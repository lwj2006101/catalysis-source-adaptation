"""Actually tune and evaluate two new acquisition controls; preserve old evidence."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,datetime,argparse,gzip
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
import robust_operator as ro

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'reference_step10'
sys.path.insert(0,str(REF/'code'))
import run_matched as old
b=old.b
EPS=(0.,.1,1.,10.)
BUDGETS=(1,2,3,5,10)
METHODS=('fixed_rbf','frozen_blend','target_mean','zero_label')


def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv(f,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(gzip.compress(f.to_csv(index=False).encode(),mtime=0) if str(p).endswith('.gz')
                  else f.to_csv(index=False).encode())


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tune(training,zs,inner,hp,directory):
    records=[]
    for source,g in training.groupby('canonical_source_id',sort=True):
        if len(g)<15:continue
        query,pool=b.random_design(g)
        if len(pool)<10:continue
        pool=np.array(sorted(pool,key=lambda i:str(training.row_id.iloc[i])))
        j=int(inner[g.index[0]])
        assert np.all(inner[g.index]==j)
        z=zs[j];y=training.C2_yield_percent.to_numpy(float)
        kp=ro.kernel(z[pool],z[pool],hp['gamma']);kpq=ro.kernel(z[pool],z[query],hp['gamma'])
        for epsilon in EPS:
            order,trace=ro.select(kp,kpq,hp['alpha'],epsilon,maximum=5)
            take=order[:5];w=ro.weights(kp[np.ix_(take,take)],kpq[take].T,hp['alpha'])
            loss=float(np.abs(w@y[pool[take]]-y[query]).mean())
            records.append(dict(source=str(source),group=str(g.leakage_group_id.iloc[0]),epsilon=epsilon,
                                mae=loss,support_indices=json.dumps(pool[take].tolist()),inner_fold=j))
    f=pd.DataFrame(records);assert len(f)>0
    scores=f.groupby(['group','epsilon']).mae.mean().groupby('epsilon').mean().reindex(EPS)
    chosen=float(scores.idxmin())
    csv(f,directory/'epsilon_tuning.csv.gz')
    write(directory/'parameters.json',dict(epsilon=chosen,scores={str(k):float(v) for k,v in scores.items()},
          fixed_rbf=hp,selection='historical pseudo-source/group-equal five-label MAE',
          n_training_sources=int(f.source.nunique()),n_training_groups=int(f.group.nunique()),
          query_outcomes_used_for_tuning=False))
    return chosen


def evaluate(data,z,base,info,hp,epsilon,designs,directory):
    y=data.C2_yield_percent.to_numpy(float);rows=[];cells=[];ledgers=[];preds=[];cids=[];qids=[]
    for source,scenario,partition,query,pool,baseline in designs:
        query=np.asarray(query,int);pool=np.array(sorted(pool,key=lambda i:str(data.row_id.iloc[i])),int)
        assert not set(pool)&set(query)
        kp=ro.kernel(z[pool],z[pool],hp['gamma']);kpq=ro.kernel(z[pool],z[query],hp['gamma'])
        orders=list(baseline)
        for policy,ee in [('rbf_matched',0.),('rbf_amplification_robust',epsilon)]:
            order,trace=ro.select(kp,kpq,hp['alpha'],ee)
            orders.append((policy,0,pool[order],{'trace':trace,'epsilon':ee}))
        lookup={int(v):i for i,v in enumerate(pool)}
        for policy,repeat,order,diagnostic in orders:
            order=np.asarray(order,int)
            assert len(order)==10 and len(set(order))==10 and set(order)<=set(pool)
            did=len(ledgers)
            ledgers.append(dict(design_id=did,source=source,scenario=scenario,partition=partition,policy=policy,
                repeat=repeat,query_indices=json.dumps(query.tolist()),pool_indices=json.dumps(pool.tolist()),
                support_indices=json.dumps(order.tolist()),diagnostic=json.dumps(diagnostic,sort_keys=True)))
            for k in BUDGETS:
                support=order[:k];ss=np.array([lookup[int(v)] for v in support])
                rr,amp,w=ro.components(kp[np.ix_(ss,ss)],kpq[ss].T,hp['alpha'])
                fixed=w@y[support]
                frozen=old.old._frozen_pred(z,y[support],base[support],query,support,base,info,k)
                values=np.column_stack([fixed,frozen['adaptive_blend'],np.full(len(query),y[support].mean()),base[query]])
                assert np.isfinite(values).all()
                cid=len(cells)
                row=dict(cell_id=cid,design_id=did,source=source,leakage_group=str(data.leakage_group_id.iloc[query[0]]),
                         scenario=scenario,partition=partition,policy=policy,repeat=repeat,k=k,n_test=len(query),
                         nominal_risk=float(rr),amplification=float(amp),fixed_alpha=hp['alpha'],fixed_gamma=hp['gamma'])
                for m,v in zip(METHODS,values.T):
                    row['mae__'+m]=float(np.abs(v-y[query]).mean());row['mse__'+m]=float(np.square(v-y[query]).mean())
                cells.append(row);preds.append(values);cids.extend([cid]*len(query));qids.extend(query)
    csv(pd.DataFrame(cells),directory/'cells.csv.gz');csv(pd.DataFrame(ledgers),directory/'support_ledger.csv.gz')
    csv(data,directory/'evaluated_input.csv.gz')
    np.savez_compressed(directory/'row_predictions.npz',predictions=np.vstack(preds),cell_id=np.array(cids,np.int32),
                        query_index=np.array(qids,np.int32),methods=np.array(METHODS))
    np.savez_compressed(directory/'geometry.npz',z=z,base=base)
    return len(cells),len(cids)


def historical(seed,fold):
    out=ROOT/'runs/operator_history_v1/parts'/f'ocm_{seed}_{fold}';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    oldpart=REF/'runs/history_v1/parts'/f'ocm_{seed}_{fold}'
    data,cfg=b.load_task('ocm_f1')
    allocation=pd.read_csv(b.SOURCE/'benchmark_splits/ocm_f1/split_ledger.csv.gz')
    allocation=allocation[(allocation.outer_seed==seed)&(allocation.fold==fold)]
    trainids=set(allocation.loc[allocation.role.eq('train'),'row_id'].astype(str))
    training=data[data.row_id.astype(str).isin(trainids)].reset_index(drop=True)
    test=pd.read_csv(oldpart/'evaluated_input.csv.gz');assert set(training.leakage_group_id).isdisjoint(test.leakage_group_id)
    original=old.old.previous.REFERENCE/'results/parts'/f'ocm_f1_{seed}_{fold}_C2_yield_percent'
    zs,inner,_,audits=old.old.previous.historical_geometry(training,cfg,original,seed,fold)
    params=json.loads((oldpart/'inherited_parameters.json').read_text())
    hp=params['controls']['ocm_f1']['composition_vopt']['5']['rbf']
    epsilon=tune(training,zs,inner,hp,out)
    ge=np.load(oldpart/'geometry.npz');completion=json.loads((oldpart/'completed.json').read_text())
    ledger=pd.read_csv(oldpart/'support_ledger.csv.gz');ledger=ledger[ledger.task=='ocm_f1']
    designs=[]
    for (source,scenario),f in ledger.groupby(['source','scenario'],sort=True):
        first=f.iloc[0]
        baseline=[(row.policy,int(row.repeat),np.array(json.loads(row.support_indices)),{}) for row in f.itertuples()]
        designs.append((str(source),scenario,0,json.loads(first.query_indices),json.loads(first.pool_indices),baseline))
    cells,records=evaluate(test,ge['z_f1'],ge['base_f1'],completion['inherited_infos']['ocm_f1'],hp,epsilon,designs,out)
    write(out/'training_source_audit.json',audits)
    write(out/'completed.json',dict(seed=seed,fold=fold,new_backbone_fits=0,cells=cells,query_records=records,
          scalar_predictions=records*len(METHODS),completed_utc=utc(),parent_part_sha256=sha(oldpart/'completed.json'),
          train_groups=sorted(set(training.leakage_group_id)),test_groups=sorted(set(test.leakage_group_id))))
    print('HISTORY DONE',seed,fold,epsilon,cells,records,flush=True)


def freeze():
    out=ROOT/'runs/operator_full_history_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():return
    orig=REF/'reference_step9/runs/full_history_v1'
    train=pd.read_csv(orig/'historical_input.csv.gz');oof=pd.read_csv(orig/'ocm_f1_training_oof.csv.gz')
    assert np.array_equal(train.row_id.astype(str),oof.row_id.astype(str))
    inner=oof.inner_fold.to_numpy(int);_,cfg=b.load_task('ocm_f1');zs={};audits=[]
    for j in sorted(set(inner)):
        tr=train.iloc[np.where(inner!=j)[0]]
        z,a=b.fitted_transform(tr,train,cfg);zs[j]=z;audits.append(a)
    params=json.loads((orig/'parameters.json').read_text())
    hp=params['controls']['ocm_f1']['composition_vopt']['5']['rbf']
    epsilon=tune(train,zs,inner,hp,out)
    write(out/'frozen.json',dict(frozen_utc=utc(),epsilon=epsilon,new_backbone_fits=0,
         parameters_sha256=sha(out/'parameters.json'),parent_frozen_sha256=sha(orig/'frozen.json'),
         code_sha256={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},
         opened_cohort_used_for_tuning=False))
    print('FROZEN',epsilon,flush=True)


def opened():
    freeze()
    out=ROOT/'runs/operator_opened_v1/parts/cohort';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    orig=REF/'runs/applications/nishimura_opened_v1/parts/cohort'
    data=pd.read_csv(orig/'evaluated_input.csv.gz');geo=np.load(orig/'geometry.npz')
    info=json.loads((orig/'completed.json').read_text())['inherited_infos']['ocm_f1']
    pp=json.loads((ROOT/'runs/operator_full_history_v1/parameters.json').read_text());hp=pp['fixed_rbf']
    full=json.loads((REF/'reference_step9/runs/full_history_v1/parameters.json').read_text())
    provenance=pd.read_csv(REF/'reference_step9/data/curated/nishimura_measured_v1/row_provenance.csv')
    lookup=provenance.set_index('row_id').catalyst_id;cat=lookup.loc[data.row_id].to_numpy(str);ids=np.array(sorted(set(cat)))
    designs=[];partition_rows=[]
    # Twenty new partitions are a within-family sensitivity, never 20 independent sources.
    for part in range(20):
        qc=np.random.default_rng(b.stable_seed('step11_catalyst_partition',part)).permutation(ids)[:29]
        query=np.where(np.isin(cat,qc))[0];pool=np.where(~np.isin(cat,qc))[0]
        assert not set(cat[query])&set(cat[pool])
        orders=old.old.selection_orders(geo['z_f1'],data.row_id.to_numpy(str),query,pool,
             old.old.alpha_schedule(info),old.old.ViewGate.from_dict(full['gate']),full['gamma'],
             b.stable_seed('step11_support_orders',part))
        baseline=[]
        for policy in ('random','operating_vopt','composition_vopt','pool_vopt'):
            baseline.extend((policy,repeat,order,diag) for repeat,order,diag in orders[policy])
        # Match the earlier selected expanded-kernel acquisition for context.
        pm=json.loads((REF/'runs/full_history_v1/parameters.json').read_text())['tasks']['ocm_f1']
        sorted_pool=np.array(sorted(pool,key=lambda i:str(data.row_id.iloc[i])))
        kp=old.mk.kernel(geo['z_f1'][sorted_pool],geo['z_f1'][sorted_pool],pm['theta'])
        kpq=old.mk.kernel(geo['z_f1'][sorted_pool],geo['z_f1'][query],pm['theta'])
        take,trace=old.mk.order(kp,kpq,kp,kpq,pm['theta']['alpha'],pm['theta']['alpha'],'centered_mse')
        baseline.append(('expanded_centered',0,sorted_pool[take],{'trace':trace}))
        designs.append((str(data.canonical_source_id.iloc[0]),'catalyst_holdout',part,query,pool,baseline))
        partition_rows.append(dict(partition=part,query_catalysts=sorted(set(cat[query])),pool_catalysts=sorted(set(cat[pool])),
                                   query_indices=query.tolist(),pool_indices=pool.tolist()))
    write(out/'partition_lock.json',dict(created_utc=utc(),partitions=partition_rows,
          labels_used_for_partitioning=False,all_partitions_included=True,source_families=1,
          interpretation='conditional partition sensitivity; no source-population CI'))
    cells,records=evaluate(data,geo['z_f1'],geo['base_f1'],info,hp,pp['epsilon'],designs,out)
    write(out/'parameters.json',pp)
    write(out/'completed.json',dict(cells=cells,query_records=records,scalar_predictions=records*len(METHODS),
          completed_utc=utc(),new_backbone_fits=0,n_partitions=20,n_sources=1,n_experiment_families=1,
          previously_opened=True))
    print('OPENED DONE',cells,records,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['history','freeze','opened'])
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if a.action=='history':
        parts=[(42,0)] if a.smoke else [(s,f) for s in old.SEEDS for f in range(5)]
        Parallel(n_jobs=a.workers)(delayed(historical)(s,f) for s,f in parts)
    elif a.action=='freeze':freeze()
    else:opened()


if __name__=='__main__':main()

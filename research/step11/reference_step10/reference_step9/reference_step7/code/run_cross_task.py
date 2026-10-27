"""Retrospective five-endpoint acquisition benchmark with frozen backbones.

125 outer models are actually refitted. Archived source-disjoint OOF responses
are reused for historical control tuning; their feature preprocessors are
rebuilt from the matching inner training groups. No target query outcome can
reach acquisition, preprocessing or control selection.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import argparse,gzip,hashlib,itertools,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'reference_step3/benchmark'
sys.path.insert(0,str(REFERENCE/'code'))
import run_source_tuned_comparison as b
import component_transfer as comp
import acquisition as acq

TASKS={'methanol_conversion':('methanol','CO2_conversion_percent'),
       'methanol_selectivity':('methanol','Methanol_selectivity_percent'),
       'methanol_sty':('sty','STY_mgMeOH_per_h_per_gcat'),
       'ocm_f0':('ocm_f0','C2_yield_percent'),'ocm_f1':('ocm_f1','C2_yield_percent')}
METHODS=b.METHODS+('local_ridge_retuned','residual_ridge_retuned','adaptive_blend_retuned','local_rbf_retuned')
RBF_GRID=[dict(alpha=a,gamma=g) for a in (.01,.1,1.) for g in (.1,1.,10.)]
BLEND_GRID=list(itertools.product(b.W0_GRID,b.C_GRID))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def csv(frame,path):
    path=Path(path);blob=frame.to_csv(index=False).encode()
    if path.suffix=='.gz':blob=gzip.compress(blob,mtime=0)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(blob);temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def orders(data,sub,z,old,pseudo=False):
    scenarios={'random':(*b.random_design(sub),{})} if pseudo else b.designs(sub)
    source=str(sub.canonical_source_id.iloc[0]);alphas={str(k):old['hyperparameters'][str(k)]['local_alpha'] for k in acq.BUDGETS}
    for scenario,(query,pool,meta) in scenarios.items():
        if len(pool)<10:continue
        pool=np.array(sorted(pool,key=lambda i:str(data.row_id.iloc[i])),int)
        for policy in acq.POLICIES:
            if policy=='random':
                for repeat in range(10 if pseudo else 50):
                    order=(np.random.default_rng(b.stable_seed('pseudo_step2',source,repeat)).permutation(pool)
                           if pseudo else b.calibration_order(data,pool,source,scenario,repeat))
                    yield scenario,query,pool,policy,repeat,order[:10],[],meta
            else:
                take,trace=acq.select_order(z[pool],z[query],policy,alphas)
                yield scenario,query,pool,policy,0,pool[take],trace,meta


def historical_geometry(training,cfg,old_part,seed,fold):
    archived=pd.read_csv(old_part/'training_oof.csv.gz')
    np.testing.assert_array_equal(training.row_id.astype(str),archived.row_id.astype(str))
    inner=archived.inner_fold.to_numpy(int);zs={};audits=[]
    splits=list(b.group_folds(training.leakage_group_id,b.stable_seed('inner',f'{seed}:{fold}')))
    for j,(tr,va) in enumerate(splits):
        assert np.all(inner[va]==j)
        z,meta=b.fitted_transform(training.iloc[tr],training,cfg);zs[j]=z
        assert set(training.leakage_group_id.iloc[tr]).isdisjoint(training.leakage_group_id.iloc[va])
        audits.append(dict(inner_fold=j,training_groups=sorted(set(training.leakage_group_id.iloc[tr])),
                          validation_groups=sorted(set(training.leakage_group_id.iloc[va])),preprocessor=meta,
                          response_origin='archived source-disjoint OOF; no newly fit inner models'))
    return zs,inner,archived.base.to_numpy(float),audits


def tune_controls(training,target,zs,inner,base,old,out):
    """Policy/budget-specific tuning on equal historical pseudo-sources.

    Same finite alpha, kernel and weight grids for all policies. Backbone and
    acquisition alpha are inherited; all controls see the same acquired rows.
    Random pseudo-supports repeat ten times; deterministic designs run once.
    """
    y=training[target].to_numpy(float);arrays={};episodes=[]
    for source,sub in training.groupby('canonical_source_id',sort=True):
        if len(sub)<15:continue
        j=int(inner[sub.index[0]]);assert np.all(inner[sub.index]==j);z=zs[j]
        for scenario,query,pool,policy,repeat,order,trace,meta in orders(training,sub,z,old,True):
            for k in acq.BUDGETS:
                support=order[:k];ya=y[support];pa=base[support];zq=z[query];za=z[support]
                combined=[b.ridge_query_and_loo(za,np.column_stack([ya,ya-pa]),zq,a) for a in b.ALPHAS]
                local=[p[:,0] for p,_ in combined];residual=[base[query]+p[:,1] for p,_ in combined]
                lloo=[None if l is None else l[:,0] for _,l in combined]
                rloo=[None if l is None else pa+l[:,1] for _,l in combined]
                rbf=[comp.local_rbf(za,ya,zq,**hp) for hp in RBF_GRID]
                for method,preds in [('local',local),('residual',residual),('rbf',rbf)]:
                    arrays.setdefault((policy,k,str(source),method),[]).append([float(np.abs(y[query]-p).mean()) for p in preds])
                episodes.append(dict(source=str(source),policy=policy,k=k,ya=ya,yq=y[query],local=local,residual=residual,lloo=lloo,rloo=rloo))
    hp={};scores=[]
    for policy in acq.POLICIES:
        hp[policy]={}
        for k in acq.BUDGETS:
            chosen={}
            for method,grid in [('local',b.ALPHAS),('residual',b.ALPHAS),('rbf',RBF_GRID)]:
                per=[]
                for (pp,kk,source,mm),values in arrays.items():
                    if (pp,kk,mm)==(policy,k,method):
                        v=np.mean(values,axis=0);per.append(v)
                        scores.extend(dict(policy=pp,k=kk,source=source,model=mm,candidate=i,mae=float(a)) for i,a in enumerate(v))
                mean=np.mean(per,axis=0);ix=int(np.argmin(mean));chosen[method]=grid[ix];chosen[method+'_index']=ix;chosen[method+'_scores']=mean.tolist()
            hp[policy][str(k)]=chosen
    blend={}
    for ep in episodes:
        choice=hp[ep['policy']][str(ep['k'])];li=choice['local_index'];ri=choice['residual_index'];dscale=old['hyperparameters'][str(ep['k'])]['blend_dscale']
        errors=[]
        for w0,c in BLEND_GRID:
            w=b.blend_weight(ep['ya'],ep['lloo'][li],ep['rloo'][ri],w0,c,dscale)
            errors.append(float(np.abs(ep['yq']-((1-w)*ep['local'][li]+w*ep['residual'][ri])).mean()))
        blend.setdefault((ep['policy'],ep['k'],ep['source']),[]).append(errors)
    for policy in acq.POLICIES:
        for k in acq.BUDGETS:
            per=[]
            for (pp,kk,source),values in blend.items():
                if (pp,kk)==(policy,k):
                    v=np.mean(values,axis=0);per.append(v)
                    scores.extend(dict(policy=pp,k=kk,source=source,model='blend',candidate=i,mae=float(a)) for i,a in enumerate(v))
            mean=np.mean(per,axis=0);w0,c=BLEND_GRID[int(np.argmin(mean))]
            hp[policy][str(k)].update(blend_w0=w0,blend_c=c,blend_dscale=old['hyperparameters'][str(k)]['blend_dscale'],blend_scores=mean.tolist(),n_pseudo_sources=len(per))
    write_json(out/'retuned_parameters.json',hp);csv(pd.DataFrame(scores),out/'historical_control_scores.csv.gz')
    return hp


def predictions(za,ya,pa,zq,pq,old,controls,k,policy,scale):
    hp=old['hyperparameters'][str(k)];frozen,w=b.predict_methods(za,ya,pa,zq,pq,hp,old['prior'],scale)
    choice=controls[policy][str(k)]
    local,lloo=b.ridge_query_and_loo(za,ya,zq,choice['local'])
    residual,rloo=b.ridge_query_and_loo(za,ya-pa,zq,choice['residual']);residual+=pq
    if rloo is not None:rloo+=pa
    rw=b.blend_weight(ya,lloo,rloo,choice['blend_w0'],choice['blend_c'],choice['blend_dscale'])
    frozen.update(local_ridge_retuned=local,residual_ridge_retuned=residual,adaptive_blend_retuned=(1-rw)*local+rw*residual,
                  local_rbf_retuned=comp.local_rbf(za,ya,zq,**choice['rbf']))
    assert tuple(frozen)==METHODS and all(np.isfinite(p).all() for p in frozen.values())
    return frozen,w,rw


def evaluate(data,target,z,base,old,controls,endpoint,seed,fold,scale,out):
    y=data[target].to_numpy(float);cells=[];supports=[];traces=[];preds=[];ci=[];qi=[];permutations=[]
    for source,sub in data.groupby('canonical_source_id',sort=True):
        for scenario,query,pool,policy,repeat,order,trace,meta in orders(data,sub,z,old):
            did=len(supports);supports.append(dict(design_id=did,source=str(source),scenario=scenario,policy=policy,repeat=repeat,
                query_indices=json.dumps(query.tolist()),pool_indices=json.dumps(pool.tolist()),support_indices=json.dumps(order.tolist()),
                query_ids=json.dumps(data.row_id.iloc[query].astype(str).tolist()),support_ids=json.dumps(data.row_id.iloc[order].astype(str).tolist()),metadata=json.dumps(meta)))
            traces.extend(dict(design_id=did,**r) for r in trace)
            for k in acq.BUDGETS:
                a=order[:k];p,w,rw=predictions(z[a],y[a],base[a],z[query],base[query],old,controls,k,policy,scale)
                cid=len(cells);rank=int(np.linalg.matrix_rank(z[a]-z[a].mean(0),tol=1e-7));assert rank<=min(z.shape[1],k-1)
                row=dict(cell_id=cid,design_id=did,endpoint=endpoint,target=target,seed=seed,fold=fold,source=str(source),
                    leakage_group=str(sub.leakage_group_id.iloc[0]),scenario=scenario,policy=policy,repeat=repeat,k=k,n_test=len(query),
                    support_rank=rank,blend_weight=w,retuned_weight=rw)
                lo,hi=b.TARGET_BOUNDS[target]
                for m,pp in p.items():
                    bounded=np.maximum(pp,lo) if hi is None else np.clip(pp,lo,hi)
                    row['mae__'+m]=float(np.abs(y[query]-pp).mean());row['mse__'+m]=float(np.mean((y[query]-pp)**2));row['bounded_mae__'+m]=float(np.abs(y[query]-bounded).mean())
                cells.append(row);preds.append(np.column_stack(list(p.values())));ci.extend([cid]*len(query));qi.extend(query.tolist())
                if k==5 and repeat==0:
                    rng=np.random.default_rng(b.stable_seed('stage7_label_correspondence',endpoint,seed,fold,source,scenario,policy));errors=[]
                    for _ in range(20):
                        yp=rng.permutation(y[a]);pp,_=b.predict_methods(z[a],yp,base[a],z[query],base[query],old['hyperparameters']['5'],old['prior'],scale)
                        errors.append(float(np.abs(y[query]-pp['adaptive_blend']).mean()))
                    permutations.append(dict(endpoint=endpoint,seed=seed,fold=fold,source=str(source),leakage_group=str(sub.leakage_group_id.iloc[0]),
                        scenario=scenario,policy=policy,k=5,aligned_mae=row['mae__adaptive_blend'],permuted_mae=float(np.mean(errors)),permutation_repeats=20))
    csv(pd.DataFrame(cells),out/'cells.csv.gz');csv(pd.DataFrame(supports),out/'support_ledger.csv.gz')
    csv(pd.DataFrame(traces),out/'acquisition_trace.csv.gz');csv(pd.DataFrame(permutations),out/'label_permutation.csv.gz');csv(data,out/'evaluated_input.csv.gz')
    nrows=sum(len(p) for p in preds)
    np.savez_compressed(out/'row_predictions.npz',predictions=np.vstack(preds) if preds else np.empty((0,len(METHODS))),cell_id=np.array(ci,np.int32),query_index=np.array(qi,np.int32),methods=np.array(METHODS,dtype='U32'))
    np.savez_compressed(out/'evaluation_geometry.npz',z=z,base=base)
    return pd.DataFrame(cells),len(supports),nrows


def run_part(endpoint,seed,fold,identity):
    start=time.monotonic();task,target=TASKS[endpoint];name=f'{endpoint}_{seed}_{fold}';out=ROOT/'results/parts'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        assert json.loads((out/'completed.json').read_text())['identity']==identity;return name
    old_part=REFERENCE/'results/parts'/f'{task}_{seed}_{fold}_{target}';old=json.loads((old_part/'done.json').read_text())
    data,cfg=b.load_task(task);ledger=pd.read_csv(b.SOURCE/'benchmark_splits'/task/'split_ledger.csv.gz');allocation=ledger[(ledger.outer_seed==seed)&(ledger.fold==fold)]
    trids=set(allocation.loc[allocation.role.eq('train'),'row_id'].astype(str));tr=np.flatnonzero(data.row_id.astype(str).isin(trids));te=np.flatnonzero(~data.row_id.astype(str).isin(trids))
    training=data.iloc[tr].reset_index(drop=True);evaluated=data.iloc[te].reset_index(drop=True)
    assert set(training.leakage_group_id).isdisjoint(evaluated.leakage_group_id)
    zs,inner,oof,audits=historical_geometry(training,cfg,old_part,seed,fold);write_json(out/'source_disjoint_audit.json',audits)
    controls=tune_controls(training,target,zs,inner,oof,old,out)
    z,preprocessor=b.fitted_transform(training,evaluated,cfg);ztrain,_=b.fitted_transform(training,training,cfg)
    model=b.estimator(old['model']).fit(ztrain,training[target].to_numpy(float));base=model.predict(z)
    archive=pd.read_csv(old_part/'zero_all_predictions.csv.gz');np.testing.assert_array_equal(evaluated.row_id.astype(str),archive.row_id.astype(str))
    base_diff=float(np.max(np.abs(base-archive.base.to_numpy(float))));np.testing.assert_allclose(base,archive.base,rtol=1e-10,atol=1e-7)
    scale=max(float(training[target].std(ddof=0)),1e-8)
    cells,ndesign,nrows=evaluate(evaluated,target,z,base,old,controls,endpoint,seed,fold,scale,out)
    original=pd.read_csv(old_part/'cells.csv.gz');random=cells[cells.policy.eq('random')]
    matched=random.merge(original,on=['source','scenario','repeat','k'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(matched)==len(random)==len(original);diff=0.
    for m in b.METHODS:
        v=float(np.max(np.abs(matched['mae__'+m+'_new']-matched['mae__'+m+'_old'])));diff=max(diff,v)
        np.testing.assert_allclose(matched['mae__'+m+'_new'],matched['mae__'+m+'_old'],rtol=1e-10,atol=1e-7)
    write_json(out/'completed.json',dict(identity=identity,endpoint=endpoint,task=task,target=target,seed=seed,fold=fold,model=old['model'],
        preprocessor=preprocessor,label_scale=scale,frozen_parameters=old,training_groups=sorted(set(training.leakage_group_id)),evaluation_groups=sorted(set(evaluated.leakage_group_id)),
        n_cells=len(cells),n_designs=ndesign,n_prediction_rows=nrows,base_maximum_difference=base_diff,archived_random_mae_maximum_difference=diff,
        archived_random_cells=len(random),new_backbone_fits=1,new_inner_model_fits=0,wall_seconds=time.monotonic()-start))
    print('DONE',name,'cells',len(cells),'seconds',round(time.monotonic()-start,1),flush=True);return name


def lock_protocol(args):
    path=ROOT/'protocol/cross_task_lock.json'
    config=dict(version='step7-cross-task-1',endpoints=TASKS,seeds=args.seeds,folds=args.folds,budgets=acq.BUDGETS,policies=acq.POLICIES,methods=METHODS,
        random_draws=50,pseudo_random_draws=10,label_permutations=20,control_alpha_grid=b.ALPHAS,rbf_grid=RBF_GRID,blend_grid=BLEND_GRID,
        primary='frozen adaptive_blend: linear_vopt minus random MAE; random-query scenario, k=5, five endpoint family',
        uncertainty='source-equal; draws then seeds averaged; leakage-group cluster bootstrap; nominal95 and Bonferroni99 intervals for five primary endpoints; endpoints not independent',
        secondary='retuned controls, budgets, coverage and pool V, stress and label permutation are exploratory',
        status='retrospective extension: these corpora and outcomes have already informed method design',
        training='backbones and original adapters frozen Step3; new policy controls tuned only on archived historical source-disjoint OOF; acquisition regularizers inherited random-design local alpha',
        data_hashes={t:sha(b.SOURCE/'inputs'/f) for t,f in b.TASK_INPUT.items()},split_hashes={t:sha(b.SOURCE/'benchmark_splits'/t/'split_ledger.csv.gz') for t in b.TASK_INPUT},
        code_hashes={f.name:sha(f) for f in (ROOT/'code').glob('*.py') if f.name in ['acquisition.py','run_cross_task.py','component_transfer.py']},
        inherited_parts={p.name:sha(p/'done.json') for p in (REFERENCE/'results/parts').iterdir()})
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if path.exists():assert json.loads(path.read_text())['identity']==identity,'Protocol changed: use a fresh output package'
    else:write_json(path,dict(identity=identity,created_utc=datetime.now(timezone.utc).isoformat(),config=config))
    return identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=[42,101,202,303,404]);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)));p.add_argument('--endpoints',nargs='+',choices=list(TASKS),default=list(TASKS));args=p.parse_args()
    identity=lock_protocol(args);jobs=list(itertools.product(args.endpoints,args.seeds,args.folds))
    Parallel(n_jobs=args.workers)(delayed(run_part)(endpoint,seed,fold,identity) for endpoint,seed,fold in jobs)
    write_json(ROOT/'verification/run_completion.json',dict(parts=len(jobs),identity=identity,new_backbone_fits=len(jobs),new_inner_model_fits=0,completed_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version()))


if __name__=='__main__':main()

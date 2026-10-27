"""Post-primary exploratory OCM geometry/predictor crossing.

Both operating-only and composition-enriched acquisition use the SAME F1
random-design alpha schedule. Both frozen predictors see identical supports.
No new tuning, outcomes in acquisition, or new backbone fits are introduced.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
import run_cross_task as r
from aggregate_cross_task import bootstrap


def main():
    primary=pd.read_csv(r.ROOT/'results/primary_family.csv')
    r.write_json(r.ROOT/'protocol/ocm_geometry_diagnostic.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        timing='added after inspecting the five-endpoint primary results; exploratory, not preregistered',
        hypothesis='Does the OCM feature representation affect acquisition and prediction separately?',
        controls='same F1 alpha schedule, same pools/query rows, F0/F1 frozen predictors crossed with six-feature/full-feature V design',
        backbone_refits=0,new_hyperparameter_tuning=0,primary_results_unchanged=True))
    cells=[];ledger=[];prediction_banks=[];cid=[];qid=[];rowbank=[];reference_matches=0
    seeds=sorted({int(x) for x in pd.read_csv(r.ROOT/'results/per_source_per_seed.csv.gz').seed})
    for seed in seeds:
        for fold in range(5):
            dirs={t:r.ROOT/'results/parts'/f'{t}_{seed}_{fold}' for t in ['ocm_f0','ocm_f1']}
            if not all((p/'completed.json').exists() for p in dirs.values()):continue
            infos={t:json.loads((p/'completed.json').read_text()) for t,p in dirs.items()}
            data={t:pd.read_csv(p/'evaluated_input.csv.gz',keep_default_na=False) for t,p in dirs.items()};geo={t:np.load(p/'evaluation_geometry.npz') for t,p in dirs.items()}
            np.testing.assert_array_equal(data['ocm_f0'].row_id.astype(str),data['ocm_f1'].row_id.astype(str));np.testing.assert_allclose(data['ocm_f0'].C2_yield_percent,data['ocm_f1'].C2_yield_percent)
            np.testing.assert_allclose(geo['ocm_f0']['z'],geo['ocm_f1']['z'][:,:6],atol=1e-12)
            d=data['ocm_f1'];y=d.C2_yield_percent.to_numpy(float);z=geo['ocm_f1']['z'];old=infos['ocm_f1']['frozen_parameters'];alphas={str(k):old['hyperparameters'][str(k)]['local_alpha'] for k in r.acq.BUDGETS}
            reference=pd.read_csv(dirs['ocm_f1']/'support_ledger.csv.gz');reference=reference[reference.policy.eq('linear_vopt')].set_index(['source','scenario'])
            row_offset=len(rowbank);rowbank.extend(d[['row_id','canonical_source_id','C2_yield_percent']].assign(seed=seed,fold=fold).to_dict('records'))
            for source,sub in d.groupby('canonical_source_id',sort=True):
                for scenario,(query,pool,meta) in r.b.designs(sub).items():
                    pool=np.array(sorted(pool,key=lambda i:str(d.row_id.iloc[i])),int)
                    for design,zz in [('operating_v_same_alpha',z[:,:6]),('composition_v_same_alpha',z)]:
                        take,trace=r.acq.select_order(zz[pool],zz[query],'linear_vopt',alphas);support=pool[take]
                        if design=='composition_v_same_alpha':
                            record=reference.loc[(source,scenario)];np.testing.assert_array_equal(support,np.array(json.loads(record.support_indices)));reference_matches+=1
                        did=len(ledger);ledger.append(dict(design_id=did,seed=seed,fold=fold,source=str(source),scenario=scenario,design=design,
                            support_ids=json.dumps(d.row_id.iloc[support].astype(str).tolist()),query_ids=json.dumps(d.row_id.iloc[query].astype(str).tolist()),
                            support_indices=json.dumps(support.tolist()),query_indices=json.dumps(query.tolist()),row_offset=row_offset))
                        for k in r.acq.BUDGETS:
                            for predictor in ['ocm_f0','ocm_f1']:
                                info=infos[predictor];zz=geo[predictor]['z'];base=geo[predictor]['base'];a=support[:k]
                                pp,w=r.b.predict_methods(zz[a],y[a],base[a],zz[query],base[query],info['frozen_parameters']['hyperparameters'][str(k)],info['frozen_parameters']['prior'],info['label_scale'])
                                cell_id=len(cells);row=dict(cell_id=cell_id,design_id=did,seed=seed,fold=fold,source=str(source),leakage_group=str(sub.leakage_group_id.iloc[0]),scenario=scenario,k=k,design=design,predictor=predictor,n_test=len(query))
                                row.update({'mae__'+m:float(np.abs(y[query]-p).mean()) for m,p in pp.items()});cells.append(row);prediction_banks.append(np.column_stack(list(pp.values())));cid.extend([cell_id]*len(query));qid.extend((row_offset+query).tolist())
    c=pd.DataFrame(cells);keys=['source','leakage_group','scenario','k','design','predictor'];metrics=['mae__'+m for m in r.b.METHODS]
    source=c.groupby(keys,as_index=False)[metrics].mean();summary=source.groupby(['scenario','k','design','predictor'],as_index=False)[metrics].mean();contrasts=[]
    for (scenario,k,predictor),g in source.groupby(['scenario','k','predictor']):
        a=g[g.design.eq('composition_v_same_alpha')];d=g[g.design.eq('operating_v_same_alpha')];m=a.merge(d,on=['source','leakage_group','scenario','k','predictor'],suffixes=('_a','_b'),validate='one_to_one')
        m['delta']=m.mae__adaptive_blend_a-m.mae__adaptive_blend_b;ci=bootstrap(m,'delta',r.b.stable_seed('step7_geometry_cross',scenario,k,predictor))
        contrasts.append(dict(scenario=scenario,k=k,predictor=predictor,contrast='composition acquisition minus operating acquisition; same alpha',delta_mae=ci['estimate'],ci_low=ci['ci_low'],ci_high=ci['ci_high'],n_sources=len(m),mae=float(m.mae__adaptive_blend_a.mean()),reference_mae=float(m.mae__adaptive_blend_b.mean())))
    out=r.ROOT/'results/ocm_geometry';out.mkdir(exist_ok=True)
    for frame,name in [(c,'cells.csv.gz'),(pd.DataFrame(ledger),'support_ledger.csv.gz'),(source,'per_source.csv'),(summary,'summary.csv'),(pd.DataFrame(contrasts),'paired_contrasts.csv'),(pd.DataFrame(rowbank),'row_bank.csv.gz')]:r.csv(frame,out/name)
    pred=np.vstack(prediction_banks);cell_id=np.asarray(cid,np.int32);query_id=np.asarray(qid,np.int32);np.savez_compressed(out/'row_predictions.npz',predictions=pred,cell_id=cell_id,query_index=query_id,methods=np.array(r.b.METHODS,dtype='U32'))
    actual=np.asarray([rowbank[i]['C2_yield_percent'] for i in query_id]);sums=np.zeros((len(c),len(r.b.METHODS)));np.add.at(sums,cell_id,np.abs(pred-actual[:,None]));n=np.bincount(cell_id,minlength=len(c));np.testing.assert_allclose(sums/n[:,None],c[metrics],rtol=1e-11,atol=1e-8)
    r.write_json(r.ROOT/'verification/ocm_geometry.json',dict(passed=True,cells=len(c),row_values_reaggregated=pred.size,original_composition_orders_matched=reference_matches,new_backbone_fits=0,timing='exploratory after primary results'))
    print(pd.DataFrame(contrasts).query("scenario=='random' and k==5").to_string(index=False),flush=True)


if __name__=='__main__':main()

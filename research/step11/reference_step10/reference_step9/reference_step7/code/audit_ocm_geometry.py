"""Read-only independent audit of the post-primary representation crossing."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
import run_cross_task as r


def audit():
    root=r.ROOT/'results/ocm_geometry'
    if not root.exists():return dict(status='not_present')
    c=pd.read_csv(root/'cells.csv.gz');ledger=pd.read_csv(root/'support_ledger.csv.gz');bank=pd.read_csv(root/'row_bank.csv.gz');a=np.load(root/'row_predictions.npz')
    cid,qid=a['cell_id'],a['query_index'];p=a['predictions'];y=bank.C2_yield_percent.to_numpy(float)
    sums=np.zeros((len(c),len(r.b.METHODS)));np.add.at(sums,cid,np.abs(p-y[qid,None]));n=np.bincount(cid,minlength=len(c));np.testing.assert_allclose(sums/n[:,None],c[['mae__'+m for m in r.b.METHODS]],rtol=1e-11,atol=1e-8)
    cache={};reconstructed=0;designs=0;maxdiff=0.
    for row in ledger.itertuples():
        key=(row.seed,row.fold)
        if key not in cache:
            cache[key]={}
            for predictor in ['ocm_f0','ocm_f1']:
                part=r.ROOT/'results/parts'/f'{predictor}_{row.seed}_{row.fold}'
                cache[key][predictor]=(json.loads((part/'completed.json').read_text()),pd.read_csv(part/'evaluated_input.csv.gz',keep_default_na=False),np.load(part/'evaluation_geometry.npz'))
        info,data,geo=cache[key]['ocm_f1'];sub=data[data.canonical_source_id.astype(str).eq(str(row.source))];query,pool,_=r.b.designs(sub)[row.scenario]
        pool=np.array(sorted(pool,key=lambda i:str(data.row_id.iloc[i])),int);alpha={str(k):info['frozen_parameters']['hyperparameters'][str(k)]['local_alpha'] for k in r.acq.BUDGETS}
        z=geo['z'][:,:6] if row.design=='operating_v_same_alpha' else geo['z'];take,_=r.acq.select_order(z[pool],z[query],'linear_vopt',alpha);support=pool[take]
        np.testing.assert_array_equal(support,json.loads(row.support_indices));np.testing.assert_array_equal(query,json.loads(row.query_indices));designs+=1
        for cell in c[(c.design_id==row.design_id)&(c.k==5)].itertuples():
            inf,d,g=cache[key][cell.predictor];z,base=g['z'],g['base'];yy=d.C2_yield_percent.to_numpy(float);support5=support[:5]
            pp,_=r.b.predict_methods(z[support5],yy[support5],base[support5],z[query],base[query],inf['frozen_parameters']['hyperparameters']['5'],inf['frozen_parameters']['prior'],inf['label_scale'])
            recon=np.column_stack([pp[m] for m in r.b.METHODS]);mask=cid==cell.cell_id;np.testing.assert_array_equal(qid[mask],row.row_offset+query);np.testing.assert_allclose(recon,p[mask],rtol=1e-11,atol=1e-8)
            maxdiff=max(maxdiff,float(np.max(np.abs(recon-p[mask]))));reconstructed+=recon.size
    return dict(passed=True,cells=len(c),row_values_reaggregated=p.size,row_values_recomputed=reconstructed,acquisition_orders_recomputed=designs,maximum_prediction_difference=maxdiff)


if __name__=='__main__':
    result=audit();out=Path(os.environ.get('CROSS_TASK_VERIFY_REPORT_DIR',r.ROOT/'verification'));out.mkdir(parents=True,exist_ok=True)
    r.write_json(out/'geometry_row_audit.json',result);print(json.dumps(result,indent=2))

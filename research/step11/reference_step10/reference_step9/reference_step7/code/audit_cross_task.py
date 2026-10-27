"""Independent row reaggregation, support reconstruction and label poison audit."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_cross_task as r


def main():
    counts=dict(row_values_reaggregated=0,row_values_recomputed=0,poisoned_label_orders_recomputed=0,source_group_checks=0,prior_ocm_orders_matched=0,prior_ocm_maes_matched=0)
    max_mae=max_mse=max_pred=0.;allcells=[];dirs=sorted((r.ROOT/'results/parts').iterdir())
    for part in dirs:
        info=json.loads((part/'completed.json').read_text());old=info['frozen_parameters'];controls=json.loads((part/'retuned_parameters.json').read_text())
        data=pd.read_csv(part/'evaluated_input.csv.gz',keep_default_na=False);target=info['target'];y=data[target].to_numpy(float)
        rows=np.load(part/'row_predictions.npz');geo=np.load(part/'evaluation_geometry.npz');z,base=geo['z'],geo['base'];pred=rows['predictions'];cid,qid=rows['cell_id'],rows['query_index']
        np.testing.assert_array_equal(rows['methods'],r.METHODS);cells=pd.read_csv(part/'cells.csv.gz');ledger=pd.read_csv(part/'support_ledger.csv.gz')
        assert np.all(np.isfinite(pred)) and not ledger.duplicated(['source','scenario','policy','repeat']).any()
        n=np.bincount(cid,minlength=len(cells));assert np.array_equal(n,cells.n_test.to_numpy())
        summed=np.zeros((len(cells),len(r.METHODS)));square=summed.copy();np.add.at(summed,cid,np.abs(pred-y[qid,None]));np.add.at(square,cid,(pred-y[qid,None])**2)
        aa=summed/n[:,None];mm=square/n[:,None]
        for reconstructed,columns,kind in [(aa,['mae__'+m for m in r.METHODS],'mae'),(mm,['mse__'+m for m in r.METHODS],'mse')]:
            actual=cells[columns].to_numpy();diff=float(np.max(np.abs(reconstructed-actual)));np.testing.assert_allclose(reconstructed,actual,rtol=1e-11,atol=1e-8)
            if kind=='mae':max_mae=max(max_mae,diff)
            else:max_mse=max(max_mse,diff)
        counts['row_values_reaggregated']+=pred.size
        assert set(info['training_groups']).isdisjoint(info['evaluation_groups']);counts['source_group_checks']+=1
        for audit in json.loads((part/'source_disjoint_audit.json').read_text()):
            assert set(audit['training_groups']).isdisjoint(audit['validation_groups']);counts['source_group_checks']+=1
        for row in ledger.itertuples():
            support=np.array(json.loads(row.support_indices),int);query=np.array(json.loads(row.query_indices),int);pool=np.array(json.loads(row.pool_indices),int)
            assert len(set(support))==10 and set(support).issubset(pool) and set(pool).isdisjoint(query)
            assert data.canonical_source_id.iloc[support].astype(str).eq(str(row.source)).all()
            np.testing.assert_array_equal(data.row_id.iloc[support].astype(str).to_numpy(),np.array(json.loads(row.support_ids)))
            if row.repeat==0:
                for cell in cells[cells.design_id.eq(row.design_id)].itertuples():
                    mask=cid==cell.cell_id;np.testing.assert_array_equal(qid[mask],query)
                    pp,_,_=r.predictions(z[support[:cell.k]],y[support[:cell.k]],base[support[:cell.k]],z[query],base[query],old,controls,cell.k,row.policy,info['label_scale'])
                    recon=np.column_stack([pp[m] for m in r.METHODS]);diff=float(np.max(np.abs(recon-pred[mask])));np.testing.assert_allclose(recon,pred[mask],rtol=1e-11,atol=1e-8)
                    max_pred=max(max_pred,diff);counts['row_values_recomputed']+=recon.size
        poison=data.copy();poison[target]=np.linspace(-1e9,1e9,len(data));expected=ledger.set_index(['source','scenario','policy','repeat'])
        for source,sub in poison.groupby('canonical_source_id',sort=True):
            for scenario,query,pool,policy,repeat,order,trace,meta in r.orders(poison,sub,z,old):
                record=expected.loc[(str(source),scenario,policy,repeat)]
                np.testing.assert_array_equal(order,np.array(json.loads(record.support_indices)));np.testing.assert_array_equal(query,np.array(json.loads(record.query_indices)))
                counts['poisoned_label_orders_recomputed']+=1
        if info['endpoint']=='ocm_f1':
            prior=r.ROOT/'reference_step6/ocm_order_reference'/f"ocm_f1_{info['seed']}_{info['fold']}_C2_yield_percent"
            ref=pd.read_csv(prior/'support_ledger.csv.gz');keys=['source','scenario','policy','repeat'];new=ledger[ledger.policy.isin(['random','linear_vopt'])];matched=new.merge(ref,on=keys,suffixes=('_new','_old'),validate='one_to_one')
            assert len(matched)==len(ref)==len(new)
            for rec in matched.itertuples():
                np.testing.assert_array_equal(json.loads(rec.support_ids_new),json.loads(rec.support_ids_old));counts['prior_ocm_orders_matched']+=1
            original=pd.read_csv(prior/'cells.csv.gz');matched=cells[cells.policy.isin(['random','linear_vopt'])].merge(original,on=keys+['k'],suffixes=('_new','_old'),validate='one_to_one')
            for method in ['base','target_mean','local_ridge','adaptive_blend']:
                np.testing.assert_allclose(matched['mae__'+method+'_new'],matched['mae__'+method+'_old'],rtol=1e-11,atol=1e-8);counts['prior_ocm_maes_matched']+=len(matched)
        allcells.append(cells);print('AUDIT',part.name,'passed',flush=True)
    cells=pd.concat(allcells,ignore_index=True);keys=['endpoint','target','scenario','source','leakage_group','policy','k'];metrics=[c for c in cells if c.startswith(('mae__','mse__','bounded_mae__'))]
    source=cells.groupby(keys+['seed'],as_index=False)[metrics].mean().groupby(keys,as_index=False)[metrics].mean().sort_values(keys).reset_index(drop=True)
    original=pd.read_csv(r.ROOT/'results/per_source_metrics.csv').sort_values(keys).reset_index(drop=True);pd.testing.assert_frame_equal(source[keys],original[keys],check_dtype=False);np.testing.assert_allclose(source[metrics],original[metrics],rtol=1e-11,atol=1e-8)
    result=counts|dict(passed=True,fits=len(dirs),maximum_row_mae_difference=max_mae,maximum_row_mse_difference=max_mse,maximum_prediction_reconstruction_difference=max_pred,
        per_source_rows_reaggregated=len(source),poison_value_range=[-1e9,1e9],query_outcomes_absent_from_prediction_and_acquisition_interfaces=True)
    out=Path(os.environ.get('CROSS_TASK_VERIFY_REPORT_DIR',r.ROOT/'verification'));out.mkdir(parents=True,exist_ok=True);r.write_json(out/'row_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Independent scalar scoring and centered-KRR reconstruction, then aggregation."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import run_matched as run
import matched_kernel as mk


def independent_kernel(x, y, theta):
    def squared(a,b):
        return np.square(a[:,None,:]-b[None,:,:]).sum(-1)/max(a.shape[1],1)
    if theta['kind']=='isotropic':
        return np.exp(-theta['gamma']*squared(x,y))
    op=np.exp(-theta['gamma_op']*squared(x[:,:6],y[:,:6]))
    chem=np.exp(-theta['gamma_chem']*squared(x[:,6:],y[:,6:]))
    if theta['kind']=='additive':return .5*(op+chem)
    if theta['kind']=='product':return op*chem
    return (op+chem+op*chem)/3


def independent_prediction(x, y, q, theta):
    k=independent_kernel(x,x,theta); kq=independent_kernel(q,x,theta)
    h=np.eye(len(x))-np.ones_like(k)/len(x)
    centered=kq-kq.mean(1)[:,None]-k.mean(0)[None,:]+k.mean()
    model=KernelRidge(alpha=theta['alpha'],kernel='precomputed').fit(h@k@h,y-y.mean())
    return y.mean()+model.predict(centered)


def audit(run_root):
    counts=dict(parts=0,scalar_values_reaggregated=0,support_orders_checked=0,
                new_design_orders_reconstructed=0,independent_kernel_query_values_reconstructed=0,
                frozen_blend_query_values_reconstructed=0,legacy_locked_rbf_cells_matched=0,
                actual_poisoned_source_frames=0,maximum_independent_prediction_error=0.,
                maximum_scalar_loss_error=0.)
    all_cells=[]
    for part in sorted((run_root/'parts').iterdir()):
        if not (part/'completed.json').exists():
            raise ValueError('Incomplete part: '+str(part))
        completion=json.loads((part/'completed.json').read_text())
        if set(completion['history_training_groups']) & set(completion['evaluation_groups']):
            raise ValueError('Source-group leakage')
        data=pd.read_csv(part/'evaluated_input.csv.gz')
        cells=pd.read_csv(part/'cells.csv.gz')
        ledger=pd.read_csv(part/'support_ledger.csv.gz')
        bank=np.load(part/'row_predictions.npz')
        predictions=bank['predictions']; cell_id=bank['cell_id']; query_id=bank['query_index']
        np.testing.assert_array_equal(bank['methods'],np.array(run.METHODS))
        if len(predictions)!=completion['n_prediction_rows'] or len(cells)!=completion['n_cells']:
            raise ValueError('Completion count mismatch')
        y=data.C2_yield_percent.to_numpy(float)
        observed=y[query_id]; n=np.bincount(cell_id,minlength=len(cells))
        np.testing.assert_array_equal(n,cells.n_test.to_numpy(int))
        for j,method in enumerate(run.METHODS):
            error=predictions[:,j]-observed
            for name,loss in [('mae',np.abs(error)),('mse',np.square(error))]:
                measured=np.bincount(cell_id,weights=loss,minlength=len(cells))/n
                stored=cells[name+'__'+method].to_numpy(float)
                diff=float(np.max(np.abs(measured-stored)))
                counts['maximum_scalar_loss_error']=max(counts['maximum_scalar_loss_error'],diff)
                np.testing.assert_allclose(measured,stored,atol=1e-9,rtol=1e-10)
        counts['scalar_values_reaggregated']+=predictions.size
        parameter=json.loads((part/'parameters.json').read_text())['tasks']
        old_parameter=json.loads((part/'inherited_parameters.json').read_text())
        infos=completion['inherited_infos']; geo=np.load(part/'geometry.npz')
        starts=np.r_[0,np.cumsum(n)]; by_design={i:g for i,g in cells.groupby('design_id',sort=False)}
        for (_,_,_), frame in ledger.groupby(['task','source','scenario'],sort=True):
            first=frame.iloc[0]; task=first.task
            z,base=geo['z_'+task[-2:]],geo['base_'+task[-2:]]
            query=np.array(json.loads(first.query_indices),int)
            pool=np.array(json.loads(first.pool_indices),int)
            p=parameter[task]; theta=p['theta']
            k,kq,c,cq=mk.matrices(z[pool],z[query],p)
            expected={}
            for policy in mk.NEW_POLICIES:
                take,_=mk.order(k,kq,c,cq,theta['alpha'],p['working_noise'],policy)
                expected[policy]=pool[take]
            source_frame=data[data.canonical_source_id.astype(str).eq(str(first.source))].copy()
            before=run.b.designs(source_frame)
            source_frame['C2_yield_percent']=np.arange(len(source_frame))*1e12
            after=run.b.designs(source_frame)
            if set(before)!=set(after):raise ValueError('Outcome-dependent source partition')
            for scenario in before:
                for a,b in zip(before[scenario][:2],after[scenario][:2]):np.testing.assert_array_equal(a,b)
            counts['actual_poisoned_source_frames']+=1
            for record in frame.itertuples():
                support=np.array(json.loads(record.support_indices),int)
                if len(set(support))!=10 or not set(support)<=set(pool) or set(support)&set(query):
                    raise ValueError('Invalid acquired sequence')
                counts['support_orders_checked']+=1
                if record.policy in expected:
                    np.testing.assert_array_equal(support,expected[record.policy])
                    counts['new_design_orders_reconstructed']+=1
                # Every deterministic support and the first random support are checked.
                if record.repeat!=0:continue
                for cell in by_design[record.design_id].itertuples():
                    take=support[:cell.k]; yy=y[take]
                    independent=independent_prediction(z[take],yy,z[query],theta)
                    actual=predictions[starts[cell.cell_id]:starts[cell.cell_id+1]]
                    error=float(np.max(np.abs(independent-actual[:,0])))
                    counts['maximum_independent_prediction_error']=max(
                        counts['maximum_independent_prediction_error'],error)
                    np.testing.assert_allclose(independent,actual[:,0],atol=1e-9,rtol=1e-10)
                    hp=old_parameter['controls'][task]['composition_vopt'][str(cell.k)]['rbf']
                    isotropic=dict(kind='isotropic',**hp)
                    np.testing.assert_allclose(independent_prediction(z[take],yy,z[query],isotropic),
                                               actual[:,1],atol=1e-9,rtol=1e-10)
                    frozen=run.old._frozen_pred(z,yy,base[take],query,take,base,infos[task],cell.k)
                    np.testing.assert_allclose(frozen['adaptive_blend'],actual[:,2],atol=1e-9)
                    np.testing.assert_allclose(frozen['local_ridge'],actual[:,3],atol=1e-9)
                    np.testing.assert_allclose(actual[:,4],np.full(len(query),yy.mean()),atol=1e-10)
                    np.testing.assert_allclose(actual[:,5],base[query],atol=1e-10)
                    iso_index=int(np.argmin(p['historical_group_equal_mae'][:9]))
                    np.testing.assert_allclose(independent_prediction(z[take],yy,z[query],p['atlas'][iso_index]),
                                               actual[:,6],atol=1e-9,rtol=1e-10)
                    counts['independent_kernel_query_values_reconstructed']+=3*len(query)
                    counts['frozen_blend_query_values_reconstructed']+=len(query)
        if part.name.startswith('ocm_'):
            prior=run.REFERENCE/'runs/history_v1/parts'/part.name
        else:
            prior=run.REFERENCE/'runs/new_cohorts/nishimura_measured_v1/parts/cohort'
        old_cells=pd.read_csv(prior/'cells.csv.gz')
        current=cells[cells.policy.eq('composition_vopt')]
        matched=current.merge(old_cells[old_cells.policy.eq('composition_vopt')],
            on=['task','source','scenario','k','repeat'],suffixes=('_new','_old'),validate='one_to_one')
        np.testing.assert_allclose(matched.mae__locked_local_rbf,
                                   matched.mae__local_rbf_retuned,atol=1e-10)
        np.testing.assert_allclose(matched.mae__frozen_blend,matched.mae__adaptive_blend,atol=1e-10)
        counts['legacy_locked_rbf_cells_matched']+=len(matched)
        counts['parts']+=1; all_cells.append(cells)
    counts.update(passed=True,audited_utc=run.utc(),
        independence='All scalar losses independently reaggregated. New kernel and locked RBF predictions reconstructed by a separate kernel implementation and sklearn; frozen blend uses inherited helpers. New design orders reuse selector; mathematical risk checks are separate.',
        new_backbone_refits=0)
    run.write_json(run_root/'verification/audit.json',counts)
    return pd.concat(all_cells,ignore_index=True),counts


def interval(frame,coverage=.99):
    if frame.leakage_group.nunique()<2:return None,None
    totals=frame.groupby('leakage_group',sort=True).delta.agg(['sum','count'])
    rng=np.random.default_rng(101011)
    draw=rng.integers(0,len(totals),size=(20000,len(totals)))
    boot=totals['sum'].to_numpy()[draw].sum(1)/totals['count'].to_numpy()[draw].sum(1)
    q=(1-coverage)/2
    return tuple(map(float,np.quantile(boot,[q,1-q])))


def aggregate(cells,run_root):
    values=[c for c in cells if c.startswith(('mae__','mse__'))]
    per_seed=cells.groupby(['source','leakage_group','task','scenario','policy','k','seed'],sort=True)[values].mean().reset_index()
    per_source=per_seed.groupby(['source','leakage_group','task','scenario','policy','k'],sort=True)[values].mean().reset_index()
    summary=per_source.groupby(['task','scenario','policy','k'],sort=True)[values].mean().reset_index()
    counts=per_source.groupby(['task','scenario','policy','k']).agg(n_sources=('source','nunique'),n_leakage_groups=('leakage_group','nunique')).reset_index()
    summary=summary.merge(counts,on=['task','scenario','policy','k'])
    run.save_csv(per_source,run_root/'aggregated/per_source.csv')
    run.save_csv(summary,run_root/'aggregated/summary.csv')
    primary=per_source[(per_source.task=='ocm_f1')&(per_source.scenario=='random')&(per_source.k==5)]
    candidate=primary[primary.policy=='hedged_mae']
    contrasts=[]
    for control in ('composition_vopt','kernel_vopt','robust_view_vopt','pool_vopt','centered_mae'):
        paired=candidate.merge(primary[primary.policy==control],on=['source','leakage_group'],suffixes=('_candidate','_control'),validate='one_to_one')
        paired['delta']=paired.mae__matched_kernel_candidate-paired.mae__matched_kernel_control
        low,high=interval(paired)
        contrasts.append(dict(control=control,candidate_mae=float(paired.mae__matched_kernel_candidate.mean()),
            control_mae=float(paired.mae__matched_kernel_control.mean()),delta_mae=float(paired.delta.mean()),
            family_adjusted_ci_low=low,family_adjusted_ci_high=high,n_sources=len(paired),
            n_leakage_groups=paired.leakage_group.nunique(),
            adjusted_support_for_advantage=high is not None and high<0,
            statistical_status='historical development diagnostic' if low is not None else 'one opened family: point estimate'))
    run.save_csv(pd.DataFrame(contrasts),run_root/'aggregated/primary_family.csv')
    run.write_json(run_root/'aggregated/aggregation.json',dict(completed_utc=run.utc(),
        reduction='average draws, then seeds within source, then equal source mean',
        primary='F1 matched-kernel predictor; k5 random query; hedged_mae vs five fixed controls',
        adjusted_intervals='99% percentile paired leakage-group bootstrap, 20000 draws; conditional development evidence',
        one_family_interval=None))
    return pd.DataFrame(contrasts)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='history_v1');args=parser.parse_args()
    root=ROOT/'runs'/args.run_name
    cells,checks=audit(root);primary=aggregate(cells,root)
    print(json.dumps(checks,indent=2));print(primary.to_string(index=False))


if __name__=='__main__':main()

"""Draw -> seed -> source aggregation; conditional leakage-group intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_cross_task as r
ROOT=r.ROOT


def bootstrap(frame,column,seed,draws=20000):
    means=frame.groupby('leakage_group',sort=True)[column].agg(['sum','count'])
    estimate=float(frame[column].mean())
    if len(means)<2:return dict(estimate=estimate,ci_low=None,ci_high=None,family_ci_low=None,family_ci_high=None,n_groups=len(means))
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(means),size=(draws,len(means)))
    values=means['sum'].to_numpy()[ix].sum(1)/means['count'].to_numpy()[ix].sum(1)
    q=np.quantile(values,[.025,.975,.005,.995])
    return dict(estimate=estimate,ci_low=float(q[0]),ci_high=float(q[1]),family_ci_low=float(q[2]),family_ci_high=float(q[3]),n_groups=len(means))


def paired(source,endpoint,scenario,k,policy,method,ref_policy,ref_method,primary=False):
    sub=source[(source.endpoint==endpoint)&(source.scenario==scenario)&(source.k==k)]
    a=sub[sub.policy==policy];d=sub[sub.policy==ref_policy]
    keys=['source','leakage_group'];g=a[keys+['mae__'+method]].merge(d[keys+['mae__'+ref_method]],on=keys,suffixes=('_a','_b'),validate='one_to_one')
    assert len(g)==len(a)==len(d)
    if method==ref_method:va=g['mae__'+method+'_a'];vd=g['mae__'+ref_method+'_b']
    else:va=g['mae__'+method];vd=g['mae__'+ref_method]
    g['delta']=va-vd
    interval=bootstrap(g,'delta',r.b.stable_seed('step7_bootstrap',endpoint,scenario,k,policy,method,ref_policy,ref_method))
    out=dict(endpoint=endpoint,scenario=scenario,k=k,policy=policy,method=method,reference_policy=ref_policy,reference_method=ref_method,
        mae=float(va.mean()),reference_mae=float(vd.mean()),delta_mae=interval.pop('estimate'),relative_reduction_percent=float(100*(1-va.mean()/vd.mean())),
        n_sources=len(g),n_improved=int(g.delta.lt(-1e-10).sum()),n_harmed=int(g.delta.gt(1e-10).sum()),primary_family=primary,**interval)
    if not primary:out['family_ci_low']=out['family_ci_high']=None
    return out


def main():
    dirs=sorted((ROOT/'results/parts').iterdir());completed=[json.loads((d/'completed.json').read_text()) for d in dirs]
    cells=pd.concat([pd.read_csv(d/'cells.csv.gz') for d in dirs],ignore_index=True)
    metrics=[f'{prefix}__{m}' for prefix in ['mae','mse','bounded_mae'] for m in r.METHODS]
    keys=['endpoint','target','scenario','source','leakage_group','policy','k']
    assert not cells.duplicated(keys+['seed','repeat']).any()
    per_seed=cells.groupby(keys+['seed'],as_index=False)[metrics+['n_test','support_rank','blend_weight','retuned_weight']].mean()
    seed_counts=per_seed.groupby(keys).seed.nunique();expected=sorted(set(cells.seed))
    assert seed_counts.eq(len(expected)).all()
    source=per_seed.groupby(keys,as_index=False)[metrics+['n_test','support_rank','blend_weight','retuned_weight']].mean()
    r.csv(cells,ROOT/'results/all_repeat_cells.csv.gz');r.csv(per_seed,ROOT/'results/per_source_per_seed.csv.gz');r.csv(source,ROOT/'results/per_source_metrics.csv')
    summary=[]
    for (endpoint,target,scenario,policy,k),g in source.groupby(['endpoint','target','scenario','policy','k'],sort=True):
        for method in r.METHODS:
            summary.append(dict(endpoint=endpoint,target=target,scenario=scenario,policy=policy,k=k,method=method,
                mae=float(g['mae__'+method].mean()),mse=float(g['mse__'+method].mean()),bounded_mae=float(g['bounded_mae__'+method].mean()),
                n_sources=len(g),n_groups=g.leakage_group.nunique(),n_query_rows=int(g.n_test.sum()),source_mean_rank=float(g.support_rank.mean())))
    r.csv(pd.DataFrame(summary),ROOT/'results/summary.csv')
    contrasts=[];primary=[]
    for (endpoint,scenario,k),g in source.groupby(['endpoint','scenario','k'],sort=True):
        for method in ['adaptive_blend','adaptive_blend_retuned','local_ridge_retuned','local_rbf_retuned']:
            for policy in r.acq.POLICIES[1:]:
                is_primary=scenario=='random' and k==5 and method=='adaptive_blend' and policy=='linear_vopt'
                row=paired(source,endpoint,scenario,k,policy,method,'random',method,is_primary);contrasts.append(row)
                if is_primary:primary.append(row)
        if k==5:
            for policy in ['temperature_cover','feature_cover','linear_pool_vopt']:
                contrasts.append(paired(source,endpoint,scenario,k,'linear_vopt','adaptive_blend',policy,'adaptive_blend'))
            for method in ['local_ridge_retuned','local_rbf_retuned']:
                contrasts.append(paired(source,endpoint,scenario,k,'linear_vopt','adaptive_blend_retuned','linear_vopt',method))
    r.csv(pd.DataFrame(contrasts),ROOT/'results/paired_contrasts.csv');r.csv(pd.DataFrame(primary),ROOT/'results/primary_family.csv')
    p=pd.concat([pd.read_csv(d/'label_permutation.csv.gz') for d in dirs],ignore_index=True)
    ps=p.groupby(['endpoint','scenario','policy','source','leakage_group'],as_index=False)[['aligned_mae','permuted_mae']].mean();ps['delta']=ps.permuted_mae-ps.aligned_mae
    rows=[]
    for (endpoint,scenario,policy),g in ps.groupby(['endpoint','scenario','policy'],sort=True):
        ci=bootstrap(g,'delta',r.b.stable_seed('step7_permutation_bootstrap',endpoint,scenario,policy))
        rows.append(dict(endpoint=endpoint,scenario=scenario,policy=policy,k=5,aligned_mae=float(g.aligned_mae.mean()),permuted_mae=float(g.permuted_mae.mean()),
            delta_mae=ci['estimate'],ci_low=ci['ci_low'],ci_high=ci['ci_high'],n_sources=len(g),n_groups=ci['n_groups'],scope='exploratory label-correspondence diagnostic; same support mean/marginal labels'))
    r.csv(p,ROOT/'results/all_label_permutations.csv.gz');r.csv(ps,ROOT/'results/label_permutation_per_source.csv');r.csv(pd.DataFrame(rows),ROOT/'results/label_permutation_summary.csv')
    verification=dict(parts=len(completed),endpoints=list(source.endpoint.unique()),seeds=expected,methods=list(r.METHODS),policies=list(r.acq.POLICIES),
        cells=len(cells),row_values=sum(d['n_prediction_rows'] for d in completed)*len(r.METHODS),designs=sum(d['n_designs'] for d in completed),
        new_backbone_fits=sum(d['new_backbone_fits'] for d in completed),new_inner_model_fits=0,archived_random_cells=sum(d['archived_random_cells'] for d in completed),
        archived_random_mae_maximum_difference=max(d['archived_random_mae_maximum_difference'] for d in completed),base_maximum_difference=max(d['base_maximum_difference'] for d in completed),
        weighting='query rows within each cell; draws within source/seed; seeds within source; sources equal. Bootstrap resamples leakage groups conditional on fitted models.',
        primary_family_size=5,primary_interval='percentile99 = Bonferroni-adjusted five-member nominal95 family; approximate conditional intervals',
        study_status='retrospective; no fresh independent experimental measurements',complete=len(completed)==125)
    r.write_json(ROOT/'verification/aggregation.json',verification)
    print(pd.DataFrame(primary)[['endpoint','mae','reference_mae','relative_reduction_percent','delta_mae','family_ci_low','family_ci_high','n_sources']].to_string(index=False),flush=True)
    print('Aggregation',json.dumps(verification),flush=True)


if __name__=='__main__':main()

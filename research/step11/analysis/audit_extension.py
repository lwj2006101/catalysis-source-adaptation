"""Independent scoring and sklearn reconstruction of every fixed-RBF prediction."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.kernel_ridge import KernelRidge
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_extension as e


def audit(root):
    totals=dict(passed=True,parts=0,cells=0,query_records=0,scalar_predictions=0,
                sklearn_reconstructed_values=0,max_loss_difference=0.,max_reconstruction_difference=0.)
    allcells=[]
    for part in sorted((root/'parts').glob('*')):
        if not (part/'completed.json').exists():continue
        f=pd.read_csv(part/'cells.csv.gz');l=pd.read_csv(part/'support_ledger.csv.gz')
        d=pd.read_csv(part/'evaluated_input.csv.gz');g=np.load(part/'geometry.npz')
        a=np.load(part/'row_predictions.npz');p=a['predictions'];ci=a['cell_id'];qi=a['query_index']
        y=d.C2_yield_percent.to_numpy(float);z=g['z'];methods=a['methods'].tolist()
        assert methods==list(e.METHODS) and len(p)==len(ci)==len(qi) and ci.max()==len(f)-1
        n=np.bincount(ci,minlength=len(f));assert np.array_equal(n,f.n_test)
        for j,m in enumerate(methods):
            errors=p[:,j]-y[qi]
            for name,values in [('mae',np.abs(errors)),('mse',errors*errors)]:
                rebuilt=np.bincount(ci,weights=values,minlength=len(f))/n
                delta=float(np.max(np.abs(rebuilt-f[name+'__'+m])))
                totals['max_loss_difference']=max(totals['max_loss_difference'],delta)
                np.testing.assert_allclose(rebuilt,f[name+'__'+m],rtol=1e-10,atol=1e-9)
        # Independent Euclidean-distance construction, kernel centering and solver.
        for design in l.itertuples():
            pool=json.loads(design.pool_indices);q=np.array(json.loads(design.query_indices),int)
            order=np.array(json.loads(design.support_indices),int)
            assert not set(pool)&set(q) and len(set(order))==10 and set(order)<=set(pool)
            rows=f[f.design_id==design.design_id]
            for row in rows.itertuples():
                s=order[:row.k];alpha=row.fixed_alpha;gamma=row.fixed_gamma/z.shape[1]
                kss=rbf_kernel(z[s],z[s],gamma=gamma);kqs=rbf_kernel(z[q],z[s],gamma=gamma)
                center=KernelCenterer();kc=center.fit_transform(kss);qkc=center.transform(kqs)
                model=KernelRidge(alpha=alpha,kernel='precomputed').fit(kc,y[s]-y[s].mean())
                independent=model.predict(qkc)+y[s].mean()
                ix=np.where(ci==row.cell_id)[0]
                assert np.array_equal(qi[ix],q)
                delta=float(np.max(np.abs(independent-p[ix,0])))
                totals['max_reconstruction_difference']=max(totals['max_reconstruction_difference'],delta)
                np.testing.assert_allclose(independent,p[ix,0],rtol=2e-9,atol=2e-8)
                totals['sklearn_reconstructed_values']+=len(q)
        complete=json.loads((part/'completed.json').read_text())
        assert complete['cells']==len(f) and complete['scalar_predictions']==p.size
        f['part']=part.name
        if 'seed' in complete:f['seed']=complete['seed'];f['fold']=complete['fold']
        allcells.append(f);totals['parts']+=1;totals['cells']+=len(f)
        totals['query_records']+=len(p);totals['scalar_predictions']+=p.size
        print('AUDITED',part.name,len(f),flush=True)
    assert totals['parts']>0
    if root.name=='operator_history_v1':assert totals['parts']==25
    df=pd.concat(allcells,ignore_index=True)
    # Repeated seeds/folds first averaged within sources, never independent replicates.
    if 'seed' in df:
        df=df.groupby(['source','leakage_group','scenario','policy','k','seed'],as_index=False).mean(numeric_only=True)
        ps=df.groupby(['source','leakage_group','scenario','policy','k'],as_index=False).mean(numeric_only=True)
        e.csv(ps,root/'aggregated/per_source.csv')
        summary=ps.groupby(['scenario','policy','k'],as_index=False).mean(numeric_only=True)
    else:
        ps=df.groupby(['partition','scenario','policy','k'],as_index=False).mean(numeric_only=True)
        e.csv(ps,root/'aggregated/per_partition.csv')
        summary=ps.groupby(['scenario','policy','k'],as_index=False).mean(numeric_only=True)
    e.csv(summary,root/'aggregated/summary.csv');e.write(root/'verification/audit.json',totals)
    return totals


if __name__=='__main__':
    name=sys.argv[1] if len(sys.argv)>1 else 'operator_history_v1'
    print(json.dumps(audit(ROOT/'runs'/name),indent=2))

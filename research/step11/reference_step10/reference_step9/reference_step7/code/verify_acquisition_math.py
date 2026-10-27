"""Independent primal posterior, scalar greedy and sklearn predictor checks."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
import run_cross_task as r


def main():
    rng=np.random.default_rng(7421);zp=rng.normal(size=(24,7));zq=rng.normal(size=(13,7));fp=np.column_stack([np.full(len(zp),10.),zp]);fq=np.column_stack([np.full(len(zq),10.),zq])
    alphas={str(k):[.1,1.,10.,100.,1.][i] for i,k in enumerate(r.acq.BUDGETS)};largest=0.;greedy_steps=0
    for policy in ['linear_vopt','linear_pool_vopt']:
        order,trace=r.acq.select_order(zp,zq,policy,alphas);query=fp if policy=='linear_pool_vopt' else fq;selected=[]
        for size in range(1,11):
            alpha=alphas[str(min(k for k in r.acq.BUDGETS if k>=size))];candidates=[i for i in range(len(zp)) if i not in selected];scores=[]
            for candidate in candidates:
                f=fp[selected+[candidate]];cov=np.linalg.inv(np.eye(f.shape[1])+f.T@f/alpha)
                scores.append(float(np.einsum('qi,ij,qj->',query,cov,query)/len(query)))
            chosen=candidates[int(np.argmin(scores))];assert chosen==order[size-1];selected.append(chosen)
            largest=max(largest,abs(min(scores)-trace[size-1]['objective']));greedy_steps+=1
    za=zp[:5];ya=rng.normal(size=5);pred,loo=r.b.ridge_query_and_loo(za,ya,zq,1.)
    np.testing.assert_allclose(pred,Ridge(alpha=1.).fit(za,ya).predict(zq),atol=1e-12)
    manual=np.array([Ridge(alpha=1.).fit(np.delete(za,i,0),np.delete(ya,i)).predict(za[i:i+1])[0] for i in range(5)])
    np.testing.assert_allclose(loo,manual,atol=1e-12)
    for policy in ['temperature_cover','feature_cover','linear_pool_vopt']:
        a,_=r.acq.select_order(zp,zq,policy,alphas);d,_=r.acq.select_order(zp,rng.normal(size=zq.shape)*1e5,policy,alphas);np.testing.assert_array_equal(a,d)
    result=dict(passed=True,primal_posterior_greedy_steps=greedy_steps,maximum_primal_dual_variance_difference=largest,
        ridge_sklearn_primal_maximum_difference=float(np.max(np.abs(pred-Ridge(alpha=1.).fit(za,ya).predict(zq)))),
        ridge_loo_refit_maximum_difference=float(np.max(np.abs(loo-manual))),query_agnostic_rules_verified=['temperature_cover','feature_cover','linear_pool_vopt'])
    out=Path(os.environ.get('CROSS_TASK_VERIFY_REPORT_DIR',r.ROOT/'verification'));out.mkdir(parents=True,exist_ok=True);r.write_json(out/'mathematical_checks.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

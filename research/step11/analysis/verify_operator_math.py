"""Numerical validation of the exact spectral covariance-ball identity."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import robust_operator as r
from run_extension import write


def main():
    rng=np.random.default_rng(11011);x=rng.normal(size=(13,5));n=4;q=9;alpha=.1;epsilon=.3
    k=r.kernel(x,x,.8);s=np.arange(n);t=np.arange(n,13)
    risk,amp,w=r.components(k[np.ix_(s,s)],k[np.ix_(t,s)],alpha)
    np.testing.assert_allclose(w.sum(1),1,atol=1e-12)
    cov=k+alpha*np.eye(len(x));B=np.column_stack([-w,np.eye(q)])
    nominal=np.trace(B@cov@B.T)/q
    np.testing.assert_allclose(nominal,risk,atol=1e-12)
    worst=np.trace(B@(cov+epsilon*np.eye(len(x)))@B.T)/q
    np.testing.assert_allclose(worst,risk+epsilon*(1+amp),atol=1e-12)
    gaps=[]
    for _ in range(1000):
        v=rng.normal(size=(len(x),len(x)));v=(v+v.T)/2;v/=np.linalg.norm(v,2)
        delta=epsilon*v
        actual=np.trace(B@(cov+delta)@B.T)/q
        assert abs(actual-nominal)<=epsilon*(1+amp)+1e-12
        gaps.append(abs(actual-nominal))
    draws=rng.multivariate_normal(np.zeros(len(x)),cov+epsilon*np.eye(len(x)),size=50000)
    losses=np.square(draws@B.T).mean(1);se=losses.std(ddof=1)/np.sqrt(len(losses))
    assert abs(losses.mean()-worst)<6*se
    shift=np.ones(len(x))*17.2
    np.testing.assert_allclose(B@shift,0,atol=1e-12)
    output=dict(passed=True,spectral_ball_upper_bound_tight=True,random_symmetric_perturbations=1000,
                gaussian_simulations=50000,simulations_are_not_catalysis_data=True,
                constant_source_shift_cancelled=True,nominal_risk=float(risk),amplification=float(amp),
                worst_working_risk=float(worst),simulation_mean=float(losses.mean()),simulation_se=float(se))
    write(ROOT/'reports/operator_math_checks.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()

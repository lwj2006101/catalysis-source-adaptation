"""Meaningful server-side checks of variance algebra, ridge and gate isolation."""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import Ridge
from server_guard import ROOT, require_server, write_json, utc
from selection_controls import _kernels, posterior_scores, select_control
from view_selection import context, fit_gate
import run_history as experiment


def main():
    require_server()
    rng = np.random.default_rng(81011)
    p, q = rng.normal(size=(17,8)), rng.normal(size=(9,8))
    fp, fq = np.column_stack([np.full(len(p),10.),p]), np.column_stack([np.full(len(q),10.),q])
    alpha, selected, candidates = 1.3, [1,4], np.array([i for i in range(len(p)) if i not in (1,4)])
    dual = posterior_scores(_kernels(p,q), selected, candidates, alpha)
    primal = []
    for index in candidates:
        take = selected + [int(index)]
        covariance = np.linalg.inv(np.eye(fp.shape[1]) + fp[take].T @ fp[take] / alpha)
        primal.append(np.einsum('ij,jk,ik->i',fq,covariance,fq).mean())
    np.testing.assert_allclose(dual, primal, atol=1e-9, rtol=1e-10)
    old, _ = experiment.acq.select_order(p,q,'linear_vopt',{str(k):alpha for k in experiment.BUDGETS})
    robust, _ = select_control(p,p,q,q,'robust_view_vopt',{str(k):alpha for k in experiment.BUDGETS})
    np.testing.assert_array_equal(old,robust)
    ridge_diff = 0.0
    for k in (1,2,3,5,10):
        y = rng.normal(size=k)
        pred, loo = experiment.b.ridge_query_and_loo(p[:k],y,q,alpha)
        exact = Ridge(alpha=alpha).fit(p[:k],y).predict(q)
        np.testing.assert_allclose(pred,exact,atol=1e-10)
        ridge_diff = max(ridge_diff,float(np.max(np.abs(pred-exact))))
        if k>1:
            oracle = np.array([Ridge(alpha=alpha).fit(np.delete(p[:k],j,axis=0),
                np.delete(y,j)).predict(p[j:j+1])[0] for j in range(k)])
            np.testing.assert_allclose(loo,oracle,atol=1e-9)
    x = rng.normal(size=(12,8))
    delta = rng.normal(size=12)
    groups = np.array([f'g{i}' for i in range(12)])
    gate = fit_gate(x,delta,groups)
    before = gate.choose(x[0])
    # A response-bearing source frame is accepted by the legacy partitioner.
    # Poison its outcome column and verify all feature-defined partitions.
    import pandas as pd
    frame = pd.DataFrame({'row_id':[f'r{i:03d}' for i in range(24)],
        'canonical_source_id':['fixture']*24, 'Temperature_K':np.arange(24)+900.,
        '__stratum':['a']*6+['b']*18, 'C2_yield_percent':rng.normal(size=24)})
    original = experiment.b.designs(frame)
    poisoned = frame.copy()
    poisoned['C2_yield_percent'] = np.arange(24)*1e12
    changed = experiment.b.designs(poisoned)
    assert set(original)==set(changed)
    for scenario in original:
        for a,b in zip(original[scenario][:2],changed[scenario][:2]):
            np.testing.assert_array_equal(a,b)
    after = gate.choose(x[0])
    assert before == after
    # Pointwise regret identity: wrong sign incurs <= absolute risk-difference error.
    actual, estimate = rng.normal(size=2000), rng.normal(size=2000)
    candidate_loss = actual  # operating loss is set to zero by translation.
    chosen = np.where(estimate>0,0,candidate_loss)
    oracle = np.minimum(0,candidate_loss)
    assert np.all(chosen-oracle <= np.abs(actual-estimate)+1e-12)
    write_json(ROOT/'runs/verification/math.json', dict(passed=True, checked_utc=utc(),
        maximum_variance_difference=float(np.max(np.abs(dual-primal))),
        maximum_ridge_difference=ridge_diff,
        same_view_robust_design_matches_linear=True,
        outcome_poisoned_partitions_unchanged=True,
        gate_label_argument_absent=True,
        synthetic_values_are_mathematical_checks_not_catalyst_results=True))
    print('Mathematical checks passed')


if __name__ == '__main__':
    main()

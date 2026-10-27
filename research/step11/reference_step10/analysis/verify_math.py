"""Independent finite-matrix and simulation checks of the new risk calculation."""
from pathlib import Path
import sys
import json
import math
import numpy as np
from sklearn.kernel_ridge import KernelRidge
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import matched_kernel as mk
import run_matched as run


def main():
    rng = np.random.default_rng(101011)
    z = rng.normal(size=(30, 9))
    errors = []
    for k in (1, 2, 3, 5, 10):
        support, query = z[:k], z[15:]
        y = rng.normal(size=k)
        theta = dict(kind='isotropic', alpha=.1, gamma=1.)
        kss, kqs = mk.kernel(support, support, theta), mk.kernel(query, support, theta)
        pred = mk.predict(kss, kqs, y, theta['alpha'])
        h = np.eye(k) - np.ones((k, k)) / k
        qc = kqs - kqs.mean(1)[:, None] - kss.mean(0)[None, :] + kss.mean()
        oracle = KernelRidge(alpha=theta['alpha'], kernel='precomputed').fit(h @ kss @ h, y-y.mean())
        independent = y.mean() + oracle.predict(qc)
        legacy = run.old.comp.local_rbf(support, y, query, alpha=.1, gamma=1.)
        np.testing.assert_allclose(pred, independent, atol=1e-10)
        np.testing.assert_allclose(pred, legacy, atol=1e-10)
        errors.append(float(np.max(np.abs(pred-independent))))
        np.testing.assert_allclose(mk.predict(kss, kqs, y+1000, .1), pred+1000, atol=1e-9)

    theta = dict(kind='product', alpha=.1, gamma_op=1., gamma_chem=.1)
    prior = dict(kind='anova', alpha=.25, gamma_op=.1, gamma_chem=10.)
    support, query = z[:10], z[15:]
    w = mk.weights(mk.kernel(support, support, theta), mk.kernel(query, support, theta), .1)
    joint = mk.kernel(np.vstack([support, query]), np.vstack([support, query]), prior)
    herror = np.column_stack([-w, np.eye(len(query))])
    covariance = joint + .25 * np.eye(len(joint))
    independent_variance = np.diag(herror @ covariance @ herror.T)
    computed = mk.risk(w, joint[:10, :10], joint[:10, 10:], .25)
    np.testing.assert_allclose(computed, independent_variance, atol=1e-10)
    gaussian = rng.normal(size=(50000, len(joint))) @ np.linalg.cholesky(covariance).T + 70.
    error = gaussian[:, 10:] - gaussian[:, :10] @ w.T
    empirical_mae = float(np.abs(error).mean())
    expected_mae = float(np.sqrt(2/math.pi) * np.sqrt(computed).mean())
    if abs(empirical_mae-expected_mae) > .025:
        raise AssertionError('Working Gaussian MAE does not match simulation')

    grids = mk.candidates('ocm_f1')
    eigen_minima = [float(np.linalg.eigvalsh(mk.kernel(z[:18], z[:18], item)).min()) for item in grids]
    if min(eigen_minima) < -1e-9:
        raise AssertionError('Non-PSD block kernel')
    source_scores = rng.uniform(1, 4, size=(8, len(grids)))
    parameter = mk.choose_parameters(source_scores, np.array([f'g{i//2}' for i in range(8)]), grids)
    kp, kpq, cp, cpq = mk.matrices(z[:15], z[15:], parameter)
    all_orders = {}
    for policy in mk.NEW_POLICIES:
        take, trace = mk.order(kp, kpq, cp, cpq, parameter['theta']['alpha'], parameter['working_noise'], policy)
        if len(set(take)) != 10 or not set(take) <= set(range(15)):
            raise AssertionError('Non-nested/invalid selection')
        if [t['size'] for t in trace] != list(range(1,11)):
            raise AssertionError('Missing greedy step')
        all_orders[policy] = take.tolist()
    run.write_json(ROOT/'runs/verification/math.json', dict(passed=True, checked_utc=run.utc(),
        centered_kernel_vs_sklearn_max_error=max(errors),
        centered_kernel_matches_previous_local_rbf=True, constant_shift_invariance=True,
        variance_vs_independent_joint_matrix_max_error=float(np.max(np.abs(computed-independent_variance))),
        monte_carlo_draws=50000, empirical_working_mae=empirical_mae, analytic_working_mae=expected_mae,
        minimum_block_kernel_eigenvalue=min(eigen_minima), all_nested_orders=all_orders,
        scientific_scope='finite mathematical/simulation checks, not catalytic measurements or confidence calibration'))
    print('Independent centered-kernel/risk/MAE checks passed')


if __name__ == '__main__':
    main()

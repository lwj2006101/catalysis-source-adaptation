"""Working risk for the actual local kernel estimator with a free intercept.

Ordinary/universal kriging and block kernels are established methods. The
experiment tests coupling them to source-local calibration and acquisition;
neither the algebra nor the Gaussian working model is a coverage guarantee.
"""
from __future__ import annotations
import itertools
import numpy as np

ALPHAS = (0.01, 0.1, 1.0)
GAMMAS = (0.1, 1.0, 10.0)
NEW_POLICIES = ('finite_gp_v', 'centered_mse', 'centered_mae',
                'hedged_mse', 'hedged_mae')


def candidates(task):
    grid = [dict(kind='isotropic', alpha=a, gamma=g)
            for a, g in itertools.product(ALPHAS, GAMMAS)]
    if task == 'ocm_f1':
        grid += [dict(kind=kind, alpha=a, gamma_op=go, gamma_chem=gc)
                 for kind, a, go, gc in itertools.product(
                     ('additive', 'product', 'anova'), ALPHAS, GAMMAS, GAMMAS)]
    return grid


def distances(x, y):
    return np.maximum(0., (np.square(x).sum(1)[:, None]
                          + np.square(y).sum(1)[None, :] - 2 * x @ y.T)
                      / max(x.shape[1], 1))


def distance_views(x, y):
    return {'isotropic': distances(x, y), 'op': distances(x[:, :6], y[:, :6]),
            'chem': distances(x[:, 6:], y[:, 6:])}


def from_distances(d, theta):
    if theta['kind'] == 'isotropic':
        return np.exp(-theta['gamma'] * d['isotropic'])
    op = np.exp(-theta['gamma_op'] * d['op'])
    chem = np.exp(-theta['gamma_chem'] * d['chem'])
    if theta['kind'] == 'additive':
        return .5 * (op + chem)
    if theta['kind'] == 'product':
        return op * chem
    if theta['kind'] == 'anova':
        return (op + chem + op * chem) / 3.
    raise ValueError('Unknown kernel kind')


def kernel(x, y, theta):
    return from_distances(distance_views(np.asarray(x, float), np.asarray(y, float)), theta)


def weights(kss, kqs, alpha):
    """BLUP weights; exactly the centered KRR predictor with a free intercept."""
    kss, kqs = np.asarray(kss, float), np.asarray(kqs, float)
    n = kss.shape[-1]
    inv = np.linalg.solve(kss + np.asarray(alpha)[..., None, None] * np.eye(n),
                          np.broadcast_to(np.eye(n), kss.shape))
    cinv = inv.sum(-1)
    denom = cinv.sum(-1)
    direct = kqs @ inv
    remainder = 1. - direct.sum(-1)
    result = direct + remainder[..., :, None] * (cinv / denom[..., None])[..., None, :]
    if not np.allclose(result.sum(-1), 1., atol=1e-9):
        raise ArithmeticError('Free-intercept weights do not preserve constants')
    return result


def predict(kss, kqs, support_y, alpha):
    return weights(kss, kqs, alpha) @ np.asarray(support_y, float)


def risk(w, css, csq, noise):
    """Error variance for observed query yields under a zero-shape-mean working GP.

The unknown constant source level cancels because every weight row sums to 1.
The response covariance may differ from the deployed estimator's kernel.
"""
    n = css.shape[-1]
    cobs = css + noise * np.eye(n)
    answer = 1. + noise - 2 * np.einsum('...qk,...kq->...q', w, csq)
    answer += np.einsum('...qk,...kl,...ql->...q', w, cobs, w)
    if answer.min() < -1e-7:
        raise ArithmeticError('Negative working error variance')
    return np.maximum(answer, 0.)


def choose_parameters(source_scores, source_groups, grid):
    """Source/group-equal scores from historical random-support validation only."""
    source_scores = np.asarray(source_scores, float)
    source_groups = np.asarray(source_groups, str)
    means = np.mean([source_scores[source_groups == g].mean(0)
                     for g in np.unique(source_groups)], axis=0)
    index = int(np.argmin(means))
    temperature = max(float(np.median(np.abs(means - np.median(means)))), 1e-8)
    logits = -(means - means.min()) / temperature
    ww = np.exp(logits - logits.max()); ww /= ww.sum()
    return dict(theta=grid[index], theta_index=index, atlas=grid,
                working_weights=ww.tolist(), working_noise=float(sum(
                    w * h['alpha'] for w, h in zip(ww, grid))),
                weight_temperature=temperature, historical_group_equal_mae=means.tolist(),
                n_historical_groups=len(np.unique(source_groups)),
                interpretation='empirical kernel-covariance averaging; not a calibrated posterior')


def matrices(zpool, zquery, parameter):
    dp, dq = distance_views(zpool, zpool), distance_views(zpool, zquery)
    theta = parameter['theta']
    kp, kpq = from_distances(dp, theta), from_distances(dq, theta)
    cp, cpq = np.zeros_like(kp), np.zeros_like(kpq)
    for weight, item in zip(parameter['working_weights'], parameter['atlas']):
        cp += weight * from_distances(dp, item)
        cpq += weight * from_distances(dq, item)
    return kp, kpq, cp, cpq


def order(kp, kpq, cp, cpq, alpha, noise, policy, maximum=10):
    """Label-free nested greedy design; caller sorts pool by stable row IDs."""
    if policy not in NEW_POLICIES:
        raise ValueError(policy)
    if maximum > len(kp) or kpq.shape[1] == 0:
        raise ValueError('Ineligible pool/query')
    selected, trace = [], []
    nominal = policy in ('centered_mse', 'centered_mae')
    if nominal:
        cp, cpq, noise = kp, kpq, alpha
    for size in range(1, maximum + 1):
        available = np.array([j for j in range(len(kp)) if j not in selected])
        scores = []
        for start in range(0, len(available), 64):
            batch = available[start:start + 64]
            take = np.column_stack([np.broadcast_to(np.asarray(selected, int),
                                      (len(batch), len(selected))), batch])
            kss = kp[take[:, :, None], take[:, None, :]]
            ksq = kpq[take]
            if policy == 'finite_gp_v':
                # Matched alpha/kernel, but the historical finite intercept prior.
                cov = kss + 100. + alpha * np.eye(size)
                cross = ksq + 100.
                solved = np.linalg.solve(cov, cross)
                score = 101. - np.einsum('bkq,bkq->b', cross, solved) / kpq.shape[1]
            else:
                w = weights(kss, ksq.transpose(0, 2, 1), alpha)
                variance = risk(w, cp[take[:, :, None], take[:, None, :]], cpq[take], noise)
                score = np.sqrt(variance).mean(-1) if policy.endswith('mae') else variance.mean(-1)
            scores.extend(score)
        scores = np.asarray(scores)
        position = int(np.argmin(scores)); chosen = int(available[position])
        selected.append(chosen)
        trace.append(dict(size=size, pool_position=chosen, objective=float(scores[position])))
    return np.asarray(selected, int), trace

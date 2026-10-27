"""Two additional established design baselines: kernel V and robust view V."""
from __future__ import annotations

import numpy as np

KERNEL_GAMMAS = (0.1, 1.0, 10.0)
BUDGETS = (1, 2, 3, 5, 10)


def _kernels(pool, query, gamma=None):
    pool, query = np.asarray(pool, float), np.asarray(query, float)
    if gamma is None:
        pp = 100.0 + pool @ pool.T
        pq = 100.0 + pool @ query.T
        diagonal = 100.0 + np.square(query).sum(1)
    else:
        dimension = max(pool.shape[1], 1)
        pp = 100.0 + np.exp(-gamma * np.square(pool[:, None] - pool[None]).sum(2) / dimension)
        pq = 100.0 + np.exp(-gamma * np.square(pool[:, None] - query[None]).sum(2) / dimension)
        diagonal = np.full(len(query), 101.0)
    return pp, pq, float(diagonal.mean())


def posterior_scores(kernel, selected, candidates, alpha):
    pp, pq, prior = kernel
    values = []
    for start in range(0, len(candidates), 64):
        batch = candidates[start:start + 64]
        takes = np.column_stack([np.broadcast_to(np.asarray(selected, int),
                                                 (len(batch), len(selected))), batch])
        gram = pp[takes[:, :, None], takes[:, None, :]]
        cross = pq[takes]
        solved = np.linalg.solve(gram + alpha * np.eye(len(selected) + 1)[None], cross)
        values.extend(prior - np.einsum('nkq,nkq->n', cross, solved) / pq.shape[1])
    values = np.asarray(values)
    if values.min() < -1e-7:
        raise ArithmeticError('Negative posterior variance exceeds numerical tolerance')
    return np.maximum(values, 0)


def select_control(op_pool, full_pool, op_query, full_query, policy,
                   alpha_by_k, gamma=1.0, maximum=10):
    assert policy in ('kernel_vopt', 'robust_view_vopt')
    assert maximum <= len(full_pool) and len(full_query)
    if policy == 'kernel_vopt':
        kernels = [_kernels(full_pool, full_query, gamma)]
    else:
        kernels = [_kernels(op_pool, op_query), _kernels(full_pool, full_query)]
    selected, trace = [], []
    for size in range(1, maximum + 1):
        candidates = np.array([i for i in range(len(full_pool)) if i not in selected])
        budget = min(k for k in BUDGETS if k >= size)
        alpha = float(alpha_by_k[str(budget)])
        scores = [posterior_scores(kernel, selected, candidates, alpha) for kernel in kernels]
        if policy == 'robust_view_vopt':
            score = np.maximum.reduce([v / max(kernel[2], 1e-12)
                                       for v, kernel in zip(scores, kernels)])
        else:
            score = scores[0]
        chosen = int(candidates[int(np.argmin(score))])
        selected.append(chosen)
        trace.append(dict(size=size, selected_pool_index=chosen,
                          objective=float(score.min()), alpha=alpha,
                          gamma=gamma if policy == 'kernel_vopt' else None))
    return np.array(selected), trace

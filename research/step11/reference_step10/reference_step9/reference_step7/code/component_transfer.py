"""Support-conditioned transfer of two frozen functional components.

The decomposition is a two-block empirical interventional Shapley allocation.
It describes a predictor, not a chemical mechanism. Only training references
are used. All target-time functions accept support labels, never query labels.
"""
from __future__ import annotations

import itertools
import numpy as np


def ridge_query_loo(za, ya, zt, alpha):
    za, ya, zt = np.asarray(za, float), np.asarray(ya, float), np.asarray(zt, float)
    xc = za - za.mean(0)
    ym = ya.mean(0)
    gram = xc @ xc.T
    inv = np.linalg.solve(gram + alpha * np.eye(len(za)), np.eye(len(za)))
    coef = inv @ (ya - ym)
    pred = ym + ((zt - za.mean(0)) @ xc.T) @ coef
    if len(za) == 1:
        return pred, None
    fitted = ym + gram @ coef
    hd = 1 / len(za) + np.einsum('ij,ji->i', gram, inv)
    denom = 1 - hd
    assert (denom > 1e-12).all()
    loo = ya - (ya - fitted) / (denom[:, None] if ya.ndim == 2 else denom)
    return pred, loo


def reference_indices(source_ids, row_ids, seed, maximum=32):
    """One randomly selected training row per source; cap with a seeded draw."""
    rng = np.random.default_rng(seed)
    ss, rr = np.asarray(source_ids, str), np.asarray(row_ids, str)
    sources = np.array(sorted(set(ss)))
    if len(sources) > maximum:
        sources = rng.choice(sources, maximum, replace=False)
    ans = []
    for source in sorted(sources):
        ids = np.flatnonzero(ss == source)
        ids = ids[np.argsort(rr[ids])]
        ans.append(int(rng.choice(ids)))
    return np.asarray(ans, int)


def decompose(model, z, reference, first_block, batch_rows=256):
    """Return exact additive allocation of interactions to two feature blocks.

    v0=mean f(reference); vA=mean f(x_A, reference_B), analogously vB.
    uA=.5*((vA-v0)+(f-vB)); uB=f-v0-uA.
    Sum reconstruction is exact even with a finite reference sample.
    """
    z, reference = np.asarray(z, float), np.asarray(reference, float)
    a = np.asarray(first_block, int)
    b = np.setdiff1d(np.arange(z.shape[1]), a)
    assert len(a) and len(b) and len(reference)
    base = model.predict(z)
    intercept = float(model.predict(reference).mean())

    def marginal(block):
        patterns, inverse = np.unique(z[:, block], axis=0, return_inverse=True)
        vals = np.empty(len(patterns))
        for start in range(0, len(patterns), batch_rows):
            part = patterns[start:start + batch_rows]
            mixed = np.broadcast_to(reference, (len(part), *reference.shape)).copy()
            mixed[:, :, block] = part[:, None, :]
            vals[start:start + len(part)] = model.predict(mixed.reshape(-1, z.shape[1])).reshape(len(part), -1).mean(1)
        return vals[inverse]

    va, vb = marginal(a), marginal(b)
    ua = .5 * ((va - intercept) + (base - vb))
    u = np.column_stack([ua, base - intercept - ua])
    np.testing.assert_allclose(intercept + u.sum(1), base, atol=1e-9, rtol=1e-12)
    return base, u, intercept


def support_geometry(za, ya, ua, zt, ut, alpha, projected=True):
    """Local fit, query innovation G, and cross-fitted support evidence D."""
    fit, loo = ridge_query_loo(za, np.column_stack([ya, ua]), zt, alpha)
    local = fit[:, 0]
    if projected:
        g = ut - fit[:, 1:]
        d = None if loo is None else ua - loo[:, 1:]
    else:
        g = ut - ua.mean(0)
        d = None if len(ua) == 1 else (ua - ua.mean(0)) * len(ua) / (len(ua) - 1)
    error = None if loo is None else ya - loo[:, 0]
    return local, g, d, error


def box_quadratic(matrix, vector):
    """Exact box QP by active-set enumeration for <=3 weights.

    Minimize w'Mw-2b'w on [0,1]^J. Positive-definite M gives a unique fit.
    """
    matrix, vector = np.asarray(matrix, float), np.asarray(vector, float)
    j = len(vector)
    assert j <= 3
    best_value, best = np.inf, None
    for status in itertools.product((-1, 0, 1), repeat=j):
        status = np.asarray(status)
        free = np.flatnonzero(status == -1)
        fixed = np.flatnonzero(status != -1)
        w = np.maximum(status, 0).astype(float)
        if len(free):
            m = matrix[np.ix_(free, free)]
            rhs = vector[free] - matrix[np.ix_(free, fixed)] @ w[fixed]
            try:
                w[free] = np.linalg.solve(m, rhs)
            except np.linalg.LinAlgError:
                w[free] = np.linalg.lstsq(m, rhs, rcond=None)[0]
        if (w < -1e-8).any() or (w > 1 + 1e-8).any():
            continue
        w = np.clip(w, 0, 1)
        value = float(w @ matrix @ w - 2 * vector @ w)
        if value < best_value - 1e-12:
            best_value, best = value, w
    assert best is not None
    return best


def fit_weights(d, error, prior, c, dscale, reliability=True, shared=False):
    prior = np.asarray(prior, float)
    if d is None or c == 'inf':
        return prior.copy(), dict(effective_rank=0, eigen_min=0., eigen_max=0.)
    gram = d.T @ d
    eig = np.linalg.eigvalsh(gram / dscale)
    diagnostic = dict(effective_rank=int((eig > 1e-8).sum()),
                      eigen_min=float(eig.min()), eigen_max=float(eig.max()))
    if shared:
        ds = d.sum(1)
        denom = float(ds @ ds) + float(c) * dscale
        w = (float(ds @ error) + float(c) * dscale * prior[0]) / denom
        return np.full(len(prior), np.clip(w, 0, 1)), diagnostic
    if reliability:
        precision = float(c) * dscale * np.linalg.solve(np.eye(len(prior)) + gram / dscale, np.eye(len(prior)))
    else:
        precision = float(c) * dscale * np.eye(len(prior))
    return box_quadratic(gram + precision, d.T @ error + precision @ prior), diagnostic


def local_rbf(za, ya, zt, alpha, gamma):
    """Centered kernel ridge with an unpenalized intercept."""
    dimension = max(za.shape[1], 1)
    sq = ((za[:, None, :] - za[None, :, :]) ** 2).sum(2) / dimension
    kt = np.exp(-gamma * sq)
    h = np.eye(len(za)) - np.ones((len(za), len(za))) / len(za)
    kc = h @ kt @ h
    beta = np.linalg.solve(kc + alpha * np.eye(len(za)), ya - ya.mean())
    qsq = ((zt[:, None, :] - za[None, :, :]) ** 2).sum(2) / dimension
    kq = np.exp(-gamma * qsq)
    kq = kq - kq.mean(1, keepdims=True) - kt.mean(0)[None, :] + kt.mean()
    return ya.mean() + kq @ beta


def augmented_ridge(za, ya, ua, zt, ut, alpha, scales):
    aa = np.column_stack([za, ua / scales])
    at = np.column_stack([zt, ut / scales])
    return ridge_query_loo(aa, ya, at, alpha)[0]


def predict_new(za, ya, ua, zt, ut, hp):
    local, g, d, error = support_geometry(za, ya, ua, zt, ut, hp['alpha'])
    output, diagnostics = {}, {}
    for method, parameters in hp['variants'].items():
        if method == 'component_no_projection':
            lm, gm, dm, em = support_geometry(za, ya, ua, zt, ut, hp['alpha'], projected=False)
        else:
            lm, gm, dm, em = local, g, d, error
        w, diagnostic = fit_weights(dm, em, parameters['prior'], parameters['c'], parameters['dscale'],
                                   parameters['reliability'], parameters['shared'])
        output[method] = lm + gm @ w
        diagnostics[method] = dict(weight_0=float(w[0]), weight_1=float(w[1]), **diagnostic)
    output['component_fixed_prior'] = local + g @ np.asarray(hp['variants']['component_transfer']['prior'])
    output['residual_same_alpha'] = local + g.sum(1)
    output['local_rbf'] = local_rbf(za, ya, zt, **hp['rbf'])
    output['component_feature_ridge'] = augmented_ridge(za, ya, ua, zt, ut, hp['augmented_alpha'], np.asarray(hp['component_scales']))
    assert all(np.isfinite(v).all() for v in output.values())
    return output, diagnostics

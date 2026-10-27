"""Historical-source selection of acquisition views, with no target outcomes.

The candidate adapts established meta-active-learning ideas to source-local
catalysis calibration. It is not claimed to invent multi-view active learning.
Its benefit over fixed composition V-optimal selection remains untested.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge

CONTEXT_NAMES = (
    'log_pool_size', 'log_query_size', 'log_operating_variation',
    'log_chemistry_variation', 'log_query_chemistry_novelty',
    'chemistry_unexplained_by_operating', 'log_chemistry_mean_shift',
    'view_order_disagreement_at_five',
)
META_ALPHA = 10.0
MARGIN_QUANTILE = 0.8


def context(op_pool, chem_pool, op_query, chem_query, op_order, chem_order):
    """A fixed eight-dimensional, label-free context; views may be wide."""
    x, c, xq, cq = [np.asarray(v, float) for v in
                     (op_pool, chem_pool, op_query, chem_query)]
    assert len(x) == len(c) and len(xq) == len(cq) and len(xq)
    d = max(c.shape[1], 1)
    xc = x - x.mean(0)
    cc = c - c.mean(0)
    op_var = float(np.square(xc).sum() / (len(x) * max(x.shape[1], 1)))
    chem_var = float(np.square(cc).sum() / (len(x) * d))
    if c.shape[1]:
        coefficients = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ cc)
        residual = cc - xc @ coefficients
        unexplained = float(np.square(residual).sum() /
                            max(float(np.square(cc).sum()), 1e-12))
        novelty = []
        for start in range(0, len(cq), 64):
            distance = np.square(cq[start:start + 64, None] - c[None]).sum(2) / d
            novelty.extend(distance.min(1))
        novelty = float(np.mean(novelty) / max(chem_var, 1e-8))
        shift = float(np.square(cq.mean(0) - c.mean(0)).mean() /
                      max(chem_var, 1e-8))
    else:
        unexplained = novelty = shift = 0.0
    a, b = set(map(int, op_order[:5])), set(map(int, chem_order[:5]))
    disagreement = 1 - len(a & b) / max(len(a | b), 1)
    values = np.array([np.log1p(len(x)), np.log1p(len(xq)), np.log1p(op_var),
                       np.log1p(chem_var), np.log1p(min(novelty, 1e8)),
                       np.clip(unexplained, 0, 1), np.log1p(min(shift, 1e8)),
                       disagreement])
    assert np.isfinite(values).all()
    return values


@dataclass
class ViewGate:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    margin: float
    fixed_view: str
    n_groups: int
    calibration: str

    def choose(self, features, conservative=True):
        estimate = float(((np.asarray(features) - self.mean) / self.scale) @
                         self.coef + self.intercept)
        threshold = self.margin if conservative else 0.0
        # delta = loss(chemistry acquisition) - loss(operating acquisition).
        view = 'operating' if estimate > threshold else 'composition'
        return view, dict(estimated_delta_mae=estimate, threshold=threshold,
                          selected_view=view, n_historical_groups=self.n_groups)

    def as_dict(self):
        return {name: (value.tolist() if isinstance(value, np.ndarray) else value)
                for name, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, value):
        value = value.copy()
        for name in ('mean', 'scale', 'coef'):
            value[name] = np.asarray(value[name], float)
        return cls(**value)


def _fit(x, delta, groups):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    model = Ridge(alpha=META_ALPHA).fit((x - mean) / scale, delta,
                                       sample_weight=weights)
    return mean, scale, model.coef_, float(model.intercept_)


def fit_gate(x, delta, groups):
    """Group-held-out empirical errors set a fixed conservative margin.

    This margin is NOT a conformal guarantee, a confidence interval, or a
    distribution-free certificate. Inherited adapter hyperparameters were
    selected on all outer-training sources; the margin is diagnostic only.
    """
    x, delta, groups = np.asarray(x, float), np.asarray(delta, float), np.asarray(groups, str)
    assert len(x) == len(delta) == len(groups) and len(x)
    assert np.isfinite(x).all() and np.isfinite(delta).all()
    unique = np.unique(groups)
    group_delta = np.array([delta[groups == group].mean() for group in unique])
    fixed = 'composition' if group_delta.mean() <= 0 else 'operating'
    if len(unique) < 6:
        return ViewGate(x.mean(0), np.ones(x.shape[1]), np.zeros(x.shape[1]),
                        -1.0, 0.0, fixed, len(unique),
                        'fewer than six groups: composition fallback; no learned gate')
    error = np.empty(len(x))
    for group in unique:
        held = groups == group
        mean, scale, coef, intercept = _fit(x[~held], delta[~held], groups[~held])
        error[held] = np.abs(((x[held] - mean) / scale) @ coef + intercept - delta[held])
    group_error = np.array([error[groups == group].mean() for group in unique])
    margin = float(np.quantile(group_error, MARGIN_QUANTILE, method='higher'))
    mean, scale, coef, intercept = _fit(x, delta, groups)
    return ViewGate(mean, scale, coef, intercept, margin, fixed, len(unique),
                    'empirical leave-leakage-group-out error margin; no coverage claim')

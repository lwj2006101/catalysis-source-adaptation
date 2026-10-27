"""Reference guards for the v5.1 revision, not a complete experiment runner.

Synthetic tests for these helpers do not validate a real training pipeline.
Fit RoleCompositionEncoder only on the current allowed training partition.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

OPERATING = [
    'Temperature_K', 'CH4_partial_pressure_bar', 'O2_partial_pressure_bar',
    'CH4_O2_ratio', 'total_pressure_bar', 'contact_time_s',
]
ROLES = {'cation': [f'Cation {i}' for i in range(1, 5)],
         'support': [f'Support {i}' for i in range(1, 3)]}
COMPOSITION_COLUMNS = [c for slots in ROLES.values() for s in slots for c in (s, s + ' mol%')]
TARGET_BOUNDS = {
    'CO2_conversion_percent': (0.0, 100.0),
    'Methanol_selectivity_percent': (0.0, 100.0),
    'C2_yield_percent': (0.0, 100.0),
    'STY_mgMeOH_per_h_per_gcat': (0.0, None),
}


def stable_seed(*parts: object) -> int:
    """Keep the original v5 seed serialization, including whitespace."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts, default=str).encode()).digest()[:4], 'big')


def fixed_partition(row_ids: Sequence[str], source: str, task: str,
                    allocation: int = 0) -> tuple[list[str], list[str]]:
    """Partition IDs without accepting labels. Allocation 0 matches v5 design()."""
    ids = sorted(str(x) for x in row_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('row_ids must be nonempty and unique')
    if allocation < 0:
        raise ValueError('allocation must be nonnegative')
    ntest = max(5, math.ceil(0.25 * len(ids)))
    if ntest >= len(ids):
        return ids, []
    seed = stable_seed('test', source) if allocation == 0 else stable_seed('test_v51', task, source, allocation)
    order = np.random.default_rng(seed).permutation(ids).tolist()
    return order[:ntest], order[ntest:]


def nested_prefixes(pool: Sequence[str], seed: int,
                    budgets: Sequence[int] = (1, 2, 3, 5, 10)) -> dict[int, list[str]]:
    ids = sorted(str(x) for x in pool)
    if len(ids) != len(set(ids)):
        raise ValueError('pool has duplicate IDs')
    if not budgets or min(budgets) <= 0 or max(budgets) > len(ids):
        raise ValueError('budget must be positive and cannot exceed pool size')
    order = np.random.default_rng(seed).permutation(ids).tolist()
    return {int(k): order[:k] for k in budgets}


def bounded_prediction(pred: Sequence[float], target: str) -> np.ndarray:
    """Secondary bound sensitivity only; never change ground-truth labels."""
    if target not in TARGET_BOUNDS:
        raise ValueError(f'No registered target bounds: {target}')
    p = np.asarray(pred, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction')
    lo, hi = TARGET_BOUNDS[target]
    return np.maximum(p, lo) if hi is None else np.clip(p, lo, hi)


def offset(pc: Sequence[float], yc: Sequence[float], pt: Sequence[float]) -> np.ndarray:
    a, b, t = (np.asarray(x, float) for x in (pc, yc, pt))
    if not len(a) or a.shape != b.shape:
        raise ValueError('Calibration vectors must be nonempty and aligned')
    return t + (b - a).mean()


def affine(pc: Sequence[float], yc: Sequence[float], pt: Sequence[float],
           lam: float, scale: float) -> np.ndarray:
    a, b, t = (np.asarray(x, float) for x in (pc, yc, pt))
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Calibration vectors must be nonempty and aligned')
    if not all(np.isfinite(x).all() for x in (a, b, t)) or not np.isfinite(scale):
        raise ValueError('Nonfinite input')
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError('Positive finite lambda is required')
    scale = max(float(scale), 1e-8)
    z = (a - a.mean()) / scale
    slope_change = float(np.dot(z, b - a) / (np.dot(z, z) + lam))
    return t + (b - a).mean() + slope_change * (t - a.mean()) / scale


def permutation_indices(k: int, draws: int = 20, seed: int = 0) -> list[np.ndarray]:
    """Enumerate small permutation spaces, else sample uniformly with replacement.

    Identity permutations are allowed. k=1 has no informative permutation.
    This is a diagnostic control, not an automatically valid permutation test.
    """
    if k < 1 or draws < 1:
        raise ValueError('k and draws must be positive')
    if k == 1:
        return []
    if math.factorial(k) <= draws:
        return [np.array(p, dtype=int) for p in itertools.permutations(range(k))]
    rng = np.random.default_rng(seed)
    return [rng.permutation(k) for _ in range(draws)]


def paired_rows(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str],
                y_col: str = 'y_true', pred_col: str = 'prediction') -> pd.DataFrame:
    """Reject missing keys/rows/labels rather than silently intersecting results."""
    keys = list(keys)
    if not keys or y_col in keys or pred_col in keys:
        raise ValueError('Specify explicit identity keys, excluding labels/predictions')
    if left.empty or right.empty:
        raise ValueError('Empty comparison')
    required = keys + [y_col, pred_col]
    for name, frame in [('left', left), ('right', right)]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {sorted(missing)}')
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f'{name}: null or duplicate keys')
        if not np.isfinite(frame[[y_col, pred_col]].to_numpy(float)).all():
            raise ValueError(f'{name}: nonfinite labels/predictions')
    m = left[required].merge(right[required], on=keys, how='outer',
                             suffixes=('_left', '_right'), indicator=True, validate='one_to_one')
    if not m['_merge'].eq('both').all():
        raise ValueError('Different scoring populations; comparison blocked')
    if not np.array_equal(m[y_col + '_left'].to_numpy(), m[y_col + '_right'].to_numpy()):
        raise ValueError('Ground truth differs across methods')
    m['error_left'] = (m[y_col + '_left'] - m[pred_col + '_left']).abs()
    m['error_right'] = (m[y_col + '_right'] - m[pred_col + '_right']).abs()
    m['delta'] = m['error_left'] - m['error_right']
    return m.drop(columns='_merge')


def cluster_bootstrap(source_metrics: pd.DataFrame, value_col: str,
                      draws: int = 100_000, seed: int = 510927,
                      source_col: str = 'source', group_col: str = 'leakage_group',
                      chunk_size: int = 1000) -> dict:
    """Source-equal estimate, group-resampled interval, conditional on fixed fits.

    Input must already have one row per source for ONE comparison/panel.
    Retain within-group source weights; do not replace with group-equal MAE.
    """
    d = source_metrics[[source_col, group_col, value_col]].copy()
    if d.empty or d[source_col].duplicated().any() or d.isna().any().any():
        raise ValueError('Need exactly one complete row per source')
    if not np.isfinite(d[value_col].to_numpy(float)).all() or draws < 1 or chunk_size < 1:
        raise ValueError('Invalid bootstrap arguments')
    agg = d.groupby(group_col, sort=True)[value_col].agg(['sum', 'count'])
    point = float(d[value_col].mean())
    if len(agg) < 2:
        return {'estimate': point, 'ci95': None, 'status': 'INSUFFICIENT_GROUPS',
                'n_sources': len(d), 'n_groups': len(agg), 'draws': 0}
    sums, counts = agg['sum'].to_numpy(float), agg['count'].to_numpy(int)
    rng = np.random.default_rng(seed)
    vals = np.empty(draws)
    for start in range(0, draws, chunk_size):
        n = min(chunk_size, draws - start)
        idx = rng.integers(0, len(agg), size=(n, len(agg)))
        vals[start:start + n] = sums[idx].sum(1) / counts[idx].sum(1)
    return {'estimate': point, 'ci95': np.quantile(vals, [.025, .975]).tolist(),
            'status': 'OK', 'n_sources': len(d), 'n_groups': len(agg), 'draws': draws,
            'seed': seed, 'scope': 'conditional_on_fixed_models_and_observed_sources'}


def _token(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Sodium 'Na' is a valid element token, never a missing-value marker.
    if not s or s in {'n.a.', 'N/A', 'NA', 'nan', 'None', 'none'}:
        return None
    return s


def composition_record(row: pd.Series) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Role-aware, slot-order-invariant descriptor, preserving reported semantics.

    Values are (reported indicator, known mol fraction sum, missing-amount count).
    Unknown composition is not physically imputed. Roles are not renormalized.
    """
    record = {}
    for role, slots in ROLES.items():
        terms: dict[str, list[float]] = {}
        for slot in slots:
            token = _token(row[slot])
            amount = row[slot + ' mol%']
            missing = pd.isna(amount)
            if token is None:
                if not missing:
                    raise ValueError(f'Amount without species label in {slot}')
                continue
            vals = terms.setdefault(token, [1.0, 0.0, 0.0])
            if missing:
                vals[2] += 1.0
            else:
                amount = float(amount)
                if not np.isfinite(amount) or amount < 0 or amount > 100.0001:
                    raise ValueError(f'Invalid reported mol percent in {slot}')
                vals[1] += amount / 100.0
        record[role] = {k: tuple(v) for k, v in sorted(terms.items())}
    return record


class RoleCompositionEncoder(TransformerMixin, BaseEstimator):
    """Fit vocabulary on training X only; track unseen species explicitly.

    This encodes reported cation/support slots only. It does NOT assert that a
    blank slot proves physical absence, infer oxidation states, identify a
    preparation method, or normalize cations plus supports to a common total.
    """
    def fit(self, X: pd.DataFrame, y: object = None):
        self._validate_frame(X)
        if X.empty:
            raise ValueError('Cannot fit an empty training partition')
        records = [composition_record(row) for _, row in X.iterrows()]
        self.vocabulary_ = {role: sorted({t for r in records for t in r[role]}) for role in ROLES}
        self.feature_names_ = []
        for role in ROLES:
            for token in self.vocabulary_[role] + ['__UNSEEN__']:
                for channel in ['reported', 'known_mol_fraction', 'missing_amount_count']:
                    self.feature_names_.append(f'chem_{role}:{token}:{channel}')
            self.feature_names_.append(f'chem_{role}:no_reported_species')
        return self

    @staticmethod
    def _validate_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Pass a DataFrame with named raw composition columns')
        missing = set(COMPOSITION_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f'Missing composition columns: {sorted(missing)}')

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_frame(X)
        if not hasattr(self, 'vocabulary_'):
            raise ValueError('Encoder not fitted')
        output = []
        for _, row in X.iterrows():
            record = composition_record(row)
            values = []
            for role in ROLES:
                terms = record[role]
                unknown = [vals for token, vals in terms.items() if token not in self.vocabulary_[role]]
                for token in self.vocabulary_[role]:
                    values.extend(terms.get(token, (0., 0., 0.)))
                values.extend(np.array(unknown, float).sum(0).tolist() if unknown else [0., 0., 0.])
                values.append(float(not terms))
            output.append(values)
        return np.asarray(output, float).reshape(len(X), len(self.feature_names_))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, 'feature_names_'):
            raise ValueError('Encoder not fitted')
        return np.asarray(self.feature_names_, dtype=object)

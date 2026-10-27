"""CO2 v5.1 protocol engine.

Evolves the frozen v4.1/v5 engine under the 2026-09-27 dated amendment
(01_MASTER_PROMPT.md + 02_EXECUTION_CONTRACT.json). Numeric behaviour for the
F0 operating representation and percent targets is intentionally bitwise
identical to the frozen engine; changes are limited to:

  1. three explicitly named panels (zero_all_sources / main_fixed_cohort /
     low_budget_fixed_cohort) that never mix denominators;
  2. target-specific bound sensitivity (percent -> [0,100], STY -> [0,inf))
     replacing the universal [0,100] clip; primary predictions stay unclipped;
  3. k=0 zero-label rows on the main cohort;
  4. a DataFrame-in pipeline so the F1 composition representation
     (RoleCompositionEncoder, vocabulary fitted on the current training
     partition only) flows through the same model-selection procedure;
  5. per-inner-fold preprocessors instead of one fixed-width scaled matrix;
  6. explicit Temperature_K univariate transform for temperature_ridge;
  7. richer cache identity in done.json;
  8. per-part adapter artifacts so E2/E3 never refit the base model.

Per-task configuration travels through the V51_TASK_CONFIG environment
variable (a frozen JSON file per task/representation, written by run_task.py);
parent and loky worker processes re-import this module, so the frozen file is
the single source of truth.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

from v51_contracts import (COMPOSITION_COLUMNS, OPERATING, ROLES, TARGET_BOUNDS,
                           RoleCompositionEncoder, cluster_bootstrap, fixed_partition)

SEEDS = [42, 101, 202, 303, 404]
BUDGETS = [1, 2, 3, 5, 10]           # positive budgets; k=0 handled separately
K_CURVE = [0, 1, 2, 3, 5, 10]        # reporting curve (main cohort)
METHODS = ['base', 'debiased', 'train_mean', 'train_median', 'target_mean', 'target_median',
           'offset_mean', 'offset_median', 'affine', 'eb', 'target_ridge', 'temperature_ridge',
           'affine_shuffled_y', 'donor_offset', 'eb_shuffled_p']
LABEL_FREE = ['base', 'debiased', 'train_mean', 'train_median']
GRID = ([dict(kind='ridge', alpha=a) for a in [.1, 1., 10., 100.]] +
        [dict(kind='gbdt', n=n, depth=d) for n in [100, 300] for d in [2, 3]] +
        [dict(kind='gbdt', n=500, depth=8)])
CONFIG = dict(engine_version='5.1', outer_seeds=SEEDS, outer_folds=5, inner_folds=3,
              folds_to_run=[0, 1, 2, 3, 4], calibration_repeats=50, pseudo_repeats=10,
              primary_budget=5, secondary_budget=10, max_budget=10,
              test_fraction=.25, min_test=5, main_min_pool=10, low_budget_max=3,
              budgets=BUDGETS, bootstrap_draws=100_000, bootstrap_seed_base=510927,
              budget_unit='deduplicated literature record; independence not established',
              prediction_bounds='unclipped primary; target-specific bound sensitivity',
              scope='Internal retrospective source-disjoint benchmark; not external validation',
              smoke=False)

_TASK_CFG_PATH = os.environ.get('V51_TASK_CONFIG')
if _TASK_CFG_PATH:
    _cfg = json.loads(Path(_TASK_CFG_PATH).read_text(encoding='utf-8'))
    TASK = _cfg['task']
    FEATURE_SET = _cfg['feature_set']
    FEATURES = list(_cfg['features'])
    USE_COMPOSITION = bool(_cfg.get('use_composition', False))
    TARGETS = list(_cfg['targets'])
    INPUT_FILE = _cfg['input']
    CONFIG = dict(CONFIG, **_cfg.get('config_extra', {}))
else:  # never used directly for science; present for --test
    TASK, FEATURE_SET, FEATURES, USE_COMPOSITION, TARGETS, INPUT_FILE = (
        'unconfigured', 'F0', [], False, [], None)

MODEL_COLUMNS = FEATURES + (COMPOSITION_COLUMNS if USE_COMPOSITION else [])
assert 'Temperature_K' in FEATURES or not _TASK_CFG_PATH  # unconfigured import = helpers only


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, default=str).encode()).digest()[:4], 'big')


def boot_seed(*parts):
    """Contract base seed 510927 combined with the comparison keys."""
    return stable_seed(CONFIG['bootstrap_seed_base'], *parts)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def folds(groups, seed, k=5):
    g = np.asarray(groups, str)
    ids = np.array(sorted(set(g)))
    ids = np.random.default_rng(seed).permutation(ids)
    for held in np.array_split(ids, k):
        te = np.flatnonzero(np.isin(g, held))
        tr = np.flatnonzero(~np.isin(g, held))
        assert not set(g[tr]) & set(g[te])
        yield tr, te


def design(d):
    """Label-blind fixed T for allocation 0; identical to the frozen v4.1/v5 engine."""
    ids = d.sort_values('row_id').index.to_numpy()
    ntest = max(5, math.ceil(.25 * len(ids)))
    if ntest >= len(ids):
        return ids, np.array([], dtype=int)
    rng = np.random.default_rng(stable_seed('test', d.canonical_source_id.iloc[0]))
    ids = rng.permutation(ids)
    return ids[:ntest], ids[ntest:]


def build_preprocessor():
    """F0: exactly the frozen v4.1/v5 numeric pipeline on a plain numpy matrix
    (bitwise reproduction). F1: ColumnTransformer so the composition encoder
    sees named raw columns; vocabulary fitted on the current training fold."""
    numeric = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                            StandardScaler())
    if USE_COMPOSITION:
        from sklearn.compose import ColumnTransformer
        return ColumnTransformer([
            ('operating', numeric, list(FEATURES)),
            ('chemistry', RoleCompositionEncoder(), list(COMPOSITION_COLUMNS)),
        ])
    return numeric


def model_matrix(dframe: pd.DataFrame):
    """F0 -> plain numpy matrix (frozen engine path); F1 -> named DataFrame."""
    return dframe[FEATURES].to_numpy(float) if not USE_COMPOSITION else dframe[MODEL_COLUMNS]


def take_X(X, idx):
    """Row selection matching the frozen engine's numpy path for F0."""
    return X[idx] if not USE_COMPOSITION else X.iloc[idx]


def build_temp_preprocessor():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler())


def model(cfg):
    if cfg['kind'] == 'ridge':
        est = Ridge(alpha=cfg['alpha'])
    else:
        est = GradientBoostingRegressor(n_estimators=cfg['n'], max_depth=cfg['depth'],
                                        learning_rate=.05, subsample=.8, min_samples_split=3,
                                        random_state=42)
    return make_pipeline(build_preprocessor(), est)


def source_mae(y, p, s):
    return pd.DataFrame({'e': abs(y - p), 's': s}).groupby('s').e.mean().mean()


@dataclass
class Prior:
    mu: float
    tau2: float
    sigma2: float


def prior_fit(r, s):
    vals = [r[s == q] for q in sorted(set(s))]
    n = np.array([len(v) for v in vals])
    means = np.array([v.mean() for v in vals])
    sigma2 = max(sum(((v - v.mean()) ** 2).sum() for v in vals) / sum(n - 1), 1e-12)
    tau2 = max(float(np.var(means, ddof=1) - sigma2 * np.mean(1 / n)), 0.)
    return Prior(float(means.mean()), float(tau2), float(sigma2))


def affine(pc, yc, pt, lam, scale):
    scale = max(float(scale), 1e-8)
    z = (pc - pc.mean()) / scale
    e = float(np.dot(z, yc - pc) / (np.dot(z, z) + lam))
    return pt + (yc - pc).mean() + e * (pt - pc.mean()) / scale


def eb(prior, pc, yc, pt):
    k = len(yc)
    w = k * prior.tau2 / (prior.sigma2 + k * prior.tau2)
    return pt + prior.mu + w * ((yc - pc).mean() - prior.mu)


def target_ridge(Xa, ya, Xt, alpha):
    if len(ya) == 1:
        return np.full(len(Xt), ya[0])
    return Ridge(alpha=alpha).fit(Xa, ya).predict(Xt)


def farthest(pool, X, ids):
    pool = np.array(sorted(pool, key=lambda i: ids[i]))
    z = X[pool]
    chosen = [int(np.argmin(((z - z.mean(0)) ** 2).sum(1)))]
    distances = ((z - z[chosen[0]]) ** 2).sum(1)
    used = np.zeros(len(pool), bool)
    used[chosen[0]] = True
    while len(chosen) < len(pool):
        score = distances.copy(); score[used] = -np.inf
        nxt = int(np.argmax(score)); chosen.append(nxt); used[nxt] = True
        distances = np.minimum(distances, ((z - z[nxt]) ** 2).sum(1))
    return pool[chosen]


def inner_fit(d, target, outer_id):
    """Training-side selection: candidates by inner source-groups, calibrator
    hyper-parameters by pseudo new-source tasks. Every preprocessor is fitted
    on its own inner training partition; F1 vocabularies may differ per fold."""
    X = model_matrix(d.reset_index(drop=True))
    y = d[target].to_numpy(float)
    s = d.canonical_source_id.to_numpy(str)
    g = d.leakage_group_id.to_numpy(str)
    splits = list(folds(g, stable_seed('inner', outer_id), CONFIG['inner_folds']))
    candidates, oofs = [], []
    numeric = X if not USE_COMPOSITION else X[FEATURES].to_numpy(float)
    for cfg in GRID:
        p = np.empty(len(d))
        for tr, te in splits:
            assert not np.isnan(numeric[tr]).all(0).any()
            m = model(cfg).fit(take_X(X, tr), y[tr])
            p[te] = m.predict(take_X(X, te))
        candidates.append(float(source_mae(y, p, s)))
        oofs.append(p)
    selected = int(np.argmin(candidates))  # first minimum == simplest config under GRID order
    p = oofs[selected]
    # per-inner-fold fitted preprocessors and label scales (variable width for F1)
    fold_pre, fold_temp, fold_scale, inner_id = [], [], [], np.empty(len(d), int)
    for j, (tr, te) in enumerate(splits):
        fold_pre.append(build_preprocessor().fit(take_X(X, tr)))
        if USE_COMPOSITION:  # F0 reuses the first scaled column, as the frozen engine did
            fold_temp.append(build_temp_preprocessor().fit(take_X(X[['Temperature_K']], tr)))
        fold_scale.append(max(float(np.std(y[tr])), 1e-8))
        inner_id[te] = j
    chosen = {'affine': 1., 'ridge': 10., 'temperature_ridge': 10.}
    cv = {kind: {str(a): [] for a in [.1, 1., 10., 100.]} for kind in chosen}
    for src, grp in d.groupby('canonical_source_id', sort=True):
        rows = grp.sort_values('row_id').index.to_numpy()
        j = int(inner_id[rows[0]])
        T, U = design(grp)
        if len(U) < 5:
            continue
        pos = {g: i for i, g in enumerate(rows)}
        Trows = np.array([pos[g] for g in T])
        Zrows = fold_pre[j].transform(take_X(X, rows))
        # frozen v4.1/v5 used the first scaled column (Temperature_K) for the
        # temperature ridge; F0 keeps that path, F1 uses a dedicated transform
        # frozen v4.1/v5 used the first scaled column (Temperature_K) for the
        # temperature ridge; F0 keeps that path, F1 uses a dedicated transform
        tempz = (Zrows[:, :1] if not USE_COMPOSITION
                 else fold_temp[j].transform(X[['Temperature_K']].iloc[rows]))
        yrows = y[rows]
        prows = p[rows]
        scale_row = fold_scale[j]
        for rep in range(CONFIG['pseudo_repeats']):
            # permute the design-ordered pool exactly as the frozen v4.1 engine did,
            # then map to local positions, so F0 hyper-parameter selection is unchanged
            A = np.array([pos[g] for g in
                          np.random.default_rng(stable_seed('pseudo', src, rep)).permutation(U)[:5]])
            for a in [.1, 1., 10., 100.]:
                q = affine(prows[A], yrows[A], prows[Trows], a, scale_row)
                cv['affine'][str(a)].append((src, float(abs(yrows[Trows] - q).mean())))
                q = target_ridge(Zrows[A], yrows[A], Zrows[Trows], a)
                cv['ridge'][str(a)].append((src, float(abs(yrows[Trows] - q).mean())))
                q = target_ridge(tempz[A], yrows[A], tempz[Trows], a)
                cv['temperature_ridge'][str(a)].append((src, float(abs(yrows[Trows] - q).mean())))
    tune = {}
    for kind, values in cv.items():
        scores = {a: float(pd.DataFrame(v, columns=['s', 'e']).groupby('s').e.mean().mean())
                  for a, v in values.items() if v}
        if scores:
            chosen[kind] = float(min(scores, key=scores.get))
        tune[kind] = scores
    pri = prior_fit(y - p, s)
    return GRID[selected], p, inner_id, pri, chosen, dict(
        grid=GRID, grid_scores=candidates, pseudo_scores=tune)


def part_identity(input_file, tr, te, d):
    rows = ''.join(f'{rid},{src},{grp}' for rid, src, grp in zip(
        d.row_id.to_numpy(str), d.canonical_source_id.to_numpy(str),
        d.leakage_group_id.to_numpy(str)))
    split = json.dumps([sorted(d.row_id.to_numpy(str)[tr]),
                        sorted(d.row_id.to_numpy(str)[te])], separators=(',', ':'))
    return dict(engine_version=CONFIG['engine_version'], task=TASK, feature_set=FEATURE_SET,
                input_hash=sha(input_file), code_hash=sha(__file__),
                contracts_hash=sha(Path(__file__).with_name('v51_contracts.py')),
                config_hash=sha(Path(_TASK_CFG_PATH)),
                rowmap_hash=hashlib.sha256(rows.encode()).hexdigest(),
                split_hash=hashlib.sha256(split.encode()).hexdigest(),
                env=dict(python=platform.python_version(), numpy=np.__version__,
                         pandas=pd.__version__, sklearn=sklearn.__version__),
                representation=dict(features=FEATURES, use_composition=USE_COMPOSITION,
                                    model_columns=MODEL_COLUMNS,
                                    vocabulary='training-partition-fitted' if USE_COMPOSITION else None))


def task(input_file, out_dir, outer_seed, fold_id, tr, te, target):
    out = Path(out_dir)
    name = f'{outer_seed}_{fold_id}_{target}'
    part = out / 'parts' / name
    part.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(input_file).reset_index(drop=True)
    identity = part_identity(input_file, tr, te, d)
    done = part / 'done.json'
    if done.exists():
        old = json.loads(done.read_text())
        stable = {k: old[k] for k in ('engine_version', 'task', 'feature_set', 'input_hash',
                                      'code_hash', 'contracts_hash', 'config_hash', 'rowmap_hash',
                                      'split_hash', 'env', 'representation')}
        assert stable == identity, f'cache identity mismatch for {name}'
        return name
    training = d.iloc[tr].reset_index(drop=True)
    oid = f'{outer_seed}:{fold_id}'
    cfg, p_oof, inner_id, pri, hp, selection = inner_fit(training, target, oid)
    # F0 bitwise anchor: feed the pandas view directly, exactly as the frozen
    # engine did (an arange fancy-index copy here changes numpy summation
    # blocking and shifts farthest-first near-ties); F1 passes the DataFrame.
    Xtrain = model_matrix(training)
    ytrain = training[target].to_numpy(float)
    m = model(cfg).fit(Xtrain, ytrain)
    y = d[target].to_numpy(float)
    Xall = model_matrix(d)
    pred = m.predict(Xall)
    z = m[:-1].transform(Xall)
    temp_z = (z[:, :1] if not USE_COMPOSITION
              else build_temp_preprocessor().fit(training[['Temperature_K']]).transform(
                  d[['Temperature_K']]))
    scale = max(float(np.std(training[target].to_numpy(float))), 1e-8)
    oof = training[['row_id', 'canonical_source_id', 'leakage_group_id']].copy()
    oof['inner_fold'] = inner_id
    oof['y_true'] = training[target].to_numpy()
    oof['y_pred'] = p_oof
    oof.to_csv(part / 'training_oof.csv', index=False)
    train_mean = float(training[target].mean())
    train_median = float(training[target].median())
    alltest = d.iloc[te][['row_id', 'canonical_source_id', 'leakage_group_id']].copy()
    alltest['outer_seed'] = outer_seed
    alltest['fold'] = fold_id
    alltest['target'] = target
    alltest['y_true'] = y[te]
    alltest['base'] = pred[te]
    alltest['debiased'] = pred[te] + pri.mu
    alltest['train_mean'] = train_mean
    alltest['train_median'] = train_median
    alltest.to_csv(part / 'zero_all.csv', index=False)
    r = training[target].to_numpy() - p_oof
    donors = pd.DataFrame({'s': training.canonical_source_id, 'r': r}).groupby('s').r.mean().sort_index()
    lo, hi = TARGET_BOUNDS[target]
    rows, ledgers = [], []
    header = ['task', 'feature_set', 'dataset_version', 'outer_seed', 'fold', 'target', 'source',
              'leakage_group', 'cohort', 'allocation', 'selector', 'repeat', 'k', 'row_id',
              'y_true'] + METHODS
    with gzip.open(part / 'predictions.csv.gz', 'wt', encoding='utf-8', newline='\n') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for src, grp in d.iloc[te].groupby('canonical_source_id', sort=True):
            T, U = design(grp)
            if len(U) < 3:
                continue
            assert not set(T) & set(U)
            cohorts = []
            if len(U) >= CONFIG['main_min_pool']:
                cohorts.append(('main_fixed_cohort', BUDGETS))
            cohorts.append(('low_budget_fixed_cohort', [1, 2, 3]))
            lg = str(grp.leakage_group_id.iloc[0])
            ids = d.row_id.to_numpy(str)
            allids = grp.index.to_numpy()
            shuffle = pred.copy()
            shuffle[allids] = pred[np.random.default_rng(
                stable_seed('shuffle_p', oid, src, target)).permutation(allids)]
            donor = float(donors.iloc[stable_seed('donor', oid, src, target) % len(donors)])
            main_cohort = any(c == 'main_fixed_cohort' for c, _ in cohorts)
            if main_cohort:  # k=0 zero-label row, identical values across methods/panels
                q0 = {'base': pred[T], 'debiased': pred[T] + pri.mu,
                      'train_mean': np.full(len(T), train_mean),
                      'train_median': np.full(len(T), train_median)}
                for j, i in enumerate(T):
                    matrix0 = [float(q0[me][j]) if me in q0 else '' for me in METHODS]
                    writer.writerow([TASK, FEATURE_SET, CONFIG['engine_version'], outer_seed,
                                     fold_id, target, src, lg, 'main_fixed_cohort', 0, 'none', 0,
                                     0, ids[i], y[i]] + matrix0)
                for me in LABEL_FREE:
                    errors = abs(y[T] - q0[me])
                    bounded = np.clip(q0[me], lo, hi) if hi is not None else np.maximum(q0[me], lo)
                    rows.append(dict(outer_seed=outer_seed, fold=fold_id, target=target,
                                     source=src, leakage_group=lg, cohort='main_fixed_cohort',
                                     allocation=0, selector='none', repeat=0, k=0, method=me,
                                     n_test=len(T), mae=float(errors.mean()),
                                     mae_bounded=float(abs(y[T] - bounded).mean()),
                                     sse=float(((y[T] - q0[me]) ** 2).sum()),
                                     sum_y=float(y[T].sum()), sum_y2=float((y[T] ** 2).sum())))
            for selector in ['random', 'farthest']:
                nreps = CONFIG['calibration_repeats'] if selector == 'random' else 1
                for rep in range(nreps):
                    order = (np.random.default_rng(stable_seed('cal', src, rep)).permutation(
                        sorted(U, key=lambda i: ids[i])) if selector == 'random' else farthest(U, z, ids))
                    assert set(order) == set(U)
                    ledgers.append(dict(outer_seed=outer_seed, fold=fold_id, target=target,
                                        source=src, selector=selector, repeat=rep, allocation=0,
                                        test_ids=json.dumps(ids[T].tolist()),
                                        ordered_pool_ids=json.dumps(ids[order].tolist())))
                    for cohort, budgets in cohorts:
                        for k in budgets:
                            A = order[:k]
                            pc, yc, pt = pred[A], y[A], pred[T]
                            shuffled_y = np.random.default_rng(
                                stable_seed('shuffle_y', src, rep, k)).permutation(yc)
                            q = {'base': pt, 'debiased': pt + pri.mu,
                                 'train_mean': np.full(len(T), train_mean),
                                 'train_median': np.full(len(T), train_median),
                                 'target_mean': np.full(len(T), yc.mean()),
                                 'target_median': np.full(len(T), np.median(yc)),
                                 'offset_mean': pt + (yc - pc).mean(),
                                 'offset_median': pt + np.median(yc - pc),
                                 'affine': affine(pc, yc, pt, hp['affine'], scale),
                                 'eb': eb(pri, pc, yc, pt),
                                 'target_ridge': target_ridge(z[A], yc, z[T], hp['ridge']),
                                 'temperature_ridge': target_ridge(temp_z[A], yc, temp_z[T],
                                                                   hp['temperature_ridge']),
                                 'affine_shuffled_y': affine(pc, shuffled_y, pt, hp['affine'], scale),
                                 'donor_offset': pt + donor,
                                 'eb_shuffled_p': eb(pri, shuffle[A], yc, shuffle[T])}
                            assert np.allclose((yc - pc).mean(), (shuffled_y - pc).mean())
                            if k == 1:
                                assert np.allclose(q['affine'], q['offset_mean'])
                            matrix = np.column_stack([q[me] for me in METHODS])
                            assert np.isfinite(matrix).all()
                            for j, i in enumerate(T):
                                writer.writerow([TASK, FEATURE_SET, CONFIG['engine_version'],
                                                 outer_seed, fold_id, target, src, lg, cohort, 0,
                                                 selector, rep, k, ids[i], y[i]] + matrix[j].tolist())
                            for me in METHODS:
                                errors = abs(y[T] - q[me])
                                bounded = (np.clip(q[me], lo, hi) if hi is not None
                                           else np.maximum(q[me], lo))
                                rows.append(dict(outer_seed=outer_seed, fold=fold_id, target=target,
                                                 source=src, leakage_group=lg, cohort=cohort,
                                                 allocation=0, selector=selector, repeat=rep, k=k,
                                                 method=me, n_test=len(T),
                                                 mae=float(errors.mean()),
                                                 mae_bounded=float(abs(y[T] - bounded).mean()),
                                                 sse=float(((y[T] - q[me]) ** 2).sum()),
                                                 sum_y=float(y[T].sum()),
                                                 sum_y2=float((y[T] ** 2).sum())))
    pd.DataFrame(rows).to_csv(part / 'cells.csv', index=False)
    pd.DataFrame(ledgers).to_csv(part / 'calibration_ledger.csv.gz', index=False, compression='gzip')
    np.savez_compressed(part / 'adapter_artifacts.npz',
                        row_id=d.row_id.to_numpy(str), source=d.canonical_source_id.to_numpy(str),
                        y=y, base_pred=pred, z=z.astype(np.float64),
                        temp_z=temp_z.astype(np.float64), test_idx=np.asarray(te),
                        prior=np.array([pri.mu, pri.tau2, pri.sigma2]),
                        hp_json=json.dumps(hp), scale=scale, model_cfg=json.dumps(cfg))
    write_json(done, identity | dict(outer_seed=outer_seed, fold=fold_id, target=target,
                                     model=cfg, hyperparameters=hp, prior=asdict(pri),
                                     prediction_scale=scale, selection=selection,
                                     n_train=len(tr), n_test_all=len(te), n_cells=len(rows),
                                     n_calibration_orders=len(ledgers)))
    print('DONE', TASK, FEATURE_SET, name, 'model', cfg, flush=True)
    return name


def summarize(out):
    out = Path(out)
    cells = pd.concat([pd.read_csv(p) for p in sorted((out / 'parts').glob('*/cells.csv'))],
                      ignore_index=True)
    parts = sorted((out / 'parts').glob('*/done.json'))
    expected = len(CONFIG['outer_seeds']) * len(CONFIG['folds_to_run']) * len(TARGETS)
    assert len(parts) == expected, f'{len(parts)} done parts != expected {expected}'
    keys = ['target', 'source', 'leakage_group', 'cohort', 'allocation', 'selector', 'k', 'method']
    per = cells.groupby(keys, as_index=False)[
        ['mae', 'mae_bounded', 'n_test', 'sse', 'sum_y', 'sum_y2']].mean()
    per.to_csv(out / 'source_metrics.csv', index=False)
    draws = CONFIG['bootstrap_draws']
    summary = []
    for keys_, a in per.groupby(['target', 'cohort', 'selector', 'k', 'method']):
        target, cohort, selector, k, method = keys_
        cb = cluster_bootstrap(a.rename(columns={'leakage_group': 'leakage_group'}),
                               'mae', draws=draws, seed=boot_seed(*keys_), source_col='source',
                               group_col='leakage_group')
        n = a.n_test.sum()
        sst = a.sum_y2.sum() - a.sum_y.sum() ** 2 / n
        summary.append(dict(target=target, cohort=cohort, selector=selector, k=int(k),
                            method=method, source_equal_mae=float(a.mae.mean()),
                            ci_low=(cb['ci95'] or [None, None])[0],
                            ci_high=(cb['ci95'] or [None, None])[1],
                            bootstrap_status=cb['status'],
                            source_equal_mae_bounded=float(a.mae_bounded.mean()),
                            row_weighted_mae=float(np.average(a.mae, weights=a.n_test)),
                            prediction_averaged_r2=float(1 - a.sse.sum() / sst) if sst > 0 else None,
                            n_sources=len(a), n_groups=int(a.leakage_group.nunique()),
                            n_test_records=int(n)))
    sm = pd.DataFrame(summary)
    sm.to_csv(out / 'summary.csv', index=False)
    comparisons = []
    for keys_, a in per.groupby(['target', 'cohort', 'selector', 'k']):
        target, cohort, selector, k = keys_
        wide = a.pivot(index=['source', 'leakage_group'], columns='method', values='mae').reset_index()
        present = [me for me in METHODS if me in wide.columns]
        assert not wide[present].isna().any().any()
        pairs = [('eb', 'base'), ('eb', 'debiased'), ('eb', 'offset_mean'), ('affine', 'offset_mean'),
                 ('eb', 'target_median'), ('eb', 'target_ridge'), ('eb', 'temperature_ridge'),
                 ('affine', 'target_ridge'), ('affine', 'base'), ('eb', 'eb_shuffled_p'),
                 ('affine', 'affine_shuffled_y'), ('eb', 'donor_offset'), ('train_mean', 'base'),
                 ('target_mean', 'base'), ('target_mean', 'train_mean')]
        for ma, mb in pairs:
            if ma not in wide.columns or mb not in wide.columns:
                continue
            zc = wide[['source', 'leakage_group', ma, mb]].copy()
            zc['delta'] = zc[ma] - zc[mb]
            cb = cluster_bootstrap(zc, 'delta', draws=draws, seed=boot_seed(*keys_, ma, mb),
                                   source_col='source', group_col='leakage_group')
            comparisons.append(dict(target=target, cohort=cohort, selector=selector, k=int(k),
                                    method=ma, reference=mb, delta_mae=float(zc.delta.mean()),
                                    ci_low=(cb['ci95'] or [None, None])[0],
                                    ci_high=(cb['ci95'] or [None, None])[1],
                                    relative_reduction=float(1 - zc[ma].mean() / zc[mb].mean()),
                                    n_improved=int((zc.delta < 0).sum()), n_sources=len(zc),
                                    n_groups=int(zc.leakage_group.nunique())))
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    zero = pd.concat([pd.read_csv(p) for p in sorted((out / 'parts').glob('*/zero_all.csv'))],
                     ignore_index=True)
    zero.to_csv(out / 'zero_all_predictions.csv.gz', index=False, compression='gzip')
    zs = []
    for target, a in zero.groupby('target'):
        for method in ['base', 'debiased', 'train_mean', 'train_median']:
            a2 = a.copy()
            a2['ae'] = abs(a2.y_true - a2[method])
            # one row per SOURCE (a source may span several leakage groups);
            # for the shipped datasets source<->group is 1:1, so this is identity there
            bysrc = (a2.groupby('canonical_source_id')
                     .agg(ae=('ae', 'mean'),
                          leakage_group=('leakage_group_id', 'first'))
                     .reset_index().rename(columns={'canonical_source_id': 'source'}))
            cb = cluster_bootstrap(bysrc, 'ae', draws=draws, seed=boot_seed('zero_all', target, method),
                                   source_col='source', group_col='leakage_group')
            zs.append(dict(panel='zero_all_sources', target=target, method=method,
                           n_sources=len(bysrc), n_rows=int(a2.row_id.nunique()),
                           source_equal_mae=float(bysrc.ae.mean()),
                           ci_low=(cb['ci95'] or [None, None])[0],
                           ci_high=(cb['ci95'] or [None, None])[1],
                           bootstrap_status=cb['status'],
                           row_weighted_mae=float(a2.ae.mean()),
                           repeated_prediction_pooled_r2=float(r2_score(a2.y_true, a2[method]))))
    write_json(out / 'zero_all_summary.json', zs)
    lc = sm.query("cohort=='main_fixed_cohort' and selector=='random'").copy()
    lc = lc.sort_values(['target', 'method', 'k'])
    lc.to_csv(out / 'learning_curve_main.csv', index=False)
    write_json(out / 'RESULTS_STATUS.json', dict(
        status='completed', engine_version=CONFIG['engine_version'], task=TASK,
        feature_set=FEATURE_SET, outer_tasks=expected, n_cell_metrics=len(cells),
        methods=METHODS, smoke=CONFIG['smoke'],
        main_cohort_sources=int(per.query("cohort=='main_fixed_cohort'").source.nunique()),
        low_budget_sources=int(per.query("cohort=='low_budget_fixed_cohort'").source.nunique()),
        bootstrap=dict(draws=draws, seed_base=CONFIG['bootstrap_seed_base'],
                       unit='leakage_group', scope='conditional on fitted models'),
        panels=dict(zero_all='all qualified sources, held-out rows, label-free methods only',
                    main_fixed_cohort='pool>=10 and fixed T shared across k and methods',
                    low_budget_fixed_cohort='sources supporting only k<=3, reported separately'),
        not_run=['external_validation', 'conformal_intervals', 'REML', 'new_reactions',
                 'prospective_physical_experiments']))


def engine_tests():
    groups = np.repeat(np.arange(11).astype(str), 4)
    for seed in SEEDS:
        hits = np.zeros(len(groups), int)
        for tr, te in folds(groups, seed):
            hits[te] += 1
            assert not set(groups[tr]) & set(groups[te])
        assert np.all(hits == 1)
    d = pd.DataFrame({'row_id': [f'r{i:03}' for i in range(20)], 'canonical_source_id': ['s'] * 20,
                      'y': np.arange(20)})
    T, U = design(d)
    d['y'] = -d.y
    T2, U2 = design(d)
    assert np.array_equal(T, T2) and np.array_equal(U, U2) and not set(T) & set(U)
    # allocation 0 of the reference partition matches design() membership on real-style ids
    sub = pd.DataFrame({'row_id': [f'OCM-CAT-{i:04}' for i in range(30)],
                        'canonical_source_id': ['s'] * 30})
    t_ref, u_ref = fixed_partition(sub.row_id, 's', 'ocm', 0)
    t_eng, u_eng = design(sub)
    assert set(map(str, t_ref)) == set(sub.row_id.to_numpy(str)[t_eng])
    assert set(map(str, u_ref)) == set(sub.row_id.to_numpy(str)[u_eng])
    pc = np.array([1., 2., 4.]); yc = np.array([4., 4., 9.]); pt = np.array([3., 5.])
    assert np.allclose((yc - pc).mean(), (yc[::-1] - pc).mean())
    assert np.allclose(affine(pc[:1], yc[:1], pt, 1., 2.), pt + yc[0] - pc[0])
    prior = Prior(1., 2., 3.)
    w = 3 * 2 / (3 + 3 * 2)
    assert np.allclose(eb(prior, pc, yc, pt), pt + 1 + w * ((yc - pc).mean() - 1))
    order = np.arange(15)
    assert all(set(order[:a]) <= set(order[:b]) for a, b in zip(BUDGETS, BUDGETS[1:]))
    print('PASS: coverage, label-blind fixed T, allocation-0 equivalence, nested budgets, '
          'offset permutation, affine k1, EB formula', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    if args.test:
        engine_tests()
        return
    assert _TASK_CFG_PATH, 'V51_TASK_CONFIG must point at a frozen per-task config file'
    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        summarize(args.out)
        return
    engine_tests()
    d = pd.read_csv(args.input)
    assert d.row_id.is_unique
    assert not d[['row_id', 'canonical_source_id', 'leakage_group_id']].isna().any().any()
    assert np.isfinite(d[TARGETS]).all().all()
    assert not np.isinf(d[FEATURES].to_numpy(float)).any()  # NaN allowed; imputer handles
    if USE_COMPOSITION:
        # composition slots may legitimately be empty dataset-wide (never listed);
        # the encoder treats empty slots as unreported, so only operating features
        # are required to carry information
        assert not d[FEATURES].isna().all().any()
    ledger, jobs = [], []
    for seed in CONFIG['outer_seeds']:
        hits = np.zeros(len(d), int)
        for j, (tr, te) in enumerate(folds(d.leakage_group_id, seed)):
            hits[te] += 1
            if j not in CONFIG['folds_to_run']:
                continue
            for i in range(len(d)):
                ledger.append(dict(outer_seed=seed, fold=j, row_id=d.row_id.iloc[i],
                                   source=d.canonical_source_id.iloc[i],
                                   leakage_group=d.leakage_group_id.iloc[i],
                                   role='test' if i in set(te) else 'train'))
            for target in TARGETS:
                jobs.append((str(args.input), str(args.out), seed, j, tr, te, target))
        assert np.all(hits == 1)
    pd.DataFrame(ledger).to_csv(args.out / 'split_ledger.csv.gz', index=False, compression='gzip')
    write_json(args.out / 'config.json', CONFIG | {'grid': GRID, 'methods': METHODS,
                                                   'task': TASK, 'feature_set': FEATURE_SET,
                                                   'model_columns': MODEL_COLUMNS})
    write_json(args.out / 'environment.json',
               dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, input_sha256=sha(args.input),
                    code_sha256=sha(__file__),
                    contracts_sha256=sha(Path(__file__).with_name('v51_contracts.py')),
                    task_config_sha256=sha(_TASK_CFG_PATH)))
    write_json(args.out / 'split_checks.json',
               dict(each_row_tested_once_per_seed=True,
                    n_outer_partitions=len(CONFIG['outer_seeds']) * 5,
                    seeds_run=CONFIG['outer_seeds'], folds_run=CONFIG['folds_to_run'],
                    n_sources=int(d.canonical_source_id.nunique()),
                    n_groups=int(d.leakage_group_id.nunique()), n_rows=len(d), smoke=CONFIG['smoke']))
    Parallel(n_jobs=args.jobs, verbose=5)(delayed(task)(*job) for job in jobs)
    summarize(args.out)


if __name__ == '__main__':
    main()

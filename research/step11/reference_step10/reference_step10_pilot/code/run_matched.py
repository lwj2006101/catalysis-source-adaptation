"""Actual source-disjoint development of predictor-matched local acquisition.

Archived outer backbones and transforms are reused; all new kernel selection,
working covariances, support orders and predictions are actually computed.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
import argparse
import datetime as dt
import gzip
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matched_kernel as mk

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reference_step9'
sys.path.insert(0, str(REFERENCE / 'code'))
import run_history as old
b = old.b
BUDGETS = (1, 2, 3, 5, 10)
SEEDS = (42, 101, 202, 303, 404)
BASE_POLICIES = ('random', 'operating_vopt', 'composition_vopt', 'feature_cover',
                 'pool_vopt', 'kernel_vopt', 'robust_view_vopt')
POLICIES = BASE_POLICIES + mk.NEW_POLICIES
METHODS = ('matched_kernel', 'locked_local_rbf', 'frozen_blend',
           'local_ridge', 'target_mean', 'zero_label')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p = Path(p)
    if not p.resolve().is_relative_to(ROOT):
        raise ValueError('Output outside project')
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def save_csv(frame, p):
    blob = frame.to_csv(index=False).encode()
    if str(p).endswith('.gz'):
        blob = gzip.compress(blob, mtime=0)
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp'); temp.write_bytes(blob); temp.replace(p)


def tune(training, zs, inner, directory):
    """Nine/90 candidate estimators validated using historical random labels.

    Transforms exclude each pseudo-source group through the archived inner
    splits. The outer evaluation sources never enter these validation losses.
    """
    source_scores = {task: [] for task in ('ocm_f0', 'ocm_f1')}
    groups, candidates_log = [], []
    prediction_count = 0
    for source, sub in training.groupby('canonical_source_id', sort=True):
        if len(sub) < 15:
            continue
        query, pool = b.random_design(sub)
        if len(pool) < 10:
            continue
        j = int(inner[sub.index[0]])
        if not np.all(inner[sub.index] == j):
            raise ValueError('Pseudo-source crosses inner folds')
        indices = sub.index.to_numpy(int)
        lookup = {int(v): k for k, v in enumerate(indices)}
        q = np.array([lookup[int(v)] for v in query])
        pp = np.array([lookup[int(v)] for v in sorted(pool, key=lambda x: str(training.row_id.iloc[x]))])
        y = training.C2_yield_percent.to_numpy(float)[indices]
        group = str(sub.leakage_group_id.iloc[0]); groups.append(group)
        for task in ('ocm_f0', 'ocm_f1'):
            z = zs[j][indices] if task == 'ocm_f1' else zs[j][indices, :6]
            grid = mk.candidates(task); d = mk.distance_views(z, z)
            kernels = np.stack([mk.from_distances(d, item) for item in grid])
            alphas = np.array([item['alpha'] for item in grid])
            losses = []
            for repeat in range(10):
                support = np.random.default_rng(b.stable_seed(
                    'step10_random_history', str(source), repeat)).permutation(pp)[:5]
                kss = kernels[:, support[:, None], support[None, :]]
                kqs = kernels[:, q[:, None], support[None, :]]
                pred = mk.weights(kss, kqs, alphas) @ y[support]
                prediction_count += pred.size
                losses.append(np.abs(pred - y[q]).mean(1))
            score = np.mean(losses, axis=0); source_scores[task].append(score)
            for i, (item, value) in enumerate(zip(grid, score)):
                candidates_log.append(dict(task=task, source=str(source), leakage_group=group,
                    candidate=i, theta=json.dumps(item, sort_keys=True), mae=float(value),
                    support_draws=10, budget=5, inner_fold=j))
    if not groups:
        raise ValueError('No eligible historical source')
    parameters = {task: mk.choose_parameters(source_scores[task], groups, mk.candidates(task))
                  for task in ('ocm_f0', 'ocm_f1')}
    write_json(directory / 'parameters.json', dict(tasks=parameters,
        historical_sources=len(groups), historical_groups=len(set(groups)),
        actual_kernel_validation_prediction_values=prediction_count,
        primary_parameter_budget=5,
        tuning='group-equal historical random-support MAE; fixed deployed kernel across policies and budgets',
        new_backbone_fits=0))
    save_csv(pd.DataFrame(candidates_log), directory / 'historical_kernel_candidates.csv.gz')
    return parameters


def evaluate(data, geometry, infos, old_parameters, parameters, baseline_ledger,
             directory, seed, fold):
    y = data.C2_yield_percent.to_numpy(float)
    row_ids = data.row_id.to_numpy(str)
    cells, ledger, preds, cell_ids, query_ids = [], [], [], [], []
    for (source, scenario), frame in baseline_ledger.groupby(['source', 'scenario'], sort=True):
        first = frame.iloc[0]
        query = np.array(json.loads(first.query_indices), int)
        pool = np.array(sorted(json.loads(first.pool_indices), key=lambda i: str(row_ids[i])), int)
        if set(pool) & set(query):
            raise ValueError('Support/query overlap')
        base_orders = [(str(row.policy), int(row.repeat), np.array(json.loads(row.support_indices), int),
                        {'predecessor_design_id':int(row.design_id)})
                       for row in frame.itertuples() if row.policy in BASE_POLICIES]
        position = {int(v): k for k, v in enumerate(pool)}
        for task in ('ocm_f0', 'ocm_f1'):
            z, base = geometry['z_' + task[-2:]], geometry['base_' + task[-2:]]
            parameter = parameters[task]; theta = parameter['theta']
            kp, kpq, cp, cpq = mk.matrices(z[pool], z[query], parameter)
            orders = list(base_orders)
            for policy in mk.NEW_POLICIES:
                take, trace = mk.order(kp, kpq, cp, cpq, theta['alpha'],
                    parameter['working_noise'], policy)
                orders.append((policy, 0, pool[take], {'trace':trace, 'theta':theta,
                               'covariance_interpretation':parameter['interpretation']}))
            for policy, repeat, order, diagnostic in orders:
                if len(set(order)) != 10 or not set(order) <= set(pool):
                    raise ValueError('Invalid support sequence')
                design_id = len(ledger)
                ledger.append(dict(design_id=design_id, task=task, source=str(source), scenario=scenario,
                    policy=policy, repeat=repeat, query_indices=json.dumps(query.tolist()),
                    pool_indices=json.dumps(pool.tolist()), support_indices=json.dumps(order.tolist()),
                    diagnostic=json.dumps(diagnostic, sort_keys=True), metadata=first.metadata))
                for k in BUDGETS:
                    support = order[:k]
                    s = np.array([position[int(v)] for v in support])
                    prediction = mk.predict(kp[np.ix_(s, s)], kpq[s].T, y[support], theta['alpha'])
                    rbf_hp = old_parameters['controls'][task]['composition_vopt'][str(k)]['rbf']
                    locked_rbf = old.comp.local_rbf(z[support], y[support], z[query], **rbf_hp)
                    frozen = old._frozen_pred(z, y[support], base[support], query, support,
                                              base, infos[task], k)
                    pp = dict(matched_kernel=prediction, locked_local_rbf=locked_rbf,
                              frozen_blend=frozen['adaptive_blend'], local_ridge=frozen['local_ridge'],
                              target_mean=np.full(len(query), y[support].mean()), zero_label=base[query])
                    values = np.column_stack([pp[m] for m in METHODS])
                    if not np.isfinite(values).all():
                        raise ArithmeticError('Nonfinite target predictions')
                    row = dict(cell_id=len(cells), design_id=design_id, task=task,
                        seed=seed, fold=fold, source=str(source),
                        leakage_group=str(data.leakage_group_id.iloc[query[0]]),
                        scenario=scenario, policy=policy, repeat=repeat, k=k, n_test=len(query))
                    for method, prediction in pp.items():
                        row['mae__' + method] = float(np.abs(prediction - y[query]).mean())
                        row['mse__' + method] = float(np.square(prediction - y[query]).mean())
                    cells.append(row); preds.append(values)
                    cell_ids.extend([row['cell_id']] * len(query)); query_ids.extend(query.tolist())
    if not cells:
        raise ValueError('No eligible target frame')
    save_csv(pd.DataFrame(cells), directory / 'cells.csv.gz')
    save_csv(pd.DataFrame(ledger), directory / 'support_ledger.csv.gz')
    save_csv(data, directory / 'evaluated_input.csv.gz')
    np.savez_compressed(directory / 'row_predictions.npz', predictions=np.vstack(preds),
                        cell_id=np.asarray(cell_ids, np.int32), query_index=np.asarray(query_ids, np.int32),
                        methods=np.asarray(METHODS, dtype='U32'))
    np.savez_compressed(directory / 'geometry.npz', **geometry)
    return len(cells), len(cell_ids)


def run_part(seed, fold, identity, run_root):
    directory = run_root / 'parts' / f'ocm_{seed}_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / 'completed.json'
    if complete.exists():
        if json.loads(complete.read_text())['identity'] != identity:
            raise ValueError('Preserve old run and choose a fresh version')
        return
    old_part = REFERENCE / 'runs/history_v1/parts' / f'ocm_{seed}_{fold}'
    inherited = json.loads((old_part / 'completed.json').read_text())
    infos = inherited['inherited_infos']
    data, cfg = b.load_task('ocm_f1')
    allocation = pd.read_csv(b.SOURCE / 'benchmark_splits/ocm_f1/split_ledger.csv.gz')
    allocation = allocation[(allocation.outer_seed == seed) & (allocation.fold == fold)]
    train_ids = set(allocation.loc[allocation.role.eq('train'), 'row_id'].astype(str))
    training = data[data.row_id.astype(str).isin(train_ids)].reset_index(drop=True)
    test = pd.read_csv(old_part / 'evaluated_input.csv.gz')
    if not set(training.leakage_group_id).isdisjoint(test.leakage_group_id):
        raise ValueError('Historical/evaluation source overlap')
    original = old.previous.REFERENCE / 'results/parts' / f'ocm_f1_{seed}_{fold}_C2_yield_percent'
    zs, inner, _, audits = old.previous.historical_geometry(training, cfg, original, seed, fold)
    parameters = tune(training, zs, inner, directory)
    old_parameters = json.loads((old_part / 'parameters.json').read_text())
    write_json(directory / 'inherited_parameters.json', old_parameters)
    write_json(directory / 'training_source_audit.json', audits)
    legacy_geometry = np.load(old_part / 'geometry.npz')
    geometry = {key:legacy_geometry[key] for key in legacy_geometry.files}
    baseline = pd.read_csv(old_part / 'support_ledger.csv.gz')
    n_cells, n_rows = evaluate(test, geometry, infos, old_parameters, parameters, baseline,
                               directory, seed, fold)
    write_json(complete, dict(identity=identity, seed=seed, fold=fold,
        n_cells=n_cells, n_prediction_rows=n_rows, n_scalar_predictions=n_rows * len(METHODS),
        new_outer_backbone_fits=0, kernel_and_support_fits_actually_executed=True,
        history_training_groups=sorted(set(training.leakage_group_id)),
        evaluation_groups=sorted(set(test.leakage_group_id)), inherited_infos=infos,
        predecessor_part_completion_sha256=sha(old_part / 'completed.json'), completed_utc=utc()))
    print('DONE', seed, fold, n_cells, n_rows, flush=True)


def experiment_lock(run_root, seeds, folds):
    config = dict(version='step10-predictor-matched-1', seeds=seeds, folds=folds,
        policies=POLICIES, methods=METHODS, budgets=BUDGETS,
        plan_sha256=sha(ROOT / 'protocol/analysis_plan.json'),
        code_hashes={p.name:sha(p) for p in (ROOT / 'code').glob('*.py')},
        predecessor_manifest_sha256=sha(REFERENCE / 'release_manifest.json'))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    p = run_root / 'experiment_lock.json'
    if p.exists() and json.loads(p.read_text())['identity'] != identity:
        raise ValueError('Code/design changed; use a new run name')
    if not p.exists():
        write_json(p, dict(identity=identity, created_utc=utc(), config=config))
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='history_v1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args()
    root = ROOT / 'runs' / args.run_name
    if not root.resolve().is_relative_to(ROOT / 'runs'):
        raise ValueError('Run outside project runs')
    root.mkdir(parents=True, exist_ok=True)
    identity = experiment_lock(root, args.seeds, args.folds)
    jobs = list(itertools.product(args.seeds, args.folds))
    Parallel(n_jobs=args.workers)(delayed(run_part)(s, f, identity, root) for s, f in jobs)
    write_json(root / 'run_completion.json', dict(status='completed', identity=identity,
        parts=len(jobs), full_historical_grid_completed=set(jobs) == set(itertools.product(SEEDS, range(5))),
        new_outer_backbone_fits=0, data_status='previously observed historical development',
        completed_utc=utc()))


if __name__ == '__main__':
    main()

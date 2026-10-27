"""Retrospective OCM view-selection experiment, reusing verified Step 7 bases.

All new fitting and outputs use the explicitly authorized project runtime. Historical query outcomes have
already informed this research program: this is development, not confirmation.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from server_guard import ROOT, require_server, sha, utc, write_json
REFERENCE = ROOT / 'reference_step7'
sys.path.insert(0, str(REFERENCE / 'code'))
import run_cross_task as previous
from selection_controls import BUDGETS, KERNEL_GAMMAS, select_control
from view_selection import context, fit_gate, ViewGate
b = previous.b
comp = previous.comp
acq = previous.acq

VIEWS = ('operating_vopt', 'composition_vopt')
POLICIES = ('random', *VIEWS, 'feature_cover', 'pool_vopt', 'kernel_vopt',
            'robust_view_vopt', 'fixed_best_view', 'plain_view_gate', 'view_gate',
            'shuffled_composition_vopt')
METHODS = (*b.METHODS, 'local_ridge_retuned', 'adaptive_blend_retuned', 'local_rbf_retuned')
SEEDS = (42, 101, 202, 303, 404)


def save_csv(frame, path):
    require_server(path)
    previous.csv(frame, path)


def alpha_schedule(info):
    return {str(k): info['frozen_parameters']['hyperparameters'][str(k)]['local_alpha']
            for k in BUDGETS}


def base_views(z, query, pool, alphas):
    op, _ = acq.select_order(z[pool, :6], z[query, :6], 'linear_vopt', alphas)
    full, _ = acq.select_order(z[pool], z[query], 'linear_vopt', alphas)
    features = context(z[pool, :6], z[pool, 6:], z[query, :6], z[query, 6:], op, full)
    return {'operating': pool[op], 'composition': pool[full]}, features


def _frozen_pred(z, y_support, base_support, query, support, base, info, k):
    old = info['frozen_parameters']
    return b.predict_methods(z[support], y_support, base_support, z[query], base[query],
                             old['hyperparameters'][str(k)], old['prior'],
                             info['label_scale'])[0]


def selection_orders(z, row_ids, query, pool, alphas, gate, gamma, seed):
    """No response, fitted target predictor, or response-bearing frame accepted."""
    pool = np.array(sorted(pool, key=lambda i: str(row_ids[i])), int)
    views, features = base_views(z, query, pool, alphas)
    chosen, diagnostic = gate.choose(features)
    plain, _ = gate.choose(features, conservative=False)
    output = {VIEWS[0]: [(0, views['operating'], {})],
              VIEWS[1]: [(0, views['composition'], {})],
              'view_gate': [(0, views[chosen], diagnostic)],
              'plain_view_gate': [(0, views[plain], {})],
              'fixed_best_view': [(0, views[gate.fixed_view], {})]}
    for policy, old_policy in [('feature_cover', 'feature_cover'),
                               ('pool_vopt', 'linear_pool_vopt')]:
        take, trace = acq.select_order(z[pool], z[query], old_policy, alphas)
        output[policy] = [(0, pool[take], {'trace': trace})]
    for policy in ('kernel_vopt', 'robust_view_vopt'):
        take, trace = select_control(z[pool, :6], z[pool], z[query, :6], z[query],
                                     policy, alphas, gamma)
        output[policy] = [(0, pool[take], {'trace': trace})]
    output['random'] = [(repeat, np.random.default_rng(b.stable_seed(
                         'step8_random', seed, repeat)).permutation(pool)[:10], {})
                        for repeat in range(50)]
    shuffles = []
    joined = np.concatenate([pool, query])
    for repeat in range(10):
        rng = np.random.default_rng(b.stable_seed('step8_chemistry_shuffle', seed, repeat))
        zz = z[joined].copy()
        zz[:, 6:] = zz[rng.permutation(len(zz)), 6:]
        take, _ = acq.select_order(zz[:len(pool)], zz[len(pool):], 'linear_vopt', alphas)
        shuffles.append((repeat, pool[take], {}))
    output['shuffled_composition_vopt'] = shuffles
    assert set(output) == set(POLICIES)
    assert all(len(set(order)) == 10 and set(order) <= set(pool)
               for records in output.values() for _, order, _ in records)
    return output


def historical_episodes(training, zs, inner, bases, infos):
    y = training.C2_yield_percent.to_numpy(float)
    alphas = alpha_schedule(infos['ocm_f1'])
    episodes, meta_x, meta_delta, groups = [], [], [], []
    for source, sub in training.groupby('canonical_source_id', sort=True):
        if len(sub) < 15:
            continue
        query, pool = b.random_design(sub)
        if len(pool) < 10:
            continue
        j = int(inner[sub.index[0]])
        assert np.all(inner[sub.index] == j)
        z = zs[j]
        pool = np.array(sorted(pool, key=lambda i: str(training.row_id.iloc[i])), int)
        views, features = base_views(z, query, pool, alphas)
        losses = []
        for view in ('operating', 'composition'):
            support = views[view][:5]
            pred = _frozen_pred(z[:, :6], y[support], bases['ocm_f0'][support],
                                query, support, bases['ocm_f0'], infos['ocm_f0'], 5)
            losses.append(float(np.abs(y[query] - pred['adaptive_blend']).mean()))
        kernel_loss = []
        for gamma in KERNEL_GAMMAS:
            take, _ = select_control(z[pool, :6], z[pool], z[query, :6], z[query],
                                      'kernel_vopt', alphas, gamma)
            support = pool[take[:5]]
            pred = _frozen_pred(z[:, :6], y[support], bases['ocm_f0'][support],
                                query, support, bases['ocm_f0'], infos['ocm_f0'], 5)
            kernel_loss.append(float(np.abs(y[query] - pred['adaptive_blend']).mean()))
        group = str(sub.leakage_group_id.iloc[0])
        episodes.append(dict(source=str(source), group=group, query=query, pool=pool,
                             z=z, features=features, view_losses=losses,
                             kernel_losses=kernel_loss))
        meta_x.append(features)
        meta_delta.append(losses[1] - losses[0])
        groups.append(group)
    if not episodes:
        raise ValueError('No eligible historical pseudo-sources')
    return episodes, np.asarray(meta_x), np.asarray(meta_delta), np.asarray(groups)


def select_gamma(episodes):
    return KERNEL_GAMMAS[int(np.argmin(np.mean([e['kernel_losses'] for e in episodes], axis=0)))]


def tune(training, zs, inner, bases, infos, directory):
    episodes, xx, delta, groups = historical_episodes(training, zs, inner, bases, infos)
    gate = fit_gate(xx, delta, groups)
    gamma = select_gamma(episodes)
    rows = []
    controls = {}
    y = training.C2_yield_percent.to_numpy(float)
    alphas = alpha_schedule(infos['ocm_f1'])
    arrays, blend_records = {}, []
    for e in episodes:
        keep = groups != e['group']
        if not np.any(keep):
            raise ValueError('At least two distinct historical leakage groups are required')
        source_gate = fit_gate(xx[keep], delta[keep], groups[keep])
        source_gamma = select_gamma([ep for ep in episodes if ep['group'] != e['group']])
        orders = selection_orders(e['z'], training.row_id.to_numpy(str), e['query'],
                                   e['pool'], alphas, source_gate, source_gamma,
                                   b.stable_seed('step8_pseudo', e['source']))
        # Ten random draws for tuning; all ten chemistry-shuffle draws.
        for policy, records in orders.items():
            if policy == 'random':
                records = records[:10]
            for repeat, order, _ in records:
                for task in ('ocm_f0', 'ocm_f1'):
                    z = e['z'][:, :6] if task == 'ocm_f0' else e['z']
                    base = bases[task]
                    for k in BUDGETS:
                        support, query = order[:k], e['query']
                        combined = [b.ridge_query_and_loo(z[support],
                                    np.column_stack([y[support], y[support] - base[support]]),
                                    z[query], alpha) for alpha in b.ALPHAS]
                        locals_ = [p[:, 0] for p, _ in combined]
                        residuals = [base[query] + p[:, 1] for p, _ in combined]
                        rbfs = [comp.local_rbf(z[support], y[support], z[query], **hp)
                                for hp in previous.RBF_GRID]
                        for method, preds in [('local', locals_), ('residual', residuals), ('rbf', rbfs)]:
                            arrays.setdefault((task, policy, k, e['source'], method), []).append(
                                [float(np.abs(y[query] - p).mean()) for p in preds])
                        blend_records.append(dict(task=task, policy=policy, k=k, source=e['source'],
                            y=y[support], yq=y[query], pa=base[support], local=locals_, residual=residuals,
                            lloo=[None if loo is None else loo[:, 0] for _, loo in combined],
                            rloo=[None if loo is None else base[support] + loo[:, 1] for _, loo in combined]))
    for task in ('ocm_f0', 'ocm_f1'):
        controls[task] = {}
        for policy in POLICIES:
            controls[task][policy] = {}
            for k in BUDGETS:
                hp = {}
                for method, grid in [('local', b.ALPHAS), ('residual', b.ALPHAS), ('rbf', previous.RBF_GRID)]:
                    source_scores = []
                    for (tt, pp, kk, source, mm), values in arrays.items():
                        if (tt, pp, kk, mm) == (task, policy, k, method):
                            score = np.mean(values, axis=0)
                            source_scores.append(score)
                            rows.extend(dict(task=tt, policy=pp, k=kk, source=source,
                                             model=mm, candidate=j, mae=float(v))
                                        for j, v in enumerate(score))
                    mean = np.mean(source_scores, axis=0)
                    index = int(np.argmin(mean))
                    hp[method] = grid[index]
                    hp[method + '_index'] = index
                controls[task][policy][str(k)] = hp
    blend_arrays = {}
    for record in blend_records:
        task, policy, k = record['task'], record['policy'], record['k']
        hp = controls[task][policy][str(k)]
        li, ri = hp['local_index'], hp['residual_index']
        dscale = infos[task]['frozen_parameters']['hyperparameters'][str(k)]['blend_dscale']
        values = []
        for w0, penalty in previous.BLEND_GRID:
            w = b.blend_weight(record['y'], record['lloo'][li], record['rloo'][ri], w0, penalty, dscale)
            pred = (1 - w) * record['local'][li] + w * record['residual'][ri]
            values.append(float(np.abs(record['yq'] - pred).mean()))
        blend_arrays.setdefault((task, policy, k, record['source']), []).append(values)
    for task in ('ocm_f0', 'ocm_f1'):
        for policy in POLICIES:
            for k in BUDGETS:
                means = []
                for (tt, pp, kk, source), values in blend_arrays.items():
                    if (tt, pp, kk) == (task, policy, k):
                        score = np.mean(values, axis=0)
                        means.append(score)
                        rows.extend(dict(task=tt, policy=pp, k=kk, source=source,
                                         model='blend', candidate=j, mae=float(v))
                                    for j, v in enumerate(score))
                w0, penalty = previous.BLEND_GRID[int(np.argmin(np.mean(means, axis=0)))]
                controls[task][policy][str(k)].update(blend_w0=w0, blend_c=penalty,
                    blend_dscale=infos[task]['frozen_parameters']['hyperparameters'][str(k)]['blend_dscale'])
    write_json(directory / 'parameters.json', dict(gate=gate.as_dict(), gamma=gamma, controls=controls,
               gate_context_names=list(__import__('view_selection').CONTEXT_NAMES),
               tuning='new gate/kernel choices exclude the pseudo-source leakage group; inherited adapter settings remain frozen'))
    save_csv(pd.DataFrame(rows), directory / 'control_candidate_scores.csv.gz')
    save_csv(pd.DataFrame([dict(source=e['source'], leakage_group=e['group'],
                    operating_mae=e['view_losses'][0], composition_mae=e['view_losses'][1],
                    delta_mae=float(d), **{n:float(v) for n,v in zip(
                        __import__('view_selection').CONTEXT_NAMES, x)})
                for e, x, d in zip(episodes, xx, delta)]), directory / 'gate_training_examples.csv')
    return gate, gamma, controls


def predict(z, base, support_y, support, query, info, choice, k):
    frozen = _frozen_pred(z, support_y, base[support], query, support, base, info, k)
    local, lloo = b.ridge_query_and_loo(z[support], support_y, z[query], choice['local'])
    residual, rloo = b.ridge_query_and_loo(z[support], support_y - base[support], z[query], choice['residual'])
    residual += base[query]
    if rloo is not None:
        rloo += base[support]
    weight = b.blend_weight(support_y, lloo, rloo, choice['blend_w0'], choice['blend_c'], choice['blend_dscale'])
    frozen.update(local_ridge_retuned=local,
                  adaptive_blend_retuned=(1 - weight) * local + weight * residual,
                  local_rbf_retuned=comp.local_rbf(z[support], support_y, z[query], **choice['rbf']))
    assert tuple(frozen) == METHODS and all(np.isfinite(p).all() for p in frozen.values())
    return frozen


def evaluate(data, zs, bases, infos, gate, gamma, controls, directory, seed, fold):
    y = data.C2_yield_percent.to_numpy(float)
    cells, ledger, predictions, cids, qids = [], [], [], [], []
    full = zs['ocm_f1']
    alphas = alpha_schedule(infos['ocm_f1'])
    for source, sub in data.groupby('canonical_source_id', sort=True):
        for scenario, (query, pool, meta) in b.designs(sub).items():
            orders = selection_orders(full, data.row_id.to_numpy(str), query, pool, alphas,
                                        gate, gamma, b.stable_seed('step8_target', source, scenario, seed))
            for policy, records in orders.items():
                for repeat, order, diagnostic in records:
                    design_id = len(ledger)
                    ledger.append(dict(design_id=design_id, source=str(source), scenario=scenario,
                        policy=policy, repeat=repeat, query_indices=json.dumps(query.tolist()),
                        pool_indices=json.dumps(pool.tolist()), support_indices=json.dumps(order.tolist()),
                        diagnostic=json.dumps(diagnostic), metadata=json.dumps(meta)))
                    for task in ('ocm_f0', 'ocm_f1'):
                        for k in BUDGETS:
                            support = order[:k]
                            pp = predict(zs[task], bases[task], y[support], support, query,
                                         infos[task], controls[task][policy][str(k)], k)
                            row = dict(cell_id=len(cells), design_id=design_id, task=task,
                                seed=seed, fold=fold, source=str(source),
                                leakage_group=str(sub.leakage_group_id.iloc[0]), scenario=scenario,
                                policy=policy, repeat=repeat, k=k, n_test=len(query))
                            for method, pred in pp.items():
                                row['mae__' + method] = float(np.abs(y[query] - pred).mean())
                                row['mse__' + method] = float(np.square(y[query] - pred).mean())
                            cells.append(row)
                            predictions.append(np.column_stack(list(pp.values())))
                            cids.extend([row['cell_id']] * len(query))
                            qids.extend(query.tolist())
    save_csv(pd.DataFrame(cells), directory / 'cells.csv.gz')
    save_csv(pd.DataFrame(ledger), directory / 'support_ledger.csv.gz')
    save_csv(data, directory / 'evaluated_input.csv.gz')
    if not predictions:
        raise ValueError('No eligible evaluation sources')
    np.savez_compressed(require_server(directory / 'row_predictions.npz'),
        predictions=np.vstack(predictions), cell_id=np.asarray(cids, np.int32),
        query_index=np.asarray(qids, np.int32), methods=np.asarray(METHODS, dtype='U32'))
    np.savez_compressed(require_server(directory / 'geometry.npz'),
        z_f0=zs['ocm_f0'], z_f1=zs['ocm_f1'], base_f0=bases['ocm_f0'], base_f1=bases['ocm_f1'])
    return len(cells), sum(len(p) for p in predictions)


def run_part(seed, fold, identity, run_root):
    directory = run_root / 'parts' / f'ocm_{seed}_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'completed.json').exists():
        assert json.loads((directory / 'completed.json').read_text())['identity'] == identity
        return
    infos = {task: json.loads((REFERENCE / 'results/parts' / f'{task}_{seed}_{fold}' /
                              'completed.json').read_text()) for task in ('ocm_f0', 'ocm_f1')}
    data, cfg = b.load_task('ocm_f1')
    allocation = pd.read_csv(b.SOURCE / 'benchmark_splits/ocm_f1/split_ledger.csv.gz')
    allocation = allocation[(allocation.outer_seed == seed) & (allocation.fold == fold)]
    tr_ids = set(allocation.loc[allocation.role.eq('train'), 'row_id'].astype(str))
    training = data[data.row_id.astype(str).isin(tr_ids)].reset_index(drop=True)
    test = data[~data.row_id.astype(str).isin(tr_ids)].reset_index(drop=True)
    assert set(training.leakage_group_id).isdisjoint(test.leakage_group_id)
    old_part = previous.REFERENCE / 'results/parts' / f'ocm_f1_{seed}_{fold}_C2_yield_percent'
    zs, inner, _, audits = previous.historical_geometry(training, cfg, old_part, seed, fold)
    bases, ztest = {}, {}
    for task in ('ocm_f0', 'ocm_f1'):
        old = previous.REFERENCE / 'results/parts' / f'{task}_{seed}_{fold}_C2_yield_percent'
        oof = pd.read_csv(old / 'training_oof.csv.gz')
        np.testing.assert_array_equal(oof.row_id.astype(str), training.row_id.astype(str))
        bases[task] = oof.base.to_numpy(float)
        part = REFERENCE / 'results/parts' / f'{task}_{seed}_{fold}'
        archived = pd.read_csv(part / 'evaluated_input.csv.gz')
        np.testing.assert_array_equal(archived.row_id.astype(str), test.row_id.astype(str))
        geometry = np.load(part / 'evaluation_geometry.npz')
        ztest[task] = (geometry['z'], geometry['base'])
    np.testing.assert_allclose(ztest['ocm_f0'][0], ztest['ocm_f1'][0][:, :6], atol=1e-12)
    gate, gamma, controls = tune(training, zs, inner, bases, infos, directory)
    write_json(directory / 'training_source_audit.json', audits)
    n_cells, n_rows = evaluate(test, {t:v[0] for t,v in ztest.items()}, {t:v[1] for t,v in ztest.items()},
                              infos, gate, gamma, controls, directory, seed, fold)
    write_json(directory / 'completed.json', dict(identity=identity, seed=seed, fold=fold,
        n_cells=n_cells, n_prediction_rows=n_rows, new_outer_backbone_fits=0,
        history_training_sources=sorted(set(training.canonical_source_id)),
        history_training_groups=sorted(set(training.leakage_group_id)),
        evaluation_groups=sorted(set(test.leakage_group_id)),
        inherited_infos=infos, completed_utc=utc()))
    print('DONE', seed, fold, n_cells, flush=True)


def experiment_lock(run_root, seeds, folds):
    config = dict(version='step8-view-gate-1', seeds=seeds, folds=folds,
        policies=POLICIES, predictors=METHODS, budgets=BUDGETS,
        primary='ocm_f0 frozen adaptive_blend; random queries; five labels; view_gate minus five fixed controls',
        primary_controls=['composition_vopt','operating_vopt','kernel_vopt','robust_view_vopt','fixed_best_view'],
        primary_uncertainty='equal sources; draws then seeds averaged; conditional leakage-group bootstrap; Bonferroni 99% intervals',
        data_status='historical development; not new confirmation',
        code_hashes={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},
        historical_input_hashes={p.name:sha(p) for p in (b.SOURCE/'inputs').glob('*.csv')},
        reference_release_manifest_sha256=sha(REFERENCE/'release_manifest.json'),
        plan_sha256=sha(ROOT/'protocol/analysis_plan.json'))
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    lock = run_root / 'experiment_lock.json'
    if lock.exists():
        assert json.loads(lock.read_text())['identity'] == identity, 'Use a new run directory after a protocol change'
    else:
        write_json(lock, dict(identity=identity, created_utc=utc(), config=config))
    return identity


def main():
    require_server()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='history_v1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    args = parser.parse_args()
    run_root = require_server(ROOT / 'runs' / args.run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    identity = experiment_lock(run_root, args.seeds, args.folds)
    jobs = list(itertools.product(args.seeds, args.folds))
    Parallel(n_jobs=args.workers)(delayed(run_part)(seed, fold, identity, run_root)
                                  for seed, fold in jobs)
    write_json(run_root / 'run_completion.json', dict(parts=len(jobs), identity=identity,
        full_historical_grid_completed=set(args.seeds)==set(SEEDS) and set(args.folds)==set(range(5)),
        status='completed', completed_utc=utc(), formal_training_location=str(ROOT)))


if __name__ == '__main__':
    main()

"""Freeze history-only method parameters, then reapply to the opened cohort.

This is development on a previously evaluated publication. Catalyst holdout
changes the question, not the provenance or independence of the dataset.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import run_matched as run


def verify_previous_freeze():
    frozen = json.loads((run.REFERENCE / 'runs/full_history_v1/frozen.json').read_text())
    for name, expected in frozen['file_hashes'].items():
        if run.sha(run.REFERENCE / name) != expected:
            raise ValueError('Previous pipeline changed: ' + name)
    return frozen


def freeze(name='full_history_v1'):
    previous = verify_previous_freeze()
    directory = run.ROOT / 'runs' / name
    if (directory / 'frozen.json').exists():
        print('Already frozen; not counted as a new fit')
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Preserve partial freeze and choose a new version')
    completion = json.loads((run.ROOT / 'runs/history_v1/run_completion.json').read_text())
    audit = json.loads((run.ROOT / 'runs/history_v1/verification/audit.json').read_text())
    if not completion['full_historical_grid_completed'] or not audit['passed']:
        raise ValueError('Complete and audit the historical grid first')
    directory.mkdir(parents=True, exist_ok=True)
    original = run.REFERENCE / 'runs/full_history_v1'
    training = pd.read_csv(original / 'historical_input.csv.gz')
    oof = pd.read_csv(original / 'ocm_f1_training_oof.csv.gz')
    np.testing.assert_array_equal(training.row_id.to_numpy(str), oof.row_id.to_numpy(str))
    inner = oof.inner_fold.to_numpy(int)
    _, cfg = run.b.load_task('ocm_f1')
    zs, audits = {}, []
    for j in sorted(set(inner)):
        tr, va = np.where(inner != j)[0], np.where(inner == j)[0]
        if not set(training.leakage_group_id.iloc[tr]).isdisjoint(training.leakage_group_id.iloc[va]):
            raise ValueError('Historical inner-group leakage')
        z, metadata = run.b.fitted_transform(training.iloc[tr], training, cfg)
        zs[int(j)] = z
        audits.append(dict(inner_fold=int(j), training_groups=sorted(set(training.leakage_group_id.iloc[tr])),
                           validation_groups=sorted(set(training.leakage_group_id.iloc[va])),
                           transform=metadata, new_backbone_fits=0))
    run.tune(training, zs, inner, directory)
    run.write_json(directory / 'training_source_audit.json', audits)
    hashes = {str(p.relative_to(run.ROOT)):run.sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    hashes.update({str(p.relative_to(run.ROOT)):run.sha(p) for p in (run.ROOT / 'code').glob('*.py')})
    hashes['protocol/analysis_plan.json'] = run.sha(run.ROOT / 'protocol/analysis_plan.json')
    run.write_json(directory / 'frozen.json', dict(frozen_utc=run.utc(), status='history_parameters_frozen',
        file_hashes=hashes, new_backbone_fits=0,
        inherited_backbone_fits_originally_executed_in_step9=previous['actual_backbone_fits'],
        previous_pipeline_freeze_sha256=run.sha(original / 'frozen.json'),
        historical_run_identity=completion['identity'],
        cohort_already_opened_in_step9=True,
        statistical_status='development; subsequent cohort application is not unseen confirmation'))
    print('New history-only parameters frozen; no new backbone fits')


def catalyst_ledger(data, z, old_parameters, infos):
    provenance = pd.read_csv(run.REFERENCE / 'data/curated/nishimura_measured_v1/row_provenance.csv')
    lookup = provenance.set_index('row_id').catalyst_id.to_dict()
    catalysts = np.array([lookup[str(row)] for row in data.row_id])
    unique = np.array(sorted(set(catalysts)))
    source = str(data.canonical_source_id.iloc[0])
    selected = np.random.default_rng(run.b.stable_seed(
        'step10_catalyst_holdout', source)).permutation(unique)[:math.ceil(.25 * len(unique))]
    query = np.where(np.isin(catalysts, selected))[0]
    pool = np.where(~np.isin(catalysts, selected))[0]
    if set(catalysts[query]) & set(catalysts[pool]):
        raise ValueError('Catalyst identity crosses held-out split')
    if len(query) < 5 or len(pool) < 10:
        raise ValueError('Ineligible catalyst holdout')
    orders = run.old.selection_orders(z, data.row_id.to_numpy(str), query, pool,
        run.old.alpha_schedule(infos['ocm_f1']),
        run.old.ViewGate.from_dict(old_parameters['gate']), old_parameters['gamma'],
        run.b.stable_seed('step10_target_catalyst_holdout', source))
    rows = []
    for policy, records in orders.items():
        if policy not in run.BASE_POLICIES:
            continue
        for repeat, order, diagnostic in records:
            rows.append(dict(design_id=len(rows), source=source, scenario='catalyst_holdout', policy=policy,
                repeat=repeat, query_indices=json.dumps(query.tolist()), pool_indices=json.dumps(pool.tolist()),
                support_indices=json.dumps(order.tolist()), diagnostic=json.dumps(diagnostic),
                metadata=json.dumps(dict(query_catalysts=sorted(set(catalysts[query])),
                    pool_catalysts=sorted(set(catalysts[pool])), n_query_rows=len(query),
                    scope='one opened publication; catalyst-disjoint stress, not a second independent cohort'))))
    return pd.DataFrame(rows)


def apply(name='nishimura_opened_v1', freeze_name='full_history_v1'):
    verify_previous_freeze()
    frozen_dir = run.ROOT / 'runs' / freeze_name
    frozen = json.loads((frozen_dir / 'frozen.json').read_text())
    for path, expected in frozen['file_hashes'].items():
        if run.sha(run.ROOT / path) != expected:
            raise ValueError('New frozen pipeline changed: ' + path)
    directory = run.ROOT / 'runs/applications' / name
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Preserve completed/partial application; do not silently overwrite')
    part = directory / 'parts/cohort'; part.mkdir(parents=True, exist_ok=True)
    original = run.REFERENCE / 'runs/new_cohorts/nishimura_measured_v1/parts/cohort'
    data = pd.read_csv(original / 'evaluated_input.csv.gz')
    geo = np.load(original / 'geometry.npz')
    geometry = {key:geo[key] for key in geo.files}
    old_parameters = json.loads((original / 'parameters.json').read_text())
    infos = json.loads((original / 'completed.json').read_text())['inherited_infos']
    parameters_record = json.loads((frozen_dir / 'parameters.json').read_text())
    parameters = parameters_record['tasks']
    baseline = pd.read_csv(original / 'support_ledger.csv.gz')
    extra = catalyst_ledger(data, geometry['z_f1'], old_parameters, infos)
    baseline = pd.concat([baseline, extra], ignore_index=True)
    run.write_json(directory / 'application_lock.json', dict(locked_utc=run.utc(),
        freeze_sha256=run.sha(frozen_dir / 'frozen.json'),
        normalized_data_sha256=run.sha(original / 'evaluated_input.csv.gz'),
        data_previously_evaluated=True, n_publications=1, n_experiment_families=1,
        prospective_measurements=False, catalyst_holdout_defined_using_ids_only=True,
        inference='point estimates; no independent-source confidence interval'))
    run.write_json(part / 'parameters.json', parameters_record)
    run.write_json(part / 'inherited_parameters.json', old_parameters)
    n_cells, n_rows = run.evaluate(data, geometry, infos, old_parameters, parameters,
                                 baseline, part, 81099, 0)
    run.write_json(part / 'completed.json', dict(identity=run.sha(frozen_dir / 'frozen.json'), seed=81099,
        fold=0, n_cells=n_cells, n_prediction_rows=n_rows, n_scalar_predictions=n_rows * len(run.METHODS),
        new_outer_backbone_fits=0, history_training_groups=sorted(set(
            pd.read_csv(run.REFERENCE / 'runs/full_history_v1/historical_input.csv.gz').leakage_group_id)),
        evaluation_groups=sorted(set(data.leakage_group_id)), inherited_infos=infos,
        completed_utc=run.utc()))
    run.write_json(directory / 'application_completion.json', dict(status='completed', completed_utc=run.utc(),
        n_cells=n_cells, n_query_records=n_rows, scalar_predictions=n_rows * len(run.METHODS),
        previously_opened_dataset=True, n_publications=1, n_experiment_families=1,
        new_catalyst_holdout=True, new_wet_lab_experiments=False))
    print('Opened-publication application completed:', n_cells, n_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'apply'))
    parser.add_argument('--name', default=None)
    parser.add_argument('--freeze-name', default='full_history_v1')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.name or args.freeze_name)
    else:
        apply(args.name or 'nishimura_opened_v1', args.freeze_name)


if __name__ == '__main__':
    main()

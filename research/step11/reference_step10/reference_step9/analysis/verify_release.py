"""Read-only integrity/evidence checks using only the Python standard library.

This checks the completed evidence and byte identities. It does not repeat
training, certify measurement semantics, or claim a second experiment.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(name):
    p = (ROOT / name).resolve()
    require(p.is_relative_to(ROOT), 'Path outside package: ' + str(name))
    return p


def load(name):
    return json.loads(path(name).read_text(encoding='utf-8'))


def rows(name):
    with path(name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    p = p.resolve()
    if p not in HASH_CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        HASH_CACHE[p] = h.hexdigest()
    return HASH_CACHE[p]


def identities(records, prefix=''):
    for item in records:
        p = path(str(Path(prefix) / item['path']))
        require(p.is_file(), 'Missing file: ' + str(p.relative_to(ROOT)))
        require(p.stat().st_size == item['bytes'], 'Byte count changed: ' + str(p))
        require(sha(p) == item['sha256'], 'SHA256 changed: ' + str(p))
    return len(records)


def mean_score(table, task, policy, method='adaptive_blend'):
    found = [r for r in table if r['task'] == task and r['policy'] == policy
             and r['scenario'] == 'random' and int(r['k']) == 5]
    require(len(found) == 1, 'Nonunique result: ' + task + '/' + policy)
    return float(found[0]['mae__' + method])


def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10),
            'Numeric mismatch: ' + label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-only', action='store_true',
                        help='Run evidence checks before the final release manifest exists.')
    args = parser.parse_args()
    result = {'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'type': 'read-only recorded evidence and byte-identity verification',
              'model_refits_performed': 0}
    if not args.inputs_only:
        manifest = load('release_manifest.json')
        result['release_manifest_files_checked'] = identities(manifest['files'])
    else:
        result['release_manifest_files_checked'] = None
        result['release_manifest_status'] = 'not checked in pre-manifest inputs-only mode'

    predecessor = load('reference_step7/release_manifest.json')
    result['step7_unchanged_manifest_files'] = identities(predecessor['files'], 'reference_step7')
    require(result['step7_unchanged_manifest_files'] == 2317, 'Incomplete Step7 predecessor')
    preserved = load('reports/step9_predecessor_preservation.json')
    result['step8_original_files_preserved'] = identities([
        dict(path=r['preserved_path'], bytes=r['bytes'], sha256=r['sha256'])
        for r in preserved['records']])
    require(result['step8_original_files_preserved'] == 2357, 'Incomplete Step8 preservation')

    lock = load('runs/history_v1/experiment_lock.json')
    completion = load('runs/history_v1/run_completion.json')
    require(completion['status'] == 'completed' and completion['full_historical_grid_completed'],
            'Historical grid is not complete')
    require(completion['identity'] == lock['identity'], 'Historical identity mismatch')
    parts = [json.loads(p.read_text()) for p in sorted(path('runs/history_v1/parts').glob('*/completed.json'))]
    require(len(parts) == completion['parts'] == 25, 'Historical part count mismatch')
    require({(p['seed'], p['fold']) for p in parts} == {
        (s, f) for s in (42, 101, 202, 303, 404) for f in range(5)}, 'Missing seed/fold')
    for part in parts:
        require(part['identity'] == lock['identity'], 'Part identity mismatch')
        require(part['new_outer_backbone_fits'] == 0, 'Wrong historical backbone-count claim')
    for name, digest in lock['config']['code_hashes'].items():
        require(sha(path('code/' + name)) == digest, 'Historical code changed: ' + name)
    require(sha(path('protocol/analysis_plan.json')) == lock['config']['plan_sha256'],
            'Locked analysis plan changed')
    require(len(lock['config']['policies']) == 11 and len(lock['config']['predictors']) == 12,
            'Policy/predictor count mismatch')
    h_audit = load('runs/history_v1/verification/audit.json')
    require(h_audit['passed'] and h_audit['parts'] == 25, 'Historical audit missing')
    h_cells = sum(p['n_cells'] for p in parts)
    h_rows = sum(p['n_prediction_rows'] for p in parts)
    require(h_cells == 93150 and h_rows == 662400, 'Historical count mismatch')
    require(h_audit['prediction_values_reaggregated'] == h_rows * 12 == 7948800,
            'Historical scalar count mismatch')
    for name in ['runs/verification/math.json', 'runs/smoke_v1/verification/audit.json',
                 'runs/history_v1/verification/target_outcome_poison.json']:
        require(load(name)['passed'], 'Required verification failed: ' + name)

    frozen = load('runs/full_history_v1/frozen.json')
    for name, digest in frozen['file_hashes'].items():
        require(sha(path(name)) == digest, 'Frozen pipeline changed: ' + name)
    models = list(path('runs/full_history_v1/models').rglob('*.joblib'))
    require(len(models) == frozen['actual_backbone_fits'] == 56, 'Actual backbone count mismatch')
    require(len(frozen['file_hashes']) == 79, 'Frozen hash count mismatch')
    require(frozen['historical_run_identity'] == lock['identity'], 'Freeze identity mismatch')
    require(frozen['new_cohort_files_read_during_fitting'] is False, 'New data used during fitting')
    result.update(historical_parts=len(parts), historical_cells=h_cells,
                  historical_query_records=h_rows,
                  historical_scalar_predictions=h_rows * 12,
                  actual_final_backbone_fits=len(models), frozen_files_checked=79)

    cohort = 'data/curated/nishimura_measured_v1/'
    application = 'runs/new_cohorts/nishimura_measured_v1/'
    validation = load(cohort + 'validation.json')
    contract = load(cohort + 'curation_contract.json')
    external_lock = load(application + 'external_evaluation_lock.json')
    for record in [validation, external_lock['cohort_audit']]:
        require(record['passed'], 'Curated-data audit failed')
        for name, field in [('ocm_normalized.csv', 'normalized_sha256'),
                            ('row_provenance.csv', 'provenance_sha256'),
                            ('curation_contract.json', 'contract_sha256')]:
            require(sha(path(cohort + name)) == record[field], 'Curated data changed: ' + name)
    require(external_lock['pipeline_freeze_sha256'] == sha(path('runs/full_history_v1/frozen.json')),
            'External pipeline identity mismatch')
    require(contract['pipeline_freeze_sha256'] == external_lock['pipeline_freeze_sha256'],
            'Curation freeze identity mismatch')
    for item in contract['raw_assets']:
        require(sha(path(item['relative_path'])) == item['sha256'], 'Raw asset changed')
    original = rows('data/raw/nishimura2023_cads.csv')
    measured = rows(cohort + 'ocm_normalized.csv')
    provenance = rows(cohort + 'row_provenance.csv')
    ledger = rows(cohort + 'inclusion_exclusion.csv')
    require(len(original) == len(ledger) == 2928, 'Raw/ledger row mismatch')
    require(len(measured) == len(provenance) == validation['n_records'] == 1739,
            'Curated record count mismatch')
    require({r['row_id'] for r in measured} == {r['row_id'] for r in provenance},
            'Provenance row identity mismatch')
    require(len({r['row_id'] for r in measured}) == 1739, 'Duplicate curated row ID')
    require(len({r['catalyst_id'] for r in provenance}) == 116, 'Catalyst count mismatch')
    require(len({r['canonical_source_id'] for r in measured}) == 1
            and len({r['leakage_group_id'] for r in measured}) == 1, 'Incorrect source grouping')
    require(all(r['outcome_origin'] == 'measured' for r in provenance), 'Nonmeasured outcome')
    require(all(0 <= float(r['C2_yield_percent']) <= 100 for r in measured), 'Invalid yield unit/range')
    for field in ('CH4_partial_pressure_bar', 'O2_partial_pressure_bar',
                  'total_pressure_bar', 'contact_time_s'):
        require(all(r[field] in ('', 'nan', 'NaN') for r in measured),
                'Unreported condition was silently filled: ' + field)
    require(sum(r['included'].lower() == 'true' for r in ledger) == 1739,
            'Inclusion ledger mismatch')
    e_complete = load(application + 'cohort_completion.json')
    e_part = load(application + 'parts/cohort/completed.json')
    e_audit = load(application + 'verification/audit.json')
    require(e_complete['passed'] and e_audit['passed'], 'External application audit failed')
    require(e_part['identity'] == external_lock['pipeline_freeze_sha256'], 'External part mismatch')
    require(e_complete['cells'] == e_part['n_cells'] == 1380, 'External cell mismatch')
    require(e_complete['prediction_rows'] == e_part['n_prediction_rows'] == 620310,
            'External query-record mismatch')
    require(e_audit['prediction_values_reaggregated'] == e_part['n_prediction_rows'] * 12 == 7443720,
            'External scalar mismatch')
    require(e_complete['n_sources'] == e_complete['n_independent_experiment_families'] == 1,
            'Incorrect independent-source claim')
    require(e_complete['prospective_new_experiments'] is False, 'Incorrect wet-lab claim')
    frozen_time = dt.datetime.fromisoformat(frozen['frozen_utc'])
    eval_time = dt.datetime.fromisoformat(external_lock['locked_utc'])
    curate_time = dt.datetime.fromisoformat(load(cohort + 'curation_rules_pre_evaluation.json')['locked_utc'])
    require(frozen_time < curate_time < eval_time, 'Freeze/curation/evaluation chronology changed')

    hist = rows('runs/history_v1/aggregated/summary.csv')
    new = rows(application + 'aggregated/summary.csv')
    primary = rows('runs/history_v1/aggregated/primary_family.csv')
    external_primary = rows(application + 'aggregated/primary_family.csv')
    require(len(primary) == len(external_primary) == 5, 'Wrong primary-family size')
    require(all(r['adjusted_support_for_advantage'].lower() == 'false' for r in primary),
            'Wrong candidate superiority claim')
    require(all(not r['family_adjusted_ci_low'] and not r['family_adjusted_ci_high']
                for r in external_primary), 'Single-family population interval created')
    facts = load('reports/step9_measured_facts.json')
    gate = mean_score(hist, 'ocm_f0', 'view_gate')
    comp = mean_score(hist, 'ocm_f0', 'composition_vopt')
    random = mean_score(new, 'ocm_f1', 'random')
    adapted = mean_score(new, 'ocm_f1', 'composition_vopt')
    rbf = mean_score(new, 'ocm_f1', 'composition_vopt', 'local_rbf_retuned')
    close(gate, facts['gate_primary_mae'], 'gate MAE')
    close(comp, facts['composition_primary_mae'], 'composition MAE')
    close(random, facts['external_f1_random_blend_mae'], 'new random MAE')
    close(adapted, facts['external_f1_composition_blend_mae'], 'new composition MAE')
    close(rbf, facts['external_f1_local_rbf_composition_mae'], 'new RBF MAE')
    close(100 * (1 - adapted / random), facts['external_f1_composition_vs_random_reduction_percent'],
          'point reduction percent')
    require(rbf < adapted and gate > comp and facts['primary_advantage_supported'] is False,
            'Scientific interpretation mismatch')
    for row in primary:
        close(float(row['candidate_mae']), gate, 'primary gate score')
        close(float(row['control_mae']), mean_score(hist, 'ocm_f0', row['control']), 'control score')
        close(float(row['delta_mae']), float(row['candidate_mae']) - float(row['control_mae']),
              'primary delta')
    result.update(external_measured_rows=len(measured), external_catalysts=116,
                  external_experiment_families=1, external_cells=1380,
                  external_query_records=620310, external_scalar_predictions=7443720,
                  gate_primary_mae=gate, fixed_composition_primary_mae=comp,
                  adjusted_gate_superiority_supported=False,
                  external_f1_random_blend_mae=random,
                  external_f1_composition_blend_mae=adapted,
                  external_f1_local_rbf_mae=rbf,
                  external_same_budget_reduction_percent=100 * (1 - adapted / random))

    qa = load('reports/step9_pdf_layout_checks.json')
    visual = load('reports/step9_pdf_visual_review.json')
    require(visual['passed'] and visual['total_pages'] == 63, 'Incomplete PDF review')
    for name, record in visual['documents'].items():
        require(sha(path(name)) == record['sha256'], 'PDF changed after visual review: ' + name)
        require(record['visually_reviewed_pages'] == list(range(1, record['pages'] + 1)),
                'Missing rendered-page review')
    require(all(not r['out_of_page_spans'] for r in qa.values()), 'PDF clipping recorded')
    for name in visual['final_latex_logs']:
        text = path(name).read_text(errors='replace')
        require('Overfull' not in text and 'undefined' not in text, 'Unresolved LaTeX layout/reference')
    result['visually_reviewed_pdf_pages'] = 63
    result['passed'] = True
    result['limits'] = ('Recorded execution and hash checks, with previous actual scalar/order audits; '
                        'not a new training run, independent laboratory replication, or a Q1 acceptance guarantee.')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

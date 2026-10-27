"""Enforce provenance and units before accepting a new OCM test cohort."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from server_guard import ROOT, require_server, sha, utc, write_json
import run_history as experiment

AUDIT_KEYS = ('original_measurements_only','historical_overlap_review_complete',
              'previous_external_overlap_review_complete','units_review_complete',
              'batch_and_replicate_review_complete','model_development_did_not_use_rows')
PROVENANCE_COLUMNS = ('row_id','original_measurement_doi','raw_asset_sha256',
    'raw_row_or_table_reference','outcome_origin','lab_id','catalyst_id',
    'measurement_run_id','replicate_group_id')


def validate(cohort):
    directory=require_server(ROOT/'data/curated'/cohort)
    contract=json.loads((directory/'curation_contract.json').read_text())
    for key in AUDIT_KEYS:
        if contract.get(key) is not True:
            raise ValueError('Curator has not completed: '+key)
    if contract.get('reaction')!='OCM' or contract.get('yield_definition')!='100 * C2 product carbon / inlet CH4 carbon':
        raise ValueError('Reaction or yield definition does not match the benchmark')
    if not contract.get('source_publication_dois') or not contract.get('raw_assets'):
        raise ValueError('Source and raw-asset provenance are required')
    data=pd.read_csv(directory/'ocm_normalized.csv')
    provenance=pd.read_csv(directory/'row_provenance.csv').fillna('')
    required={'row_id','canonical_source_id','leakage_group_id','Temperature_K',
              'CH4_partial_pressure_bar','O2_partial_pressure_bar','CH4_O2_ratio',
              'total_pressure_bar','contact_time_s','C2_yield_percent',
              *experiment.b.COMPOSITION_COLUMNS}
    if not required <= set(data):
        raise ValueError('Missing normalized input columns: '+str(sorted(required-set(data))))
    if not set(PROVENANCE_COLUMNS)<=set(provenance):
        raise ValueError('Row-level provenance schema is incomplete')
    if not data.row_id.is_unique or not provenance.row_id.is_unique:
        raise ValueError('Repeated row identifiers')
    if set(data.row_id)!=set(provenance.row_id):
        raise ValueError('Normalized rows and provenance rows disagree')
    if not provenance.outcome_origin.eq('measured').all():
        raise ValueError('Predictions or simulated yields cannot serve as measured outcomes')
    if provenance.original_measurement_doi.eq('').any() or provenance.raw_row_or_table_reference.eq('').any():
        raise ValueError('An original measurement/table reference is missing')
    if not np.isfinite(data[['Temperature_K','CH4_O2_ratio','C2_yield_percent']].to_numpy(float)).all():
        raise ValueError('Required reported conditions or measured responses are missing')
    if not data.C2_yield_percent.between(0,100).all():
        raise ValueError('Physical outcome QC requires a recorded exclusion; do not clip labels')
    for _, g in data.groupby('canonical_source_id'):
        if g.leakage_group_id.nunique()!=1:
            raise ValueError('A source must belong to a single experiment-family leakage group')
    sources = experiment.b.SOURCE/'inputs'
    known=set()
    for file in sources.glob('*.csv'):
        frame=pd.read_csv(file)
        if 'canonical_source_id' in frame:
            known.update(frame.canonical_source_id.astype(str))
    old_external=ROOT/'reference_prior_external/external_model_input.csv'
    if not old_external.exists():
        raise ValueError('Previous external cohort is missing; cannot declare new independence')
    frame=pd.read_csv(old_external)
    known.update(frame.canonical_source_id.astype(str))
    if known & set(data.canonical_source_id.astype(str)):
        raise ValueError('Previously used source identities appear in the new cohort')
    # Identity checks do not detect repackaged measurements. A curator must
    # additionally review DOI/bibliography, features, responses and lab/batches.
    for asset in contract['raw_assets']:
        path=require_server(ROOT/asset['relative_path'])
        if sha(path)!=asset['sha256']:
            raise ValueError('Raw asset hash mismatch')
    active=[len(g) for _,g in data.groupby('canonical_source_id') if len(g)>=15]
    if not active:
        raise ValueError('No source supplies at least 15 compatible records')
    report=dict(passed=True,validated_utc=utc(),cohort=cohort,
        normalized_sha256=sha(directory/'ocm_normalized.csv'),
        provenance_sha256=sha(directory/'row_provenance.csv'),
        contract_sha256=sha(directory/'curation_contract.json'),
        n_records=len(data),n_sources=data.canonical_source_id.nunique(),
        n_experiment_family_groups=data.leakage_group_id.nunique(),
        eligible_sources=len(active),
        condition_missing_fractions={c:float(data[c].isna().mean()) for c in
            ('CH4_partial_pressure_bar','O2_partial_pressure_bar','total_pressure_bar','contact_time_s')},
        caveat='Curator attestation is required for semantic provenance; row identifiers alone do not establish independent measurements')
    write_json(directory/'validation.json',report)
    return report


def main():
    require_server()
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.cohort),indent=2))


if __name__=='__main__':
    main()

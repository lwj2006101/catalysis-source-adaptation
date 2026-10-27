"""Deterministic, provenance-based curation before any cohort scoring.

Only the five explicitly identified experimental validation phases enter.
The paper's mixed literature/HTS List0 and CADS controls/unpublished blocks
are excluded. All validation phases form one publication/experiment family.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from server_guard import sha, utc, write_json
import run_history as experiment

DOI = '10.1039/d3cy00596h'
COHORT = 'nishimura_measured_v1'
PHASES = ('SVR-1', 'SVR-2', 'Bayesian-1', 'Bayesian-2', 'Bayesian-3')
SUPPORTS = {'SiO2':'Si', 'TiO2':'Ti', 'MgO':'Mg', 'CaO':'Ca', 'BaO':'Ba',
            'ZnO':'Zn', 'La2O3':'La', 'Y2O3':'Y'}


def numeric(value):
    return pd.to_numeric(str(value).strip().replace(',', '.'), errors='coerce')


def species(value):
    value = str(value).strip()
    return '' if value.lower() in ('none', '', 'nan') else value


def main():
    dest = ROOT / 'data/curated' / COHORT
    if (ROOT / 'runs/new_cohorts' / COHORT).exists():
        raise ValueError('Cohort already evaluated: preserve its curation and results')
    dest.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((ROOT / 'runs/full_history_v1/frozen.json').read_text())
    for name, expected in freeze['file_hashes'].items():
        assert sha(ROOT / name) == expected, name
    raw_path = ROOT / 'data/raw/nishimura2023_cads.csv'
    raw = pd.read_csv(raw_path, keep_default_na=False)
    raw_sha = sha(raw_path)
    source = 'OCM-NISHIMURA-2023-d3cy00596h'
    group = 'OCM-JAIST-NISHIMURA-2023-EXPERIMENT-FAMILY'
    write_json(dest / 'curation_rules_pre_evaluation.json', {
        'locked_utc':utc(), 'pipeline_freeze_sha256':sha(ROOT/'runs/full_history_v1/frozen.json'),
        'cohort':COHORT, 'included_resource_phases':PHASES,
        'excluded_resource_phases':'STD, Refernce/reference, Unpublished: controls or insufficient origin specificity',
        'non_response_rows':'temperature missing or nonnumeric yield: exclude; these include water-summary lines',
        'numeric_parsing':'decimal comma to dot; reported Celsius plus 273.15',
        'repeated_condition_rule':'retain first original CSV occurrence, using only furnace, feed ratio, sorted metals, support and temperature; no averaging labels',
        'same_publication_grouping':'all five validation phases are one source and one experiment family',
        'support_token_mapping':SUPPORTS, 'loading':'no mol% denominator inferred from mmol loading per support mass',
        'unreported_conditions':'leave partial pressures, total pressure and contact time missing; no assumed 1 bar',
        'physical_outcome_rule':'exclude reported yields outside 0..100, never clip',
        'external_prediction_performance_computed_before_this_lock':False})
    rows, provenance, decisions, seen = [], [], [], {}
    for index, row in raw.iterrows():
        raw_line = index + 2
        phase = row['Resource']
        metals = sorted(species(row[c]) for c in ('M1','M2','M3') if species(row[c]))
        support = species(row['Support'])
        temp, ratio, response = [numeric(row[c]) for c in ('Temp(℃)','CH4/O2 ratio','C2 yield(%)')]
        reason = ''
        duplicate_of = ''
        if phase not in PHASES:
            reason = 'excluded_control_or_unresolved_origin_phase'
        elif not np.isfinite([temp, ratio, response]).all():
            reason = 'no_complete_reported_temperature_ratio_and_response'
        elif not 0 <= response <= 100:
            reason = 'reported_yield_outside_physical_bounds'
        elif support not in SUPPORTS:
            reason = 'support_not_compatible_with_single_cation_oxide_mapping'
        else:
            condition = (str(row['furnace length(mm)']), float(ratio), tuple(metals), support, float(temp))
            if condition in seen:
                reason = 'repeated_catalyst_condition_first_occurrence_retained'
                duplicate_of = seen[condition]
            else:
                seen[condition] = raw_line
        row_id = f'NISHIMURA-2023-CSV-{raw_line:05d}'
        decisions.append(dict(raw_csv_line=raw_line, row_id=row_id, resource=phase,
            included=not reason, exclusion_reason=reason, duplicate_of_raw_csv_line=duplicate_of))
        if reason:
            continue
        record = {c:np.nan for c in experiment.b.COMPOSITION_COLUMNS}
        for slot, element in enumerate(metals, 1):
            record[f'Cation {slot}'] = element
        record['Support 1'] = SUPPORTS[support]
        record.update(row_id=row_id,canonical_source_id=source,leakage_group_id=group,
            Temperature_K=float(temp+273.15),CH4_partial_pressure_bar=np.nan,
            O2_partial_pressure_bar=np.nan,CH4_O2_ratio=float(ratio),total_pressure_bar=np.nan,
            contact_time_s=np.nan,C2_yield_percent=float(response),experimental_phase=phase,
            furnace_length_mm=int(row['furnace length(mm)']),original_support_formula=support)
        rows.append(record)
        catalyst = '|'.join([*metals, support])
        catalyst_id = 'CAT-' + hashlib.sha256(catalyst.encode()).hexdigest()[:16]
        provenance.append(dict(row_id=row_id,original_measurement_doi=DOI,raw_asset_sha256=raw_sha,
            raw_row_or_table_reference=f'author CADS CSV line {raw_line}; Resource={phase}; original paper section 3 validation phase, Table S3 counterpart',
            outcome_origin='measured',lab_id='JAIST-NISHIMURA-OCM',catalyst_id=catalyst_id,
            measurement_run_id=f'CSV-block-{index//16:03d}-{phase}',
            replicate_group_id=catalyst_id+'-T-'+str(temp)+'-ratio-'+str(ratio),
            original_cation_slots='|'.join(str(row[c]) for c in ('M1','M2','M3')),
            reported_support_formula=support))
    normalized = pd.DataFrame(rows)
    prov = pd.DataFrame(provenance)
    experiment.save_csv(normalized, dest/'ocm_normalized.csv')
    experiment.save_csv(prov, dest/'row_provenance.csv')
    experiment.save_csv(pd.DataFrame(decisions), dest/'inclusion_exclusion.csv')
    # Search approximate fingerprints for possible copied measurements. New
    # original validation phases and their provenance, not dates alone, establish
    # their eligibility. A shared catalyst or rounded yield is not a copied run.
    previous = [ROOT/'reference_step7/reference_step3/benchmark/source_snapshot/inputs/ocm_paired_raw_chemistry.csv',
                ROOT/'reference_prior_external/external_model_input.csv']
    old_rows = []
    for path in previous:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            cations = sorted(species(row[f'Cation {k}']) for k in range(1,5) if species(row[f'Cation {k}']))
            supports = sorted(species(row[f'Support {k}']) for k in range(1,3) if species(row[f'Support {k}']))
            old_rows.append(dict(dataset=path.name,old_row_id=row.row_id,source=row.canonical_source_id,
                cations=tuple(cations),supports=tuple(supports),temp=float(row.Temperature_K),
                ratio=float(row.CH4_O2_ratio),response=float(row.C2_yield_percent)))
    overlap = []
    for row in normalized.itertuples(index=False, name=None):
        item = dict(zip(normalized.columns,row))
        cations = tuple(sorted(species(item[f'Cation {k}']) for k in range(1,5) if species(item[f'Cation {k}'])))
        support = (item['Support 1'],)
        for old in old_rows:
            if (old['cations']==cations and old['supports']==support and
                abs(old['temp']-item['Temperature_K'])<=1.0 and
                abs(old['ratio']-item['CH4_O2_ratio'])<=1e-6 and
                abs(old['response']-item['C2_yield_percent'])<=0.15):
                overlap.append(dict(new_row_id=item['row_id'],old_dataset=old['dataset'],old_row_id=old['old_row_id'],
                    old_source=old['source'],temperature_difference_K=old['temp']-item['Temperature_K'],
                    yield_difference_pp=old['response']-item['C2_yield_percent'],
                    review='rounded-condition coincidence; new study explicitly measures the validation phase; absent full old/new reactor identity, not proof of copied measurements'))
    columns=['new_row_id','old_dataset','old_row_id','old_source','temperature_difference_K','yield_difference_pp','review']
    experiment.save_csv(pd.DataFrame(overlap,columns=columns),dest/'overlap_review.csv')
    # Report bibliographic evidence and the unavoidable independence limits.
    bibliography=(ROOT/'reference_prior_external/old_bibliography.csv').read_text().lower()
    external=pd.read_csv(ROOT/'reference_prior_external/external_model_input.csv')
    assert DOI not in bibliography
    assert not external.reference.fillna('').str.lower().str.contains(DOI,regex=False).any()
    measured=raw[raw.Resource.isin(PHASES)].copy()
    for c in ('C2 yield(%)','C2H6 yield(%)','C2H4 yield(%)','CH4 Conv(%)','CO yield(%)','CO2 yield(%)','C missing(%)'):
        measured[c]=measured[c].map(numeric)
    diff_c2=np.abs(measured['C2 yield(%)']-measured['C2H6 yield(%)']-measured['C2H4 yield(%)'])
    diff_carbon=np.abs(measured['CH4 Conv(%)']-measured['C2 yield(%)']-measured['CO yield(%)']-measured['CO2 yield(%)']-measured['C missing(%)'])
    assert diff_c2.max()<=0.100001 and diff_carbon.max()<=0.200001
    assets=[{'relative_path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in
            (raw_path,ROOT/'data/raw/nishimura2023_original_article.pdf')]
    contract={key:True for key in ('original_measurements_only','historical_overlap_review_complete',
        'previous_external_overlap_review_complete','units_review_complete',
        'batch_and_replicate_review_complete','model_development_did_not_use_rows')}
    contract.update(status='curated_before_first_frozen_application',reaction='OCM',
        yield_definition='100 * C2 product carbon / inlet CH4 carbon',source_publication_dois=[DOI],raw_assets=assets,
        evidence='Original article sections 2.1, 2.2, 2.4 and 3 identify the five measured validation phases, and Table S3; data availability links author CADS measurements. Mixed List0, prediction lists, unpublished and control rows are not evaluated.',
        outcome_units_evidence='Reported C2 carbon yield; ethane plus ethylene yield agrees within 0.1 pp, and conversion minus carbon products agrees with reported missing carbon within 0.2 pp. Carbon-basis interpretation also follows the original N2-internal-standard method; not an outlet mole fraction.',
        source_evidence='Historical and previous external records originate from compilations through 2019. The paper reports freshly prepared and measured validation phases in 2023; publication date alone is not the inclusion argument.',
        replicate_handling='One first row per catalyst-condition signature across validation phases; no retained repeated signature.',
        limitations=['One publication and one experiment family, not five independent studies.',
                    'Publication/lab separation beyond reported provenance cannot be certified for all historical labs.',
                    'Four operating variables and all mol percentages are unreported in this mapped representation and remain missing.',
                    'Original curve measurements share catalyst runs; this offline query split does not test unseen catalysts or a prospective campaign.'],
        possible_rounded_fingerprint_matches=len(overlap),pipeline_freeze_sha256=sha(ROOT/'runs/full_history_v1/frozen.json'))
    write_json(dest/'curation_contract.json',contract)
    summary=dict(raw_rows=len(raw),eligible_resource_rows=int(raw.Resource.isin(PHASES).sum()),
        final_measured_rows=len(normalized),unique_catalysts=int(prov.catalyst_id.nunique()),
        n_sources=1,n_experiment_family_groups=1,exclusion_counts=pd.Series([d['exclusion_reason'] for d in decisions if not d['included']]).value_counts().to_dict(),
        n_possible_rounded_fingerprint_matches=len(overlap),C2_sum_max_disagreement_pp=float(diff_c2.max()),
        carbon_balance_max_disagreement_pp=float(diff_carbon.max()),curated_utc=utc())
    write_json(dest/'curation_summary.json',summary)
    (dest/'units_and_conditions.txt').write_text(
        'Temperature: original Celsius + 273.15 K. Feed CH4/O2 ratio is reported.\n'
        'C2 yield: carbon percent relative to inlet methane, reported directly; labels are not clipped.\n'
        'Validation protocol: 270 mm furnace, CH4/O2/N2 21/7/3 mL/min, 50 mg catalyst. Furnace and gas-flow provenance is retained; residence time is not derivable without an appropriate bed/gas-volume definition.\n'
        'Total pressure, partial pressures and residence/contact time: not explicitly resolved, so all remain NaN. Historical-only medians are used by the frozen transformer.\n'
        'Single-cation oxide supports are mapped to their support-cation symbols to match the historical schema, preserving original formulas in provenance. Oxidation state and support morphology are not inferred.\n'
        'Each precursor is loaded at a reported mmol-per-support-mass recipe; no historical mol-percent denominator is asserted, so all mol% fields remain NaN.\n'
        'All five phases are one publication/lab family. Same catalyst-condition occurrences are removed in original CSV order, without inspecting prediction errors or averaging responses.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

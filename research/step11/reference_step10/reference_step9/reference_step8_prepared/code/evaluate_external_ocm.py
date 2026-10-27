"""One frozen application to a genuinely new, curator-audited measured cohort."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from server_guard import ROOT, require_server, sha, utc, write_json
from check_external_data import validate
from view_selection import ViewGate
import run_history as experiment


def main():
    require_server()
    p=argparse.ArgumentParser()
    p.add_argument('--cohort',required=True)
    p.add_argument('--freeze-run-name',default='full_history_v1')
    args=p.parse_args()
    frozen_root=require_server(ROOT/'runs'/args.freeze_run_name)
    frozen=json.loads((frozen_root/'frozen.json').read_text())
    for name,expected in frozen['file_hashes'].items():
        if sha(require_server(ROOT/name))!=expected:
            raise ValueError('Frozen pipeline changed: '+name)
    audit=validate(args.cohort)
    directory=require_server(ROOT/'runs/new_cohorts'/args.cohort)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Preserve the existing cohort evaluation; no silent repeat or retune')
    directory.mkdir(parents=True,exist_ok=True)
    cohort_root=require_server(ROOT/'data/curated'/args.cohort)
    data=pd.read_csv(cohort_root/'ocm_normalized.csv').reset_index(drop=True)
    # Reuse the historical composition parser to define a label-free stress stratum.
    patterns=[]
    for _,row in data.iterrows():
        record=experiment.b.composition_record(row)
        terms=[value for role in record.values() for value in role.values()]
        patterns.append('' if not terms or any(value[2]>0 for value in terms)
                        else json.dumps(record,sort_keys=True))
    data['__stratum']=patterns
    infos=json.loads((frozen_root/'infos.json').read_text())
    parameters=json.loads((frozen_root/'parameters.json').read_text())
    zs,bases={},{}
    for task in ('ocm_f0','ocm_f1'):
        training,cfg=experiment.b.load_task(task)
        z,metadata=experiment.b.fitted_transform(training,data,cfg)
        model=joblib.load(frozen_root/'models'/task/'full_backbone.joblib')
        zs[task],bases[task]=z,model.predict(z)
    np.testing.assert_allclose(zs['ocm_f0'],zs['ocm_f1'][:,:6],atol=1e-12)
    # Stable source partition, fixed once. Random support orders use fifty draws;
    # none of these draws is an independent publication or laboratory.
    write_json(directory/'external_evaluation_lock.json',dict(locked_utc=utc(),cohort_audit=audit,
        pipeline_freeze_sha256=sha(frozen_root/'frozen.json'),
        primary='F0 frozen adaptive_blend; k=5; view_gate versus same five fixed controls',
        type='retrospective published measurements; not a prospective wet-lab campaign'))
    cells,rows=experiment.evaluate(data,zs,bases,infos,ViewGate.from_dict(parameters['gate']),
        parameters['gamma'],parameters['controls'],directory,81099,0)
    # Place the same record format under parts so ordinary audit and aggregation
    # can verify the complete scalar prediction bank without special scoring.
    parts=directory/'parts/cohort'
    parts.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ('cells.csv.gz','support_ledger.csv.gz','evaluated_input.csv.gz','row_predictions.npz','geometry.npz'):
        shutil.move(str(directory/name),str(parts/name))
    write_json(parts/'parameters.json',parameters)
    write_json(parts/'completed.json',dict(identity=sha(frozen_root/'frozen.json'),seed=81099,fold=0,
        n_cells=cells,n_prediction_rows=rows,new_outer_backbone_fits=0,inherited_infos=infos,
        history_training_groups=sorted(set(infos['ocm_f1']['training_groups'])),
        evaluation_groups=sorted(set(data.leakage_group_id)),completed_utc=utc()))
    from audit_results import audit as audit_results
    from aggregate_results import aggregate
    audit_results(directory)
    primary=aggregate(directory,data_scope='locked_application_to_curator_audited_published_measurements')
    write_json(directory/'cohort_completion.json',dict(completed_utc=utc(),passed=True,
        cells=cells,prediction_rows=rows,n_sources=int(audit['n_sources']),
        n_independent_experiment_families=int(audit['n_experiment_family_groups']),
        uncertainty='one source/family: point estimates only; no population confidence claim' if
                    audit['n_experiment_family_groups']<2 else
                    'conditional paired source/family bootstrap; no independent-laboratory claim without provenance',
        published_before_method_freeze=True,prospective_new_experiments=False,
        missing_condition_fractions=audit['condition_missing_fractions']))
    print(primary.to_string(index=False))


if __name__=='__main__':
    main()

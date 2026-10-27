"""Fit and freeze the final OCM pipeline without opening new cohort files."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from server_guard import ROOT, require_server, sha, utc, write_json
import run_history as experiment
b = experiment.b


def fit_task(data, cfg, task, directory):
    target='C2_yield_percent'
    y=data[target].to_numpy(float)
    sources=data.canonical_source_id.to_numpy(str)
    seed=81001
    parts=list(b.group_folds(data.leakage_group_id,b.stable_seed('inner',f'{seed}:full')))
    oof=np.empty((len(data),len(b.GRID)))
    inner=np.empty(len(data),int)
    geometries,scales,audits=[],[],[]
    model_dir=directory/'models'/task
    model_dir.mkdir(parents=True,exist_ok=True)
    count=0
    for fold,(train,held) in enumerate(parts):
        z,metadata=b.fitted_transform(data.iloc[train],data,cfg)
        geometries.append(z)
        inner[held]=fold
        scales.append(max(float(y[train].std()),1e-8))
        audits.append(dict(inner_fold=fold,training_groups=sorted(set(data.leakage_group_id.iloc[train])),
            validation_groups=sorted(set(data.leakage_group_id.iloc[held])),preprocessor=metadata))
        for candidate,settings in enumerate(b.GRID):
            model=b.estimator(settings).fit(z[train],y[train])
            oof[held,candidate]=model.predict(z[held])
            joblib.dump(model,require_server(model_dir/f'inner_{fold}_candidate_{candidate}.joblib'))
            count+=1
    scores=[b.source_mean(np.abs(y-oof[:,i]),sources) for i in range(len(b.GRID))]
    chosen=int(np.argmin(scores))
    prediction=oof[:,chosen]
    hp,adapter_audit=b.tune_adapters(data,y,prediction,geometries,inner,scales,10,experiment.BUDGETS)
    prior=b.prior_fit(y-prediction,sources)
    z,metadata=b.fitted_transform(data,data,cfg)
    model=b.estimator(b.GRID[chosen]).fit(z,y)
    joblib.dump(model,require_server(model_dir/'full_backbone.joblib'))
    count+=1
    info=dict(label_scale=max(float(y.std()),1e-8),frozen_parameters=dict(model=b.GRID[chosen],
        hyperparameters=hp,prior=prior),preprocessor=metadata,model_selection_scores=scores,
        actual_backbone_fits=count,training_groups=sorted(set(data.leakage_group_id)))
    experiment.save_csv(pd.DataFrame(adapter_audit),directory/f'{task}_adapter_candidates.csv.gz')
    write_json(directory/f'{task}_inner_source_audit.json',audits)
    experiment.save_csv(pd.DataFrame(dict(row_id=data.row_id,inner_fold=inner,base=prediction)),
                        directory/f'{task}_training_oof.csv.gz')
    return info,{i:v for i,v in enumerate(geometries)},inner,prediction


def main():
    require_server()
    parser=argparse.ArgumentParser()
    parser.add_argument('--history-run-name',default='history_v1')
    parser.add_argument('--freeze-run-name',default='full_history_v1')
    args=parser.parse_args()
    history=require_server(ROOT/'runs'/args.history_run_name)
    full=require_server(ROOT/'runs'/args.freeze_run_name)
    if (full/'frozen.json').exists():
        print('A final pipeline is already frozen; use it without retuning.')
        return
    if full.exists() and any(full.iterdir()):
        raise ValueError('Partial full-history run exists; preserve it and review before a fresh version')
    full.mkdir(parents=True,exist_ok=True)
    completion=history/'run_completion.json'
    if not completion.exists() or not json.loads(completion.read_text()).get('full_historical_grid_completed'):
        raise ValueError('Complete the locked 25-part historical grid before final freezing')
    if not (history/'verification/audit.json').exists() or not json.loads((history/'verification/audit.json').read_text()).get('passed'):
        raise ValueError('Historical result audit is required before final freezing')
    history_lock=json.loads((history/'experiment_lock.json').read_text())
    for name,expected in history_lock['config']['code_hashes'].items():
        if sha(ROOT/'code'/name)!=expected:
            raise ValueError('Code changed after historical evaluation; run a fresh version: '+name)
    infos,bases={},{}
    for task in ('ocm_f0','ocm_f1'):
        data,cfg=b.load_task(task)
        info,zs,inner,base=fit_task(data,cfg,task,full)
        infos[task]=info
        bases[task]=base
        if task=='ocm_f1':
            training,geometries,inner_ids=data,zs,inner
    gate,gamma,controls=experiment.tune(training,geometries,inner_ids,bases,infos,full)
    write_json(full/'infos.json',infos)
    experiment.save_csv(training,full/'historical_input.csv.gz')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(full.rglob('*')) if p.is_file()}
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').glob('*.py'))})
    hashes['protocol/analysis_plan.json']=sha(ROOT/'protocol/analysis_plan.json')
    write_json(full/'frozen.json',dict(frozen_utc=utc(),status='frozen_before_new_cohort_evaluation',
        historical_run_identity=history_lock['identity'],historical_run_path=str(history),
        file_hashes=hashes,actual_backbone_fits=sum(i['actual_backbone_fits'] for i in infos.values()),
        new_cohort_files_read_during_fitting=False,
        external_result_selection='Evaluate every locked policy; do not pick a new winner using cohort outcomes'))
    print('Final historical pipeline frozen; new cohorts can now be evaluated.')


if __name__=='__main__':
    main()

"""Read-only scientific-record and release integrity verification (stdlib only)."""
from pathlib import Path
import ast
import csv
import gzip
import hashlib
import json
import math
import struct
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads((ROOT/path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def csv_rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as f:return list(csv.DictReader(f))


def array_header(stream):
    require(stream.read(6)==b'\x93NUMPY','Invalid NPY header')
    version=tuple(stream.read(2))
    length=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
    return ast.literal_eval(stream.read(length).decode('utf-8' if version==(3,0) else 'latin1'))


def check_part(part,predictors):
    completion=json.loads((part/'completed.json').read_text())
    with zipfile.ZipFile(part/'row_predictions.npz') as z:
        with z.open('predictions.npy') as stream:
            h=array_header(stream)
        require(tuple(h['shape'])==(completion['n_prediction_rows'],len(predictors)),'Prediction array shape mismatch: '+str(part))
        require(h['descr']=='<f8','Lossless float64 predictions required')
        with z.open('methods.npy') as stream:
            mh=array_header(stream); raw=stream.read()
        require(mh['descr'].endswith('U32'),'Unexpected predictor ID encoding')
        encoding='utf-32-le' if mh['descr'][0]=='<' else 'utf-32-be'
        actual=[raw[i:i+128].decode(encoding).rstrip('\x00') for i in range(0,len(raw),128)]
        require(actual==predictors,'Prediction column identities changed')
    rows=csv_rows(part/'cells.csv.gz')
    require(len(rows)==completion['n_cells'],'Cell count mismatch')
    require(set(map(int,(r['cell_id'] for r in rows)))==set(range(len(rows))),'Cell IDs incomplete')
    for r in rows:
        for method in predictors:
            require(math.isfinite(float(r['mae__'+method])) and math.isfinite(float(r['mse__'+method])),'Nonfinite actual loss')
    require(completion['n_scalar_predictions']==completion['n_prediction_rows']*len(predictors),'Scalar prediction count mismatch')
    require(set(completion['history_training_groups']).isdisjoint(completion['evaluation_groups']),'Outer group leakage')
    require(completion['new_outer_backbone_fits']==0,'Current backbone count changed')
    return completion


def verify():
    result={}
    previous=load('reference_step9/release_manifest.json')
    for item in previous['files']:
        p=ROOT/'reference_step9'/item['path']
        require(p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Predecessor changed: '+item['path'])
    result['unchanged_predecessor_files']=len(previous['files'])
    oldfreeze=load('reference_step9/runs/full_history_v1/frozen.json')
    for path,digest in oldfreeze['file_hashes'].items():
        require(sha(ROOT/'reference_step9'/path)==digest,'Previous frozen pipeline changed')
    require(oldfreeze['actual_backbone_fits']==56,'Original 56 fits lost')
    result['verified_previous_frozen_files']=len(oldfreeze['file_hashes'])
    lock=load('protocol/pre_execution_lock.json')
    for path,digest in lock['code_hashes'].items():
        require(sha(ROOT/'code'/path)==digest,'Full-grid scientific code lock changed')
    require(sha(ROOT/'protocol/analysis_plan.json')==lock['analysis_plan_sha256'],'Full-grid plan changed')
    plan=load('protocol/analysis_plan.json');predictors=plan['predictors']
    require(len(predictors)==7 and plan['candidate']=='hedged_mae','Locked candidate or predictor family changed')
    frozen=load('runs/full_history_v1/frozen.json')
    for path,digest in frozen['file_hashes'].items():require(sha(ROOT/path)==digest,'Current history freeze changed: '+path)
    require(frozen['cohort_already_opened_in_step9'] and frozen['new_backbone_fits']==0,'Incorrect freeze scope')
    result['scientific_code_and_parameter_locks_passed']=True
    for p in list((ROOT/'code').glob('*.py'))+list((ROOT/'analysis').glob('*.py')):
        ast.parse(p.read_text(),filename=str(p))
    result['current_python_sources_parse']=True
    mathcheck=load('runs/verification/math.json'); require(mathcheck['passed'],'Math verification failed')
    require(mathcheck['monte_carlo_draws']==50000,'Math simulation count changed')
    pilot=load('runs/pilot_v1/verification/audit.json'); require(pilot['passed'],'Stored six-predictor pilot audit failed')
    oldpilot=load('reference_step10_pilot/protocol/pre_execution_lock.json')
    for name,digest in oldpilot['code_hashes'].items():
        require(sha(ROOT/'reference_step10_pilot/code'/name)==digest,'Six-predictor pilot snapshot changed')
    check_part(ROOT/'runs/pilot_v1/parts/ocm_42_0',predictors[:6])
    result['math_and_preserved_six_predictor_pilot_passed']=True
    hc=load('runs/history_v1/run_completion.json')
    require(hc['full_historical_grid_completed'] and hc['parts']==25,'Historical grid incomplete')
    parts=sorted((ROOT/'runs/history_v1/parts').glob('*/completed.json'))
    records=[check_part(p.parent,predictors) for p in parts]
    require(len(records)==25 and {(r['seed'],r['fold']) for r in records}=={(s,f) for s in [42,101,202,303,404] for f in range(5)},'Missing seed/fold')
    ha=load('runs/history_v1/verification/audit.json');ea=load('runs/applications/nishimura_opened_v1/verification/audit.json')
    require(ha['passed'] and ea['passed'],'Actual full audits failed')
    require(ha['scalar_values_reaggregated']==sum(r['n_scalar_predictions'] for r in records),'Historical audit count mismatch')
    epart=check_part(ROOT/'runs/applications/nishimura_opened_v1/parts/cohort',predictors)
    require(ea['scalar_values_reaggregated']==epart['n_scalar_predictions'],'Application audit count mismatch')
    ec=load('runs/applications/nishimura_opened_v1/application_completion.json')
    require(ec['previously_opened_dataset'] and ec['n_publications']==1 and ec['n_experiment_families']==1 and not ec['new_wet_lab_experiments'],'Application independence overstated')
    eroot=ROOT/'runs/applications/nishimura_opened_v1'
    eprimary=csv_rows(eroot/'aggregated/primary_family.csv')
    require(all(not r['family_adjusted_ci_low'] and not r['family_adjusted_ci_high'] for r in eprimary),'One-family population intervals present')
    primary=csv_rows(ROOT/'runs/history_v1/aggregated/primary_family.csv')
    require(len(primary)==5 and all(r['adjusted_support_for_advantage'].lower()=='false' for r in primary),'Primary result misreported')
    fact=load('reports/step10_measured_facts.json')
    require(fact['historical_cells']==sum(r['n_cells'] for r in records)==82350,'History facts mismatch')
    require(fact['application_cells']==epart['n_cells']==1830,'Application facts mismatch')
    require(fact['total_scalar_predictions']==ha['scalar_values_reaggregated']+ea['scalar_values_reaggregated']==9795380,'Total predictions mismatch')
    partition=load('reports/step10_catalyst_partition_audit.json'); require(partition['passed'],'Catalyst audit failed')
    held=partition['scenarios']['catalyst_holdout']
    require(len(held['query_row_ids'])==435 and len(held['pool_row_ids'])==1304,'Catalyst row counts changed')
    require(len(held['query_catalyst_ids'])==29 and len(held['pool_catalyst_ids'])==87,'Catalyst counts changed')
    require(set(held['query_row_ids']).isdisjoint(held['pool_row_ids']) and set(held['query_catalyst_ids']).isdisjoint(held['pool_catalyst_ids']),'Catalyst identity leakage')
    result.update(historical_cells=82350,application_cells=1830,actual_scalar_predictions=9795380,
          all_stored_actual_audits_passed=True,new_backbone_refits=0,new_independent_families=0,
          primary_candidate_superiority_supported=False)
    layout=load('reports/step10_pdf_layout_checks.json');visual=load('reports/step10_pdf_visual_review.json')
    require(visual['passed'] and len(layout)==7,'PDF QA incomplete')
    for path,record in layout.items():
        require(not record['out_of_page_spans'] and not record['out_of_page_images'],'PDF clipping')
        require(record['sha256']==sha(ROOT/path),'PDF changed after QA')
        require(visual['files'][path]['visually_reviewed_pages']==list(range(1,record['pages']+1)),'Missing page review')
        require(visual['files'][path]['sha256']==record['sha256'],'PDF visual review hash changed')
    result['visually_reviewed_pdf_pages']=sum(r['pages'] for r in layout.values())
    require(result['visually_reviewed_pdf_pages']==76,'Final PDF count changed')
    for name in ['00_第10步进度说明.txt','01_现有成果与路径索引.txt','AGENT_START_HERE.txt','README.txt']:
        require((ROOT/name).is_file(),'Missing handoff file')
    require('可能有没想到的地方，需要你自己检查一下。' in (ROOT/'AGENT_START_HERE.txt').read_text(),'Required continuation phrase missing')
    manifest=ROOT/'release_manifest.json'
    if manifest.exists():
        entries=json.loads(manifest.read_text())['files']
        delivered={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=manifest}
        require(delivered=={r['path'] for r in entries},'Manifest file inventory mismatch')
        for item in entries:
            p=ROOT/item['path']
            require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Delivered checksum changed: '+item['path'])
        result['current_manifest_files_verified']=len(entries)
    else:result['current_manifest_status']='created after pre-packaging verification'
    result['passed']=True
    return result


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))

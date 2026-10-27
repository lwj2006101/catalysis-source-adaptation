"""Read-only verification of the complete Step11 release, not a model rerun."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def predecessor_files():
    m=read('reference_step10/release_manifest.json')
    return [ROOT/'reference_step10'/x['path'] for x in m['files']]+[ROOT/'reference_step10/release_manifest.json']


def release_files():
    # Deliver the exact complete predecessor manifest, excluding only new caches.
    paths=predecessor_files()
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        if rel.parts[0] in {'reference_step10','tmp','_local_input'}:continue
        if '__pycache__' in rel.parts or path.suffix in {'.pyc','.pyo'}:continue
        if rel.parts[0]=='manuscripts' and path.suffix in {'.aux','.log','.out','.toc','.synctex'}:continue
        if rel.as_posix()=='release_manifest.json':continue
        paths.append(path)
    return sorted(paths,key=lambda x:x.relative_to(ROOT).as_posix())


def verify(require_manifest=True):
    needed=['AGENT_START_HERE.txt','README.txt','requirements.txt',
            'manuscripts/manuscript_en.pdf','manuscripts/supplementary_en.pdf',
            'manuscripts/第11步_成果与投稿说明.pdf','submission/cover_letter.pdf',
            'submission/highlights.txt','submission/graphical_abstract.pdf',
            'submission/graphical_abstract.png','submission/投稿操作说明.txt',
            'analysis/finalize_author_info.py','analysis/build_submission.py']
    for name in needed:assert (ROOT/name).is_file(),f'Missing {name}'
    parent=read('reference_step10/release_manifest.json')
    assert len(parent['files'])==3290
    for record in parent['files']:
        path=ROOT/'reference_step10'/record['path']
        assert path.stat().st_size==record['bytes'],f'Predecessor size changed: {record["path"]}'
        assert sha(path)==record['sha256'],f'Predecessor changed: {record["path"]}'
    for path,digest in read('protocol/execution_lock.json')['files'].items():
        assert sha(ROOT/path)==digest,f'Execution-lock mismatch: {path}'
    hist=read('runs/operator_history_v1/verification/audit.json')
    opened=read('runs/operator_opened_v1/verification/audit.json')
    facts=read('reports/step11_measured_facts.json')
    assert hist==facts['historical_audit'] and opened==facts['opened_audit']
    assert hist['passed'] and opened['passed']
    assert hist['parts']==25 and hist['cells']==42525 and opened['cells']==5600
    parts=sorted((ROOT/'runs/operator_history_v1/parts').glob('ocm_*'))
    assert len(parts)==25
    completions=[json.loads((p/'completed.json').read_text()) for p in parts]
    for key in ['cells','query_records','scalar_predictions']:
        assert sum(x[key] for x in completions)==hist[key],f'Historical completion mismatch: {key}'
    for part in parts:
        for name in ['row_predictions.npz','geometry.npz','support_ledger.csv.gz','evaluated_input.csv.gz','parameters.json','epsilon_tuning.csv.gz']:
            assert (part/name).is_file(),f'Missing historical run evidence: {part.name}/{name}'
    c=read('runs/operator_opened_v1/parts/cohort/completed.json')
    assert c['n_partitions']==20 and c['n_experiment_families']==1 and c['previously_opened']
    for key in ['cells','query_records','scalar_predictions']:assert c[key]==opened[key]
    for name in ['row_predictions.npz','geometry.npz','support_ledger.csv.gz','evaluated_input.csv.gz','partition_lock.json']:
        assert (ROOT/'runs/operator_opened_v1/parts/cohort'/name).is_file()
    assert hist['scalar_predictions']+opened['scalar_predictions']==10948000==facts['actual_new_scalar_predictions']
    assert hist['sklearn_reconstructed_values']+opened['sklearn_reconstructed_values']==2737000
    assert facts['new_independent_measured_families']==0
    math=read('reports/operator_math_checks.json');assert math['passed'] and math['spectral_ball_upper_bound_tight']
    smoke=read('reports/portable_smoke_reproduction.json')
    assert smoke['passed'] and smoke['actual_fresh_predictions']==54180 and smoke['all_saved_arrays_identical']
    assert smoke['cached_completion_not_reused']
    qa=read('reports/pdf_quality_assurance.json');assert qa['passed'] and qa['all_pages_rendered_and_visually_reviewed']
    for record in qa['documents']:
        assert sha(ROOT/record['path'])==record['sha256'],f'PDF changed since visual QA: {record["path"]}'
    highlights=(ROOT/'submission/highlights.txt').read_text().splitlines()
    assert len(highlights)==5 and all(len(x)<=85 for x in highlights)
    files=release_files()
    manifest_status='not yet created'
    if require_manifest:
        m=read('release_manifest.json')
        actual={p.relative_to(ROOT).as_posix() for p in files}
        assert actual=={x['path'] for x in m['files']},'Delivered file list differs from manifest'
        for record in m['files']:
            path=ROOT/record['path']
            assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],f'Release checksum mismatch: {record["path"]}'
        manifest_status='every delivered file verified'
    status=read('submission/submission_readiness.json')
    return dict(passed=True,predecessor_original_files_unchanged=len(parent['files']),
                predecessor_manifest_unchanged=sha(ROOT/'reference_step10/release_manifest.json')==read('protocol/step11_analysis_plan.json')['parent_manifest_sha256'],
                code_execution_lock_verified=True,historical_parts=25,within_family_partitions=20,
                actual_scalar_predictions=10948000,independently_reconstructed_rbf_values=2737000,
                math_and_fresh_portability_passed=True,pdf_visual_qa_passed=True,
                delivered_files_excluding_root_manifest=len(files),manifest=manifest_status,
                scientific_manuscript_complete=status['scientific_manuscript_complete'],
                author_identity_filled=status['author_identity_filled'],article_submitted=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2,ensure_ascii=False))

"""Package the complete release, preserving all predecessor manifest files."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean():
    old=json.loads((ROOT/'reference_step9/release_manifest.json').read_text())
    preserved={'reference_step9/'+r['path'] for r in old['files']}|{'reference_step9/release_manifest.json'}
    removed=[]
    for p in list(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT))
        transient=('__pycache__' in p.parts or p.suffix=='.pyc' or
                   p.name.startswith('fontlist-v') or
                   (p.parent==ROOT/'manuscripts' and p.suffix in ['.aux','.out','.log']))
        if transient and rel not in preserved:
            p.unlink();removed.append(rel)
    for p in sorted(ROOT.rglob('__pycache__'),reverse=True):
        if p.is_dir() and not any(p.iterdir()):p.rmdir()
    return removed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clean-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT.parent/'第10步_预测器匹配选样与论文完整包.zip')
    args=parser.parse_args()
    removed=clean()
    if args.clean_only:
        print(json.dumps(dict(removed_transient_files=len(removed)),ensure_ascii=False));return
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='release_manifest.json':
            entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        elif p.is_file() and p!=ROOT/'release_manifest.json':
            # The immutable predecessor manifest is part of this release's payload.
            entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(version='step10-executed-complete-release-1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='full_history_and_opened_catalyst_stress_executed_and_audited',package_root='step10',
        predecessor_archive_sha256='1db4a170fdc35ca054765841ef8f060c92e4c86cf0e18fbebdcaeecfabacc7bb',
        historical_parts=25,acquisition_policies=12,predictors=7,scalar_query_predictions=9795380,
        new_backbone_refits=0,new_independent_publications=0,opened_catalyst_holdout=True,
        primary_candidate_superiority_supported=False,prospective_wet_lab=False,
        verification_entry='analysis/verify_release.py',
        hash_scope='All delivered regular files except this root manifest; new transient caches excluded.',files=entries)
    (ROOT/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for item in entries:z.write(ROOT/item['path'],'step10/'+item['path'])
        z.write(ROOT/'release_manifest.json','step10/release_manifest.json')
    with zipfile.ZipFile(args.output) as z:
        error=z.testzip()
        if error:raise ValueError('ZIP CRC failed: '+error)
        actual=set(z.namelist()); expected={'step10/'+r['path'] for r in entries}|{'step10/release_manifest.json'}
        if actual!=expected:raise ValueError('ZIP inventory mismatch')
        for item in entries:
            h=hashlib.sha256()
            with z.open('step10/'+item['path']) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=item['sha256']:raise ValueError('ZIP content changed: '+item['path'])
    print(json.dumps(dict(status='complete',zip=str(args.output.resolve()),bytes=args.output.stat().st_size,
        sha256=sha(args.output),manifest_files=len(entries),zip_crc_passed=True,
        all_zip_file_hashes_verified=True,removed_transient_files=len(removed)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

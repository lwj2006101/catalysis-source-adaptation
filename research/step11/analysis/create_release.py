"""Create and independently verify one complete numbered release ZIP."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,zipfile,hashlib
from verify_release import ROOT,sha,release_files,verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT.parents[1]/'output/第11步_可投稿论文与完整复现包.zip'))
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if out.is_relative_to(ROOT):raise ValueError('Save the ZIP outside the delivered project tree')
    parent=json.loads((ROOT/'reference_step10/release_manifest.json').read_text())
    preflight=verify(require_manifest=False)
    assert preflight['predecessor_manifest_unchanged']
    (ROOT/'reports/predecessor_preservation.json').write_text(json.dumps(
        dict(passed=True,original_manifested_files=len(parent['files']),all_original_hashes_unchanged=True,
             original_manifest_sha256=sha(ROOT/'reference_step10/release_manifest.json')),
        indent=2)+'\n')
    (ROOT/'reports/release_preflight.json').write_text(json.dumps(preflight,indent=2,ensure_ascii=False)+'\n')
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in release_files()]
    manifest=dict(version='step11-complete-scientific-manuscript-1',package_root='step11',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_manuscript_complete=True,article_submitted=False,
                  author_metadata_supplied=False,quarter1_guarantee=False,
                  predecessor_manifest_sha256=sha(ROOT/'reference_step10/release_manifest.json'),
                  actual_new_scalar_predictions=10948000,new_independent_measured_families=0,
                  hash_scope='Every delivered regular file except this root manifest; includes the complete unchanged predecessor.',
                  exclusions=['new temporary directories','new unaccepted/restricted raw downloads','new Python and LaTeX caches'],
                  files=entries)
    mf=ROOT/'release_manifest.json';mf.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    verified=verify()
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for record in entries:z.write(ROOT/record['path'],'step11/'+record['path'])
        z.write(mf,'step11/release_manifest.json')
    with zipfile.ZipFile(out) as z:
        assert len(z.namelist())==len(entries)+1
        for record in entries:
            digest=hashlib.sha256()
            with z.open('step11/'+record['path']) as f:
                for block in iter(lambda:f.read(2**20),b''):digest.update(block)
            assert digest.hexdigest()==record['sha256'],f'ZIP content mismatch: {record["path"]}'
        assert z.read('step11/release_manifest.json')==mf.read_bytes()
    receipt=dict(passed=True,zip_path=str(out),bytes=out.stat().st_size,sha256=sha(out),
                 members=len(entries)+1,all_zip_member_hashes_verified=True,release_verification=verified)
    out.with_suffix('.verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

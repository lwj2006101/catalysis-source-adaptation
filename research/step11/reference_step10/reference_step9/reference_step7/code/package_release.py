"""Create a complete, checksummed numbered release; never package caches/fonts."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]


def included(path):
    return path.is_file() and not any(n in ['__pycache__','build_fonts'] for n in path.parts) and path.suffix not in ['.pyc','.aux','.out','.log']


def main():
    files=[p for p in sorted(ROOT.rglob('*')) if included(p) and p.name!='release_manifest.json'];entries=[]
    for p in files:
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha))
    (ROOT/'release_manifest.json').write_text(json.dumps(dict(version='step7-release-1',files=entries),ensure_ascii=False,indent=2)+'\n')
    destination=ROOT.parent/'第7步_跨任务选样验证与论文更新包.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'release_manifest.json']:z.write(p,'step7/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None;count=len(z.infolist())
    with destination.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,files=count,sha256=sha),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

"""Portable verification and genuine fresh refits for this numbered stage."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]


def run(root,script,args=(),env=None):subprocess.run([sys.executable,str(root/'code'/script),*args],check=True,env=env)


def verify_manifest():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for entry in manifest['files']:
        path=ROOT/entry['path'];assert path.is_file() and path.stat().st_size==entry['bytes'],entry['path']
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==entry['sha256'],entry['path']
    print('Manifest passed',len(manifest['files']),'files',flush=True)


def fresh_copy(path):
    path=Path(path).resolve()
    if path==ROOT or ROOT in path.parents:raise ValueError('fresh-dir must be outside the extracted release')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):raise ValueError('fresh-dir must be absent or empty')
    def ignore(src,names):
        skip={n for n in names if n in ['__pycache__','build_fonts'] or n.endswith(('.pyc','.aux','.log','.out'))}
        if Path(src).resolve()==ROOT:skip.update(set(names)&{'results','verification','release_manifest.json'})
        if Path(src).resolve()==ROOT/'protocol':skip.update(set(names)&{'cross_task_lock.json'})
        return skip
    shutil.copytree(ROOT,path,ignore=ignore,dirs_exist_ok=True);(path/'verification').mkdir(exist_ok=True);return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['verify','smoke','full'],required=True);parser.add_argument('--fresh-dir');parser.add_argument('--workers',type=int,default=4);parser.add_argument('--documents',action='store_true');args=parser.parse_args()
    if args.mode=='verify':
        verify_manifest()
        with tempfile.TemporaryDirectory(prefix='cross_task_verify_') as temp:
            env=dict(os.environ,CROSS_TASK_VERIFY_REPORT_DIR=temp);run(ROOT,'verify_acquisition_math.py',env=env);run(ROOT,'audit_cross_task.py',env=env);run(ROOT,'audit_ocm_geometry.py',env=env)
        return
    if not args.fresh_dir:parser.error('Fresh refits require --fresh-dir; completed caches do not count as reproduction')
    fresh=fresh_copy(args.fresh_dir)
    if args.mode=='smoke':run(fresh,'run_cross_task.py',['--workers','1','--seeds','42','--folds','0','--endpoints','ocm_f1'])
    else:run(fresh,'run_cross_task.py',['--workers',str(args.workers)])
    run(fresh,'aggregate_cross_task.py');run(fresh,'verify_acquisition_math.py');run(fresh,'audit_cross_task.py')
    if args.mode=='full':run(fresh,'diagnose_ocm_geometry.py');run(fresh,'audit_ocm_geometry.py')
    if args.documents:
        if args.mode!='full':parser.error('--documents requires --mode full')
        run(fresh,'build_cross_task_artifacts.py')
    print('Fresh refit completed',fresh,flush=True)


if __name__=='__main__':main()

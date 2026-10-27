"""Portable verification, a genuinely fresh compute run, and PDF rebuilding."""
from pathlib import Path
import argparse,subprocess,sys,shutil,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]


def command(args,cwd=ROOT):
    subprocess.run(args,cwd=cwd,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','paper','compute','smoke'])
    p.add_argument('--destination',default='tmp/fresh_reproduction');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.action=='check':
        command([sys.executable,'analysis/verify_release.py'])
    elif a.action=='paper':
        command([sys.executable,'analysis/build_results.py'])
        for file in ['manuscript_en.tex','supplementary_en.tex']:
            for _ in range(2):command(['pdflatex','-interaction=nonstopmode','-halt-on-error',file],ROOT/'manuscripts')
    else:
        dest=(ROOT/a.destination).resolve()
        if not dest.is_relative_to(ROOT) or dest==ROOT:raise ValueError('Choose a new directory under the project root')
        if dest.exists() and any(dest.iterdir()):raise ValueError('Choose an empty destination to perform a fresh run')
        dest.mkdir(parents=True,exist_ok=True)
        shutil.copytree(ROOT/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'protocol',dest/'protocol')
        (dest/'analysis').mkdir()
        for name in ['audit_extension.py','verify_operator_math.py']:
            shutil.copy2(ROOT/'analysis'/name,dest/'analysis'/name)
        (dest/'reference_step10').symlink_to(ROOT/'reference_step10',target_is_directory=True)
        args=[sys.executable,'code/run_extension.py','history','--workers',str(a.workers)]
        if a.action=='smoke':args+=['--smoke']
        command(args,dest)
        if a.action=='compute':
            command([sys.executable,'analysis/audit_extension.py','operator_history_v1'],dest)
            command([sys.executable,'code/run_extension.py','opened'],dest)
            command([sys.executable,'analysis/audit_extension.py','operator_opened_v1'],dest)
        else:
            name='runs/operator_history_v1/parts/ocm_42_0'
            x=np.load(ROOT/name/'row_predictions.npz');y=np.load(dest/name/'row_predictions.npz')
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
            output=dict(passed=True,fresh_directory=str(dest.relative_to(ROOT)),
                        actual_fresh_predictions=int(y['predictions'].size),all_saved_arrays_identical=True,
                        cached_completion_not_reused=True)
            (ROOT/'reports/portable_smoke_reproduction.json').write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps(output,indent=2))


if __name__=='__main__':main()

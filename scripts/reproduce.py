"""Launch original extension code in a fresh directory without overwriting evidence."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'research/step11'

def run(args, cwd):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','smoke','compute'])
    parser.add_argument('--destination',default='work/fresh')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.action == 'check':
        run([str(ROOT/'scripts/verify_release.py')],ROOT)
        return
    import numpy as np
    dest = (ROOT/args.destination).resolve()
    if not dest.is_relative_to(ROOT/'work') or dest == ROOT/'work':
        parser.error('Choose a new subdirectory under work/.')
    if dest.exists() and any(dest.iterdir()):
        parser.error('Destination must be empty, to avoid reusing cached predictions.')
    if args.workers < 1:
        parser.error('workers must be positive')
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['code','protocol']:
        shutil.copytree(ARCHIVE/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
    (dest/'analysis').mkdir()
    for name in ['audit_extension.py','verify_operator_math.py']:
        shutil.copyfile(ARCHIVE/'analysis'/name,dest/'analysis'/name)
    (dest/'reference_step10').symlink_to(ARCHIVE/'reference_step10',target_is_directory=True)
    command = ['code/run_extension.py','history','--workers',str(args.workers)]
    if args.action == 'smoke':
        command.append('--smoke')
    run(command,dest)
    if args.action == 'compute':
        run(['analysis/audit_extension.py','operator_history_v1'],dest)
        run(['code/run_extension.py','opened'],dest)
        run(['analysis/audit_extension.py','operator_opened_v1'],dest)
        result = {'completed':True,'fresh_extension_run':True,'all_historical_analyses_rerun':False}
    else:
        part = Path('runs/operator_history_v1/parts/ocm_42_0/row_predictions.npz')
        with np.load(ARCHIVE/part,allow_pickle=False) as archived, np.load(dest/part,allow_pickle=False) as fresh:
            assert archived.files == fresh.files
            for key in archived.files:
                np.testing.assert_array_equal(archived[key],fresh[key])
            n = int(fresh['predictions'].size)
        result = {'passed':True,'fresh_scalar_predictions':n,'all_arrays_identical':True,
                  'backbones_retrained':False,'cached_completion_reused':False}
    (dest/'public_reproduction_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()

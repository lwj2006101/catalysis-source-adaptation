"""Compile a working copy of the current manuscript, preserving release evidence."""
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language',choices=['cn','en'],default='cn')
    a = p.parse_args()
    if not shutil.which('xelatex'):
        p.error('Install XeLaTeX / TeX Live first.')
    dest = ROOT/'work'/('paper-'+a.language)
    shutil.copytree(ROOT/'paper/source',dest,dirs_exist_ok=True)
    for _ in range(2):
        subprocess.run(['xelatex','-halt-on-error','-interaction=nonstopmode',f'manuscript_{a.language}.tex'],cwd=dest,check=True)
    print(dest/f'manuscript_{a.language}.pdf')

if __name__ == '__main__':
    main()

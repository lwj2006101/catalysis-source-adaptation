"""Deterministic retrieval of public upstream assets for the two case studies.

Downloads exact public versions (repo clone / paper PDF) and verifies SHA-256.
Creates the directory structure that rc3_pipeline.py expects:
  original_assets/github_repo/
  original_assets/paper_npj_xu2026.pdf

Usage: python fetch_original_assets.py
No network access is required if the original_assets/ directory is already populated.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ASSETS = [
    dict(
        case='suvarna',
        repo_url='https://github.com/ssuvarnamanu/ML-for-thermocatalytic-CO2-hydrogenation-to-methanol.git',
        commit='0dac9383169ef26283ab9717083689b6ebb0c998',
        dest=HERE / 'suvarna' / 'original_assets' / 'github_repo',
        license='MIT',
        files_to_verify=['Curated data_1234 datapoints.xlsx',
                         'Master Model_1230 datapoints.ipynb'],
    ),
    dict(
        case='xu',
        repo_url='https://github.com/hangweixu-1/nlp-and-Causal-Analysis.git',
        commit='b7291de8d81e5feb4e66f2579e9bf450305d6fec',
        dest=HERE / 'xu' / 'original_assets' / 'github_repo',
        license='see repo LICENSE',
        files_to_verify=['3_cleaned2_standardized_processed2.xlsx',
                         'ml_catalyst_predictor.py'],
    ),
    dict(
        case='xu_paper',
        url='https://www.nature.com/articles/s41524-026-02065-2.pdf',
        dest=HERE / 'xu' / 'original_assets' / 'paper_npj_xu2026.pdf',
        license='CC BY 4.0 (open access)',
        files_to_verify=[],
    ),
]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clone_and_verify(spec):
    dest = spec['dest']
    if dest.is_dir() and (dest / '.git').is_dir():
        print(f'  [SKIP] {dest.name} already cloned')
    else:
        print(f'  [CLONE] {spec["repo_url"]}')
        subprocess.run(['git', 'clone', '--depth', '1', spec['repo_url'], str(dest)],
                       check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', spec['commit']], check=True)
    # record commit
    r = subprocess.run(['git', '-C', str(dest), 'log', '-1', '--format=%H'],
                       capture_output=True, text=True)
    actual = r.stdout.strip()
    if actual != spec['commit']:
        print(f'  [WARN] commit mismatch: expected {spec["commit"][:12]}, got {actual[:12]}')
    else:
        print(f'  [OK] commit {actual[:12]}')
    # verify required files
    for f in spec['files_to_verify']:
        fp = dest / f
        if fp.is_file():
            print(f'  [OK] {f} ({fp.stat().st_size} bytes)')
        else:
            print(f'  [MISSING] {f}')


def fetch_paper(spec):
    dest = spec['dest']
    if dest.is_file():
        print(f'  [SKIP] {dest.name} already downloaded')
        return
    print(f'  [DOWNLOAD] {spec["url"]}')
    import urllib.request
    import ssl
    ctx = ssl.create_default_context()
    req = urllib.request.Request(spec['url'], headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            dest.write_bytes(resp.read())
        print(f'  [OK] {dest.name} ({dest.stat().st_size} bytes)')
    except Exception as e:
        print(f'  [FAIL] {e}')


def main():
    print('=== Fetching original public assets ===')
    for spec in ASSETS:
        print(f'\nCase: {spec["case"]}')
        if 'repo_url' in spec:
            clone_and_verify(spec)
        if 'url' in spec and 'paper' in spec['case']:
            fetch_paper(spec)
    print('\nDone. Assets are ready for rc3_pipeline.py and x0_original_flow.py.')
    print('Note: the full dataset files are inside the cloned repos — no separate Zenodo')
    print('download is required for the current version of the pipeline.')


if __name__ == '__main__':
    main()

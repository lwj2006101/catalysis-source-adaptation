"""Retrieve optional raw sources at recorded URLs and verify original SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'research/step11/reference_step10/reference_step9/data/raw'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true',help='Show assets without downloading.')
    args = parser.parse_args()
    for name in ['nishimura2023_cads.csv','jp4c07397_si_001.pdf']:
        receipt = json.loads((RAW/(name+'.receipt.json')).read_text())
        url = receipt['source']['verified_download_url']
        print(json.dumps({'name':name,'source':receipt['source']['landing_url'],'url':url,'sha256':receipt['sha256']},ensure_ascii=False))
        if args.list:
            continue
        dest = RAW/name
        if dest.exists():
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == receipt['sha256'], 'Existing asset differs from the archive.'
            continue
        request = urllib.request.Request(url,headers={'User-Agent':'catalysis-source-adaptation/0.1.0'})
        fd, temporary = tempfile.mkstemp(dir=RAW,suffix='.download')
        try:
            with os.fdopen(fd,'wb') as output,urllib.request.urlopen(request,timeout=60) as response:
                while block := response.read(2**20):
                    output.write(block)
            assert hashlib.sha256(Path(temporary).read_bytes()).hexdigest() == receipt['sha256'], 'Source bytes changed; keep the archive hash and inspect the new version.'
            os.replace(temporary,dest)
        finally:
            Path(temporary).unlink(missing_ok=True)

if __name__ == '__main__':
    main()

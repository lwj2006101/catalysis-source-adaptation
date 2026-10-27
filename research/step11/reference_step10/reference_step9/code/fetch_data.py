"""Retrieve publicly advertised source assets in the authorized project runtime.

Downloads are candidate assets, never proof of new independent experiments.
An original publication can contain literature, HTS, proposed and measured rows.
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from server_guard import ROOT, require_server, sha, utc, write_json


def main():
    require_server()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT/'protocol/data_sources.json').read_text())
    item = next((v for v in registry['sources'] if v['key']==args.source), None)
    if item is None:
        raise ValueError('Unknown source key')
    if not item.get('verified_download_url'):
        raise ValueError('No verified file link: use the original landing page to resolve it first')
    dest = require_server(ROOT/'data/raw'/item['local_name'])
    receipt = dest.with_suffix(dest.suffix+'.receipt.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if sha(dest) != old['sha256']:
            raise ValueError('Previously retrieved asset changed')
        print('Existing verified candidate asset:',dest.name)
        return
    if dest.exists():
        raise ValueError('Unreceipted asset exists: preserve it and use a new source version')
    request = urllib.request.Request(item['verified_download_url'],headers={'User-Agent':'CatalystResearch/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        blob=response.read()
        content_type=response.headers.get('Content-Type','')
        final_url=response.url
    if not blob or (item['local_name'].endswith('.pdf') and not blob.startswith(b'%PDF')):
        raise ValueError('Response is not the requested scientific asset')
    if item['local_name'].endswith('.csv') and b'<html' in blob[:2000].lower():
        raise ValueError('HTML returned instead of CSV; possible sign-in or error page')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(blob)
    write_json(receipt,dict(source=item,downloaded_utc=utc(),sha256=sha(dest),
        bytes=len(blob),content_type=content_type,final_url=final_url,
        status='raw candidate only; original measurement provenance and overlap not yet audited'))
    print('Retrieved raw candidate asset:',dest.name,len(blob),'bytes')


if __name__=='__main__':
    main()

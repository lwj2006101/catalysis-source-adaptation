"""Offline integrity verification; accounts explicitly for public-release edits."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT/item['path']
        assert path.is_file(), f"Missing: {item['path']}"
        assert path.stat().st_size == item['bytes'], f"Size mismatch: {item['path']}"
        assert sha(path) == item['sha256'], f"Hash mismatch: {item['path']}"
    edits = json.loads((ROOT/'docs/release_changes.json').read_text())
    changes = {x['path']: x for x in edits['changes']}
    omissions = {x['path']: x for x in edits['omissions']}
    historic = json.loads((ROOT/'research/step11/release_manifest.json').read_text())
    unchanged = 0
    for item in historic['files']:
        name = 'research/step11/' + item['path']
        path = ROOT/name
        if name in omissions:
            assert omissions[name]['original_sha256'] == item['sha256']
            if path.exists():
                assert sha(path) == item['sha256'], f'Optional restored asset differs: {name}'
        elif name in changes:
            assert changes[name]['original_sha256'] == item['sha256']
            assert sha(path) == changes[name]['public_sha256']
        else:
            assert sha(path) == item['sha256'], f'Undeclared archived change: {name}'
            unchanged += 1
    print(json.dumps({'passed': True, 'public_files_checked': len(manifest['files']),
                      'unchanged_original_files': unchanged, 'declared_metadata_redactions': len(changes),
                      'declared_optional_raw_omissions': len(omissions)}, indent=2))

if __name__ == '__main__':
    main()

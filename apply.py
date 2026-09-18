"""Apply only to the pinned source, then verify the tested result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/sgl-workspace/sglang'))
    parser.add_argument('--profile', choices=('cached', 'long', 'independent', 'dspark'), default='cached')
    parser.add_argument('--check', action='store_true', help='Verify an already patched tree.')
    args = parser.parse_args()
    profile = json.loads((ROOT / 'versions.json').read_text())['profiles'][args.profile]
    files = profile['files']
    if all(digest(args.source / name) == hashes['after'] for name, hashes in files.items()):
        print(f'Verified {args.profile}: {len(files)} patched source files.')
        return
    if args.check:
        raise SystemExit('Patched source hashes do not match.')
    for name, hashes in files.items():
        if digest(args.source / name) != hashes['before']:
            raise SystemExit(f'Upstream source mismatch: {name}. Use the pinned image.')
    for name, hashes in files.items():
        if hashes['before'] is not None:
            saved = ROOT / 'upstream' / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source / name, saved)
    subprocess.run(
        ['patch', '--batch', '--forward', '--fuzz=0', '-p1'],
        input=(ROOT / profile['patch']).read_bytes(),
        cwd=args.source,
        check=True,
    )
    for name, hashes in files.items():
        if digest(args.source / name) != hashes['after']:
            raise SystemExit(f'Patched source mismatch: {name}')
    print(f'Applied and verified {args.profile}: {len(files)} source files.')


if __name__ == '__main__':
    main()

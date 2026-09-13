"""Download the pinned checkpoint and its reference message encoder."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default='/weights')
    parser.add_argument('--metadata-only', action='store_true', help='Fetch tokenizer, encoder, config, and license without weights.')
    args = parser.parse_args()
    versions = json.loads(Path(__file__).with_name('versions.json').read_text())
    snapshot_download(
        repo_id=versions['model'],
        revision=versions['model_revision'],
        local_dir=args.destination,
        max_workers=8,
        allow_patterns=['*.json', '*.py', 'LICENSE'] + ([] if args.metadata_only else ['model-*.safetensors']),
    )


if __name__ == '__main__':
    main()

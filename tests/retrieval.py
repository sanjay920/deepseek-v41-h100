"""Check exact-length prompts with passcodes near 10%, 50%, and 90%."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer


def build_prompt(tokens, model_dir):
    spec = importlib.util.spec_from_file_location('reference_encoding', model_dir / 'encoding/encoding.py')
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    rng = random.Random(20260912 + tokens)
    records = [
        f'Record {i}: warehouse {rng.randrange(100,999)} received {rng.randrange(100,999)} units of item {rng.randrange(10000,99999)}.\n'
        for i in range(tokens // 10 + 100)
    ]
    filler = tokenizer.encode(''.join(records), add_special_tokens=False).ids
    expected = {key: f'{rng.getrandbits(48):012x}' for key in ('alpha', 'beta', 'gamma')}
    marker = '__DSV41_LONG_CONTEXT_BODY__'
    prompt = (
        'Read the records below. Three special audit entries contain passcodes. '
        'Remember those exact passcodes.\n' + marker + '\n'
        'Return only a JSON object mapping alpha, beta, and gamma to their '
        'special audit passcodes. Do not include any other text.'
    )
    encoded = encoder.encode_messages([{'role': 'user', 'content': prompt}], thinking_mode='chat')
    before, after = encoded.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False).ids
    suffix = tokenizer.encode(after, add_special_tokens=False).ids
    needles = [
        tokenizer.encode(f'\nSPECIAL AUDIT ENTRY: {key} passcode is {value}.\n', add_special_tokens=False).ids
        for key, value in expected.items()
    ]
    filler_count = tokens - len(prefix) - len(suffix) - sum(map(len, needles))
    if not 0 < filler_count <= len(filler):
        raise ValueError('Token budget is too small for the fixture.')
    ids, positions = list(prefix), []
    previous = 0
    for cut, needle in zip([filler_count // 10, filler_count // 2, filler_count * 9 // 10], needles):
        ids.extend(filler[previous:cut])
        positions.append(len(ids))
        ids.extend(needle)
        previous = cut
    ids.extend(filler[previous:filler_count])
    ids.extend(suffix)
    assert len(ids) == tokens
    return tokenizer, ids, expected, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--model-dir', type=Path, default=Path('/model'))
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--dry-run', action='store_true', help='Build and hash the fixture without inference.')
    args = parser.parse_args()
    tokenizer, ids, expected, positions = build_prompt(args.tokens, args.model_dir)
    result = {
        'input_tokens': len(ids), 'needle_positions': positions, 'expected': expected,
        'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    body = {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 128}, 'stream': True}
    req = urllib.request.Request(
        args.url.rstrip('/') + '/generate', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    started = time.perf_counter()
    first = None
    first_count = 0
    text, meta, done = '', {}, False
    try:
        with urllib.request.urlopen(req, timeout=7200) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    done = True
                    break
                item = json.loads(payload)
                if 'error' in item:
                    raise RuntimeError(item['error'])
                text, meta = item.get('text', text), item.get('meta_info', meta)
                if first is None and text:
                    first = time.perf_counter()
                    first_count = len(tokenizer.encode(text, add_special_tokens=False).ids)
        ended = time.perf_counter()
        try:
            actual = json.loads(text.strip())
        except ValueError:
            actual = None
        count = len(tokenizer.encode(text, add_special_tokens=False).ids)
        result.update(
            passed=done and actual == expected and meta.get('prompt_tokens') == len(ids),
            text=text, output_tokens=count, cached_tokens=meta.get('cached_tokens', 0),
            total_seconds=ended-started, ttft_seconds=None if first is None else first-started,
            decode_tokens_per_second=None if first is None else (count-first_count)/(ended-first),
        )
    except Exception as exc:
        result.update(passed=False, error=str(exc))
        if isinstance(exc, urllib.error.HTTPError):
            result['error_body'] = exc.read().decode()
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

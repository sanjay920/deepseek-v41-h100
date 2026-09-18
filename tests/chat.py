"""Measure real streamed Chat Completions over current independent histories."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--output-tokens', type=int, default=512)
p.add_argument('--output-dir', type=Path, default=Path('/runs'))
a = p.parse_args()
if Path(a.name).name != a.name or a.name in ('.', '..'):
    p.error('--name must be a filename, not a path')
folder = a.output_dir
folder.mkdir(parents=True, exist_ok=True)
history_dir = folder / 'independent'
history_dir.mkdir(parents=True, exist_ok=True)
out = folder / f'{a.name}.json'
assert not out.exists(), 'Preserve earlier receipts.'
prior = json.loads(a.source.read_text())
assert prior['passed'] and prior['args']['clients'] == 30
assert prior.get('history_includes_wave'), 'Require persisted current histories.'
histories = [json.loads(Path(f).read_text()) for f in prior['history_files']]
assert len(histories) == 30
result = {'passed': False, 'source': str(a.source), 'url': a.url, 'rows': [],
          'history_files': [], 'phase': 'cache_probe',
          'note': 'Loopback Chat Completions, full text histories, streamed output. Wall time includes client serialization, API tokenization and admission, but excludes initial cold fill.'}


def save():
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(out)


def request(index, prompt, maximum):
    messages = histories[index]['messages'] + [{'role': 'user', 'content': prompt}]
    started = time.perf_counter()
    body = {'model': '/model', 'messages': messages, 'temperature': 0,
            'max_tokens': maximum, 'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'thinking': False}}
    req = urllib.request.Request(a.url + '/v1/chat/completions',
          data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    first = None
    content = ''
    reasoning = ''
    usage = None
    finish = None
    done = False
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            value = line[5:].strip()
            if value == '[DONE]':
                done = True
                break
            event = json.loads(value)
            if 'error' in event:
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or ''
                if first is None and (text or thought):
                    first = time.perf_counter()
                content += text
                reasoning += thought
                assert not delta.get('tool_calls'), 'No tools were requested.'
                finish = choice.get('finish_reason') or finish
    ended = time.perf_counter()
    assert done and usage is not None and first is not None
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    row = {'context': index, 'text': content, 'reasoning': reasoning, 'usage': usage,
           'finish_reason': finish, 'ttft_seconds': first-started,
           'seconds': ended-started, 'decode_seconds': ended-first,
           'cache_hit_passed': cached >= histories[index]['target_context'] - 256}
    messages.append({'role': 'assistant', 'content': content})
    return row, messages


save()
try:
    # Prove the template matches before sending all 30 large requests. Preserve
    # this probe's actual continuation so history0 is not resumed as an old fork.
    row, messages = request(0, 'Return exactly CHAT_READY.', 32)
    result['probe'] = row
    histories[0]['messages'] = messages
    probe_file = history_dir / f'{a.name}-probe-history0.json'
    probe_file.write_text(json.dumps(histories[0]))
    result['probe_history_file'] = str(probe_file)
    save()
    assert row['cache_hit_passed'] and row['text'].strip() == 'CHAT_READY', row
    result['phase'] = 'wave'
    save()
    updates = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=30) as executor:
        pending = {executor.submit(request, i,
            f'Return consecutive integers from {1000+i*1000} through {1999+i*1000}, separated by comma and space. No explanation.',
            a.output_tokens): i for i in range(30)}
        for future in as_completed(pending):
            row, messages = future.result()
            i = row['context']
            initial = [int(n) for n in re.findall(r'\d+', row['text'])[:16]]
            row['counting_prefix_passed'] = initial == list(range(1000+i*1000, 1016+i*1000))
            row['passed'] = (row['cache_hit_passed'] and row['counting_prefix_passed']
                             and row['usage']['completion_tokens'] == a.output_tokens)
            result['rows'].append(row)
            updates[i] = messages
            save()
    result['wall_seconds'] = time.perf_counter() - started
    # Keep large artifact serialization outside the measured wave.
    for i, messages in sorted(updates.items()):
        history_file = history_dir / f'{a.name}-history{i}.json'
        history_file.write_text(json.dumps({'target_context': histories[i]['target_context'], 'messages': messages}))
        result['history_files'].append({'context': i, 'path': str(history_file)})
    result['output_tokens'] = sum(row['usage']['completion_tokens'] for row in result['rows'])
    result['aggregate_wall_tokens_per_second'] = result['output_tokens']/result['wall_seconds']
    result['passed'] = len(result['rows']) == 30 and all(row['passed'] for row in result['rows'])
    result['phase'] = 'complete'
    assert result['passed'], 'Chat API cache/output gate failed.'
except Exception as error:
    result['error'] = str(error)
    raise
finally:
    save()
print(json.dumps({k: result[k] for k in ('passed', 'wall_seconds', 'output_tokens', 'aggregate_wall_tokens_per_second')}))

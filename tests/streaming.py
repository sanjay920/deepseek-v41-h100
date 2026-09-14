"""Delivered output timing; shared by the independent-conversation benchmark."""
import hashlib
import json
import time
import urllib.request


def post(url, body, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def stream(url, body, barrier, *, on_first_token=None):
    data = json.dumps(body).encode()
    input_sha256=hashlib.sha256(json.dumps(body['input_ids']).encode()).hexdigest()
    barrier.wait(timeout=60)
    start = time.perf_counter()
    events, text, meta, done = [], '', {}, False
    previous = 0
    try:
        req = urllib.request.Request(url + '/generate', data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1200) as response:
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
                text = item.get('text', text)
                meta = item.get('meta_info', meta)
                count = meta.get('completion_tokens', 0)
                if count > previous:
                    if previous == 0 and on_first_token is not None:
                        on_first_token(body['rid'])
                    events.append([time.perf_counter(), count - previous])
                    previous = count
        end = time.perf_counter()
        finish = meta.get('finish_reason', {})
        eos = int(finish.get('type') == 'stop' and finish.get('matched') == 1)
        if eos and events:
            events[-1][1] -= 1
        events = [e for e in events if e[1] > 0]
        count = previous - eos
        return {'rid': body['rid'], 'input_tokens': len(body['input_ids']),
                'input_sha256':input_sha256,
                'start': start, 'end': end, 'events': events, 'text': text,
                'output_tokens': count, 'cached_tokens': meta.get('cached_tokens', 0),
                'ttft_seconds': events[0][0] - start if events else None,
                'decode_tokens_per_second': ((count-events[0][1])/(end-events[0][0])) if events else None,
                'done': done, 'server_meta': meta}
    except Exception as exc:
        return {'rid': body['rid'], 'start': start, 'end': time.perf_counter(),
                'error': repr(exc), 'done': False, 'events': events, 'text': text,
                'server_meta': meta, 'output_tokens': previous}


def summarize(rows):
    start, end = min(r['start'] for r in rows), max(r['end'] for r in rows)
    total = sum(r['output_tokens'] for r in rows)
    good = [r for r in rows if r['events'] and r['done']]
    overlap = None
    if len(good) == len(rows):
        lo = max(r['events'][0][0] for r in good)
        hi = min(r['events'][-1][0] for r in good)
        if hi-lo >= 0.5:
            n = sum(n for r in good for t,n in r['events'] if lo < t <= hi)
            overlap = {'seconds': hi-lo, 'output_tokens': n,
                       'aggregate_tokens_per_second': n/(hi-lo)}
    ttfts = sorted(r['ttft_seconds'] for r in good)
    rates = [r['decode_tokens_per_second'] for r in good]
    return {'requests': len(rows), 'completed': len(good), 'output_tokens': total,
            'wall_seconds': end-start, 'aggregate_wall_tokens_per_second': total/(end-start),
            'all_streams_overlap': overlap,
            'ttft_min_seconds': min(ttfts) if ttfts else None,
            'ttft_max_seconds': max(ttfts) if ttfts else None,
            'mean_stream_decode_tokens_per_second': sum(rates)/len(rates) if rates else None}

"""Real distinct long conversations: cold retrieval, cache persistence, concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import threading
import time
import urllib.request
from tokenizers import Tokenizer
import cached
from cached import fixture, generate
from streaming import stream, summarize, post

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:30000')
p.add_argument('--tokens',type=int,default=400000)
p.add_argument('--clients',type=int,default=30)
p.add_argument('--output-tokens',type=int,default=512)
p.add_argument('--name',required=True)
p.add_argument('--workload',choices=['counting','prose'],default='counting')
p.add_argument('--resume',type=Path)
p.add_argument('--fixtures-only',action='store_true')
p.add_argument('--model-dir',type=Path,default=Path('/model'))
p.add_argument('--output-dir',type=Path,default=Path('/runs/independent'))
a=p.parse_args()
if not 1 <= a.clients <= 30:
    p.error('--clients must be between 1 and 30')
if Path(a.name).name != a.name or a.name in ('.', '..'):
    p.error('--name must be a filename, not a path')
folder=a.output_dir
folder.mkdir(parents=True,exist_ok=True)
path=folder/f'{a.name}.json'
spec=importlib.util.spec_from_file_location('reference_encoding',a.model_dir/'encoding/encoding.py')
encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
encode_messages=encoder.encode_messages
cached.encode_messages=encode_messages
tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
encode=lambda messages:tokenizer.encode(encode_messages(messages,thinking_mode='chat'),add_special_tokens=False).ids
result={'name':a.name,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'passed':False,'primes':[],'cache_checks':[],'pending_request_ids':[],'history_files':[]}
histories=[]
sources=[]


def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)


def wave_prompt(workload,i):
    if workload=='counting':
        return f'Return consecutive integers from {1000+i*1000} through {1999+i*1000}, separated by comma and space. No explanation.'
    topics=['database recovery','TCP congestion control','CPU scheduling','compiler optimization','cache eviction','distributed consensus']
    return f'Write a detailed 1200-word tutorial about {topics[i%len(topics)]}, with examples, mechanisms, and failure cases. Plain prose.'


def abort(signum,frame):
    for rid in result['pending_request_ids']:
        try:
            with post(a.url+'/abort_request',{'rid':rid},timeout=5) as response:response.read()
        except Exception:pass
    result['interrupted']=True;save()
    raise SystemExit(128+signum)


signal.signal(signal.SIGTERM,abort)
signal.signal(signal.SIGINT,abort)
for i in range(a.clients):
    f=folder/f'fixture-{a.tokens}-{i}.json'
    if f.exists():source=json.loads(f.read_text())
    else:
        content,codes=fixture(tokenizer,a.tokens,seed=20260914+a.tokens+i*1000003)
        source={'content':content,'codes':codes,'seed':20260914+a.tokens+i*1000003}
        f.write_text(json.dumps(source))
    sources.append(source)
    histories.append([{'role':'user','content':source['content']}])
result['prefixes']=[{'tokens':len(encode(h)),'sha256':hashlib.sha256(json.dumps(encode(h)).encode()).hexdigest()} for h in histories]
assert len({s['sha256'] for s in result['prefixes']})==a.clients
save()
if a.fixtures_only:
    result.update(passed=True,fixture_only=True);save()
    print(json.dumps({'event':'fixtures_ready','clients':a.clients,'total_tokens':sum(s['tokens'] for s in result['prefixes'])}),flush=True)
    raise SystemExit(0)
with urllib.request.urlopen(a.url+'/server_info',timeout=30) as response:result['server_before']=json.load(response)
start_context=0
if a.resume:
    prior=json.loads(a.resume.read_text())
    assert prior['args']['clients']<=a.clients and prior['prefixes']==result['prefixes'][:prior['args']['clients']]
    assert len(prior['primes'])==len(prior['history_files'])
    start_context=len(prior['primes'])
    histories[:start_context]=[json.loads(Path(f).read_text())['messages'] for f in prior['history_files']]
    if prior.get('wave') and not prior.get('history_includes_wave'):
        for row in prior['wave']['rows']:
            i=row['context']
            messages=histories[i]+[{'role':'user','content':wave_prompt(prior['args']['workload'],i)}]
            assert hashlib.sha256(json.dumps(encode(messages)).encode()).hexdigest()==row['input_sha256']
            histories[i]=messages+[{'role':'assistant','content':row['text']}]
    result['primes']=prior['primes']
    for i in range(start_context):
        f=folder/f'{a.name}-history{i}.json'
        f.write_text(json.dumps({'target_context':a.tokens,'messages':histories[i]}))
        result['history_files'].append(str(f))
if start_context<a.clients:
    for i in range(start_context,a.clients):
        rid=f'{a.name}-prime{i}-{time.time_ns()}'
        result['pending_request_ids']=[rid];save()
        print(json.dumps({'event':'prime_start','context':i,'rid':rid}),flush=True)
        row=generate(a.url,tokenizer,histories[i],rid,128)
        try:row['correct']=json.loads(row['text'])==sources[i]['codes']
        except ValueError:row['correct']=False
        row['context']=i
        result['primes'].append(row);result['pending_request_ids']=[];save()
        print(json.dumps({'event':'prime_done','context':i,'correct':row['correct'],'input_tokens':row['input_tokens'],'cached_tokens':row['cached_tokens'],'ttft':row['ttft_seconds']}),flush=True)
        if not row['correct']:raise SystemExit('Long retrieval failed')
        histories[i].append({'role':'assistant','content':row['text']})
        f=folder/f'{a.name}-history{i}.json'
        f.write_text(json.dumps({'target_context':a.tokens,'messages':histories[i]}))
        result['history_files'].append(str(f));save()
for i in range(a.clients):
    messages=histories[i]+[{'role':'user','content':'Recall the three audit passcodes. Return only their JSON object.'}]
    rid=f'{a.name}-cache{i}-{time.time_ns()}'
    result['pending_request_ids']=[rid];save()
    row=generate(a.url,tokenizer,messages,rid,128)
    try:correct=json.loads(row['text'])==sources[i]['codes']
    except ValueError:correct=False
    row.update(context=i,correct=correct,cache_hit_passed=row['cached_tokens']>=result['prefixes'][i]['tokens']-256)
    result['cache_checks'].append(row);result['pending_request_ids']=[];save()
    print(json.dumps({'event':'cache_check','context':i,'correct':correct,'cached_tokens':row['cached_tokens'],'ttft':row['ttft_seconds']}),flush=True)
    if not correct or not row['cache_hit_passed']:raise SystemExit('Independent cache-persistence check failed')
    histories[i]=messages+[{'role':'assistant','content':row['text']}]
result['history_files']=[]
for i,h in enumerate(histories):
    f=folder/f'{a.name}-history{i}.json'
    f.write_text(json.dumps({'target_context':a.tokens,'messages':h}))
    result['history_files'].append(str(f))
bodies=[]
for i,h in enumerate(histories):
    prompt=wave_prompt(a.workload,i)
    rid=f'{a.name}-wave{i}-{time.time_ns()}'
    bodies.append({'rid':rid,'input_ids':encode(h+[{'role':'user','content':prompt}]),'stream':True,
                   'sampling_params':{'temperature':0,'max_new_tokens':a.output_tokens}})
result['pending_request_ids']=[b['rid'] for b in bodies];save()
barrier=threading.Barrier(a.clients)
rows=[]
print(json.dumps({'event':'concurrent_start','clients':a.clients}),flush=True)
with ThreadPoolExecutor(max_workers=a.clients) as executor:
    futures={executor.submit(stream,a.url,b,barrier):i for i,b in enumerate(bodies)}
    for f in as_completed(futures):
        row=f.result();i=futures[f]
        row['context']=i
        row['cache_hit_passed']=row.get('cached_tokens',0)>=result['prefixes'][i]['tokens']-256
        if a.workload=='counting':
            expected=', '.join(str(n) for n in range(1000+i*1000,2000+i*1000))
            row['content_passed']=bool(row['text'].strip()) and expected.startswith(row['text'].strip())
        else:row['content_passed']=row['output_tokens']>=128
        rows.append(row);result['pending_request_ids'].remove(row['rid']);save()
rows.sort(key=lambda r:r['context'])
wave=summarize(rows)
wave['rows']=rows
wave['passed']=all(r['done'] and r['cache_hit_passed'] and r['content_passed'] for r in rows)
result['wave']=wave;result['passed']=wave['passed'];save()
if result['passed']:
    for row in rows:
        i=row['context']
        histories[i].extend([{'role':'user','content':wave_prompt(a.workload,i)},
                             {'role':'assistant','content':row['text']}])
        Path(result['history_files'][i]).write_text(json.dumps({'target_context':a.tokens,'messages':histories[i]}))
    result['history_includes_wave']=True;save()
print(json.dumps({k:v for k,v in wave.items() if k!='rows'}),flush=True)
raise SystemExit(0 if result['passed'] else 1)

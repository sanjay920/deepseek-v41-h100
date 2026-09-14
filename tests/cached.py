"""Measure a growing cached conversation using the pinned reference encoder."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import urllib.request
from tokenizers import Tokenizer

def fixture(tokenizer, target, seed=None):
    rng = random.Random(20260912 + target if seed is None else seed)
    records = [
        f'Record {i}: warehouse {rng.randrange(100,999)} received {rng.randrange(100,999)} units of item {rng.randrange(10000,99999)}.\n'
        for i in range(target//10+100)
    ]
    filler = tokenizer.encode(''.join(records), add_special_tokens=False).ids
    codes = {key: f'{rng.getrandbits(48):012x}' for key in ('alpha','beta','gamma')}
    markers = [tokenizer.encode(f'\nSPECIAL AUDIT ENTRY: {k} passcode is {v}.\n', add_special_tokens=False).ids for k,v in codes.items()]
    room = target - sum(map(len,markers)) - 100
    output = []
    prior = 0
    for cut, marker in zip([room//10,room//2,room*9//10],markers):
        output.extend(filler[prior:cut]); output.extend(marker); prior=cut
    output.extend(filler[prior:room])
    content = ('Read these records and remember the three special audit passcodes.\n'
               +tokenizer.decode(output,skip_special_tokens=False)
               +'\nReturn only a JSON object with keys alpha, beta, and gamma and their exact passcodes.')
    return content,codes

def generate(url, tokenizer, messages, rid, maximum, return_logprob=False, sampling_overrides=None):
    ids = tokenizer.encode(encode_messages(messages,thinking_mode='chat'),add_special_tokens=False).ids
    body = {'rid':rid,'input_ids':ids,'stream':True,
            'sampling_params':{'temperature':0,'max_new_tokens':maximum}}
    body['sampling_params'].update(sampling_overrides or {})
    if return_logprob:
        body.update(return_logprob=True,logprob_start_len=len(ids)-1,top_logprobs_num=10)
    request=urllib.request.Request(url+'/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter(); first=None; first_count=0; text=''; meta={}; done=False
    last_text=None; last_count=0
    with urllib.request.urlopen(request,timeout=7200) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':done=True;break
            data=json.loads(payload)
            if 'error' in data:raise RuntimeError(data['error'])
            old_text=text
            text=data.get('text',text); meta=data.get('meta_info',meta)
            if text and text!=old_text:
                last_text=time.perf_counter()
                last_count=meta.get('completion_tokens',len(tokenizer.encode(text,add_special_tokens=False).ids))
            if first is None and text:
                first=time.perf_counter()
                first_count=meta.get('completion_tokens',len(tokenizer.encode(text,add_special_tokens=False).ids))
                print(json.dumps({'event':'first_text','request_id':rid,'seconds':first-start}),flush=True)
    end=time.perf_counter()
    finish=meta.get('finish_reason',{})
    eos=finish.get('type')=='stop' and finish.get('matched')==1
    count=meta.get('completion_tokens',0)-int(eos)
    return {'request_id':rid,'input_tokens':len(ids),'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'text':text,'stream_done':done,'output_tokens':count,'retokenized_output_tokens':len(tokenizer.encode(text,add_special_tokens=False).ids),
            'cached_tokens':meta.get('cached_tokens',0),'ttft_seconds':None if first is None else first-start,
            'fresh_input_tokens':len(ids)-meta.get('cached_tokens',0),
            'total_seconds':end-start,'decode_tokens_per_second':None if first is None else max(0,count-first_count)/(end-first),
            'active_decode_tokens_per_second':None if first is None or last_text is None or last_text<=first else (last_count-first_count)/(last_text-first),
            'finish_delay_seconds':None if last_text is None else end-last_text,
            'sampling_params':body['sampling_params'],'server_meta':meta}

def main():
    global encode_messages
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=400000)
    parser.add_argument('--model-dir',type=Path,default=Path('/model'))
    parser.add_argument('--url',default='http://127.0.0.1:30000')
    parser.add_argument('--output',type=Path,default=Path('/runs/cached.json'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--sampled',action='store_true',help='Also run explanation and creative-writing turns at temperature 0.6.')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('reference_encoding',args.model_dir/'encoding/encoding.py')
    encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    encode_messages=encoder.encode_messages
    tokenizer=Tokenizer.from_file(str(args.model_dir/'tokenizer.json'))
    content,codes=fixture(tokenizer,args.tokens)
    messages=[{'role':'user','content':content}]
    ids=tokenizer.encode(encode_messages(messages,thinking_mode='chat'),add_special_tokens=False).ids
    result={'passed':False,'target_context':args.tokens,'input_tokens':len(ids),
            'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'turns':[]}
    if args.dry_run:
        result.pop('passed')
        result['dry_run']=True
        print(json.dumps(result,indent=2));return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counting='Return the integers from 1 through 60 in order, separated by commas. Output only the sequence.'
    retrieval='Recall the three special audit passcodes from the records. Return only a JSON object with keys alpha, beta, and gamma. Also include key squares containing the first twenty positive integer squares in order.'
    tasks=[('prime',None,128,0)]
    for n in range(2):
        tasks += [(f'counting_{n}',counting,200,0),(f'retrieval_{n}',retrieval,256,0)]
    tasks += [
        ('code','Write a complete Python module containing first_unique_char(text). Return the first character occurring exactly once, or None if there is none. Treat uppercase and lowercase as different. Include a short docstring and a unittest.TestCase with at least six tests, covering empty input, repeated characters, spaces, and case sensitivity. Return Python only, without Markdown fences.',800,0),
        ('prose','Explain how a cache hit and a cache miss differ, using a library as an analogy. Then explain why a long conversation can still take work to generate its next word even when earlier text is cached. About 200 words, plain prose.',512,0),
    ]
    if args.sampled:
        tasks += [
            ('counting_2',counting,200,0),('retrieval_2',retrieval,256,0),
            ('explanation','Explain the difference between latency and throughput in an inference server. Include two simple numerical examples, and explain why adding concurrent requests can improve throughput while making each request slower. About 250 words, plain prose.',768,0.6),
            ('creative','Write a scene of about 350 words in which two friends repair an old radio during a power outage. Include natural dialogue, a small disagreement, and an understated ending. Use concrete details and no headings.',768,0.6),
        ]
    try:
        for name,prompt,limit,temperature in tasks:
            if prompt is not None:messages.append({'role':'user','content':prompt})
            result['pending_turn']=name
            args.output.write_text(json.dumps(result,indent=2))
            sampling={'temperature':temperature}
            if temperature:sampling['top_p']=0.95
            response=generate(args.url,tokenizer,messages,f'cached-{name}-{time.time_ns()}',limit,sampling_overrides=sampling)
            if name=='prime':
                correct=json.loads(response['text'])==codes
            elif name.startswith('counting'):
                correct=response['text'].strip()==', '.join(str(i) for i in range(1,61))
            elif name.startswith('retrieval'):
                correct=json.loads(response['text'])=={**codes,'squares':[i*i for i in range(1,21)]}
            elif name=='code':
                parsed=ast.parse(response['text'])
                names=[n.name for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef)]
                correct='first_unique_char' in names and sum(n.startswith('test_') for n in names)>=6
                args.output.with_name('generated.py').write_text(response['text'])
            else:
                correct=response['output_tokens']>=128
            cache_ok=name=='prime' or response['cached_tokens']>=len(ids)-256
            response.update(turn=name,passed=correct and cache_ok and response['stream_done'])
            result['turns'].append(response)
            print(json.dumps({k:response[k] for k in ('turn','passed','input_tokens','output_tokens','cached_tokens','ttft_seconds','decode_tokens_per_second')}),flush=True)
            assert response['passed'],f'Failed turn: {name}'
            messages.append({'role':'assistant','content':response['text']})
        result['passed']=True;result['messages']=messages;result.pop('pending_turn',None)
    finally:args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

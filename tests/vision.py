"""Exercise the actual image-input path with a controlled two-color image."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('/runs/vision.json'))
parser.add_argument('--url',default='http://127.0.0.1:30000')
parser.add_argument('--history',type=Path)
parser.add_argument('--width',type=int,default=1920)
parser.add_argument('--height',type=int,default=1080)
args=parser.parse_args()
canvas=Image.new('RGB',(args.width,args.height),(255,0,0))
ImageDraw.Draw(canvas).rectangle((args.width//2,0,args.width-1,args.height-1),fill=(0,0,255))
buffer=io.BytesIO();canvas.save(buffer,format='PNG');png=buffer.getvalue()
body={'model':'/model','temperature':0,'max_tokens':64,'messages':[{'role':'user','content':[
    {'type':'text','text':'Name the color of the left half and the right half of this image. Return only JSON with keys left and right, using lowercase color names.'},
    {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},
]}]}
history=None
if args.history:
    history=json.loads(args.history.read_text())
    body['messages']=history['messages']+body['messages']
start=time.perf_counter()
req=urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=600) as response: result=json.load(response)
text=result['choices'][0]['message']['content'].strip()
try: actual=json.loads(text)
except ValueError:actual=None
record={'passed':actual=={'left':'red','right':'blue'},'seconds':time.perf_counter()-start,'image_size':[args.width,args.height],'image_sha256':hashlib.sha256(png).hexdigest(),'response':result}
if history:
    details=result['usage'].get('prompt_tokens_details') or {}
    record['cached_tokens']=details.get('cached_tokens',0)
    record['cache_hit_passed']=record['cached_tokens']>=history['target_context']-512
    record['passed']=record['passed'] and record['cache_hit_passed'] and details.get('image_tokens',0)>0
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
assert record['passed'],record

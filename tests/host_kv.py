"""Actual packed KV writes and FlashMLA reads from registered host memory."""
import argparse
import json
from pathlib import Path
import torch
import triton
from sglang.srt.mem_cache.mapped_host_kv import HostKVAllocation
from sglang.kernels.ops.attention.dsv4 import fused_store_cache
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,default=4096)
p.add_argument('--batch',type=int,default=6)
p.add_argument('--name',default='host-kv-small')
p.add_argument('--context-tokens',type=int)
p.add_argument('--page',type=int,default=256)
p.add_argument('--dtype',choices=['uint8','float8'],default='uint8')
a=p.parse_args()
from sglang.srt.mem_cache.host_kv_staging import gather_packed_rows
root=Path('/runs')
root.mkdir(exist_ok=True,parents=True)
path=root/f'{a.name}.json'
result={'passed':False,'tokens':a.tokens,'batch':a.batch,'phase':'allocate'}
path.write_text(json.dumps(result,indent=2))
torch.cuda.set_device(0)
torch.manual_seed(771)
page=a.page
exposed_dtype=torch.uint8 if a.dtype=='uint8' else torch.float8_e4m3fn
pages=triton.cdiv(a.tokens,page)
row_bytes=triton.cdiv(page*584,576)*576
host=HostKVAllocation((pages,row_bytes))
gpu=torch.zeros((pages,row_bytes),dtype=torch.uint8,device='cuda')
result.update(host_bytes=host.host.numel(),host_device_pointer_equal=host.host.data_ptr()==host.cuda.data_ptr())
for start in range(0,pages*page,65536):
    count=min(65536,pages*page-start)
    values=(torch.randn(count,512,device='cuda')*.2).bfloat16()
    loc=torch.arange(start,start+count,device='cuda',dtype=torch.int64)
    for cache in (gpu,host.cuda):
        fused_store_cache(input=values,cache=cache,indices=loc,page_size=page,type='flashmla')
    torch.cuda.synchronize()
del values,loc
result['packed_writes_exact']=torch.equal(gpu.cpu(),host.host)
result['phase']='attention'
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
q=(torch.randn(a.batch,1,64,512,device='cuda')*.2).bfloat16()
sink=torch.randn(64,device='cuda')
swa=torch.zeros((1,128*584),dtype=torch.uint8,device='cuda')
x=(torch.randn(128,512,device='cuda')*.2).bfloat16()
fused_store_cache(input=x,cache=swa,indices=torch.arange(128,device='cuda'),page_size=128,type='flashmla')
swa=swa.view(1,128,1,584)
swa_idx=torch.arange(128,device='cuda',dtype=torch.int32).view(1,1,128).repeat(a.batch,1,1)
def make_indices():
    span=a.context_tokens or a.tokens
    if a.context_tokens:
        assert a.tokens==a.batch*a.context_tokens
    value=torch.randint(0,span,(a.batch,1,512),device='cuda',dtype=torch.int32).sort(-1).values
    if a.context_tokens:
        value+=torch.arange(a.batch,device='cuda',dtype=torch.int32).view(-1,1,1)*span
    base=(torch.arange(a.batch,device='cuda',dtype=torch.int32)*span if a.context_tokens else torch.zeros(a.batch,device='cuda',dtype=torch.int32))
    value[:,:,:4]=base[:,None,None]+torch.arange(4,device='cuda',dtype=torch.int32)[None,None,:]
    return value
idx=make_indices()
lengths=torch.full((a.batch,),512,device='cuda',dtype=torch.int32)
swa_lengths=torch.full((a.batch,),128,device='cuda',dtype=torch.int32)
md_gpu=get_mla_metadata()[0]
md_host=get_mla_metadata()[0]


def run(cache,metadata,selected=None):
    cache=cache.view(exposed_dtype)
    return flash_mla_with_kvcache(q=q,k_cache=swa.view(exposed_dtype),head_dim_v=512,
        block_table=None,cache_seqlens=None,tile_scheduler_metadata=metadata,
        softmax_scale=512**-.5,is_fp8_kvcache=True,indices=swa_idx,
        topk_length=swa_lengths,attn_sink=sink,
        extra_k_cache=cache[:,:page*584].view(cache.shape[0],page,1,584),
        extra_indices_in_kvcache=idx if selected is None else selected,extra_topk_length=lengths)[0]


expected=run(gpu,md_gpu)
actual=run(host.cuda,md_host)
torch.cuda.synchronize()
result['attention_exact']=torch.equal(expected,actual)
result['gpu_ms']=triton.testing.do_bench_cudagraph(lambda:run(gpu,md_gpu),rep=100)
result['host_ms']=triton.testing.do_bench_cudagraph(lambda:run(host.cuda,md_host),rep=100)
md_staged=get_mla_metadata()[0]
staged,local=gather_packed_rows(host.cuda.view(exposed_dtype),idx,page)
staged_result=run(staged,md_staged,local)
torch.cuda.synchronize()
result['staged_exact']=torch.equal(expected,staged_result)
def staged_once():
    gathered,selected=gather_packed_rows(host.cuda.view(exposed_dtype),idx,page,staged)
    return run(gathered,md_staged,selected)
result['staged_once_ms']=triton.testing.do_bench_cudagraph(staged_once,rep=100)
def staged_group():
    gathered,selected=gather_packed_rows(host.cuda.view(exposed_dtype),idx,page,staged)
    for _ in range(6):run(gathered,md_staged,selected)
def host_group():
    for _ in range(6):run(host.cuda,md_host)
result['staged_six_layers_ms']=triton.testing.do_bench_cudagraph(staged_group,rep=100)
result['host_six_layers_ms']=triton.testing.do_bench_cudagraph(host_group,rep=100)
graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
update_locs=(torch.arange(a.batch,device='cuda',dtype=torch.int64)[:,None]*(a.context_tokens or 4)+torch.arange(4,device='cuda')) .flatten()
updates=torch.randn(update_locs.numel(),512,device='cuda').bfloat16()
def write(cache):
    fused_store_cache(input=updates,cache=cache,indices=update_locs,page_size=page,type='flashmla')
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    run(host.cuda,md_host);run(host.cuda,md_host)
torch.cuda.current_stream().wait_stream(stream)
with torch.cuda.graph(graph,stream=stream):
    write(host.cuda)
    captured=run(host.cuda,md_host)
    captured_staged=staged_once()
checks=[]
staged_checks=[]
for seed in range(5):
    torch.manual_seed(seed)
    q.copy_((torch.randn_like(q.float())*.2).bfloat16())
    idx.copy_(make_indices())
    updates.copy_(torch.randn_like(updates))
    write(gpu)
    expected=run(gpu,md_gpu)
    graph.replay();torch.cuda.synchronize()
    checks.append(torch.equal(expected,captured))
    staged_checks.append(torch.equal(expected,captured_staged))
result.update(graph_exact=checks,staged_graph_exact=staged_checks,phase='complete',
              staging_module='sglang.srt.mem_cache.host_kv_staging',
              context_tokens=a.context_tokens,
              passed=result['packed_writes_exact'] and result['attention_exact'] and result['staged_exact'] and all(checks) and all(staged_checks))
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
graph.reset()
torch.cuda.synchronize()
assert result['passed']

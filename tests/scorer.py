"""Exact masked-score, selected-index, and graph replay gates for tile skipping."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import triton
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import fp4_index_logits_decode as candidate

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('/runs/scorer.json'))
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
source=ROOT/'upstream/python/sglang/kernels/ops/attention/dsv4/sm90_fp4_indexer.py'
spec=importlib.util.spec_from_file_location('original_scorer',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.fp4_index_logits_decode
torch.manual_seed(2718)
results=[]


def make_mask(rows,length,lens,mode):
    mask=torch.zeros((rows,(length+7)//8),dtype=torch.bool,device='cuda')
    for b in range(rows):
        if mode=='none':continue
        chosen=torch.randperm(mask.shape[1],device='cuda')[:min(2048,mask.shape[1])]
        mask[b,chosen]=True
    mask=mask.repeat_interleave(8,dim=1)[:,:length].contiguous()
    return mask & (torch.arange(length,device='cuda')[None,:]<lens[:,None])


with torch.inference_mode():
    for rows,length,page in ((1,2048,128),(6,65536,256),(1,409920,128),(6,409920,256),(6,1048576,128)):
        pages=triton.cdiv(length,page)
        table=torch.randint(0,256,(pages,page*68),dtype=torch.uint8,device='cuda')
        table[:,page*64:]=torch.randint(123,130,(pages,page*4),dtype=torch.uint8,device='cuda')
        slots=torch.randperm(pages*page,device='cuda')[:length].expand(rows,-1)
        q=(torch.randn(rows,32,128,device='cuda')*.2).bfloat16()
        weights=(torch.randn(rows,32,device='cuda')*.02).bfloat16()
        lens=torch.full((rows,),length,dtype=torch.int64,device='cuda')
        mask=make_mask(rows,length,lens,'random')
        def ref():return original(q,weights,slots,lens,table,page).masked_fill(~mask,-torch.inf)
        def alt():return candidate(q,weights,slots,lens,table,page,candidate_mask=mask).masked_fill(~mask,-torch.inf)
        source_exact=torch.equal(original(q,weights,slots,lens,table,page),candidate(q,weights,slots,lens,table,page))
        expected=ref();actual=alt()
        exact=torch.equal(expected,actual)
        topk_exact=torch.equal(expected.topk(512,sorted=False).indices,actual.topk(512,sorted=False).indices)
        native_ms=triton.testing.do_bench_cudagraph(ref,rep=120)
        skipped_ms=triton.testing.do_bench_cudagraph(alt,rep=120)
        graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):alt()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph,stream=stream):graphed=alt()
        checks=[]
        for visible in (0,1,63,64,65,length//2,length-1,length):
            lens.fill_(visible)
            mask.copy_(make_mask(rows,length,lens,'random'))
            q.copy_((torch.randn_like(q.float())*.2).bfloat16())
            expected=ref();graph.replay();torch.cuda.synchronize()
            checks.append({'visible':visible,'exact':torch.equal(expected,graphed),
                           'topk_exact':torch.equal(expected.topk(512,sorted=False).indices,graphed.topk(512,sorted=False).indices)})
        mask.zero_();expected=ref();graph.replay();torch.cuda.synchronize()
        checks.append({'visible':length,'mask':'empty','exact':torch.equal(expected,graphed),
                       'topk_exact':torch.equal(expected.topk(512,sorted=False).indices,graphed.topk(512,sorted=False).indices)})
        passed=source_exact and exact and topk_exact and all(c['exact'] and c['topk_exact'] for c in checks)
        row={'rows':rows,'length':length,'page_size':page,'exact':exact,'topk_exact':topk_exact,
             'source_scores_exact':source_exact,'slot_row_stride':slots.stride(0),
             'native_ms':native_ms,'skipped_ms':skipped_ms,'speedup':native_ms/skipped_ms,'graph_checks':checks,'passed':passed}
        results.append(row);print(json.dumps(row),flush=True)
        args.output.write_text(json.dumps({'passed':all(r['passed'] for r in results),'results':results},indent=2))
        assert passed
        graph.reset();del graph,stream,expected,actual,graphed,table,slots,q,weights,lens,mask
        torch.cuda.empty_cache()

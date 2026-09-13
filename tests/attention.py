"""Compare selected-entry KV gathering with full-store FlashMLA attention."""
import json
import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd
from sglang.srt.layers.attention.dsv4.compact_prefill_gather import compact_prefill_gather


def main():
    torch.manual_seed(923)
    results = []
    with torch.inference_mode():
        for main_count, rows in ((8192, 64), (131072, 256), (1048576, 256)):
            swa_count = 384
            main = torch.randn(main_count, 1, 512, device='cuda', dtype=torch.bfloat16)
            swa = torch.randn(swa_count, 1, 512, device='cuda', dtype=torch.bfloat16)
            main_ids = torch.randperm(main_count, device='cuda')
            swa_ids = torch.randperm(swa_count, device='cuda')
            indices = torch.cat([
                torch.randint(main_count, (rows, 512), device='cuda', dtype=torch.int32),
                torch.randint(main_count, main_count+swa_count, (rows, 128), device='cuda', dtype=torch.int32),
            ], dim=1)
            indices[0] = -1
            indices[1, 100:] = -1
            lengths = (indices >= 0).sum(dim=1).int()
            cm, cs, ci = compact_prefill_gather(main_ids, swa_ids, indices)
            original = torch.cat([main[main_ids], swa[swa_ids]])
            compact = torch.cat([main[cm], swa[cs]])
            valid = indices >= 0
            values_exact = torch.equal(original[indices[valid].long()], compact[ci[valid].long()])
            # The kernel uses the same 64-head shape as the server's padded queries.
            q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
            sink = torch.zeros(64, device='cuda', dtype=torch.float32)
            expected = flash_mla_sparse_fwd(q=q, kv=original, indices=indices.unsqueeze(1), sm_scale=512**-.5, d_v=512, attn_sink=sink, topk_length=lengths)[0]
            actual = flash_mla_sparse_fwd(q=q, kv=compact, indices=ci.unsqueeze(1), sm_scale=512**-.5, d_v=512, attn_sink=sink, topk_length=lengths)[0]
            results.append({
                'main_tokens': main_count, 'query_rows': rows,
                'original_kv_rows': len(original), 'compact_kv_rows': len(compact),
                'selected_values_exact': values_exact, 'attention_exact': torch.equal(expected, actual),
                'max_error': float((expected-actual).abs().max()),
            })
            del main, swa, main_ids, swa_ids, indices, lengths, cm, cs, ci
            del original, compact, valid, q, sink, expected, actual
            torch.cuda.empty_cache()
    passed = all(item['selected_values_exact'] and item['attention_exact'] for item in results)
    print(json.dumps({'passed': passed, 'results': results}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

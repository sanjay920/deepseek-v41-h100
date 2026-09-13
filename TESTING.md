# Tests

[measurements.json](results/measurements.json) and
[validation.json](results/validation.json) summarize recorded runs. They separate
the 4K, 400K cached, 1M, and A100 configurations. The 400K speed measurements
precede the parser-only fix; its GPU sources and settings were unchanged.

[Package verification](results/package-validation.json): both images built and
passed source-hash checks; the 400K and 1M fixtures matched their recorded hashes.
The shipped parser, chat, tool, and image checks passed. API tests targeted the
existing deployment with identical GPU files; packaging did not reload the model
or rerun the GPU primitive benchmarks.

## Checks without loading weights

The Docker build checks upstream hashes, applies the selected patch with zero
fuzz, then verifies the resulting files against the tested hashes.

```bash
docker run --rm deepseek-v41-h100:cached python3 /opt/experiment/apply.py --check
docker run --rm deepseek-v41-h100:cached python3 /opt/experiment/tests/tool_parser.py
```

The parser check covers 916 stream chunkings, including parallel calls, typed
arguments, JSON bodies, and zero-argument calls.

The benchmark scripts require `encoding/encoding.py` alongside the tokenizer.
If reusing a weights folder without it, run `download.py --metadata-only` with
that folder mounted at `/weights`; this fetches the pinned metadata without weights.

Run GPU checks **before starting the model server**:

```bash
mkdir -p runs
docker run --rm --gpus device=0 deepseek-v41-h100:cached \
  python3 /opt/experiment/tests/attention.py
docker run --rm --gpus device=0 -v "$PWD/runs:/runs" deepseek-v41-h100:cached \
  python3 /opt/experiment/tests/scorer.py
```

These compare selected KV values, FlashMLA outputs, masked index scores,
selected indices, and dynamic graph replay, including 1M-entry tables.

## API and cache checks

With `./serve.sh cached "$PWD/weights"` running:

```bash
docker exec deepseek-v41 python3 /opt/experiment/tests/smoke.py
docker exec deepseek-v41 python3 /opt/experiment/tests/tools.py
docker exec deepseek-v41 python3 /opt/experiment/tests/vision.py
docker exec deepseek-v41 python3 /opt/experiment/tests/cached.py --sampled
docker exec deepseek-v41 python3 /opt/experiment/tests/vision.py \
  --history /runs/cached.json --output /runs/cached-vision.json
```

The launcher mounts `./runs` at `/runs` for these outputs. `tools.py` checks auto,
required, named, and disabled tool choice, streamed arguments, reasoning, parallel
calls, typed values, and tool-result handling. Tool results are simulated; no
generated command is executed. Schema-enforcement cases explicitly use `strict=True`.

`cached.py` uses the original synthetic fixture and growing conversation through
`/generate`. It separates cold prefill from cached generation and checks actual
cache reuse. Its code check validates syntax and requested structure; prose is
a throughput test. Sampled output varies. Use an otherwise idle server: other
conversations may evict the prefix and invalidate the cached comparison.

The original generated Python sample also passed 10 of its own tests and 2,000
independent cases. To check a newly generated sample in an isolated container:

```bash
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 64 --memory 256m --cpus 1 \
  -v "$PWD/runs/generated.py:/candidate.py:ro" deepseek-v41-h100:cached \
  python3 /opt/experiment/tests/code.py
```

For the `long` profile, `tests/retrieval.py --tokens 1048000` reproduces the
exact-length 1M fixture. It can take about an hour. `--dry-run` builds and hashes
the input without inference. `VERIFY_INDEXER=1 ./serve.sh long "$PWD/weights"`
enables the original comparison-only indexer checks through 131,072 visible keys.

## Limits observed

- One earlier raw-scorer comparison differed by 7.6e-6. Selected keys matched.
- Split-K changes FP32 accumulation order. Its primitive tests were not all
  bit-exact; the 128-token full-model comparison at matched cache states was exact.
- A probability comparison with different cache boundaries failed. The
  controlled comparison used the same conversation and cache-generation order.
- The first high-resolution image at 400K exhausted a temporary mask allocation.
  Preallocated masks and shared slot reads fixed it; the subsequent test passed.
- Tool tests passed separately, including a 400,520-token tool call. A later
  combined cached tool-result/image test was interrupted by live traffic and is
  **not** counted as passed. The 1M profile's large-image path is unqualified.

The separate A100 runtime passed seven API checks and 36 exact comparisons
against its own baseline. It is not a supported target of these launch scripts.

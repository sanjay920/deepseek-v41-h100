# Tests

## DSpark with 30 independent histories

[September 15 results](results/dspark.json): all 30 cold 400K histories retrieved
their three planted codes. Cache and output checks passed for counting and prose
with 512 output tokens, continued prose with 1,024, and final counting with 128.
With the cache full,
a separate 400K tool call, cached tool-result round trip and 1920×1080 image
request passed. All 30 original histories then passed another cache check.
Seven basic API cases, twelve short tool cases and image checks also passed.
Full server logs showed no retractions or fatal errors; two allocator mapping
warnings recovered without failing requests.

The separate Chat Completions test sent full text histories from one process
with 30 threads. All 30 cache checks and counting prefixes passed, with 512 output
tokens each: 15,360 tokens in 35.10 seconds, or 437.65 tokens/s. That includes
client serialization, API tokenization and admission. It excludes initial cold
fill and writing large history artifacts. Client buffering affects individual
stream timestamps; the whole-wave rate is the useful measurement here.

The new profile fixes excessive cache eviction during speculative allocation,
lets requests enter as soon as one slot is free, bounds prefill workspaces and
avoids redundant score copies. Image MLP weights are created directly in host
memory. New memory paths were compared with original kernel and weight-loading
outputs. Prose samples were reviewed for coherence, not used as a model-quality
evaluation. These tests do not cover mixed cold/warm traffic or 30 simultaneous
image requests. Faster steady decoding did not improve every whole-wave result.

On an otherwise idle `dspark` server, this repeats the workload and checks:

```bash
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name dspark-count
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name dspark-prose --workload prose \
  --resume /runs/independent/dspark-count.json
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name dspark-prose1024 --workload prose \
  --output-tokens 1024 --resume /runs/independent/dspark-prose.json
docker exec deepseek-v41 python3 /opt/experiment/tests/tools.py \
  --long --output /runs/dspark-tools-long.json
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name dspark-final-cache --output-tokens 128 \
  --resume /runs/independent/dspark-prose1024.json
docker exec deepseek-v41 python3 /opt/experiment/tests/chat.py \
  --source /runs/independent/dspark-final-cache.json --name dspark-chat
```

The first command takes roughly four hours to load the histories. The long tool
test uses a separate fixture and adds another cold fill. Each continuation keeps
the preceding replies; use unique output names. The chat test saves its own
updated histories, so its earlier native source histories are no longer current.
The original run qualified two histories before expanding to 30 and included an
additional image fixture. These commands repeat the workload, not identical
conversation bytes or timings. Check server logs for retractions and fatal
errors too; a completed response alone is insufficient.

[DSpark package verification](results/dspark-package-validation.json) separates
package checks from the recorded GPU/model tests. Packaging does not repeat the
four-hour run or reload the serving model.

## Earlier ordinary-decoding profile

[Recorded results](results/independent.json): 30 distinct synthetic histories,
three planted codes per history, then counting and prose waves of 512 output
tokens per request. All 90 original code checks and all 30 cache checks before
each wave passed. Counting matched the requested sequence prefixes. Six prose
samples were reviewed for coherence; this was not a model-quality evaluation.

Rates use delivered output tokens, excluding EOS. The common interval runs from
the last stream's first token to the first stream's last token. Its aggregate
rate divided by 30 gives 27.8 tokens/s for counting and 24.3 for prose. Averaging
each stream from its own first token through completion gives 23.6 and 20.1;
early streams can wait while other requests are admitted. Full-wave rates include
that admission but exclude the earlier cold fills and cache-check requests.
Tokenization and request serialization happen before wave timing.

The first full test returned to an older version of one history after an image
turn. That branch missed cache. Its actual refill exchange became the current
history, and all 30 were checked again; the other 29 were never reloaded. The
reported waves passed from those current histories. Old conversation forks are
not guaranteed to stay cached. Keep each generated reply when continuing a test.

With the long cache populated, a separate 400K tool call, cached tool-result
round trip, and large-image request with tools passed. Separate checks covered
seven basic API cases, twelve tool cases, and a 1920×1080 image. Primitive checks
covered host-KV writes, staged attention, graph replay, index selection, and
bounded sliding-window cache references.

Earlier DSpark candidates reached about 2,200 aggregate tokens/s at **16K** context but
failed cache or memory checks. They are not successful 400K results and are not
included in the `independent` profile. The later `dspark` profile above passed
the long-context gates. Mixed cold-prefill/decode traffic was not
benchmarked; this is not a measured hardware ceiling.

To repeat the workload on an otherwise idle `independent` server:

```bash
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name count
docker exec deepseek-v41 python3 /opt/experiment/tests/independent.py \
  --clients 30 --tokens 400000 --name prose --workload prose \
  --resume /runs/independent/count.json
```

The first command fills the histories sequentially and can take about two hours.
Both commands check cache reuse before measuring a concurrent wave. The script
stops on a failed retrieval or cache check and saves its result under `/runs`.
Use `--fixtures-only` to build and hash the inputs without inference. The test
prompts request long answers, but the measured responses stop at 512 tokens.

To check the host-memory path on an idle GPU **before loading the server**:

```bash
docker run --rm --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v "$PWD/runs:/runs" deepseek-v41-h100:independent \
  python3 /opt/experiment/tests/host_kv.py
```

[Independent package verification](results/independent-package-validation.json)
records the build, source hashes, fixture reproduction, and command checks.
Packaging does not rerun the two-hour benchmark or reload the serving model.

## Earlier profiles

[measurements.json](results/measurements.json) and
[validation.json](results/validation.json) summarize recorded runs. They describe
the earlier 4K, single-conversation 400K cached, 1M, and A100 configurations.
The 400K speed measurements precede the parser-only fix; its GPU sources and
settings were unchanged.

[Package verification](results/package-validation.json): both images built and
passed source-hash checks; the 400K and 1M fixtures matched their recorded hashes.
The shipped parser, chat, tool, and image checks passed. API tests targeted the
existing deployment with identical GPU files; packaging did not reload the model
or rerun the GPU primitive benchmarks.

## Checks without loading weights

The Docker build checks the original files, applies the selected patch without
adjusting mismatched lines, and checks that the result matches the tested code.

```bash
docker run --rm deepseek-v41-h100:cached python3 /opt/experiment/apply.py --check
docker run --rm deepseek-v41-h100:cached python3 /opt/experiment/tests/tool_parser.py
docker run --rm deepseek-v41-h100:dspark python3 /opt/experiment/apply.py --profile dspark --check
```

The parser test tries 916 ways of splitting four example responses into chunks.
It covers parallel calls, typed arguments, JSON bodies, and zero-argument calls.
These are parser tests, not 916 model requests.

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

These compare attention values, index scores, selected keys, and CUDA graph
outputs, including 1M-entry tables. They test the kernels, not model quality.

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

One generated Python function passed its 10 tests and 2,000 additional input
cases. To check a newly generated sample in an isolated container:

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

- One earlier score comparison differed by 7.6e-6. The selected keys matched.
- Split-K changes the order of floating-point sums. Kernel results did not
  always match exactly. The 128-token model comparison matched when both runs
  used the same conversation and cache state. A comparison with different cache
  states failed.
- The first high-resolution image at 400K exhausted a temporary mask allocation.
  Preallocated masks and shared slot reads fixed it; the subsequent test passed.
- Tool tests passed separately, including a 400,520-token tool call. A later
  combined cached tool-result/image test was interrupted by live traffic and is
  **not** counted as passed. The 1M profile's large-image path is unqualified.

The separate A100 runtime passed seven API checks and 36 exact comparisons
against its own baseline. It is not a supported target of these launch scripts.

# DeepSeek-V4.1-Flash on 4×H100

An experiment for fun and learning. Native weights, all target experts, image
input, and tool calling. Engram tables live in host RAM.

## 30 independent 400K conversations

On four H100 80 GB GPUs, with each conversation already cached:

| Profile | Workload | Output per request | Aggregate generation | Aggregate including admission |
| --- | --- | ---: | ---: | ---: |
| `independent` | Counting | 512 tokens | 834 tokens/s | 543 tokens/s |
| `dspark` | Counting | 512 tokens | 1,125 tokens/s | 549 tokens/s |
| `independent` | Prose | 512 tokens | 729 tokens/s | 469 tokens/s |
| `dspark` | Prose | 512 tokens | 1,018 tokens/s | 395 tokens/s |
| `dspark` | Continued prose | 1,024 tokens | 1,154 tokens/s | 664 tokens/s |

Generation rates cover the interval when all 30 streams were producing output.
The last column covers the entire wave, including admission and completion.
These are pretokenized `/generate` tests from September 14–15, not a guarantee
for every response. Divide by 30 for the average per conversation.

DSpark improved steady decoding. Whole-wave counting was roughly unchanged;
short prose was slower. Cold 400K fill took about **3.5 minutes** for `independent`
and **7.7 minutes** for `dspark`. Cold loading is excluded from the table.

A separate `dspark` Chat Completions test measured **438 aggregate tokens/s**:
30 cached histories, 512 output tokens each, with full text input and client/API
overhead included. It used one Python process with 30 threads, not an agent harness.

Both profiles passed retrieval and cache checks for all 30 histories, plus tools
and image checks with the cache populated.
**The 3,000 aggregate tokens/s target was not reached.**
[Ordinary results](results/independent.json) · [DSpark results](results/dspark.json) · [Tests](TESTING.md)

Both profiles put the main conversation KV cache in host RAM and copy
selected entries to GPU for attention. Index keys, target sliding-window state,
and experts stay on GPU. The configured KV pool adds about 71 GiB of host
RAM across four GPUs, separate from about 189 GiB for Engram. No experts or
attention entries are removed. It saves GPU memory; transfers still take time.
DSpark also stores input embeddings, RoPE tables, image MLP weights and draft
sliding-window state in host RAM. Its text output head stays on GPU.

## Run

Requires 4×H100 80 GB with NVLink, Docker with NVIDIA GPU support, and a
CUDA 13-compatible driver. Tested with about 885 GiB of system RAM. Minimum RAM
has not been measured; loading needs more than the resident pools alone.
Weights use about 475 GiB of disk.

```bash
docker build --build-arg PROFILE=dspark -t deepseek-v41-h100:dspark .
mkdir -p weights runs
docker run --rm -v "$PWD/weights:/weights" deepseek-v41-h100:dspark \
  python3 /opt/experiment/download.py
./serve.sh dspark "$PWD/weights"
```

For ordinary decoding, replace `dspark` with `independent` in those commands.
API: `http://localhost:30000/v1`, model: `/model`. The port binds to loopback.
The 409,600-token limit includes the prompt, history, images, and output.
Both profiles allow 32 active requests and reserve 13,107,200 cache tokens.
`dspark` proposes three tokens per step; `independent` disables speculation.
Restarting clears the cache.
Model and container versions are pinned in [versions.json](versions.json).

## Earlier single-conversation results

These use separate profiles, with one active request:

| Profile | Test | Input | Generation speed |
| --- | --- | --- | ---: |
| `cached` | Python function with tests | ~400K cached | 203 tokens/s |
| `cached` | Three prose samples | ~400K cached | 107–130 tokens/s |
| `long` | Retrieval of three planted codes | 1,048,000 tokens | 16.94 tokens/s |

Those profiles took 8.3 minutes at cold 400K and 57 minutes at cold 1M before
generation started. Speeds exclude that wait and EOS. They describe these
samples, not every response. [Recorded runs](results/measurements.json).

```bash
docker build --build-arg PROFILE=cached -t deepseek-v41-h100:cached .
./serve.sh cached "$PWD/weights"
```

For 1M, substitute `long` for `cached` in both commands. The `cached` profile
uses DSpark; `long` disables it and has text-only validation at 1M.
Run one profile at a time. `./serve.sh short "$PWD/weights"` selects the earlier
4K profile using the pinned upstream image.

A separate runtime on 8×A100 40 GB reached 14.77 tokens/s. Its results are
included; that runtime is not.

Built on [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[SGLang](https://github.com/sgl-project/sglang).

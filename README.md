# DeepSeek-V4.1-Flash on 4×H100

An experiment for fun and learning. Native weights, all target experts, image
input, and tool calling. Engram tables live in host RAM.

## 30 independent 400K conversations

On four H100 80 GB GPUs, with each conversation already cached:

| Workload | Aggregate generation | Average per conversation | Aggregate including admission |
| --- | ---: | ---: | ---: |
| Counting | 834 tokens/s | 27.8 tokens/s | 543 tokens/s |
| Varied prose | 729 tokens/s | 24.3 tokens/s | 469 tokens/s |

Each request generated 512 tokens. Generation rates cover the interval when all
30 streams were producing output. Including admission, the effective average was
**16–18 tokens/s per conversation**. First tokens arrived in 1.2–11.7 seconds.
These were pretokenized `/generate` requests, not a client application benchmark.

All 30 histories passed retrieval and cache checks. A cold 400K history took about
3.5 minutes to process; initial cache loading is excluded from the table.
**The 3,000 aggregate tokens/s target was not reached.**
[Measurements and checks](results/independent.json) · [Test details](TESTING.md)

This profile also puts the main conversation KV cache in host RAM and copies
selected entries to GPU for attention. Index keys, sliding-window state, and
compute weights stay on GPU. The configured KV pool adds about 71 GiB of host
RAM across four GPUs, separate from about 189 GiB for Engram. No experts or
attention entries are removed. It saves GPU memory; transfers still take time.

## Run

Requires 4×H100 80 GB with NVLink, Docker with NVIDIA GPU support, and a
CUDA 13-compatible driver. Tested with about 885 GiB of system RAM. Minimum RAM
has not been measured; loading needs more than the resident pools alone.
Weights use about 475 GiB of disk.

```bash
docker build --build-arg PROFILE=independent -t deepseek-v41-h100:independent .
mkdir -p weights runs
docker run --rm -v "$PWD/weights:/weights" deepseek-v41-h100:independent \
  python3 /opt/experiment/download.py
./serve.sh independent "$PWD/weights"
```

API: `http://localhost:30000/v1`, model: `/model`. The port binds to loopback.
The 409,600-token limit includes the prompt, history, images, and output.
The profile allows 32 active requests and reserves 13,107,200 cache tokens.
It uses ordinary decoding, with DSpark disabled. Restarting clears the cache.
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

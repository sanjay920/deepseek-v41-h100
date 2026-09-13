# DeepSeek-V4.1-Flash on 4×H100

An inference experiment for fun and learning. Native weights, all target experts
retained, Engram tables in host RAM. Includes image input and OpenAI tool calling.

Recorded on 4×H100 80 GB, one active request:

| Workload | Context | Generated tokens/s |
| --- | --- | ---: |
| Python | ~400K cached | 203 |
| Explanations | ~400K cached | 114–130 |
| Creative writing | ~400K cached | 107 |
| Three-passcode retrieval | 1,048,000 input tokens | 16.94 |

These rates start after the first text chunk and exclude EOS. **Cold prefill
took 8.3 minutes at 400K and 57 minutes at 1M.** A cache miss, changed prefix,
or queued request can still mean a long wait. Speeds vary with the response.
[Measurements and conditions](results/measurements.json).

## Run

Requires four H100 80 GB GPUs connected by NVLink, Docker with NVIDIA GPU support,
and a driver compatible with the image's CUDA 13 runtime. The checkpoint uses
about 475 GiB of disk. Engram alone occupies about 189 GiB of host RAM; loading
needs additional memory. A minimum host-RAM configuration was not established.

```bash
docker build --build-arg PROFILE=cached -t deepseek-v41-h100:cached .
mkdir -p weights runs
docker run --rm -v "$PWD/weights:/weights" deepseek-v41-h100:cached \
  python3 /opt/experiment/download.py
./serve.sh cached "$PWD/weights"
```

API: `http://localhost:30000/v1`, model: `/model`. The port binds to loopback.
The **409,600-token limit includes prompt, history, images, and output**.
The launcher enables `deepseekv41` tool parsing and `deepseek-v41` reasoning
parsing; without the tool parser, native DSML can appear as ordinary text.

| Profile | Total context limit | DSpark | Container image |
| --- | ---: | --- | --- |
| `cached` | 409,600 | Yes | `deepseek-v41-h100:cached` |
| `long` | 1,048,576 | No | `deepseek-v41-h100:long` |
| `short` | 4,096 | Yes | Pinned upstream image |

Build the earlier 1M profile separately:

```bash
docker build --build-arg PROFILE=long -t deepseek-v41-h100:long .
./serve.sh long "$PWD/weights"
```

`./serve.sh short "$PWD/weights"` uses the unmodified GPU runtime. Its earlier
short-prompt measurements were 217–262 tokens/s; those are separate from the
400K measurements above. Model and container revisions are pinned in
[versions.json](versions.json).

## What changed

The patches bound prefill buffers and gather only selected KV entries. The
cached profile also sizes communication buffers for 256-token chunks, selects
native Hopper split-K tiles for small dense operations, skips index tiles
excluded by the model's candidate mask, and removes duplicate prefill allocations.
The build verifies every patched file against the tested source hashes.

Seven basic API checks and a 400K cached conversation passed; a 1920×1080 image
with that history also passed. Tool calling was checked separately. Matched-cache comparisons reproduced
128 token probabilities and their top-10 entries exactly. Some primitive tests
were not bit-exact; this is not a claim of universal model parity or vision quality.
[Tests, commands, and limitations](TESTING.md).

A separate adapted runtime on 8×A100 40 GB reached 14.77 tokens/s. Its results
are recorded here; that runtime is not included.

Built on [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[SGLang](https://github.com/sgl-project/sglang).

# DeepSeek-V4.1-Flash on 4×H100

An experiment for fun and learning. Uses the native weights and all target
experts, with Engram tables in host RAM. Image input and tool calling are enabled.

## Measured results

One request at a time on four H100 80 GB GPUs:

| Test | Input | Generation speed |
| --- | --- | ---: |
| One Python function with tests | ~400K cached tokens | 203 tokens/s |
| Three prose samples | ~400K cached tokens | 107–130 tokens/s |
| Retrieval of three planted codes | 1,048,000 tokens | 16.94 tokens/s |

**Uncached input took 8.3 minutes at 400K and 57 minutes at 1M before generation
started.** The speeds above exclude that wait and the end-of-sequence token.
They are measurements of these samples, not a guarantee for other requests.
[Prompts, output lengths, temperatures, and timings](results/measurements.json).

## Run

Requires 4×H100 80 GB with NVLink, Docker with NVIDIA GPU support, and a
CUDA 13-compatible driver. The weights use about 475 GiB of disk. Engram uses
about 189 GiB of host RAM, with more needed during loading. Minimum RAM has
not been measured.

```bash
docker build --build-arg PROFILE=cached -t deepseek-v41-h100:cached .
mkdir -p weights runs
docker run --rm -v "$PWD/weights:/weights" deepseek-v41-h100:cached \
  python3 /opt/experiment/download.py
./serve.sh cached "$PWD/weights"
```

API: `http://localhost:30000/v1`, model: `/model`. The port binds to loopback.
The 409,600-token limit includes the prompt, history, images, and output.
Model and container versions are pinned in [versions.json](versions.json).

For 1M context:

```bash
docker build --build-arg PROFILE=long -t deepseek-v41-h100:long .
./serve.sh long "$PWD/weights"
```

The 1M profile disables DSpark to make room for the cache; its validation was
text-only. The cached profile keeps DSpark enabled. `./serve.sh short "$PWD/weights"`
runs the earlier 4K profile using the pinned upstream image.

## Changes and checks

The patches reduce temporary memory use, tune Hopper dense kernels, and skip
scoring attention blocks the model has already excluded. The launcher also
enables the V4.1 tool and reasoning parsers.

A 400K cached conversation and a 1920×1080 image with that history passed their
checks. Tool calling was tested separately. On two fixed prompts, 128 generated
tokens and their top-10 probabilities matched the previous working runtime.
These are limited checks, not a model-quality evaluation.
[Tests, commands, and known limits](TESTING.md).

A separate runtime on 8×A100 40 GB reached 14.77 tokens/s. Its results are
included; that runtime is not.

Built on [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[SGLang](https://github.com/sgl-project/sglang).

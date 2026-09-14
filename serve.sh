#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: $0 cached|independent|long|short PATH_TO_WEIGHTS" >&2
  exit 2
fi
base_image=lmsysorg/sglang@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860
requests=1
graphs=(--cuda-graph-max-bs-decode 1)
runtime_env=(-e HF_HUB_OFFLINE=1)
case "$1" in
  independent)
    image=deepseek-v41-h100:independent
    context=409600; capacity=13107200; chunk=1024; memory=0.985
    requests=32
    graphs=(--cuda-graph-bs-decode 1 2 4 8 16 24 32)
    extra=(--disable-flashinfer-autotune --swa-prefix-tails 16 --enable-metrics
      --enable-session-radix-cache --random-seed 73599507
      --enforce-disable-flashinfer-allreduce-fusion)
    runtime_env+=(-e DSV41_HOST_KV=1 -e DSV41_PREFER_FINISHED_CACHE=1
      -e PYTORCH_ALLOC_CONF=expandable_segments:True
      -e SGLANG_MEMORY_SAVER_CUDA_GRAPH=1 -e SGLANG_SWA_EVICTION_INTERVAL=16
      -e NCCL_BUFFSIZE=1048576 -e NCCL_MAX_CTAS=8)
    ;;
  cached)
    image=deepseek-v41-h100:cached
    context=409600; capacity=413696; chunk=256; memory=0.985
    extra=(--disable-flashinfer-autotune --speculative-algorithm DSPARK --speculative-dspark-block-size 5)
    ;;
  long)
    image=deepseek-v41-h100:long
    context=1048576; capacity=1052672; chunk=256; memory=0.98
    extra=(--disable-flashinfer-autotune)
    ;;
  short)
    image=$base_image
    context=4096; capacity=8192; chunk=512; memory=0.985
    extra=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
    ;;
  *) echo "Choose cached, independent, long, or short." >&2; exit 2 ;;
esac
model_dir=$(cd "$2" && pwd)
test -f "$model_dir/model.safetensors.index.json"
if [[ "$1" != short ]]; then
  actual_profile=$(docker image inspect --format '{{ index .Config.Labels "deepseek-v41.profile" }}' "$image")
  if [[ "$actual_profile" != "$1" ]]; then
    echo "Image profile mismatch. Build with --build-arg PROFILE=$1 -t $image ." >&2
    exit 2
  fi
fi
repo_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$repo_dir/.cache/sglang" "$repo_dir/.cache/tilelang" "$repo_dir/runs"
shadow=0
if [[ ${VERIFY_INDEXER:-0} == 1 ]]; then shadow=131072; fi

exec docker run --rm --name deepseek-v41 --gpus all --ipc=host --ulimit memlock=-1 \
  -p 127.0.0.1:30000:30000 \
  -v "$model_dir:/model:ro" \
  -v "$repo_dir/.cache/sglang:/root/.cache" \
  -v "$repo_dir/.cache/tilelang:/root/.tilelang/cache" \
  -v "$repo_dir/runs:/runs" \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
  -e DSV41_PREFILL_SHADOW_MAX_KEYS="$shadow" "${runtime_env[@]}" \
  "$image" sglang serve --model-path /model --trust-remote-code \
  --tp 4 --ep-size 4 --context-length "$context" --max-total-tokens "$capacity" \
  --chunked-prefill-size "$chunk" --mem-fraction-static "$memory" \
  --max-running-requests "$requests" "${graphs[@]}" \
  --enable-cache-report --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 \
  --host 0.0.0.0 --port 30000 "${extra[@]}"

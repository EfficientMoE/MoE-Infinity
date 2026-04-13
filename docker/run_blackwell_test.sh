#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="moe-infinity-blackwell"
GPU_ID="${GPU_ID:-0}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
OFFLOAD_DIR="${OFFLOAD_DIR:-/tmp/moe-offload-blackwell}"

echo "=== MoE-Infinity Blackwell MXFP4 Test ==="
echo "GPU:            ${GPU_ID}"
echo "HF cache:       ${HF_CACHE}"
echo "Offload dir:    ${OFFLOAD_DIR}"
echo ""

mkdir -p "${OFFLOAD_DIR}" "${HF_CACHE}" "$(pwd)/results"

# ---------- Build ----------
echo "Building Docker image (first build ~20 min, subsequent <2 min)..."
DOCKER_BUILDKIT=1 docker build \
    -t "${IMAGE_NAME}" \
    -f docker/Dockerfile.blackwell \
    .

# ---------- Unit Tests ----------
echo ""
echo "=== Running MXFP4 kernel unit tests ==="
docker run --rm \
    --gpus "\"device=${GPU_ID}\"" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${IMAGE_NAME}"

# ---------- Benchmark (optional) ----------
if [[ "${RUN_BENCHMARK:-0}" == "1" ]]; then
    echo ""
    echo "=== Running MXFP4 benchmark ==="
    docker run --rm \
        --gpus "\"device=${GPU_ID}\"" \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=16g \
        -v "${OFFLOAD_DIR}:/offload" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "$(pwd)/results:/workspace/MoE-Infinity/results" \
        -e CUDA_VISIBLE_DEVICES=0 \
        "${IMAGE_NAME}" \
        python benchmarks/mxfp4_benchmark.py \
            --model openai/gpt-oss-20b \
            --mode both \
            --offload-dir /offload \
            --num-requests 10 \
            --output-json results/blackwell_mxfp4_benchmark.json

    echo ""
    echo "Results saved to results/blackwell_mxfp4_benchmark.json"
    cat results/blackwell_mxfp4_benchmark.json 2>/dev/null || true
fi

echo ""
echo "=== Done ==="

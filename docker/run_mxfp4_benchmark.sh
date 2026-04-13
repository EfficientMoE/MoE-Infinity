#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="moe-infinity-mxfp4-bench"
OFFLOAD_DIR="${OFFLOAD_DIR:-/tmp/moe-offload-bench}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
GPU_ID="${GPU_ID:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"

echo "=== MoE-Infinity MXFP4 Benchmark ==="
echo "GPU:            ${GPU_ID}"
echo "Offload dir:    ${OFFLOAD_DIR}"
echo "HF cache:       ${HF_CACHE}"
echo "Num requests:   ${NUM_REQUESTS}"
echo ""

mkdir -p "${OFFLOAD_DIR}" "${HF_CACHE}"

echo "Building Docker image (this takes ~15-20 min first time)..."
DOCKER_BUILDKIT=1 docker build \
    -t "${IMAGE_NAME}" \
    -f docker/Dockerfile.benchmark \
    .

echo ""
echo "Running benchmark..."
docker run --rm \
    --gpus "\"device=${GPU_ID}\"" \
    --shm-size=16g \
    -v "${OFFLOAD_DIR}:/offload" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "$(pwd)/results:/workspace/MoE-Infinity/results" \
    -e CUDA_VISIBLE_DEVICES=0 \
    "${IMAGE_NAME}" \
    --model openai/gpt-oss-20b \
    --mode both \
    --offload-dir /offload \
    --num-requests "${NUM_REQUESTS}" \
    --output-json results/mxfp4_benchmark.json

echo ""
echo "Results saved to results/mxfp4_benchmark.json"
cat results/mxfp4_benchmark.json 2>/dev/null || true

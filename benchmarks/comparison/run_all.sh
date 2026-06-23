#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OFFLOAD_DIR=""
RESULTS_DIR="${SCRIPT_DIR}/results"
SKIP_BUILD=false
ONLY=""

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --offload-dir DIR   Directory for MoE-Infinity expert offload storage (required for MoE-Infinity)"
    echo "  --results-dir DIR   Directory for JSON results (default: benchmarks/comparison/results/)"
    echo "  --skip-build        Skip Docker image builds"
    echo "  --only FRAMEWORK    Run only one framework: moe_infinity|vllm|llamacpp"
    echo "  --help              Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offload-dir) OFFLOAD_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
HF_CACHE="${HOME}/.cache/huggingface"
mkdir -p "${HF_CACHE}"

if [[ "${SKIP_BUILD}" == "false" ]]; then
    echo "[run_all] Building Docker images..."
    docker build -t moe-infinity-bench -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"
    docker build -t bench-vllm -f "${SCRIPT_DIR}/Dockerfile.vllm" "${REPO_ROOT}"
    docker build -t bench-llamacpp -f "${SCRIPT_DIR}/Dockerfile.llamacpp" "${REPO_ROOT}"
fi

run_benchmark() {
    local framework="$1"
    shift
    local exit_code=0
    "$@" || exit_code=$?
    if [[ ${exit_code} -eq 0 ]] || [[ ${exit_code} -eq 2 ]] || [[ ${exit_code} -eq 3 ]] || [[ ${exit_code} -eq 4 ]]; then
        echo "[run_all] ${framework} finished with expected exit code ${exit_code}"
    else
        echo "[run_all] WARNING: ${framework} exited unexpectedly with code ${exit_code}" >&2
    fi
}

if [[ -z "${ONLY}" ]] || [[ "${ONLY}" == "moe_infinity" ]]; then
    echo "[run_all] Running MoE-Infinity benchmark..."
    run_benchmark "moe_infinity" docker run \
        --gpus "device=0" \
        --rm \
        -v "${RESULTS_DIR}:/results" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${OFFLOAD_DIR:-/tmp/moe-offload}:/offload" \
        moe-infinity-bench \
        python3 benchmarks/comparison/run_moe_infinity.py \
            --offload-dir /offload \
            --output-dir /results
    echo "[run_all] Cooldown 60s (thermal management)..."
    sleep 60
fi

if [[ -z "${ONLY}" ]] || [[ "${ONLY}" == "vllm" ]]; then
    echo "[run_all] Running vLLM benchmark..."
    run_benchmark "vllm" docker run \
        --gpus "device=0" \
        --rm \
        -v "${RESULTS_DIR}:/results" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        bench-vllm \
        python3 /workspace/run_vllm.py \
            --output-dir /results
    echo "[run_all] Cooldown 60s (thermal management)..."
    sleep 60
fi

if [[ -z "${ONLY}" ]] || [[ "${ONLY}" == "llamacpp" ]]; then
    echo "[run_all] Running llama.cpp benchmark..."
    run_benchmark "llamacpp" docker run \
        --gpus "device=0" \
        --rm \
        -v "${RESULTS_DIR}:/results" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        bench-llamacpp \
        python3 /workspace/run_llamacpp.py \
            --output-dir /results
fi

echo "[run_all] Aggregating results..."
python3 "${SCRIPT_DIR}/aggregate_results.py" \
    --results-dir "${RESULTS_DIR}" \
    --format markdown

echo "[run_all] Done. Table saved to ${RESULTS_DIR}/comparison_table.md"

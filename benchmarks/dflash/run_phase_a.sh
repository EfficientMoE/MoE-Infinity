#!/usr/bin/env bash
# ===========================================================================
# run_phase_a.sh -- one-command PD-DFlash Phase-A "measure-first" matrix.
#
# Runs the §8 B0-B3 route-ahead serving experiment on ONE RTX PRO 6000
# (sm_120, capability 12.0) for both required MoE targets with FP4-offloaded
# experts, then aggregates the raw rows into a result-matrix JSON that
# benchmarks.dflash.report / validate_result_matrix consumes. This is the
# hardware harness for plan Task 3 (docs/superpowers/plans/
# 2026-08-14-pd-dflash-serving-scheduler.md); B2 is emitted BLOCKED until the
# 2-D scheduler (Task 6) lands, so it is passed via --allow-blocked B2.
#
# USAGE
#   benchmarks/dflash/run_phase_a.sh
#
# All inputs are environment variables (shown with their defaults). Override
# any of them inline, e.g.:
#   QWEN_OFFLOAD=/data/qwen-fp4 MEASURED_H2D_GBPS=48 \
#     benchmarks/dflash/run_phase_a.sh
#
# REQUIRED on the GPU box (defaults assume this project's conventions):
#   HF_HOME                cached checkpoints (default /mnt/raid0nvme0/public/huggingface)
#   CUDA_VISIBLE_DEVICES   the single RTX PRO 6000 to use (default 0)
#   QWEN_OFFLOAD           dir of FP4-offloaded Qwen3-Coder-30B-A3B experts
#   GPTOSS_OFFLOAD         dir of FP4-offloaded gpt-oss-20b experts (needs #137)
#
# KEY KNOBS
#   DEVICE_MEMORY_RATIO    weight-resident fraction; MUST be < 0.9 to force
#                          offload for B0/B1/B2 (default 0.85)
#   MEASURED_H2D_GBPS      measured host->GPU expert bandwidth (GB/s) for the
#                          hide inequality; NOT a theoretical PCIe number
#   SLO_MS                 per-round SLO for goodput@SLO (optional)
#   BASELINES              default "B0 B1 B2 B3"
#   BLOCK_SIZES            default "8 16"
#   CONCURRENCY            default "1 2 4 8 16 32"
#   REQUESTS / WARMUP / SEED   default 64 / 5 / 1408
#   PD_DFLASH_BUILD=1      rebuild the native sm_120 extensions first
#                          (MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0)
#
# OUTPUTS (under $OUTPUT_DIR, default /tmp/pd-dflash-results)
#   raw/qwen.json, raw/gpt-oss.json   one JSON row per (model,baseline,B,c,repeat)
#   result_matrix.json                grouped {model|B|c: {baseline: row}}
#   summary.csv, summary.md           human-readable aggregation
# ===========================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/raid0nvme0/public/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MOE_ENABLE_SM120="${MOE_ENABLE_SM120:-1}"
# The runner validates the device itself; this mirrors the pytest gate name so
# any nested gated assertions also run on the box.
export MOE_DFLASH_SERVING_GPU="${MOE_DFLASH_SERVING_GPU:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pd-dflash-results}"
BASELINES="${BASELINES:-B0 B1 B2 B3}"
BLOCK_SIZES="${BLOCK_SIZES:-8 16}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
REQUESTS="${REQUESTS:-64}"
WARMUP="${WARMUP:-5}"
SEED="${SEED:-1408}"
DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.85}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Coder-30B-A3B}"
QWEN_DRAFT="${QWEN_DRAFT:-z-lab/Qwen3-Coder-30B-A3B-DFlash}"
QWEN_OFFLOAD="${QWEN_OFFLOAD:-/mnt/raid0nvme0/offload/qwen3-coder-30b-a3b-fp4}"

GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-20b}"
GPTOSS_DRAFT="${GPTOSS_DRAFT:-z-lab/gpt-oss-20b-DFlash}"
GPTOSS_OFFLOAD="${GPTOSS_OFFLOAD:-/mnt/raid0nvme0/offload/gpt-oss-20b-fp4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR/raw"

echo "[run_phase_a] repo=$REPO_ROOT out=$OUTPUT_DIR device=$CUDA_VISIBLE_DEVICES"
echo "[run_phase_a] device_memory_ratio=$DEVICE_MEMORY_RATIO (must be < 0.9 to offload)"

if awk "BEGIN{exit !($DEVICE_MEMORY_RATIO >= 0.9)}"; then
  echo "[run_phase_a] ERROR: DEVICE_MEMORY_RATIO=$DEVICE_MEMORY_RATIO >= 0.9 will not offload B0/B1/B2" >&2
  exit 2
fi

if [[ "${PD_DFLASH_BUILD:-0}" == "1" ]]; then
  echo "[run_phase_a] building native sm_120 extensions"
  MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}" \
    pip install --no-build-isolation -e .
fi

extra_args=()
if [[ -n "${MEASURED_H2D_GBPS:-}" ]]; then
  extra_args+=(--measured-h2d-gbps "$MEASURED_H2D_GBPS")
fi
if [[ -n "${SLO_MS:-}" ]]; then
  extra_args+=(--slo-ms "$SLO_MS")
fi

run_model() {
  local model="$1" draft="$2" offload="$3" output="$4"
  echo "[run_phase_a] === $model -> $output ==="
  if [[ ! -d "$offload" ]]; then
    echo "[run_phase_a] WARNING: offload dir '$offload' missing; the runner will" \
         "refuse B0/B1/B2 unless experts are genuinely offloaded" >&2
  fi
  python -m benchmarks.dflash.pd_dflash_serving \
    --model "$model" \
    --draft "$draft" \
    --offload-dir "$offload" \
    --baseline $BASELINES \
    --block-size $BLOCK_SIZES \
    --concurrency $CONCURRENCY \
    --requests "$REQUESTS" \
    --warmup-rounds "$WARMUP" \
    --seed "$SEED" \
    --device-memory-ratio "$DEVICE_MEMORY_RATIO" \
    "${extra_args[@]}" \
    --output "$output"
}

run_model "$QWEN_MODEL" "$QWEN_DRAFT" "$QWEN_OFFLOAD" "$OUTPUT_DIR/raw/qwen.json"
run_model "$GPTOSS_MODEL" "$GPTOSS_DRAFT" "$GPTOSS_OFFLOAD" "$OUTPUT_DIR/raw/gpt-oss.json"

echo "[run_phase_a] aggregating result matrix"
python -m benchmarks.dflash.report \
  --input "$OUTPUT_DIR/raw/qwen.json" "$OUTPUT_DIR/raw/gpt-oss.json" \
  --matrix-json "$OUTPUT_DIR/result_matrix.json" \
  --csv "$OUTPUT_DIR/summary.csv" \
  --markdown "$OUTPUT_DIR/summary.md" \
  --allow-blocked B2 || {
    echo "[run_phase_a] report gate FAILED (missing/invalid baseline); inspect" \
         "$OUTPUT_DIR/result_matrix.json" >&2
    exit 1
  }

echo "[run_phase_a] done:"
echo "  raw rows:      $OUTPUT_DIR/raw/{qwen,gpt-oss}.json"
echo "  result matrix: $OUTPUT_DIR/result_matrix.json"
echo "  summary:       $OUTPUT_DIR/summary.{csv,md}"

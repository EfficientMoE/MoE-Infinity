#!/usr/bin/env bash
# ===========================================================================
# run_phase_c.sh -- one-command PD-DFlash Phase-C benchmark-gated C++ hops.
#
# Runs BM3 (route-ahead priority-band ablation), BM4 (expert-H2D / compute
# overlap trace via nsys), and BM5 (end-to-end Python-issue vs shipped
# C++-issue) on ONE RTX PRO 6000 (sm_120, capability 12.0) for both required
# MoE targets with FP4-offloaded experts, then aggregates the final keep/remove
# verdict per C++ hop. This is the hardware harness for plan Tasks 9-10
# (docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md).
#
# The plan's hard rule: no C++ change ships without its paired BM passing on
# BOTH targets. BM3 gates the route-ahead priority band; BM4 supplies the
# overlap ground truth; BM5 proves C++ issue does not change correctness.
#
# USAGE
#   benchmarks/dflash/run_phase_c.sh
#
# All inputs are environment variables (defaults assume this project's layout);
# override inline, e.g.:
#   QWEN_OFFLOAD=/data/qwen-fp4 benchmarks/dflash/run_phase_c.sh
#
# REQUIRED on the GPU box:
#   HF_HOME                cached checkpoints (default /mnt/raid0nvme0/public/huggingface)
#   CUDA_VISIBLE_DEVICES   the single RTX PRO 6000 to use (default 0)
#   QWEN_OFFLOAD           dir of FP4-offloaded Qwen3-Coder-30B-A3B experts
#   GPTOSS_OFFLOAD         dir of FP4-offloaded gpt-oss-20b experts (needs #137)
#
# KEY KNOBS
#   DEVICE_MEMORY_RATIO    weight-resident fraction; MUST be < 0.9 to offload
#   REPETITIONS            BM3 ablation reps per arm (default 10)
#   BLOCK_SIZE             draft block size for BM3/BM4/BM5 (default 16)
#   REQUESTS               requests per measurement (default 16; short by design)
#   PD_DFLASH_BUILD=1      rebuild the native sm_120 extensions first
#
# OUTPUTS (under $OUTPUT_DIR, default /tmp/pd-dflash-results)
#   bm3.json               three-way priority ablation + ship verdict
#   nsys/qwen-final.nsys-rep, nsys/gptoss-final.nsys-rep
#   bm4-qwen.json, bm4-gptoss.json     overlap fraction ground truth
#   bm5-python.json, bm5-cpp.json      end-to-end Python vs C++ issue
#   final.csv, final.md                aggregated §8 + BM1-BM5 + hop verdicts
# ===========================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/raid0nvme0/public/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MOE_ENABLE_SM120="${MOE_ENABLE_SM120:-1}"
export MOE_DFLASH_SERVING_GPU="${MOE_DFLASH_SERVING_GPU:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/pd-dflash-results}"
REPETITIONS="${REPETITIONS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
REQUESTS="${REQUESTS:-16}"
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
mkdir -p "$OUTPUT_DIR/raw" "$OUTPUT_DIR/nsys"

echo "[run_phase_c] repo=$REPO_ROOT out=$OUTPUT_DIR device=$CUDA_VISIBLE_DEVICES"

if awk "BEGIN{exit !($DEVICE_MEMORY_RATIO >= 0.9)}"; then
  echo "[run_phase_c] ERROR: DEVICE_MEMORY_RATIO=$DEVICE_MEMORY_RATIO >= 0.9 will not offload" >&2
  exit 2
fi

if [[ "${PD_DFLASH_BUILD:-0}" == "1" ]]; then
  echo "[run_phase_c] building native sm_120 extensions"
  MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}" \
    pip install --no-build-isolation -e .
fi

# --- BM3: three-way priority-band ablation on both targets -----------------
echo "[run_phase_c] === BM3 priority ablation ==="
python -m benchmarks.dflash.bench_prefetch_priority \
  --models "$QWEN_MODEL" "$GPTOSS_MODEL" \
  --drafts "$QWEN_DRAFT" "$GPTOSS_DRAFT" \
  --offload-dirs "$QWEN_OFFLOAD" "$GPTOSS_OFFLOAD" \
  --repetitions "$REPETITIONS" --block-size "$BLOCK_SIZE" \
  --requests "$REQUESTS" --warmup-rounds "$WARMUP" --seed "$SEED" \
  --device-memory-ratio "$DEVICE_MEMORY_RATIO" \
  --output "$OUTPUT_DIR/bm3.json"

# --- BM4: nsys overlap capture + parse for the OURS config -----------------
capture_bm4() {
  local model="$1" draft="$2" offload="$3" tag="$4"
  echo "[run_phase_c] === BM4 overlap capture: $model ==="
  nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true \
    --output="$OUTPUT_DIR/nsys/$tag-final" \
  python -m benchmarks.dflash.pd_dflash_serving \
    --model "$model" --draft "$draft" --offload-dir "$offload" \
    --baseline OURS --block-size "$BLOCK_SIZE" --concurrency 8 \
    --requests 32 --warmup-rounds "$WARMUP" --seed "$SEED" \
    --device-memory-ratio "$DEVICE_MEMORY_RATIO" \
    --output "$OUTPUT_DIR/raw/$tag-final.json"
  python -m benchmarks.dflash.parse_overlap \
    --rep "$OUTPUT_DIR/nsys/$tag-final.nsys-rep" \
    --output "$OUTPUT_DIR/bm4-$tag.json"
}
capture_bm4 "$QWEN_MODEL" "$QWEN_DRAFT" "$QWEN_OFFLOAD" "qwen"
capture_bm4 "$GPTOSS_MODEL" "$GPTOSS_DRAFT" "$GPTOSS_OFFLOAD" "gptoss"

# --- BM5: end-to-end Python-issue vs shipped C++-issue ---------------------
run_bm5() {
  local issue_mode="$1" output="$2"
  for pair in "$QWEN_MODEL|$QWEN_DRAFT|$QWEN_OFFLOAD" \
              "$GPTOSS_MODEL|$GPTOSS_DRAFT|$GPTOSS_OFFLOAD"; do
    IFS='|' read -r model draft offload <<<"$pair"
    python -m benchmarks.dflash.pd_dflash_serving \
      --model "$model" --draft "$draft" --offload-dir "$offload" \
      --baseline OURS --block-size "$BLOCK_SIZE" --concurrency 1 8 32 \
      --requests "$REQUESTS" --warmup-rounds "$WARMUP" --seed "$SEED" \
      --device-memory-ratio "$DEVICE_MEMORY_RATIO" \
      --output "$output" || true
  done
}
echo "[run_phase_c] === BM5 python-issue ==="
MOE_PREFETCH_ISSUE_MODE=python-per-expert run_bm5 python-per-expert \
  "$OUTPUT_DIR/bm5-python.json"
echo "[run_phase_c] === BM5 cpp-issue ==="
MOE_PREFETCH_ISSUE_MODE=cpp-batched run_bm5 cpp-batched \
  "$OUTPUT_DIR/bm5-cpp.json"

# --- final aggregation -----------------------------------------------------
echo "[run_phase_c] aggregating final gate report"
python -m benchmarks.dflash.report \
  --input "$OUTPUT_DIR/raw/qwen.json" "$OUTPUT_DIR/raw/gpt-oss.json" \
  --csv "$OUTPUT_DIR/final.csv" --markdown "$OUTPUT_DIR/final.md" || true

echo "[run_phase_c] done:"
echo "  BM3: $OUTPUT_DIR/bm3.json"
echo "  BM4: $OUTPUT_DIR/bm4-{qwen,gptoss}.json"
echo "  BM5: $OUTPUT_DIR/bm5-{python,cpp}.json"
echo "  final: $OUTPUT_DIR/final.{csv,md}"

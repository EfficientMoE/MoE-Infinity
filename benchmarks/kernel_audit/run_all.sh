#!/usr/bin/env bash
# Full sm_120 kernel-audit campaign on one GPU (the GPU must be otherwise idle).
#   GPU=0 bash benchmarks/kernel_audit/run_all.sh [stage...]
# Stages (default: all, in this order):
#   ceilings expert_time expert_ncu layer grouped graph real loop attn_time attn_ncu small busy autotune
# Env: PY (python, default .venv/bin/python), KAUDIT_REAL_TRACE (routing-trace
# npz for the `real` stage; skipped when unset), NCU / NCU_PY (Nsight Compute).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
R=benchmarks/kernel_audit/results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p "$R"
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(ceilings expert_time expert_ncu layer grouped graph real loop attn_time attn_ncu small busy autotune)
run() { echo "=== $* ($(date +%T))"; "$PY" -m benchmarks.kernel_audit.run_cases "$@" 2>&1 | grep -v Warning; }
fresh() { rm -f "$R/$1_time.jsonl"; }
for s in "${stages[@]}"; do
  case $s in
    ceilings) "$PY" -m benchmarks.kernel_audit.ceilings > "$R/ceilings.log" 2>&1 ;;
    expert_time) fresh expert; run --suite expert --mode time --iters 15 --out $R/expert_time.jsonl ;;
    expert_ncu) "$PY" -m benchmarks.kernel_audit.ncu_driver --suite expert --tag expert \
        --impl "^(cutlass_fused|fp8_prod|gptoss_resident|gptoss_offload|v4_fp4_prod|cublas_ref)$" \
        --exclude "mixtral-8x7b\|(50K|200K)" ;;
    layer) fresh expert_layer; run --suite expert_layer --mode time --out $R/expert_layer_time.jsonl ;;
    grouped) fresh expert_grouped; run --suite expert_grouped --mode time --out $R/expert_grouped_time.jsonl ;;
    graph) fresh expert_layer_graph; run --suite expert_layer_graph --mode time --out $R/expert_layer_graph_time.jsonl ;;
    real) fresh expert_real; run --suite expert_real --mode time --out $R/expert_real_time.jsonl ;;
    loop) fresh gptoss_loop; run --suite gptoss_loop --mode time --out $R/gptoss_loop_time.jsonl ;;
    attn_time)
      fresh attn_prefill; fresh attn_decode
      run --suite attn_prefill --mode time --out $R/attn_prefill_time.jsonl
      # The in-tree Triton decode kernel faults once the KV pool exceeds 2^31 elements
      # (Mixtral b>=256 @ 9K); skip those points.
      run --suite attn_decode --mode time --iters 10 --out $R/attn_decode_time.jsonl \
        --exclude "mixtral-8x7b\|b(256@(9K|12K)|64@50K|16@200K)\|triton" ;;
    attn_ncu)
      "$PY" -m benchmarks.kernel_audit.ncu_driver --suite attn_prefill --tag attn_prefill
      "$PY" -m benchmarks.kernel_audit.ncu_driver --suite attn_decode --tag attn_decode \
        --match "b(1|16|64)@(1K|9K|50K|200K)" --exclude "sdpa_fallback|paged_v1|mixtral" ;;
    small)
      for x in router gdn kv_write; do fresh $x; run --suite $x --mode time --iters 10 --out $R/${x}_time.jsonl; done ;;
    busy)
      rm -f $R/busy.jsonl
      run --suite attn_decode --mode busy --impl "sdpa_fallback|hf_eager_decode" --match "b(1|16)@(1K|9K)\|" --out $R/busy.jsonl
      run --suite attn_decode --mode busy --impl "paged_v1" --match "b1@1K" --out $R/busy.jsonl
      run --suite kv_write --mode busy --out $R/busy.jsonl
      run --suite gdn --mode busy --out $R/busy.jsonl
      run --suite router --mode busy --match "\|(b1|1K|50K)\|" --out $R/busy.jsonl
      run --suite gptoss_loop --mode busy --out $R/busy.jsonl ;;
    autotune) "$PY" -m benchmarks.kernel_audit.autotune_probe ;;
  esac
done
echo "=== done $(date +%T)"

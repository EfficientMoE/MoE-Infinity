#!/usr/bin/env bash
# Run each GLM MoE-building spec-decode test in its OWN process (one MoE per process).
set -u
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/raid0nvme0/public/huggingface/hub}
export MOE_GLM_TINY=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=.venv/bin/python3
rc=0
for t in test_glm_mtp test_glm_mtp_stats; do
  echo "=== $t ==="
  timeout 400 "$PY" -m pytest "tests/python/integration/$t.py" -q || rc=1
done
echo "=== dflash adapter (CPU) ==="
"$PY" -m pytest tests/python/unit/test_glm_dflash_adapter.py -q || rc=1
exit $rc

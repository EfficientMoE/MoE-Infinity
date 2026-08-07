import json
import os
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MOE_GLM_TINY"),
    reason="Set MOE_GLM_TINY=1 to run FP8 reload parity test",
)

_WORKER = """
import sys, os, json, torch
os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
ckpt, offload_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
from moe_infinity import MoE
m = MoE(ckpt, {"offload_path": offload_path, "device_memory_ratio": 0.8})
ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
toks = m.generate(ids, max_new_tokens=6)[0].tolist()
with open(out_file, "w") as f:
    json.dump({"tokens": toks}, f)
"""


def _generate_in_subprocess(ckpt, offload_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    try:
        result = subprocess.run(
            [sys.executable, "-c", _WORKER, ckpt, offload_path, out_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Worker failed:\n{result.stderr[-3000:]}")
        with open(out_file) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out_file)
        except OSError:
            pass


def test_fp8_reload_parity():
    from tests.python.integration._glm_tiny import build_tiny_glm_fp8

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_glm_fp8(
            os.path.join(tmp, "tiny"), quantize_shared=True
        )
        store = os.path.join(tmp, "store")

        fresh = _generate_in_subprocess(ckpt, store)
        reload = _generate_in_subprocess(ckpt, store)

        assert fresh["tokens"] == reload["tokens"], (
            f"fp8 reload-from-store diverged from fresh offload: "
            f"{reload['tokens']} vs {fresh['tokens']}"
        )
        assert len(fresh["tokens"]) > 4

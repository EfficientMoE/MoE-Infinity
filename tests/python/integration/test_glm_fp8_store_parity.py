import os
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MOE_GLM_TINY"),
    reason="Set MOE_GLM_TINY=1 to run FP8 store parity test",
)

_WORKER = """
import sys, os, json, torch
os.environ.setdefault("HF_HUB_CACHE", "/mnt/raid0nvme0/public/huggingface/hub")
ckpt, offload_path, fp8_in_store, out_file = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
from moe_infinity import MoE
config = {"offload_path": offload_path, "device_memory_ratio": 0.8, "glm_fp8_in_store": fp8_in_store}
m = MoE(ckpt, config)
ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
out = m.generate(ids, max_new_tokens=6)[0].tolist()
with open(out_file, "w") as f:
    json.dump(out, f)
"""


def _generate_in_subprocess(ckpt, offload_path, fp8_in_store):
    import json
    import tempfile

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    try:
        result = subprocess.run(
            [sys.executable, "-c", _WORKER, ckpt, offload_path, "1" if fp8_in_store else "0", out_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Worker failed (fp8_in_store={fp8_in_store}):\n{result.stderr[-3000:]}")
        with open(out_file) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out_file)
        except OSError:
            pass


def test_fp8_store_parity():
    from tests.python.integration._glm_tiny import build_tiny_glm_fp8

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_glm_fp8(os.path.join(tmp, "tiny"))

        out_a = _generate_in_subprocess(ckpt, os.path.join(tmp, "off_a"), False)
        out_b = _generate_in_subprocess(ckpt, os.path.join(tmp, "off_b"), True)

        assert out_a == out_b, f"Parity failed: Option A={out_a}, Option B={out_b}"

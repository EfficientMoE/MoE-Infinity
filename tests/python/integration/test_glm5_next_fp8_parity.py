import json
import os
import subprocess
import sys
import tempfile

import pytest

pytest.importorskip(
    "transformers.models.glm5_next",
    reason="transformers >= 5.16 with glm5_next required",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MOE_GLM5_NEXT_TINY") != "1",
    reason="Set MOE_GLM5_NEXT_TINY=1 to run glm5_next FP8 parity tests",
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
    json.dump(toks, f)
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
            timeout=600,
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


def test_glm5_next_fp8_store_parity():
    from tests.python.integration._glm5_next_tiny import (
        build_tiny_glm5_next_fp8,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_glm5_next_fp8(os.path.join(tmp, "tiny"))

        out_a = _generate_in_subprocess(ckpt, os.path.join(tmp, "off_a"))
        out_b = _generate_in_subprocess(ckpt, os.path.join(tmp, "off_b"))

        assert (
            out_a == out_b
        ), f"Independent fresh stores diverged: {out_a} vs {out_b}"


def test_glm5_next_fp8_reload_parity():
    from tests.python.integration._glm5_next_tiny import (
        build_tiny_glm5_next_fp8,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_glm5_next_fp8(os.path.join(tmp, "tiny"))
        store = os.path.join(tmp, "store")

        fresh = _generate_in_subprocess(ckpt, store)
        reloaded = _generate_in_subprocess(ckpt, store)

        assert (
            fresh == reloaded
        ), f"Reloaded store diverged from fresh store: {fresh} vs {reloaded}"

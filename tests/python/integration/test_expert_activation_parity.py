"""Offloaded MoE-Infinity vs HF eager logits for clamped-SwiGLU experts (#317).

Tiny random-weight MiniMax-M3 (SwiGLU-OAI) and GLM-5.3-Flash (clamped SiLU)
checkpoints whose routed-expert gate/up projections exceed ``swiglu_limit``.
The routed experts run through the native dispatcher (``MoEMLP`` ->
``fused_moe_ffn_into``); a plain-SiLU epilogue fails these tests.

GLM-5.3-Flash needs transformers >= 5.16 (``.venv-tf516``).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("MOE_EXPERT_ACT_PARITY") != "1",
        reason="set MOE_EXPERT_ACT_PARITY=1 to run the tiny-model GPU parity",
    ),
]

_REPO = Path(__file__).resolve().parents[3]

_INPUT_IDS = [[1, 17, 42, 99, 7, 256, 11, 5, 63, 128, 3, 77]]

_HF_WORKER = r"""
import sys, torch, importlib
import torch.nn.functional as F
from transformers import AutoConfig
ckpt, cls_path, plain_silu, out_path = sys.argv[1:]
mod_name, cls_name = cls_path.rsplit(".", 1)
mod = importlib.import_module(mod_name)
if plain_silu == "1":
    # What the native path computed before #317: silu(gate) * up.
    for obj in vars(mod).values():
        if isinstance(obj, type) and hasattr(obj, "_apply_gate"):
            def _plain(self, gate_up):
                gate, up = gate_up.chunk(2, dim=-1)
                return F.silu(gate) * up
            obj._apply_gate = _plain
model = getattr(mod, cls_name).from_pretrained(
    ckpt, dtype=torch.bfloat16, attn_implementation="eager"
).cuda().eval()
ids = torch.tensor(%s, device="cuda")
with torch.no_grad():
    torch.save(model(ids).logits.float().cpu(), out_path)
""" % (_INPUT_IDS,)

_MOE_WORKER = r"""
import sys, torch
from moe_infinity import MoE
ckpt, offload_path, out_path = sys.argv[1:]
model = MoE(ckpt, {"offload_path": offload_path, "device_memory_ratio": 0.8})
ids = torch.tensor(%s, device="cuda")
with torch.no_grad():
    logits = model(ids).logits.float().cpu()
torch.save(logits, out_path)
""" % (_INPUT_IDS,)


def _run(script, args, tmp):
    import torch

    out = os.path.join(tmp, f"logits_{abs(hash(tuple(args)))}.pt")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), env.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *args, out],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"worker failed:\n{result.stderr[-4000:]}")
    return torch.load(out)


def _check_parity(ckpt, cls_path, tmp):
    import torch

    ref = _run(_HF_WORKER, [ckpt, cls_path, "0"], tmp)
    plain = _run(_HF_WORKER, [ckpt, cls_path, "1"], tmp)
    got = _run(_MOE_WORKER, [ckpt, os.path.join(tmp, "offload")], tmp)

    assert got.shape == ref.shape
    assert torch.isfinite(got).all()

    def err(a):
        return ((a - ref).norm() / ref.norm()).item()

    bug_err, our_err = err(plain), err(got)
    print(
        f"rel-L2 vs HF eager: offloaded={our_err:.4f} plain-silu={bug_err:.4f}"
    )
    # The clamps matter at these magnitudes ...
    assert bug_err > 0.1, bug_err
    # ... and the offloaded path matches the HF activation (bf16 tolerance).
    assert our_err < 0.03, our_err
    assert our_err < bug_err / 5
    assert (got.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9


def test_minimax_m3_offloaded_logits_match_hf():
    if importlib.util.find_spec("transformers.models.minimax_m3_vl") is None:
        pytest.skip("transformers lacks minimax_m3_vl")
    from tests.python.integration._expert_activation_tiny import (
        build_tiny_minimax_m3,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_minimax_m3(os.path.join(tmp, "tiny"))
        _check_parity(
            ckpt,
            "transformers.models.minimax_m3_vl.modeling_minimax_m3_vl."
            "MiniMaxM3SparseForConditionalGeneration",
            tmp,
        )


def test_glm5_next_offloaded_logits_match_hf():
    if importlib.util.find_spec("transformers.models.glm5_next") is None:
        pytest.skip("transformers >= 5.16 with glm5_next required")
    from tests.python.integration._expert_activation_tiny import (
        build_tiny_glm5_next_clamped,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = build_tiny_glm5_next_clamped(os.path.join(tmp, "tiny"))
        _check_parity(
            ckpt,
            "transformers.models.glm5_next.modeling_glm5_next."
            "Glm5NextForConditionalGeneration",
            tmp,
        )

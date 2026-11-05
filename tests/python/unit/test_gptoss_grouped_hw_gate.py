"""The GPT-OSS grouped MXFP4 MoE op (#313) only runs on the RTX PRO 6000.

Other GPUs must keep the moe-store wrapper's per-expert loop: the engine op is
not registered and the op itself refuses to run.
"""

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.utils import hw_profile

MIB = 1024 * 1024
KIB = 1024

DEVICES = {
    "rtx_pro_6000": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "rtx_5090": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=170,
        L2_cache_size=96 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "h100": SimpleNamespace(
        major=9,
        minor=0,
        multi_processor_count=132,
        L2_cache_size=50 * MIB,
        shared_memory_per_block_optin=227 * KIB,
    ),
}


@pytest.fixture
def fake_gpu(monkeypatch):
    def use(name):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: DEVICES[name]
        )
        hw_profile._gates.cache_clear()

    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    yield use
    hw_profile._gates.cache_clear()


@pytest.mark.parametrize(
    "name, enabled",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
def test_gate(fake_gpu, name, enabled):
    from moe_infinity.runtime.engine_ops import gptoss_grouped_pro6000_enabled

    fake_gpu(name)
    assert gptoss_grouped_pro6000_enabled() is enabled


@pytest.mark.parametrize(
    "name, registered",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
def test_engine_op_registered_only_on_pro6000(
    fake_gpu, monkeypatch, name, registered
):
    hooks = pytest.importorskip("moe_store.hooks")
    pytest.importorskip("triton")
    if not hasattr(hooks.ops, "fused_moe_mxfp4"):
        pytest.skip("installed moe-store predates fused_moe_mxfp4")
    from moe_infinity.runtime import engine_ops

    fake_gpu(name)
    monkeypatch.setattr(engine_ops, "_registered", False)
    monkeypatch.setattr(hooks.ops, "fused_moe_mxfp4", None)
    try:
        engine_ops.register_engine_ops()
    except ImportError as exc:  # optional native pieces on CPU-only CI
        pytest.skip(f"engine ops unavailable: {exc}")
    assert (hooks.ops.fused_moe_mxfp4 is not None) is registered


@pytest.mark.parametrize("name", ["rtx_5090", "h100"])
def test_op_refuses_other_gpus(fake_gpu, name):
    pytest.importorskip("triton")
    from moe_infinity.kernel import mxfp4_moe

    fake_gpu(name)
    mxfp4_moe._pro6000_ok.cache_clear()
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    ids = torch.zeros(1, 1, dtype=torch.int64)
    w = torch.ones(1, 1, dtype=torch.bfloat16)
    u8 = torch.zeros(1, 2, 32, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="RTX PRO 6000"):
        mxfp4_moe.fused_moe_mxfp4(x, ids, w, u8, u8, w, u8, u8, w)
    mxfp4_moe._pro6000_ok.cache_clear()

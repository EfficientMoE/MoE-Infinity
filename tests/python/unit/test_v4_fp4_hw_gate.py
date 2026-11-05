"""Hardware gate of the DeepSeek-V4 grouped FP4 MoE (CPU only).

``v4_moe_forward_sm120_pro6000`` uses sm_120a block-scaled MMA and was tuned
on the RTX PRO 6000; every other device must keep Path B
(``v4_expert_forward``: BF16 dequant + cuBLAS).
"""

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.runtime.engine_ops import v4_moe_sm120_pro6000_supported
from moe_infinity.utils import hw_profile

MIB, KIB = 1024 * 1024, 1024


def _gpu(cc, sms, l2_mib, smem_kib):
    return SimpleNamespace(
        major=cc[0],
        minor=cc[1],
        multi_processor_count=sms,
        L2_cache_size=l2_mib * MIB,
        shared_memory_per_block_optin=smem_kib * KIB,
    )


DEVICES = {
    "rtx_pro_6000": _gpu((12, 0), 188, 128, 99),
    "rtx_5090": _gpu((12, 0), 170, 96, 99),
    "h100": _gpu((9, 0), 132, 50, 227),
    "b200": _gpu((10, 0), 148, 126, 227),
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
    "name, grouped",
    [
        ("rtx_pro_6000", True),
        ("rtx_5090", False),  # sm_120, but 170 SMs / 96 MiB L2
        ("h100", False),
        ("b200", False),
    ],
)
def test_grouped_moe_gate(fake_gpu, name, grouped):
    fake_gpu(name)
    assert v4_moe_sm120_pro6000_supported("cuda:0") is grouped


def test_override_never_enables_on_non_sm120(fake_gpu, monkeypatch):
    fake_gpu("h100")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert not v4_moe_sm120_pro6000_supported()


@pytest.mark.parametrize("name", ["h100", "rtx_5090"])
def test_moe_store_adapter_keeps_path_b(fake_gpu, monkeypatch, name):
    hooks = pytest.importorskip("moe_store.hooks")
    if not hasattr(hooks.ops, "v4_moe_sm120_pro6000_supported"):
        pytest.skip("moe-store without the grouped FP4 MoE gate hook")
    from moe_store.wrappers.deepseek_v4 import official_offload_adapter as ad

    class Ext:
        def v4_moe_forward_sm120_pro6000(self, *args):
            raise AssertionError("gated kernel called on " + name)

    fake_gpu(name)
    monkeypatch.setattr(hooks.ops, "v4_fp4_ext", Ext())
    monkeypatch.setattr(
        hooks.ops,
        "v4_moe_sm120_pro6000_supported",
        v4_moe_sm120_pro6000_supported,
    )
    monkeypatch.setattr(ad, "_V4_FP4", None)
    assert ad.grouped_native_available("cuda:0") is False
    fake_gpu("rtx_pro_6000")
    assert ad.grouped_native_available("cuda:0") is True

"""FP8 experts (#310) pick the sm120 / RTX PRO 6000 kernels only on that card.

CPU test: fakes the device properties and checks the dispatch decision and
what the offload layer tells the C++ dispatcher. Other GPUs (H100, RTX 5090)
must keep the previous dequant-to-BF16 path.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from moe_infinity.kernel import fp8_experts
from moe_infinity.kernel.fp8_experts import (
    FP8_EXPERTS_DEQUANT_BF16,
    FP8_EXPERTS_SM120_PRO6000,
    fp8_expert_backend,
)
from moe_infinity.utils import hw_profile

MIB = 1024 * 1024
KIB = 1024
REPO = Path(__file__).resolve().parents[3]

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
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: DEVICES[name]
        )
        hw_profile._gates.cache_clear()

    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    yield use
    hw_profile._gates.cache_clear()


@pytest.mark.parametrize(
    "name, backend",
    [
        ("rtx_pro_6000", FP8_EXPERTS_SM120_PRO6000),
        ("rtx_5090", FP8_EXPERTS_DEQUANT_BF16),
        ("h100", FP8_EXPERTS_DEQUANT_BF16),
    ],
)
def test_backend_by_device(fake_gpu, name, backend):
    fake_gpu(name)
    assert fp8_expert_backend() == backend
    assert fp8_expert_backend("cuda:0") == backend
    assert fp8_experts.fp8_sm120_pro6000_allowed() is (
        backend == FP8_EXPERTS_SM120_PRO6000
    )


def test_unvalidated_override_does_not_lift_sm120(fake_gpu, monkeypatch):
    # The override lifts the tuning gate only; an H100 still lacks sm_120a.
    fake_gpu("h100")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert fp8_expert_backend() == FP8_EXPERTS_DEQUANT_BF16


class _Dispatcher:
    def __init__(self):
        self.scales = None
        self.allowed = []

    def set_scales(self, scales):
        self.scales = scales

    def set_fp8_sm120_pro6000_experts(self, allowed):
        self.allowed.append(allowed)


@pytest.mark.parametrize(
    "name, allowed",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
def test_offload_tells_dispatcher(fake_gpu, name, allowed):
    from moe_infinity.runtime.model_offload import OffloadEngine

    fake_gpu(name)
    dispatcher = _Dispatcher()
    engine = SimpleNamespace(
        _blockwise_fp8_scales={"w": torch.ones(1)},
        expert_dispatcher=dispatcher,
    )
    OffloadEngine.deliver_fp8_scales_to_dispatcher(engine)
    assert dispatcher.scales is not None
    assert dispatcher.allowed == [allowed]


def test_cpp_gate_checks_both_gates():
    src = (REPO / "extensions/kernel/fp8_moe/fp8_moe.cu").read_text()
    body = re.search(
        r"bool fp8_moe_sm120_pro6000_supported\(.*?\n}\n", src, re.S
    ).group(0)
    assert "MOE_ENABLE_SM120_FP8" in body
    assert "Gate::kSm120" in body and "Gate::kRtxPro6000" in body
    module = (REPO / "core/parallel/expert_module.cpp").read_text()
    # MoEMLP only binds the native kernels through the gated check.
    assert "fp8_moe_sm120_pro6000_supported(" in module
    assert "fp8_moe_native_supported" not in module

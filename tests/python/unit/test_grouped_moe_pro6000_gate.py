"""The grouped decode MoE path (#306) is RTX PRO 6000 only."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from moe_infinity.runtime.grouped_moe_pro6000 import (
    grouped_decode_pro6000_enabled,
)
from moe_infinity.utils import hw_profile

MIB = 1024 * 1024
KIB = 1024
_REPO = Path(__file__).resolve().parents[3]

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
    monkeypatch.delenv("MOE_GROUPED_DECODE", raising=False)
    yield use
    hw_profile._gates.cache_clear()


@pytest.mark.parametrize(
    "name, grouped",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
def test_grouped_decode_only_on_rtx_pro_6000(fake_gpu, name, grouped):
    fake_gpu(name)
    assert grouped_decode_pro6000_enabled() is grouped


def test_kill_switch_wins_on_rtx_pro_6000(fake_gpu, monkeypatch):
    fake_gpu("rtx_pro_6000")
    monkeypatch.setenv("MOE_GROUPED_DECODE", "0")
    assert grouped_decode_pro6000_enabled() is False


def test_native_dispatcher_and_kernel_check_the_pro6000_gate():
    dispatcher = (_REPO / "core/parallel/expert_dispatcher.cpp").read_text()
    eligible = re.search(
        r"bool ExpertDispatcher::GroupedEligible\(.*?\n}\n", dispatcher, re.S
    ).group(0)
    assert "grouped_pro6000_ok_" in eligible
    kernel = (
        _REPO / "extensions/kernel/grouped_moe_decode_pro6000.cu"
    ).read_text()
    assert "Gate::kRtxPro6000" in kernel


@pytest.mark.parametrize(
    "config, kind",
    [
        (SimpleNamespace(model_type="qwen3_moe", hidden_act="silu"), 0),
        (SimpleNamespace(model_type="mixtral", hidden_act="silu"), 0),
        (
            SimpleNamespace(
                model_type="glm5_flash", hidden_act="silu", swiglu_limit=7.0
            ),
            1,
        ),
        (
            SimpleNamespace(
                model_type="minimax_m3_vl",
                text_config=SimpleNamespace(
                    model_type="minimax_m3_vl_text", hidden_act="silu"
                ),
            ),
            2,
        ),
        (SimpleNamespace(model_type="x", hidden_act="gelu"), -1),
    ],
)
def test_activation_kind_matches_expert_activation_rules(config, kind):
    from moe_infinity.runtime.grouped_moe_pro6000 import (
        grouped_decode_activation_kind,
    )

    assert grouped_decode_activation_kind(config) == kind


def test_dispatcher_rejects_non_silu_activation():
    dispatcher = (_REPO / "core/parallel/expert_dispatcher.cpp").read_text()
    eligible = re.search(
        r"bool ExpertDispatcher::GroupedEligible\(.*?\n}\n", dispatcher, re.S
    ).group(0)
    assert "grouped_activation_kind_" in eligible

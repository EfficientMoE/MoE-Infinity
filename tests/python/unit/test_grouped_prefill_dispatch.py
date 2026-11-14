"""Dispatcher selection of the grouped prefill kernels (#307), on the CPU."""

import re
from pathlib import Path

import pytest
import torch

from moe_infinity.runtime import grouped_moe_pro6000 as g
from moe_infinity.utils import hw_profile
from tests.python.unit.test_grouped_moe_pro6000_gate import DEVICES

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_gpu(monkeypatch):
    def use(name):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: DEVICES[name]
        )
        hw_profile._gates.cache_clear()

    for var in (
        "MOE_INFINITY_UNVALIDATED_HW",
        "MOE_GROUPED_PREFILL",
        "MOE_GROUPED_PREFILL_MIN_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)
    yield use
    hw_profile._gates.cache_clear()


def test_defaults_keep_current_behaviour(fake_gpu):
    fake_gpu("rtx_pro_6000")
    assert g.grouped_prefill_enabled() is False
    for tokens in (1, 256):
        assert g.select_grouped_mode(tokens, False) == g.DECODE
    for tokens in (257, 2048, 12_288, 200_000):
        assert g.select_grouped_mode(tokens, False) == g.NONE


def test_prefill_needs_flag_and_rtx_pro_6000(fake_gpu, monkeypatch):
    monkeypatch.setenv("MOE_GROUPED_PREFILL", "1")
    fake_gpu("rtx_pro_6000")
    assert g.grouped_prefill_enabled() is True
    for name in ("rtx_5090", "h100"):
        fake_gpu(name)
        assert g.grouped_prefill_enabled() is False
    fake_gpu("rtx_pro_6000")
    monkeypatch.setenv("MOE_GROUPED_PREFILL", "0")
    assert g.grouped_prefill_enabled() is False


def test_selection_thresholds():
    on = True
    min_tokens = g.GROUPED_PREFILL_DEFAULT_MIN_TOKENS
    assert g.select_grouped_mode(0, on) == g.NONE
    assert g.select_grouped_mode(256, on) == g.DECODE
    assert g.select_grouped_mode(257, on) == g.NONE
    assert g.select_grouped_mode(min_tokens - 1, on) == g.NONE
    assert g.select_grouped_mode(min_tokens, on) == g.PREFILL
    assert g.select_grouped_mode(50_000, on) == g.PREFILL
    # A custom threshold applies, but never inside the decode range.
    assert g.select_grouped_mode(512, on, 512) == g.PREFILL
    assert g.select_grouped_mode(511, on, 512) == g.NONE
    assert g.select_grouped_mode(100, on, 1) == g.DECODE
    assert g.select_grouped_mode(257, on, 1) == g.PREFILL


def test_min_tokens_env(monkeypatch):
    monkeypatch.delenv("MOE_GROUPED_PREFILL_MIN_TOKENS", raising=False)
    assert g.grouped_prefill_min_tokens() == 2048
    monkeypatch.setenv("MOE_GROUPED_PREFILL_MIN_TOKENS", "4096")
    assert g.grouped_prefill_min_tokens() == 4096
    for bad in ("0", "-5", "abc", ""):
        monkeypatch.setenv("MOE_GROUPED_PREFILL_MIN_TOKENS", bad)
        assert g.grouped_prefill_min_tokens() == 2048


def test_workspace_cap():
    # Qwen3-30B-A3B (top-8, I = 768) at 12K tokens fits; 50K tokens of a
    # top-8, I = 2048 model does not and keeps per-expert execution.
    assert g.grouped_workspace_fits(12_288, 8, 768)
    assert g.grouped_workspace_fits(12_288, 8, 2048)
    assert not g.grouped_workspace_fits(50_000, 8, 2048)


def test_constants_match_native_header():
    select = (_REPO / "core/parallel/grouped_moe_select.h").read_text()
    decode = (
        _REPO / "extensions/kernel/grouped_moe_decode_pro6000.h"
    ).read_text()
    assert re.search(r"kGroupedPrefillDefaultMinTokens\s*=\s*2048", select)
    assert re.search(
        r"kGroupedMaxActBytes\s*=\s*std::int64_t\(1\) << 30", select
    )
    assert re.search(r"kGroupedMoePro6000MaxTokens\s*=\s*256", decode)
    assert g.GROUPED_DECODE_MAX_TOKENS == 256
    assert g.GROUPED_PREFILL_DEFAULT_MIN_TOKENS == 2048
    assert g.GROUPED_MAX_ACT_BYTES == 1 << 30


def _function(source, signature):
    return re.search(signature + r".*?\n}\n", source, re.S).group(0)


def test_dispatcher_wiring():
    cpp = (_REPO / "core/parallel/expert_dispatcher.cpp").read_text()
    eligible = _function(cpp, r"bool ExpertDispatcher::GroupedEligible\(")
    # Off by default, RTX PRO 6000 gated, shared SiLU-only restriction.
    assert "grouped_prefill_enabled_" in eligible
    assert "grouped_prefill_ok_" in eligible
    assert "SelectGroupedMoeMode" in eligible
    assert "grouped_moe_prefill_supported" in eligible
    assert "grouped_activation_kind_" in eligible
    exec_ = _function(cpp, r"void ExpertDispatcher::GroupedExec\(")
    assert "grouped_moe_prefill_cfg" in exec_
    assert "grouped_moe_decode_pro6000(" in exec_
    # The non-atomic bf16 combine would race with index_add_ on other streams.
    assert re.search(r"/\*combine_mode=\*/\s*0", exec_)
    header = (_REPO / "core/parallel/expert_dispatcher.h").read_text()
    assert re.search(r"grouped_prefill_enabled_\{false\}", header)
    binding = (_REPO / "core/python/py_archer_prefetch.cpp").read_text()
    assert '"set_grouped_prefill"' in binding

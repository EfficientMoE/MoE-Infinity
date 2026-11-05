"""The RTX PRO 6000-tuned GPT-OSS offload kernels are not chosen elsewhere.

CPU-only: fakes the device properties like ``test_hw_profile.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from moe_infinity.runtime import gptoss_offload
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
def fake_gpus(monkeypatch):
    def use(*names):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i: DEVICES[names[i]],
        )
        hw_profile._gates.cache_clear()

    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    yield use
    hw_profile._gates.cache_clear()


@pytest.mark.parametrize("name", ["h100", "rtx_5090"])
def test_other_gpus_keep_previous_path(fake_gpus, name):
    fake_gpus(name)
    assert not gptoss_offload.gptoss_mxfp4_pro6000_enabled()


def test_rtx_pro_6000_uses_mxfp4_kernels(fake_gpus):
    fake_gpus("rtx_pro_6000")
    assert gptoss_offload.gptoss_mxfp4_pro6000_enabled()


def test_no_cuda_keeps_previous_path(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not gptoss_offload.gptoss_mxfp4_pro6000_enabled()

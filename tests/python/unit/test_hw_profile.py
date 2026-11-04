from types import SimpleNamespace

import pytest
import torch

from moe_infinity.utils import hw_profile
from moe_infinity.utils.hw_profile import RTX_PRO_6000, SM120, supports

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
    "b200": SimpleNamespace(
        major=10,
        minor=0,
        multi_processor_count=148,
        L2_cache_size=126 * MIB,
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
    "name, sm120, pro6000",
    [
        ("rtx_pro_6000", True, True),
        ("rtx_5090", True, False),
        ("h100", False, False),
        ("b200", False, False),
    ],
)
def test_gates_by_device(fake_gpu, name, sm120, pro6000):
    fake_gpu(name)
    assert supports(SM120) is sm120
    assert supports(RTX_PRO_6000) is pro6000
    assert supports(RTX_PRO_6000, "cuda:0") is pro6000


def test_override_lifts_tuning_gate_only(fake_gpu, monkeypatch):
    fake_gpu("h100")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert supports(RTX_PRO_6000)
    assert not supports(SM120)


def test_cpu_and_unknown_gate(fake_gpu, monkeypatch):
    fake_gpu("rtx_pro_6000")
    assert not supports(SM120, "cpu")
    with pytest.raises(ValueError):
        supports("sm90")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not supports(SM120)

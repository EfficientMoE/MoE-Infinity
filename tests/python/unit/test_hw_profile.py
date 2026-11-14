from types import SimpleNamespace

import pytest
import torch

from moe_infinity.utils import hw_profile
from moe_infinity.utils.hw_profile import (
    RTX_PRO_6000,
    RTX_PRO_6000_MAXQ,
    RTX_PRO_6000_SERVER,
    SM120,
    pro6000_edition,
    supports,
)

MIB = 1024 * 1024
KIB = 1024

DEVICES = {
    "rtx_pro_6000": SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "rtx_pro_6000_server": SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
        memory_clock_rate=12481000,
    ),
    "rtx_pro_6000_maxq": SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
        memory_clock_rate=14001000,
    ),
    "rtx_5090": SimpleNamespace(
        name="NVIDIA GeForce RTX 5090",
        major=12,
        minor=0,
        multi_processor_count=170,
        L2_cache_size=96 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "h100": SimpleNamespace(
        name="NVIDIA H100 SXM5 80GB",
        major=9,
        minor=0,
        multi_processor_count=132,
        L2_cache_size=50 * MIB,
        shared_memory_per_block_optin=227 * KIB,
    ),
    "b200": SimpleNamespace(
        name="NVIDIA B200",
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
    monkeypatch.delenv("MOE_INFINITY_PRO6000_EDITION", raising=False)
    yield use
    hw_profile._gates.cache_clear()


@pytest.mark.parametrize(
    "name, sm120, pro6000, server, maxq, edition",
    [
        ("rtx_pro_6000", True, True, False, False, None),
        ("rtx_pro_6000_server", True, True, True, False, "server"),
        ("rtx_pro_6000_maxq", True, True, False, True, "maxq"),
        ("rtx_5090", True, False, False, False, None),
        ("h100", False, False, False, False, None),
        ("b200", False, False, False, False, None),
    ],
)
def test_gates_by_device(
    fake_gpu, name, sm120, pro6000, server, maxq, edition
):
    fake_gpu(name)
    assert supports(SM120) is sm120
    assert supports(RTX_PRO_6000) is pro6000
    assert supports(RTX_PRO_6000, "cuda:0") is pro6000
    assert supports(RTX_PRO_6000_SERVER) is server
    assert supports(RTX_PRO_6000_MAXQ) is maxq
    assert pro6000_edition() == edition


def test_edition_env_override(fake_gpu, monkeypatch):
    fake_gpu("rtx_pro_6000_server")
    monkeypatch.setenv("MOE_INFINITY_PRO6000_EDITION", "maxq")
    hw_profile._gates.cache_clear()
    assert supports(RTX_PRO_6000_MAXQ)
    assert not supports(RTX_PRO_6000_SERVER)
    assert pro6000_edition() == "maxq"


def test_override_lifts_tuning_gate_only(fake_gpu, monkeypatch):
    fake_gpu("h100")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert supports(RTX_PRO_6000)
    assert supports(RTX_PRO_6000_SERVER)
    assert supports(RTX_PRO_6000_MAXQ)
    assert not supports(SM120)


def test_cpu_and_unknown_gate(fake_gpu, monkeypatch):
    fake_gpu("rtx_pro_6000")
    assert not supports(SM120, "cpu")
    assert pro6000_edition("cpu") is None
    with pytest.raises(ValueError):
        supports("sm90")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not supports(SM120)

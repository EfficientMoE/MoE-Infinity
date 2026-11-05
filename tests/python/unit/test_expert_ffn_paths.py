"""Hardware gate of the expert-FFN prefill paths (#307 / #308), on the CPU."""

import pytest
import torch

from moe_infinity.utils import expert_ffn_paths as paths
from moe_infinity.utils import hw_profile
from tests.python.unit.test_hw_profile import DEVICES


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


@pytest.mark.parametrize("name", ["h100", "rtx_5090", "b200"])
def test_other_gpus_keep_previous_paths(fake_gpu, name):
    fake_gpu(name)
    assert paths.expert_ffn_path() == paths.CUTLASS_GENERIC
    assert paths.grouped_prefill_path() == paths.PER_EXPERT
    assert paths.grouped_prefill_path("cuda:0") == paths.PER_EXPERT


def test_rtx_pro_6000_runs_tuned_paths(fake_gpu):
    fake_gpu("rtx_pro_6000")
    assert paths.expert_ffn_path() == paths.CUTLASS_PRO6000
    assert paths.grouped_prefill_path() == paths.GROUPED_PRO6000


def test_override_enables_tuned_paths_for_retuning(fake_gpu, monkeypatch):
    fake_gpu("rtx_5090")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert paths.expert_ffn_path() == paths.CUTLASS_PRO6000
    assert paths.grouped_prefill_path() == paths.GROUPED_PRO6000

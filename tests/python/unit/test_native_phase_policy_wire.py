import importlib

import pytest
import torch


def _load_prefetch_handle(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        prefetch_lib = importlib.import_module("moe_infinity._store")
    except Exception:
        pytest.skip("moe_infinity._store extension not built")
    return prefetch_lib.prefetch_handle(f"{tmp_path}/", 0.5)


@pytest.mark.gpu
def test_expert_policy_stats_expose_single_manager_snapshot(tmp_path):
    engine = _load_prefetch_handle(tmp_path)

    stats = engine.get_expert_policy_stats()

    assert isinstance(stats, dict)
    assert "resident_bytes" in stats
    assert "resident_count" in stats
    assert "capacity_bytes" in stats
    assert stats["resident_bytes"] == 0
    assert stats["resident_count"] == 0


@pytest.mark.gpu
def test_demand_and_prefetch_share_one_resident_byte_snapshot(tmp_path):
    engine = _load_prefetch_handle(tmp_path)

    first = engine.get_expert_policy_stats()
    second = engine.get_expert_policy_stats()

    assert list(first.keys()) == list(second.keys())
    resident_keys = [key for key in first if key == "resident_bytes"]
    assert len(resident_keys) == 1
    assert first["resident_bytes"] == second["resident_bytes"]

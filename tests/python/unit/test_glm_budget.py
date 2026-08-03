import pytest


class _Props96:
    total_memory = 96 * 1024 ** 3
    multi_processor_count = 188


class _Props80:
    total_memory = 80 * 1024 ** 3
    multi_processor_count = 108


def _patch_cuda(monkeypatch, props):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props, raising=True)


def test_expert_and_kv_budget_within_total(monkeypatch):
    from moe_infinity.memory.memory_coordinator import MemoryCoordinator

    total = _Props96.total_memory
    _patch_cuda(monkeypatch, _Props96())

    coord = MemoryCoordinator(device_memory_ratio=0.5, kv_cache_memory_ratio=0.15)
    got = coord.total_gpu_memory_bytes(0)
    assert got == total

    prev_ec = None
    for dmr in (0.2, 0.5, 0.9):
        kvr = min(0.05, 1.0 - dmr)
        coord2 = MemoryCoordinator(device_memory_ratio=dmr, kv_cache_memory_ratio=kvr)
        ec = coord2.expert_cache_bytes(0)
        kv = coord2.kv_cache_bytes(0)
        assert 0 <= ec <= total
        assert 0 <= kv <= total
        assert ec + kv <= total
        if prev_ec is not None:
            assert ec >= prev_ec
        prev_ec = ec


def test_remaining_bytes_non_negative(monkeypatch):
    from moe_infinity.memory.memory_coordinator import MemoryCoordinator

    total = _Props80.total_memory
    _patch_cuda(monkeypatch, _Props80())

    coord = MemoryCoordinator(device_memory_ratio=0.4, kv_cache_memory_ratio=0.25)
    rem = coord.remaining_bytes(0)
    assert rem >= 0
    assert rem == total - coord.expert_cache_bytes(0) - coord.kv_cache_bytes(0)


def test_budget_constraint_enforced():
    from moe_infinity.memory.memory_coordinator import MemoryCoordinator

    with pytest.raises(ValueError):
        MemoryCoordinator(device_memory_ratio=0.8, kv_cache_memory_ratio=0.3)


def test_from_config_glm_seq4k(monkeypatch):
    from moe_infinity.memory.memory_coordinator import MemoryCoordinator

    total = _Props96.total_memory
    _patch_cuda(monkeypatch, _Props96())

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coord = MemoryCoordinator.from_config(
            {"device_memory_ratio": 0.7, "kv_cache_memory_ratio": 0.15, "use_native_engine": True}
        )
    assert coord.device_memory_ratio == pytest.approx(0.7)
    assert coord.kv_cache_memory_ratio == pytest.approx(0.15)
    ec = coord.expert_cache_bytes(0)
    kv = coord.kv_cache_bytes(0)
    assert ec + kv <= total


def test_from_config_glm_seq32k(monkeypatch):
    from moe_infinity.memory.memory_coordinator import MemoryCoordinator

    total = _Props96.total_memory
    _patch_cuda(monkeypatch, _Props96())

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coord = MemoryCoordinator.from_config(
            {"device_memory_ratio": 0.4, "kv_cache_memory_ratio": 0.25, "use_native_engine": True}
        )
    assert coord.device_memory_ratio == pytest.approx(0.4)
    assert coord.kv_cache_memory_ratio == pytest.approx(0.25)
    ec = coord.expert_cache_bytes(0)
    kv = coord.kv_cache_bytes(0)
    assert ec + kv <= total

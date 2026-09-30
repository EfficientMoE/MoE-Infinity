from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_resident_on_gpu_is_bound():
    store = pytest.importorskip("moe_infinity._store")
    if "mock" in type(store).__module__ or not getattr(store, "__file__", None):
        pytest.skip("_store is mocked in this environment")
    dispatcher = getattr(store, "expert_dispatcher", None)
    if dispatcher is None:
        pytest.skip("expert_dispatcher extension class unavailable")
    assert hasattr(dispatcher, "resident_on_gpu")


def test_residency_probe_uses_atomic_publication():
    node_header = (ROOT / "core/model/model_topology.h").read_text()
    node_source = (ROOT / "core/model/model_topology.cpp").read_text()
    dispatcher_source = (
        ROOT / "core/parallel/expert_dispatcher.cpp"
    ).read_text()

    assert "std::atomic<bool> resident_on_gpu" in node_header
    assert "resident_on_gpu.store" in node_source
    assert "resident_on_gpu.load" in dispatcher_source

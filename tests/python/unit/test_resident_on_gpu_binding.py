import pytest


def test_resident_on_gpu_is_bound():
    store = pytest.importorskip("moe_infinity._store")
    if "mock" in type(store).__module__ or not getattr(store, "__file__", None):
        pytest.skip("_store is mocked in this environment")
    dispatcher = getattr(store, "expert_dispatcher", None)
    if dispatcher is None:
        pytest.skip("expert_dispatcher extension class unavailable")
    assert hasattr(dispatcher, "resident_on_gpu")

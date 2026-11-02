"""Decode-graph blockers name the hook / component that prevents capture."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from moe_infinity.runtime.attention_types import DecodeGraphCapability


def _offload_engine(**attrs: object):
    from moe_infinity.runtime.model_offload import OffloadEngine

    engine = OffloadEngine.__new__(OffloadEngine)
    engine._enable_kv_cache_offload = False
    engine._kv_cache_manager = None
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine


def test_offload_engine_hooks_report_their_count() -> None:
    engine = _offload_engine(forward_hooks=[object()] * 3, backward_hooks=[])
    capability = engine.decode_graph_capability()
    assert capability == DecodeGraphCapability(False, "active_model_hooks")
    assert "3 forward / 0 backward hooks" in capability.detail


@pytest.mark.parametrize(
    ("attr", "reason"),
    [
        ("archer_engine", "archer_callbacks"),
        ("expert_dispatcher", "expert_dispatcher"),
        ("expert_prefetcher", "transfer_scheduler"),
    ],
)
def test_offload_engine_components_are_named(attr: str, reason: str) -> None:
    capability = _offload_engine(**{attr: object()}).decode_graph_capability()
    assert capability.reason == reason
    assert capability.detail


def test_resident_offload_engine_is_eligible() -> None:
    capability = _offload_engine(
        forward_hooks=[], backward_hooks=[]
    ).decode_graph_capability()
    assert capability.safe and capability.reason == "eligible"


def _native_moe(**attrs: object):
    from moe_infinity.entrypoints.big_modeling import MoE

    moe = MoE.__new__(MoE)
    moe.engine = None
    for name, value in attrs.items():
        setattr(moe, name, value)
    return moe


def test_idle_native_transfer_scheduler_does_not_block() -> None:
    storage = types.SimpleNamespace(owner_id="owner")
    moe = _native_moe(
        _native_transfer_scheduler=types.SimpleNamespace(_handlers={}),
        _native_kv_offload_coordinator=None,
        _native_paged_kv_storage=storage,
    )
    capability = moe.decode_graph_capability()
    assert capability.safe
    assert capability.storage_owner_id == "owner"


def test_native_transfer_handlers_block_and_are_listed() -> None:
    from moe_infinity.engine.transfer_types import TransferType

    moe = _native_moe(
        _native_transfer_scheduler=types.SimpleNamespace(
            _handlers={TransferType.KV_SWAP_OUT: lambda request: None}
        ),
        _native_kv_offload_coordinator=None,
        _native_paged_kv_storage=types.SimpleNamespace(owner_id="owner"),
    )
    capability = moe.decode_graph_capability()
    assert capability.reason == "transfer_scheduler"
    assert "KV_SWAP_OUT" in capability.detail


def test_missing_native_storage_explains_itself() -> None:
    moe = _native_moe(
        _native_transfer_scheduler=None,
        _native_kv_offload_coordinator=None,
        _native_paged_kv_storage=None,
    )
    capability = moe.decode_graph_capability()
    assert capability.reason == "native_paged_required"
    assert "PagedKVStorage" in capability.detail


def test_graph_runner_stats_carry_the_blocker_detail() -> None:
    from moe_infinity.serving.cuda_graph import CudaGraphRunner

    blocked = DecodeGraphCapability(
        False, "active_model_hooks", detail="2 forward hooks"
    )
    model_runner = types.SimpleNamespace(
        decode_graph_capability=lambda: blocked
    )
    storage = types.SimpleNamespace(owner_id="owner", block_size=16)
    runner = CudaGraphRunner(model_runner, storage, enabled=True)
    assert runner._check_capability(None) == "active_model_hooks"
    assert runner.stats()["fallback_details"] == {
        "active_model_hooks": "2 forward hooks"
    }


def test_serve_forwards_decode_graph_options(monkeypatch) -> None:
    import moe_infinity.entrypoints.openai as openai_pkg

    captured: dict[str, object] = {}
    fake_server = types.ModuleType("api_server_v2")
    fake_server.app = object()
    fake_server.initialize_with_model = lambda **kwargs: captured.update(kwargs)
    module_name = "moe_infinity.entrypoints.openai.api_server_v2"
    monkeypatch.setitem(sys.modules, module_name, fake_server)
    monkeypatch.setattr(openai_pkg, "api_server_v2", fake_server, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: None),
    )

    moe = _native_moe(
        engine_config=types.SimpleNamespace(),
        model=types.SimpleNamespace(config=types.SimpleNamespace()),
        max_seq_length=128,
    )
    moe.serve(
        offload_dir="/tmp/unused",
        enable_decode_cuda_graphs=True,
        decode_cuda_graph_batch_sizes=[1, 2],
        decode_cuda_graph_context_sizes=[256],
        decode_cuda_graph_warmup_iters=3,
        decode_cuda_graph_max_memory_bytes=1 << 20,
    )

    assert captured["enable_decode_cuda_graphs"] is True
    assert captured["decode_cuda_graph_batch_sizes"] == (1, 2)
    assert captured["decode_cuda_graph_context_sizes"] == (256,)
    assert captured["decode_cuda_graph_warmup_iters"] == 3
    assert captured["decode_cuda_graph_max_memory_bytes"] == 1 << 20


def test_serve_keeps_decode_graphs_off_by_default(monkeypatch) -> None:
    import moe_infinity.entrypoints.openai as openai_pkg

    captured: dict[str, object] = {}
    fake_server = types.ModuleType("api_server_v2")
    fake_server.app = object()
    fake_server.initialize_with_model = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(
        sys.modules,
        "moe_infinity.entrypoints.openai.api_server_v2",
        fake_server,
    )
    monkeypatch.setattr(openai_pkg, "api_server_v2", fake_server, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: None),
    )
    moe = _native_moe(
        engine_config=types.SimpleNamespace(),
        model=types.SimpleNamespace(config=types.SimpleNamespace()),
        max_seq_length=128,
    )
    moe.serve(offload_dir="/tmp/unused")
    assert captured["enable_decode_cuda_graphs"] is False


def test_registry_bound_shim_subclasses_are_discovered() -> None:
    from moe_infinity.serving.model_runner import ModelRunner

    class Qwen3PagedAttention(torch.nn.Module):
        layer_idx = 0

        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            _ = backend, metadata

        @classmethod
        def clear_paged_context(cls) -> None:
            pass

    # PagedAttentionLayerRegistry rebinds each module to a per-layer subclass.
    bound = type("Qwen3PagedAttentionLayer0", (Qwen3PagedAttention,), {})
    module = Qwen3PagedAttention()
    module.__class__ = bound
    model = types.SimpleNamespace(modules=lambda: [module])
    runner = ModelRunner(
        model, types.SimpleNamespace(), device=torch.device("cpu")
    )

    assert runner._get_paged_attention_classes() == [bound]
    assert runner._get_qwen3_paged_attention_modules() == [module]

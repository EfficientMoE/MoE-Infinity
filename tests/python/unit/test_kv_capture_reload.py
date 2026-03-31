import importlib.machinery
import sys
import types

import torch

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub

from moe_infinity.runtime.model_offload import OffloadEngine


def _build_past_key_values(dtype: torch.dtype):
    return (
        (
            torch.randn(2, 4, 8, dtype=dtype),
            torch.randn(2, 4, 8, dtype=dtype),
        ),
        (
            torch.randn(2, 4, 8, dtype=dtype),
            torch.randn(2, 4, 8, dtype=dtype),
        ),
    )


def _new_engine_stub() -> OffloadEngine:
    engine = OffloadEngine.__new__(OffloadEngine)
    engine._captured_kv = {}
    return engine


def test_roundtrip_fp16() -> None:
    engine = _new_engine_stub()
    past_key_values = _build_past_key_values(torch.float16)

    OffloadEngine._capture_kv_cache(
        engine, seq_id=0, past_key_values=past_key_values
    )

    assert 0 in engine._captured_kv

    restored = OffloadEngine._reload_kv_cache(engine, seq_id=0)
    assert restored is not None
    for (orig_k, orig_v), (new_k, new_v) in zip(past_key_values, restored):
        assert torch.allclose(orig_k, new_k, rtol=1e-5)
        assert torch.allclose(orig_v, new_v, rtol=1e-5)

    assert 0 not in engine._captured_kv


def test_roundtrip_bf16() -> None:
    engine = _new_engine_stub()
    past_key_values = _build_past_key_values(torch.bfloat16)

    OffloadEngine._capture_kv_cache(
        engine, seq_id=0, past_key_values=past_key_values
    )

    assert 0 in engine._captured_kv

    restored = OffloadEngine._reload_kv_cache(engine, seq_id=0)
    assert restored is not None
    for (orig_k, orig_v), (new_k, new_v) in zip(past_key_values, restored):
        assert torch.allclose(orig_k, new_k, rtol=1e-5)
        assert torch.allclose(orig_v, new_v, rtol=1e-5)

    assert 0 not in engine._captured_kv


def test_capture_frees_on_reload() -> None:
    engine = _new_engine_stub()
    past_key_values = _build_past_key_values(torch.float16)

    OffloadEngine._capture_kv_cache(
        engine, seq_id=11, past_key_values=past_key_values
    )
    assert 11 in engine._captured_kv

    _ = OffloadEngine._reload_kv_cache(engine, seq_id=11)
    assert 11 not in engine._captured_kv

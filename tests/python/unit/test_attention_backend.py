import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for module_name in list(sys.modules):
    if module_name == "moe_infinity" or module_name.startswith("moe_infinity."):
        del sys.modules[module_name]

_ = sys.modules.setdefault("nvtx", MagicMock())
_ = sys.modules.setdefault("moe_infinity._store", MagicMock())
_ = sys.modules.setdefault("moe_infinity._engine", MagicMock())


def test_import_attention_backend():
    from moe_infinity.runtime.attention_backend import AttentionBackend

    assert AttentionBackend is not None


def test_import_attention_metadata():
    from moe_infinity.runtime.attention_backend import AttentionMetadata

    assert AttentionMetadata is not None


def test_import_placeholder_backend():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    assert PlaceholderAttentionBackend is not None


def test_attention_metadata_has_required_fields():
    from moe_infinity.runtime.attention_backend import AttentionMetadata

    meta = AttentionMetadata(
        is_prefill=True,
        block_table=None,
        slot_mapping=None,
    )
    assert meta.is_prefill is True
    assert meta.block_table is None
    assert meta.slot_mapping is None


def test_placeholder_backend_implements_protocol():
    from moe_infinity.runtime.attention_backend import (
        AttentionBackend,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert isinstance(backend, AttentionBackend)


def test_placeholder_backend_supports_dtype():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert backend.supports_dtype(torch.float16) is True
    assert backend.supports_dtype(torch.bfloat16) is True
    assert backend.supports_dtype(torch.float32) is True


def test_placeholder_backend_get_kv_cache_shape():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    shape = backend.get_kv_cache_shape(
        num_blocks=8, block_size=16, num_kv_heads=4, head_size=64
    )
    assert isinstance(shape, tuple)
    assert len(shape) >= 3


def test_placeholder_backend_forward_returns_none_or_tensor():
    from moe_infinity.runtime.attention_backend import (
        AttentionMetadata,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    query = torch.zeros(1, 4, 64)
    key = torch.zeros(1, 4, 64)
    value = torch.zeros(1, 4, 64)
    meta = AttentionMetadata(
        is_prefill=True, block_table=None, slot_mapping=None
    )
    result = backend.forward(
        query, key, value, kv_cache=None, attn_metadata=meta
    )
    assert result is None or isinstance(result, torch.Tensor)


def test_attention_backend_is_runtime_checkable():
    from moe_infinity.runtime.attention_backend import (
        AttentionBackend,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert isinstance(backend, AttentionBackend)

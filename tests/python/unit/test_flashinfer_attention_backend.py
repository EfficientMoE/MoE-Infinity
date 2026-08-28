import types

import pytest
import torch

from moe_infinity.runtime import attention_backend as attention_backend_module
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec
from moe_infinity.serving.kv_cache import PagedKVCache


class _FakePrefillWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_args = None
        self.run_args = None

    def plan(self, *args, **kwargs) -> None:
        self.plan_args = (args, kwargs)

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        self.run_args = (query, kv_cache)
        return torch.zeros_like(query)


class _FakeDecodeWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_args = None
        self.run_args = None

    def plan(self, *args, **kwargs) -> None:
        self.plan_args = (args, kwargs)

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        self.run_args = (query, kv_cache)
        return torch.zeros_like(query)


def _spec() -> KVCacheSpec:
    return KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )


def _prefill_metadata(num_tokens: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        max_seq_len=num_tokens,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        is_prefill=True,
    )


def _decode_metadata(seq_len: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        max_seq_len=seq_len,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.long),
        is_prefill=False,
    )


def _enable_fake_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=_FakePrefillWrapper,
        BatchDecodeWithPagedKVCacheWrapper=_FakeDecodeWrapper,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        True,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_flashinfer_module",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_workspace",
        lambda device: torch.empty(
            1024,
            dtype=torch.uint8,
            device=device,
        ),
    )


def test_flashinfer_kv_cache_layout_nhd_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )
    assert backend._fi_kv_cache is not None
    assert backend._fi_kv_cache.shape == (10, 2, 4, 2, 8)


def test_flashinfer_prefill_metadata_is_int32_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )

    query = torch.randn(4, 4, 8)
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    out = backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=_prefill_metadata(num_tokens=4),
    )

    assert out.shape == (4, 4, 8)
    assert backend._fi_prefill is not None
    assert backend._fi_prefill.plan_args is not None
    plan_args = backend._fi_prefill.plan_args[0]
    assert plan_args[0].dtype == torch.int32
    assert plan_args[1].dtype == torch.int32
    assert plan_args[2].dtype == torch.int32
    assert plan_args[3].dtype == torch.int32


def test_flashinfer_decode_metadata_is_int32_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )

    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_kv_flashinfer(
        key=key,
        value=value,
        slot_mapping=torch.arange(4, dtype=torch.long),
    )

    out = backend.forward(
        query=torch.randn(1, 4, 8),
        key=key[:1],
        value=value[:1],
        attention_metadata=_decode_metadata(seq_len=4),
    )
    assert out.shape == (1, 4, 8)
    assert backend._fi_decode is not None
    assert backend._fi_decode.plan_args is not None
    plan_args = backend._fi_decode.plan_args[0]
    assert plan_args[0].dtype == torch.int32
    assert plan_args[1].dtype == torch.int32
    assert plan_args[2].dtype == torch.int32


def test_write_kv_flashinfer_writes_expected_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=2,
        device=torch.device("cpu"),
    )

    key = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    value = key + 1000.0
    slot_mapping = torch.tensor([0, 3, 4], dtype=torch.long)

    backend.write_kv_flashinfer(key=key, value=value, slot_mapping=slot_mapping)
    assert backend._fi_kv_cache is not None
    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        block_id = slot // backend.spec.block_size
        token_offset = slot % backend.spec.block_size
        torch.testing.assert_close(
            backend._fi_kv_cache[block_id, 0, token_offset], key[i]
        )
        torch.testing.assert_close(
            backend._fi_kv_cache[block_id, 1, token_offset], value[i]
        )


def test_fallback_prefill_without_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        False,
    )
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )

    out = backend.forward(
        query=torch.randn(4, 4, 8),
        key=torch.randn(4, 2, 8),
        value=torch.randn(4, 2, 8),
        attention_metadata=_prefill_metadata(num_tokens=4),
    )
    assert out.shape == (4, 4, 8)
    assert backend._fi_prefill is None


def test_fallback_decode_without_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        False,
    )
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_kv(key=key, value=value, slot_mapping=torch.arange(4))

    out = backend.forward(
        query=torch.randn(1, 4, 8),
        key=key[:1],
        value=value[:1],
        attention_metadata=_decode_metadata(seq_len=4),
    )
    assert out.shape == (1, 4, 8)
    assert backend._fi_decode is None


def _make_serving_cache(num_blocks: int, block_size: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_layered_store_checkpoint_restores_both_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    checkpoint = backend.block_store.checkpoint([1])
    key = torch.full((2, 2, 8), 7.0)
    value = torch.full((2, 2, 8), 9.0)
    slots = torch.tensor([4, 5])
    backend.write_kv(key, value, slots)
    backend.write_kv_flashinfer(key, value, slots)

    backend.block_store.restore([1], checkpoint)

    payload = backend.block_store.export_blocks([1])
    assert torch.count_nonzero(payload.k_cache) == 0
    assert torch.count_nonzero(payload.v_cache) == 0
    assert torch.count_nonzero(payload.fi_kv_cache) == 0


def test_swap_exports_and_restores_runtime_backend_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    cache = _make_serving_cache(num_blocks=4, block_size=4)
    cache.set_block_store(
        backend.block_store, logical_capacity=cache.num_blocks
    )
    cache.allocate_sequence(3, num_tokens=4)
    key = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8)
    value = key + 100.0
    backend.write_kv(key, value, torch.arange(4))
    backend.write_kv_flashinfer(key, value, torch.arange(4))

    cache.swap_out(3)
    cache.free_gpu_blocks(3)
    cache.swap_in(3)

    restored = backend.block_store.export_blocks(cache.get_block_table(3))
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 0], key.reshape(1, 4, 2, 8)
    )
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 1], value.reshape(1, 4, 2, 8)
    )


@pytest.mark.skipif(
    not flashinfer_utils.HAS_FLASHINFER or not torch.cuda.is_available(),
    reason="requires flashinfer + CUDA",
)
def test_flashinfer_workspace_reuse_across_batches() -> None:
    backend = attention_backend_module.PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float16,
            block_size=4,
        ),
        num_gpu_blocks=16,
        device=torch.device("cuda"),
    )

    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([4], dtype=torch.int32, device="cuda"),
        max_seq_len=4,
        num_prefill_tokens=4,
        num_decode_tokens=0,
        slot_mapping=torch.arange(4, dtype=torch.long, device="cuda"),
        is_prefill=True,
    )

    query = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
    key = torch.randn(4, 2, 16, dtype=torch.float16, device="cuda")
    value = torch.randn(4, 2, 16, dtype=torch.float16, device="cuda")

    workspace0 = backend._fi_workspace
    backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=metadata,
    )
    backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=metadata,
    )
    assert backend._fi_workspace is workspace0

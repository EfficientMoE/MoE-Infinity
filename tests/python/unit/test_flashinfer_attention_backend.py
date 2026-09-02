import types

import pytest
import torch

from moe_infinity.engine.memory_resize import ResizeReceipt
from moe_infinity.runtime import attention_backend as attention_backend_module
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec


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
        attention_backend_module.flashinfer_utils, "HAS_FLASHINFER", True
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_flashinfer_module",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_workspace",
        lambda device: torch.empty(1024, dtype=torch.uint8, device=device),
    )


class _CompleteEvent:
    def query(self) -> bool:
        return True


def _resize_receipt() -> ResizeReceipt:
    return ResizeReceipt(
        device_id=0,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(_CompleteEvent(),),
        admissions_paused=True,
    )


def test_flashinfer_resize_recreates_store_and_both_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=8, device=torch.device("cpu")
    )
    old_store = backend._fi_kv_cache
    old_prefill = backend._fi_prefill
    old_decode = backend._fi_decode
    backend.resize_num_blocks(0, 4, _resize_receipt())
    assert backend._fi_kv_cache is not old_store
    assert backend._fi_kv_cache is not None
    assert backend._fi_kv_cache.shape[0] == 4
    assert backend._fi_prefill is not old_prefill
    assert backend._fi_decode is not old_decode
    assert backend._fi_prefill is not backend._fi_decode

    backend.forward(
        query=torch.randn(4, 4, 8),
        key=torch.randn(4, 2, 8),
        value=torch.randn(4, 2, 8),
        attention_metadata=_prefill_metadata(4),
    )
    backend.forward(
        query=torch.randn(1, 4, 8),
        key=torch.randn(1, 2, 8),
        value=torch.randn(1, 2, 8),
        attention_metadata=_decode_metadata(4),
    )
    assert backend._fi_prefill.plan_args is not None
    assert max(backend._fi_prefill.plan_args[0][2].tolist()) < 4
    assert backend._fi_decode.plan_args is not None
    assert max(backend._fi_decode.plan_args[0][1].tolist()) < 4


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

"""The split-KV decode kernels are tuned for the RTX PRO 6000 only; every other
GPU must dispatch to the previous decode-attention paths."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from moe_infinity.kernel import fused_decode_attn, paged_attention_ops
from moe_infinity.utils import hw_profile

MIB = 1024 * 1024
KIB = 1024

DEVICES = {
    "rtx_pro_6000": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "rtx_5090": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=170,
        L2_cache_size=96 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "h100": SimpleNamespace(
        major=9,
        minor=0,
        multi_processor_count=132,
        L2_cache_size=50 * MIB,
        shared_memory_per_block_optin=227 * KIB,
    ),
}


@pytest.fixture
def fake_gpu(monkeypatch):
    def use(name):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: DEVICES[name]
        )
        hw_profile._gates.cache_clear()
        fused_decode_attn._PRO6000_GATE.clear()

    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    yield use
    hw_profile._gates.cache_clear()
    fused_decode_attn._PRO6000_GATE.clear()


def _fake_call(head_dim: int, kv_dtype: torch.dtype):
    query = SimpleNamespace(
        device=torch.device("cuda:0"), shape=(2, 8, head_dim)
    )
    cache = SimpleNamespace(dtype=kv_dtype)
    return query, cache, cache


@pytest.mark.parametrize(
    "name, expected",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.int8])
def test_split_kv_kernel_only_on_rtx_pro_6000(
    fake_gpu, name, expected, head_dim, kv_dtype
):
    fake_gpu(name)
    assert fused_decode_attn.use_pro6000_decode_attention(0) is expected
    assert (
        paged_attention_ops._use_pro6000_kernel(*_fake_call(head_dim, kv_dtype))
        is expected
    )


def test_cpu_device_never_uses_the_gpu_kernel(fake_gpu):
    fake_gpu("rtx_pro_6000")
    assert not fused_decode_attn.use_pro6000_decode_attention(
        torch.device("cpu")
    )


def test_previous_paths_selected_on_other_gpus(fake_gpu, monkeypatch):
    """On an H100 paged_attention_fwd must reach the previous dispatch."""
    fake_gpu("h100")
    calls = []
    monkeypatch.setattr(
        paged_attention_ops,
        "_paged_attention_previous",
        lambda *a, **k: calls.append(a) or "previous",
    )
    query = SimpleNamespace(
        ndim=3, shape=(1, 8, 128), is_cuda=True, device=torch.device("cuda:0")
    )
    key = SimpleNamespace(ndim=5, is_cuda=True, dtype=torch.bfloat16)
    value = SimpleNamespace(ndim=4, is_cuda=True, dtype=torch.bfloat16)
    table = SimpleNamespace(ndim=2, is_cuda=True)
    lens = SimpleNamespace(ndim=1, is_cuda=True)
    out = paged_attention_ops.paged_attention_fwd(
        query, key, value, table, lens, 0.1, 2, 16
    )
    assert out == "previous" and len(calls) == 1

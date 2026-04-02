"""A/B kernel benchmark: FlashInfer paged attention vs naive PyTorch SDPA.

Runs across multiple head/batch/sequence configurations to show
kernel-level speedup from FlashInfer integration.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def benchmark_ms(fn, iters: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def run_benchmarks() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    assert HAS_FLASHINFER, "FlashInfer required"

    device = torch.device("cuda")

    configs = [
        # (label, batch_size, num_heads, head_dim, block_size, kv_lens)
        ("Small (b=2, h=8, d=64)", 2, 8, 64, 16, [64, 48]),
        ("Medium (b=4, h=16, d=128)", 4, 16, 128, 16, [128, 96, 64, 32]),
        (
            "Large (b=8, h=32, d=128)",
            8,
            32,
            128,
            16,
            [256, 192, 128, 96, 64, 48, 32, 16],
        ),
        (
            "DeepSeek-V2 (b=4, h=128, d=128)",
            4,
            128,
            128,
            16,
            [512, 384, 256, 128],
        ),
    ]

    header = f"{'Config':<40} {'FlashInfer(ms)':>14} {'Naive(ms)':>12} {'Speedup':>10}"
    print(header)
    print("-" * 80)

    for label, batch_size, num_heads, head_dim, block_size, kv_lens in configs:
        qo_lens = [min(kv, 4) for kv in kv_lens]
        total_q = sum(qo_lens)

        qo_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        for i, ql in enumerate(qo_lens):
            qo_indptr[i + 1] = qo_indptr[i] + ql

        num_pages = batch_size
        paged_kv_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=device
        )
        paged_kv_indices = torch.arange(
            batch_size, dtype=torch.int32, device=device
        )
        paged_kv_last_page_len = torch.tensor(
            kv_lens, dtype=torch.int32, device=device
        )

        q = torch.randn(
            total_q, num_heads, head_dim, dtype=torch.float16, device=device
        )
        kv_cache = torch.randn(
            num_pages,
            2,
            block_size,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )

        # FlashInfer path
        workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD"
        )
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            block_size,
            causal=False,
        )

        def run_fi(w=wrapper, _q=q, _kv=kv_cache):
            w.run(_q, _kv)

        # Naive SDPA path
        def run_naive(
            _q=q,
            _kv=kv_cache,
            _qo_indptr=qo_indptr,
            _kv_lens=kv_lens,
            _bs=batch_size,
        ):
            for seq_id in range(_bs):
                q_start = int(_qo_indptr[seq_id])
                q_end = int(_qo_indptr[seq_id + 1])
                kv_len = _kv_lens[seq_id]
                q_seq = _q[q_start:q_end].transpose(0, 1).unsqueeze(0)
                k_seq = _kv[seq_id, 0, :kv_len].transpose(0, 1).unsqueeze(0)
                v_seq = _kv[seq_id, 1, :kv_len].transpose(0, 1).unsqueeze(0)
                F.scaled_dot_product_attention(
                    q_seq, k_seq, v_seq, dropout_p=0.0, is_causal=False
                )

        fi_ms = benchmark_ms(run_fi)
        naive_ms = benchmark_ms(run_naive)
        speedup = naive_ms / fi_ms if fi_ms > 0 else float("inf")
        print(f"{label:<40} {fi_ms:>14.4f} {naive_ms:>12.4f} {speedup:>9.2f}x")


if __name__ == "__main__":
    print("=== FlashInfer vs Naive SDPA — Kernel A/B Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer: {flashinfer.__version__ if HAS_FLASHINFER else 'N/A'}")
    print(f"PyTorch: {torch.__version__}")
    print()
    run_benchmarks()

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class Result:
    name: str
    status: str
    detail: str


def run_with_result(name: str, fn) -> Result:
    try:
        detail = fn()
        return Result(name=name, status="PASS", detail=str(detail))
    except Blocked as exc:
        return Result(name=name, status="BLOCKED", detail=str(exc))
    except Exception as exc:
        return Result(
            name=name, status="FAIL", detail=f"{type(exc).__name__}: {exc}"
        )


class Blocked(RuntimeError):
    pass


def cuda_env_summary() -> str:
    torch_version = getattr(torch, "__version__", "unknown")
    torch_version_module = torch.__dict__.get("version", None)
    torch_cuda = getattr(torch_version_module, "cuda", None)
    cuda_available = torch.cuda.is_available()
    return f"torch={torch_version}, torch_cuda={torch_cuda}, cuda_available={cuda_available}"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise Blocked(
            "CUDA unavailable for FlashInfer runtime. "
            f"{cuda_env_summary()}. "
            "Needed: working NVIDIA driver/runtime (nvidia-smi visible) and a CUDA-capable GPU."
        )


def import_flashinfer() -> Any:
    try:
        return importlib.import_module("flashinfer")
    except Exception as exc:
        raise Blocked(
            "FlashInfer import failed. "
            f"Reason: {type(exc).__name__}: {exc}. "
            "Needed: flashinfer-python wheel matching torch+CUDA (e.g., cu124/torch2.5) and a valid CUDA runtime."
        )


def make_mock_paged_kv(
    device: torch.device,
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 16,
) -> dict[str, Any]:
    qo_lens = [3, 2]
    kv_lens = [5, 4]
    if batch_size != 2:
        raise ValueError("mock batch currently expects batch_size=2")

    qo_indptr = torch.tensor(
        [0, qo_lens[0], qo_lens[0] + qo_lens[1]],
        dtype=torch.int32,
        device=device,
    )
    paged_kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(
        kv_lens, dtype=torch.int32, device=device
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q = torch.randn(
        sum(qo_lens), num_heads, head_dim, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        2,
        2,
        block_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    for page_idx, kv_len in enumerate(kv_lens):
        kv_cache[page_idx, 0, :kv_len] = torch.randn(
            kv_len, num_heads, head_dim, dtype=dtype, device=device
        )
        kv_cache[page_idx, 1, :kv_len] = torch.randn(
            kv_len, num_heads, head_dim, dtype=dtype, device=device
        )

    q_decode = torch.randn(
        batch_size, num_heads, head_dim, dtype=dtype, device=device
    )

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "batch_size": batch_size,
        "qo_lens": qo_lens,
        "kv_lens": kv_lens,
        "qo_indptr": qo_indptr,
        "paged_kv_indptr": paged_kv_indptr,
        "paged_kv_indices": paged_kv_indices,
        "paged_kv_last_page_len": paged_kv_last_page_len,
        "q": q,
        "q_decode": q_decode,
        "kv_cache": kv_cache,
    }


def reference_prefill_outputs(data: dict[str, Any]) -> torch.Tensor:
    outputs = []
    q = data["q"]
    kv_cache = data["kv_cache"]
    qo_indptr = data["qo_indptr"].tolist()
    kv_lens = data["kv_lens"]
    for seq_id in range(data["batch_size"]):
        q_start, q_end = qo_indptr[seq_id], qo_indptr[seq_id + 1]
        kv_len = kv_lens[seq_id]
        q_seq = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_seq = kv_cache[seq_id, 0, :kv_len].transpose(0, 1).unsqueeze(0)
        v_seq = kv_cache[seq_id, 1, :kv_len].transpose(0, 1).unsqueeze(0)
        out_seq = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, dropout_p=0.0, is_causal=False
        )
        outputs.append(out_seq.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def reference_decode_outputs(data: dict[str, Any]) -> torch.Tensor:
    q_decode = data["q_decode"]
    kv_cache = data["kv_cache"]
    kv_lens = data["kv_lens"]
    outputs = []
    for seq_id in range(data["batch_size"]):
        q_seq = q_decode[seq_id : seq_id + 1].transpose(0, 1).unsqueeze(0)
        kv_len = kv_lens[seq_id]
        k_seq = kv_cache[seq_id, 0, :kv_len].transpose(0, 1).unsqueeze(0)
        v_seq = kv_cache[seq_id, 1, :kv_len].transpose(0, 1).unsqueeze(0)
        out_seq = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, dropout_p=0.0, is_causal=False
        )
        outputs.append(out_seq.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


def benchmark_ms(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def test_import_and_version() -> str:
    flashinfer = import_flashinfer()
    version = getattr(flashinfer, "__version__", "unknown")
    has_prefill = hasattr(flashinfer, "BatchPrefillWithPagedKVCacheWrapper")
    has_decode = hasattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper")
    if not has_prefill or not has_decode:
        raise RuntimeError("Required paged attention wrappers not found")
    return f"flashinfer_version={version}, wrappers_present=True"


def test_paged_kv_tensor_construction() -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_mock_paged_kv(device=device)
    shape = tuple(data["kv_cache"].shape)
    expected = (2, 2, 16, 8, 64)
    if shape != expected:
        raise RuntimeError(
            f"Unexpected paged KV shape: got {shape}, expected {expected}"
        )
    return f"kv_cache_shape={shape}, device={device}, dtype={data['kv_cache'].dtype}"


def test_prefill_correctness() -> str:
    require_cuda()
    flashinfer = import_flashinfer()
    device = torch.device("cuda")
    data = make_mock_paged_kv(device=device)
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        data["qo_indptr"],
        data["paged_kv_indptr"],
        data["paged_kv_indices"],
        data["paged_kv_last_page_len"],
        data["num_heads"],
        data["num_heads"],
        data["head_dim"],
        data["block_size"],
        causal=False,
    )
    fi_out = wrapper.run(data["q"], data["kv_cache"])
    ref_out = reference_prefill_outputs(data)
    if not torch.allclose(fi_out, ref_out, atol=1e-2, rtol=1e-2):
        max_abs = (fi_out - ref_out).abs().max().item()
        raise RuntimeError(f"Prefill mismatch: max_abs_diff={max_abs:.6f}")
    return "allclose(atol=1e-2, rtol=1e-2)=True"


def test_decode_correctness() -> str:
    require_cuda()
    flashinfer = import_flashinfer()
    device = torch.device("cuda")
    data = make_mock_paged_kv(device=device)
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        data["paged_kv_indptr"],
        data["paged_kv_indices"],
        data["paged_kv_last_page_len"],
        data["num_heads"],
        data["num_heads"],
        data["head_dim"],
        data["block_size"],
        pos_encoding_mode="NONE",
        data_type=data["q_decode"].dtype,
    )
    fi_out = wrapper.run(data["q_decode"], data["kv_cache"])
    ref_out = reference_decode_outputs(data)
    if not torch.allclose(fi_out, ref_out, atol=1e-2, rtol=1e-2):
        max_abs = (fi_out - ref_out).abs().max().item()
        raise RuntimeError(f"Decode mismatch: max_abs_diff={max_abs:.6f}")
    return "allclose(atol=1e-2, rtol=1e-2)=True"


def test_latency() -> str:
    require_cuda()
    flashinfer = import_flashinfer()
    device = torch.device("cuda")
    data = make_mock_paged_kv(device=device)
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        data["qo_indptr"],
        data["paged_kv_indptr"],
        data["paged_kv_indices"],
        data["paged_kv_last_page_len"],
        data["num_heads"],
        data["num_heads"],
        data["head_dim"],
        data["block_size"],
        causal=False,
    )

    def run_flashinfer() -> None:
        wrapper.run(data["q"], data["kv_cache"])

    def run_naive() -> None:
        reference_prefill_outputs(data)

    fi_ms = benchmark_ms(run_flashinfer)
    naive_ms = benchmark_ms(run_naive)
    speedup = naive_ms / fi_ms if fi_ms > 0 else float("inf")
    return f"flashinfer_ms={fi_ms:.4f}, naive_ms={naive_ms:.4f}, speedup={speedup:.2f}x"


def test_coexistence() -> str:
    flashinfer = import_flashinfer()
    _ = flashinfer.__version__

    try:
        importlib.import_module("moe_infinity._store")
        return "Imported flashinfer and moe_infinity._store via package import"
    except Exception as first_exc:
        candidates: list[str] = []
        repo_candidate_pattern = os.path.join(
            os.path.dirname(__file__), "..", "..", "moe_infinity", "_store*.so"
        )
        candidates.extend(glob.glob(os.path.abspath(repo_candidate_pattern)))
        for search_root in sys.path:
            candidates.extend(
                glob.glob(
                    os.path.join(search_root, "moe_infinity", "_store*.so")
                )
            )
        candidates = sorted(set([p for p in candidates if os.path.isfile(p)]))
        if not candidates:
            raise Blocked(
                "moe_infinity._store import path unavailable. "
                f"Package import error: {type(first_exc).__name__}: {first_exc}. "
                "No _store shared library found on sys.path."
            )

        store_path = candidates[0]
        spec = importlib.util.spec_from_file_location("_store", store_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to create import spec for {store_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            "Imported flashinfer and loaded _store extension directly "
            f"from {store_path} after package import failure ({type(first_exc).__name__})."
        )


def main() -> None:
    print("=== FlashInfer Integration Validation (GO/NO-GO spike) ===")
    print(f"Environment: {cuda_env_summary()}")

    tests = [
        ("FlashInfer import/version", test_import_and_version),
        ("Paged KV tensor construction", test_paged_kv_tensor_construction),
        (
            "BatchPrefillWithPagedKVCacheWrapper correctness",
            test_prefill_correctness,
        ),
        (
            "BatchDecodeWithPagedKVCacheWrapper correctness",
            test_decode_correctness,
        ),
        ("FlashInfer latency vs naive attention", test_latency),
        ("FlashInfer + moe_infinity._store coexistence", test_coexistence),
    ]

    results = [run_with_result(name, fn) for name, fn in tests]

    print()
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    blocked = sum(1 for r in results if r.status == "BLOCKED")
    print()
    print(f"Summary: PASS={passed}, FAIL={failed}, BLOCKED={blocked}")

    if blocked > 0:
        print("Overall verdict: BLOCKED")
    elif failed > 0:
        print("Overall verdict: NO-GO")
    else:
        print("Overall verdict: GO")


if __name__ == "__main__":
    main()

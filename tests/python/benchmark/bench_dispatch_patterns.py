from __future__ import annotations

import statistics

import pytest

torch = pytest.importorskip("torch")

from tests.python.ops.conftest import BF16_ATOL, BF16_RTOL, requires_cuda

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5632
BATCH_SIZES = (1, 8, 32, 64, 128)

DTYPE_TOLERANCES = (
    (torch.float32, 1e-5, 1e-3),
    (torch.float16, 5e-3, 5e-3),
    (torch.bfloat16, BF16_ATOL, BF16_RTOL),
)


def _skip_if_dtype_unsupported(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this CUDA device")


def _matmul_out(lhs, rhs, out):
    if hasattr(torch, "matmul_out"):
        try:
            torch.matmul_out(lhs, rhs, out=out)
            return out
        except TypeError:
            try:
                torch.matmul_out(out, lhs, rhs)
                return out
            except TypeError:
                pass

    try:
        torch.matmul(lhs, rhs, out=out)
    except TypeError:
        out.copy_(torch.matmul(lhs, rhs))
    return out


def _silu_out(x, out):
    if hasattr(torch, "silu_out"):
        try:
            torch.silu_out(x, out=out)
            return out
        except TypeError:
            try:
                torch.silu_out(out, x)
                return out
            except TypeError:
                pass

    try:
        torch.silu(x, out=out)
    except (TypeError, AttributeError):
        out.copy_(torch.nn.functional.silu(x))
    return out


def _mul_out(lhs, rhs, out):
    if hasattr(torch, "mul_out"):
        try:
            torch.mul_out(lhs, rhs, out=out)
            return out
        except TypeError:
            try:
                torch.mul_out(out, lhs, rhs)
                return out
            except TypeError:
                pass

    try:
        torch.mul(lhs, rhs, out=out)
    except TypeError:
        out.copy_(lhs * rhs)
    return out


def _forward_preallocated(
    hidden_states,
    gate_proj,
    up_proj,
    down_proj,
):
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    intermediate_size = gate_proj.shape[0]

    gate_out = torch.empty(
        (batch_size, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    up_out = torch.empty_like(gate_out)
    act_out = torch.empty_like(gate_out)
    fused_out = torch.empty_like(gate_out)
    output = torch.empty(
        (batch_size, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    _matmul_out(hidden_states, gate_proj.t(), gate_out)
    _matmul_out(hidden_states, up_proj.t(), up_out)
    _silu_out(gate_out, act_out)
    _mul_out(act_out, up_out, fused_out)
    _matmul_out(fused_out, down_proj.t(), output)
    return output


def _forward_standard(
    hidden_states,
    gate_proj,
    up_proj,
    down_proj,
):
    gate = torch.matmul(hidden_states, gate_proj.t())
    up = torch.matmul(hidden_states, up_proj.t())
    fused = torch.nn.functional.silu(gate) * up
    return torch.matmul(fused, down_proj.t())


def _make_preallocated_runner(
    hidden_states,
    gate_proj,
    up_proj,
    down_proj,
):
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    intermediate_size = gate_proj.shape[0]

    gate_out = torch.empty(
        (batch_size, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    up_out = torch.empty_like(gate_out)
    act_out = torch.empty_like(gate_out)
    fused_out = torch.empty_like(gate_out)
    output = torch.empty(
        (batch_size, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    def _run():
        _matmul_out(hidden_states, gate_proj.t(), gate_out)
        _matmul_out(hidden_states, up_proj.t(), up_out)
        _silu_out(gate_out, act_out)
        _mul_out(act_out, up_out, fused_out)
        _matmul_out(fused_out, down_proj.t(), output)
        return output

    return _run


def _measure_latency_ms(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    return statistics.fmean(times_ms), statistics.pstdev(times_ms)


@requires_cuda
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCES)
def test_mlp_forward_dispatch_patterns_numerical_equivalence(
    batch_size,
    dtype,
    atol,
    rtol,
):
    _skip_if_dtype_unsupported(dtype)

    torch.manual_seed(1234)
    device = torch.device("cuda")

    # Scale down for fp16 to avoid overflow (hidden_size=2048 * intermediate_size=5632
    # causes accumulation overflow with unit-variance random weights)
    scale = (
        0.01
        if dtype == torch.float16
        else 0.1
        if dtype == torch.bfloat16
        else 1.0
    )

    hidden_states = (
        torch.randn(batch_size, HIDDEN_SIZE, device=device, dtype=dtype) * scale
    )
    gate_proj = (
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
        * scale
    )
    up_proj = (
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
        * scale
    )
    down_proj = (
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype)
        * scale
    )

    output_a = _forward_preallocated(
        hidden_states, gate_proj, up_proj, down_proj
    )
    output_b = _forward_standard(hidden_states, gate_proj, up_proj, down_proj)

    assert torch.allclose(output_a, output_b, atol=atol, rtol=rtol), (
        f"Outputs diverged for dtype={dtype}, batch_size={batch_size}; "
        f"max_abs_diff={(output_a - output_b).abs().max().item():.6e}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCES)
def test_accumulation_patterns_numerical_equivalence(
    dtype,
    atol,
    rtol,
):
    _skip_if_dtype_unsupported(dtype)

    torch.manual_seed(2026)
    device = torch.device("cuda")

    num_experts = 8
    num_tokens = 32
    hidden_size = 512
    top_k = 2

    expert_outputs = torch.randn(
        num_experts, num_tokens, hidden_size, device=device, dtype=dtype
    )

    router_logits = torch.randn(num_tokens, num_experts, device=device)
    router_weights = torch.softmax(router_logits, dim=-1).to(dtype)
    topk_experts = torch.topk(router_logits, k=top_k, dim=-1).indices
    router_mask = torch.zeros(
        num_tokens, num_experts, device=device, dtype=torch.bool
    )
    router_mask.scatter_(1, topk_experts, True)

    final_hidden_states_a = torch.zeros(
        num_tokens, hidden_size, device=device, dtype=dtype
    )
    for expert_idx in range(num_experts):
        token_mask = router_mask[:, expert_idx]
        if not token_mask.any():
            continue
        token_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)
        weighted_output = expert_outputs[
            expert_idx, token_indices
        ] * router_weights[token_indices, expert_idx].unsqueeze(1)
        final_hidden_states_a.index_add_(0, token_indices, weighted_output)

    final_hidden_states_b = torch.zeros_like(final_hidden_states_a)
    for expert_idx in range(num_experts):
        token_mask = router_mask[:, expert_idx]
        if not token_mask.any():
            continue
        final_hidden_states_b[token_mask] += expert_outputs[
            expert_idx, token_mask
        ] * router_weights[token_mask, expert_idx].unsqueeze(1)

    assert torch.allclose(
        final_hidden_states_a,
        final_hidden_states_b,
        atol=atol,
        rtol=rtol,
    ), "General-token accumulation paths diverged"

    single_token_outputs = torch.randn(
        num_experts, 1, hidden_size, device=device, dtype=dtype
    )
    single_router_weights = torch.softmax(
        torch.randn(1, num_experts, device=device), dim=-1
    ).to(dtype)

    final_hidden_states_single_a = torch.zeros(
        1, hidden_size, device=device, dtype=dtype
    )
    for expert_idx in range(num_experts):
        final_hidden_states_single_a.add_(
            single_token_outputs[expert_idx]
            * single_router_weights[:, expert_idx]
        )

    final_hidden_states_single_b = torch.zeros_like(
        final_hidden_states_single_a
    )
    single_token_mask = torch.ones(1, device=device, dtype=torch.bool)
    for expert_idx in range(num_experts):
        final_hidden_states_single_b[single_token_mask] += single_token_outputs[
            expert_idx, single_token_mask
        ] * single_router_weights[single_token_mask, expert_idx].unsqueeze(1)

    assert torch.allclose(
        final_hidden_states_single_a,
        final_hidden_states_single_b,
        atol=atol,
        rtol=rtol,
    ), "Batch-size=1 accumulation paths diverged"


@requires_cuda
def test_dispatch_pattern_latency_benchmark():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    atol, rtol = (
        (BF16_ATOL, BF16_RTOL) if dtype == torch.bfloat16 else (5e-3, 5e-3)
    )

    torch.manual_seed(7)
    device = torch.device("cuda")

    gate_proj = torch.randn(
        INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype
    )
    up_proj = torch.randn(
        INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype
    )
    down_proj = torch.randn(
        HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype
    )

    print("\nDispatch pattern latency benchmark (ms)")
    print("batch | prealloc_mean ± std | standard_mean ± std")
    print("------|----------------------|--------------------")

    for batch_size in BATCH_SIZES:
        hidden_states = torch.randn(
            batch_size, HIDDEN_SIZE, device=device, dtype=dtype
        )

        preallocated_runner = _make_preallocated_runner(
            hidden_states, gate_proj, up_proj, down_proj
        )

        def standard_runner():
            return _forward_standard(
                hidden_states, gate_proj, up_proj, down_proj
            )

        output_a = preallocated_runner()
        output_b = standard_runner()
        assert torch.allclose(output_a, output_b, atol=atol, rtol=rtol)

        prealloc_mean, prealloc_std = _measure_latency_ms(preallocated_runner)
        standard_mean, standard_std = _measure_latency_ms(standard_runner)

        print(
            f"{batch_size:>5} | "
            f"{prealloc_mean:>8.3f} ± {prealloc_std:<8.3f} | "
            f"{standard_mean:>8.3f} ± {standard_std:<8.3f}"
        )

        assert prealloc_mean > 0.0
        assert standard_mean > 0.0

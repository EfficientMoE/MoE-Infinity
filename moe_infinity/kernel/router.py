import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def launch_softmax(input_tensor):
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if BLOCK_SIZE <= 2048 else 2
    output = torch.empty_like(input_tensor)
    softmax_kernel[(n_rows,)](
        output,
        input_tensor,
        input_tensor.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return output


@triton.jit
def fused_softmax_topk_kernel_nobias(
    hidden_ptr,
    weight_ptr,
    routing_mask_ptr,
    routing_weight_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    normalize_topk: tl.constexpr,
):
    batch_id = tl.program_id(0)
    off_e = tl.arange(0, BLOCK_E)

    logits = tl.zeros([BLOCK_E], dtype=tl.float32)

    for h in range(H):
        h_val = tl.load(hidden_ptr + batch_id * H + h)
        w_ptr = weight_ptr + off_e * H + h
        valid = off_e < E
        w_val = tl.load(w_ptr, mask=valid, other=0.0)
        logits = tl.where(valid, logits + h_val * w_val, logits)

    logits = tl.where(off_e < E, logits, -float("inf"))

    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    remaining = tl.where(off_e < E, probs, -float("inf"))
    top_mask = tl.zeros([BLOCK_E], dtype=tl.int1)

    for _k in range(TOPK):
        cur_max = tl.max(remaining, axis=0)
        is_max = remaining == cur_max
        first = is_max & (tl.cumsum(is_max.to(tl.int32), axis=0) == 1)
        top_mask = top_mask | first
        remaining = tl.where(first, -float("inf"), remaining)

    top_probs = tl.where(top_mask, probs, tl.zeros([BLOCK_E], dtype=tl.float32))

    if normalize_topk:
        top_sum = tl.sum(top_probs, axis=0)
        top_probs = top_probs / top_sum

    tl.store(routing_mask_ptr + batch_id * E + off_e, top_mask, mask=off_e < E)
    tl.store(
        routing_weight_ptr + batch_id * E + off_e,
        top_probs.to(tl.bfloat16),
        mask=off_e < E,
    )


def launch_fused_softmax_topk_nobias(
    hidden_states, weight, top_k, normalize_topk=True
):
    B, H = hidden_states.shape
    E = weight.shape[0]
    dtype = hidden_states.dtype

    routing_mask = torch.zeros(
        (B, E), dtype=torch.bool, device=hidden_states.device
    )
    routing_weight = torch.zeros(
        (B, E), dtype=dtype, device=hidden_states.device
    )

    BLOCK_E = triton.next_power_of_2(E)

    fused_softmax_topk_kernel_nobias[(B,)](
        hidden_states,
        weight,
        routing_mask,
        routing_weight,
        B=B,
        H=H,
        E=E,
        TOPK=top_k,
        BLOCK_E=BLOCK_E,
        normalize_topk=normalize_topk,
    )

    return routing_mask, routing_weight

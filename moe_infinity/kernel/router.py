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
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def fused_softmax_topk_kernel(
    hidden_ptr,  # [B, H]
    weight_ptr,  # [H, E]
    routing_mask_ptr,  # [B, E] (bool)
    routing_weight_ptr,  # [B, E] (float16)
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    normalize_topk: tl.constexpr,  # New!
):
    batch_id = tl.program_id(0)
    off_e = tl.arange(0, BLOCK_E)

    # [E] logits init
    logits = tl.zeros([BLOCK_E], dtype=tl.float32)

    for h in range(0, H):
        hidden_val = tl.load(hidden_ptr + batch_id * H + h)
        weight_row = weight_ptr + h * E + off_e
        w = tl.load(weight_row, mask=off_e < E, other=0.0)
        logits += hidden_val * w

    logits += tl.load(bias_ptr + off_e, mask=off_e < E, other=0.0)

    # Compute softmax
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Top-k selection (insertion sort)
    top_vals = tl.full([TOPK], -float("inf"), dtype=tl.float32)
    top_idxs = tl.full([TOPK], -1, dtype=tl.int32)

    for i in range(BLOCK_E):
        p = probs[i]
        idx = i

        # insert into sorted list
        for j in range(TOPK):
            if p > top_vals[j]:
                for k in range(TOPK - 1, j, -1):
                    top_vals[k] = top_vals[k - 1]
                    top_idxs[k] = top_idxs[k - 1]
                top_vals[j] = p
                top_idxs[j] = idx
                break

    # normalize
    if normalize_topk:
        sum_top = tl.sum(top_vals)
        top_vals = top_vals / sum_top

    for i in range(TOPK):
        expert_idx = top_idxs[i]
        expert_val = top_vals[i]
        if expert_idx >= 0:
            tl.store(routing_mask_ptr + batch_id * E + expert_idx, True)
            tl.store(
                routing_weight_ptr + batch_id * E + expert_idx,
                expert_val.to(tl.float16),
            )


def launch_fused_softmax_topk(hidden_states, weight, bias, top_k):
    B, H = hidden_states.shape
    E = weight.shape[1]
    dtype = hidden_states.dtype

    routing_mask = torch.zeros(
        (B, E), dtype=torch.bool, device=hidden_states.device
    )
    routing_weight = torch.zeros(
        (B, E), dtype=dtype, device=hidden_states.device
    )

    BLOCK_E = 32  # Must divide E

    fused_softmax_topk_kernel[(B,)](
        hidden_states,
        weight,
        bias,
        routing_mask,
        routing_weight,
        B=B,
        H=H,
        E=E,
        TOPK=top_k,
        BLOCK_E=BLOCK_E,
    )

    return routing_mask, routing_weight


@triton.jit
def fused_softmax_topk_kernel_nobias(
    hidden_ptr,  # [B, H]
    weight_ptr,  # [E, H]
    routing_mask_ptr,  # [B, E]
    routing_weight_ptr,  # [B, E]
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    normalize_topk: tl.constexpr,
):
    batch_id = tl.program_id(0)
    off_e = tl.arange(0, BLOCK_E)

    logits = tl.full([BLOCK_E], -float("inf"), dtype=tl.float32)

    for h in range(H):
        h_val = tl.load(hidden_ptr + batch_id * H + h)
        w_ptr = weight_ptr + off_e * H + h
        valid = off_e < E
        w_val = tl.load(w_ptr, mask=valid, other=0.0)
        logits = tl.where(valid, logits + h_val * w_val, logits)

    # Softmax
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Top-k (insertion sort)
    top_vals = tl.full([TOPK], -float("inf"), dtype=tl.float32)
    top_idxs = tl.full([TOPK], -1, dtype=tl.int32)

    for i in range(BLOCK_E):
        if i < E:
            p = tl.load(probs + batch_id * E + i)
            idx = i

            for j in range(TOPK):
                if p > top_vals[j]:
                    for k in range(TOPK - 1, j, -1):
                        top_vals[k] = top_vals[k - 1]
                        top_idxs[k] = top_idxs[k - 1]
                    top_vals[j] = p
                    top_idxs[j] = idx
                    break

    if normalize_topk:
        sum_top = tl.sum(top_vals)
        top_vals = top_vals / sum_top

    for i in range(TOPK):
        idx = top_idxs[i]
        val = top_vals[i]
        if idx >= 0:
            tl.store(routing_mask_ptr + batch_id * E + idx, True)
            tl.store(
                routing_weight_ptr + batch_id * E + idx, val.to(tl.float16)
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

    BLOCK_E = 128
    assert BLOCK_E >= E, "BLOCK_E must be greater than or equal to E"

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

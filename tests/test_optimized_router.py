import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def expert_routing_small_batch_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    router_logits_ptr,
    router_mask_ptr,
    routing_weights_ptr,
    B,
    D,
    E,
    K,
    norm_topk_prob: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Optimized kernel for small batch size with many experts.
    Parallelizes over experts instead of batch.
    """
    # Grid: (num_expert_blocks, B)
    expert_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    if batch_id >= B:
        return

    # Process a block of experts
    expert_start = expert_block_id * BLOCK_SIZE_E
    expert_idx = tl.arange(0, BLOCK_SIZE_E) + expert_start
    expert_mask = expert_idx < E

    # Compute dot product for this block of experts
    hidden_offset = batch_id * D
    acc = tl.zeros([BLOCK_SIZE_E], dtype=tl.float32)

    # Chunked matrix multiplication
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_idx = tl.arange(0, BLOCK_SIZE_D) + d_start
        d_mask = d_idx < D

        # Load hidden states once
        hidden_vals = tl.load(
            hidden_states_ptr + hidden_offset + d_idx, mask=d_mask, other=0.0
        ).to(tl.float32)

        # Load weights for all experts in block
        weight_offset = expert_idx[:, None] * D + d_idx[None, :]
        weight_vals = tl.load(
            gate_weight_ptr + weight_offset,
            mask=expert_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate
        acc = acc + tl.sum(weight_vals * hidden_vals[None, :], axis=1)

    # Add bias and store logits
    bias_vals = tl.load(
        gate_bias_ptr + expert_idx, mask=expert_mask, other=0.0
    ).to(tl.float32)

    logits = acc + bias_vals

    # Store logits for this block
    logits_offset = batch_id * E + expert_idx
    tl.store(router_logits_ptr + logits_offset, logits, mask=expert_mask)


@triton.jit
def softmax_topk_small_batch_kernel(
    router_logits_ptr,
    router_mask_ptr,
    routing_weights_ptr,
    B,
    E,
    K,
    norm_topk_prob: tl.constexpr,
):
    """
    Softmax and top-k selection for small batch.
    One thread block handles one sample entirely.
    """
    batch_id = tl.program_id(0)

    if batch_id >= B:
        return

    base_offset = batch_id * E

    # Step 1: Find max for numerical stability
    max_logit = -float("inf")
    for e in range(E):
        logit = tl.load(router_logits_ptr + base_offset + e).to(tl.float32)
        max_logit = tl.maximum(max_logit, logit)

    # Step 2: Compute exp sum
    exp_sum = 0.0
    for e in range(E):
        logit = tl.load(router_logits_ptr + base_offset + e).to(tl.float32)
        exp_sum = exp_sum + tl.exp(logit - max_logit)

    # Step 3: Initialize outputs
    for e in range(E):
        tl.store(router_mask_ptr + base_offset + e, False)
        tl.store(routing_weights_ptr + base_offset + e, 0.0)

    # Step 4: Top-k selection
    selected_sum = 0.0

    for k in range(K):
        best_idx = -1
        best_prob = -1.0

        # Find best unselected expert
        for e in range(E):
            is_selected = tl.load(router_mask_ptr + base_offset + e)

            if is_selected == 0:
                logit = tl.load(router_logits_ptr + base_offset + e).to(
                    tl.float32
                )
                prob = tl.exp(logit - max_logit) / exp_sum

                if prob > best_prob:
                    best_prob = prob
                    best_idx = e

        # Select this expert
        if best_idx >= 0 and best_prob > 0:
            tl.store(router_mask_ptr + base_offset + best_idx, True)
            tl.store(routing_weights_ptr + base_offset + best_idx, best_prob)
            selected_sum = selected_sum + best_prob

    # Step 5: Normalize if requested
    if norm_topk_prob and selected_sum > 0:
        for e in range(E):
            is_selected = tl.load(router_mask_ptr + base_offset + e)
            if is_selected:
                weight = tl.load(routing_weights_ptr + base_offset + e)
                tl.store(
                    routing_weights_ptr + base_offset + e, weight / selected_sum
                )


@triton.jit
def fused_gate_softmax_topk_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    router_logits_ptr,
    router_mask_ptr,
    routing_weights_ptr,
    topk_indices_ptr,
    topk_values_ptr,
    B,
    D,
    E,
    K,
    norm_topk_prob: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Fully fused kernel for B=1 case.
    Computes everything in a single pass.
    """
    batch_id = tl.program_id(0)

    if batch_id >= B:
        return

    hidden_offset = batch_id * D

    # Online computation of logits, max, and exp_sum
    online_max = -float("inf")
    online_sum = 0.0

    # First pass: compute logits and track statistics
    for e in range(E):
        acc = 0.0

        # Compute dot product
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_idx = tl.arange(0, BLOCK_SIZE_D) + d_start
            d_mask = d_idx < D

            hidden_vals = tl.load(
                hidden_states_ptr + hidden_offset + d_idx,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            weight_vals = tl.load(
                gate_weight_ptr + e * D + d_idx, mask=d_mask, other=0.0
            ).to(tl.float32)

            acc = acc + tl.sum(hidden_vals * weight_vals)

        # Add bias
        bias = tl.load(gate_bias_ptr + e).to(tl.float32)
        logit = acc + bias

        # Store logit
        tl.store(router_logits_ptr + batch_id * E + e, logit)

        # Update online max and sum
        if logit > online_max:
            online_sum = online_sum * tl.exp(online_max - logit)
            online_max = logit

        online_sum = online_sum + tl.exp(logit - online_max)

    # Initialize outputs
    for e in range(E):
        tl.store(router_mask_ptr + batch_id * E + e, False)
        tl.store(routing_weights_ptr + batch_id * E + e, 0.0)

    # Top-k selection with softmax probabilities
    selected_sum = 0.0

    for k in range(K):
        best_idx = -1
        best_prob = -1.0

        for e in range(E):
            is_selected = tl.load(router_mask_ptr + batch_id * E + e)

            if is_selected == 0:
                logit = tl.load(router_logits_ptr + batch_id * E + e).to(
                    tl.float32
                )
                prob = tl.exp(logit - online_max) / online_sum

                if prob > best_prob:
                    best_prob = prob
                    best_idx = e

        if best_idx >= 0 and best_prob > 0:
            tl.store(router_mask_ptr + batch_id * E + best_idx, True)
            tl.store(routing_weights_ptr + batch_id * E + best_idx, best_prob)
            tl.store(topk_indices_ptr + batch_id * K + k, best_idx)
            tl.store(topk_values_ptr + batch_id * K + k, best_prob)
            selected_sum = selected_sum + best_prob

    # Normalize if requested
    if norm_topk_prob and selected_sum > 0:
        for e in range(E):
            is_selected = tl.load(router_mask_ptr + batch_id * E + e)
            if is_selected == 1:
                weight = tl.load(routing_weights_ptr + batch_id * E + e)
                tl.store(
                    routing_weights_ptr + batch_id * E + e,
                    weight / selected_sum,
                )

        # Also update topk_values
        for k in range(K):
            val = tl.load(topk_values_ptr + batch_id * K + k)
            tl.store(topk_values_ptr + batch_id * K + k, val / selected_sum)


def expert_routing_small_batch(
    hidden_states,
    gate_weight,
    gate_bias,
    num_experts,
    top_k,
    norm_topk_prob=False,
):
    """
    Optimized expert routing for small batch sizes with many experts.
    """
    B, D = hidden_states.shape
    E = num_experts
    K = top_k

    # Ensure contiguous
    hidden_states = hidden_states.contiguous()
    gate_weight = gate_weight.contiguous()
    gate_bias = gate_bias.contiguous()

    # Allocate outputs
    device = hidden_states.device
    dtype = hidden_states.dtype
    router_logits = torch.empty(B, E, dtype=torch.float32, device=device)
    router_mask = torch.zeros(B, E, dtype=torch.bool, device=device)
    routing_weights = torch.zeros(B, E, dtype=dtype, device=device)

    if B == 1:
        # For B=1, use fully fused kernel
        topk_indices = torch.empty(B, K, dtype=torch.int64, device=device)
        topk_values = torch.empty(B, K, dtype=dtype, device=device)

        BLOCK_SIZE_D = min(64, triton.next_power_of_2(D))
        grid = (B,)

        fused_gate_softmax_topk_kernel[grid](
            hidden_states,
            gate_weight,
            gate_bias,
            router_logits,
            router_mask,
            routing_weights,
            topk_indices,
            topk_values,
            B,
            D,
            E,
            K,
            norm_topk_prob,
            BLOCK_SIZE_D,
        )
    else:
        # For small B > 1, use two-kernel approach
        # Kernel 1: Parallel over experts
        BLOCK_SIZE_E = min(32, triton.next_power_of_2(E))
        BLOCK_SIZE_D = min(64, triton.next_power_of_2(D))
        num_expert_blocks = triton.cdiv(E, BLOCK_SIZE_E)

        grid = (num_expert_blocks, B)

        expert_routing_small_batch_kernel[grid](
            hidden_states,
            gate_weight,
            gate_bias,
            router_logits,
            router_mask,
            routing_weights,
            B,
            D,
            E,
            K,
            norm_topk_prob,
            BLOCK_SIZE_E,
            BLOCK_SIZE_D,
        )

        # Kernel 2: Softmax and top-k per sample
        grid = (B,)

        softmax_topk_small_batch_kernel[grid](
            router_logits, router_mask, routing_weights, B, E, K, norm_topk_prob
        )

    # Convert types
    router_logits = router_logits.to(dtype)

    return router_logits, router_mask, routing_weights


def prepare_expert_route_optimized(
    gate_module, hidden_states, num_experts, top_k, norm_topk_prob=False
):
    """
    Optimized routing that automatically selects the best implementation.
    """
    B, D = hidden_states.shape

    # Extract weights and bias
    gate_weight = gate_module.weight
    gate_bias = (
        gate_module.bias
        if gate_module.bias is not None
        else torch.zeros(
            num_experts, device=hidden_states.device, dtype=hidden_states.dtype
        )
    )

    # Choose implementation based on batch size
    if B <= 32:  # Small batch
        return expert_routing_small_batch(
            hidden_states,
            gate_weight,
            gate_bias,
            num_experts,
            top_k,
            norm_topk_prob,
        )
    else:
        # For larger batches, PyTorch is often more efficient
        router_logits = gate_module(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1
        )

        if norm_topk_prob:
            routing_weights_topk = (
                routing_weights_topk
                / routing_weights_topk.sum(dim=-1, keepdim=True)
            )

        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        router_mask = torch.zeros(
            B, num_experts, dtype=torch.bool, device=hidden_states.device
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            B,
            num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        routing_weights_mask.scatter_add_(
            1, selected_experts, routing_weights_topk
        )

        return router_logits, router_mask, routing_weights_mask


# Test and benchmark
if __name__ == "__main__":
    import time

    # Test configurations
    configs = [
        (1, 2048, 128, 8),  # Your specific case
        (1, 768, 128, 8),  # Smaller hidden dim
        (4, 2048, 128, 8),  # Small batch
        (32, 2048, 128, 8),  # Medium batch
        (1, 2048, 64, 4),  # Fewer experts
        (1, 2048, 256, 16),  # More experts
    ]

    for B, D, E, K in configs:
        print(f"\nConfiguration: B={B}, D={D}, E={E}, K={K}")

        device = torch.cuda.current_device()
        dtype = torch.float16

        # Create inputs
        hidden_states = torch.randn(B, D, device=device, dtype=dtype)
        gate = torch.nn.Linear(D, E, device=device, dtype=dtype)

        # Test correctness
        router_logits, router_mask, routing_weights = (
            prepare_expert_route_optimized(
                gate, hidden_states, E, K, norm_topk_prob=True
            )
        )

        # Reference implementation
        router_logits_ref = gate(hidden_states)
        routing_weights_ref = F.softmax(
            router_logits_ref, dim=1, dtype=torch.float
        )
        routing_weights_topk, selected_experts = torch.topk(
            routing_weights_ref, K, dim=-1
        )
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1, keepdim=True
        )
        routing_weights_topk = routing_weights_topk.to(dtype)

        router_mask_ref = torch.zeros(B, E, dtype=torch.bool, device=device)
        router_mask_ref.scatter_(1, selected_experts, True)

        routing_weights_mask_ref = torch.zeros(B, E, dtype=dtype, device=device)
        routing_weights_mask_ref.scatter_add_(
            1, selected_experts, routing_weights_topk
        )

        # Check correctness
        logits_match = torch.allclose(
            router_logits, router_logits_ref, rtol=1e-2, atol=1e-3
        )
        mask_match = torch.equal(router_mask, router_mask_ref)
        weights_match = torch.allclose(
            routing_weights, routing_weights_mask_ref, rtol=1e-2, atol=1e-3
        )

        print(
            f"Correctness - Logits: {logits_match}, Mask: {mask_match}, Weights: {weights_match}"
        )

        # Benchmark
        n_iters = 100

        # Warmup
        for _ in range(10):
            _ = prepare_expert_route_optimized(gate, hidden_states, E, K, True)
            torch.cuda.synchronize()

        # Optimized implementation
        start = time.time()
        for _ in range(n_iters):
            _ = prepare_expert_route_optimized(gate, hidden_states, E, K, True)
        torch.cuda.synchronize()
        opt_time = time.time() - start

        # PyTorch baseline
        start = time.time()
        for _ in range(n_iters):
            router_logits = gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights_topk, selected_experts = torch.topk(
                routing_weights, K, dim=-1
            )
            routing_weights_topk = (
                routing_weights_topk
                / routing_weights_topk.sum(dim=-1, keepdim=True)
            )
            routing_weights_topk = routing_weights_topk.to(dtype)

            router_mask = torch.zeros(B, E, dtype=torch.bool, device=device)
            router_mask.scatter_(1, selected_experts, True)

            routing_weights_mask = torch.zeros(B, E, dtype=dtype, device=device)
            routing_weights_mask.scatter_add_(
                1, selected_experts, routing_weights_topk
            )
        torch.cuda.synchronize()
        pytorch_time = time.time() - start

        print(
            f"Optimized: {opt_time / n_iters * 1000:.3f} ms, PyTorch: {pytorch_time / n_iters * 1000:.3f} ms"
        )
        print(f"Speedup: {pytorch_time / opt_time:.2f}x")

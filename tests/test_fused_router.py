import time

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def fused_gate_softmax_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    router_logits_ptr,
    softmax_out_ptr,
    B,
    D,
    E,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
):
    """
    Fused kernel for gate computation and softmax.
    """
    pid = tl.program_id(0)

    if pid >= B:
        return

    # Step 1: Compute gate(hidden_states) = hidden_states @ gate_weight.T + gate_bias
    hidden_offset = pid * D

    # Process experts in blocks
    for expert_block_start in range(0, E, BLOCK_SIZE_E):
        expert_idx = tl.arange(0, BLOCK_SIZE_E) + expert_block_start
        expert_mask = expert_idx < E

        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE_E], dtype=tl.float32)

        # Compute dot product
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_idx = tl.arange(0, BLOCK_SIZE_D) + d_start
            d_mask = d_idx < D

            # Load hidden states
            hidden_vals = tl.load(
                hidden_states_ptr + hidden_offset + d_idx,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            # Load weights
            weight_offset = expert_idx[:, None] * D + d_idx[None, :]
            weight_vals = tl.load(
                gate_weight_ptr + weight_offset,
                mask=expert_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Accumulate
            acc = acc + tl.sum(weight_vals * hidden_vals[None, :], axis=1)

        # Add bias
        bias_vals = tl.load(
            gate_bias_ptr + expert_idx, mask=expert_mask, other=0.0
        ).to(tl.float32)

        logits = acc + bias_vals

        # Store logits
        tl.store(
            router_logits_ptr + pid * E + expert_idx, logits, mask=expert_mask
        )

    # Step 2: Compute softmax
    # Find max
    max_logit = -float("inf")
    for e in range(0, E, BLOCK_SIZE_E):
        e_idx = tl.arange(0, BLOCK_SIZE_E) + e
        e_mask = e_idx < E
        logits = tl.load(
            router_logits_ptr + pid * E + e_idx,
            mask=e_mask,
            other=-float("inf"),
        ).to(tl.float32)
        block_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, block_max)

    # Compute exp and sum
    exp_sum = 0.0
    for e in range(0, E, BLOCK_SIZE_E):
        e_idx = tl.arange(0, BLOCK_SIZE_E) + e
        e_mask = e_idx < E
        logits = tl.load(
            router_logits_ptr + pid * E + e_idx,
            mask=e_mask,
            other=-float("inf"),
        ).to(tl.float32)
        exp_vals = tl.exp(logits - max_logit)
        tl.store(softmax_out_ptr + pid * E + e_idx, exp_vals, mask=e_mask)
        exp_sum = exp_sum + tl.sum(tl.where(e_mask, exp_vals, 0.0))

    # Normalize
    for e in range(0, E, BLOCK_SIZE_E):
        e_idx = tl.arange(0, BLOCK_SIZE_E) + e
        e_mask = e_idx < E
        exp_vals = tl.load(
            softmax_out_ptr + pid * E + e_idx, mask=e_mask, other=0.0
        ).to(tl.float32)
        probs = exp_vals / exp_sum
        tl.store(softmax_out_ptr + pid * E + e_idx, probs, mask=e_mask)


@triton.jit
def scatter_topk_kernel(
    topk_values_ptr,
    topk_indices_ptr,
    router_mask_ptr,
    routing_weights_mask_ptr,
    B,
    E,
    K,
    norm_topk_prob: tl.constexpr,
):
    """Scatter top-k values to create masks."""
    pid = tl.program_id(0)
    if pid >= B:
        return

    # Initialize outputs
    for e in range(E):
        tl.store(router_mask_ptr + pid * E + e, False)
        tl.store(routing_weights_mask_ptr + pid * E + e, 0.0)

    # Compute sum for normalization if needed
    topk_sum = 1.0  # Default to 1.0 to avoid division issues
    if norm_topk_prob:
        topk_sum = 0.0
        for k in range(K):
            val = tl.load(topk_values_ptr + pid * K + k).to(tl.float32)
            topk_sum = topk_sum + val
        # Avoid division by zero
        topk_sum = tl.maximum(topk_sum, 1e-10)

    # Scatter top-k values
    for k in range(K):
        idx = tl.load(topk_indices_ptr + pid * K + k).to(tl.int32)
        val = tl.load(topk_values_ptr + pid * K + k)

        # Always perform the division but use topk_sum=1.0 when not normalizing
        val_normalized = (val.to(tl.float32) / topk_sum).to(val.dtype)

        tl.store(router_mask_ptr + pid * E + idx, True)
        tl.store(routing_weights_mask_ptr + pid * E + idx, val_normalized)


class FusedExpertRouting(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        gate_weight,
        gate_bias,
        num_experts,
        top_k,
        norm_topk_prob,
    ):
        B, D = hidden_states.shape
        E = num_experts
        K = top_k

        # Ensure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        gate_weight = gate_weight.contiguous()
        gate_bias = gate_bias.contiguous()

        # Allocate intermediate tensors
        router_logits = torch.empty(
            B, E, dtype=torch.float32, device=hidden_states.device
        )
        softmax_probs = torch.empty(
            B, E, dtype=torch.float32, device=hidden_states.device
        )

        # Step 1: Fused gate + softmax computation
        BLOCK_SIZE_D = min(32, triton.next_power_of_2(D))
        BLOCK_SIZE_E = min(32, triton.next_power_of_2(E))

        grid = (B,)
        fused_gate_softmax_kernel[grid](
            hidden_states,
            gate_weight,
            gate_bias,
            router_logits,
            softmax_probs,
            B,
            D,
            E,
            BLOCK_SIZE_D,
            BLOCK_SIZE_E,
        )

        # Step 2: Use PyTorch's efficient top-k
        routing_weights_topk, selected_experts = torch.topk(
            softmax_probs, K, dim=-1
        )

        # Convert types
        router_logits = router_logits.to(hidden_states.dtype)
        routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

        # Step 3: Scatter to create masks
        router_mask = torch.zeros(
            B, E, dtype=torch.bool, device=hidden_states.device
        )
        routing_weights_mask = torch.zeros(
            B, E, dtype=hidden_states.dtype, device=hidden_states.device
        )

        scatter_topk_kernel[grid](
            routing_weights_topk,
            selected_experts,
            router_mask,
            routing_weights_mask,
            B,
            E,
            K,
            norm_topk_prob,
        )

        # Save for backward
        ctx.save_for_backward(
            hidden_states,
            gate_weight,
            router_logits,
            router_mask,
            routing_weights_mask,
        )
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob

        return router_logits, router_mask, routing_weights_mask

    @staticmethod
    def backward(
        ctx, grad_router_logits, grad_router_mask, grad_routing_weights_mask
    ):
        # Implement backward pass if needed
        raise NotImplementedError("Backward pass not implemented yet")


def fused_prepare_expert_route(
    hidden_states,
    gate_weight,
    gate_bias,
    num_experts,
    top_k,
    norm_topk_prob=False,
):
    """
    Fused implementation of prepare_expert_route using Triton.

    This version uses a hybrid approach:
    1. Fused kernel for gate computation and softmax
    2. PyTorch's efficient top-k operation
    3. Fused kernel for scatter operations
    """
    return FusedExpertRouting.apply(
        hidden_states,
        gate_weight,
        gate_bias,
        num_experts,
        top_k,
        norm_topk_prob,
    )


# Pure PyTorch implementation with only scatter fusion
def prepare_expert_route_scatter_only(
    gate_module, hidden_states, num_experts, top_k, norm_topk_prob=False
):
    """
    Minimal fusion - only fuse the scatter operations.
    This is often the most practical approach.
    """
    # Use PyTorch for everything except scatter
    router_logits = gate_module(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights_topk, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1
    )

    if norm_topk_prob:
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1, keepdim=True
        )

    routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

    # Only fuse the scatter operations
    B, E = router_logits.shape
    router_mask = torch.zeros(
        B, E, dtype=torch.bool, device=hidden_states.device
    )
    routing_weights_mask = torch.zeros(
        B, E, dtype=hidden_states.dtype, device=hidden_states.device
    )

    # Simple scatter without normalization (PyTorch handles it above)
    grid = (B,)
    scatter_topk_kernel[grid](
        routing_weights_topk,
        selected_experts,
        router_mask,
        routing_weights_mask,
        B,
        E,
        top_k,
        False,  # Normalization already done
    )

    return router_logits, router_mask, routing_weights_mask


# Example usage
if __name__ == "__main__":
    # Test configuration
    B, D, E, K = 1, 2048, 128, 8
    device = torch.cuda.current_device()

    # Create test inputs
    hidden_states = torch.randn(B, D, device=device, dtype=torch.float16)
    gate_weight = torch.randn(E, D, device=device, dtype=torch.float16)
    gate_bias = torch.randn(E, device=device, dtype=torch.float16)

    router_logits, router_mask, routing_weights_mask = (
        fused_prepare_expert_route(
            hidden_states, gate_weight, gate_bias, E, K, norm_topk_prob=True
        )
    )

    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    print("Testing hybrid fused implementation...")
    start_fused = time.perf_counter()
    router_logits, router_mask, routing_weights_mask = (
        fused_prepare_expert_route(
            hidden_states, gate_weight, gate_bias, E, K, norm_topk_prob=True
        )
    )
    stream.synchronize()
    end_fused = time.perf_counter()
    print(
        f"Fused implementation time: {(end_fused - start_fused) * 1000:.2f} ms"
    )

    print(f"Router logits shape: {router_logits.shape}")
    print(f"Router mask shape: {router_mask.shape}")
    print(f"Routing weights mask shape: {routing_weights_mask.shape}")

    # Reference implementation
    gate_module = torch.nn.Linear(
        D, E, device=device, dtype=hidden_states.dtype
    )
    gate_module.weight.data = gate_weight
    gate_module.bias.data = gate_bias

    start_ref = time.perf_counter()
    router_logits_ref = gate_module(hidden_states)
    routing_weights_ref = F.softmax(router_logits_ref, dim=1, dtype=torch.float)
    routing_weights_topk, selected_experts = torch.topk(
        routing_weights_ref, K, dim=-1
    )
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
        dim=-1, keepdim=True
    )
    routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

    router_mask_ref = torch.zeros(B, E, dtype=torch.bool, device=device)
    router_mask_ref.scatter_(1, selected_experts, True)

    routing_weights_mask_ref = torch.zeros(
        B, E, dtype=hidden_states.dtype, device=device
    )
    routing_weights_mask_ref.scatter_add_(
        1, selected_experts, routing_weights_topk
    )
    stream.synchronize()
    end_ref = time.perf_counter()
    print(
        f"Reference implementation time: {(end_ref - start_ref) * 1000:.2f} ms"
    )

    # Check correctness
    print(
        f"\nRouter logits match: {torch.allclose(router_logits, router_logits_ref, rtol=1e-2, atol=1e-3)}"
    )
    print(f"Router mask match: {torch.equal(router_mask, router_mask_ref)}")
    print(
        f"Routing weights mask match: {torch.allclose(routing_weights_mask, routing_weights_mask_ref, rtol=1e-2, atol=1e-3)}"
    )

    # Test scatter-only version
    print("\nTesting scatter-only implementation...")
    start_scatter = time.perf_counter()
    router_logits_v2, router_mask_v2, routing_weights_mask_v2 = (
        prepare_expert_route_scatter_only(
            gate_module, hidden_states, E, K, norm_topk_prob=True
        )
    )
    stream.synchronize()
    end_scatter = time.perf_counter()
    print(
        f"Scatter-only implementation time: {(end_scatter - start_scatter) * 1000:.2f} ms"
    )

    print(
        f"Scatter-only logits match: {torch.allclose(router_logits_v2, router_logits_ref, rtol=1e-2, atol=1e-3)}"
    )
    print(
        f"Scatter-only mask match: {torch.equal(router_mask_v2, router_mask_ref)}"
    )
    print(
        f"Scatter-only weights match: {torch.allclose(routing_weights_mask_v2, routing_weights_mask_ref, rtol=1e-2, atol=1e-3)}"
    )

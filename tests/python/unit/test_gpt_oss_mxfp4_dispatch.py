import pytest
import torch


@pytest.mark.gpu
def test_native_mxfp4_gate_up_dequant_is_exact():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity._v4_fp4 import mxfp4_dequant
    from moe_infinity.kernel.mxfp4_gemm import mxfp4_dequantize

    torch.manual_seed(137)
    blocks = torch.randint(
        0, 256, (5760, 1440), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        120, 135, (5760, 90), dtype=torch.uint8, device="cuda"
    )
    expected = mxfp4_dequantize(
        blocks, scales, dtype=torch.bfloat16, block_size=32
    )
    actual = mxfp4_dequant(blocks, scales)

    relative_error = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-12)
    ).max()
    assert actual.shape == (5760, 2880)
    assert actual.dtype == torch.bfloat16
    assert relative_error.item() == 0.0


@pytest.mark.gpu
def test_dequantized_option_a_matches_resident_expert_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity._v4_fp4 import mxfp4_dequant
    except Exception:
        pytest.skip("native MXFP4 dequant extension not built")

    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    torch.manual_seed(137)
    tokens, hidden, intermediate = 3, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate_blocks = torch.randint(
        0,
        256,
        (2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    gate_scales = torch.randint(
        120,
        135,
        (2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    down_blocks = torch.randint(
        0,
        256,
        (hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    down_scales = torch.randint(
        120,
        135,
        (hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    gate_bias = torch.randn(
        2 * intermediate, dtype=torch.bfloat16, device="cuda"
    )
    down_bias = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

    resident_gate_up = fused_mxfp4_gemm(
        x, gate_blocks, gate_scales, gate_bias
    )
    resident_gate, resident_up = (
        resident_gate_up[:, ::2],
        resident_gate_up[:, 1::2],
    )
    resident_activated = (resident_up.clamp(-7, 7) + 1) * (
        resident_gate.clamp(max=7)
        * torch.sigmoid(resident_gate.clamp(max=7) * 1.702)
    )
    resident = fused_mxfp4_gemm(
        resident_activated.to(torch.bfloat16),
        down_blocks,
        down_scales,
        down_bias,
    )

    gate_weight = mxfp4_dequant(gate_blocks, gate_scales)
    down_weight = mxfp4_dequant(down_blocks, down_scales)
    option_a_gate_up = x @ gate_weight.t() + gate_bias
    option_a_gate, option_a_up = (
        option_a_gate_up[:, ::2],
        option_a_gate_up[:, 1::2],
    )
    option_a_activated = (option_a_up.clamp(-7, 7) + 1) * (
        option_a_gate.clamp(max=7)
        * torch.sigmoid(option_a_gate.clamp(max=7) * 1.702)
    )
    option_a = option_a_activated @ down_weight.t() + down_bias

    gate_weight_f = gate_weight.float()
    down_weight_f = down_weight.float()
    golden_gate_up = x.float() @ gate_weight_f.t() + gate_bias.float()
    golden_gate, golden_up = golden_gate_up[:, ::2], golden_gate_up[:, 1::2]
    golden_activated = (golden_up.clamp(-7, 7) + 1) * (
        golden_gate.clamp(max=7)
        * torch.sigmoid(golden_gate.clamp(max=7) * 1.702)
    )
    golden = golden_activated @ down_weight_f.t() + down_bias.float()

    # Bound bf16 rounding by the down-GEMM magnitude instead of comparing two
    # cancellation-sensitive bf16 paths directly.
    envelope = 8 * (2**-8) * (
        golden_activated.abs() @ down_weight_f.abs().t()
        + down_bias.float().abs()
    ) + 1e-2
    assert ((option_a.float() - golden).abs() <= envelope).all()
    assert ((resident.float() - golden).abs() <= envelope).all()


def _native_dispatch(tmp_path, hidden_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity.runtime.model_offload import _load_prefetch_lib

        prefetch_lib = _load_prefetch_lib()
    except Exception as exc:
        pytest.skip(f"native Archer extension unavailable: {exc}")

    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    class Config:
        hidden_size = 64
        intermediate_size = 32
        num_local_experts = 2
        num_experts_per_tok = 1

    torch.manual_seed(137)
    hidden = Config.hidden_size
    intermediate = Config.intermediate_size
    mlp = SyncGptOssMLP(Config())
    tensors = [
        torch.randint(0, 256, (2 * intermediate, hidden // 2), dtype=torch.uint8),
        torch.randint(120, 135, (2 * intermediate, hidden // 32), dtype=torch.uint8),
        torch.randn(2 * intermediate, dtype=torch.bfloat16),
        torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8),
        torch.randint(120, 135, (hidden, intermediate // 32), dtype=torch.uint8),
        torch.randn(hidden, dtype=torch.bfloat16),
    ]
    mlp.experts.gate_up_proj.requires_grad_(False)
    mlp.experts.down_proj.requires_grad_(False)
    mlp.experts.gate_up_proj.data = tensors[0].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.gate_up_proj_scales.data = tensors[1].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.gate_up_proj_bias.data = tensors[2].unsqueeze(0).repeat(2, 1)
    mlp.experts.down_proj.data = tensors[3].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.down_proj_scales.data = tensors[4].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.down_proj_bias.data = tensors[5].unsqueeze(0).repeat(2, 1)

    hidden_states = torch.randn(
        3, hidden, dtype=hidden_dtype, device="cuda:0"
    )
    expected = mlp._expert_forward_mxfp4(hidden_states, 0)

    engine = prefetch_lib.prefetch_handle(f"{tmp_path}/", 0.5)
    expert_tensor_ids = [list(range(6)), list(range(6, 12))]
    for tensor_id, tensor in zip(expert_tensor_ids[0], tensors):
        engine.offload(tensor, tensor_id)
    for tensor_id, tensor in zip(expert_tensor_ids[1], tensors):
        engine.offload(tensor, tensor_id)
    dense_tensor_ids = list(
        range(12, 12 + 2 * torch.cuda.device_count())
    )
    for tensor_id in dense_tensor_ids:
        engine.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    split = len(dense_tensor_ids) // 2
    dense_before = [
        (f"model.dense.{index}", [[tensor_id]])
        for index, tensor_id in enumerate(dense_tensor_ids[:split])
    ]
    dense_after = [
        (f"model.dense.{index + split}", [[tensor_id]])
        for index, tensor_id in enumerate(dense_tensor_ids[split:])
    ]
    engine.set_topology(
        dense_before
        + [("model.layers.0.mlp.experts", expert_tensor_ids)]
        + dense_after
    )

    dispatcher = prefetch_lib.expert_dispatcher(2, 1, 0, 6, 1)
    dispatcher.register_expert(0, 0, expert_tensor_ids[0], "")
    dispatcher.register_expert(0, 1, expert_tensor_ids[1], "")
    torch.cuda.set_device(hidden_states.device)
    router_mask = torch.zeros((3, 2), dtype=torch.bool, device="cuda:0")
    router_mask[:, 0] = True
    router_weights = torch.zeros((3, 2), dtype=torch.bfloat16, device="cuda:0")
    router_weights[:, 0] = 1
    dispatcher.set_inputs(
        hidden_states,
        router_mask,
        router_weights,
    )
    dispatcher.set_expected_queue(1)
    dispatcher.enqueue_expert(0, 0, 0, False)
    dispatcher.notify_fetch_start()
    actual = dispatcher.wait_expert()

    return actual, expected, hidden_states, tensors


@pytest.mark.gpu
def test_native_dispatcher_matches_resident_expert_forward(tmp_path):
    actual, resident, hidden_states, tensors = _native_dispatch(
        tmp_path, torch.bfloat16
    )
    from moe_infinity._v4_fp4 import mxfp4_dequant

    gate_weight = mxfp4_dequant(tensors[0].cuda(), tensors[1].cuda()).float()
    down_weight = mxfp4_dequant(tensors[3].cuda(), tensors[4].cuda()).float()
    golden_gate_up = (
        hidden_states.float() @ gate_weight.t() + tensors[2].cuda().float()
    )
    golden_gate, golden_up = golden_gate_up[:, ::2], golden_gate_up[:, 1::2]
    golden_activated = (golden_up.clamp(-7, 7) + 1) * (
        golden_gate.clamp(max=7)
        * torch.sigmoid(golden_gate.clamp(max=7) * 1.702)
    )
    golden = golden_activated @ down_weight.t() + tensors[5].cuda().float()
    envelope = 8 * (2**-8) * (
        golden_activated.abs() @ down_weight.abs().t()
        + tensors[5].cuda().float().abs()
    ) + 1e-2

    assert ((actual.to(golden.device) - golden).abs() <= envelope).all()
    assert ((resident.float() - golden).abs() <= envelope).all()


@pytest.mark.gpu
def test_native_dispatcher_rejects_mismatched_hidden_dtype(tmp_path):
    actual, _, _, _ = _native_dispatch(tmp_path, torch.float32)

    assert torch.count_nonzero(actual).item() == 0

"""Task 2: rich on-device forward helper + 5-layer hidden-state capture.

Covers `_native_model_forward_rich` and `extract_context_feature` in
`moe_infinity/entrypoints/big_modeling.py`:

1. Capture returns the 5-layer concat with last dim == 5 * hidden, on the
   model device (NOT the CPU-detach path of `_native_model_forward`), with
   the model dtype.
2. Capture-on greedy decode is byte-identical to capture-off greedy decode
   (the capture must not perturb the forward).

The tiny gpt-oss-like target is built INLINE here (no T5-fixture imports —
those are owned by a parallel task) and runs on CPU or GPU, fp32, seeded.
"""

from types import SimpleNamespace

import torch
from transformers.models.gpt_oss import GptOssConfig, GptOssForCausalLM

from moe_infinity.entrypoints.big_modeling import (
    MoE,
    extract_context_feature,
)

TINY_HIDDEN = 64
TINY_LAYERS = 6
# Scaled stand-in for the real gpt-oss-120b DFlash ids (1, 9, 17, 25, 33).
TINY_LAYER_IDS = (0, 1, 2, 3, 4)
DEVICE = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]


def _tiny_config() -> GptOssConfig:
    return GptOssConfig(
        vocab_size=512,
        hidden_size=TINY_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=128,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 512,
            "rope_theta": 150000.0,
        },
        tie_word_embeddings=False,
    )


def _tiny_model(seed: int = 0) -> GptOssForCausalLM:
    torch.manual_seed(seed)
    return GptOssForCausalLM(_tiny_config()).to(DEVICE).eval()


def _moe_shell(model: GptOssForCausalLM) -> MoE:
    """Bare MoE instance (no checkpoint load) exposing the forward helpers."""
    shell = MoE.__new__(MoE)
    shell.model = model
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    return shell


def _greedy_ids(shell: MoE, rich: bool, n_tokens: int) -> list[int]:
    shell._cached_past_key_values = None
    generated: list[int] = []
    step_ids = list(PROMPT)
    for step in range(n_tokens):
        meta = None if step == 0 else SimpleNamespace(is_prefill=False)
        if rich:
            logits, _, _ = shell._native_model_forward_rich(
                step_ids, meta, logits_to_keep=1
            )
            next_id = int(logits[0, -1].argmax().item())
        else:
            logits = shell._native_model_forward(step_ids, meta)
            next_id = int(logits[-1].argmax().item())
        generated.append(next_id)
        step_ids = [next_id]
    return generated


def test_rich_forward_capture_shape_device_dtype():
    model = _tiny_model()
    shell = _moe_shell(model)
    param = next(model.parameters())

    logits, hidden_states, past_kv = shell._native_model_forward_rich(
        PROMPT, None, logits_to_keep=1
    )

    vocab = model.config.vocab_size
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 1, vocab)
    # On-device: same device/dtype as the model weights (the baseline helper
    # would have detached to CPU here).
    assert logits.device == param.device
    assert logits.dtype == param.dtype
    if param.device.type == "cuda":
        assert logits.device.type == "cuda"

    assert isinstance(hidden_states, tuple)
    assert len(hidden_states) == TINY_LAYERS + 1  # embeddings + layers
    assert hidden_states[0].shape == (1, len(PROMPT), TINY_HIDDEN)

    feat = extract_context_feature(hidden_states, layer_ids=TINY_LAYER_IDS)
    assert feat.shape == (1, len(PROMPT), 5 * TINY_HIDDEN)
    assert feat.device == param.device
    assert feat.dtype == param.dtype

    assert past_kv is not None
    assert past_kv.get_seq_length() == len(PROMPT)


def test_extract_context_feature_default_layer_ids_mapping():
    # 35 entries (embeddings + 34 layers) so the real default ids index in.
    width = 8
    hidden_states = tuple(
        torch.full((1, 3, width), float(i)) for i in range(35)
    )
    feat = extract_context_feature(hidden_states)
    assert feat.shape == (1, 3, 5 * width)
    # Default ids (1, 9, 17, 25, 33) map to tuple indices id + 1.
    for slot, src_idx in enumerate((2, 10, 18, 26, 34)):
        block = feat[0, :, slot * width : (slot + 1) * width]
        assert torch.all(block == float(src_idx))


def test_extract_context_feature_out_of_range_raises():
    hidden_states = tuple(torch.zeros(1, 2, 4) for _ in range(3))
    try:
        extract_context_feature(hidden_states, layer_ids=(5,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range layer id")


def test_logits_to_keep_passthrough():
    model = _tiny_model()
    shell = _moe_shell(model)

    full_logits, hidden_states, _ = shell._native_model_forward_rich(
        PROMPT, None, logits_to_keep=0
    )
    assert full_logits.shape == (1, len(PROMPT), model.config.vocab_size)
    # Hidden-state capture is full-length even when logits are kept in full.
    assert hidden_states[-1].shape[1] == len(PROMPT)

    shell._cached_past_key_values = None
    kept_logits, _, _ = shell._native_model_forward_rich(
        PROMPT, None, logits_to_keep=1
    )
    assert kept_logits.shape == (1, 1, model.config.vocab_size)
    # The kept row is the last row of the full-logits forward. GEMM-shape
    # differences allow low-bit FP wobble on GPU, so the contract is
    # argmax-identical (the DFlash losslessness rule) + numerically tight.
    assert torch.allclose(
        full_logits[:, -1:, :], kept_logits, rtol=1e-4, atol=1e-5
    )
    assert int(full_logits[0, -1].argmax().item()) == int(
        kept_logits[0, -1].argmax().item()
    )


def test_capture_on_off_greedy_byte_identical():
    model = _tiny_model()
    shell = _moe_shell(model)

    baseline_ids = _greedy_ids(shell, rich=False, n_tokens=16)
    capture_ids = _greedy_ids(shell, rich=True, n_tokens=16)

    assert len(baseline_ids) == 16
    assert capture_ids == baseline_ids

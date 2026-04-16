import importlib.machinery
import sys
import types

import torch  # pyright: ignore[reportMissingImports]


def _ensure_flash_attn_stub_has_spec() -> None:
    flash_attn_module = sys.modules.get("flash_attn")
    if flash_attn_module is None:
        flash_attn_module = types.ModuleType("flash_attn")
        flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None
        )
        sys.modules["flash_attn"] = flash_attn_module
    elif getattr(flash_attn_module, "__spec__", None) is None:
        flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None
        )


_ensure_flash_attn_stub_has_spec()

from transformers import DeepseekV2Config, DeepseekV3Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2RotaryEmbedding,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
)

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)


def _zero_position_ids(batch_size: int, seq_len: int, device: torch.device):
    return torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)


def _zero_attention_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
):
    return torch.zeros(
        (batch_size, 1, seq_len, seq_len), device=device, dtype=dtype
    )


def _build_v2_position_embeddings(config, hidden_states, position_ids):
    rotary_emb = DeepseekV2RotaryEmbedding(config)
    return rotary_emb(hidden_states, position_ids)


def _build_v3_position_embeddings(config, hidden_states, position_ids):
    rotary_emb = DeepseekV3RotaryEmbedding(config)
    return rotary_emb(hidden_states, position_ids)


@requires_cuda
def test_deepseek_v2_mla_forward_is_finite_and_deterministic(seed_everything):
    config = DeepseekV2Config(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=64,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    attn = DeepseekV2Attention(config, layer_idx=0).cuda().bfloat16().eval()

    hidden_states = torch.randn(
        2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    attention_mask = _zero_attention_mask(
        hidden_states.size(0),
        hidden_states.size(1),
        hidden_states.device,
        hidden_states.dtype,
    )
    position_ids = _zero_position_ids(
        hidden_states.size(0), hidden_states.size(1), hidden_states.device
    )
    position_embeddings = _build_v2_position_embeddings(
        config, hidden_states, position_ids
    )

    with torch.no_grad():
        out_1, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        out_2, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

    assert out_1.shape == hidden_states.shape
    assert out_1.dtype == torch.bfloat16
    assert torch.isfinite(out_1).all()
    torch.testing.assert_close(out_1, out_2, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
def test_deepseek_v3_attention_forward_is_finite_and_deterministic(
    seed_everything,
):
    config = DeepseekV3Config(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=64,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    attn = DeepseekV3Attention(config, layer_idx=0).cuda().bfloat16().eval()

    hidden_states = torch.randn(
        2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    attention_mask = _zero_attention_mask(
        hidden_states.size(0),
        hidden_states.size(1),
        hidden_states.device,
        hidden_states.dtype,
    )
    position_ids = _zero_position_ids(
        hidden_states.size(0), hidden_states.size(1), hidden_states.device
    )
    position_embeddings = _build_v3_position_embeddings(
        config, hidden_states, position_ids
    )

    with torch.no_grad():
        out_1, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        out_2, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

    assert out_1.shape == hidden_states.shape
    assert out_1.dtype == torch.bfloat16
    assert torch.isfinite(out_1).all()
    torch.testing.assert_close(out_1, out_2, rtol=BF16_RTOL, atol=BF16_ATOL)

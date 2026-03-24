import math

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

from moe_infinity.models.modeling_arctic.configuration_arctic import (
    ArcticConfig,
)
from moe_infinity.models.modeling_arctic.modeling_arctic import (
    ArcticAttention,
    ArcticSdpaAttention,
)
from moe_infinity.models.modeling_deepseek_v2.configuration_deepseek import (
    DeepseekV2Config,
)
from moe_infinity.models.modeling_deepseek_v2.modeling_deepseek import (
    DeepseekV2Attention,
)
from moe_infinity.models.modeling_deepseek_v3.configuration_deepseek import (
    DeepseekV3Config,
)
from moe_infinity.models.modeling_deepseek_v3.modeling_deepseek import (
    DeepseekV3Attention,
)
from moe_infinity.models.modeling_grok.modeling_grok1 import (
    MultiHeadAttention,
)
from moe_infinity.models.modeling_grok.modeling_grok1 import (
    apply_rotary_pos_emb as grok_apply_rotary_pos_emb,
)
from moe_infinity.models.modeling_grok.modeling_grok1 import (
    repeat_kv as grok_repeat_kv,
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

    with torch.no_grad():
        out_1, _, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        out_2, _, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
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

    with torch.no_grad():
        out_1, _, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        out_2, _, _ = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    assert out_1.shape == hidden_states.shape
    assert out_1.dtype == torch.bfloat16
    assert torch.isfinite(out_1).all()
    torch.testing.assert_close(out_1, out_2, rtol=BF16_RTOL, atol=BF16_ATOL)


@requires_cuda
def test_arctic_attention_eager_matches_sdpa(seed_everything):
    config = ArcticConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )

    eager = ArcticAttention(config, layer_idx=0).cuda().bfloat16().eval()
    sdpa = ArcticSdpaAttention(config, layer_idx=0).cuda().bfloat16().eval()
    sdpa.load_state_dict(eager.state_dict())

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

    with torch.no_grad():
        eager_out, _, _ = eager(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        sdpa_out, _, _ = sdpa(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    assert torch.isfinite(eager_out).all()
    assert torch.isfinite(sdpa_out).all()
    torch.testing.assert_close(
        eager_out, sdpa_out, rtol=BF16_RTOL, atol=BF16_ATOL
    )


@requires_cuda
def test_grok_attention_matches_sdpa_reference(seed_everything):
    hidden_size = 128
    num_heads = 8
    seq_len = 8
    batch_size = 2

    grok_attn = (
        MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=64,
            attn_output_multiplier=1.0 / math.sqrt(hidden_size // num_heads),
            max_attn_val=1e4,
        )
        .cuda()
        .bfloat16()
        .eval()
    )

    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    attention_mask = _zero_attention_mask(
        batch_size, seq_len, hidden_states.device, hidden_states.dtype
    )
    position_ids = _zero_position_ids(batch_size, seq_len, hidden_states.device)

    with torch.no_grad():
        custom_out, _, _ = grok_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        query_states = (
            grok_attn.q_proj(hidden_states)
            .view(batch_size, seq_len, grok_attn.num_heads, grok_attn.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            grok_attn.k_proj(hidden_states)
            .view(
                batch_size,
                seq_len,
                grok_attn.num_key_value_heads,
                grok_attn.head_dim,
            )
            .transpose(1, 2)
        )
        value_states = (
            grok_attn.v_proj(hidden_states)
            .view(
                batch_size,
                seq_len,
                grok_attn.num_key_value_heads,
                grok_attn.head_dim,
            )
            .transpose(1, 2)
        )

        cos, sin = grok_attn.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = grok_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        key_states = grok_repeat_kv(key_states, grok_attn.num_key_value_groups)
        value_states = grok_repeat_kv(
            value_states, grok_attn.num_key_value_groups
        )

        ref_out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        ref_out = (
            ref_out.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, hidden_size)
        )
        ref_out = grok_attn.o_proj(ref_out)

    assert custom_out.shape == hidden_states.shape
    assert torch.isfinite(custom_out).all()
    assert torch.isfinite(ref_out).all()
    torch.testing.assert_close(
        custom_out, ref_out, rtol=BF16_RTOL, atol=BF16_ATOL
    )

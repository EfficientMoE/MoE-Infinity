import inspect

import pytest
import torch

pytest.importorskip(
    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    reason="transformers >= 5.12 required",
)

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaMoE as _GlmMoeDsaMoE,
)

_HAS_HF_ROUTER = hasattr(_GlmMoeDsaMoE, "route_tokens_to_experts")


def _tiny_config():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )

    return GlmMoeDsaConfig(
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        hidden_size=64,
        moe_intermediate_size=32,
        n_shared_experts=1,
        num_hidden_layers=2,
        vocab_size=256,
    )


def test_no_handrolled_softmax():
    import moe_infinity.models.glm_moe_dsa as mod

    src = inspect.getsource(mod)
    assert "torch.softmax" not in src
    assert "F.softmax" not in src


def test_exported():
    import moe_infinity.models as m
    from moe_infinity.models import SyncGlmMoeDsaMoEBlock

    assert SyncGlmMoeDsaMoEBlock is not None
    assert "SyncGlmMoeDsaMoEBlock" in m.__all__


@pytest.mark.skipif(
    not _HAS_HF_ROUTER,
    reason="transformers build lacks GlmMoeDsaMoE.route_tokens_to_experts to compare against",
)
def test_routing_parity():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMoE,
    )

    from moe_infinity.models.glm_moe_dsa import SyncGlmMoeDsaMoEBlock

    cfg = _tiny_config()
    torch.manual_seed(0)
    ref = GlmMoeDsaMoE(cfg)
    block = SyncGlmMoeDsaMoEBlock(cfg)
    block.gate.load_state_dict(ref.gate.state_dict())

    x = torch.randn(4, cfg.hidden_size)
    with torch.no_grad():
        r_idx, r_w = ref.route_tokens_to_experts(ref.gate(x))
        b_idx, b_w = block._route(x)

    assert torch.equal(r_idx.sort(-1).values, b_idx.sort(-1).values)
    assert torch.allclose(r_w.sort(-1).values, b_w.sort(-1).values, atol=1e-5)


def test_executor_stays_unset():
    from moe_infinity.models.glm_moe_dsa import SyncGlmMoeDsaMoEBlock

    cfg = _tiny_config()
    block = SyncGlmMoeDsaMoEBlock(cfg)
    assert block.expert_executor is None


def test_forward_shape():
    from moe_infinity.models.glm_moe_dsa import SyncGlmMoeDsaMoEBlock

    cfg = _tiny_config()
    block = SyncGlmMoeDsaMoEBlock(cfg).eval()
    x = torch.randn(2, 5, cfg.hidden_size)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

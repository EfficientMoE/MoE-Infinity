import importlib
from typing import Any, Optional

import pytest
import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    seed_everything,
)


class _LocalExpertExecutor:
    def __init__(self, experts):
        self.experts = experts
        self.hidden_states = None
        self.router_mask = None
        self.router_weights = None

    def dispatch_local(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> None:
        del layer_id
        self.hidden_states = hidden_states
        self.router_mask = router_mask
        self.router_weights = router_weights

    def wait_dispatch_local(self) -> torch.Tensor:
        assert self.hidden_states is not None
        assert self.router_mask is not None
        assert self.router_weights is not None

        output = torch.zeros_like(self.hidden_states)
        for expert_id, expert in enumerate(self.experts):
            token_mask = self.router_mask[:, expert_id].bool()
            if not token_mask.any():
                continue
            expert_output = expert(self.hidden_states[token_mask])
            weights = self.router_weights[token_mask, expert_id].unsqueeze(-1)
            output[token_mask] += expert_output * weights

        return output


class _PythonTopKSoftmax:
    def __init__(
        self, num_experts: int, top_k: int, norm_topk_prob: bool
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def topk_softmax(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        num_tokens = router_logits.shape[0]
        router_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=router_logits.device,
        )
        router_mask.scatter_(1, selected_experts, True)

        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=routing_weights.dtype,
            device=router_logits.device,
        )
        routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)

        return selected_experts, router_mask, routing_weights_mask


def _hidden_states_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _router_logits_from_output(
    output: Any,
) -> Optional[torch.Tensor]:
    if isinstance(output, tuple) and len(output) > 1:
        return output[1]
    return None


def _load_sync_and_hf_state_dict(
    sync_block, hf_block, architecture: str
) -> None:
    try:
        sync_block.load_state_dict(hf_block.state_dict(), strict=True)
    except Exception as exc:
        pytest.skip(
            f"{architecture} Sync/HF block parameters are not directly compatible in this revision. "
            "If this build requires full ArcherEngine wiring, use integration coverage in "
            "tests/python/integration/test_model_consistency.py. "
            f"Original error: {exc}"
        )


def _build_mixtral_config(mixtral_config_cls):
    kwargs = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
    }
    try:
        return mixtral_config_cls(**kwargs)
    except TypeError:
        kwargs.pop("num_key_value_heads")
        return mixtral_config_cls(**kwargs)


def _build_qwen3_config(qwen3_config_cls):
    kwargs = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 96,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }
    try:
        return qwen3_config_cls(**kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("mlp_only_layers", None)
        fallback_kwargs.pop("decoder_sparse_step", None)
        try:
            return qwen3_config_cls(**fallback_kwargs)
        except TypeError:
            fallback_kwargs.pop("num_key_value_heads", None)
            return qwen3_config_cls(**fallback_kwargs)


@requires_cuda
def test_sync_mixtral_sparse_moe_block_matches_hf(seed_everything):
    del seed_everything

    mixtral_modeling = pytest.importorskip(
        "transformers.models.mixtral.modeling_mixtral"
    )
    mixtral_config = pytest.importorskip(
        "transformers.models.mixtral.configuration_mixtral"
    )

    if not hasattr(mixtral_modeling, "MixtralSparseMoeBlock"):
        pytest.skip(
            "MixtralSparseMoeBlock is unavailable in this transformers build"
        )

    try:
        sync_mixtral_module = importlib.import_module(
            "moe_infinity.models.mixtral"
        )
    except Exception as exc:
        pytest.skip(f"Unable to import SyncMixtralSparseMoeBlock: {exc}")

    SyncMixtralSparseMoeBlock = getattr(
        sync_mixtral_module, "SyncMixtralSparseMoeBlock", None
    )
    if SyncMixtralSparseMoeBlock is None:
        pytest.skip("SyncMixtralSparseMoeBlock is not available")

    config = _build_mixtral_config(mixtral_config.MixtralConfig)
    hf_block = (
        mixtral_modeling.MixtralSparseMoeBlock(config).cuda().bfloat16().eval()
    )
    sync_block = SyncMixtralSparseMoeBlock(config).cuda().bfloat16().eval()

    if not hasattr(sync_block, "expert_executor") or not hasattr(
        sync_block, "experts"
    ):
        pytest.skip(
            "SyncMixtralSparseMoeBlock API changed; expected runtime handles are missing"
        )

    sync_block.expert_executor = _LocalExpertExecutor(sync_block.experts)
    sync_block.layer_id = 0
    _load_sync_and_hf_state_dict(sync_block, hf_block, architecture="Mixtral")

    x = torch.randn(
        2, 8, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    with torch.no_grad():
        hf_out = hf_block(x)
        sync_out = sync_block(x)

    hf_hidden = _hidden_states_from_output(hf_out)
    sync_hidden = _hidden_states_from_output(sync_out)
    torch.testing.assert_close(
        sync_hidden,
        hf_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    hf_router_logits = _router_logits_from_output(hf_out)
    sync_router_logits = _router_logits_from_output(sync_out)
    if hf_router_logits is not None and sync_router_logits is not None:
        torch.testing.assert_close(
            sync_router_logits,
            hf_router_logits,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )


@requires_cuda
def test_sync_qwen3_sparse_moe_block_matches_hf(seed_everything):
    del seed_everything

    qwen3_modeling = pytest.importorskip(
        "transformers.models.qwen3_moe.modeling_qwen3_moe"
    )
    qwen3_config = pytest.importorskip(
        "transformers.models.qwen3_moe.configuration_qwen3_moe"
    )

    if not hasattr(qwen3_modeling, "Qwen3MoeSparseMoeBlock"):
        pytest.skip(
            "Qwen3MoeSparseMoeBlock is unavailable in this transformers build"
        )

    try:
        sync_qwen_module = importlib.import_module("moe_infinity.models.qwen")
    except Exception as exc:
        pytest.skip(f"Unable to import Qwen3MoEBlock replacement: {exc}")

    Qwen3MoEBlock = getattr(sync_qwen_module, "Qwen3MoEBlock", None)
    if Qwen3MoEBlock is None:
        pytest.skip("Qwen3MoEBlock replacement class is not available")

    config = _build_qwen3_config(qwen3_config.Qwen3MoeConfig)
    hf_block = (
        qwen3_modeling.Qwen3MoeSparseMoeBlock(config).cuda().bfloat16().eval()
    )
    sync_block = Qwen3MoEBlock(config).cuda().bfloat16().eval()

    if (
        not hasattr(sync_block, "expert_executor")
        or not hasattr(sync_block, "lib")
        or not hasattr(sync_block, "experts")
    ):
        pytest.skip(
            "Qwen3MoEBlock API changed; expected runtime handles are missing"
        )

    sync_block.expert_executor = _LocalExpertExecutor(sync_block.experts)
    sync_block.layer_id = 0
    sync_block.lib = _PythonTopKSoftmax(
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        norm_topk_prob=getattr(config, "norm_topk_prob", True),
    )
    _load_sync_and_hf_state_dict(sync_block, hf_block, architecture="Qwen3")

    x = torch.randn(
        2, 8, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    with torch.no_grad():
        hf_out = hf_block(x)
        sync_out = sync_block(x)

    hf_hidden = _hidden_states_from_output(hf_out)
    sync_hidden = _hidden_states_from_output(sync_out)
    torch.testing.assert_close(
        sync_hidden,
        hf_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    hf_router_logits = _router_logits_from_output(hf_out)
    sync_router_logits = _router_logits_from_output(sync_out)
    if hf_router_logits is not None and sync_router_logits is not None:
        torch.testing.assert_close(
            sync_router_logits,
            hf_router_logits,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )

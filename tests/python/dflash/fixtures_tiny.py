"""Tiny, CPU-deterministic gpt-oss target + DFlash drafter doubles.

Public fixture module for the DFlash autonomous test suite. It provides:

* ``build_tiny_target`` — a real (but tiny) ``GptOssForCausalLM`` in fp32 on CPU,
  faithful to the target contract downstream tasks rely on: ``output_hidden_states``
  (``num_hidden_layers + 1`` states), ``logits_to_keep``, ``embed_tokens`` /
  ``lm_head``, and a ``DynamicCache`` past that supports ``.crop()``.
* ``TinyDFlashDrafter`` / ``build_tiny_drafter`` — a matching drafter stand-in that
  mirrors the RFC §1.2 forward contract: reuses the target ``embed_tokens`` +
  ``lm_head``, projects the concatenated 5-layer context feature (``fc``: 5H→H,
  then RMSNorm) and KV-injects it into every layer with NON-causal attention.
* ``plain_greedy_decode`` — the ground-truth greedy generator.
* ``set_determinism`` / ``context_feature_from_hidden_states`` helpers.

By design this module contains NO DFlash acceptance / verify / rollback logic;
that state machine is the code under test in Task 6.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.spec_decode import read_dflash_config

TINY_HIDDEN = 32
TINY_VOCAB = 64
TINY_NUM_LAYERS = 6
TINY_TARGET_LAYER_IDS = (1, 2, 3, 4, 5)
TINY_BLOCK_SIZE = 10
TINY_MASK_TOKEN_ID = TINY_VOCAB - 1

_TARGET_HEADS = 4
_TARGET_KV_HEADS = 2
_TARGET_HEAD_DIM = 8
_DRAFTER_LAYERS = 2
_DRAFTER_HEADS = 4


def set_determinism(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def make_tiny_target_config(
    *,
    num_hidden_layers: int = TINY_NUM_LAYERS,
    hidden_size: int = TINY_HIDDEN,
    vocab_size: int = TINY_VOCAB,
) -> Any:
    from transformers import GptOssConfig

    return GptOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=_TARGET_HEADS,
        num_key_value_heads=_TARGET_KV_HEADS,
        head_dim=_TARGET_HEAD_DIM,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sliding_window=128,
        attn_implementation="eager",
        tie_word_embeddings=False,
        rope_scaling={"rope_type": "default"},
    )


def build_tiny_target(seed: int = 0, **config_kwargs: Any) -> Any:
    from transformers import GptOssForCausalLM

    set_determinism(seed)
    config = make_tiny_target_config(**config_kwargs)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GptOssForCausalLM(config)
    return model.to(torch.float32).eval()


def make_tiny_drafter_config(
    target_config: Any = None,
    *,
    block_size: int = TINY_BLOCK_SIZE,
    target_layer_ids: Sequence[int] = TINY_TARGET_LAYER_IDS,
    mask_token_id: int = TINY_MASK_TOKEN_ID,
    hidden_size: Optional[int] = None,
    vocab_size: Optional[int] = None,
    num_target_layers: Optional[int] = None,
) -> SimpleNamespace:
    if target_config is not None:
        if hidden_size is None:
            hidden_size = int(target_config.hidden_size)
        if vocab_size is None:
            vocab_size = int(target_config.vocab_size)
        if num_target_layers is None:
            num_target_layers = int(target_config.num_hidden_layers)
    return SimpleNamespace(
        block_size=block_size,
        hidden_size=hidden_size if hidden_size is not None else TINY_HIDDEN,
        vocab_size=vocab_size if vocab_size is not None else TINY_VOCAB,
        num_target_layers=(
            num_target_layers
            if num_target_layers is not None
            else TINY_NUM_LAYERS
        ),
        dflash_config={
            "mask_token_id": mask_token_id,
            "target_layer_ids": list(target_layer_ids),
        },
    )


def context_feature_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    target_layer_ids: Sequence[int] = TINY_TARGET_LAYER_IDS,
) -> torch.Tensor:
    return torch.cat([hidden_states[i + 1] for i in target_layer_ids], dim=-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


class _NonCausalBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_norm = _RMSNorm(hidden_size)
        self.mlp_norm = _RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _heads(self, x: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, noise: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        batch, block, hidden = noise.shape
        ctx_len = ctx.shape[1]
        normed = self.attn_norm(noise)
        # KV spans the injected context followed by the block; every query
        # attends over all of it (non-causal) — the core DFlash injection.
        kv_source = torch.cat([ctx, normed], dim=1)
        q = self._heads(self.q_proj(normed), batch, block)
        k = self._heads(self.k_proj(kv_source), batch, ctx_len + block)
        v = self._heads(self.v_proj(kv_source), batch, ctx_len + block)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch, block, hidden)
        )
        noise = noise + self.o_proj(attended)
        gated = self.down_proj(
            F.silu(self.gate_proj(self.mlp_norm(noise)))
            * self.up_proj(self.mlp_norm(noise))
        )
        return noise + gated


class TinyDFlashDrafter(nn.Module):
    is_causal = False

    def __init__(
        self,
        config: Any,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        *,
        num_layers: int = _DRAFTER_LAYERS,
        num_heads: int = _DRAFTER_HEADS,
    ) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.target_layer_ids = list(config.target_layer_ids)
        # Reuse the target modules by reference without registering them as
        # drafter submodules (they belong to, and are placed by, the target).
        object.__setattr__(self, "embed_tokens", embed_tokens)
        object.__setattr__(self, "lm_head", lm_head)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * hidden, hidden, bias=False
        )
        self.hidden_norm = _RMSNorm(hidden)
        self.layers = nn.ModuleList(
            [_NonCausalBlock(hidden, num_heads) for _ in range(num_layers)]
        )
        self.norm = _RMSNorm(hidden)

    def forward(
        self, block_ids: torch.Tensor, context_feature: torch.Tensor
    ) -> torch.Tensor:
        noise = self.embed_tokens(block_ids)
        ctx = self.hidden_norm(self.fc(context_feature))
        hidden = noise
        for layer in self.layers:
            hidden = layer(hidden, ctx)
        return self.norm(hidden)


def build_tiny_drafter(
    target: Any,
    seed: int = 1,
    *,
    block_size: int = TINY_BLOCK_SIZE,
    target_layer_ids: Sequence[int] = TINY_TARGET_LAYER_IDS,
    num_layers: int = _DRAFTER_LAYERS,
    num_heads: int = _DRAFTER_HEADS,
) -> TinyDFlashDrafter:
    draft_config_ns = make_tiny_drafter_config(
        target.config, block_size=block_size, target_layer_ids=target_layer_ids
    )
    config = read_dflash_config(draft_config_ns)
    if max(config.target_layer_ids) + 1 > int(target.config.num_hidden_layers):
        raise ValueError(
            f"tiny drafter target_layer_ids {config.target_layer_ids} exceed target "
            f"depth {target.config.num_hidden_layers}"
        )

    embed_tokens = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if lm_head is None:
        lm_head = target.lm_head

    set_determinism(seed)
    drafter = TinyDFlashDrafter(
        config,
        embed_tokens,
        lm_head,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    return drafter.to(torch.float32).eval()


@torch.no_grad()
def plain_greedy_decode(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    generated = input_ids.clone()

    output = model(generated, use_cache=True)
    past = output.past_key_values
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and int(next_token.item()) == int(
            eos_token_id
        ):
            break
        output = model(next_token, past_key_values=past, use_cache=True)
        past = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


__all__ = [
    "TINY_BLOCK_SIZE",
    "TINY_HIDDEN",
    "TINY_MASK_TOKEN_ID",
    "TINY_NUM_LAYERS",
    "TINY_TARGET_LAYER_IDS",
    "TINY_VOCAB",
    "TinyDFlashDrafter",
    "build_tiny_drafter",
    "build_tiny_target",
    "context_feature_from_hidden_states",
    "make_tiny_drafter_config",
    "make_tiny_target_config",
    "plain_greedy_decode",
    "set_determinism",
]

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Optional

import torch

from moe_infinity.entrypoints.big_modeling import extract_context_feature
from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    build_block,
    committed_tokens,
)


@dataclass
class DFlashConfig:
    block_size: int
    mask_token_id: int
    target_layer_ids: List[int]
    num_target_layers: int
    hidden_size: int
    vocab_size: int


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def read_dflash_config(draft_hf_config: Any) -> DFlashConfig:
    dflash = _get(draft_hf_config, "dflash_config", {}) or {}
    text_config = _get(draft_hf_config, "text_config", draft_hf_config)

    block_size = _get(draft_hf_config, "block_size", _get(dflash, "block_size"))
    mask_token_id = _get(dflash, "mask_token_id", _get(draft_hf_config, "mask_token_id"))
    target_layer_ids = _get(dflash, "target_layer_ids", _get(draft_hf_config, "target_layer_ids"))
    num_target_layers = _get(draft_hf_config, "num_target_layers", _get(dflash, "num_target_layers"))

    if block_size is None or mask_token_id is None or target_layer_ids is None:
        raise ValueError(
            "draft config is missing required DFlash fields "
            "(block_size, dflash_config.mask_token_id, dflash_config.target_layer_ids)"
        )

    return DFlashConfig(
        block_size=int(block_size),
        mask_token_id=int(mask_token_id),
        target_layer_ids=[int(x) for x in target_layer_ids],
        num_target_layers=int(num_target_layers) if num_target_layers is not None else -1,
        hidden_size=int(_get(text_config, "hidden_size")),
        vocab_size=int(_get(text_config, "vocab_size")),
    )


DFLASH_BLOCK_SIZE = 10
DFLASH_TARGET_LAYER_IDS = [1, 9, 17, 25, 33]


def validate_pairing(draft_cfg: DFlashConfig, target_hf_config: Any) -> None:
    target_text = _get(target_hf_config, "text_config", target_hf_config)
    t_hidden = int(_get(target_text, "hidden_size"))
    t_vocab = int(_get(target_text, "vocab_size"))
    t_layers = int(_get(target_text, "num_hidden_layers"))

    if draft_cfg.hidden_size != t_hidden:
        raise ValueError(
            f"DFlash drafter hidden_size {draft_cfg.hidden_size} != target hidden_size {t_hidden}"
        )
    if draft_cfg.vocab_size != t_vocab:
        raise ValueError(
            f"DFlash drafter vocab_size {draft_cfg.vocab_size} != target vocab_size {t_vocab}"
        )
    if draft_cfg.mask_token_id >= t_vocab:
        raise ValueError(
            f"DFlash mask_token_id {draft_cfg.mask_token_id} is outside target vocab_size {t_vocab}"
        )
    if draft_cfg.block_size != DFLASH_BLOCK_SIZE:
        raise ValueError(
            f"DFlash block_size {draft_cfg.block_size} != expected {DFLASH_BLOCK_SIZE}"
        )
    if draft_cfg.target_layer_ids != DFLASH_TARGET_LAYER_IDS:
        raise ValueError(
            f"DFlash target_layer_ids {draft_cfg.target_layer_ids} != expected {DFLASH_TARGET_LAYER_IDS}"
        )
    highest_capture = max(draft_cfg.target_layer_ids) + 1
    if highest_capture > t_layers:
        raise ValueError(
            f"DFlash target_layer_ids reference target layer {max(draft_cfg.target_layer_ids)} "
            f"(capture index {highest_capture}) but target has only {t_layers} layers"
        )


def validate_drafter_module(draft_model: Any, draft_cfg: DFlashConfig) -> None:
    fc = _get(draft_model, "fc")
    if fc is None:
        raise ValueError(
            "DFlash drafter is missing the required `fc` projection layer "
            "(expected a Linear consuming the concatenated 5-layer hidden feature)"
        )
    in_features = _get(fc, "in_features")
    expected = len(draft_cfg.target_layer_ids) * draft_cfg.hidden_size
    if in_features != expected:
        raise ValueError(
            f"DFlash drafter fc.in_features {in_features} != expected {expected} "
            f"({len(draft_cfg.target_layer_ids)} * hidden_size {draft_cfg.hidden_size})"
        )


def validate_drafter(
    draft_model: Any,
    target_hf_config: Any,
    draft_cfg: Optional[DFlashConfig] = None,
) -> DFlashConfig:
    if draft_cfg is None:
        draft_cfg = read_dflash_config(_get(draft_model, "config"))
    validate_pairing(draft_cfg, target_hf_config)
    validate_drafter_module(draft_model, draft_cfg)
    return draft_cfg


def _resolve_input_embeddings(target: Any) -> Any:
    getter = getattr(target, "get_input_embeddings", None)
    if callable(getter):
        emb = getter()
        if emb is not None:
            return emb
    inner = getattr(target, "model", target)
    emb = getattr(inner, "embed_tokens", None)
    if emb is not None:
        return emb
    raise ValueError(
        "could not resolve target embed_tokens for DFlash drafter weight sharing"
    )


def _resolve_output_embeddings(target: Any) -> Any:
    getter = getattr(target, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    head = getattr(target, "lm_head", None)
    if head is not None:
        return head
    raise ValueError(
        "could not resolve target lm_head for DFlash drafter weight sharing"
    )


def bind_shared_weights(draft_model: Any, target: Any) -> tuple[Any, Any]:
    embed_tokens = _resolve_input_embeddings(target)
    lm_head = _resolve_output_embeddings(target)
    draft_model.embed_tokens = embed_tokens
    draft_model.lm_head = lm_head
    return embed_tokens, lm_head


def _infer_cuda_device(model: Any) -> str:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return str(dev)
    try:
        for param in model.parameters():
            if param.device.type == "cuda":
                return str(param.device)
    except Exception:
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_stop_ids(target: Any, stop_token_ids: Optional[List[int]]) -> List[int]:
    if stop_token_ids is not None:
        return list(stop_token_ids)
    eos = _get(_get(target, "config", target), "eos_token_id")
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(x) for x in eos]


def rollback_target_cache(
    target_kv: Any,
    sliding_snaps: dict[int, tuple[torch.Tensor, torch.Tensor, int]],
    prev_start: int,
    committed: int,
    block_size: int,
) -> None:
    """Roll the target KV cache back to ``prev_start + committed`` in place.

    Slice-only rollback (``DynamicCache.crop``) is WRONG for sliding-window
    layers: ``DynamicSlidingWindowLayer.update`` retains only the last
    ``sliding_window - 1`` tokens, so the verify forward evicts prefix tokens
    that a partial accept must restore (and ``crop`` itself raises once the
    cumulative length reaches the window). Instead the sliding layers are
    snapshotted before the verify forward (see ``generate``) and rebuilt here
    as ``snapshot prefix ++ kept block tokens``, re-truncated to the window;
    full-attention layers retain every token and crop cleanly. ``kv_offset``
    is recomputed from ``cumulative_length``, so only ``keys`` / ``values`` /
    ``cumulative_length`` are touched.
    """
    from transformers.cache_utils import DynamicSlidingWindowLayer

    new_len = prev_start + committed
    for i, layer in enumerate(target_kv.layers):
        if isinstance(layer, DynamicSlidingWindowLayer):
            old_k, old_v, old_len = sliding_snaps[i]
            assert old_len == prev_start
            block_k = layer.keys[:, :, -block_size:, :]
            block_v = layer.values[:, :, -block_size:, :]
            keep_k = block_k[:, :, :committed, :]
            keep_v = block_v[:, :, :committed, :]
            full_k = torch.cat([old_k, keep_k], dim=-2)
            full_v = torch.cat([old_v, keep_v], dim=-2)
            layer.keys = full_k[:, :, -layer.sliding_window + 1 :, :]
            layer.values = full_v[:, :, -layer.sliding_window + 1 :, :]
            layer.cumulative_length = new_len
        else:
            layer.crop(new_len)


class NativeStepTrace(NamedTuple):
    """Per-step state accounting for the native DFlash loop (diagnostics).

    The bonus token is emitted but NOT cached, so after every step the target
    cache length equals ``start`` while the absolute emitted length is exactly
    one ahead -- conflating the two is the bonus-token trap (oracle ruling #3).
    """

    prev_start: int  # cached_len at step entry
    # accepted drafts actually committed this step, in [0, block_size - 1];
    # smaller than the accept-rule result when the step is truncated by a
    # stop token or the max_new_tokens budget (drafts beyond the cut are
    # dropped), so ``start == prev_start + accept + 1`` holds in every branch
    accept: int
    start: int  # cached_len after commit == prev_start + accept + 1
    emitted_len: int  # generated tokens emitted so far (accepted drafts + bonus)
    target_cache_len: int  # target_kv.get_seq_length() after crop
    draft_cache_len: Optional[int]  # context-KV length after crop (KV drafter only)


class DFlashSpeculator:
    def __init__(
        self,
        moe: Any,
        draft_model_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import AutoModel

        self.moe = moe
        self.target = getattr(moe, "model", moe)
        self.device = device or _infer_cuda_device(self.target)

        self.draft = (
            AutoModel.from_pretrained(
                draft_model_path, trust_remote_code=True, dtype=dtype
            )
            .to(self.device)
            .eval()
        )

        self.config = read_dflash_config(self.draft.config)
        validate_drafter(self.draft, self.target.config, draft_cfg=self.config)
        self.embed_tokens, self.lm_head = bind_shared_weights(self.draft, self.target)
        self._init_native_runtime()

    @classmethod
    def from_models(
        cls,
        moe: Any,
        draft_model: Any,
        config: Optional[DFlashConfig] = None,
        device: Optional[str] = None,
    ) -> "DFlashSpeculator":
        """Build a speculator from already-instantiated models (no checkpoint load).

        ``moe`` may be the MoE wrapper -- every target forward then routes
        through ``moe._native_model_forward_rich`` so experts stay on the
        standard ExpertExecutor dispatch -- or a bare HF causal LM, which is
        called directly with ``output_hidden_states=True``. The 120B pairing
        constants are not applied here; the module-level ``fc`` contract is.
        """
        self = cls.__new__(cls)
        self.moe = moe
        on_moe_engine = callable(getattr(moe, "_native_model_forward_rich", None))
        self.target = moe.model if on_moe_engine else moe
        if device is None:
            if on_moe_engine:
                device = _infer_cuda_device(self.target)
            else:
                dev = getattr(self.target, "device", None)
                device = (
                    str(dev)
                    if isinstance(dev, torch.device)
                    else str(next(self.target.parameters()).device)
                )
        self.device = device
        self.draft = draft_model
        if config is None:
            config = read_dflash_config(_get(self.draft, "config"))
        self.config = config
        validate_drafter_module(self.draft, self.config)
        self.embed_tokens, self.lm_head = bind_shared_weights(self.draft, self.target)
        self._init_native_runtime()
        return self

    def _init_native_runtime(self) -> None:
        # The reference DFlashDraftModel.forward takes ``noise_embedding`` and
        # maintains a DynamicCache of projected context KV; stateless drafters
        # (tiny fixtures) take ``(block_ids, context_feature)`` instead.
        self._drafter_has_kv_cache = (
            "noise_embedding" in inspect.signature(self.draft.forward).parameters
        )
        # Sliding-window targets retain only the last ``sliding_window - 1``
        # tokens per cache update; the verify block must fit in that span or
        # the snapshot/rebuild rollback cannot recover the block's K/V.
        sliding_window = getattr(
            getattr(self.target, "config", None), "sliding_window", None
        )
        if sliding_window:
            assert int(self.config.block_size) <= int(sliding_window) - 1, (
                f"DFlash block_size {self.config.block_size} must be <= "
                f"target sliding_window - 1 ({int(sliding_window) - 1})"
            )
        self.step_trace: List[NativeStepTrace] = []
        self.last_target_cache: Any = None
        self.last_draft_cache: Any = None

    def _configure_target_hooks(self, input_ids: torch.Tensor) -> None:
        configure = getattr(self.moe, "_configure_hook", None)
        if callable(configure):
            configure(input_ids)
        eval_fn = getattr(self.target, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def _forward_target(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        logits_to_keep: int = 0,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Rich target forward -> on-device (logits, hidden_states, past_key_values).

        ``logits_to_keep=1`` slices to the last position and is for the
        prefill/anchor step only; the verify step MUST pass ``0`` (full
        logits) or the acceptance rule breaks.
        """
        rich = getattr(self.moe, "_native_model_forward_rich", None)
        if callable(rich):
            metadata = (
                None
                if past_key_values is None
                else SimpleNamespace(is_prefill=False)
            )
            token_ids = [int(t) for t in input_ids[0].tolist()]
            return rich(token_ids, metadata, logits_to_keep=logits_to_keep)
        kwargs: dict[str, Any] = {
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
        }
        if logits_to_keep:
            kwargs["logits_to_keep"] = int(logits_to_keep)
        outputs = self.target(input_ids, **kwargs)
        return outputs.logits, outputs.hidden_states, outputs.past_key_values

    def _run_drafter(
        self,
        block: torch.Tensor,
        context_feature: torch.Tensor,
        start: int,
        draft_kv: Any,
    ) -> torch.Tensor:
        """One non-causal drafter pass over ``block`` -> hidden [1, block_size, H].

        Both drafter contracts KV-inject the 5-layer target feature into
        EVERY drafter layer internally. The KV drafter is fed suffix-only
        features and its cache is cropped back to ``start`` after the pass:
        the cache accumulates projected context KV only, so this block's
        noise KV is discarded (mirrors the reference ``spec_generate``).
        """
        if self._drafter_has_kv_cache:
            noise_embedding = self.embed_tokens(block)
            position_ids = torch.arange(
                draft_kv.get_seq_length(),
                start + block.shape[1],
                device=block.device,
                dtype=torch.long,
            ).unsqueeze(0)
            drafter_out = self.draft(
                target_hidden=context_feature,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                past_key_values=draft_kv,
                use_cache=True,
                is_causal=False,
            )
            draft_kv.crop(start)
            return drafter_out
        return self.draft(block, context_feature)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Native greedy DFlash draft->verify->rollback loop (RFC 1.2).

        Per step: draft a block of ``block_size`` candidates with the
        non-causal drafter, verify the whole block in ONE target forward with
        full logits, accept the leading drafts the target's argmax agrees
        with, emit the accepted drafts plus the target's bonus token, and roll
        both KV caches back to the committed prefix. The bonus token is
        emitted but NOT cached -- it becomes the next step's anchor, so
        ``cached_len`` (``start``) always trails the emitted count by one.

        Stop handling: ``stop_token_ids`` (falling back to the target
        config's ``eos_token_id``) truncates the emitted output at the first
        stop id, inclusive, and ends the loop; ``max_new_tokens`` truncates a
        block that would overshoot the remaining budget. Neither cache is
        allowed past the last kept token: keeping ``k`` of a step's
        ``accept + 1`` emitted tokens (``k - 1`` is the stop index) commits
        ``min(k, accept) + 1`` cached tokens -- the anchor plus the first
        ``min(k, accept)`` drafts -- because the verify forward only
        produced KV for block tokens, never for the bonus.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"DFlashSpeculator.generate expects input_ids of shape [1, seq], got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "DFlashSpeculator v1 supports batch==1 only; got input_ids "
                f"with batch size {input_ids.shape[0]}"
            )
        if float(temperature) > 0:
            raise ValueError(
                "DFlashSpeculator v1 is greedy-only (temperature=0.0); "
                "sampled speculative decoding is out of scope"
            )

        from transformers import DynamicCache
        from transformers.cache_utils import DynamicSlidingWindowLayer

        input_ids = input_ids.to(self.device)
        self._configure_target_hooks(input_ids)

        num_prompt_tokens = int(input_ids.shape[1])
        block_size = int(self.config.block_size)
        layer_ids = list(self.config.target_layer_ids)
        max_new_tokens = int(max_new_tokens)

        logits, hidden_states, target_kv = self._forward_target(
            input_ids, past_key_values=None, logits_to_keep=1
        )
        anchor = int(logits[:, -1, :].argmax(dim=-1).item())
        context_feature = extract_context_feature(hidden_states, layer_ids)

        stop_ids = set(_resolve_stop_ids(self.target, stop_token_ids))

        emitted: List[int] = [anchor]
        start = num_prompt_tokens
        draft_kv = DynamicCache() if self._drafter_has_kv_cache else None

        self.step_trace = []
        if stop_ids and anchor in stop_ids and max_new_tokens >= 1:
            # The prefill anchor is itself a stop token: emit it and halt
            # before any block is drafted (nothing past EOS may be emitted
            # or cached; the anchor/bonus is never cached).
            self.last_target_cache = target_kv
            self.last_draft_cache = draft_kv
            new_ids = torch.tensor(
                [emitted], dtype=torch.long, device=input_ids.device
            )
            return torch.cat([input_ids, new_ids], dim=1)

        while len(emitted) < max_new_tokens:
            prev_start = start
            block = build_block(anchor, self.config.mask_token_id, block_size).to(
                self.device
            )

            drafter_out = self._run_drafter(block, context_feature, start, draft_kv)
            draft_logits = self.lm_head(drafter_out)[:, -(block_size - 1) :, :]
            block[:, 1:] = draft_logits.argmax(dim=-1)

            # Snapshot sliding layers BEFORE the verify forward: its update
            # evicts all but the last ``sliding_window - 1`` tokens, so the
            # prefix must be captured here to rebuild after a partial accept.
            sliding_snaps: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
            for i, layer in enumerate(target_kv.layers):
                if isinstance(layer, DynamicSlidingWindowLayer):
                    sliding_snaps[i] = (
                        layer.keys.clone(),
                        layer.values.clone(),
                        int(layer.cumulative_length),
                    )

            logits, hidden_states, target_kv = self._forward_target(
                block, past_key_values=target_kv, logits_to_keep=0
            )
            posterior = logits.argmax(dim=-1)

            accept = acceptance_length(block, posterior)
            committed = committed_tokens(block, posterior, accept)

            # This step's emitted tokens are [d_1 .. d_accept, bonus]; the
            # verify forward produced KV only for [anchor, d_1 .. d_accept].
            # Keeping k emitted tokens (stop index k - 1) therefore commits
            # min(k, accept) + 1 cached tokens: the anchor plus the first
            # min(k, accept) drafts. The bonus is never cached, so a cut at
            # the bonus still commits the full accept + 1 block prefix.
            step_tokens = [int(t) for t in committed.emitted[0].tolist()]
            keep = accept + 1
            stop = False
            if stop_ids:
                for j, tok in enumerate(step_tokens):
                    if tok in stop_ids:
                        keep = j + 1
                        stop = True
                        break
            remaining = max_new_tokens - len(emitted)
            if keep > remaining:
                keep = remaining
                stop = True

            emitted.extend(step_tokens[:keep])
            cache_committed = min(keep, accept) + 1
            start = prev_start + cache_committed
            rollback_target_cache(
                target_kv,
                sliding_snaps,
                prev_start=prev_start,
                committed=cache_committed,
                block_size=block_size,
            )
            assert int(target_kv.get_seq_length()) == start

            self.step_trace.append(
                NativeStepTrace(
                    prev_start=prev_start,
                    accept=cache_committed - 1,
                    start=start,
                    emitted_len=len(emitted),
                    target_cache_len=int(target_kv.get_seq_length()),
                    draft_cache_len=(
                        int(draft_kv.get_seq_length()) if draft_kv is not None else None
                    ),
                )
            )
            if stop:
                break

            suffix = extract_context_feature(hidden_states, layer_ids)[
                :, : accept + 1, :
            ]
            if self._drafter_has_kv_cache:
                context_feature = suffix
            else:
                context_feature = torch.cat([context_feature, suffix], dim=1)

            anchor = int(committed.bonus[0, 0].item())

        self.last_target_cache = target_kv
        self.last_draft_cache = draft_kv

        new_ids = torch.tensor(
            [emitted[:max_new_tokens]], dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, new_ids], dim=1)


__all__ = [
    "DFlashConfig",
    "DFlashSpeculator",
    "read_dflash_config",
    "validate_pairing",
    "validate_drafter",
    "validate_drafter_module",
    "bind_shared_weights",
]

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

import torch

from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    acceptance_lengths,
    build_block,
    build_block_with_prefixes,
    committed_tokens,
    committed_tokens_ragged,
)
from moe_infinity.spec_decode._dflash_sample_ops import (
    acceptance_sampled,
    committed_tokens_sampled,
    warped_probs,
)
from moe_infinity.spec_decode._route_ahead_ctx import route_ahead_context
from moe_infinity.spec_decode._route_ahead_stats import RouteAheadStats

if TYPE_CHECKING:
    from moe_infinity.engine.generation_loop import GenerationEngine
    from moe_infinity.engine.types import SamplingParams


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


def _extract_context_feature(
    hidden_states: Sequence[torch.Tensor], layer_ids: Sequence[int]
) -> torch.Tensor:
    # Lazy import avoids the package cycle big_modeling -> spec_decode -> dflash
    # while preserving big_modeling.extract_context_feature as the one contract.
    from moe_infinity.entrypoints.big_modeling import extract_context_feature

    return extract_context_feature(hidden_states, layer_ids)


def read_dflash_config(draft_hf_config: Any) -> DFlashConfig:
    dflash = _get(draft_hf_config, "dflash_config", {}) or {}
    text_config = _get(draft_hf_config, "text_config", draft_hf_config)

    block_size = _get(draft_hf_config, "block_size", _get(dflash, "block_size"))
    mask_token_id = _get(
        dflash, "mask_token_id", _get(draft_hf_config, "mask_token_id")
    )
    target_layer_ids = _get(
        dflash, "target_layer_ids", _get(draft_hf_config, "target_layer_ids")
    )
    num_target_layers = _get(
        draft_hf_config, "num_target_layers", _get(dflash, "num_target_layers")
    )

    if block_size is None or mask_token_id is None or target_layer_ids is None:
        raise ValueError(
            "draft config is missing required DFlash fields "
            "(block_size, dflash_config.mask_token_id, dflash_config.target_layer_ids)"
        )

    return DFlashConfig(
        block_size=int(block_size),
        mask_token_id=int(mask_token_id),
        target_layer_ids=[int(x) for x in target_layer_ids],
        num_target_layers=int(num_target_layers)
        if num_target_layers is not None
        else -1,
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
    if draft_cfg.block_size < 2:
        raise ValueError(
            f"DFlash block_size {draft_cfg.block_size} must be >= 2 (anchor + >=1 draft token)"
        )
    if not draft_cfg.target_layer_ids or any(
        i < 0 for i in draft_cfg.target_layer_ids
    ):
        raise ValueError(
            f"DFlash target_layer_ids {draft_cfg.target_layer_ids} must be non-empty and non-negative"
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
    # Match the bound shared embed_tokens/lm_head device: an offloaded backbone
    # is resident on the LAST visible GPU, so first-cuda-param/cuda:0 would put
    # the drafter's block on a different GPU than its shared weights.
    try:
        embed = _resolve_input_embeddings(model)
        embed_device = getattr(getattr(embed, "weight", None), "device", None)
        if (
            isinstance(embed_device, torch.device)
            and embed_device.type == "cuda"
        ):
            return str(embed_device)
    except Exception:
        pass
    try:
        for param in model.parameters():
            if param.device.type == "cuda":
                return str(param.device)
    except Exception:
        pass
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    return "cpu"


def _resolve_stop_ids(
    target: Any, stop_token_ids: Optional[List[int]]
) -> List[int]:
    if stop_token_ids is not None:
        return list(stop_token_ids)
    eos = _get(_get(target, "config", target), "eos_token_id")
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(x) for x in eos]


@dataclass(frozen=True)
class SlidingWindowCacheSnapshot:
    keys: torch.Tensor
    values: torch.Tensor
    cumulative_length: int


@dataclass(frozen=True)
class LinearAttentionCacheSnapshot:
    conv_states: Optional[torch.Tensor]
    recurrent_states: Optional[torch.Tensor]
    is_conv_states_initialized: bool
    is_recurrent_states_initialized: bool
    has_previous_state: bool


@dataclass(frozen=True)
class TargetCacheSnapshot:
    sliding: dict[int, SlidingWindowCacheSnapshot]
    linear: dict[int, LinearAttentionCacheSnapshot]


_LINEAR_CACHE_FIELDS = (
    "conv_states",
    "recurrent_states",
    "is_conv_states_initialized",
    "is_recurrent_states_initialized",
    "has_previous_state",
)


def snapshot_target_cache(target_kv: Any) -> TargetCacheSnapshot:
    """Clone rollback-sensitive state before a speculative verify forward."""
    try:
        from transformers.cache_utils import (
            DynamicSlidingWindowLayer,
            LinearAttentionCacheLayerMixin,
        )
    except ImportError as exc:
        raise RuntimeError(
            "unsupported transformers cache API: DFlash hybrid rollback "
            "requires DynamicSlidingWindowLayer and "
            "LinearAttentionCacheLayerMixin"
        ) from exc

    sliding: dict[int, SlidingWindowCacheSnapshot] = {}
    linear: dict[int, LinearAttentionCacheSnapshot] = {}
    for index, layer in enumerate(target_kv.layers):
        present_linear_fields = [
            field for field in _LINEAR_CACHE_FIELDS if hasattr(layer, field)
        ]
        is_linear = isinstance(layer, LinearAttentionCacheLayerMixin)
        if is_linear or present_linear_fields:
            if len(present_linear_fields) != len(_LINEAR_CACHE_FIELDS):
                missing = sorted(
                    field
                    for field in _LINEAR_CACHE_FIELDS
                    if field not in present_linear_fields
                )
                raise RuntimeError(
                    "unsupported transformers linear-attention cache contract "
                    f"for layer {index}: missing fields {missing}"
                )
            conv_states = layer.conv_states
            recurrent_states = layer.recurrent_states
            linear[index] = LinearAttentionCacheSnapshot(
                conv_states=(
                    conv_states.clone() if conv_states is not None else None
                ),
                recurrent_states=(
                    recurrent_states.clone()
                    if recurrent_states is not None
                    else None
                ),
                is_conv_states_initialized=bool(
                    layer.is_conv_states_initialized
                ),
                is_recurrent_states_initialized=bool(
                    layer.is_recurrent_states_initialized
                ),
                has_previous_state=bool(layer.has_previous_state),
            )

        if isinstance(layer, DynamicSlidingWindowLayer):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            cumulative_length = getattr(layer, "cumulative_length", None)
            if keys is None or values is None or cumulative_length is None:
                raise RuntimeError(
                    "unsupported transformers sliding-window cache contract "
                    f"for layer {index}: expected initialized keys, values, "
                    "and cumulative_length"
                )
            sliding[index] = SlidingWindowCacheSnapshot(
                keys=keys.clone(),
                values=values.clone(),
                cumulative_length=int(cumulative_length),
            )

    return TargetCacheSnapshot(sliding=sliding, linear=linear)


def rollback_target_cache(
    target_kv: Any,
    snapshot: TargetCacheSnapshot,
    prev_start: int,
    committed: int,
    block_size: int,
    *,
    block: Optional[torch.Tensor] = None,
    replay: Optional[Callable[[torch.Tensor, Any], Any]] = None,
) -> None:
    """Restore a partial verify and replay its committed prefix exactly."""
    if committed < 0 or committed > block_size:
        raise ValueError(
            f"committed must be in [0, {block_size}], got {committed}"
        )
    if committed == block_size:
        return

    # Legacy full/sliding-only callers (the batched path) remain slice based.
    # Hybrid linear attention cannot use this branch because its recurrent
    # state is not croppable; batch-1 callers provide replay below.
    if replay is None:
        if snapshot.linear:
            raise RuntimeError(
                "hybrid linear-attention cache rollback requires committed-"
                "prefix replay; no replay callback was provided"
            )
        new_len = prev_start + committed
        for index, layer in enumerate(target_kv.layers):
            sliding = snapshot.sliding.get(index)
            if sliding is None:
                layer.crop(new_len)
                continue
            block_k = layer.keys[:, :, -block_size:, :]
            block_v = layer.values[:, :, -block_size:, :]
            full_k = torch.cat(
                [sliding.keys, block_k[:, :, :committed, :]], dim=-2
            )
            full_v = torch.cat(
                [sliding.values, block_v[:, :, :committed, :]], dim=-2
            )
            layer.keys = full_k[:, :, -layer.sliding_window + 1 :, :]
            layer.values = full_v[:, :, -layer.sliding_window + 1 :, :]
            layer.cumulative_length = new_len
        return

    if block is None or block.ndim != 2 or int(block.shape[1]) != block_size:
        raise ValueError(
            "partial cache replay requires block with shape "
            f"[batch, {block_size}]"
        )

    for index, layer in enumerate(target_kv.layers):
        linear = snapshot.linear.get(index)
        if linear is not None:
            for field in ("conv_states", "recurrent_states"):
                saved = getattr(linear, field)
                if saved is None:
                    setattr(layer, field, None)
                else:
                    # Reassign a fresh clone rather than in-place copy_: the
                    # GatedDeltaNet state is an inference tensor (the target
                    # forward runs under inference/no_grad), and an in-place
                    # copy_ on it raises a version-counter bump error, silently
                    # aborting the restore and breaking greedy losslessness.
                    setattr(layer, field, saved.clone())
            layer.is_conv_states_initialized = linear.is_conv_states_initialized
            layer.is_recurrent_states_initialized = (
                linear.is_recurrent_states_initialized
            )
            layer.has_previous_state = linear.has_previous_state

        sliding = snapshot.sliding.get(index)
        if sliding is not None:
            if sliding.cumulative_length != prev_start:
                raise RuntimeError(
                    f"sliding cache layer {index} snapshot length "
                    f"{sliding.cumulative_length} != expected {prev_start}"
                )
            layer.keys = sliding.keys.clone()
            layer.values = sliding.values.clone()
            layer.cumulative_length = sliding.cumulative_length
        else:
            crop = getattr(layer, "crop", None)
            if callable(crop):
                crop(prev_start)
            elif linear is None:
                raise RuntimeError(
                    "unsupported transformers cache layer for DFlash rollback: "
                    f"layer {index} is {type(layer).__name__}"
                )

    replayed_cache = replay(block[:, :committed], target_kv)
    if replayed_cache is not None and replayed_cache is not target_kv:
        raise RuntimeError(
            "target replay replaced the DynamicCache object; in-place hybrid "
            "rollback requires the original cache instance"
        )


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
    emitted_len: (
        int  # generated tokens emitted so far (accepted drafts + bonus)
    )
    target_cache_len: int  # target_kv.get_seq_length() after crop
    draft_cache_len: Optional[
        int
    ]  # context-KV length after crop (KV drafter only)


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
        self.embed_tokens, self.lm_head = bind_shared_weights(
            self.draft, self.target
        )
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
        on_moe_engine = callable(
            getattr(moe, "_native_model_forward_rich", None)
        )
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
        validate_pairing(self.config, self.target.config)
        validate_drafter_module(self.draft, self.config)
        self.embed_tokens, self.lm_head = bind_shared_weights(
            self.draft, self.target
        )
        self._init_native_runtime()
        return self

    def _init_native_runtime(self) -> None:
        # The reference DFlashDraftModel.forward takes ``noise_embedding`` and
        # maintains a DynamicCache of projected context KV; stateless drafters
        # (tiny fixtures) take ``(block_ids, context_feature)`` instead.
        self._drafter_has_kv_cache = (
            "noise_embedding"
            in inspect.signature(self.draft.forward).parameters
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
        # Per-row new-token counts set by the batched path (its ragged rows
        # are right-padded in the returned rectangle). None on the batch==1
        # path, whose output is never padded.
        self.last_generated_lengths: Optional[List[int]] = None
        # Track A5 metrics handle; None (default) = instrumentation off, zero
        # overhead. Set via ``enable_route_ahead_stats``.
        self.route_ahead_stats: Optional[RouteAheadStats] = None

    def enable_route_ahead_stats(self) -> RouteAheadStats:
        """Opt in to Track A5 route-ahead coverage/waste instrumentation.

        Creates (or resets) the ``RouteAheadStats`` recorder consumed by the
        verify-time route-ahead context; read the counters back from
        ``self.route_ahead_stats`` after ``generate()``. Pure observer --
        enabling it never changes routing, prefetch, or emitted tokens.
        """
        if self.route_ahead_stats is None:
            self.route_ahead_stats = RouteAheadStats()
        else:
            self.route_ahead_stats.reset()
        return self.route_ahead_stats

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
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Rich target forward -> on-device (logits, hidden_states, past_key_values).

        ``logits_to_keep=1`` slices to the last position and is for the
        prefill/anchor step only; the verify step MUST pass ``0`` (full
        logits) or the acceptance rule breaks. ``attention_mask`` /
        ``position_ids`` are the Track-C batched-path plumbing (left-padded
        rows need explicit per-row RoPE positions); the batch==1 call sites
        never pass them, so the single-sequence forward is byte-identical.
        """
        rich = getattr(self.moe, "_native_model_forward_rich", None)
        if callable(rich):
            metadata = (
                None
                if past_key_values is None
                else SimpleNamespace(is_prefill=False)
            )
            token_ids = [int(t) for t in input_ids[0].tolist()]
            result = rich(token_ids, metadata, logits_to_keep=logits_to_keep)
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(
                    "_native_model_forward_rich must return "
                    "(logits, hidden_states, past_key_values)"
                )
            logits, hidden_states, past_key_values = result
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError(
                    "native rich forward logits must be a Tensor"
                )
            return logits, hidden_states, past_key_values
        kwargs: dict[str, Any] = {
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
        }
        if logits_to_keep:
            kwargs["logits_to_keep"] = int(logits_to_keep)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        outputs = self.target(input_ids, **kwargs)
        return outputs.logits, outputs.hidden_states, outputs.past_key_values

    def _resolve_route_ahead_prefetcher(self) -> Any:
        """Prefetcher handle bound to the verify-time route-ahead context.

        Production MoE shell: ``moe.engine.expert_prefetcher``
        (``runtime/model_offload.py:865``). Tiny fixtures / bare-HF targets
        have no offload engine -> ``None``; the executor seam then falls back
        to its own configured prefetcher and no-ops when none exists
        (resident mode / ``speculative_prefetch`` config off).
        """
        engine = getattr(self.moe, "engine", None)
        return getattr(engine, "expert_prefetcher", None)

    def _verify_target_block(
        self,
        block: torch.Tensor,
        target_kv: Any,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Verify forward under the Track A3 route-ahead context.

        The context flags executor-backed MoE layers to pin + prefetch their
        ACTUAL routed expert union before reading any expert weight
        (cache warming only; routing/outputs unchanged). Token reset in
        ``finally`` makes the activation exception-safe, so a failed verify
        cannot leak route-ahead state into non-spec decode.

        A4 audit: NO resident-only guard exists anywhere on the spec path --
        ``_init_native_runtime``/``generate`` here,
        ``big_modeling._resolve_spec_strategy`` (big_modeling.py:721), and
        ``generation_loop._spec_strategy_applies`` (generation_loop.py:113)
        gate only on batch size and greedy sampling, never on expert
        residency -- so offloaded executor-backed models (DeepSeek/Qwen/
        Mixtral) already run DFlash with this seam active. gpt-oss stays
        excluded structurally, not by assertion: ``model_offload.py:954``
        never wires ``expert_executor`` into ``SyncGptOssMLP`` (its forward
        runs a resident Python expert loop), ``parse_expert_id`` yields no
        per-expert ids for it (hf_config.py:216-223), and its experts are
        force-resident (model_offload.py:1114-1123).

        A5: when ``self.route_ahead_stats`` is set, the step's observations
        are opened here (``begin_step``) and the handle is carried by the
        context; ``generate`` commits the step after the accept rule.
        """
        stats = getattr(self, "route_ahead_stats", None)
        if stats is not None:
            stats.begin_step()
        # The mask/position kwargs are forwarded only when set so the batch==1
        # call keeps its exact pre-Track-C signature (test doubles wrap it).
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with route_ahead_context(
            self._resolve_route_ahead_prefetcher(), stats=stats
        ):
            return self._forward_target(
                block, past_key_values=target_kv, logits_to_keep=0, **kwargs
            )

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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Union[int, Sequence[int]] = 256,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
        top_k: int = 0,
        top_p: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Native DFlash draft->verify->rollback loop (RFC 1.2).

        Per step: draft a block of ``block_size`` candidates with the
        non-causal drafter, verify the whole block in ONE target forward with
        full logits, accept the leading drafts the target's argmax agrees
        with, emit the accepted drafts plus the target's bonus token, and roll
        both KV caches back to the committed prefix. The bonus token is
        emitted but NOT cached -- it becomes the next step's anchor, so
        ``cached_len`` (``start``) always trails the emitted count by one.

        Sampling: ``temperature == 0`` is the greedy path above (byte-for-byte
        the v1 contract). ``temperature > 0`` (optionally with ``top_k`` /
        ``top_p``) runs the LOSSLESS speculative-sampling accept rule of
        ``_dflash_sample_ops``: drafts are drawn from the drafter's warped
        slot distributions and verified per slot against the identically
        warped target conditionals (rejection sampling + residual
        correction), so the emitted stream follows the exact distribution of
        plain sampled generation from the target. Greedy-vs-sampled only
        changes how tokens are CHOSEN; the commit/rollback bookkeeping below
        is shared.

        Stop handling: ``stop_token_ids`` (falling back to the target
        config's ``eos_token_id``) truncates the emitted output at the first
        stop id, inclusive, and ends the loop; ``max_new_tokens`` truncates a
        block that would overshoot the remaining budget. Neither cache is
        allowed past the last kept token: keeping ``k`` of a step's
        ``accept + 1`` emitted tokens (``k - 1`` is the stop index) commits
        ``min(k, accept) + 1`` cached tokens -- the anchor plus the first
        ``min(k, accept)`` drafts -- because the verify forward only
        produced KV for block tokens, never for the bonus.

        Batching (Track C): ``input_ids`` with batch > 1 dispatches to
        ``_generate_batched`` -- greedy-only (``temperature`` must be 0),
        bare-HF-target-only (the MoE rich-forward seam stays batch==1), with
        LEFT-padded prompts described by ``attention_mask`` (omit it when all
        prompts share one length) and a scalar or per-sequence
        ``max_new_tokens``. The ragged per-row outputs are right-padded into
        the returned rectangle; each row's true new-token count is exposed as
        ``self.last_generated_lengths``. ``attention_mask`` is ignored on the
        batch==1 path.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"DFlashSpeculator.generate expects input_ids of shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        if float(temperature) < 0:
            raise ValueError(
                f"DFlashSpeculator.generate: temperature must be >= 0, got {temperature}"
            )
        if input_ids.shape[0] == 1:
            budget = max_new_tokens
            if isinstance(budget, Sequence):
                if len(budget) != 1:
                    raise ValueError(
                        f"per-sequence max_new_tokens has {len(budget)} entries "
                        f"for batch size 1"
                    )
                budget = budget[0]
            return self._generate_single(
                input_ids,
                max_new_tokens=int(budget),
                temperature=float(temperature),
                stop_token_ids=stop_token_ids,
                top_k=top_k,
                top_p=top_p,
            )
        if float(temperature) > 0:
            raise NotImplementedError(
                "batched DFlash (batch > 1) is greedy-only for now; "
                f"got temperature {temperature}"
            )
        if callable(getattr(self.moe, "_native_model_forward_rich", None)):
            raise NotImplementedError(
                "batched DFlash (batch > 1) requires a bare HF target; the MoE "
                "rich-forward seam is batch==1 (engine-gated) only"
            )
        return self._generate_batched(
            input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _generate_single(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: Optional[List[int]],
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """The v1 single-sequence loop (batch==1), byte-identical to pre-Track-C."""
        sampled = float(temperature) > 0

        from transformers import DynamicCache

        input_ids = input_ids.to(self.device)
        self._configure_target_hooks(input_ids)

        num_prompt_tokens = int(input_ids.shape[1])
        block_size = int(self.config.block_size)
        layer_ids = list(self.config.target_layer_ids)
        max_new_tokens = int(max_new_tokens)

        logits, hidden_states, target_kv = self._forward_target(
            input_ids, past_key_values=None, logits_to_keep=1
        )
        if sampled:
            anchor = int(
                torch.multinomial(
                    warped_probs(logits[0, -1], temperature, top_k, top_p),
                    num_samples=1,
                ).item()
            )
        else:
            anchor = int(logits[:, -1, :].argmax(dim=-1).item())
        context_feature = _extract_context_feature(hidden_states, layer_ids).to(
            self.device
        )

        stop_ids = set(_resolve_stop_ids(self.target, stop_token_ids))

        emitted: List[int] = [anchor]
        start = num_prompt_tokens
        draft_kv = DynamicCache() if self._drafter_has_kv_cache else None

        self.step_trace = []
        self.last_generated_lengths = None
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
            block = build_block(
                anchor, self.config.mask_token_id, block_size
            ).to(self.device)

            drafter_out = self._run_drafter(
                block, context_feature, start, draft_kv
            )
            draft_logits = self.lm_head(drafter_out)[:, -(block_size - 1) :, :]
            draft_probs: Optional[torch.Tensor] = None
            if sampled:
                # The accept test divides by the drafter's OWN warped slot
                # distributions, so drafts must be genuine draws from them --
                # argmax drafts would void the losslessness proof.
                draft_probs = warped_probs(
                    draft_logits[0], temperature, top_k, top_p
                )
                block[:, 1:] = torch.multinomial(
                    draft_probs, num_samples=1
                ).squeeze(-1)
            else:
                block[:, 1:] = draft_logits.argmax(dim=-1)

            cache_snapshot = snapshot_target_cache(target_kv)

            logits, hidden_states, target_kv = self._verify_target_block(
                block, target_kv
            )

            if sampled:
                assert draft_probs is not None
                decision = acceptance_sampled(
                    draft_probs,
                    warped_probs(logits[0], temperature, top_k, top_p),
                    block[0, 1:],
                )
                accept = decision.accept
                committed = committed_tokens_sampled(
                    block, decision.accept, decision.final_token
                )
            else:
                posterior = logits.argmax(dim=-1).to(self.device)
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
                cache_snapshot,
                prev_start=prev_start,
                committed=cache_committed,
                block_size=block_size,
                block=block,
                replay=(
                    (
                        lambda prefix, cache: self._forward_target(
                            prefix,
                            past_key_values=cache,
                            logits_to_keep=0,
                        )[2]
                    )
                    if cache_snapshot.linear
                    else None
                ),
            )
            assert int(target_kv.get_seq_length()) == start

            if self.route_ahead_stats is not None:
                # A5: finalize this verify step's coverage/waste accounting
                # with the kept prefix the accept rule just fixed. Read-only.
                self.route_ahead_stats.commit_step(kept_rows=cache_committed)

            self.step_trace.append(
                NativeStepTrace(
                    prev_start=prev_start,
                    accept=cache_committed - 1,
                    start=start,
                    emitted_len=len(emitted),
                    target_cache_len=int(target_kv.get_seq_length()),
                    draft_cache_len=(
                        int(draft_kv.get_seq_length())
                        if draft_kv is not None
                        else None
                    ),
                )
            )
            if stop:
                break

            suffix = _extract_context_feature(hidden_states, layer_ids).to(
                self.device
            )[:, : accept + 1, :]
            if self._drafter_has_kv_cache:
                context_feature = suffix
            else:
                context_feature = torch.cat([context_feature, suffix], dim=1)

            anchor = int(committed.bonus[0, 0].item())

        self.last_target_cache = target_kv
        self.last_draft_cache = draft_kv

        new_ids = torch.tensor(
            [emitted[:max_new_tokens]],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, new_ids], dim=1)

    @torch.no_grad()
    def _generate_batched(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Union[int, Sequence[int]],
        stop_token_ids: Optional[List[int]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Track-C batched loop: one prefill + one verify per step for all rows.

        Greedy correctness rests on two facts: (1) the emitted stream of a
        greedy verify step is always the target's argmax continuation given
        the committed prefix -- accepted drafts matched the target by
        definition and the bonus/correction IS the target's argmax -- so
        drafts (and hence batching) can only change HOW MANY tokens a step
        commits, never WHICH tokens are emitted; (2) a row's verify logits
        depend only on its own cache row, block prefix, and RoPE positions,
        which the left-pad ``attention_mask`` + per-row ``position_ids``
        plumbing reproduce exactly.

        Per-sequence rollback (C1): HF ``DynamicCache`` is dense -- one
        ``cumulative_length`` per layer and slot-index-based causal/sliding
        masks (``create_causal_mask`` reads ``q_offset`` from the cache, not
        per-row positions) -- so rows cannot physically hold ragged lengths.
        Instead every row rolls back to ``prev_start + min_cc`` where
        ``min_cc`` is the SMALLEST per-row commit among still-active rows
        (``rollback_target_cache`` is reused unchanged: the sliding-window
        snapshot/rebuild is per-row inside the batched tensors). Rows that
        committed more than ``min_cc`` carry the un-cached tail of their
        already-emitted (hence target-true) tokens as the KNOWN PREFIX of
        their next block (``build_block_with_prefixes``); the drafter only
        fills the MASK slots past it, the verify re-caches the prefix, and
        the prefix's re-confirmation tokens are skipped on emission
        (``pending - 1`` of them). Slot distances equal true token distances
        under this uniform-length scheme, so sliding-window masks stay exact.
        """
        from transformers import DynamicCache

        batch, padded_prompt = int(input_ids.shape[0]), int(input_ids.shape[1])
        block_size = int(self.config.block_size)
        layer_ids = list(self.config.target_layer_ids)
        mask_token_id = int(self.config.mask_token_id)

        if isinstance(max_new_tokens, Sequence):
            budgets = [int(x) for x in max_new_tokens]
            if len(budgets) != batch:
                raise ValueError(
                    f"per-sequence max_new_tokens has {len(budgets)} entries "
                    f"for batch size {batch}"
                )
        else:
            budgets = [int(max_new_tokens)] * batch
        if any(b < 0 for b in budgets):
            raise ValueError(f"max_new_tokens must be >= 0, got {budgets}")

        input_ids = input_ids.to(self.device)
        self._configure_target_hooks(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(
                device=self.device, dtype=torch.long
            )
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        if attention_mask.min() < 0 or attention_mask.max() > 1:
            raise ValueError("attention_mask must be 0/1 valued")
        if int(attention_mask[:, -1].min()) != 1:
            raise ValueError(
                "batched DFlash requires LEFT-padded prompts: every row's last "
                "token must be real (attention_mask[:, -1] == 1)"
            )
        steps = attention_mask[:, 1:] - attention_mask[:, :-1]
        if int(steps.min()) < 0:
            raise ValueError(
                "batched DFlash requires LEFT-padded prompts: each "
                "attention_mask row must be 0*1* (pads first, then real tokens)"
            )
        pads = padded_prompt - attention_mask.sum(dim=1)

        prefill_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
        logits, hidden_states, target_kv = self._forward_target(
            input_ids,
            past_key_values=None,
            logits_to_keep=1,
            attention_mask=attention_mask,
            position_ids=prefill_position_ids,
        )
        anchors = logits[:, -1, :].argmax(dim=-1)
        context_feature = _extract_context_feature(hidden_states, layer_ids).to(
            self.device
        )
        stop_ids = set(_resolve_stop_ids(self.target, stop_token_ids))

        emitted: List[List[int]] = [[int(anchors[b])] for b in range(batch)]
        finished: List[bool] = [
            budgets[b] <= 0 or (bool(stop_ids) and int(anchors[b]) in stop_ids)
            for b in range(batch)
        ]
        start = padded_prompt
        draft_kv = DynamicCache() if self._drafter_has_kv_cache else None
        self.step_trace = []

        def active(b: int) -> bool:
            return not finished[b] and len(emitted[b]) < budgets[b]

        while any(active(b) for b in range(batch)):
            prev_start = start
            pendings = [
                len(emitted[b]) - (start - padded_prompt) for b in range(batch)
            ]
            prefixes = [
                emitted[b][start - padded_prompt :] if active(b) else []
                for b in range(batch)
            ]
            block = build_block_with_prefixes(
                prefixes, mask_token_id, block_size
            ).to(self.device)

            drafter_out = self._run_drafter(
                block, context_feature, start, draft_kv
            )
            draft_logits = self.lm_head(drafter_out)[:, -(block_size - 1) :, :]
            for b in range(batch):
                if active(b) and pendings[b] < block_size:
                    block[b, pendings[b] :] = draft_logits[
                        b, pendings[b] - 1 :
                    ].argmax(dim=-1)

            cache_snapshot = snapshot_target_cache(target_kv)

            block_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        batch,
                        start - padded_prompt + block_size,
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
            block_position_ids = torch.arange(
                start, start + block_size, device=self.device, dtype=torch.long
            ).unsqueeze(0) - pads.unsqueeze(1)
            logits, hidden_states, target_kv = self._verify_target_block(
                block,
                target_kv,
                attention_mask=block_attention,
                position_ids=block_position_ids,
            )
            posterior = logits.argmax(dim=-1).to(self.device)
            accepts = acceptance_lengths(block, posterior)
            step_committed = committed_tokens_ragged(block, posterior, accepts)

            # Per-row emission with the re-fed prefix skipped: of this step's
            # ``accept + 1`` emitted tokens the first ``pending - 1`` are
            # re-confirmations of tokens already emitted (the known prefix),
            # so row b newly emits ``step_tokens[pending - 1:]``. Keeping k of
            # those commits ``min(pending - 1 + k, accept) + 1`` cached tokens
            # (the v1 rule ``min(k, accept) + 1`` at pending == 1).
            step_cc: dict[int, int] = {}
            for b in range(batch):
                if not active(b):
                    continue
                pending = pendings[b]
                accept = accepts[b]
                step_tokens = [
                    int(t) for t in step_committed[b].emitted[0].tolist()
                ]
                new_tokens = step_tokens[pending - 1 :]
                keep = len(new_tokens)
                stop = False
                if stop_ids:
                    for j, tok in enumerate(new_tokens):
                        if tok in stop_ids:
                            keep = j + 1
                            stop = True
                            break
                remaining = budgets[b] - len(emitted[b])
                if keep > remaining:
                    keep = remaining
                    stop = True
                emitted[b].extend(new_tokens[:keep])
                step_cc[b] = min(pending - 1 + keep, accept) + 1
                if stop or len(emitted[b]) >= budgets[b]:
                    finished[b] = True

            continuing = [b for b in step_cc if active(b)]
            min_cc = (
                min(step_cc[b] for b in continuing)
                if continuing
                else min(step_cc.values())
            )
            rollback_target_cache(
                target_kv,
                cache_snapshot,
                prev_start=prev_start,
                committed=min_cc,
                block_size=block_size,
            )
            start = prev_start + min_cc
            assert int(target_kv.get_seq_length()) == start

            if self.route_ahead_stats is not None:
                self.route_ahead_stats.commit_step(kept_rows=min_cc)

            for b, cc_b in step_cc.items():
                self.step_trace.append(
                    NativeStepTrace(
                        prev_start=prev_start,
                        accept=cc_b - 1,
                        start=start,
                        emitted_len=len(emitted[b]),
                        target_cache_len=int(target_kv.get_seq_length()),
                        draft_cache_len=(
                            int(draft_kv.get_seq_length())
                            if draft_kv is not None
                            else None
                        ),
                    )
                )
            if not continuing:
                break

            suffix = _extract_context_feature(hidden_states, layer_ids).to(
                self.device
            )[:, :min_cc, :]
            if self._drafter_has_kv_cache:
                context_feature = suffix
            else:
                context_feature = torch.cat([context_feature, suffix], dim=1)

        self.last_target_cache = target_kv
        self.last_draft_cache = draft_kv

        new_lengths = [min(len(emitted[b]), budgets[b]) for b in range(batch)]
        self.last_generated_lengths = new_lengths
        pad_id = _get(_get(self.target, "config", self.target), "pad_token_id")
        pad_id = 0 if pad_id is None else int(pad_id)
        width = max(new_lengths) if new_lengths else 0
        new_ids = torch.full(
            (batch, width), pad_id, dtype=torch.long, device=input_ids.device
        )
        for b in range(batch):
            n = new_lengths[b]
            if n:
                new_ids[b, :n] = torch.tensor(
                    emitted[b][:n], dtype=torch.long, device=input_ids.device
                )
        return torch.cat([input_ids, new_ids], dim=1)

    def run(
        self,
        *,
        engine: GenerationEngine,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> List[int]:
        """``SpecDecodeStrategy`` adapter: engine list contract -> native loop.

        The engine calls this only after its greedy gate applied (batch==1,
        temperature==0, top_p==1, top_k==0); the sampling params are
        forwarded verbatim, so every production call takes the greedy path,
        while direct ``generate()`` callers may opt into the lossless sampled
        path. Target forwards route through this speculator's OWN
        ``self.moe`` rich helper -- never through ``engine`` -- so the
        standard expert dispatch is preserved. ``max_new_tokens`` comes from
        ``sampling_params.max_tokens`` and the stop ids from
        ``engine.eos_token_id`` (the native loop falls back to the target
        config's eos when the engine has none). Returns ONLY the newly
        generated token ids (prompt stripped); the engine wraps them in a
        ``GenerationResult`` and ``MoE.generate`` re-prepends the prompt.
        """
        del request_id  # protocol-conformance parameter; the loop is stateless
        input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long)
        max_new_tokens = int(getattr(sampling_params, "max_tokens", 256))
        eos = getattr(engine, "eos_token_id", None)
        stop_ids = [int(eos)] if isinstance(eos, int) and eos >= 0 else None
        output = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=float(getattr(sampling_params, "temperature", 0.0)),
            stop_token_ids=stop_ids,
            top_k=int(getattr(sampling_params, "top_k", 0) or 0),
            top_p=float(getattr(sampling_params, "top_p", 1.0)),
        )
        return [int(t) for t in output[0, len(prompt_token_ids) :].tolist()]


__all__ = [
    "DFlashConfig",
    "DFlashSpeculator",
    "read_dflash_config",
    "validate_pairing",
    "validate_drafter",
    "validate_drafter_module",
    "bind_shared_weights",
]

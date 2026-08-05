from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch


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
        if not hasattr(self.draft, "spec_generate"):
            raise TypeError(
                f"loaded draft model {type(self.draft).__name__} has no spec_generate; "
                "expected a DFlash drafter checkpoint (e.g. z-lab/gpt-oss-120b-DFlash)"
            )

        self.config = read_dflash_config(self.draft.config)
        validate_drafter(self.draft, self.target.config, draft_cfg=self.config)
        self.embed_tokens, self.lm_head = bind_shared_weights(self.draft, self.target)

    def _configure_target_hooks(self, input_ids: torch.Tensor) -> None:
        configure = getattr(self.moe, "_configure_hook", None)
        if callable(configure):
            configure(input_ids)
        eval_fn = getattr(self.target, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"DFlashSpeculator.generate expects input_ids of shape [1, seq], got {tuple(input_ids.shape)}"
            )

        input_ids = input_ids.to(self.device)
        stops = _resolve_stop_ids(self.target, stop_token_ids)
        self._configure_target_hooks(input_ids)

        return self.draft.spec_generate(
            target=self.target,
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            stop_token_ids=stops,
            temperature=float(temperature),
        )


__all__ = [
    "DFlashConfig",
    "DFlashSpeculator",
    "read_dflash_config",
    "validate_pairing",
    "validate_drafter",
    "validate_drafter_module",
    "bind_shared_weights",
]

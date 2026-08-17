from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _infer_device(model: Any) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    try:
        for p in model.parameters():
            if p.device.type == "cuda":
                return p.device
    except Exception:
        pass
    return torch.device("cpu")


def _infer_dtype(model: Any) -> torch.dtype:
    try:
        for p in model.parameters():
            return p.dtype
    except Exception:
        pass
    return torch.bfloat16


def _resolve_stop_ids(
    model: Any, stop_token_ids: Optional[List[int]]
) -> List[int]:
    if stop_token_ids is not None:
        return list(stop_token_ids)
    cfg = getattr(model, "config", None)
    eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    return [int(x) for x in eos]


class _GlmMtpLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype) -> None:
        super().__init__()
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaDecoderLayer,
            GlmMoeDsaRMSNorm,
        )

        h = config.hidden_size
        self.hnorm = GlmMoeDsaRMSNorm(h, config.rms_norm_eps)
        self.enorm = GlmMoeDsaRMSNorm(h, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        self.decoder_layer = GlmMoeDsaDecoderLayer(config, layer_idx)
        self.shared_head_norm = GlmMoeDsaRMSNorm(h, config.rms_norm_eps)
        self.to(dtype)

    def forward(
        self,
        last_hidden: torch.Tensor,
        token_embed: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: Any,
    ) -> torch.Tensor:
        h_n = self.hnorm(last_hidden)
        e_n = self.enorm(token_embed)
        combined = self.eh_proj(torch.cat([h_n, e_n], dim=-1))
        pos_emb = rotary_emb(combined, position_ids=position_ids)
        out, _ = self.decoder_layer(
            combined,
            position_embeddings=pos_emb,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return self.shared_head_norm(out)


class GlmMtpSpeculator:
    def __init__(self, moe: Any) -> None:
        self.moe = moe
        hf_model = getattr(moe, "model", moe)
        self.hf_model = hf_model
        self.device = _infer_device(hf_model)
        self.dtype = _infer_dtype(hf_model)
        self.last_stats: Dict[str, Any] = {}

        cfg = hf_model.config
        self._build_mtp_layer(cfg)

    def _build_mtp_layer(self, cfg: Any) -> None:
        import copy

        mtp_cfg = copy.deepcopy(cfg)
        num_layers = getattr(mtp_cfg, "num_hidden_layers", 4)
        mtp_layer_idx = num_layers

        if hasattr(mtp_cfg, "mlp_layer_types") and mtp_cfg.mlp_layer_types:
            orig = list(mtp_cfg.mlp_layer_types)
            if len(orig) <= mtp_layer_idx:
                orig.extend(["sparse"] * (mtp_layer_idx - len(orig) + 1))
            orig[mtp_layer_idx] = "sparse"
            mtp_cfg.mlp_layer_types = orig

        if hasattr(mtp_cfg, "layer_types") and mtp_cfg.layer_types:
            orig = list(mtp_cfg.layer_types)
            if len(orig) <= mtp_layer_idx:
                orig.extend([orig[-1]] * (mtp_layer_idx - len(orig) + 1))
            mtp_cfg.layer_types = orig

        if hasattr(mtp_cfg, "indexer_types") and mtp_cfg.indexer_types:
            orig = list(mtp_cfg.indexer_types)
            if len(orig) <= mtp_layer_idx:
                orig.extend(["full"] * (mtp_layer_idx - len(orig) + 1))
            orig[mtp_layer_idx] = "full"
            mtp_cfg.indexer_types = orig

        torch.manual_seed(42)
        self.mtp_layer = _GlmMtpLayer(mtp_cfg, mtp_layer_idx, self.dtype).to(
            self.device
        )
        self.mtp_layer.eval()

    def _greedy_token(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _configure_hooks(self, input_ids: torch.Tensor) -> None:
        configure = getattr(self.moe, "_configure_hook", None)
        if callable(configure):
            configure(input_ids)

    def _forward(self, seq: torch.Tensor):
        hf = self.hf_model
        backbone_out = hf.model(seq, use_cache=False)
        last_hidden = backbone_out.last_hidden_state
        logits = hf.lm_head(last_hidden)
        return logits, last_hidden

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if temperature != 0.0:
            warnings.warn(
                "GlmMtpSpeculator.generate: temperature != 0 is not supported; "
                "falling back to greedy (temperature=0).",
                RuntimeWarning,
                stacklevel=2,
            )

        input_ids = input_ids.to(self.device)
        self._configure_hooks(input_ids)
        self.hf_model.eval()
        stops = set(_resolve_stop_ids(self.hf_model, stop_token_ids))

        hf = self.hf_model
        embed_tokens = hf.model.embed_tokens
        rotary_emb = hf.model.rotary_emb
        lm_head = hf.lm_head

        seq = input_ids.clone()
        generated = 0

        _steps = 0
        _accepted = 0
        _per_step_accepted: List[int] = []

        while generated < max_new_tokens:
            logits, last_hidden = self._forward(seq)
            next_tok = self._greedy_token(logits)

            if int(next_tok.item()) in stops:
                seq = torch.cat([seq, next_tok], dim=1)
                generated += 1
                _steps += 1
                _per_step_accepted.append(0)
                break

            tok_embed = embed_tokens(next_tok)
            seq_len = seq.shape[1] + 1
            pos_ids = torch.tensor(
                [[seq_len - 1]], device=self.device, dtype=torch.long
            )
            mtp_hidden = self.mtp_layer(
                last_hidden[:, -1:, :], tok_embed, pos_ids, rotary_emb
            )
            proposed_logits = lm_head(mtp_hidden)
            proposed_tok = proposed_logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

            verify_seq = torch.cat([seq, next_tok], dim=1)
            verify_logits, _ = self._forward(verify_seq)
            verify_tok = self._greedy_token(verify_logits)

            _steps += 1
            if (
                int(proposed_tok.item()) == int(verify_tok.item())
                and generated + 2 <= max_new_tokens
            ):
                seq = torch.cat([seq, next_tok, proposed_tok], dim=1)
                generated += 2
                _accepted += 1
                _per_step_accepted.append(1)
                if int(proposed_tok.item()) in stops:
                    break
            else:
                seq = torch.cat([seq, next_tok], dim=1)
                generated += 1
                _per_step_accepted.append(0)

        _expert_fetch_events = None
        if os.environ.get("MOE_INFINITY_PROFILE_IO") == "1":
            try:
                from moe_infinity.profiling.io_profiler import IOProfiler

                profiler = IOProfiler.instance()
                _expert_fetch_events = {
                    "cpu_to_gpu": getattr(profiler, "cpu_to_gpu_count", None),
                    "expert_compute": getattr(
                        profiler, "expert_compute_count", None
                    ),
                }
            except Exception:
                pass

        self.last_stats = {
            "steps": _steps,
            "accepted": _accepted,
            "mean_accept_len": 1.0
            + (_accepted / _steps if _steps > 0 else 0.0),
            "per_step_accepted": _per_step_accepted,
            "expert_fetch_events": _expert_fetch_events,
        }

        return seq


def run_glm_mtp_instrumented(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    spec = GlmMtpSpeculator(model)
    spec.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    return spec.last_stats


__all__ = ["GlmMtpSpeculator", "run_glm_mtp_instrumented"]

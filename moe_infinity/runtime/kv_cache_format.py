from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class KVCacheFormatName(str, Enum):
    NATIVE = "native"
    INT8_SYM = "int8_sym"


@dataclass(frozen=True)
class KVCacheFormat:
    name: KVCacheFormatName
    payload_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None

    @classmethod
    def parse(cls, value: str) -> "KVCacheFormat":
        if value == "native":
            return cls(KVCacheFormatName.NATIVE, None, None)
        if value == "int8_sym":
            return cls(KVCacheFormatName.INT8_SYM, torch.int8, torch.float16)
        raise ValueError(f"unsupported KV cache format: {value}")

    def page_size_bytes(
        self,
        *,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        execution_dtype: torch.dtype = torch.float16,
    ) -> int:
        values = 2 * block_size * num_kv_heads * head_dim
        if self.name is KVCacheFormatName.NATIVE:
            return (
                values * torch.empty((), dtype=execution_dtype).element_size()
            )
        scale_values = 2 * block_size * num_kv_heads
        return values + scale_values * 2


@dataclass(frozen=True)
class KVCacheModelInfo:
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    is_mla: bool


@dataclass(frozen=True)
class KVCacheFormatDecision:
    requested_format: KVCacheFormat
    effective_format: KVCacheFormat
    execution_backend: str
    reason: str | None


@dataclass(frozen=True)
class KVCacheBackendCapabilities:
    flashinfer_available: bool
    native_int8_binding_available: bool
    sdpa_available: bool
    native_int8_unavailable_reason: str | None


def resolve_kv_cache_format(
    *,
    requested: str,
    model: KVCacheModelInfo,
    device: torch.device,
    backend_preference: str,
    capabilities: KVCacheBackendCapabilities,
    allow_fallback: bool,
) -> KVCacheFormatDecision:
    requested_format = KVCacheFormat.parse(requested)
    native = KVCacheFormat.parse("native")
    if requested_format.name is KVCacheFormatName.NATIVE:
        return KVCacheFormatDecision(requested_format, native, "existing", None)
    format_failure = None
    if model.is_mla:
        format_failure = "mla_not_validated"
    elif model.num_attention_heads % model.num_kv_heads != 0:
        format_failure = "invalid_gqa_ratio"
    elif model.head_dim % 8 != 0:
        format_failure = "head_dim_not_divisible_by_8"
    if format_failure is not None:
        if not allow_fallback:
            raise RuntimeError(
                f"KV cache format int8_sym unavailable: {format_failure}"
            )
        return KVCacheFormatDecision(
            requested_format, native, "existing", format_failure
        )
    flashinfer_reason = (
        "flashinfer_no_int8_sym_contract"
        if backend_preference == "flashinfer"
        and capabilities.flashinfer_available
        else None
    )
    if device.type == "cuda" and capabilities.native_int8_binding_available:
        return KVCacheFormatDecision(
            requested_format, requested_format, "native_int8", flashinfer_reason
        )
    if capabilities.sdpa_available:
        reason = (
            "cpu_sdpa_dequant"
            if device.type != "cuda"
            else capabilities.native_int8_unavailable_reason
            or "native_int8_binding_missing"
        )
        return KVCacheFormatDecision(
            requested_format,
            requested_format,
            "sdpa_dequant",
            flashinfer_reason or reason,
        )
    if not allow_fallback:
        raise RuntimeError(
            "KV cache format int8_sym unavailable: no_int8_execution_backend"
        )
    return KVCacheFormatDecision(
        requested_format, native, "existing", "no_int8_execution_backend"
    )


def quantize_tokenwise_symmetric(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.all(torch.isfinite(x)):
        raise ValueError("KV cache input must contain only finite values")
    amax = x.float().abs().amax(dim=-1)
    raw_scale = amax / 127.0
    zero = torch.zeros((), dtype=torch.float16, device=x.device)
    inf = torch.full((), float("inf"), dtype=torch.float16, device=x.device)
    min_scale = torch.nextafter(zero, inf).float()
    raw_scale = torch.clamp(raw_scale, min=min_scale)
    candidate = raw_scale.to(torch.float16)
    scale = torch.where(
        candidate.float() < raw_scale,
        torch.nextafter(candidate, inf),
        candidate,
    )
    if not torch.all(torch.isfinite(scale)):
        raise ValueError("KV values require an unrepresentable FP16 scale")
    q = torch.clamp(
        torch.round(x.float() / scale.float().unsqueeze(-1)), -127, 127
    )
    return q.to(torch.int8), scale

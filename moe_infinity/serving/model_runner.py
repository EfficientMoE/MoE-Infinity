# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

import torch

from moe_infinity.boundary.hybrid_attention_layout import (
    linear_attention_layer_indices,
    paged_kv_layer_indices,
)
from moe_infinity.boundary.qwen3_5_moe_mrope import (
    has_vision_grid,
    mrope_prompt_positions,
    supports_mrope_positions,
)
from moe_infinity.memory.expert_policy import (
    ExpertPhase,
    expert_phase_scope,
)
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.attention_types import (
    PagedBatchLengths,
)

from .batch import BatchMetadata, _slice_batch
from .linear_attention_state import LinearAttentionStateCache

if TYPE_CHECKING:
    from moe_infinity.runtime.attention_backend import PrefixReuseCapability
logger = logging.getLogger(__name__)

# Shims whose layers the runner registers with the paged backend (prefix reuse,
# chunked prefill, layered store binding). Matched by class name, like
# ``_get_paged_attention_classes``.
_LAYER_REGISTERED_SHIMS = frozenset(
    {"Qwen3PagedAttention", "OlmoePagedAttention", "Qwen3_5MoePagedAttention"}
)

# Hybrid models (linear-attention + full-attention layers) keep per-sequence
# recurrent state outside the paged KV cache. Reusing a cached KV prefix or
# continuing a chunk would skip that state, so both stay off for them.
HYBRID_PREFIX_REUSE_DISABLED_REASON = (
    "hybrid-linear-attention-state-unsupported"
)
HYBRID_CHUNKED_PREFILL_DISABLED_REASON = (
    "hybrid_linear_attention_state_unsupported"
)


def paged_layer_index(module: object) -> int:
    """Index of ``module``'s plane in the paged KV store.

    Hybrid shims expose a dense ``paged_layer_idx`` (the ordinal among the
    full-attention layers); every other shim is indexed by ``layer_idx``.
    """
    paged_idx = getattr(module, "paged_layer_idx", None)
    if isinstance(paged_idx, int) and not isinstance(paged_idx, bool):
        return paged_idx
    return int(getattr(module, "layer_idx"))


@runtime_checkable
class _ExpertTracerLike(Protocol):
    def create_entry(self) -> object: ...


@runtime_checkable
class _ExpertLayerModuleLike(Protocol):
    seq_id_list: list[object]


@dataclass
class PreparedDecodeBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    active_rows: torch.Tensor
    attention_metadata: RuntimeAttentionMetadata
    batch_bucket: int
    context_bucket: int
    real_batch_size: int = 0

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(tensor.data_ptr() for tensor in self.tensor_values())

    def tensor_values(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.position_ids,
            self.attention_mask,
            self.active_rows,
            self.attention_metadata.block_tables,
            self.attention_metadata.seq_lens,
            self.attention_metadata.slot_mapping,
        )


class ModelRunner:
    model: object
    engine: object
    device: torch.device
    seq_id_list: list[object]

    def __init__(
        self,
        model: object,
        engine: object,
        device: Optional[torch.device] = None,
        *,
        kv_store: object | None = None,
        attention_backend: object | None = None,
        owner_id: str | None = None,
        paged_kv_storage: object = None,
        paged_attention_registry: object = None,
        decode_graph_capability_provider: object = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.device = self._resolve_device(device)
        self.seq_id_list = []
        self.kv_store = kv_store
        self._bound_attention_backend = attention_backend
        self._owner_id = owner_id
        self.paged_kv_storage = paged_kv_storage
        self.paged_attention_registry = paged_attention_registry
        self.decode_graph_capability_provider = decode_graph_capability_provider
        self._warned_no_paged_shim = False
        self._linear_attention_state: LinearAttentionStateCache | None = None
        self._linear_attention_state_resolved = False
        # Per-sequence M-RoPE offset of decode positions (VL Qwen3.5 only).
        self._mrope_deltas: dict[int, int] = {}
        if kv_store is not None or attention_backend is not None:
            self._validate_store_identity(kv_store, attention_backend, owner_id)

    @staticmethod
    def _validate_store_identity(
        kv_store: object | None,
        attention_backend: object | None,
        owner_id: str | None,
    ) -> None:
        if kv_store is None or attention_backend is None:
            raise RuntimeError(
                "ModelRunner requires both kv_store and attention_backend "
                "when either is provided"
            )
        backend_store = getattr(attention_backend, "store", None)
        if backend_store is not kv_store:
            raise RuntimeError(
                "KV store identity mismatch between ModelRunner and backend"
            )
        store_owner = getattr(kv_store, "owner_id", None)
        if owner_id is not None and store_owner != owner_id:
            raise RuntimeError(
                "KV store identity mismatch: owner_id does not match store"
            )
        self.paged_kv_storage = paged_kv_storage
        self.paged_attention_registry = paged_attention_registry
        self.decode_graph_capability_provider = decode_graph_capability_provider
        self._warned_no_paged_shim = False

    def get_paged_kv_storage(self) -> object:
        return self.paged_kv_storage

    def decode_graph_capability(self) -> "DecodeGraphCapability":
        from moe_infinity.runtime.attention_types import (
            DecodeGraphCapability,
            PagedLayerWriteProof,
        )
        from moe_infinity.runtime.paged_kv_storage import canonical_device

        provider = self.decode_graph_capability_provider
        if provider is None:
            engine_capability_fn = getattr(
                self.engine, "decode_graph_capability", None
            )
            if callable(engine_capability_fn):
                provider = self.engine
        if provider is None:
            return DecodeGraphCapability(False, "missing_capability")

        provider_capability = provider.decode_graph_capability()
        if not provider_capability.safe:
            return provider_capability

        backend = self._get_attention_backend()
        if backend is None:
            return DecodeGraphCapability(False, "native_paged_required")

        backend_capability_fn = getattr(
            backend, "decode_graph_capability", None
        )
        if callable(backend_capability_fn):
            backend_capability = backend_capability_fn()
            if not backend_capability.safe:
                return backend_capability

        storage = self.paged_kv_storage
        backend_storage = getattr(backend, "storage", None)
        if storage is None or backend_storage is None:
            return DecodeGraphCapability(False, "kv_storage_mismatch")
        if storage is not backend_storage:
            return DecodeGraphCapability(False, "kv_storage_mismatch")

        kv_cache = getattr(self.engine, "kv_cache", None)
        cache_storage = getattr(kv_cache, "storage", None)
        if kv_cache is not None and cache_storage is not storage:
            return DecodeGraphCapability(False, "kv_storage_mismatch")

        storage_device = canonical_device(storage.spec.device)
        if storage_device != canonical_device(self.device):
            return DecodeGraphCapability(False, "kv_storage_mismatch")
        backend_device = getattr(backend, "device", None)
        if (
            backend_device is not None
            and canonical_device(backend_device) != storage_device
        ):
            return DecodeGraphCapability(False, "kv_storage_mismatch")

        registry = self.paged_attention_registry
        if registry is None:
            return DecodeGraphCapability(False, "paged_class_unregistered")
        registry_reason = getattr(registry, "reason", "eligible")
        if registry_reason != "eligible":
            return DecodeGraphCapability(False, registry_reason)
        bindings = list(getattr(registry, "bindings", []))
        if not bindings:
            return DecodeGraphCapability(False, "paged_class_unregistered")

        if any(
            not getattr(binding, "has_write_proof", False)
            for binding in bindings
        ):
            return DecodeGraphCapability(False, "layer_write_unproven")

        proofs: list[PagedLayerWriteProof] = []
        for binding in bindings:
            proofs.append(
                PagedLayerWriteProof(
                    class_fqn=binding.class_fqn,
                    layer_idx=binding.layer_idx,
                    storage_owner_id=binding.storage_owner_id,
                    writer="moe_infinity.kernel.paged_kv_write.paged_kv_write_",
                    writes_before_attention=True,
                    allocation_free=True,
                )
            )

        return DecodeGraphCapability(
            True,
            "eligible",
            storage_owner_id=storage.owner_id,
            layer_write_proofs=tuple(proofs),
        )
        self._warned_no_paged_shim = False

    def prepare_inputs(
        self, batch: BatchMetadata, *, left_pad: bool = False
    ) -> dict[str, Any]:
        num_seqs = len(batch.seq_ids)
        query_lengths = batch.query_lengths
        query_offsets = batch.query_offsets
        context_lengths = batch.context_lengths
        max_seq_len = max(query_lengths, default=0)

        input_ids = torch.zeros(
            (num_seqs, max_seq_len),
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for seq_idx in range(num_seqs):
            start = query_offsets[seq_idx]
            end = query_offsets[seq_idx + 1]
            seq_tokens = batch.input_token_ids[start:end]
            seq_len = query_lengths[seq_idx]

            if len(seq_tokens) != seq_len:
                raise ValueError(
                    "batch metadata is inconsistent: query_lengths and "
                    "query_offsets disagree"
                )
            if seq_len == 0:
                continue

            token_tensor = torch.tensor(
                seq_tokens,
                dtype=torch.long,
                device=self.device,
            )
            context_len = context_lengths[seq_idx]
            position_tensor = torch.arange(
                context_len,
                context_len + seq_len,
                dtype=torch.long,
                device=self.device,
            )

            start_col = max_seq_len - seq_len if left_pad else 0
            end_col = start_col + seq_len
            input_ids[seq_idx, start_col:end_col] = token_tensor
            position_ids[seq_idx, start_col:end_col] = position_tensor
            attention_mask[seq_idx, start_col:end_col] = 1

        prepared: dict[str, Any] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        if batch.multimodal_inputs is not None:
            for key, value in batch.multimodal_inputs.items():
                prepared[key] = (
                    value.to(self.device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
        return prepared

    def _expert_phase(self, batch: BatchMetadata) -> ExpertPhase:
        if batch.is_prefill and all(batch.is_prefill):
            return ExpertPhase.PREFILL
        if batch.is_prefill and not any(batch.is_prefill):
            return ExpertPhase.DECODE
        return ExpertPhase.MIXED

    def prepare_batch_side_effects(self, batch: BatchMetadata) -> None:
        self._configure_expert_tracing(len(batch.seq_ids))
        self._advance_request_id()

    def execute(
        self,
        batch: BatchMetadata,
        past_key_values: object = None,
    ) -> torch.Tensor:
        self.prepare_batch_side_effects(batch)
        return self._execute(batch, past_key_values=past_key_values, rich=False)

    def execute_rich(
        self,
        batch: BatchMetadata,
        *,
        cache_handles: tuple[object, ...] = (),
    ) -> object:
        """Run the normal serving forward and retain speculative state."""
        self.prepare_batch_side_effects(batch)
        return self._execute(batch, rich=True, cache_handles=cache_handles)

    def _execute(
        self,
        batch: BatchMetadata,
        past_key_values: object = None,
        *,
        rich: bool,
        cache_handles: tuple[object, ...] = (),
    ) -> Any:
        if batch.total_tokens == 0:
            logits = self._empty_logits()
            if not rich:
                return logits
            from moe_infinity.spec_decode.protocols import RichForwardResult

            shared = cache_handles[0] if cache_handles else None
            return RichForwardResult(
                logits=logits,
                hidden_states=(),
                cache_handle=shared,
                cache_handles=cache_handles or (shared,) * len(batch.seq_ids),
                row_offsets=tuple(batch.token_offsets),
                row_lengths=tuple(batch.seq_lengths),
            )

        linear_state = (
            self._get_linear_attention_state()
            if past_key_values is None
            else None
        )
        if linear_state is not None and self._uses_paged_context():
            if rich:
                raise RuntimeError(
                    "speculative (rich) execution is not supported for hybrid "
                    "linear-attention models on the paged serving path: their "
                    "recurrent state cannot be rolled back"
                )
            return self._execute_linear_attention_hybrid(batch, linear_state)

        model_inputs = self.prepare_inputs(batch)

        forward_kwargs: dict[str, object] = {
            **model_inputs,
            "use_cache": True,
        }
        if rich:
            forward_kwargs["output_hidden_states"] = True
        if past_key_values is not None:
            forward_kwargs["past_key_values"] = past_key_values

        with expert_phase_scope(self._expert_phase(batch)):
            outputs = self._forward_with_optional_paged_context(
                forward_kwargs, batch=batch
            )

        logits = self._pack_logits(outputs, model_inputs, batch)
        if not rich:
            return logits

        from moe_infinity.spec_decode.protocols import RichForwardResult

        hidden_value = getattr(outputs, "hidden_states", None)
        if hidden_value is None:
            raise ValueError("rich model output must include hidden_states")
        token_mask = model_inputs["attention_mask"].to(dtype=torch.bool)
        hidden_states = tuple(
            hidden[token_mask.to(device=hidden.device)]
            if hidden.dim() == 3
            else hidden
            for hidden in hidden_value
        )
        cache_handle = getattr(outputs, "past_key_values", None)
        row_handles = cache_handles or (cache_handle,) * len(batch.seq_ids)
        return RichForwardResult(
            logits=logits,
            hidden_states=hidden_states,
            cache_handle=cache_handle,
            cache_handles=row_handles,
            row_offsets=tuple(batch.token_offsets),
            row_lengths=tuple(batch.seq_lengths),
        )

    def _pack_logits(
        self,
        outputs: object,
        model_inputs: dict[str, Any],
        batch: BatchMetadata,
    ) -> torch.Tensor:
        logits = self._extract_logits(outputs)
        if logits.dim() == 3:
            token_mask = model_inputs["attention_mask"].to(
                dtype=torch.bool, device=logits.device
            )
            logits = logits[token_mask]
        elif logits.dim() != 2:
            raise ValueError(
                f"model output logits must be rank-2 or rank-3, got rank-{logits.dim()}"
            )

        if logits.size(0) != batch.total_tokens:
            raise ValueError(
                f"packed logits row count must match batch.total_tokens; got {logits.size(0)} vs {batch.total_tokens}"
            )
        return logits

    def _get_linear_attention_state(self) -> LinearAttentionStateCache | None:
        """Per-sequence linear-attention state for hybrid models, else None."""
        if not self._linear_attention_state_resolved:
            self._linear_attention_state_resolved = True
            config = getattr(self.model, "config", None)
            linear_layers = linear_attention_layer_indices(config)
            if linear_layers:
                self._linear_attention_state = LinearAttentionStateCache(
                    config, linear_layers
                )
        return self._linear_attention_state

    def _uses_paged_context(self) -> bool:
        return bool(
            self._get_paged_attention_classes()
            and self._get_attention_backend() is not None
        )

    def sequence_state_ids(self) -> set[int]:
        """Sequences holding runner-side state (recurrent state, M-RoPE)."""
        state = self._linear_attention_state
        ids = set(self._mrope_deltas)
        if state is not None:
            ids |= state.seq_ids()
        return ids

    def release_sequence_state(self, seq_ids: Iterable[int]) -> None:
        """Drop runner-side state of finished, aborted or reset sequences."""
        seq_ids = list(seq_ids)
        state = self._linear_attention_state
        if state is not None:
            state.release(seq_ids)
        for seq_id in seq_ids:
            self._mrope_deltas.pop(seq_id, None)

    def offload_sequence_state(self, seq_ids: Iterable[int]) -> None:
        """Move recurrent state to host memory while KV is swapped out."""
        state = self._linear_attention_state
        if state is not None:
            state.offload(seq_ids)

    def clear_sequence_state(self) -> None:
        state = self._linear_attention_state
        if state is not None:
            state.clear()
        self._mrope_deltas.clear()

    def sequence_state_reserve_bytes(self, max_sequences: int) -> int:
        """Device bytes to reserve for ``max_sequences`` running sequences."""
        config = getattr(self.model, "config", None)
        dtype = getattr(self.model, "dtype", torch.float32)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float32
        per_sequence = LinearAttentionStateCache.bytes_per_sequence(
            config, dtype
        )
        return max(0, int(max_sequences)) * per_sequence

    @staticmethod
    def _linear_attention_row_groups(batch: BatchMetadata) -> list[list[int]]:
        # Rows starting a sequence share one forward (left-padded so padding
        # precedes the real tokens of the recurrent layers); continuing rows
        # are grouped by query length so they need no padding at all.
        groups: dict[int, list[int]] = {}
        for row, (context_len, query_len) in enumerate(
            zip(batch.context_lengths, batch.query_lengths)
        ):
            key = 0 if context_len == 0 else int(query_len)
            groups.setdefault(key, []).append(row)
        return list(groups.values())

    def _execute_linear_attention_hybrid(
        self, batch: BatchMetadata, state: LinearAttentionStateCache
    ) -> torch.Tensor:
        groups = self._linear_attention_row_groups(batch)
        if len(groups) == 1:
            return self._execute_linear_attention_group(batch, state)

        packed: torch.Tensor | None = None
        for rows in groups:
            sub_batch = _slice_batch(batch, rows)
            if any(batch.context_lengths[row] for row in rows):
                sub_batch.multimodal_inputs = None
            self.prepare_batch_side_effects(sub_batch)
            sub_logits = self._execute_linear_attention_group(sub_batch, state)
            if packed is None:
                packed = sub_logits.new_empty(
                    (batch.total_tokens, sub_logits.size(-1))
                )
            for sub_row, row in enumerate(rows):
                src = sub_batch.query_offsets[sub_row]
                dst = batch.query_offsets[row]
                length = batch.query_lengths[row]
                packed[dst : dst + length] = sub_logits[src : src + length]
        assert packed is not None
        return packed

    def _execute_linear_attention_group(
        self, batch: BatchMetadata, state: LinearAttentionStateCache
    ) -> torch.Tensor:
        fresh = all(context == 0 for context in batch.context_lengths)
        cache = state.build_cache(
            batch.seq_ids, batch.context_lengths, device=self.device
        )
        model_inputs = self.prepare_inputs(batch, left_pad=fresh)
        mrope_positions, mrope_deltas = self._mrope_position_ids(
            batch, model_inputs
        )
        forward_kwargs: dict[str, object] = {
            **model_inputs,
            "use_cache": True,
            "past_key_values": cache,
        }
        if mrope_positions is not None:
            # The token types only feed HF's own position derivation.
            forward_kwargs["position_ids"] = mrope_positions
            forward_kwargs.pop("mm_token_type_ids", None)
        with expert_phase_scope(self._expert_phase(batch)):
            outputs = self._forward_with_optional_paged_context(
                forward_kwargs,
                batch=batch,
                query_padding="left" if fresh else "right",
            )
        logits = self._pack_logits(outputs, model_inputs, batch)
        state.commit(
            batch.seq_ids, batch.context_lengths, batch.query_lengths, cache
        )
        for seq_id, delta in mrope_deltas.items():
            if delta:
                self._mrope_deltas[seq_id] = delta
            else:
                self._mrope_deltas.pop(seq_id, None)
        return logits

    def _mrope_position_ids(
        self, batch: BatchMetadata, model_inputs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, dict[int, int]]:
        """Build HF's ``[4, batch, seq]`` M-RoPE positions for VL Qwen3.5.

        Row 0 is the text position (as ``prepare_inputs`` laid it out, with
        either padding side); rows 1-3 are the temporal/height/width rotary
        positions. Image/video prompts get ``get_rope_index`` positions and
        a per-sequence delta that offsets every later decode position, the
        same way ``generate`` applies ``rope_deltas``. Returns ``None`` when
        no row needs M-RoPE, leaving text-only batches unchanged.
        """
        deltas: dict[int, int] = {}
        if not supports_mrope_positions(self.model):
            return None, deltas
        text_positions = model_inputs["position_ids"]
        token_mask = model_inputs["attention_mask"].to(torch.bool)
        rotary = text_positions.unsqueeze(0).repeat(3, 1, 1)
        row_inputs = batch.row_multimodal_inputs
        needs_mrope = False
        for row, seq_id in enumerate(batch.seq_ids):
            if batch.context_lengths[row] == 0:
                inputs = row_inputs[row] if row_inputs else None
                deltas[seq_id] = 0
                if not has_vision_grid(inputs):
                    continue
                start = batch.query_offsets[row]
                end = batch.query_offsets[row + 1]
                positions, delta = mrope_prompt_positions(
                    self.model,
                    batch.input_token_ids[start:end],
                    cast(dict[str, Any], inputs),
                    device=text_positions.device,
                )
                rotary[:, row, token_mask[row]] = positions.to(rotary.dtype)
                deltas[seq_id] = delta
                needs_mrope = True
            else:
                delta = self._mrope_deltas.get(seq_id, 0)
                if delta:
                    rotary[:, row, token_mask[row]] += delta
                    needs_mrope = True
        if not needs_mrope:
            return None, deltas
        return torch.cat([text_positions.unsqueeze(0), rotary], dim=0), deltas

    def _forward_with_optional_paged_context(
        self,
        forward_kwargs: dict[str, object],
        *,
        batch: BatchMetadata,
        query_padding: str = "right",
    ) -> object:
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            _ = eval_fn()

        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise ValueError("model must define callable forward()")

        paged_attention_classes = self._get_paged_attention_classes()
        backend = self._get_attention_backend()
        use_paged_context = bool(
            paged_attention_classes and backend is not None
        )

        # Backend present but no shim matched => paged cache is never wired into
        # attention and decode silently degenerates; surface it (see issue #191).
        if backend is not None and not paged_attention_classes:
            _msg = (
                "MoE-Infinity serving: no paged-attention shim matched the "
                f"served model ({type(self.model).__name__}); the paged KV "
                "cache is NOT wired into its attention, so multi-token decode "
                "output will be INCORRECT. Implement a *PagedAttention shim for "
                "this model (reference: moe_store/wrappers/qwen3_paged_attention.py)."
            )
            if os.environ.get("MOE_STRICT_PAGED_ATTENTION", "0") == "1":
                raise RuntimeError(_msg)
            if not self._warned_no_paged_shim:
                logger.warning(_msg)
                self._warned_no_paged_shim = True

        with torch.no_grad():
            if not use_paged_context:
                return forward_fn(**forward_kwargs)
            metadata = self._build_runtime_attention_metadata(
                batch, query_padding=query_padding
            )
            for attn_cls in paged_attention_classes:
                attn_cls.set_paged_context(backend, metadata)
            try:
                return forward_fn(**forward_kwargs)
            finally:
                for attn_cls in paged_attention_classes:
                    attn_cls.clear_paged_context()

    def _require_paged_kv_storage(self) -> Any:
        storage = self.paged_kv_storage
        if storage is None:
            raise ValueError("ModelRunner has no bound PagedKVStorage")
        return storage

    def allocate_decode_buffers(
        self, *, batch_bucket: int, context_bucket: int
    ) -> PreparedDecodeBuffers:
        from moe_infinity.runtime.paged_kv_storage import canonical_device

        storage = self._require_paged_kv_storage()
        device = storage.spec.device
        if canonical_device(self.device) != canonical_device(device):
            raise ValueError(
                "ModelRunner device does not match PagedKVStorage device"
            )

        max_blocks = math.ceil(context_bucket / storage.spec.block_size)
        metadata = RuntimeAttentionMetadata(
            block_tables=torch.zeros(
                (batch_bucket, max_blocks), dtype=torch.int32, device=device
            ),
            seq_lens=torch.ones(batch_bucket, dtype=torch.int32, device=device),
            max_seq_len=context_bucket,
            num_prefill_tokens=0,
            num_decode_tokens=batch_bucket,
            slot_mapping=torch.zeros(
                batch_bucket, dtype=torch.int64, device=device
            ),
            is_prefill=False,
            kv_storage_owner_id=storage.owner_id,
        )
        return PreparedDecodeBuffers(
            input_ids=torch.zeros(
                (batch_bucket, 1), dtype=torch.long, device=device
            ),
            position_ids=torch.zeros(
                (batch_bucket, 1), dtype=torch.long, device=device
            ),
            attention_mask=torch.ones(
                (batch_bucket, 1), dtype=torch.long, device=device
            ),
            active_rows=torch.zeros(
                batch_bucket, dtype=torch.bool, device=device
            ),
            attention_metadata=metadata,
            batch_bucket=batch_bucket,
            context_bucket=context_bucket,
        )

    def copy_decode_batch(
        self,
        batch: BatchMetadata,
        buffers: PreparedDecodeBuffers,
        scratch_block_ids: list[int],
    ) -> None:
        storage = self._require_paged_kv_storage()
        device = storage.spec.device
        for tensor in buffers.tensor_values():
            if tensor.device != device:
                raise ValueError(
                    "prepared buffer device does not match storage device"
                )

        real = len(batch.seq_ids)
        if real > buffers.batch_bucket:
            raise ValueError("batch size exceeds prepared batch bucket")
        if any(length != 1 for length in batch.seq_lengths):
            raise ValueError("prepared decode requires one-token sequences")
        if any(batch.is_prefill):
            raise ValueError("prepared decode cannot contain prefill rows")

        metadata = buffers.attention_metadata
        if metadata.kv_storage_owner_id != storage.owner_id:
            raise ValueError("prepared buffers have a foreign storage owner")

        block_size = storage.spec.block_size
        max_blocks = metadata.block_tables.shape[1]
        for row in batch.block_tables:
            for block_id in row:
                if not 0 <= block_id < storage.num_blocks:
                    raise ValueError(
                        f"block id {block_id} outside authoritative storage"
                    )

        padded = buffers.batch_bucket - real
        scratch = list(scratch_block_ids)
        if len(scratch) != padded:
            raise ValueError(
                "one unique scratch block id is required per padded row"
            )
        if len(set(scratch)) != len(scratch):
            raise ValueError("scratch block ids must be unique")
        reserved = storage.graph_scratch_blocks
        for block_id in scratch:
            if block_id not in reserved:
                raise ValueError(f"scratch block id {block_id} is not reserved")

        buffers.input_ids.zero_()
        buffers.position_ids.zero_()
        buffers.attention_mask.fill_(1)
        buffers.active_rows.zero_()
        metadata.block_tables.zero_()
        metadata.seq_lens.fill_(1)
        metadata.slot_mapping.zero_()

        for row_idx in range(real):
            context_len = batch.context_lengths[row_idx]
            seq_len = context_len + 1
            block_table = batch.block_tables[row_idx]
            if seq_len > buffers.context_bucket:
                raise ValueError("sequence length exceeds context bucket")
            token_pos = context_len
            block_idx = token_pos // block_size
            if block_idx >= max_blocks or block_idx >= len(block_table):
                raise ValueError("block table too short for prepared context")
            start = batch.token_offsets[row_idx]
            token_id = batch.input_token_ids[start]

            buffers.input_ids[row_idx, 0] = token_id
            buffers.position_ids[row_idx, 0] = context_len
            buffers.active_rows[row_idx] = True
            for col, block_id in enumerate(block_table):
                metadata.block_tables[row_idx, col] = block_id
            metadata.seq_lens[row_idx] = seq_len
            slot = block_table[block_idx] * block_size + token_pos % block_size
            metadata.slot_mapping[row_idx] = slot

        for offset, block_id in enumerate(scratch):
            row_idx = real + offset
            metadata.block_tables[row_idx, 0] = block_id
            metadata.slot_mapping[row_idx] = block_id * block_size

        buffers.real_batch_size = real

    def forward_prepared_decode(
        self, buffers: PreparedDecodeBuffers
    ) -> torch.Tensor:
        registry = self.paged_attention_registry
        if registry is None:
            raise ValueError(
                "prepared decode requires a paged attention registry"
            )

        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            _ = eval_fn()
        forward_fn = getattr(self.model, "forward", None)
        if not callable(forward_fn):
            raise ValueError("model must define callable forward()")

        registry.install_metadata(buffers.attention_metadata)
        try:
            with torch.no_grad():
                outputs = forward_fn(
                    input_ids=buffers.input_ids,
                    position_ids=buffers.position_ids,
                    attention_mask=buffers.attention_mask,
                    use_cache=True,
                )
        finally:
            registry.clear_metadata()

        logits = self._extract_logits(outputs)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits

    def _configure_expert_tracing(self, num_sequences: int) -> None:
        tracer = getattr(self.engine, "expert_tracer", None)
        if not isinstance(tracer, _ExpertTracerLike) or num_sequences <= 0:
            self.seq_id_list = []
        else:
            self.seq_id_list = [
                tracer.create_entry() for _ in range(num_sequences)
            ]

        for module in getattr(self.engine, "expert_layer_modules", []):
            if isinstance(module, _ExpertLayerModuleLike):
                module.seq_id_list = self.seq_id_list

    def _advance_request_id(self) -> None:
        generator = getattr(self.engine, "_generate_request_id", None)
        if callable(generator):
            _ = generator()
            return

        request_id = getattr(self.engine, "request_id", None)
        if isinstance(request_id, int):
            setattr(self.engine, "request_id", request_id + 1)

    def _empty_logits(self) -> torch.Tensor:
        vocab_size = self._resolve_vocab_size()
        return torch.empty(
            (0, vocab_size), dtype=torch.float32, device=self.device
        )

    def _build_runtime_attention_metadata(
        self, batch: BatchMetadata, *, query_padding: str = "right"
    ) -> RuntimeAttentionMetadata:
        block_size = self._resolve_block_size()
        max_blocks = max((len(row) for row in batch.block_tables), default=0)
        block_tables = torch.zeros(
            (len(batch.block_tables), max_blocks),
            dtype=torch.int32,
            device=self.device,
        )
        for row_idx, row in enumerate(batch.block_tables):
            if row:
                block_tables[row_idx, : len(row)] = torch.tensor(
                    row,
                    dtype=torch.int32,
                    device=self.device,
                )

        query_lengths = batch.query_lengths
        query_offsets = batch.query_offsets
        context_lengths = batch.context_lengths
        kv_seq_lengths = batch.kv_seq_lengths
        seq_lens_values = list(kv_seq_lengths)

        slot_mapping = torch.tensor(
            self._build_slot_mapping(batch, block_size),
            dtype=torch.int64,
            device=self.device,
        )

        num_prefill_tokens = sum(
            seq_len
            for seq_len, is_prefill in zip(query_lengths, batch.is_prefill)
            if is_prefill
        )
        num_decode_tokens = batch.total_tokens - num_prefill_tokens

        return RuntimeAttentionMetadata(
            block_tables=block_tables,
            lengths=PagedBatchLengths(
                query_lengths=torch.tensor(
                    query_lengths, dtype=torch.int32, device=self.device
                ),
                query_offsets=torch.tensor(
                    query_offsets, dtype=torch.int32, device=self.device
                ),
                context_lengths=torch.tensor(
                    context_lengths, dtype=torch.int32, device=self.device
                ),
                kv_seq_lengths=torch.tensor(
                    kv_seq_lengths, dtype=torch.int32, device=self.device
                ),
            ),
            max_seq_len=max(seq_lens_values, default=0),
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            is_prefill=bool(batch.is_prefill and all(batch.is_prefill)),
            query_padding=query_padding,
        )

    @staticmethod
    def _build_slot_mapping(batch: BatchMetadata, block_size: int) -> list[int]:
        slots: list[int] = []
        query_lengths = batch.query_lengths
        context_lengths = batch.context_lengths
        for seq_idx, seq_len in enumerate(query_lengths):
            block_table = batch.block_tables[seq_idx]
            context_len = context_lengths[seq_idx]
            for token_idx in range(seq_len):
                token_pos = context_len + token_idx
                block_idx = token_pos // block_size
                token_offset = token_pos % block_size
                if block_idx >= len(block_table):
                    raise ValueError(
                        "batch metadata block table is too short for sequence length"
                    )
                slots.append(block_table[block_idx] * block_size + token_offset)
        return slots

    def _resolve_block_size(self) -> int:
        kv_cache = getattr(self.engine, "kv_cache", None)
        block_size = getattr(kv_cache, "block_size", None)
        if isinstance(block_size, int) and block_size > 0:
            return block_size

        backend = self._get_attention_backend()
        spec = getattr(backend, "spec", None)
        spec_block_size = getattr(spec, "block_size", None)
        if isinstance(spec_block_size, int) and spec_block_size > 0:
            return spec_block_size

        return 1

    def _get_attention_backend(self) -> object | None:
        if self._bound_attention_backend is not None:
            return self._bound_attention_backend
        getter = getattr(self.engine, "get_attention_backend", None)
        if callable(getter):
            backend = getter()
            if backend is not None:
                return backend

        for attr_name in (
            "attention_backend",
            "_attention_backend",
            "_native_attention_backend",
        ):
            backend = getattr(self.engine, attr_name, None)
            if backend is not None:
                return backend
        return None

    def get_prefix_reuse_capability(
        self, cache: object | None = None
    ) -> "PrefixReuseCapability":
        from moe_infinity.runtime.attention_backend import (
            LayerRegistration,
            PrefixReuseCapability,
        )

        _ = cache
        registrations = self._collect_qwen_layer_registrations()
        expected_layers = self._resolve_expected_layers()
        if expected_layers is None or expected_layers <= 0:
            return PrefixReuseCapability.disabled(
                "incomplete-paged-layer-registry"
            )

        layer_indices = [layer_idx for layer_idx, _ in registrations]
        if len(set(layer_indices)) != len(layer_indices) or set(
            layer_indices
        ) != set(range(expected_layers)):
            return PrefixReuseCapability.disabled(
                "incomplete-paged-layer-registry"
            )

        backend = self._get_attention_backend()
        if backend is None:
            return PrefixReuseCapability.disabled(
                "prefix-aware-prefill-unavailable"
            )

        flashinfer_enabled = getattr(backend, "_flashinfer_enabled", None)
        if not callable(flashinfer_enabled) or not flashinfer_enabled():
            return PrefixReuseCapability.disabled(
                "prefix-aware-prefill-unavailable"
            )

        create_store = getattr(backend, "create_layered_store", None)
        register_layers = getattr(backend, "register_layers", None)
        if not callable(create_store) or not callable(register_layers):
            return PrefixReuseCapability.disabled(
                "prefix-aware-prefill-unavailable"
            )

        store = create_store(layer_count=expected_layers)
        register_layers(
            [
                LayerRegistration(layer_idx=layer_idx, module_id=module_id)
                for layer_idx, module_id in registrations
            ]
        )
        # The layered store above still backs the paged KV of a hybrid model;
        # only reuse is refused, because a cached KV prefix carries no state
        # for the linear-attention layers.
        if self._is_hybrid_linear_attention_model():
            return PrefixReuseCapability.disabled(
                HYBRID_PREFIX_REUSE_DISABLED_REASON
            )
        return PrefixReuseCapability.active(backend, store)

    def _collect_qwen_layer_registrations(self) -> list[tuple[int, int]]:
        registrations: list[tuple[int, int]] = []
        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return registrations
        modules = modules_fn()
        if not isinstance(modules, Iterable):
            return registrations
        for module in modules:
            if module.__class__.__name__ not in _LAYER_REGISTERED_SHIMS:
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if not isinstance(layer_idx, int):
                continue
            registrations.append((paged_layer_index(module), id(module)))
        return registrations

    def _is_hybrid_linear_attention_model(self) -> bool:
        config = getattr(self.model, "config", None)
        return paged_kv_layer_indices(config) is not None

    def _resolve_expected_layers(self) -> int | None:
        config = getattr(self.model, "config", None)
        paged_layers = paged_kv_layer_indices(config)
        if paged_layers is not None:
            return len(paged_layers)
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int):
            return num_hidden_layers
        return None

    def _get_paged_attention_classes(self) -> list[type[Any]]:
        paged_class_names = {
            "DeepseekV2PagedAttention",
            "DeepseekV3PagedAttention",
            "Qwen3PagedAttention",
            "OlmoePagedAttention",
            "Qwen3_5MoePagedAttention",
        }
        classes: list[type[Any]] = []
        seen: set[type[Any]] = set()

        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return classes

        modules = modules_fn()
        if not isinstance(modules, Iterable):
            return classes

        for module in modules:
            cls = module.__class__
            if cls in seen:
                continue
            if cls.__name__ not in paged_class_names:
                continue
            if not hasattr(cls, "set_paged_context") or not hasattr(
                cls, "clear_paged_context"
            ):
                continue
            seen.add(cls)
            classes.append(cls)

        return classes

    def _get_qwen3_paged_attention_modules(self) -> list[Any]:
        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return []
        modules = modules_fn()
        if not isinstance(modules, Iterable):
            return []
        found: list[Any] = []
        for module in modules:
            if module.__class__.__name__ in _LAYER_REGISTERED_SHIMS and hasattr(
                module, "layer_idx"
            ):
                found.append(module)
        return found

    def get_attention_backend(self) -> object | None:
        return self._get_attention_backend()

    def supports_chunked_prefill(self) -> bool:
        from moe_infinity.runtime.attention_backend import (
            LayeredPagedKVStore,
            PagedAttentionBackend,
        )

        if self._is_hybrid_linear_attention_model():
            return False
        backend = self._get_attention_backend()
        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0)
        )
        layer_indices = [int(module.layer_idx) for module in qwen_modules]
        return bool(
            qwen_modules
            and sorted(layer_indices) == list(range(expected_layers))
            and len(set(layer_indices)) == len(layer_indices)
            and isinstance(backend, PagedAttentionBackend)
            and isinstance(
                getattr(backend, "_block_store", None), LayeredPagedKVStore
            )
            and backend.supports_chunked_prefill()
        )

    def chunked_prefill_unavailable_reason(self) -> str:
        from moe_infinity.runtime.attention_backend import (
            LayeredPagedKVStore,
            PagedAttentionBackend,
        )

        if self._is_hybrid_linear_attention_model():
            return HYBRID_CHUNKED_PREFILL_DISABLED_REASON
        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0)
        )
        indices = [int(module.layer_idx) for module in qwen_modules]
        if not qwen_modules or sorted(indices) != list(range(expected_layers)):
            return "incomplete_qwen3_paged_layer_registry"
        backend = self._get_attention_backend()
        if not isinstance(backend, PagedAttentionBackend) or not isinstance(
            getattr(backend, "_block_store", None), LayeredPagedKVStore
        ):
            return "layered_paged_kv_store_unavailable"
        if not backend.supports_chunked_prefill():
            return "paged_backend_lacks_chunk_history"
        return "none"

    def _resolve_vocab_size(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None:
            config_vocab_size = getattr(config, "vocab_size", None)
            if isinstance(config_vocab_size, int):
                return config_vocab_size

        vocab_size = getattr(self.model, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size

        raise ValueError("unable to infer vocab_size from model")

    @staticmethod
    def _extract_logits(outputs: object) -> torch.Tensor:
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            as_dict = dict(outputs)
            logits = as_dict.get("logits")
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]

        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "model forward output does not include tensor logits"
            )
        return logits

    @staticmethod
    def _resolve_device(device: Optional[torch.device]) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        return device


__all__ = ["ModelRunner"]

# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4-Flash expert tensor indexing for MoE-Infinity offloading.

DeepSeek-V4-Flash stores routed experts in the DeepSeek-native checkpoint
format (NOT HuggingFace format). Each routed expert is defined by SIX tensors:

    layers.{L}.ffn.experts.{E}.w1.weight   int8   [2*inter, hidden/2]  (FP4 E2M1 packed)
    layers.{L}.ffn.experts.{E}.w1.scale    f8_e8m0 [2*inter, hidden/32]
    layers.{L}.ffn.experts.{E}.w2.weight   int8   [hidden, inter/2]
    layers.{L}.ffn.experts.{E}.w2.scale    f8_e8m0 [hidden, inter/64]  (block over inter)
    layers.{L}.ffn.experts.{E}.w3.weight   int8   [2*inter? ...]       (gate, == w1 shape)
    layers.{L}.ffn.experts.{E}.w3.scale    f8_e8m0

Concrete verified shapes for the public checkpoint (hidden=4096, inter=2048):
    w1.weight int8 [2048, 2048]   w1.scale f8_e8m0 [2048, 128]
    w2.weight int8 [4096, 1024]   w2.scale f8_e8m0 [4096, 64]
    w3.weight int8 [2048, 2048]   w3.scale f8_e8m0 [2048, 128]

This module does NOT depend on batchgen. It reads the native safetensors index
directly and produces per-expert ``ExpertBundle`` objects so that MoE-Infinity's
offload store can register/stream each expert independently.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

EXPERT_PROJ_NAMES: Tuple[str, ...] = ("w1", "w2", "w3")
EXPERT_PART_NAMES: Tuple[str, ...] = ("weight", "scale")

FP4_PACK_FACTOR = 2
FP4_SCALE_BLOCK = 32


@dataclass(frozen=True)
class TensorRef:
    """A single on-disk tensor: its checkpoint key, dtype, shape, and shard."""

    key: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    shard: str


@dataclass
class ExpertBundle:
    """A single routed expert = 6 tensors + quantization metadata.

    ``tensor_ids`` is populated lazily when the bundle is registered into the
    offload store (one stable uint32 id per tensor, same order as ``tensors``).
    """

    layer_id: int
    expert_id: int
    tensors: List[TensorRef]
    quant_format: str = "fp4_e2m1"
    scale_format: str = "ue8m0"
    block_size: int = FP4_SCALE_BLOCK
    hidden_size: int = 0
    intermediate_size: int = 0
    swiglu_limit: float = 0.0
    tensor_ids: Optional[List[int]] = field(default=None)

    def key_for(self, proj: str, part: str) -> str:
        return (
            f"layers.{self.layer_id}.ffn.experts.{self.expert_id}.{proj}.{part}"
        )

    def part(self, proj: str, part: str) -> TensorRef:
        idx = EXPERT_PROJ_NAMES.index(proj) * len(EXPERT_PART_NAMES) + (
            EXPERT_PART_NAMES.index(part)
        )
        return self.tensors[idx]

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)


class DeepSeekV4ExpertTensorIndexer:
    """Enumerates routed-expert tensors from a native DeepSeek-V4 checkpoint.

    ``config`` may be a dict or object exposing num_hidden_layers,
    n_routed_experts, hidden_size, moe_intermediate_size, swiglu_limit; when
    absent these are read from ``config.json`` in ``ckpt_dir``.
    """

    def __init__(self, ckpt_dir: str, config: Optional[object] = None):
        self.ckpt_dir = ckpt_dir
        index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"missing safetensors index: {index_path}")
        with open(index_path) as f:
            self._weight_map: Dict[str, str] = json.load(f)["weight_map"]

        self._cfg = self._load_config(ckpt_dir, config)
        self.num_hidden_layers: int = int(self._cfg["num_hidden_layers"])
        self.n_routed_experts: int = int(self._cfg["n_routed_experts"])
        self.hidden_size: int = int(self._cfg["hidden_size"])
        self.intermediate_size: int = int(self._cfg["moe_intermediate_size"])
        self.swiglu_limit: float = float(self._cfg.get("swiglu_limit", 0.0))

        self._header_cache: Dict[str, Dict[str, dict]] = {}

    @staticmethod
    def _load_config(ckpt_dir: str, config: Optional[object]) -> Dict:
        if config is not None:
            get = (
                config.get
                if isinstance(config, dict)
                else lambda k, d=None: getattr(config, k, d)
            )
            cfg = {
                "num_hidden_layers": get("num_hidden_layers"),
                "n_routed_experts": get("n_routed_experts"),
                "hidden_size": get("hidden_size"),
                "moe_intermediate_size": get("moe_intermediate_size"),
                "swiglu_limit": get("swiglu_limit", 0.0),
            }
            if all(
                cfg[k] is not None
                for k in (
                    "num_hidden_layers",
                    "n_routed_experts",
                    "hidden_size",
                    "moe_intermediate_size",
                )
            ):
                return cfg
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path) as f:
            raw = json.load(f)
        return {
            "num_hidden_layers": raw["num_hidden_layers"],
            "n_routed_experts": raw["n_routed_experts"],
            "hidden_size": raw["hidden_size"],
            "moe_intermediate_size": raw["moe_intermediate_size"],
            "swiglu_limit": raw.get("swiglu_limit", 0.0),
        }

    _ST_DTYPE_MAP = {
        "I8": torch.int8,
        "U8": torch.uint8,
        "F8_E8M0": torch.float8_e8m0fnu,
        "F8_E4M3": torch.float8_e4m3fn,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "I64": torch.int64,
    }

    def _shard_header(self, shard: str) -> Dict[str, dict]:
        if shard not in self._header_cache:
            import struct

            path = os.path.join(self.ckpt_dir, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self._header_cache[shard] = hdr
        return self._header_cache[shard]

    def _tensor_ref(self, key: str) -> TensorRef:
        shard = self._weight_map.get(key)
        if shard is None:
            raise KeyError(f"tensor key not in index: {key}")
        meta = self._shard_header(shard)[key]
        dtype = self._ST_DTYPE_MAP.get(meta["dtype"])
        if dtype is None:
            raise ValueError(
                f"unmapped safetensors dtype {meta['dtype']} for {key}"
            )
        return TensorRef(
            key=key,
            dtype=dtype,
            shape=tuple(meta["shape"]),
            shard=shard,
        )

    def bundle(self, layer_id: int, expert_id: int) -> ExpertBundle:
        refs: List[TensorRef] = []
        for proj in EXPERT_PROJ_NAMES:
            for part in EXPERT_PART_NAMES:
                key = f"layers.{layer_id}.ffn.experts.{expert_id}.{proj}.{part}"
                refs.append(self._tensor_ref(key))
        return ExpertBundle(
            layer_id=layer_id,
            expert_id=expert_id,
            tensors=refs,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            swiglu_limit=self.swiglu_limit,
        )

    def bundles_for_layer(self, layer_id: int) -> List[ExpertBundle]:
        return [self.bundle(layer_id, e) for e in range(self.n_routed_experts)]

    def load_tensor(self, ref: TensorRef) -> torch.Tensor:
        from safetensors import safe_open

        path = os.path.join(self.ckpt_dir, ref.shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(ref.key)

    def load_bundle_tensors(self, bundle: ExpertBundle) -> List[torch.Tensor]:
        return [self.load_tensor(ref) for ref in bundle.tensors]

# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

# EfficientMoE Team

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from transformers import HfArgumentParser


@dataclass
class ArcherConfig:
    offload_path: str = field(
        default="", metadata={"help": "Path to parameter storage"}
    )
    trace_capacity: int = field(
        default=1000, metadata={"help": "Capacity of trace"}
    )
    trace_path: Optional[os.PathLike[str]] = field(
        default=None, metadata={"help": "Path to trace file"}
    )
    perfect_cache_file: str = field(init=False)
    device_per_node: int = field(init=False)
    prefetch: bool = field(
        default=False, metadata={"help": "Enable prefetching"}
    )
    speculative_prefetch: bool = field(
        default=False,
        metadata={
            "help": "Enable speculative expert prefetching using router logits from layer L to predict L+1 experts."
        },
    )
    speculative_prefetch_overlap: bool = field(
        default=False,
        metadata={
            "help": "When True, fire speculative prefetch BEFORE the layer-L barrier in dispatch_local so PCIe transfers overlap with layer-L compute. When False (default), prefetch fires after the barrier (legacy behavior). Requires speculative_prefetch=True. Currently exposes a cache-pressure failure mode (see .sisyphus/findings/ibp-feasibility/SUMMARY.md) when device_memory_ratio is high; lower device_memory_ratio if you enable this and observe 'All cached expert locked' warnings."
        },
    )
    device_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of device memory to use"},
    )
    num_threads: int = field(
        default=4,
        metadata={
            "help": "Number of parallel expert compute threads per GPU. Higher values overlap expert forward passes on separate CUDA streams, reducing pipeline bubbles."
        },
    )
    host_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of host memory to use"},
    )
    kv_cache_memory_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of GPU memory reserved for KV cache blocks. Default 0.0 (disabled). Set > 0 alongside enable_kv_cache_offload=True. Must satisfy: device_memory_ratio + kv_cache_memory_ratio <= 1.0"
        },
    )
    use_native_engine: bool = field(
        default=True,
        metadata={
            "help": "Enable native serving engine path. Default True. Set False to keep HuggingFace generate() path."
        },
    )
    enable_attention_offload: bool = field(
        default=False,
        metadata={
            "help": "Enable attention backend offloading. Default False (uses HuggingFace attention)."
        },
    )
    enable_kv_cache_offload: bool = field(
        default=False,
        metadata={
            "help": "Enable KV cache CPU offloading. Default False. Requires C++ extension support."
        },
    )
    attention_backend: str = field(
        default="default",
        metadata={
            "help": "Attention backend name. 'default' = no-op PlaceholderAttentionBackend."
        },
    )
    overlap_prefetch_policy: str = field(
        default="off",
        metadata={
            "help": "off, observe, or enforce overlap-window byte admission for speculative expert prefetch."
        },
    )
    overlap_prefetch_ewma_alpha: float = field(
        default=0.2,
        metadata={
            "help": "EWMA smoothing factor in (0, 1] for compute/bandwidth/queue/issue calibration."
        },
    )
    overlap_prefetch_safety_factor: float = field(
        default=0.8,
        metadata={
            "help": "Fraction in (0, 1] of measured compute time usable as the transfer overlap window."
        },
    )
    overlap_prefetch_cold_start_experts: int = field(
        default=1,
        metadata={
            "help": "Max experts admitted before both a compute and a transfer sample exist."
        },
    )
    overlap_prefetch_max_window_bytes: int = field(
        default=256 * 1024 * 1024,
        metadata={
            "help": "Upper bound on the per-layer admitted prefetch window in bytes."
        },
    )
    overlap_prefetch_max_inflight_bytes: int = field(
        default=512 * 1024 * 1024,
        metadata={
            "help": "Upper bound on outstanding speculative prefetch bytes."
        },
    )
    gpu_only_expert_routing: bool = field(
        default=False,
        metadata={
            "help": "Enable GPU-only expert routing; incompatible with an active overlap-prefetch policy in the first release."
        },
    )

    @classmethod
    def load_from_file(cls, config_path: Union[str, os.PathLike]):
        parser = HfArgumentParser(cls)
        config = parser.parse_json_file(json_file=config_path)[0]
        return config

    @classmethod
    def load_from_json(cls, config_json: dict):
        if "glm_fp8_in_store" in config_json:
            warnings.warn(
                "glm_fp8_in_store is deprecated and ignored: GLM-5.2-FP8 routed "
                "experts are always kept FP8 in the host store.",
                DeprecationWarning,
                stacklevel=2,
            )
            config_json = {
                k: v for k, v in config_json.items() if k != "glm_fp8_in_store"
            }
        parser = HfArgumentParser(cls)
        config = parser.parse_dict(config_json)[0]
        return config

    def __post_init__(self):
        self.perfect_cache_file = os.path.join(
            self.offload_path, "perfect_cache"
        )

        self.device_per_node = (
            torch.cuda.device_count()
        )  # always run on heterogeneous nodes

        if self.trace_path is not None:
            self.trace_path = os.path.abspath(self.trace_path)
            if os.path.isdir(self.trace_path):
                raise ValueError(
                    "The trace path should be a file, not a directory."
                )

        kv_autocorrected = False
        if self.use_native_engine and self.kv_cache_memory_ratio == 0.0:
            self.kv_cache_memory_ratio = 0.15
            kv_autocorrected = True
            warnings.warn(
                "kv_cache_memory_ratio was 0.0 with use_native_engine=True; auto-set to 0.15.",
                UserWarning,
                stacklevel=2,
            )

        if (
            kv_autocorrected
            and self.device_memory_ratio + self.kv_cache_memory_ratio > 1.0
        ):
            self.device_memory_ratio = max(
                0.0, 1.0 - self.kv_cache_memory_ratio
            )
            warnings.warn(
                f"device_memory_ratio auto-adjusted to {self.device_memory_ratio:.2f} to satisfy memory budget.",
                UserWarning,
                stacklevel=2,
            )

        if not 0.0 <= self.device_memory_ratio <= 1.0:
            raise ValueError(
                f"device_memory_ratio must be in [0, 1], got {self.device_memory_ratio}"
            )
        if not 0.0 <= self.kv_cache_memory_ratio <= 1.0:
            raise ValueError(
                f"kv_cache_memory_ratio must be in [0, 1], got {self.kv_cache_memory_ratio}"
            )
        if self.device_memory_ratio + self.kv_cache_memory_ratio > 1.0:
            raise ValueError(
                f"device_memory_ratio ({self.device_memory_ratio}) + kv_cache_memory_ratio ({self.kv_cache_memory_ratio}) > 1.0"
            )

        valid_policies = ("off", "observe", "enforce")
        if self.overlap_prefetch_policy not in valid_policies:
            raise ValueError(
                f"overlap_prefetch_policy must be one of {valid_policies}, got {self.overlap_prefetch_policy!r}"
            )
        if not 0.0 < self.overlap_prefetch_ewma_alpha <= 1.0:
            raise ValueError(
                f"overlap_prefetch_ewma_alpha must be in (0, 1], got {self.overlap_prefetch_ewma_alpha}"
            )
        if not 0.0 < self.overlap_prefetch_safety_factor <= 1.0:
            raise ValueError(
                f"overlap_prefetch_safety_factor must be in (0, 1], got {self.overlap_prefetch_safety_factor}"
            )
        if self.overlap_prefetch_cold_start_experts < 0:
            raise ValueError(
                f"overlap_prefetch_cold_start_experts must be >= 0, got {self.overlap_prefetch_cold_start_experts}"
            )
        if self.overlap_prefetch_max_window_bytes < 0:
            raise ValueError(
                f"overlap_prefetch_max_window_bytes must be >= 0, got {self.overlap_prefetch_max_window_bytes}"
            )
        if self.overlap_prefetch_max_inflight_bytes < 0:
            raise ValueError(
                f"overlap_prefetch_max_inflight_bytes must be >= 0, got {self.overlap_prefetch_max_inflight_bytes}"
            )
        if (
            self.overlap_prefetch_policy == "enforce"
            and self.overlap_prefetch_max_window_bytes
            > self.overlap_prefetch_max_inflight_bytes
        ):
            raise ValueError(
                f"overlap_prefetch_max_window_bytes ({self.overlap_prefetch_max_window_bytes}) must be <= "
                f"overlap_prefetch_max_inflight_bytes ({self.overlap_prefetch_max_inflight_bytes}) when policy is enforce"
            )
        if (
            self.gpu_only_expert_routing
            and self.overlap_prefetch_policy != "off"
        ):
            raise ValueError(
                "gpu_only_expert_routing cannot be combined with "
                "overlap_prefetch_policy=observe|enforce in the first release"
            )

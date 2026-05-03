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
    # master_addr: str = field(
    #     default="127.0.0.1",
    #     metadata={"help": "Hosts for running archer"},
    # )
    # master_port: str = field(
    #     default=29500,
    #     metadata={"help": "Port for running archer"},
    # )
    # device_per_node: int = field(
    #     default=1,
    #     metadata={"help": "Number of devices per node"},
    # )
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
        default=1, metadata={"help": "Number of threads for each GPU exec"}
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

    @classmethod
    def load_from_file(cls, config_path: Union[str, os.PathLike]):
        parser = HfArgumentParser(cls)
        config = parser.parse_json_file(json_file=config_path)[0]
        return config

    @classmethod
    def load_from_json(cls, config_json: dict):
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

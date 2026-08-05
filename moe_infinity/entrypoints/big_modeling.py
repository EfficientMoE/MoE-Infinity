# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false

import os
import warnings
from typing import Any, Callable, Dict, Optional, Union

import torch

import moe_infinity

warnings.filterwarnings("ignore")


def extract_context_feature(hidden_states, layer_ids=(1, 9, 17, 25, 33)):
    """Concatenate post-layer hidden states for DFlash drafter conditioning.

    Args:
        hidden_states: Per-layer hidden states from an HF forward with
            ``output_hidden_states=True``. Index 0 is the embedding output, so
            decoder layer ``layer_id``'s output lives at ``layer_id + 1``.
        layer_ids: Decoder layers to concatenate; defaults to the gpt-oss-120b
            DFlash contract ``(1, 9, 17, 25, 33)``.

    Returns:
        Tensor ``[batch, seq_len, len(layer_ids) * hidden_size]`` on the same
        device/dtype as the inputs (e.g. ``[B, L, 14400]`` for 120B).
    """
    if hidden_states is None or len(hidden_states) == 0:
        raise ValueError(
            "hidden_states must be a non-empty tuple; "
            "run the forward with output_hidden_states=True"
        )
    selected = []
    for layer_id in layer_ids:
        idx = int(layer_id) + 1
        if idx >= len(hidden_states):
            raise ValueError(
                f"layer_id {layer_id} maps to hidden_states[{idx}], but only "
                f"{len(hidden_states)} entries were returned"
            )
        selected.append(hidden_states[idx])
    return torch.cat(selected, dim=-1)


class MoE:
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model_name_or_path (`str` or `os.PathLike`): The model to load. It can be:
            - a name of HuggingFace Transformers model
            - a path to a file containing a whole model state dict
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        config (`Dict` or `os.PathLike`): The MoE-Infinity configuration. It can be:
            - a Python dictionary containing the configuration
            - a path to a JSON file containing the configuration

    Example:

    ```python
    >>> from moe_infinity import MoE

    >>> checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    >>> config = "config.json"
    >>> model = MoE(checkpoint, config)

    >>> # You can now use the model as usual
    >>> input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    >>> outputs = model.generate(input_ids)
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        config: Union[str, os.PathLike, Dict] = None,
    ) -> None:
        try:
            from accelerate.utils.versions import is_torch_version
        except Exception:
            is_torch_version = None

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required to load model checkpoints"
            ) from exc

        from moe_infinity.common.constants import MODEL_MAPPING_NAMES
        from moe_infinity.runtime import OffloadEngine
        from moe_infinity.utils import ArcherConfig, get_checkpoint_paths
        from moe_infinity.utils.hf_config import ensure_config_compat
        from moe_infinity.utils.quantization import (
            detect_quantization,
            validate_quantization_support,
        )

        # TODO: remove the torch version check once older versions are supported
        if is_torch_version is not None and not is_torch_version(">=", "2.0"):
            raise RuntimeError(
                "The `load_checkpoint_and_dispatch` function requires PyTorch >= 2.0. "
                "Please update PyTorch."
            )

        if config is None:
            default_config_path = os.path.join(
                os.path.dirname(__file__), "config.json"
            )
            if not os.path.exists(default_config_path):
                raise RuntimeError(
                    "The `load_checkpoint_and_dispatch` function requires a configuration file. "
                    f"Please provide a configuration file or create a default one at {default_config_path}."
                )
            config = default_config_path

        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        model_config = ensure_config_compat(model_config)

        quant_info = detect_quantization(model_config, "")
        if quant_info is not None:
            validate_quantization_support(quant_info, model_name_or_path)

        architectures = getattr(model_config, "architectures", None)
        if not architectures or not isinstance(architectures, list):
            raise RuntimeError("Unable to resolve model architecture")
        architecture = str(architectures[0]).lower()

        arch = None
        # Longest key first so specific arches ("qwen3_5", "deepseek_v3") win
        # over the shorter keys they contain ("qwen3", "deepseek").
        for supp_arch in sorted(MODEL_MAPPING_NAMES, key=len, reverse=True):
            if supp_arch in architecture:
                arch = supp_arch
                break
        if arch is None:
            raise RuntimeError(
                f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
                f"Please provide a model that is supported by the function. "
                f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
            )
        self.arch = arch
        model_cls = MODEL_MAPPING_NAMES[arch]
        # with init_empty_weights():
        #     self.model = model_cls(model_config)
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
            quant_info = detect_quantization(model_config, model_path)
            if quant_info is not None:
                validate_quantization_support(quant_info, model_name_or_path)
            checkpoint_paths = get_checkpoint_paths(model_path)
        else:
            checkpoint_paths = None
            # get the checkpoint download path from huggingface hub
            model_path = snapshot_download(
                model_name_or_path,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
                ignore_patterns=["flax*", "tf*"],
            )
            if model_path is None:
                raise RuntimeError(
                    f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                    f"Please provide a valid checkpoint."
                )

            quant_info = detect_quantization(model_config, model_path)
            if quant_info is not None:
                validate_quantization_support(quant_info, model_name_or_path)

            checkpoint_paths = get_checkpoint_paths(model_path)

        if isinstance(config, dict):
            engine_config = ArcherConfig.load_from_json(config)
        else:
            engine_config = ArcherConfig.load_from_file(config)

        self.use_native_engine = bool(
            getattr(engine_config, "use_native_engine", True)
        )
        # Qwen3.5-MoE interleaves linear (GatedDeltaNet) and full attention; the
        # native paged-KV engine assumes uniform full attention across layers, so
        # Phase 1 drives generation through HF's own forward instead.
        if getattr(model_config, "model_type", "") in ("qwen3_5_moe", "glm_moe_dsa"):
            self.use_native_engine = False
        default_max_seq_length = getattr(
            model_config, "max_position_embeddings", None
        )
        self.max_seq_length = (
            int(default_max_seq_length)
            if isinstance(default_max_seq_length, int)
            else 4096
        )

        native_components = self._build_native_components(
            model_config=model_config,
            engine_config=engine_config,
        )
        attention_backend = native_components.get("attention_backend")
        kv_cache_manager = native_components.get("kv_cache_manager")

        self.engine = OffloadEngine(
            engine_config.trace_capacity,
            model_config,
            attention_backend=attention_backend,
            enable_attention_offload=attention_backend is not None,
            kv_cache_manager=kv_cache_manager,
            enable_kv_cache_offload=(
                bool(
                    getattr(
                        engine_config,
                        "enable_kv_cache_offload",
                        False,
                    )
                )
                and kv_cache_manager is not None
            ),
        )
        self.engine.ckpt_files = checkpoint_paths
        # self.engine.save(config.offload_path, checkpoint_paths)
        is_flash_attn_available = False
        try:
            import flash_attn

            is_flash_attn_available = True

            if (
                arch == "deepseek"
                or arch == "deepseek_v3"
                or arch == "nllb"
                or arch == "gptoss"
                or arch == "qwen3_5"
            ):
                is_flash_attn_available = False
        except ImportError:
            print(
                "[WARNING] FlashAttention is not available in the current environment. Using default attention."
            )
        with self.engine.init(cls=model_cls, ar_config=engine_config):
            self.model = model_cls.from_pretrained(
                model_name_or_path,
                config=model_config,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_available else "eager"
                ),
                is_flash_attn_available=is_flash_attn_available,
                trust_remote_code=True,
            )

        self._ensure_generation_mixin()
        self.engine_config = engine_config

    def _ensure_generation_mixin(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin

        if not hasattr(self.model, "generate"):
            cls = self.model.__class__
            self.model.__class__ = type(
                cls.__name__,
                (GenerationMixin, cls),
                {},
            )
        if getattr(self.model, "generation_config", None) is None:
            self.model.generation_config = GenerationConfig.from_model_config(
                self.model.config
            )

    @staticmethod
    def _resolve_model_int_attr(
        model_config: object, *names: str
    ) -> Optional[int]:
        get_text = getattr(model_config, "get_text_config", None)
        text_config = (
            get_text()
            if callable(get_text)
            else getattr(model_config, "text_config", None)
        )
        for cfg in (model_config, text_config):
            if cfg is None:
                continue
            for name in names:
                value = getattr(cfg, name, None)
                if isinstance(value, int):
                    return value
        return None

    @staticmethod
    def _resolve_torch_dtype(model_config: object) -> torch.dtype:
        torch_dtype = getattr(model_config, "dtype", None)
        if torch_dtype is None:
            torch_dtype = getattr(model_config, "torch_dtype", None)
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            mapping = {
                "float16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "float": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            return mapping.get(torch_dtype.replace("torch.", ""), torch.float16)
        return torch.float16

    def _build_native_components(
        self, model_config: object, engine_config: object
    ) -> dict[str, object]:
        if not self.use_native_engine:
            self._native_memory_coordinator = None
            self._native_kv_cache_manager = None
            self._native_attention_backend = None
            self._native_transfer_scheduler = None
            self._native_scheduler = None
            self._native_generation_engine = None
            self._native_kv_offload_coordinator = None
            return {}

        from moe_infinity.engine.generation_loop import GenerationEngine
        from moe_infinity.engine.scheduler import Scheduler
        from moe_infinity.engine.unified_transfer_scheduler import (
            UnifiedTransferScheduler,
        )
        from moe_infinity.memory.kv_cache_manager import KVCacheManager
        from moe_infinity.memory.memory_coordinator import MemoryCoordinator
        from moe_infinity.runtime.attention_backend import PagedAttentionBackend
        from moe_infinity.runtime.attention_types import KVCacheSpec

        model_num_layers = self._resolve_model_int_attr(
            model_config,
            "num_hidden_layers",
            "num_layers",
            "n_layer",
        )
        num_layers = model_num_layers if model_num_layers is not None else 1

        num_attention_heads = self._resolve_model_int_attr(
            model_config,
            "num_attention_heads",
            "n_head",
        )
        if num_attention_heads is None:
            num_attention_heads = 1

        num_kv_heads = self._resolve_model_int_attr(
            model_config,
            "num_key_value_heads",
            "num_kv_heads",
            "n_head_kv",
        )
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        head_dim = self._resolve_model_int_attr(model_config, "head_dim")
        if head_dim is None:
            hidden_size = self._resolve_model_int_attr(
                model_config, "hidden_size", "n_embd"
            )
            if hidden_size is None:
                hidden_size = max(1, num_attention_heads * 128)
            head_dim = max(1, hidden_size // max(1, num_attention_heads))

        vocab_size = self._resolve_model_int_attr(model_config, "vocab_size")
        if vocab_size is None:
            vocab_size = 32000

        eos_token_id = getattr(model_config, "eos_token_id", 2)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else 2
        if not isinstance(eos_token_id, int):
            eos_token_id = 2

        if (
            self.use_native_engine
            and getattr(engine_config, "kv_cache_memory_ratio", 0.0) == 0.0
        ):
            setattr(engine_config, "kv_cache_memory_ratio", 0.15)
            warnings.warn(
                "kv_cache_memory_ratio was 0.0 with use_native_engine=True; auto-set to 0.15.",
                UserWarning,
                stacklevel=2,
            )

        memory_coordinator = MemoryCoordinator.from_config(
            {
                "device_memory_ratio": float(
                    getattr(engine_config, "device_memory_ratio", 0.75)
                ),
                "kv_cache_memory_ratio": float(
                    getattr(engine_config, "kv_cache_memory_ratio", 0.15)
                ),
                "use_native_engine": True,
            }
        )

        kv_spec = KVCacheSpec(
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=self._resolve_torch_dtype(model_config),
            block_size=16,
        )

        block_size_bytes = kv_spec.page_size_bytes * max(1, int(num_layers))
        num_gpu_blocks = max(
            1, memory_coordinator.compute_num_kv_blocks(block_size_bytes)
        )
        num_cpu_blocks = max(32, num_gpu_blocks * 2)

        kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=kv_spec.block_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            attention_backend = PagedAttentionBackend(
                spec=kv_spec,
                num_gpu_blocks=num_gpu_blocks,
                device=device,
            )
        except Exception:
            attention_backend = None
        transfer_scheduler = UnifiedTransferScheduler()
        kv_offload_coordinator = None
        if getattr(engine_config, "enable_kv_cache_offload", False):
            from moe_infinity.engine.kv_cache_offload_coordinator import (
                KVCacheOffloadCoordinator,
            )

            # kv_tensors=None: The actual KV tensor storage (PagedAttentionBackend
            # k_cache/v_cache) is not yet available at coordinator construction time.
            # Call coordinator.set_kv_tensors(tensors) after model initialisation
            # to enable real tensor copies. Until then, handlers are registered but
            # copy operations are no-ops (guarded by `if self._kv_tensors is None`).
            # This is intentional engine-path scaffolding per the plan scope.
            kv_offload_coordinator = KVCacheOffloadCoordinator(
                kv_tensors=None,
                block_pool=None,
                config=engine_config,
            )
            kv_offload_coordinator.register_with_scheduler(transfer_scheduler)

        scheduler = Scheduler(
            kv_cache_manager=kv_cache_manager,
            transfer_scheduler=transfer_scheduler,
        )

        generation_engine = GenerationEngine(
            kv_cache_manager=kv_cache_manager,
            kv_spec=kv_spec,
            num_layers=int(num_layers),
            vocab_size=int(vocab_size),
            model_forward_fn=self._native_model_forward,
            eos_token_id=int(eos_token_id),
            max_seq_length=self.max_seq_length,
        )

        max_seq_length = self._resolve_model_int_attr(
            model_config,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        )
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
            generation_engine.max_seq_length = max_seq_length

        self._native_memory_coordinator = memory_coordinator
        self._native_kv_cache_manager = kv_cache_manager
        self._native_attention_backend = attention_backend
        self._native_transfer_scheduler = transfer_scheduler
        self._native_scheduler = scheduler
        self._native_generation_engine = generation_engine
        self._native_kv_offload_coordinator = kv_offload_coordinator

        return {
            "memory_coordinator": memory_coordinator,
            "kv_cache_manager": kv_cache_manager,
            "attention_backend": attention_backend,
            "transfer_scheduler": transfer_scheduler,
            "kv_offload_coordinator": kv_offload_coordinator,
            "scheduler": scheduler,
            "generation_engine": generation_engine,
        }

    def _native_model_forward(
        self, token_ids: list[int], _attention_metadata: object
    ) -> torch.Tensor:
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda:0")
        else:
            model_device = getattr(self.model, "device", None)
            if isinstance(
                model_device, torch.device
            ) and model_device.type not in ("meta", "cpu"):
                input_tensor = input_tensor.to(model_device)

        is_prefill = True
        if _attention_metadata is not None:
            is_prefill = bool(getattr(_attention_metadata, "is_prefill", True))

        paged_attention_classes = self._get_paged_attention_classes()
        use_paged_context = bool(
            paged_attention_classes
            and _attention_metadata is not None
            and getattr(self, "_native_attention_backend", None) is not None
        )

        extra_kwargs: dict = {}

        if not use_paged_context:
            # Non-paged path: use HF's KV cache for autoregressive decode.
            # During prefill, use_cache=True captures past_key_values for
            # subsequent decode steps.  During decode, the cached KV provides
            # context from all prior tokens and lets HF auto-compute correct
            # position_ids.
            extra_kwargs["use_cache"] = True
            if not is_prefill:
                cached_kv = getattr(self, "_cached_past_key_values", None)
                if cached_kv is not None:
                    extra_kwargs["past_key_values"] = cached_kv
        else:
            # Paged path: the paged attention backend manages its own KV
            # cache, but the model still needs correct position_ids for
            # rotary embeddings during decode steps.
            if not is_prefill and _attention_metadata is not None:
                seq_lens = getattr(_attention_metadata, "seq_lens", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    current_pos = int(seq_lens[0].item()) - 1
                    extra_kwargs["position_ids"] = torch.tensor(
                        [[current_pos]], device=input_tensor.device
                    )

        with torch.no_grad():
            if not use_paged_context:
                outputs = self.model(input_tensor, **extra_kwargs)
            else:
                backend = self._native_attention_backend
                for attn_cls in paged_attention_classes:
                    attn_cls.set_paged_context(backend, _attention_metadata)
                try:
                    outputs = self.model(input_tensor, **extra_kwargs)
                finally:
                    for attn_cls in paged_attention_classes:
                        attn_cls.clear_paged_context()

        # Capture HF KV cache for next decode step (non-paged path only).
        if not use_paged_context:
            past_kv = getattr(outputs, "past_key_values", None)
            if past_kv is not None:
                self._cached_past_key_values = past_kv

        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]
        if logits is None:
            raise RuntimeError("model forward did not return logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model logits must be a torch.Tensor")
        if logits.ndim == 3:
            return logits[0, -len(token_ids) :, :].detach().to("cpu")
        if logits.ndim == 2:
            return logits.detach().to("cpu")
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")

    def _native_model_forward_rich(
        self,
        token_ids: list[int],
        _attention_metadata: object = None,
        logits_to_keep: int = 0,
    ) -> tuple[torch.Tensor, tuple, object]:
        """On-device forward for speculative decoding: hidden-state capture.

        Single HF forward with ``output_hidden_states=True`` returning
        ``(logits, hidden_states, past_key_values)`` on the model device —
        unlike `_native_model_forward`, nothing is detached to CPU. The cache
        contract mirrors the baseline (non-paged path reads/writes
        ``self._cached_past_key_values``) so callers can roll the returned
        ``DynamicCache`` back via ``crop()``.

        ``logits_to_keep`` passthrough: ``1`` for the anchor/prefill step
        (last-position logits only); ``0`` (the default) keeps full logits and
        MUST be used for the verify step. Experts flow through the exact same
        ``self.model(...)`` call as the baseline path, so the standard
        ExpertExecutor dispatch (and its ``speculative_prefetch`` hook) is
        preserved — nothing here bypasses expert dispatch.
        """
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda:0")
        else:
            model_device = getattr(self.model, "device", None)
            if isinstance(
                model_device, torch.device
            ) and model_device.type not in ("meta", "cpu"):
                input_tensor = input_tensor.to(model_device)

        is_prefill = True
        if _attention_metadata is not None:
            is_prefill = bool(getattr(_attention_metadata, "is_prefill", True))

        paged_attention_classes = self._get_paged_attention_classes()
        use_paged_context = bool(
            paged_attention_classes
            and _attention_metadata is not None
            and getattr(self, "_native_attention_backend", None) is not None
        )

        extra_kwargs: dict = {"output_hidden_states": True}
        if logits_to_keep:
            extra_kwargs["logits_to_keep"] = int(logits_to_keep)

        if not use_paged_context:
            # Same HF KV-cache contract as the baseline: prefill captures
            # past_key_values; decode steps consume the cached KV.
            extra_kwargs["use_cache"] = True
            if not is_prefill:
                cached_kv = getattr(self, "_cached_past_key_values", None)
                if cached_kv is not None:
                    extra_kwargs["past_key_values"] = cached_kv
        else:
            if not is_prefill and _attention_metadata is not None:
                seq_lens = getattr(_attention_metadata, "seq_lens", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    current_pos = int(seq_lens[0].item()) - 1
                    extra_kwargs["position_ids"] = torch.tensor(
                        [[current_pos]], device=input_tensor.device
                    )

        with torch.no_grad():
            if not use_paged_context:
                outputs = self.model(input_tensor, **extra_kwargs)
            else:
                backend = self._native_attention_backend
                for attn_cls in paged_attention_classes:
                    attn_cls.set_paged_context(backend, _attention_metadata)
                try:
                    outputs = self.model(input_tensor, **extra_kwargs)
                finally:
                    for attn_cls in paged_attention_classes:
                        attn_cls.clear_paged_context()

        if not use_paged_context:
            past_kv = getattr(outputs, "past_key_values", None)
            if past_kv is not None:
                self._cached_past_key_values = past_kv

        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]
        if logits is None:
            raise RuntimeError("model forward did not return logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model logits must be a torch.Tensor")

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "model forward did not return hidden_states; "
                "output_hidden_states=True is required"
            )

        past_key_values = getattr(outputs, "past_key_values", None)
        return logits, hidden_states, past_key_values

    def _get_paged_attention_classes(self) -> list[type[Any]]:
        paged_class_names = {
            "DeepseekV2PagedAttention",
            "DeepseekV3PagedAttention",
        }
        classes: list[type[Any]] = []
        seen: set[type[Any]] = set()

        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return classes

        for module in modules_fn():
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

    def _configure_hook(self, input_ids: torch.LongTensor):
        import transformers

        from moe_infinity.models import (
            apply_rotary_pos_emb,
        )

        if self.arch == "mixtral":
            import moe_infinity.models.mixtral  # noqa: F401

        batch_size = input_ids.shape[0]
        self.seq_id_list = [
            self.engine.expert_tracer.create_entry() for _ in range(batch_size)
        ]
        for module in self.engine.expert_layer_modules:
            module.seq_id_list = self.seq_id_list

    def _resolve_spec_strategy(self, speculative_draft):
        """Attach/detach the DFlash speculator on the native engine per call.

        ``speculative_draft`` may be a drafter checkpoint path (loaded via
        ``DFlashSpeculator``), an already-built ``DFlashSpeculator``, or an
        instantiated draft module (wrapped via ``from_models``). Construction
        is memoized by the passed value; ``None``/``False`` detaches so the
        standard path runs, without discarding the memoized speculator.
        """
        engine = self._native_generation_engine
        if not speculative_draft:
            engine.spec_strategy = None
            return
        cached = getattr(self, "_dflash_speculator", None)
        source = getattr(self, "_dflash_speculator_source", None)
        same = source is speculative_draft or (
            isinstance(source, str)
            and isinstance(speculative_draft, (str, os.PathLike))
            and source == str(speculative_draft)
        )
        if cached is None or not same:
            from moe_infinity.spec_decode import DFlashSpeculator

            if isinstance(speculative_draft, DFlashSpeculator):
                cached = speculative_draft
            elif isinstance(speculative_draft, (str, os.PathLike)):
                cached = DFlashSpeculator(self, str(speculative_draft))
            else:
                cached = DFlashSpeculator.from_models(self, speculative_draft)
            self._dflash_speculator = cached
            self._dflash_speculator_source = speculative_draft
        engine.spec_strategy = cached

    def generate(self, input_ids: torch.LongTensor, **kwargs) -> Any:
        """
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation. If `past` is used, only `bos_token_id` is used as
                prompt.
            speculative_draft: optional DFlash drafter (checkpoint path,
                `DFlashSpeculator`, or draft module). When given, greedy
                batch-1 decoding routes through the native speculative
                strategy on the engine; omitted/`None`/`False` uses the
                standard path (per-call, never sticky).
            **kwargs: Additional arguments for the generation method. Check the HuggingFace documentation of the model's
                `generate` method for the supported arguments.

        Returns:
            `torch.LongTensor` of shape `(batch_size, sequence_length)`:
                The generated sequences. Sequences shorter than `min_length` are padded with `pad_token_id`.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("default", DeprecationWarning)
            warnings.warn(
                "MoE.generate() is deprecated. Use MoE.serve() for continuous batching "
                "with higher throughput. MoE.generate() will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        speculative_draft = kwargs.pop("speculative_draft", None)

        if (
            not self.use_native_engine
            or self._native_generation_engine is None
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            if speculative_draft:
                if input_ids.ndim == 2 and input_ids.shape[0] != 1:
                    raise NotImplementedError(
                        "speculative_draft (DFlash) v1 supports batch==1 "
                        f"only; got batch size {input_ids.shape[0]}"
                    )
                raise ValueError(
                    "speculative_draft (DFlash) requires the MoE-Infinity "
                    "native engine (use_native_engine=True)"
                )
            self._configure_hook(input_ids)
            self.model.eval()
            with torch.no_grad():
                return self.model.generate(input_ids, **kwargs)

        from moe_infinity.engine.types import SamplingParams

        self._resolve_spec_strategy(speculative_draft)
        self._configure_hook(input_ids)
        self._cached_past_key_values = None
        self.model.eval()

        prompt_token_ids = [int(token) for token in input_ids[0].tolist()]
        if len(prompt_token_ids) > self.max_seq_length:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} exceeds max_seq_length {self.max_seq_length}"
            )

        max_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 256))
        do_sample = kwargs.get("do_sample", None)
        if do_sample is False:
            sampling_temperature = 0.0
        else:
            sampling_temperature = float(kwargs.get("temperature", 1.0))
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            top_p=float(kwargs.get("top_p", 1.0)),
            top_k=int(kwargs.get("top_k", 0)),
            max_tokens=int(max_tokens) if max_tokens is not None else 256,
        )
        try:
            result = self._native_generation_engine.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        finally:
            self._cached_past_key_values = None

        output_ids = prompt_token_ids + result.output_token_ids
        return torch.tensor(
            [output_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        device_memory_ratio: float = 0.75,
        kv_cache_ratio: float = 0.25,
        max_batch_size: int = 32,
        enable_prefix_caching: bool = False,
        offload_dir: Optional[str] = None,
    ) -> None:
        """
        Start the OpenAI-compatible continuous batching server.

        This is the recommended entry point for production serving.
        Replaces the deprecated MoE.generate() for serving use cases.

        Args:
            host: Server host (default: 0.0.0.0)
            port: Server port (default: 8000)
            device_memory_ratio: GPU memory fraction for caching (default: 0.75)
            kv_cache_ratio: Fraction of device_memory_ratio for KV cache (default: 0.25)
            max_batch_size: Maximum concurrent sequences (default: 32)
            enable_prefix_caching: Enable hash-based prefix caching (default: False)
            offload_dir: Path to offload directory (required)
        """

        if offload_dir is None:
            raise ValueError("offload_dir is required for MoE.serve()")

        import importlib

        from moe_infinity.entrypoints.openai import api_server_v2

        model_name = getattr(
            getattr(self.model, "config", None), "_name_or_path", None
        )
        api_server_v2.initialize_with_model(
            moe_model=self,
            model_name=model_name,
            tok=getattr(self, "tokenizer", None),
            max_seq_length=self.max_seq_length,
            device_memory_ratio=device_memory_ratio,
            kv_cache_ratio=kv_cache_ratio,
            max_batch_size=max_batch_size,
            enable_prefix_caching=enable_prefix_caching,
        )

        uvicorn = importlib.import_module("uvicorn")
        uvicorn.run(api_server_v2.app, host=host, port=port)

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """

        self._configure_hook(input_ids)

        return self.model(input_ids, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """
        return self.forward(*args, **kwargs)

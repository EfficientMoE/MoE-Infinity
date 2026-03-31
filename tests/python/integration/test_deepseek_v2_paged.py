import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from moe_infinity.runtime.attention_types import KVCacheSpec


def _load_deepseek_v2_modeling_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    models_dir = repo_root / "moe_infinity" / "models"
    v2_dir = models_dir / "modeling_deepseek_v2"

    models_pkg = types.ModuleType("moe_infinity.models")
    models_pkg.__path__ = [str(models_dir)]
    sys.modules["moe_infinity.models"] = models_pkg

    v2_pkg = types.ModuleType("moe_infinity.models.modeling_deepseek_v2")
    v2_pkg.__path__ = [str(v2_dir)]
    sys.modules["moe_infinity.models.modeling_deepseek_v2"] = v2_pkg

    module_name = "moe_infinity.models.modeling_deepseek_v2.modeling_deepseek"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        module_name,
        v2_dir / "modeling_deepseek.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load modeling_deepseek module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODEL_MODULE = _load_deepseek_v2_modeling_module()
ATTENTION_CLASSES = MODEL_MODULE.ATTENTION_CLASSES
DeepseekV2PagedAttention = MODEL_MODULE.DeepseekV2PagedAttention


def test_kv_cache_spec_from_config() -> None:
    config = SimpleNamespace(kv_lora_rank=512)
    spec_params = DeepseekV2PagedAttention.get_kv_cache_spec_for_config(config)
    assert spec_params == {"num_kv_heads": 1, "head_dim": 512}


def test_kv_cache_spec_standard() -> None:
    config = SimpleNamespace(
        kv_lora_rank=None,
        num_key_value_heads=8,
        num_attention_heads=16,
        hidden_size=4096,
    )
    spec_params = DeepseekV2PagedAttention.get_kv_cache_spec_for_config(config)
    assert spec_params == {"num_kv_heads": 8, "head_dim": 256}


def test_attention_classes_registration() -> None:
    assert "paged" in ATTENTION_CLASSES
    assert ATTENTION_CLASSES["paged"] is DeepseekV2PagedAttention


def test_mla_block_size() -> None:
    spec_params = DeepseekV2PagedAttention.get_kv_cache_spec_for_config(
        SimpleNamespace(kv_lora_rank=512)
    )
    spec = KVCacheSpec(**spec_params, dtype=torch.float16, block_size=16)
    assert spec.page_size_bytes == 32768

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_v2_dir = ROOT / "moe_infinity" / "models" / "modeling_deepseek_v2"
_v3_dir = ROOT / "moe_infinity" / "models" / "modeling_deepseek_v3"
if not _v2_dir.is_dir() or not _v3_dir.is_dir():
    pytest.skip(
        "vendored modeling_deepseek_v2/v3 removed; migrated to upstream transformers",
        allow_module_level=True,
    )

_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.models", ROOT / "moe_infinity" / "models")
_ensure_package(
    "moe_infinity.models.modeling_deepseek_v2",
    _v2_dir,
)
_ensure_package(
    "moe_infinity.models.modeling_deepseek_v3",
    _v3_dir,
)

v2_config_module = _load_module(
    "moe_infinity.models.modeling_deepseek_v2.configuration_deepseek",
    _v2_dir / "configuration_deepseek.py",
)
v2_modeling = _load_module(
    "moe_infinity.models.modeling_deepseek_v2.modeling_deepseek",
    _v2_dir / "modeling_deepseek.py",
)
v3_config_module = _load_module(
    "moe_infinity.models.modeling_deepseek_v3.configuration_deepseek",
    _v3_dir / "configuration_deepseek.py",
)
v3_modeling = _load_module(
    "moe_infinity.models.modeling_deepseek_v3.modeling_deepseek",
    ROOT
    / "moe_infinity"
    / "models"
    / "modeling_deepseek_v3"
    / "modeling_deepseek.py",
)

DeepseekV2Config = v2_config_module.DeepseekV2Config
DeepseekV3Config = v3_config_module.DeepseekV3Config


@dataclass
class _BackendCall:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_metadata: object
    scale: float | None


class _RecordingBackend:
    def __init__(self, fill_value: float = 0.5) -> None:
        self.fill_value = fill_value
        self.calls: list[_BackendCall] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: object = None,
        attn_metadata: object = None,
        attention_metadata: object = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        _ = (kv_cache, attn_metadata)
        self.calls.append(
            _BackendCall(
                query=query,
                key=key,
                value=value,
                attention_metadata=attention_metadata,
                scale=scale,
            )
        )
        return torch.full(
            (query.shape[0], query.shape[1], value.shape[2]),
            fill_value=self.fill_value,
            dtype=query.dtype,
            device=query.device,
        )


def _make_v2_attention():
    config = DeepseekV2Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        attention_dropout=0.0,
    )
    return v2_modeling.DeepseekV2PagedAttention(config, layer_idx=0).eval()


def _make_v3_attention():
    config = DeepseekV3Config(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        n_shared_experts=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_group=None,
        topk_group=None,
        attention_dropout=0.0,
    )
    return v3_modeling.DeepseekV3PagedAttention(config, layer_idx=0).eval()


@pytest.fixture(autouse=True)
def _clear_paged_context():
    v2_modeling.DeepseekV2PagedAttention.clear_paged_context()
    v3_modeling.DeepseekV3PagedAttention.clear_paged_context()
    yield
    v2_modeling.DeepseekV2PagedAttention.clear_paged_context()
    v3_modeling.DeepseekV3PagedAttention.clear_paged_context()


def test_deepseek_v3_paged_attention_uses_backend() -> None:
    attn = _make_v3_attention()
    backend = _RecordingBackend(fill_value=0.25)
    metadata = object()
    v3_modeling.DeepseekV3PagedAttention.set_paged_context(backend, metadata)

    hidden_states = torch.randn(1, 3, attn.hidden_size)
    attention_mask = torch.zeros(1, 1, 3, 3)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)

    outputs, attn_weights, past_key_value = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )

    assert outputs.shape == hidden_states.shape
    assert attn_weights is None
    assert past_key_value is None
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.attention_metadata is metadata
    assert call.query.shape == (3, attn.num_heads, attn.q_head_dim)
    assert call.key.shape == (3, attn.num_heads, attn.q_head_dim)
    assert call.value.shape == (3, attn.num_heads, attn.v_head_dim)
    assert call.scale == pytest.approx(attn.softmax_scale)


def test_deepseek_v3_paged_attention_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (
        torch.randn(1, 1, 1),
        torch.randn(1, 1, 1, 1),
        object(),
    )

    def _fake_super_forward(*args, **kwargs):
        _ = (args, kwargs)
        return sentinel

    monkeypatch.setattr(
        v3_modeling.DeepseekV3Attention, "forward", _fake_super_forward
    )
    v3_modeling.DeepseekV3PagedAttention.clear_paged_context()

    attn = v3_modeling.DeepseekV3PagedAttention.__new__(
        v3_modeling.DeepseekV3PagedAttention
    )
    torch.nn.Module.__init__(attn)

    outputs = attn.forward(hidden_states=torch.zeros(1, 1, 1))
    assert outputs is sentinel


def test_deepseek_v2_paged_attention_uses_backend() -> None:
    attn = _make_v2_attention()
    backend = _RecordingBackend(fill_value=1.5)
    metadata = object()
    v2_modeling.DeepseekV2PagedAttention.set_paged_context(backend, metadata)

    hidden_states = torch.randn(1, 3, attn.hidden_size)
    attention_mask = torch.zeros(1, 1, 3, 3)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)

    outputs, attn_weights, past_key_value = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )

    assert outputs.shape == hidden_states.shape
    assert attn_weights is None
    assert past_key_value is None
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.attention_metadata is metadata
    assert call.query.shape == (3, attn.num_heads, attn.q_head_dim)
    assert call.key.shape == (3, attn.num_heads, attn.q_head_dim)
    assert call.value.shape == (3, attn.num_heads, attn.v_head_dim)
    assert call.scale == pytest.approx(attn.softmax_scale)


def test_deepseek_v2_paged_attention_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = (
        torch.randn(1, 1, 1),
        torch.randn(1, 1, 1, 1),
        object(),
    )

    def _fake_super_forward(*args, **kwargs):
        _ = (args, kwargs)
        return sentinel

    monkeypatch.setattr(
        v2_modeling.DeepseekV2Attention, "forward", _fake_super_forward
    )
    v2_modeling.DeepseekV2PagedAttention.clear_paged_context()

    attn = v2_modeling.DeepseekV2PagedAttention.__new__(
        v2_modeling.DeepseekV2PagedAttention
    )
    torch.nn.Module.__init__(attn)

    outputs = attn.forward(hidden_states=torch.zeros(1, 1, 1))
    assert outputs is sentinel

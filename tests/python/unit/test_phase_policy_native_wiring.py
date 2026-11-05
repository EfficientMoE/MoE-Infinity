# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest

from moe_infinity.runtime.model_offload import OffloadEngine
from moe_infinity.utils import ArcherConfig


class _RecordingHandle:
    def __init__(self, accept=True) -> None:
        self.calls = []
        self.eviction_calls = []
        self.accept = accept

    def configure_phase_policy(
        self,
        enabled,
        prefill_admission,
        decode_admission,
        prefill_weight,
        decode_weight,
        starvation_limit,
    ) -> None:
        self.calls.append(
            (
                enabled,
                prefill_admission,
                decode_admission,
                prefill_weight,
                decode_weight,
                starvation_limit,
            )
        )

    def configure_eviction_policy(self, name, **kwargs) -> bool:
        self.eviction_calls.append((name, kwargs))
        return self.accept


class _RecordingDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def configure_phase_policy(self, *args) -> None:
        self.calls.append(args)


def _bare_engine(archer_config, model_config=None) -> OffloadEngine:
    engine = object.__new__(OffloadEngine)
    engine.archer_config = archer_config
    engine.archer_engine = _RecordingHandle()
    engine.expert_dispatcher = _RecordingDispatcher()
    engine.expert_prefetcher = SimpleNamespace(phase_policy=None)
    engine.config = model_config or SimpleNamespace(num_experts_per_tok=8)
    engine.num_experts = 128
    return engine


def _config(enabled, monkeypatch, **kwargs) -> ArcherConfig:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    return ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
        phase_specific_expert_policy=enabled,
        **kwargs,
    )


def test_enabled_configures_handle_not_dispatcher(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch))
    engine._configure_native_phase_policy()

    # Admission modes reach the binding as native AdmissionMode values.
    assert engine.archer_engine.calls == [(True, 1, 0, 1.0, 4.0, 8)]
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is True


def test_disabled_assigns_settings_but_does_not_configure(monkeypatch) -> None:
    engine = _bare_engine(_config(False, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == []
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is False


def test_legacy_eviction_policy_leaves_native_engine_alone(monkeypatch) -> None:
    engine = _bare_engine(_config(False, monkeypatch))
    engine._configure_native_eviction_policy()

    assert engine.archer_engine.eviction_calls == []


def test_eviction_policy_configures_handle(monkeypatch) -> None:
    engine = _bare_engine(
        _config(
            False,
            monkeypatch,
            expert_eviction_policy="lcnu",
            expert_eviction_lcnu_alpha=0.2,
        )
    )
    engine._configure_native_eviction_policy()

    assert engine.archer_engine.eviction_calls == [
        (
            "lcnu",
            {
                "num_layers": 0,
                "lcnu_alpha": 0.2,
                "lcnu_beta": 0.5,
                "lcnu_prior": 8 / 128,
            },
        )
    ]
    assert engine.archer_engine.calls == []


def test_eviction_policy_without_top_k_keeps_native_prior(monkeypatch) -> None:
    engine = _bare_engine(
        _config(False, monkeypatch, expert_eviction_policy="lru"),
        model_config=SimpleNamespace(),
    )
    engine._configure_native_eviction_policy()

    name, kwargs = engine.archer_engine.eviction_calls[0]
    assert name == "lru"
    assert "lcnu_prior" not in kwargs


def test_eviction_policy_is_skipped_for_gptq(monkeypatch) -> None:
    engine = _bare_engine(
        _config(False, monkeypatch, expert_eviction_policy="lcnu"),
        model_config=SimpleNamespace(
            num_experts_per_tok=8,
            quantization_config={"quant_method": "gptq"},
        ),
    )
    with pytest.warns(UserWarning, match="ignored for GPTQ"):
        engine._configure_native_eviction_policy()

    assert engine.archer_engine.eviction_calls == []


def test_rejected_eviction_policy_raises(monkeypatch) -> None:
    engine = _bare_engine(
        _config(False, monkeypatch, expert_eviction_policy="lcnu")
    )
    engine.archer_engine.accept = False

    with pytest.raises(RuntimeError, match="rejected"):
        engine._configure_native_eviction_policy()


def test_eviction_policy_needs_native_binding(monkeypatch) -> None:
    engine = _bare_engine(
        _config(False, monkeypatch, expert_eviction_policy="lcnu")
    )
    engine.archer_engine = SimpleNamespace()

    with pytest.raises(RuntimeError, match="configure_eviction_policy"):
        engine._configure_native_eviction_policy()

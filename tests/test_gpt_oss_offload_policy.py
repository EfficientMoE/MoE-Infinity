from types import SimpleNamespace
from unittest.mock import Mock

from moe_infinity.runtime.model_offload import (
    _gpt_oss_offload_enabled,
    _wire_expert_collaborators,
)


def _engine_config(ratio):
    return SimpleNamespace(device_memory_ratio=ratio)


def test_gpt_oss_default_ratio_keeps_resident_fallback():
    assert not _gpt_oss_offload_enabled("gpt_oss", _engine_config(0.9))


def test_gpt_oss_low_ratio_enables_dispatcher():
    assert _gpt_oss_offload_enabled("gpt_oss", _engine_config(0.5))


def test_policy_does_not_disable_other_architectures():
    assert _gpt_oss_offload_enabled("mixtral", _engine_config(0.9))


def test_low_ratio_wires_all_gpt_oss_collaborators():
    module = SimpleNamespace()
    collaborators = {
        "expert_executor": Mock(),
        "expert_prefetcher": Mock(),
        "expert_tracer": Mock(),
        "expert_predictor": Mock(),
        "expert_tensor_map": {(0, 0): 7},
    }

    _wire_expert_collaborators(module, True, **collaborators)

    for name, value in collaborators.items():
        assert getattr(module, name) is value


def test_resident_mode_leaves_gpt_oss_collaborators_unset():
    module = SimpleNamespace(
        expert_executor=None,
        expert_prefetcher=None,
        expert_tracer=None,
        expert_predictor=None,
        expert_tensor_map=None,
    )
    _wire_expert_collaborators(
        module,
        False,
        expert_executor=Mock(),
        expert_prefetcher=Mock(),
        expert_tracer=Mock(),
        expert_predictor=Mock(),
        expert_tensor_map={(0, 0): 7},
    )

    assert module.expert_executor is None
    assert module.expert_prefetcher is None
    assert module.expert_tracer is None
    assert module.expert_predictor is None
    assert module.expert_tensor_map is None

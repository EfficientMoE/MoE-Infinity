from types import SimpleNamespace

from moe_infinity.runtime.model_offload import _gpt_oss_offload_enabled


def _engine_config(ratio):
    return SimpleNamespace(device_memory_ratio=ratio)


def test_gpt_oss_default_ratio_keeps_resident_fallback():
    assert not _gpt_oss_offload_enabled("gpt_oss", _engine_config(0.9))


def test_gpt_oss_low_ratio_enables_dispatcher():
    assert _gpt_oss_offload_enabled("gpt_oss", _engine_config(0.5))


def test_policy_does_not_disable_other_architectures():
    assert _gpt_oss_offload_enabled("mixtral", _engine_config(0.9))

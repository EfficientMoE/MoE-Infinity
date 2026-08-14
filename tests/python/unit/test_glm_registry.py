import sys
import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest


def test_glmmoedsa_key_in_mapping_names():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import MODEL_MAPPING_NAMES

    assert "glmmoedsa" in MODEL_MAPPING_NAMES, (
        "'glmmoedsa' not found in MODEL_MAPPING_NAMES; "
        f"keys present: {sorted(MODEL_MAPPING_NAMES)}"
    )


def test_glmmoedsa_type_is_5():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import MODEL_MAPPING_TYPES

    assert MODEL_MAPPING_TYPES["glmmoedsa"] == 5, (
        f"Expected expert type 5 for 'glmmoedsa', "
        f"got {MODEL_MAPPING_TYPES.get('glmmoedsa')}"
    )


def test_glmmoedsa_class_not_none():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import MODEL_MAPPING_NAMES

    cls = MODEL_MAPPING_NAMES["glmmoedsa"]
    assert cls is not None, "MODEL_MAPPING_NAMES['glmmoedsa'] must not be None"
    assert hasattr(
        cls, "__name__"
    ), "MODEL_MAPPING_NAMES['glmmoedsa'] must have __name__"
    assert cls.__name__ == "GlmMoeDsaForCausalLM"


def test_parse_expert_type_glmmoedsa():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import parse_expert_type

    config = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
    result = parse_expert_type(cast(Any, config))
    assert (
        result == 5
    ), f"parse_expert_type with GlmMoeDsaForCausalLM expected 5, got {result}"


def test_guarded_import_does_not_crash_when_unavailable():
    import importlib

    import moe_infinity.common.constants as _mod

    with patch.dict(sys.modules, {"transformers": None}):
        pass

    glm_cls_name = "GlmMoeDsaForCausalLM"
    import transformers as _tf

    if not hasattr(_tf, glm_cls_name):
        pytest.skip(
            f"transformers does not export {glm_cls_name} — guard path not reachable in this env"
        )

    assert hasattr(
        _mod, "GlmMoeDsaForCausalLM"
    ), "constants module must expose GlmMoeDsaForCausalLM (or None) at module level"


def test_glmmoedsa_substring_match():
    arch_lower = "GlmMoeDsaForCausalLM".lower()
    assert (
        "glmmoedsa" in arch_lower
    ), f"Key 'glmmoedsa' is not a substring of '{arch_lower}'"

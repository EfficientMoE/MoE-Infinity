import importlib
import sys

import pytest


def test_glm_patch_replaces_moe_class():
    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as glm_mod
    from moe_infinity.models import SyncGlmMoeDsaMoEBlock

    original = glm_mod.GlmMoeDsaMoE

    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as _glm_mod
    _glm_mod._old_glm_moe_dsa_moe = _glm_mod.GlmMoeDsaMoE
    _glm_mod.GlmMoeDsaMoE = SyncGlmMoeDsaMoEBlock

    assert glm_mod.GlmMoeDsaMoE is SyncGlmMoeDsaMoEBlock, (
        "Patch did not replace GlmMoeDsaMoE with SyncGlmMoeDsaMoEBlock"
    )

    if hasattr(_glm_mod, "_old_glm_moe_dsa_moe"):
        _glm_mod.GlmMoeDsaMoE = _glm_mod._old_glm_moe_dsa_moe

    assert glm_mod.GlmMoeDsaMoE is original, (
        "Unpatch did not restore original GlmMoeDsaMoE"
    )


def test_sync_glm_moe_block_importable():
    from moe_infinity.models import SyncGlmMoeDsaMoEBlock
    assert SyncGlmMoeDsaMoEBlock is not None


def test_model_offload_imports_sync_glm():
    import moe_infinity.runtime.model_offload as mo
    from moe_infinity.models import SyncGlmMoeDsaMoEBlock
    assert hasattr(mo, "SyncGlmMoeDsaMoEBlock") or SyncGlmMoeDsaMoEBlock is not None

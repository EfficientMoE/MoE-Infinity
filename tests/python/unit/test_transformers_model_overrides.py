import importlib
import importlib.util

from moe_store.wrappers import SyncGlmMoeDsaMoEBlock


def test_glm_override_install_and_restore_preserve_original_class():
    module_name = "moe_infinity.boundary.transformers_model_overrides"
    assert importlib.util.find_spec(module_name) is not None
    overrides = importlib.import_module(module_name)
    glm_module = importlib.import_module(
        "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa"
    )
    original = glm_module.GlmMoeDsaMoE
    installed = False

    try:
        overrides.install_transformers_model_overrides()
        installed = True
        assert glm_module.GlmMoeDsaMoE is SyncGlmMoeDsaMoEBlock

        overrides.restore_context_transformers_model_overrides()
        assert glm_module.GlmMoeDsaMoE is original
    finally:
        if installed:
            overrides.restore_context_transformers_model_overrides()
            overrides.restore_runtime_transformers_model_overrides()
        glm_module.GlmMoeDsaMoE = original

"""TDD tests for GptOssMLP monkey-patching in OffloadEngine."""

import importlib.machinery
import sys


def _ensure_flash_attn_stub_has_spec():
    flash_attn_module = sys.modules.get("flash_attn")
    if (
        flash_attn_module is not None
        and getattr(flash_attn_module, "__spec__", None) is None
    ):
        flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn", loader=None
        )


def test_gpt_oss_mlp_original_exists():
    """Original GptOssMLP must be accessible in transformers."""
    _ensure_flash_attn_stub_has_spec()
    import transformers.models.gpt_oss.modeling_gpt_oss as mod

    assert hasattr(mod, "GptOssMLP"), "GptOssMLP must exist in transformers"
    original = mod.GptOssMLP
    assert original.__name__ == "GptOssMLP"


def test_sync_gpt_oss_mlp_is_different_from_original():
    """SyncGptOssMLP must be a different class from GptOssMLP."""
    _ensure_flash_attn_stub_has_spec()
    import transformers.models.gpt_oss.modeling_gpt_oss as mod

    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    assert mod.GptOssMLP is not SyncGptOssMLP, (
        "SyncGptOssMLP must be different from GptOssMLP (before patching)"
    )


def test_monkey_patch_mechanism():
    """Verify the monkey-patch mechanism works correctly (save/replace/restore)."""
    _ensure_flash_attn_stub_has_spec()
    import transformers.models.gpt_oss.modeling_gpt_oss as mod

    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    original = mod.GptOssMLP

    # Simulate patch
    setattr(mod, "_old_gpt_oss_mlp", mod.GptOssMLP)
    mod.GptOssMLP = SyncGptOssMLP

    try:
        assert mod.GptOssMLP is SyncGptOssMLP, (
            "After patch: GptOssMLP should be SyncGptOssMLP"
        )
        assert getattr(mod, "_old_gpt_oss_mlp") is original, (
            "Saved original must match"
        )
    finally:
        # Simulate restore
        mod.GptOssMLP = getattr(mod, "_old_gpt_oss_mlp")
        delattr(mod, "_old_gpt_oss_mlp")

    # After restore
    assert mod.GptOssMLP is original, (
        "After restore: GptOssMLP should be original"
    )
    assert not hasattr(mod, "_old_gpt_oss_mlp"), (
        "Backup should be deleted after restore"
    )

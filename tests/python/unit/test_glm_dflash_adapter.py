import warnings

from moe_infinity.spec_decode.glm_dflash import (
    glm_dflash_available,
    glm_dflash_drafter_for,
)


def test_no_glm_drafter_returns_none():
    assert glm_dflash_drafter_for("zai-org/GLM-5.2-FP8") is None


def test_availability_false_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert glm_dflash_available("zai-org/GLM-5.2-FP8") is False
    assert any("MTP" in str(x.message) for x in w)


def test_registered_drafter_available(monkeypatch):
    import moe_infinity.spec_decode.glm_dflash as mod

    monkeypatch.setitem(
        mod._GLM_DFLASH_DRAFTERS, "zai-org/GLM-9", "z-lab/GLM-9-DFlash"
    )
    assert mod.glm_dflash_drafter_for("zai-org/GLM-9") == "z-lab/GLM-9-DFlash"
    assert mod.glm_dflash_available("zai-org/GLM-9") is True

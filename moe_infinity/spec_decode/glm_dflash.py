from __future__ import annotations

import warnings
from typing import Any, Optional

from moe_infinity.spec_decode.dflash import read_dflash_config, validate_pairing

_GLM_DFLASH_DRAFTERS = {
    # populated when z-lab ships GLM DFlash drafters; e.g.:
    # "zai-org/GLM-5.1": "z-lab/GLM-5.1-DFlash",
    # "zai-org/GLM-5.2": "z-lab/GLM-5.2-DFlash",
}


def glm_dflash_drafter_for(target_model: str) -> Optional[str]:
    return _GLM_DFLASH_DRAFTERS.get(target_model)


def glm_dflash_available(target_model: str) -> bool:
    drafter = glm_dflash_drafter_for(target_model)
    if drafter is None:
        warnings.warn(
            f"No z-lab GLM DFlash drafter registered for {target_model}; "
            "use GLM's built-in MTP speculative decoding instead (T19).",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def validate_glm_pairing(draft_hf_config: Any, target_hf_config: Any) -> None:
    draft_cfg = read_dflash_config(draft_hf_config)
    validate_pairing(draft_cfg, target_hf_config)

from transformers import (
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OPTForCausalLM,
    PretrainedConfig,
    Qwen3MoeForCausalLM,
)

from ..models.modeling_arctic import (
    ArcticForCausalLM,
)  # TODO: Replace this with huggingface transformers
from ..models.modeling_deepseek_v2 import DeepseekV2ForCausalLM
from ..models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ..models.modeling_grok.modeling_grok1 import (
    Grok1ModelForCausalLM,
)  # TODO: Replace this with huggingface transformers

# Models requiring trust_remote_code class resolution (class is None here;
# resolved dynamically in big_modeling.py via auto_map).
MODEL_TRUST_REMOTE_CODE = {"minimax_m2", "kimi_vl"}

MODEL_MAPPING_NAMES = {
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "grok": Grok1ModelForCausalLM,
    "arctic": ArcticForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "qwen3": Qwen3MoeForCausalLM,
    # trust_remote_code models: class resolved dynamically in big_modeling.py
    "minimax_m2": None,
    "kimi_vl": None,
}

MODEL_MAPPING_TYPES = {
    "nllb": 2,
    "mixtral": 4,
    "grok": 4,
    "arctic": 4,
    "deepseek": 5,
    "deepseek_v3": 5,
    "qwen3": 5,
    "minimax_m2": 5,
    "kimi_vl": 5,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = config.architectures[0].lower()
    model_type = getattr(config, "model_type", "").lower()
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture or supp_arch == model_type:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    return MODEL_MAPPING_TYPES[arch]

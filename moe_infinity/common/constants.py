from transformers import (
    DbrxForCausalLM,
    DeepseekV2ForCausalLM,
    DeepseekV3ForCausalLM,
    GptOssForCausalLM,
    JambaForCausalLM,
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OlmoeForCausalLM,
    OPTForCausalLM,
    PretrainedConfig,
    Qwen3MoeForCausalLM,
)

from ..models.modeling_arctic import (
    ArcticForCausalLM,
)  # TODO: Replace this with huggingface transformers
from ..models.modeling_grok.modeling_grok1 import (
    Grok1ModelForCausalLM,
)  # TODO: Replace this with huggingface transformers

MODEL_MAPPING_NAMES = {
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "grok": Grok1ModelForCausalLM,
    "arctic": ArcticForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "gptoss": GptOssForCausalLM,
    "qwen3": Qwen3MoeForCausalLM,
    "dbrx": DbrxForCausalLM,
    "olmoe": OlmoeForCausalLM,
    "jamba": JambaForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "nllb": 2,
    "mixtral": 4,
    "opt": 3,
    "grok": 4,
    "arctic": 4,
    "deepseek": 5,
    "deepseek_v3": 5,
    "gptoss": 4,
    "qwen3": 5,
    "dbrx": 4,
    "olmoe": 4,
    "jamba": 4,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = (
        config.architectures[0].lower() if config.architectures else ""
    )
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    if arch in ("grok", "arctic"):
        import warnings

        warnings.warn(
            f"'{arch}' model support is deprecated and will be removed in a future version. "
            "This model architecture is not available in HuggingFace transformers.",
            DeprecationWarning,
            stacklevel=2,
        )

    return MODEL_MAPPING_TYPES[arch]

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

try:
    from transformers import DeepseekV4ForCausalLM
except ImportError:
    DeepseekV4ForCausalLM = None

MODEL_MAPPING_NAMES = {
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "deepseekv4": DeepseekV4ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
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
    "deepseekv4": 5,
    "deepseek_v3": 5,
    "deepseek": 5,
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

    return MODEL_MAPPING_TYPES[arch]

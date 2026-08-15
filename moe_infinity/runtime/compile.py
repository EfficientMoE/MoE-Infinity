import os

import torch
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from moe_infinity.models.mixtral import MixtralExpertMLP

EXPERT_CLS = {
    "deepseek_v2": DeepseekV2MLP,
    "deepseek_v3": DeepseekV3MLP,
    "mixtral": MixtralExpertMLP,
    "qwen3_moe": Qwen3MoeMLP,
}


# compile a single expert
def script_expert(save_dir, expert_type, config, **kwargs):
    """
    Compile a single expert.
    """
    expert_instance = EXPERT_CLS[expert_type](config, **kwargs)
    # compile the forward function of the expert
    module = torch.jit.script(expert_instance)
    torch.jit.save(
        module,
        os.path.join(save_dir, "expert.pt"),
    )

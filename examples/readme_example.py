import argparse
import os

from transformers import AutoTokenizer

from moe_infinity import MoE

SUPPORTED_MODELS = [
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "Qwen/Qwen3-30B-A3B",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-VL-A3B-Thinking",
]

parser = argparse.ArgumentParser(
    description="MoE-Infinity minimal inference example"
)
parser.add_argument(
    "--checkpoint",
    default="deepseek-ai/DeepSeek-V2-Lite-Chat",
    help="HuggingFace model checkpoint (default: deepseek-ai/DeepSeek-V2-Lite-Chat)",
)
parser.add_argument(
    "--offload_dir",
    default=os.path.join(os.path.expanduser("~"), "moe-infinity"),
    help="Directory for offloading expert weights",
)
parser.add_argument(
    "--prompt",
    default="What is the capital of France?",
    help="Input prompt for the model",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.checkpoint, trust_remote_code=True
)

config = {
    "offload_path": os.path.join(
        args.offload_dir, args.checkpoint.split("/")[-1]
    ),
    "device_memory_ratio": 0.75,  # lower on OOM
}

model = MoE(args.checkpoint, config)

# Apply chat template if the tokenizer supports it; otherwise use raw prompt.
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
else:
    input_text = args.prompt

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
}

output_ids = model.generate(input_ids, **generate_kwargs)
# Decode only the newly generated tokens (skip the prompt).
new_tokens = output_ids[0][input_ids.shape[1]:]
output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(output_text)

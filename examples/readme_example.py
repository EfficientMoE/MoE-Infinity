import os

from transformers import AutoTokenizer

from moe_infinity import MoE

user_home = os.path.expanduser("~")

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote=True)

config = {
    "offload_path": os.path.join(user_home, "moe-infinity"),
    "device_memory_ratio": 0.75,  # 75% of the device memory is used for caching, change the value according to your device memory size on OOM
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)

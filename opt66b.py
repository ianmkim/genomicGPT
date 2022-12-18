#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
model_name = 'facebook/opt-66b'
text = """
Q: On average, Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. How many punches did he throw?\n
A: Let's think step by step:\n
"""

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
input_ids = tokenizer(text, return_tensors="pt").input_ids


free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f"{free_in_GB-2}GB"

n_gpus = torch.cuda.device_count()

max_memory = {i: max_memory for i in range(n_gpus)}

print(max_memory)

print("Loading models")
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="offload",
            load_in_8bit=True)

print(model.hf_device_map)
exit()

print("Generating new ids")
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

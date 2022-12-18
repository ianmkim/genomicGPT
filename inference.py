#!/usr/bin/env python3

import torch
from os import listdir
from os.path import isfile, join

from transformers import (
    AutoConfig,
    AutoTokenizer,
    OPTForCausalLM,
    pipeline,
)

MODEL_NAME = "facebook/opt-2.7b"
CHECKPOINT_DIRECTORY = "genomicGPT" 

def choose_latest_checkpoint(dirname:str) -> str:
    checkpoints = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ".pth" in f]
    checkpoints = sorted(checkpoints)
    return join(dirname, checkpoints[-1])
    

if __name__ == "__main__":
    model_path = choose_latest_checkpoint("genomicGPT")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = OPTForCausalLM.from_pretrained(model_path,
                                    from_tf=bool(".ckpt" in model_path),
                                    config=config,
                                    local_files_only=False)

    generator = pipeline(
              'text-generation', 
              model=model, 
              tokenizer=tokenizer,
              config=config,
              use_fast=False,
              do_sample=True,
            )

    prompt = """
Sequence: msvptmawmmlllgllaygsgvdsqtvvtqepsfsvspggtvtltcglssgsvstsyypswyqqtpgqaprtliystntrssgvpdrfsgsilgnkaaltitgaqaddesdyycvlymgsgi
Function:"""

    print(prompt)

    gen = generator(
        prompt,
        max_length=96,
    ) 
    
    print(gen)

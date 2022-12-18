# Genomic GPT
An experiment that finetunes a pretrained LLM (OPT-2.7B and GPT-175B) based on protein sequence -> protein function pairing data scraped from the UniProt database level 4 and 5 confidence level annotations.

Uses colossal AI to optimize for local finetuning of the OPT model on dual 4090s.

## Prerequisites
You must install ColossalAI and build from source according to instructions [here](https://github.com/hpcaitech/ColossalAI#Installation)

You should also have at least 40GB+ of VRAM for finetuning the OPT model if you wish to do it locally. Note that while ColossalAI provides orders of magnitude speedups in certain finetuning situations, because of its design, it cannot offload weights to RAM. Therefore, all of the model must fit into your GPU

## Usage
```
# make dataset will output a jsonl file in the correct format to data directory
# the GPT flag determines whether to preprocess for GPT-3 or OPT
./make_dataset.py -f <path to raw tsv> -o <path to output> --gpt

# run the training script
./run_clm.sh
```

The original checkpoint was trained on 1.5+ million protein sequence -> protein function annotation data scraped from uniprot. This repository provides only a 200 row small version of the dataset for testing purposes in the `rawdata` directory



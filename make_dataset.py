#!/usr/bin/env python3

import re
import pandas as pd 
from pathlib import Path
import argparse

from tqdm import tqdm
from typing import List

def load_file_for_oai(filename:str) -> pd.DataFrame:
    """
    Load a tsv file and prepares it for OpenAI GPT-3 finetuning API
    """
    data = pd.read_csv(filename, delimiter="\t")
    only_func_seq = data[["Function [CC]", "Sequence"]]

    batches = []
    explanations = {}

    for idx, row in tqdm(list(only_func_seq.iterrows())): 
        clean_func = re.sub(r"{.*?}", "", str(row["Function [CC]"]))
        funcs_for_seq = clean_func.replace(" .", "").split("FUNCTION: ")[1:]

        for func in funcs_for_seq:
            completed_text = " " + func.lower()
            if completed_text not in explanations:
                prompt_text =  f"{str(row['Sequence'])}".lower() 
                explanations[completed_text] = True

                batches.append({
                    "prompt": prompt_text,
                    "completion": completed_text,
                })

                break

    df = pd.DataFrame(batches)
    return df


def load_file(filename:str) -> pd.DataFrame:
    """
    Loads a tsv file and prepares it for OPT-2.7B finetuning
    """
    data = pd.read_csv(filename, delimiter="\t")
    only_func_seq = data[["Function [CC]", "Sequence"]]

    batches = []
    explanations = {} 

    for idx, row in tqdm(list(only_func_seq.iterrows())): 
        clean_func = re.sub(r"{.*?}", "", str(row["Function [CC]"]))
        funcs_for_seq = clean_func.replace(" .", "").split("FUNCTION: ")[1:]

        for func in funcs_for_seq:
            if func not in explanations:
                explanations[func] = True

                prep_text =  f"Sequence: {str(row['Sequence'])}\n" + "Function: " + func + "<EOS>"
                batches.append({
                    "text": prep_text,
                })

                break

    df = pd.DataFrame(batches)
    return df
            

def make_dataset(filenames:List[str], gpt3:bool=False):
    if len(filenames) <= 0:
        return

    if gpt3:
        final_df = load_file_for_oai(filenames[0])
    else:
        final_df = load_file(filenames[0])
    for filename in filenames[1:]:
        final_df = final_df.append(load_file(filename))

    return final_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", 
            help="path to training data tsv file", 
            type=str, 
            required=True)
    parser.add_argument("-o", "--output",
            help="path to where to saved the processed output file",
            type=str,
            required=True)
    parser.add_argument("-g", "--gpt",
            help="toggles whether to prepare data for GPT-3 as opposed to OPT",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False)

    args = parser.parse_args()

    final_data = make_dataset([
        args.filename,
        #"data/data2.tsv", # this is short
        #"data/data1.tsv" # this is long
    ], gpt3=args.gpt)

    print("Writing final data to JSON file")
    final_data.to_json(args.output, orient="records", lines=True)

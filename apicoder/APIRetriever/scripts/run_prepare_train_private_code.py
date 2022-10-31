import json
import os
import glob
from argparse import ArgumentParser

from transformers import AutoTokenizer
from tqdm import tqdm

import sys
# -------------------------------------------------------------------------------------------------
# you need to change this path to your own `APICoder-CodeGenAPI` path, better to use absolute path
# -------------------------------------------------------------------------------------------------
sys.path.append('../../../CodeGenAPI/')
from scripts.get_libs_info_from_code import (
    get_dict_of_api_name_lib_api_paths,
    get_dict_of_api_path_api_signature_and_api_desp,
    get_first_sentence_from_api_desp,
    normalizer_api_desp
)

# -------------------------------------------------------------------------------------------------
# your need to change this path to the path of your `crawl_code` path, better to use absolute path
# -------------------------------------------------------------------------------------------------
YOUR_CRAWLED_API_PATH = "PrivateLibrary/data/API-Doc"
api_path_api_signature_and_api_desp = get_dict_of_api_path_api_signature_and_api_desp(
    YOUR_CRAWLED_API_PATH, 
    "pandas,numpy,monkey,beatnum,torchdata", 
    "datetime", 
    "False"
)

# -------------------------------------------------------------------------------------------------
# your need to change the below path to the your own ones, better to use absolute path
# -------------------------------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument('--input', type=str, default="PrivateLibrary/APIRetriever/data/train/unprocessed-train-data", help="each jsonl file in such path contains many json lines, where each line's format is {'code_doc': '', 'positive_APIs': ['A', ...], 'negative_APIs': ['B', ...]}")
parser.add_argument('--data_mode', type=str, default="", help="the prefix of the input jsonl file, default is empty")
parser.add_argument('--output', type=str, default="PrivateLibrary/APIRetriever/data/train/processed-train-data", help="the output path")
parser.add_argument('--tokenizer', type=str, required=False, default='/your/path/of/bert-base-uncased')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

if not os.path.exists(args.output):
    os.makedirs(args.output)

all_data_paths = glob.glob(os.path.join(args.input, f'{args.data_mode}*.jsonl'))
print(f"Now, all data paths are: {all_data_paths}")

# -------------------------------------------------------------------------------------------------
# your training data name default is `private_data_train.json`, you can change it to your own name
# -------------------------------------------------------------------------------------------------
with open(os.path.join(args.output, 'private_data_train.json'), 'w+') as f:
    for data_path in tqdm(all_data_paths):
        data_reader = open(data_path, 'r')
        for line in tqdm(data_reader):
            group = {}
            # dict_keys(['code_block', 'code_doc', 'code_all_doc', 'positive_APIs', 'negative_APIs'])
            line_dict = json.loads(line)
            comment, positive_apis, negative_apis = line_dict["code_doc"], line_dict["positive_APIs"], line_dict["negative_APIs"]
            query = tokenizer.encode(comment, add_special_tokens=False, max_length=256, truncation=True)
            
            group['query'] = query
            group['positives'] = []
            group['negatives'] = []
            for positive_api in positive_apis:
                if api_path_api_signature_and_api_desp.get(positive_api) is None: 
                    continue
                positive_api_info_dict = api_path_api_signature_and_api_desp[positive_api]
                if positive_api_info_dict['api_signature'] == "": 
                    continue
                positive_api_prompt = f"{positive_api_info_dict['api_name']}{positive_api_info_dict['api_signature']}: {get_first_sentence_from_api_desp(normalizer_api_desp(positive_api_info_dict['api_description']))}"
                text = tokenizer.encode(positive_api_prompt, add_special_tokens=False, max_length=256, truncation=True)
                group['positives'].append(text)
            for negative_api in negative_apis:
                if api_path_api_signature_and_api_desp.get(negative_api) is None: 
                    continue
                negative_api_info_dict = api_path_api_signature_and_api_desp[negative_api]
                if negative_api_info_dict['api_signature'] == "": 
                    continue
                negative_api_prompt = f"{negative_api_info_dict['api_name']}{negative_api_info_dict['api_signature']}: {get_first_sentence_from_api_desp(normalizer_api_desp(negative_api_info_dict['api_description']))}"
                text = tokenizer.encode(negative_api_prompt, add_special_tokens=False, max_length=256, truncation=True)
                group['negatives'].append(text)
            if len(group['positives']) == 0 or len(group['negatives']) == 0 or len(group['query']) == 0:
                print("Skip this group")
                continue
            f.write(json.dumps(group) + '\n')

print(f"Done!")
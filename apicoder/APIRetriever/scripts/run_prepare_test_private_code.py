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
sys.path.append('../../../APICoder-CodeGenAPI/')
from scripts.get_libs_info_from_code import (
    get_dict_of_api_name_lib_api_paths,
    get_dict_of_api_path_api_signature_and_api_desp,
    get_first_sentence_from_api_desp,
    normalizer_api_desp
)
from APICoder.get_lib_comment_for_eval import (
    get_code_and_comment_by_lib_name_and_task_id,
    get_one_instance_by_lib_name
)
from APICoder.get_api_info_by_name import (
    get_api_name_4_api_sign_and_desps,
    get_all_api_info_prompt_list_by_api_name
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

parser = ArgumentParser()

parser.add_argument('--base_model_dir', type=str, default="/your/base/dir/including/`eval_datas(benchmarks)`/and/others/")
parser.add_argument('--benchmarks', type=list, default=["pandas", "numpy", "monkey", "beatnum", "torchdata"])
parser.add_argument('--output_dir', type=str, default="PrivateLibrary/APIRetriever/data/inference")
parser.add_argument('--tokenizer', type=str, required=False, default='your/path/of/bert-base-uncased/')

args = parser.parse_args()

base_model_dir, benchmarks, output_dir = args.base_model_dir, args.benchmarks, args.output_dir
benchmark_dir = os.path.join(base_model_dir, "eval_datas")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

x = [0, 1000, 10000, 100000, 1000000] # the unique id of each code comment

'''
for `code comment` of 5 benchamarks, making query corpus (total 5 files, named: xxx_comment.json)
'''
for idx, library_name in enumerate(tqdm(benchmarks)):
    if not os.path.exists(os.path.join(output_dir, library_name)):
        os.makedirs(os.path.join(output_dir, library_name))
    comment_out_name = os.path.join(output_dir, library_name + "_comment.json")
    comment_writer = open(comment_out_name, 'w+')
    base_id = x[idx]

    lib_iter_obj = get_one_instance_by_lib_name(library_name, base_dir=base_model_dir)
    for this_instance_dict in tqdm(lib_iter_obj):
        # dict_keys(['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test'])
        task_id = this_instance_dict["task_id"]
        text_id = base_id + int(task_id.split("/")[-1])
        code_comment_solution = get_code_and_comment_by_lib_name_and_task_id(library_name, task_id, base_model_dir)
        this_code, this_comment, this_solution = code_comment_solution[0], code_comment_solution[1], code_comment_solution[2]
        save_dict = {
            "text_id": text_id,
            "task_id": task_id,
            "text": this_comment
        }
        comment_writer.write(json.dumps(save_dict) + "\n")

    comment_writer.close()

'''
for `API information` of 5 benchamarks, making corpus (total 5 files, named: xxx_api.json)
'''
y = [1000000, 1100000, 1200000, 1300000, 1400000] # the unique id of each API
for idx, library_name in enumerate(tqdm(benchmarks)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    api_out_name = os.path.join(output_dir, library_name + "_api.json")
    api_writer = open(api_out_name, 'w+')
    base_id = y[idx]
    api_name_4_api_sign_and_desps = get_api_name_4_api_sign_and_desps(library_name, base_model_dir)
    total_api, now_number = len(api_name_4_api_sign_and_desps), 0
    for api_name, api_path_info_dict in tqdm(api_name_4_api_sign_and_desps.items(), total=total_api):
        for api_idx, (api_path, api_info_list) in enumerate(api_path_info_dict.items()):
            api_signature, api_description = api_info_list[0].strip(), get_first_sentence_from_api_desp(normalizer_api_desp(api_info_list[1]))
            if api_signature == "":
                continue
            api_info_prompt=f"{api_name}{api_signature}: {api_description}"
            text_id = base_id + now_number
            save_dict = {
                "text_id": text_id,
                "text": api_info_prompt
            }
            now_number+=1
            api_writer.write(json.dumps(save_dict) + "\n")

    api_writer.close()

print(f"Done!")
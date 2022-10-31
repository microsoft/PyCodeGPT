import torch
import os
import time
import json
import traceback
import numpy as np
from typing import Dict, List
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
import multiprocessing
import multiprocessing.dummy
import math
import docformatter
import autopep8
from func_timeout import func_timeout, FunctionTimedOut

import sys
# ------------------------------------------------------------------------------------------------
sys.path.append("your/absolute/path/of/.../APICoder-CodeGenAPI/scripts") # must be absolute path
# ------------------------------------------------------------------------------------------------
from file_utils import get_files, read_lines, load_githup_stars_dict, Data_Abs_Dir
from multiprocessing_utils import MultiprocessingArgs, run_with_multiprocessing
from ast_utils import *
from codegen_tokenization import create_custom_gpt2_tokenizer, create_model
from get_libs_info_from_code import * # contain `craft_code_context_and_doc_pairs` function

# ------------------------------------------------------------------------------------------------
sys.path.append("your/absolute/path/of/.../APICoder-CodeGenAPI") # must be absolute path
# ------------------------------------------------------------------------------------------------
from nl2code import load_pretrained_tokenizer, resolve_model_name_or_path

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def encode_pycode(
    input_files: List[str], 
    output_prefix: str, 
    domain: str, 
    lib_name_api_name_api_paths: Dict[str, Dict[str, str]],
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]],
    tokenizer: PreTrainedTokenizer, 
    logger: logging.Logger, 
    logging_freq: int=10000, 
    rank: int=0):

    logger.info("Process {} start to process {} files:\n{}".format(str(rank), len(input_files), "\n".join(input_files)))
    results = defaultdict(int)

    try:
        # ------------------------------------------------------------------------------------------------
        # the name of saved jsonl file, your can change it
        # ------------------------------------------------------------------------------------------------
        output_path = output_prefix + "_comment_api.jsonl"
        retrievel_writer = open(output_path, "w+")

        for line in read_lines(input_files):
            try:
                results["cnt_processed"] += 1
                example = json.loads(line)
                norm_text = example['content']

                max_time_out = 5
                try:
                    norm_text = func_timeout(max_time_out, autopep8.fix_code, args=(norm_text,))
                except FunctionTimedOut:
                    results[f"time_out_exception_{max_time_out}s"] += 1
                    continue
                except Exception as e:
                    results["auto_pep_execution_error"] += 1
                    continue

                text_list = norm_text.split("\n\n\n")
                try:
                    api_code_pairs_list = func_timeout(
                        max_time_out, 
                        craft_code_context_and_doc_pairs, 
                        args=(
                            norm_text,
                            text_list,
                            "",
                            lib_name_api_name_api_paths,
                            api_path_api_signature_and_api_desp,
                            "only function"
                        )
                    )
                except FunctionTimedOut:
                    results[f"code_doc_time_out_exception_{max_time_out}s"] += 1
                    continue
                except Exception as e:
                    logger.info("[Warning] Process {} crafe code doc execution: ".format(rank) + traceback.format_exc())
                    results["code_doc_execution_error"] += 1
                    continue
                
                try:
                    if len(api_code_pairs_list) == 0:
                        results["no_api_code_pairs"] += 1
                        continue
                except Exception as e:
                    logger.info("[Error] Process {} no api code pairs error: ".format(rank) + traceback.format_exc())

                try:
                    for api_code_pair in api_code_pairs_list:
                        retrievel_writer.write(json.dumps(api_code_pair) + "\n")
                except Exception as e:
                    logger.info("[Error] Process {} write api error: ".format(rank) + traceback.format_exc())
                    results["write_api_code_pairs_error"] += 1
                
                results["cnt_saved"] += 1

                if results["cnt_processed"] % logging_freq == 0:
                    logger.info("Process {} processed {} lines, saved = {}, {}.".format(
                        rank, results["cnt_processed"], results["cnt_saved"], ", ".join([f"{k} = {v}" for k, v in results.items()])))

            except Exception as e:
                results["failed"] += 1
                logger.info("[error] Process {}: ".format(rank) + traceback.format_exc())
    
        retrievel_writer.close()
        logger.info("Process {} start to finalize dataset ...".format(rank))
        logger.info("Process {} processed all over, {}".format(
            rank, ", ".join([f"{k} = {v}" for k, v in results.items()]))
        )

        results['num_processes'] = 1
        
    except Exception as e:
        results["failed"] += 1
        logger.info("[error] Process {}: ".format(rank) + traceback.format_exc())

    return {
        **results,
        'inputs': input_files,
    }

def encode_pycode_multiprocessing(args: MultiprocessingArgs):
    split_name = args.kwargs['split']
    rank_prefix = str(args.process_idx) if args.process_idx > 0 else ''

    return encode_pycode(
        input_files=args.inputs,
        output_prefix=os.path.join(args.output_dir, split_name + rank_prefix),
        domain=args.domain,
        lib_name_api_name_api_paths=args.lib_name_api_name_api_paths,
        api_path_api_signature_and_api_desp=args.api_path_api_signature_and_api_desp,
        tokenizer=args.kwargs['tokenizer'],
        logger=args.logger,
        logging_freq=args.logging_freq,
        rank=args.process_idx
    )

def save_pretrained_into_data_dir(output_dir: str, model_name: str):
    resolved_model_path = resolve_model_name_or_path(model_name)
    tokenizer = create_custom_gpt2_tokenizer()
    model = create_model(resolved_model_path, fp16=True)

    saved_path = os.path.join(output_dir, 'model')

    tokenizer.save_pretrained(saved_path)
    model.save_pretrained(saved_path)
    print('Save {} model to {} over.'.format(model_name, saved_path))

def try_parse_node_rank():
    node_idx, node_size = None, None
    if 'DistributedNodes' in os.environ:
        node_idx, node_size = map(int, os.environ['DistributedNodes'].split('_'))

    return node_idx, node_size

def process_main(
    input_file_path: str, 
    output_dir: str, 
    domain: str, 
    lib_name_api_name_api_paths: Dict[str, Dict[str, str]],
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]],
    split_name: str, 
    is_debug: str, 
    tokenizer_name: str, 
    num_processes: int, 
    logging_freq: int=10000
    ):
    assert split_name in ['train', 'valid', 'dev', 'test']

    num_processes = min(num_processes, multiprocessing.cpu_count())

    tokenizer = create_custom_gpt2_tokenizer()
    print('load {} {} over, size = {}.'.format(tokenizer.__class__.__name__, tokenizer_name, len(tokenizer)))

    all_input_files = get_files(os.path.join(input_file_path, split_name), "*.json*.gz")
    print("Read input files from over, size = {}.".format(len(all_input_files)))

    run_with_multiprocessing(
        run_name=f'{domain}_{split_name}_codegen',
        process_func=encode_pycode_multiprocessing,
        inputs=all_input_files,
        output_dir=output_dir,
        domain=domain,
        lib_name_api_name_api_paths=lib_name_api_name_api_paths,
        api_path_api_signature_and_api_desp=api_path_api_signature_and_api_desp,
        is_debug=is_debug,
        num_processes=num_processes,
        logging_freq=logging_freq,
        tokenizer=tokenizer,
        split=split_name
    )
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run processing for extracting the pair of API and code doc.')
    parser.add_argument('-i', '--input_names', type=str, required=True)
    parser.add_argument("--private_data_path", type=str, default="../../data/")
    parser.add_argument('-model', '--model_name', type=str, required=True)
    parser.add_argument('-p', '--pattern', type=str, default='*.json*.gz')
    parser.add_argument('-o', '--output_name', type=str, default=None)
    parser.add_argument('-split', '--split_name', type=str, default='valid')
    parser.add_argument('--private_libs', type=str)
    parser.add_argument('--build_in_libs', type=str)
    parser.add_argument('-t', '--num_processes', type=int, default=16)
    parser.add_argument('-d', '--domain', type=str, default='PrivateLibrary', required=True, choices=['PrivateLibrary'])
    parser.add_argument('-isdebug', '--is_debug', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--contain_build_in', type=str, default='True', choices=['True', 'False'])

    parser.add_argument('--logging_freq', type=int, default=10)
    args = parser.parse_args()

    split_name = args.split_name
    tokenizer_name = args.model_name

    output_dir = args.output_name
    os.makedirs(output_dir, exist_ok=True)

    lib_name_api_name_api_paths = get_dict_of_api_name_lib_api_paths(
        args.private_data_path, 
        args.private_libs, 
        args.build_in_libs, 
        args.contain_build_in
    )
    api_path_api_signature_and_api_desp = get_dict_of_api_path_api_signature_and_api_desp(
        args.private_data_path, 
        args.private_libs, 
        args.build_in_libs, 
        args.contain_build_in
    )
    process_main(
        input_file_path=args.input_names,
        output_dir=output_dir,
        domain=args.domain,
        lib_name_api_name_api_paths=lib_name_api_name_api_paths,
        api_path_api_signature_and_api_desp=api_path_api_signature_and_api_desp,
        is_debug=args.is_debug,
        tokenizer_name=tokenizer_name,
        num_processes=args.num_processes,
        split_name=split_name,
        logging_freq=args.logging_freq,
    )

    # Bind processed data with correspoinding pretrained model
    if split_name == 'train':
        node_idx, node_size = try_parse_node_rank()
        if node_idx is None or node_idx == 0:
            save_pretrained_into_data_dir(output_dir, tokenizer_name)
    pass
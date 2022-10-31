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

from file_utils import get_files, read_lines, load_githup_stars_dict, Data_Abs_Dir
from multiprocessing_utils import MultiprocessingArgs, run_with_multiprocessing
from ast_utils import *
from codegen_tokenization import create_custom_gpt2_tokenizer, create_model
from get_libs_info_from_code import *

import fairseq.data.indexed_dataset as indexed_dataset

import sys
sys.path.append("..")
from nl2code import load_pretrained_tokenizer, resolve_model_name_or_path

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def extract_feature(
    example: Dict, 
    total_api_funcs: int, 
    total_api_docs: int, 
    total_retrieved_api_funcs: int
    ) -> np.ndarray:

    feature_array = []
    feature_array.append(math.log10(example['stars'] + 1))

    feature = example['feature']
    feature_array.append(feature['ut_score'])
    feature_array.append(feature['func_score'])
    feature_array.append(total_api_funcs)
    feature_array.append(total_api_docs)
    feature_array.append(total_retrieved_api_funcs)

    feature_array = np.array(feature_array, dtype=np.float32)
    assert len(feature_array) == 6
    return feature_array

def encode_pycode(
    input_files: List[str], 
    output_prefix: str, 
    domain: str, 
    lib_name_api_name_api_paths: Dict[str, Dict[str, str]],
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]],
    tokenizer: PreTrainedTokenizer, 
    logger: logging.Logger, 
    logging_freq: int=10000, 
    rank: int=0,
    is_debug: str="False",
    perturbation_probability: float=0.0,
    stype: str="v2"):

    logger.info("Process {} start to process {} files:\n{}".format(str(rank), len(input_files), "\n".join(input_files)))
    
    features = []
    results = defaultdict(int)

    try:
        builder = indexed_dataset.make_builder(
            out_file=indexed_dataset.data_file_path(output_prefix),
            impl='mmap',
            vocab_size=len(tokenizer)
        )
        raw_output_path = output_prefix+"_raw_merged_code.jsonl"

        # judge the dir path of raw_output_path is exist or not
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
        raw_merged_writer = open(raw_output_path, "w+")
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id

        repo_stars = load_githup_stars_dict()

        logger.info("Rank {}, tokenizer bos token id = {}, eos token id = {}.".format(rank, bos_token_id, eos_token_id))


        for line in read_lines(input_files):
            try:
                results["cnt_processed"] += 1
                example = json.loads(line)
                norm_text = example['content']

                if 'stars' not in example:
                    example['stars'] = repo_stars.get(example['repo_name'], 0)

                example['stars'] = max(0, example['stars'])

                # default feature value
                if 'feature' not in example:
                    example['feature'] = {
                        'ut_score': 1.0,
                        "func_score": 1.0,
                    }

                try:
                    max_time_out = 5
                    try:
                        norm_text = func_timeout(max_time_out, autopep8.fix_code, args=(norm_text,))
                    except FunctionTimedOut:
                        results[f"fix_code_time_out_exception_{max_time_out}s"] += 1
                        continue
                    except Exception as e:
                        results["auto_pep_execution_error"] += 1
                        continue

                    text_list = norm_text.split("\n\n\n")

                    try:
                        if stype == "v2": # hold prompt: [start]\n# [end]
                            if is_debug == "False":
                                sketch_norm_text, total_api_funcs, total_api_docs, total_retrieved_api_funcs = func_timeout(
                                    max_time_out, 
                                    craft_api_merged_corpus, 
                                    args=(
                                        norm_text,
                                        text_list,
                                        "",
                                        lib_name_api_name_api_paths,
                                        api_path_api_signature_and_api_desp,
                                        "only function",
                                        perturbation_probability
                                    )
                                )
                            else:
                                sketch_norm_text, total_api_funcs, total_api_docs, total_retrieved_api_funcs = craft_api_merged_corpus(
                                    norm_text,
                                    text_list,
                                    "",
                                    lib_name_api_name_api_paths,
                                    api_path_api_signature_and_api_desp,
                                    "only function",
                                    perturbation_probability
                                )
                        elif stype == "v3": # wild prompt: Please use the following APIs to solve the programming problem
                            raise NotImplementedError("Not implemented yet.")
                        else:
                            raise ValueError("stype should be v2 or v3.")
                    except FunctionTimedOut:
                        results[f"api_merge_time_out_exception_{max_time_out}s"] += 1
                        continue
                    except Exception as e:
                        results["api_merge_execution_error"] += 1
                        continue
                except Exception as e:
                    results["transform_to_sketch_failed"] += 1
                    continue

                if total_api_funcs == 0 or total_retrieved_api_funcs == 0:
                    results["no_any_api_funcs"] += 1
                    continue
                raw_merged_writer.write(json.dumps({
                    "api_code": sketch_norm_text
                })+"\n")
                input_ids = [bos_token_id] + tokenizer.encode(sketch_norm_text) + [eos_token_id]
                results['tokens_in_mb'] += len(input_ids) / 1e6
                builder.add_item(torch.tensor(input_ids))
                
                feature = extract_feature(example, total_api_funcs, total_api_docs, total_retrieved_api_funcs)
                features.append(feature)
                results["cnt_saved"] += 1

                if results["cnt_processed"] % logging_freq == 0:
                    logger.info("Process {} processed {} lines, saved = {}, {}.".format(
                        rank, results["cnt_processed"], results["cnt_saved"], ", ".join([f"{k} = {v}" for k, v in results.items()]))
                    )

                    logger.info("Rank {}, sample: feature = {}.".format(
                        rank, feature.tolist()
                    ))

            except:
                results["failed"] += 1
                logger.info("Process {}: ".format(rank) + traceback.format_exc())

        logger.info("Process {} start to finalize dataset ...".format(rank))

        raw_merged_writer.close()
        builder.finalize(indexed_dataset.index_file_path(output_prefix))
        features = np.stack(features, axis=0)
        np.save(output_prefix + "_features.npy", features)

        logger.info("Process {} processed all over, {}, avg features = {}.".format(
            rank, ", ".join([f"{k} = {v}" for k, v in results.items()]), " ".join(map(str, np.mean(features, axis=0).tolist())))
        )

        results['num_processes'] = 1
        
    except:
        results["failed"] += 1
        logger.info("Process {}: ".format(rank) + traceback.format_exc())

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
        rank=args.process_idx,
        is_debug=args.is_debug,
        perturbation_probability=args.perturbation_probability,
        stype=args.stype
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
    perturbation_probability: float,
    stype: str,
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
        perturbation_probability=perturbation_probability,
        stype=stype,
        logging_freq=logging_freq,
        tokenizer=tokenizer,
        split=split_name
    )
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run processing for pycode')
    parser.add_argument('-i', '--input_names', type=str, required=True)
    parser.add_argument("--private_data_path", type=str, default="/mnt/v-dzan/datasets/CERT/PrivateLibrary/Train/")
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
    parser.add_argument('-pp', '--perturbation_probability', type=float, default=0.0)
    parser.add_argument('--style', type=str, default='v2', choices=['v1', 'v2', 'v3', 'v4', 'v5'])

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
        perturbation_probability=args.perturbation_probability,
        stype=args.style,
        logging_freq=args.logging_freq,
    )

    # Bind processed data with correspoinding pretrained model
    if split_name == 'train':
        node_idx, node_size = try_parse_node_rank()
        if node_idx is None or node_idx == 0:
            save_pretrained_into_data_dir(output_dir, tokenizer_name)
    pass
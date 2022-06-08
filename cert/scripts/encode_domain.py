# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import os
import logging
import time
import json
import traceback
import numpy as np
from typing import Dict, List
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import multiprocessing
import multiprocessing.dummy
import math
import ipdb
import docformatter
import autopep8
from func_timeout import func_timeout, FunctionTimedOut

from file_utils import get_files, read_lines, load_githup_stars_dict, Data_Abs_Dir
from multiprocessing_utils import MultiprocessingArgs, run_with_multiprocessing
from ast_utils import *

import fairseq.data.indexed_dataset as indexed_dataset

import sys
sys.path.append("..")
from nl2code import load_pretrained_tokenizer, resolve_model_name_or_path

def extract_feature(example: Dict) -> np.ndarray:
    feature_array = []
    feature_array.append(math.log10(example['stars'] + 1))

    feature = example['feature']
    feature_array.append(feature['ut_score'])
    feature_array.append(feature['func_score'])

    feature_array = np.array(feature_array, dtype=np.float32)
    assert len(feature_array) == 3
    return feature_array

def encode_pycode(input_files: List[str], output_prefix: str, domain: str, type_name: str, tokenizer: PreTrainedTokenizer, logger: logging.Logger, logging_freq: int=10000, rank: int=0):
    logger.info("Process {} start to process {} files:\n{}".format(str(rank), len(input_files), "\n".join(input_files)))

    features = []
    results = defaultdict(int)

    try:
        builder = indexed_dataset.make_builder(
            out_file=indexed_dataset.data_file_path(output_prefix),
            impl='mmap',
            vocab_size=len(tokenizer)
        )

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
                if type_name=="normal":
                    input_ids = [bos_token_id] + tokenizer.encode(norm_text) + [eos_token_id]
                elif type_name=="sketcher":
                    try:
                        norm_sketch = transform_code_to_sketch(norm_text)
                    except Exception as e:
                        results["transform_to_sketch_failed"] += 1
                        continue
                    input_ids = [bos_token_id] + tokenizer.encode(norm_sketch) + [eos_token_id]
                elif type_name=="generator":
                    try:
                        norm_sketch = transform_code_to_sketch(norm_text)
                        max_time_out = 5
                        try:
                            norm_text = func_timeout(max_time_out, autopep8.fix_code, args=(norm_text,))
                            norm_sketch = func_timeout(max_time_out, autopep8.fix_code, args=(norm_sketch,))
                        except FunctionTimedOut:
                            results[f"time_out_exception_{max_time_out}s"] += 1
                            continue
                        except Exception as e:
                            results["auto_pep_execution_error"] += 1
                            continue
                        sketch_list, text_list = norm_sketch.split("\n\n\n"), norm_text.split("\n\n\n")
                        if len(sketch_list) != len(text_list):
                            results["sketch_text_len_dismatch"] += 1
                            continue
                        sketch_norm_text = craft_merged_corpus(sketch_list, text_list, linker="\n# [segmentation]\n")
                    except Exception as e:
                        results["transform_to_sketch_failed"] += 1
                        continue

                    input_ids = [bos_token_id] + tokenizer.encode(sketch_norm_text) + [eos_token_id]
                else:
                    raise ValueError("type_name must be normal, sketcher or generator.")

                results['tokens_in_mb'] += len(input_ids) / 1e6
                builder.add_item(torch.tensor(input_ids))
                feature = extract_feature(example)
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
                logger.error("Process {}: ".format(rank) + traceback.format_exc())

        logger.info("Process {} start to finalize dataset ...".format(rank))

        builder.finalize(indexed_dataset.index_file_path(output_prefix))
        features = np.stack(features, axis=0)
        np.save(output_prefix + "_features.npy", features)

        logger.info("Process {} processed all over, {}, avg features = {}.".format(
            rank, ", ".join([f"{k} = {v}" for k, v in results.items()]), " ".join(map(str, np.mean(features, axis=0).tolist())))
        )

        results['num_processes'] = 1
        
    except:
        results["failed"] += 1
        logger.error("Process {}: ".format(rank) + traceback.format_exc())

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
        type_name=args.type_name,
        tokenizer=args.kwargs['tokenizer'],
        logger=args.logger,
        logging_freq=args.logging_freq,
        rank=args.process_idx
    )

def save_pretrained_into_data_dir(output_dir: str, model_name: str):
    resolved_model_path = resolve_model_name_or_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    model = AutoModelForCausalLM.from_pretrained(resolved_model_path)

    saved_path = os.path.join(output_dir, 'model')

    tokenizer.save_pretrained(saved_path)
    model.save_pretrained(saved_path)
    print('Save {} model to {} over.'.format(model_name, saved_path))

def try_parse_node_rank():
    node_idx, node_size = None, None
    if 'DistributedNodes' in os.environ:
        node_idx, node_size = map(int, os.environ['DistributedNodes'].split('_'))

    return node_idx, node_size

def process_main(input_file_path: str, output_dir: str, domain: str, type_name: str, split_name: str, is_debug: str, tokenizer_name: str, num_processes: int, logging_freq: int=10000):
    assert split_name in ['train', 'valid', 'dev', 'test']

    num_processes = min(num_processes, multiprocessing.cpu_count())

    tokenizer = load_pretrained_tokenizer(tokenizer_name)
    print('load {} {} over, size = {}.'.format(tokenizer.__class__.__name__, tokenizer_name, len(tokenizer)))

    all_input_files = get_files(os.path.join(input_file_path, split_name), "*.json*.gz")
    print("Read input files from over, size = {}.".format(len(all_input_files)))

    run_with_multiprocessing(
        run_name=f'{domain}_{type_name}_{split_name}',
        process_func=encode_pycode_multiprocessing,
        inputs=all_input_files,
        output_dir=output_dir,
        domain=domain,
        is_debug=is_debug,
        type_name=type_name,
        num_processes=num_processes,
        logging_freq=logging_freq,
        tokenizer=tokenizer,
        split=split_name
    )
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run processing for pycode')
    parser.add_argument('-i', '--input_names', type=str, required=True)
    parser.add_argument('-model', '--model_name', type=str, required=True)
    parser.add_argument('-p', '--pattern', type=str, default='*.json*.gz')
    parser.add_argument('-o', '--output_name', type=str, default=None)
    parser.add_argument('-split', '--split_name', type=str, default='valid')
    parser.add_argument('-t', '--num_processes', type=int, default=16)
    parser.add_argument('-d', '--domain', type=str, default='Pandas', required=True, choices=['Pandas', 'Numpy', 'NLTK'])
    parser.add_argument('-type','--type_name', type=str, default='normal', required=True, choices=['normal', 'sketcher', 'generator'])
    parser.add_argument('-isdebug', '--is_debug', type=str, default='False', choices=['True', 'False'])

    parser.add_argument('--logging_freq', type=int, default=100)
    args = parser.parse_args()

    split_name = args.split_name
    tokenizer_name = args.model_name

    output_dir = args.output_name
    os.makedirs(output_dir, exist_ok=True)

    process_main(
        input_file_path=args.input_names,
        output_dir=output_dir,
        domain=args.domain,
        type_name=args.type_name,
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
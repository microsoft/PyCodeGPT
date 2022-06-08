# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
from secrets import choice
from typing import Dict

import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.base import Pipeline

from tqdm import tqdm
import ipdb

from human_eval.data import write_jsonl, read_problems

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
torch.set_num_threads(16)

def remove_samples_in_comments(prompt: str) -> str:
    prompt = prompt.replace('\n', '#N#')
    p = re.compile(r'(#N#)+    (>>>|Example).*"""')
    return p.sub('\n    """', prompt).replace('#N#', '\n')

def load_generation_pipe(model_name_or_path: str, gpu_device: int=0):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=gpu_device
    )

    print("load generation pipeline from {} over, vocab size = {}, eos id = {}, gpu device = {}.".format(
        model_name_or_path, len(tokenizer), tokenizer.eos_token_id, gpu_device)
    )

    return pipe

def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()

def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    set_seed(123)

    code_gens = pipe(prompt,
        num_return_sequences=num_completions,
        **gen_kwargs
    )

    return [first_block(code_gen["generated_text"][len(prompt):]) for code_gen in code_gens]

def evaluate_on_human_eval(
    model_name_or_path: str,
    temperature: float,
    top_p: float,
    num_samples_per_task: int,
    max_new_tokens: int,
    gpu_device: int,
    domain: str,
    model_version: str,
    overwrite: bool=True,
    fetch_from_huggingface: bool=True
    ) -> str:

    if os.path.exists(model_name_or_path):
        output_dir = model_name_or_path
    elif fetch_from_huggingface:
        output_dir = "output/{}".format(model_name_or_path.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)
    else:
        return None

    eval_name = f"{domain}_API_eval.{model_version}.t{temperature}.p{top_p}.l{max_new_tokens}.n{num_samples_per_task}"
    saved_path = os.path.join(output_dir, f"{eval_name}.samples.jsonl")

    print(f"saved_path: {saved_path}")

    if not overwrite and os.path.exists(saved_path):
        return saved_path

    pipe: Pipeline = load_generation_pipe(model_name_or_path, gpu_device=gpu_device)
    gen_kwargs = {
        "do_sample": True, # Default: True
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": 0,
        "pad_token_id": pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id else pipe.tokenizer.eos_token_id,
        "eos_token_id": pipe.tokenizer.eos_token_id
    }

    problems = read_problems()
    samples = []
    generate_batch_size = min(50, num_samples_per_task)

    bos_token = pipe.tokenizer.bos_token if pipe.tokenizer.bos_token else pipe.tokenizer.eos_token

    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"]
        for _ in range(num_samples_per_task // generate_batch_size):
            input_prompt = bos_token + prompt
            gen_results = complete_code(pipe, input_prompt, num_completions=generate_batch_size, **gen_kwargs)
            for gen_result in gen_results:
                # samples.append(dict(task_id=task_id, completion=truncate(gen_result)))
                samples.append(dict(task_id=task_id, completion=gen_result))

    write_jsonl(saved_path, samples)
    return saved_path

def run_samples_test(model_name_or_path: str):
    pipe = load_generation_pipe(model_name_or_path)

    prompt = 'def convert_hours_minutes(hours):'
    complete_code(pipe, prompt, num_completions=4)

    prompt = '''def area_of_rectangle(a: float, b: float):
    """Returns the area of the rectangle."""'''
    complete_code(pipe, prompt, num_completions=2)

    prompt = '''def get_urls_from_html(html):
    Get all embedded URLs in a HTML string.'''
    complete_code(pipe, prompt, num_completions=4)

    prompt = '''def is_sorted(lst):
    """
    Given a list of numbers, return whether or not they are sorted
    in ascending order. If list has more than 1 duplicate of the same
    number, return False. Assume no negative numbers and only integers.
    """'''
    complete_code(pipe, prompt, 200, num_completions=4)

    prompt = '''def is_sorted(lst):
    """
    Given a list of numbers, return whether or not they are sorted in ascending order.
    If list has more than 1 duplicate of the same number, return False. Assume no negative numbers and only integers.
    """'''
    complete_code(pipe, prompt, 200, num_completions=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation for Code Generation Model.')

    parser.add_argument('-model', '--model_name_or_path', type=str, required=True)
    parser.add_argument('-n', '--num_completions', type=int, default=100)
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-p', '--top_p', type=float, default=0.95)
    parser.add_argument('-l', '--max_new_tokens', type=int, default=100)
    parser.add_argument('-gpu', "--gpu_device", type=int, default=0)
    parser.add_argument('-d', '--domain', type=str, default="Pandas", choices=["Pandas", "Numpy", "NLTK"])
    parser.add_argument('-mv', '--model_version', type=str, default="CERT", choices=["PYCODEGPT", "PYCODEGPT_XL", "CERT"])

    args = parser.parse_args() 

    print(evaluate_on_human_eval(
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples_per_task=args.num_completions,
        gpu_device=args.gpu_device,
        max_new_tokens=args.max_new_tokens,
        domain=args.domain,
        model_version=args.model_version
    ))
    pass
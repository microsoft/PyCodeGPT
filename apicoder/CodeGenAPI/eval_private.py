import os
import re
from typing import Dict

import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from transformers.pipelines.base import Pipeline

from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from nl2code.modeling_codegen import CodeGenForCausalLM


device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
torch.set_num_threads(16)

def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion

def remove_samples_in_comments(prompt: str) -> str:
    prompt = prompt.replace('\n', '#N#')
    p = re.compile(r'(#N#)+    (>>>|Example).*"""')
    return p.sub('\n    """', prompt).replace('#N#', '\n')

def create_model(ckpt, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return CodeGenForCausalLM.from_pretrained(ckpt)

def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t

def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t

def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t

def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    t.padding_side = 'left'
    t.pad_token = 50256
    return t

def load_generation_pipe(model_name_or_path: str, gpu_device: int=0):
    model = create_model(ckpt=model_name_or_path, fp16=True)
    tokenizer = create_custom_gpt2_tokenizer()
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
    fetch_from_huggingface: bool=True,
    api_number: int=0,
    human_in_the_loop: str="False",
    user_name: str="machine",
    make_sense: str=""
) -> str:

    if os.path.exists(model_name_or_path):
        output_dir = model_name_or_path
    elif fetch_from_huggingface:
        output_dir = "output/{}".format(model_name_or_path.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)
    else:
        return None

    eval_name = f"official_{domain}{make_sense}.API_number_{str(api_number)}.{model_version}.hm_{human_in_the_loop}.{user_name}.t{temperature}.p{top_p}.l{max_new_tokens}.n{num_samples_per_task}"
    saved_path = os.path.join(output_dir, f"{eval_name}.samples.jsonl")

    print(f"saved_path: {saved_path}")

    if not overwrite and os.path.exists(saved_path):
        return saved_path

    pipe: Pipeline = load_generation_pipe(model_name_or_path, gpu_device=gpu_device)
    gen_kwargs = {
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": 0,
        "pad_token_id": pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id else pipe.tokenizer.eos_token_id,
        "use_cache": True
    }

    problems = read_problems()
    samples = []
    generate_batch_size = min(25, num_samples_per_task) # default batch size is 50

    bos_token = pipe.tokenizer.bos_token if pipe.tokenizer.bos_token else pipe.tokenizer.eos_token

    def convert_back(gen_result: str, domain: str) -> str:
        domain_key_words_path = f"./data/{domain.lower()}_key_words.jsonl"
        domain_key_words_reader = open(domain_key_words_path, "r")
        total_line = ""
        for line in domain_key_words_reader.readlines():
            total_line += line
        line_dict = eval(total_line)
        for old, new in line_dict.items():
            gen_result.replace(new, old)
        return gen_result

    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"].strip()
        for _ in range(num_samples_per_task // generate_batch_size):
            input_prompt = bos_token + prompt
            gen_results = complete_code(pipe, input_prompt, num_completions=generate_batch_size, **gen_kwargs)
            for gen_result in gen_results:
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
    parser = argparse.ArgumentParser(description='Run evaluation for Code Generation Model on CodeGen model.')

    parser.add_argument('-model', '--model_name_or_path', type=str, required=True)
    parser.add_argument('-n', '--num_completions', type=int, default=100)
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-p', '--top_p', type=float, default=0.95)
    parser.add_argument('-l', '--max_new_tokens', type=int, default=100)
    parser.add_argument('-gpu', "--gpu_device", type=int, default=0)
    parser.add_argument('-d', '--domain', type=str, default="Pandas", choices=["Pandas", "Numpy", "Monkey", "BeatNum", "TorchData"])
    parser.add_argument('-mv', '--model_version', type=str, default="CERT", choices=["CodeGen", "CodeGen_XL", "CERT", "API_Coder"])
    parser.add_argument('--api_number', default=0, help="API number for API_Coder")
    parser.add_argument('--human_in_the_loop', default="False", help="是否是human in the loop")
    parser.add_argument('--user_name', default="machine", help="human的名字，如果不是human就默认是machine")
    parser.add_argument('--make_sense', default="", help="prompt是否是friendly的？")

    args = parser.parse_args() 

    print(evaluate_on_human_eval(
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples_per_task=args.num_completions,
        gpu_device=args.gpu_device,
        max_new_tokens=args.max_new_tokens,
        domain=args.domain,
        model_version=args.model_version,
        api_number=args.api_number,
        human_in_the_loop=args.human_in_the_loop,
        user_name=args.user_name,
        make_sense=args.make_sense
    ))
    pass
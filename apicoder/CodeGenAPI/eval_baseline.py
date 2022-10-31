import os
import re

import torch
from transformers import pipeline, set_seed
from transformers import GPTNeoForCausalLM, AutoTokenizer, GPT2LMHeadModel, RobertaTokenizer, T5ForConditionalGeneration, BartTokenizer, BartForCausalLM, BartModel
from transformers.pipelines.base import Pipeline

from tqdm import tqdm

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
torch.set_num_threads(16)
LEVEL_TOKEN = '   '

def extract_first_block(string):
    return re.split('\nclass |\ndef |\n#|\nif |\nprint', string)[0].rstrip()

def remove_samples_in_comments(prompt: str) -> str:
    prompt = prompt.replace('\n', '#N#')
    p = re.compile(r'(#N#)+    (>>>|Example).*"""')
    return p.sub('\n    """', prompt).replace('#N#', '\n')

temperature = 0.2
max_length = 50
do_normalize = True
remove_examples = False

def generate_code(model: GPTNeoForCausalLM, tokenizer: RobertaTokenizer, text: str, num_generated=10):

    #text = text.lstrip('\n')
    if remove_examples:
        text = remove_samples_in_comments(text)

    text = text.replace("    ", '\t') if do_normalize else text

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    beam_outputs = model.generate(
        input_ids,
        do_sample=True,
        # num_beams=5, # default is 1 or comment this line to not use beam search
        no_repeat_ngram_size=8,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length = max_length + input_ids.size(1),
        top_p=0.9,
        num_return_sequences=num_generated
    )

    gen_results, results_set = [], set([])
    for _, beam_output in enumerate(beam_outputs):
        gen_output = beam_output[input_ids.size(1):]
        gen_text = tokenizer.decode(gen_output, skip_special_tokens=True)
        gen_text = extract_first_block(gen_text)

        if do_normalize:
            gen_text = gen_text.replace('\t', '    ')

        gen_results.append(gen_text)
        results_set.add(gen_text)

    return gen_results

def load_generation_pipe(model_name_or_path: str, model_type='gpt-neo'):
    if model_type == 'gpt-neo':
        model = GPTNeoForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif model_type == 'codet5':
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    elif model_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif model_type == 'bart':
        model = BartModel.from_pretrained(model_name_or_path)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    else:
        raise ValueError('model_type must be one of `gpt-neo`, `codet5`, `gpt2`, `bart`')

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
    print("load generation pipeline from {} over, vocab size = {}, eos id = {}.".format(model_name_or_path, len(tokenizer), tokenizer.eos_token_id))
    return pipe

def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)

    if do_normalize:
        prompt = prompt.replace("    ", '\t')

    code_gens = pipe(prompt,
                    num_return_sequences=num_completions,
                    num_beams=4,
                    max_length=max_length,
                    no_repeat_ngram_size=8)

    code_strings = []
    for code_gen in code_gens:
        generated_code = extract_first_block(code_gen['generated_text'][len(prompt):])
        if do_normalize:
            generated_code = generated_code.replace('\t', '    ')
        code_strings.append(prompt + generated_code)

    print(('\n'+'='*80 + '\n').join(code_strings))
    print('\n' + '='*80)
    return code_strings

def run_human_eval(
    model_name_or_path: str, 
    output_dir: str="./output/", 
    model_type='gpt-neo', 
    num_samples_per_task: int=50, 
    library: str="pandas",
    api_number=0,
    user_name="machine"
    ):
    from human_eval.data import write_jsonl, read_problems

    model_name = os.path.split(model_name_or_path.rstrip('/'))[-1]

    if os.path.exists(model_name_or_path):
        output_dir = model_name_or_path

    print('Run evaluation for {}, model_type = {}.'.format(model_name, model_type))
    pipe: Pipeline = load_generation_pipe(model_name_or_path, model_type)

    problems = read_problems()
    samples = []
    generate_batch_size = min(50, num_samples_per_task)
    for task_id in tqdm(problems):
        for _ in range(num_samples_per_task // generate_batch_size):            
            gen_results = generate_code(pipe.model, pipe.tokenizer, problems[task_id]["prompt"], generate_batch_size)
            for gen_result in gen_results:
                samples.append(dict(task_id=task_id, completion=gen_result))

    saved_path = os.path.join(output_dir, f"official_{library}_{user_name}_{model_type}_apinum_{str(api_number)}_temp_{temperature}.samples.jsonl")
    write_jsonl(saved_path, samples)

    print("Run prediction over, save to {}".format(saved_path))
    pass

def run_samples_test(model_name_or_path: str, model_type: str):
    pipe = load_generation_pipe(model_name_or_path, model_type)

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

    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-n', '--num_samples_per_task', type=int, default=200) # default to 100
    parser.add_argument('-type', '--model_type', type=str, default='codet5')
    parser.add_argument('-temp', '--temperature', type=float, default=0.0)
    parser.add_argument('-max_len', '--max_length', type=float, default=100)
    parser.add_argument('-lib', '--library', type=str, default='pandas', choices=["Pandas", "Numpy", "Monkey", "BeatNum", "TorchData"])
    parser.add_argument('--api_number', default=0, help="数据集是提示几个api number")
    parser.add_argument('--user_name', type=str, help="运行的类型，是human还是machine")

    args = parser.parse_args()

    temperature = args.temperature
    max_length = args.max_length
    if not os.path.exists(args.model):
        do_normalize = False

    print('Do_normalize: {}, temperature: {}, remove examples: {}'.format(do_normalize, temperature, remove_examples))
    run_human_eval(args.model, model_type=args.model_type, num_samples_per_task=args.num_samples_per_task, library=args.library, api_number=args.api_number, user_name=args.user_name)

    pass
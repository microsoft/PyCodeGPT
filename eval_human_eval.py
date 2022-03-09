import os
import re
import argparse
from tqdm import tqdm

from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.base import Pipeline

from human_eval.data import write_jsonl, read_problems

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

def extract_function_block(string):
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()

def run_code_generation(pipe, prompt, num_completions=1, **gen_kwargs):
    set_seed(123)

    code_gens = pipe(prompt,
        num_return_sequences=num_completions,
        **gen_kwargs
    )

    return [extract_function_block(code_gen["generated_text"][len(prompt):]) for code_gen in code_gens]

def evaluate_on_human_eval(
    model_name_or_path: str,
    temperature: float,
    top_p: float,
    num_samples_per_task: int,
    max_new_tokens: int,
    gpu_device: int,
    output_dir: str,
    ) -> str:

    pipe: Pipeline = load_generation_pipe(model_name_or_path, gpu_device=gpu_device)
    eval_name = f"human_eval.t{temperature}.p{top_p}.l{max_new_tokens}.n{num_samples_per_task}"

    if output_dir is None:
        if os.path.exists(model_name_or_path):
            output_dir = model_name_or_path
        else:
            raise ValueError("Output dir can't be null if you are not evaluation a local model.")

    os.makedirs(output_dir, exist_ok=True)
    saved_path = os.path.join(output_dir, f"{eval_name}.samples.jsonl")

    gen_kwargs = {
        "do_sample": True,
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
        # Strip operation is important as new tokenizer will not treat '\n' as a independent token
        prompt = problems[task_id]["prompt"].strip()

        for _ in range(num_samples_per_task // generate_batch_size):
            input_prompt = bos_token + prompt
            gen_results = run_code_generation(pipe, input_prompt, num_completions=generate_batch_size, **gen_kwargs)
            for gen_result in gen_results:
                samples.append(dict(task_id=task_id, completion=gen_result))
    
    write_jsonl(saved_path, samples)
    print("Run generation over, save {} samples to {}.".format(len(samples), saved_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation for code generation model on human-eval.')

    parser.add_argument('-model', '--model_name_or_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-n', '--num_completions', type=int, default=100)
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-p', '--top_p', type=float, default=0.95)
    parser.add_argument('-l', '--max_new_tokens', type=int, default=100)
    parser.add_argument('-gpu', "--gpu_device", type=int, default=0)

    args = parser.parse_args()

    evaluate_on_human_eval(
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples_per_task=args.num_completions,
        max_new_tokens=args.max_new_tokens,
        gpu_device=args.gpu_device,
        output_dir=args.output_dir,
    )
    pass
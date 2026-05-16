
# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import json
import tiktoken
from utils.model_utils import load_gpt2_model
from utils.gen_text import generate_text


def get_prompts(filepath):

    # Load the JSON prompts file
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Unable to find JSON prompts file {filepath}')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        json_prompts = json.load(f)
    prompt_data = {int(k): v for k, v in json_prompts.items()}

    return prompt_data


def postprocess_model_output(tokens_out):

    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token = 50256

    out = []
    for t in tokens_out:
        if t == eos_token:
            break
        out.append(t)

    return tokenizer.decode(out)


def dump_responses(model_responses, annotations, filepath):

    formatted = []
    for id in model_responses.keys():
        lines = f"\n{80 * '='}\n"
        lines += f"id: {id}\n{40 * '-'}\n"
        lines += f"{model_responses[id]}\n{40 * '-'}\n"
        lines += f"Dataset annotation: {annotations[id]}"
        formatted.append(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(formatted))


def test_prompt(
        model_dir,
        model_name,
        prompts_filepath,
        responses_filepath,
        id_range=None
):

    prompt_data = get_prompts(prompts_filepath)

    # Load the model
    print(f'Loading model `{model_name}` from directory {model_dir}')
    model = load_gpt2_model(model_dir, model_name)

    model_responses = {}
    annotations = {}

    for index, example in prompt_data.items():
        if id_range is not None:
            if index < id_range[0] or index > id_range[1]:
                continue

        prompt = example["prompt"]
        if prompt.startswith('### Task: answer question\n\n'):
            task = 'answer question'
        elif prompt.startswith('### Task: simplify text\n\n'):
            task = 'simplify text'
        elif prompt.startswith('### Task: classify news\n\n'):
            task = 'classify news'
        else:
            raise ValueError(
                f'Unable to identify the task to perform from input prompt:\n{prompt}\n'
                '\nValid tasks are: "follow instructions", "simplify text", "classify news"'
            )
        
        print(f'Prompt id: {index}    Task: {task}')

        if model.lora_config is not None:
            model.lora_adapter_tasks = {
                "answer question": 0,
                "simplify text": 1,
                "classify news": 2
            }

            adapter_idx = model.lora_adapter_tasks[task]
            model.activate_adapter(adapter_idx)

        model.compile()
 
        sampling_method = 'top_k' if task == 'simplify text' else 'greedy'
        output_len = 100

        tokens_out = generate_text(
            model,
            [prompt],   # The function takes a list of prompts.
            output_len=output_len,
            sampling_method=sampling_method,
            temperature=0.8,
            top_k=20
        )

        model_responses[index] = postprocess_model_output(tokens_out[0])
        annotations[index] = example["annotation"]

    dump_responses(model_responses, annotations, responses_filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model_dir',
        help='Directory where the model files are',
        required=True,
        type=str
    )
    parser.add_argument(
        '--model_name',
        help='Name of the model',
        required=True,
        type=str
    )
    parser.add_argument(
        '--prompts_filepath',
        help='Path to the JSON containing the prompts',
        required=True,
        type=str
    )
    parser.add_argument(
        '--responses_filepath',
        help='Path to the output .txt file that contains the prompts and their model responses',
        required=True,
        type=str
    )
    parser.add_argument(
        '--id_range',
        help='Range of prompt IDs to test',
        type=str
    )
    args = parser.parse_args()

    id_range = eval(args.id_range) if args.id_range is not None else None

    test_prompt(
        args.model_dir,
        args.model_name,
        args.prompts_filepath,
        args.responses_filepath,
        id_range=id_range
    )

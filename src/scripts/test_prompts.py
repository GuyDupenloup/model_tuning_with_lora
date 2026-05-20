# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import json
import tiktoken
import tensorflow as tf

from utils.model_utils import load_gpt2_model
from utils.gen_text import generate_text


def get_prompts(filepath):

    # Load the JSON prompts file
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Unable to find JSON prompts file {filepath}')
    
    with open(filepath, "r", encoding="utf-8") as f:
        json_prompts = json.load(f)
    prompt_data = {int(k): v for k, v in json_prompts.items()}

    tokenizer = tiktoken.get_encoding("gpt2")
    seq_len = 1024
    pad_token = 50256

    for _, example in prompt_data.items():
        prompt = example["prompt"]

        # Get the task to perform
        if prompt.startswith("### Task: answer question\n\n"):
            task = "answer question"
        elif prompt.startswith("### Task: simplify text\n\n"):
            task = "simplify text"
        elif prompt.startswith("### Task: answer question\n\n"):
            task = "answer question"
        else:
            raise ValueError(
                f"Unable to identify the task to perform from input prompt:\n{prompt}\n"
                "\nValid tasks are: 'follow instructions', 'simplify text', 'classify news'"
            )
        example["task"] = task

        # Tokenize the prompt, truncate and pad
        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = prompt_ids[:seq_len]
        prompt_ids += [pad_token] * (seq_len - len(prompt_ids))
        example["prompt_ids"]

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


def test_prompts(project_root, model_size):
    
    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")
    
    # Load the model with LoRA adapters
    model_dir = os.path.join(project_root, f"gpt2_{model_size}", "trained_models")
    model_name = "lora_adapters"

    # print(f">>Loading model `{model_name}` from directory {model_dir}")
    # model = load_gpt2_model(model_dir, model_name)

    # Load the prompts file
    prompts_fn = os.path.join(project_root, f"gpt2_{model_size}", "tests", "example_prompts.json")
    print(f">> Loading prompts file {prompts_fn}")

    prompt_data = get_prompts(prompts_fn)

    model_responses = {}
    annotations = {}

    for index, example in prompt_data.items():

        # Activate the task adapter
        task = example["task"]
        adapter_tasks = model.lora_config["tasks"]
        print(f"Prompt id: {index}    Task: {example['task']}")

        if example["task"] == "answer question":
            adapter_selector = [1, 0, 0]
        elif example["task"] == "simplify text":
            adapter_selector = [0, 1, 0]
        elif example["task"] == "classify news":
            adapter_selector = [0, 0, 1]
        else:
            adapter_selector = [0, 0, 0]

        infer_prompts = tf.constant([example["prompt_ids"] for _ in range(12)], dtype=tf.int32)
        infer_adapter_selector = tf.constant([adapter_selector for _ in range(12)], dtype=tf.int32)

        print('==>', tf.shape(infer_prompts))
        print('==>', tf.shape(infer_adapter_selector))
        
        # Activate the adapter
        # model.activate_adapter(adapter)
        # model.compile()
 
        sampling_method = 'top_k' if task == 'simplify text' else 'greedy'

        tokens_out = generate_text(
            model,
            prompts=infer_prompts,
            adapter_selector=infer_adapter_selector,
            output_len=100,
            sampling_method=sampling_method,
            temperature=0.8,
            top_k=20
        )

        model_responses[index] = postprocess_model_output(tokens_out[0])
        annotations[index] = example["annotation"]

    # responses_fn = os.path.join(project_root, f"gpt2_{model_size}", "tests", "lora_responses.txt")
    # print(f"Writing model responses to file {responses_fn}")
    # dump_responses(model_responses, annotations, responses_fn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--project_root",
        help="Project root directory",
        required=True,
        type=str
    )
    parser.add_argument(
        "--model_size",
        help="Model size, one of '124M', '355M', '774M', '1542M'",
        type=str,
        default="124M"
    )
    args = parser.parse_args()

    test_prompts(
        args.project_root,
        args.model_size
    )

# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import json
import tiktoken
import tensorflow as tf

from utils.model_utils import load_gpt2_model
from utils.gen_text import generate_text


def get_prompt_data(filepath, tokenizer, seq_len=1024, pad_token=50256):
    """
    Loads a JSON file containing example prompts, truncates/pads them to `seq_len`,
    and generates the attention masks.

    JSON file syntax:
    {
       "0": {
           "prompt": "### Task: answer question\n\n### Context: The investors eventually left Tesla penniless. 
               He had to work at various electrical repair jobs and even as a ditch digger for $2 per day.\n\n
               ### Question: What else did Tesla do for work at this time?\n\n### Answer: ",
           "reference": "various electrical repair jobs"
        },
        "1": {
           "prompt": "### Task: simplify text\n\n### Text: He settled in London, devoting himself chiefly 
               to practical teaching .\n\n### Simplified: ",
           "reference": "He teaches in London."
        },
        "2": {
           "prompt": "### Task: classify news\n\n### News: Serena takes China title Serena Williams got back
               to winning ways with victory over US Open champion Svetlana Kuznetsova in the final of 
               the China Open on Sunday.\n\n### Label: ",
           "reference": "Sports"
        }
    }

    The function outputs a list of dictionaries, one per example with the following items:
        "id":
            Unique ID of the example
        "prompt": 
            Prompt text
        "prompt_ids":
            Prompt token IDs
            List of length `seq_len`
        "task":
            Task to perform (answer question, simplify text, or classify news)
        "attention_mask":
            Specifies which token positions to attend to (hides padding)
            List of length `seq_len`
        "reference":
            Reference text from the dataset

    """

    with open(filepath, "r", encoding="utf-8") as f:
        json_prompts = json.load(f)
    json_data = {int(k): v for k, v in json_prompts.items()}

    prompt_data = []

    for id, json_example in json_data.items():

        prompt = json_example["prompt"]
        example = {
            "id": id,
            "prompt": json_example["prompt"],
            "reference": json_example["reference"]
        }

        # Get the task to perform
        if prompt.startswith("### Task: answer question\n\n"):
            task = "answer question"
        elif prompt.startswith("### Task: simplify text\n\n"):
            task = "simplify text"
        elif prompt.startswith("### Task: classify news\n\n"):
            task = "classify news"
        else:
            raise ValueError(
                f"Unable to identify the task to perform from input prompt:\n{prompt}\n"
                "\nValid tasks are: 'follow instructions', 'simplify text', 'classify news'"
            )
        example["task"] = task

        # Tokenize the prompt and truncate it if needed
        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = prompt_ids[:seq_len]

        # Create the attention mask
        attention_mask = [1] * len(prompt_ids)

        # Pad prompt sequence and attention mask to `seq_len`
        if len(prompt_ids) < seq_len:
            pad_len = seq_len - len(prompt_ids)
            prompt_ids += [pad_token] * pad_len
            attention_mask += [0] * pad_len

        example["prompt_ids"] = prompt_ids
        example["attention_mask"] = attention_mask

        prompt_data.append(example)

    return prompt_data


def dump_responses(model_responses, references, filepath):
    """
    Write prompts, model responses and reference answers to a text file.
    """

    formatted = []
    for id in model_responses.keys():
        lines = f"\n{80 * '='}\n"
        lines += f"id: {id}\n{40 * '-'}\n"
        lines += f"{model_responses[id]}\n{40 * '-'}\n"
        lines += f"Reference: {references[id]}"
        formatted.append(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(formatted))


def test_prompts(project_root, model_size):
    """
    Test example prompts provided in a JSON file.
    
    """

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")
    
    # Load the prompts file
    prompts_fn = os.path.join(project_root, f"gpt2_{model_size}", "tests", "example_prompts.json")
    if not os.path.isfile(prompts_fn):
        raise FileNotFoundError(f'Unable to find JSON prompts file {prompts_fn}')
    
    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token = 50256

    print(f">> Loading prompts file {prompts_fn}")
    prompt_data = get_prompt_data(prompts_fn, tokenizer)

    # Load the model with LoRA adapters
    model_dir = os.path.join(project_root, f"gpt2_{model_size}", "trained_models")
    model_name = "lora_adapters"

    print(f">>Loading model `{model_name}` from directory {model_dir}")
    model = load_gpt2_model(model_dir, model_name)

    lora_tasks = model.lora_config["tasks"]

    model_responses = {}
    annotations = {}

    for example in prompt_data:

        task = example["task"]
        print(f"Prompt id: {example['id']}    Task: {task}")

        # Get the index of the adapter trained for the task
        adapter = lora_tasks.index(task)

        if task == "simplify text":
            sampling_params = {"method": "top_k", "temperature": 0.8, "top_k": 20}
        else:
            sampling_params = {"method": "greedy"}

        model_inputs = {
            "input_ids": tf.constant([example["prompt_ids"]], dtype=tf.int32),
            "attention_mask": tf.constant([example["attention_mask"]], dtype=tf.int32),
            "adapter": tf.constant([adapter], dtype=tf.int32)
        }
        sampling_params = [sampling_params]

        model_outputs = generate_text(
            model,
            model_inputs=model_inputs,
            output_len=100,
            sampling_params=sampling_params
        )
    
        # We work with batches of one element.
        tokens_out = model_outputs[0]

        # Truncate the list of tokens before the first pad token,
        # which marks the end of the model response
        tokens_out = tokens_out[:tokens_out.index(eos_token)]
        model_responses[id] = tokenizer.decode(tokens_out)

        annotations[id] = example["reference"]

    responses_fn = os.path.join(
        project_root, f"gpt2_{model_size}", "tests", "lora_responses.txt"
    )
    print(f"Writing model responses to file {responses_fn}")
    dump_responses(model_responses, annotations, responses_fn)


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
        help="GPT-2 model size, one of ('124M', '355M', '774M', '1542M')",
        type=str,
        default="124M"
    )
    args = parser.parse_args()

    test_prompts(
        args.project_root,
        args.model_size
    )

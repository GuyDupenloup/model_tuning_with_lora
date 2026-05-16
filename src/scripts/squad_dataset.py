# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import numpy as np
from datasets import load_dataset
import tiktoken
from utils.dataset_utils import write_dataset_tfrecords


def tokenize_example(tokenizer, context_text, question_text, answer_text, seq_len, pad_token=50256):

    header_1 = tokenizer.encode("### Task: answer question\n\n### Context: ")
    context_ids = tokenizer.encode(context_text)
    header_2 = tokenizer.encode("\n\n### Question: ")
    question_ids = tokenizer.encode(question_text)
    header_3 = tokenizer.encode("\n\n### Answer: ")
    answer_ids = tokenizer.encode(answer_text) + [pad_token]

    # Truncate the context to the maximum length it can take
    max_context_len = seq_len - len(header_1 + header_2 + question_ids + header_3 + answer_ids)
    assert max_context_len > 1
    truncated = len(context_ids) > max_context_len
    context_ids = context_ids[:max_context_len]

    # Create prompt, attention mask, and loss mask
    prefix = header_1 + context_ids + header_2 + question_ids + header_3
    prompt = prefix + answer_ids

    attention_mask = [1] * len(prompt)
    loss_mask = [0] * len(prefix) + [1] * len(answer_ids)

    # Pad to seq_len
    if len(prompt) < seq_len:
        pad_len = seq_len - len(prompt)
        prompt += [pad_token] * pad_len
        attention_mask += [0] * pad_len
        loss_mask += [0] * pad_len

    return prompt, attention_mask, loss_mask, truncated


def tokenize_dataset(dataset, tokenizer, seq_len):

    prompts = []
    loss_masks = []
    attention_masks = []
    truncated_prompts = 0

    for example in dataset:
  
        # Some examples have large numbers of leading/trailing blanks.
        context = example["context"].strip()
        question = example["question"].strip()
        answer = example["answers"]["text"][0].strip()  # We take the first answer.

        prompt_x, attention_mask_x, loss_mask_x, truncated_x = tokenize_example(
            tokenizer, context, question, answer, seq_len
        )

        prompts.append(prompt_x)
        attention_masks.append(attention_mask_x)
        loss_masks.append(loss_mask_x)
        truncated_prompts += int(truncated_x)

    print(f"Truncated prompts: {truncated_prompts}")

    # Wrap outputs in dictionary
    return {
            "input_ids": np.array(prompts, dtype=np.int32),
            "attention_mask": np.array(attention_masks, dtype=np.int32),
            "loss_mask": np.array(loss_masks, dtype=np.int32)
        }


def parse_and_write_dataset(project_root):

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")
    
    dataset_name = "squad"
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Loading dataset `{dataset_name}`")
    dataset = load_dataset(dataset_name)

    train_set = dataset["train"]
    val_set = dataset["validation"]

    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(val_set)
    
    # Load GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Input sequence length
    seq_len = 900
    print(f"\ninput sequence length: {seq_len}")

    print("\nTokenizing training set")
    print(f"Examples: {train_size}")
    train_data = tokenize_dataset(train_set, tokenizer, seq_len)

    print("\nTokenizing validation set")
    print(f"Examples: {val_size}")
    val_data = tokenize_dataset(val_set, tokenizer, seq_len)

    print("\nTokenizing test set (same as validation set)")
    print(f"Examples: {val_size}")
    test_data = tokenize_dataset(val_set, tokenizer, seq_len)

    print("\nSaving dataset files")
    metadata = {
        "dataset_name": dataset_name,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "seq_len": seq_len
    }

    write_dataset_tfrecords(dataset_dir, metadata, train_data, val_data, test_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--project_root",
        help="Project root directory",
        required=True,
        type=str,
        default=None
    )

    args = parser.parse_args()
    parse_and_write_dataset(args.project_root)
 
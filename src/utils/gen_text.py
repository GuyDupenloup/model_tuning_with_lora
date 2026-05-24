# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import numpy as np
import tensorflow as tf


def sample_next_token(logits, sampling_method='greedy', temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample the next token from a language model's logits using different sampling methods.

    Parameters:
        logits: 2D numpy array of model logits for the current step (batch_size, vocab_size)
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider (only for top_k sampling)
        top_p: float in (0,1], cumulative probability threshold for nucleus sampling (top-p)

    Returns:
        next_token_ids: 1D numpy array of sampled token indices (batch_size,)
    """
    batch_size, vocab_size = logits.shape
    next_token_ids = np.zeros(batch_size, dtype=int)

    def softmax(x):
        x = x.astype(np.float64)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    for i in range(batch_size):
        if sampling_method == "greedy":
            next_token_ids[i] = np.argmax(logits[i])

        elif sampling_method == "temperature":
            probs = softmax(logits[i] / temperature)
            next_token_ids[i] = np.random.choice(vocab_size, p=probs)

        elif sampling_method == "top_k":
            logits_i = logits[i] / temperature
            top_k_indices = np.argpartition(logits_i, -top_k)[-top_k:]
            top_k_logits = logits_i[top_k_indices]
            top_k_probs = softmax(top_k_logits)
            next_token_ids[i] = np.random.choice(top_k_indices, p=top_k_probs)

        elif sampling_method == "top_p":
            probs = softmax(logits[i] / temperature)
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum_probs, top_p)
            cutoff_idx = max(1, cutoff_idx)
            top_p_indices = sorted_indices[:cutoff_idx]
            top_p_probs = sorted_probs[:cutoff_idx]
            top_p_probs = top_p_probs / np.sum(top_p_probs)
            next_token_ids[i] = np.random.choice(top_p_indices, p=top_p_probs)

    return next_token_ids


def check_next_token_sampling_params(sampling_method, temperature, top_k, top_p):
    if sampling_method not in ("greedy", "temperature", "top_k", "top_p"):
        raise ValueError("Supported sampling methods are 'greedy', 'temperature', 'top_k', and 'top_p'")

    if sampling_method in ("temperature", "top_k", "top_p"):
        if temperature <= 0:
            raise ValueError("temperature argument must be > 0")

    if sampling_method == "top_k":
        if top_k < 1:
            raise ValueError("top-k argument must be >= 1")

    if sampling_method == "top_p":
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top-p argument must be > 0.0 and <= 1.0")
        

def generate_text(
    model,
    model_inputs,
    output_len,
    sampling_method="top_k",
    temperature=0.8,
    top_k=20,
    top_p=1.0,
):
    """
    Generates output texts given a list of input prompts and a max number of output tokens.

    Arguments:
        model: GPT-2 Keras model
        prompts: List of input texts to start from
        output_len: Number of tokens to generate
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider for top-k sampling
        top_p: float in (0.0, 1.0], cumulative probability threshold for nucleus sampling (top-p)

    Returns:
        List of output texts, one for each input prompt
    """
    check_next_token_sampling_params(sampling_method, temperature, top_k, top_p)

    eos_token = 50256

    tokens_out = model_inputs["input_ids"].numpy()
    masks_out  = model_inputs["attention_mask"].numpy()
    adapter_selector = model_inputs["adapter_selector"]

    batch_size = tokens_out.shape[0]
    finished = [False] * batch_size

    for _ in range(output_len):

        # Exit loop if all sequences are done
        if all(finished):
            break

        hidden_states = model({
            "input_ids": tf.constant(tokens_out, dtype=tf.int32),
            "attention_mask": tf.constant(masks_out, dtype=tf.int32),
            "adapter_selector": adapter_selector
        }).numpy()

        # Get the indices of the last tokens before padding
        last_token_indices = masks_out.sum(axis=1) - 1

        logits = hidden_states[range(batch_size), last_token_indices, :]  # (batch, vocab_size)

        predicted_tokens = sample_next_token(
            logits,
            sampling_method=sampling_method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        pad_starts = masks_out.sum(axis=1)
        for b in range(batch_size):
            if finished[b]:
                continue
            if predicted_tokens[b] == eos_token:
                finished[b] = True
                continue
            tokens_out[b, pad_starts[b]] = predicted_tokens[b]
            masks_out[b, pad_starts[b]] = 1

    return tokens_out

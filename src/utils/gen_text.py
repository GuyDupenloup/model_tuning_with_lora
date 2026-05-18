# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import numpy as np
import tensorflow as tf
import tiktoken


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
    model: "tf.keras.Model",
    prompts: list,
    output_len: int,
    sampling_method: str = "top_k",
    temperature: float = 0.8,
    top_k: int = 20,
    top_p: float = 1.0,

) -> list:
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

    pad_token = 50256
    tokenizer = tiktoken.get_encoding('gpt2')

    # Encode prompts
    tokens_out = [tokenizer.encode(prompt) for prompt in prompts]

    max_context_len = 1024  # GPT-2 context limit (adjust if needed)

    for _ in range(output_len):
        batch_size = len(tokens_out)
        input_ids = []
        attention_masks = []

        for tokens in tokens_out:
            current_tokens = tokens[-max_context_len:]
            seq_len = len(current_tokens)

            padded_input = current_tokens
            attention_mask = [1] * seq_len

            input_ids.append(padded_input)
            attention_masks.append(attention_mask)

        # Run model
        inputs = {
            'input_ids': tf.constant(input_ids, dtype=tf.int32),
            'attention_mask': tf.constant(attention_masks, dtype=tf.int32)
        }
        hidden_states = model(inputs)

        # Sample next tokens
        next_tokens = []
        for i in range(batch_size):
            current_tokens = tokens_out[i][-max_context_len:]
            last_token_index = len(current_tokens) - 1

            logits = hidden_states[i, last_token_index, :]
            logits = np.squeeze(logits.numpy())

            next_token = sample_next_token(
                logits[np.newaxis, :],
                sampling_method=sampling_method,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            next_tokens.append(next_token[0])

        # Append tokens
        for i in range(batch_size):
            tokens_out[i].append(next_tokens[i])

        # Early stop if all sequences hit EOS ---
        if all(t == pad_token for t in next_tokens):
            break

    return tokens_out

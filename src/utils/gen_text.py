# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import numpy as np
import tensorflow as tf

def sample_next_token(logits, params):
    """
    Samples the next token from a language model's logits 
    using different sampling methods.

    Arguments:
        logits:
            Model logits for the current step, 1D numpy array with shape (vocab_size,).
        params:
            Sampling parameters, a dictionary with the following items:
                "method":
                    The sampling method to use.
                    A string, one of ('greedy', 'temperature', 'top_k', 'top_p').
                "temperature":
                    Temperature scaling.
                    A float > 0, defaults to 0.8 if not specified.
                "top_k":
                    Number of top-k tokens to consider ('top_k' method).
                    An integer >= 1, defaults to 20 if not specified.
                "top_p":
                    Cumulative probability threshold for nucleus sampling ('top_p' method).
                    A float in (0, 1], defaults to 0.9 if not specified.
 
    Returns:
        An integer, index in the logits of the sampled next token.
    """

    def softmax(x):
        x = x.astype(np.float64)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # Get parameter values applying defaults
    method, temperature, top_k, top_p = (
        params["method"],
        params.get("temperature", 0.8),
        params.get("top_k", 20),
        params.get("top_p", 0.9),
    )

    if method == "greedy":
        next_token = np.argmax(logits)

    elif method == "temperature":
        probs = softmax(logits / temperature)
        vocab_size = logits.shape[0]
        next_token = np.random.choice(vocab_size, p=probs)

    elif method == "top_k":
        scaled_logits = logits / temperature
        
        # Select the top-k token indices (unordered)
        top_k_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
        top_k_logits = scaled_logits[top_k_indices]
        
        # Convert to probabilities and sample
        top_k_probs = softmax(top_k_logits)
        next_token = np.random.choice(top_k_indices, p=top_k_probs)

    elif method == "top_p":
        probs = softmax(logits / temperature)
        
        # Sort tokens by probability in descending order
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Find the cutoff index where cumulative probability exceeds top_p
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = max(1, cutoff_idx)  # Keep at least one token

        # Sample from the nucleus
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_probs = top_p_probs / np.sum(top_p_probs)  # Renormalize
        next_token = np.random.choice(top_p_indices, p=top_p_probs)

    return next_token


def generate_token_sequences(
    model,
    model_inputs,
    output_len,
    sampling_params
):
    """
    Generates output tokens given a batch of sequences of input tokens
    (the prompts) and a maximum number of output tokens to generate.

    Arguments:
        model:
            GPT-2 Keras model

        model_inputs:
            A dictionary with the following items:
                "input_ids":
                    Input token sequences
                    A tensor with shape (batch, seq_len)
               "attention_mask":
                    Attention masks specifying which token positions to attend to
                    A tensor with shape (batch, seq_len)
                "adapter":
                    Index of the LoRA adapter to activate
                    Present only if the model has adapters and one of them is activated 

        sampling_params:
            A list of sampling parameter dictionaries, one per sequence in the batch
            See sample_next_token() for the dictionary format

        output_len:
            Maximum length of output token sequences

    Returns:
        Full token sequences (prompts + generated tokens)
        A numpy array with shape (batch, seq_len)
    """

    # tiktoken has no dedicated <EOS> token. The first pad token
    # in an input sequence marks the end of the sequence.
    eos_token = 50256

    tokens_out = model_inputs["input_ids"].numpy()
    masks_out  = model_inputs["attention_mask"].numpy()

    batch_size = tokens_out.shape[0]
    finished = [False] * batch_size

    for _ in range(output_len):

        # Exit loop if all sequences are done
        if all(finished):
            break

        hidden_states = model({
            "input_ids": tf.constant(tokens_out, dtype=tf.int32),
            "attention_mask": tf.constant(masks_out, dtype=tf.int32),
            "adapter": model_inputs["adapter"]
        }).numpy()

        # Get the indices of the last tokens before padding
        last_token_indices = masks_out.sum(axis=1) - 1

        logits = hidden_states[range(batch_size), last_token_indices, :]  # (batch, vocab_size)

        pad_starts = masks_out.sum(axis=1)

        for b in range(batch_size):
            if finished[b]:
                continue

            # Sample the next token
            next_token = sample_next_token(logits[b], sampling_params[b])

            if next_token == eos_token:
                finished[b] = True
                continue

            tokens_out[b, pad_starts[b]] = next_token
            masks_out[b, pad_starts[b]] = 1

    return tokens_out

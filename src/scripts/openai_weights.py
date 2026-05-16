# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
from pathlib import Path
import argparse
import numpy as np
from transformers import TFGPT2LMHeadModel
from utils.model_utils import create_gpt2_language_model, print_model_variables
from transformers import TFGPT2LMHeadModel


def save_gpt2_openai_weights(model_size, filepath):

    """
    Gets OpenAI GPT2 weights and save them to numpy arrays.

    To get the weights for a model of a given size, a model of the same size
    is instantiated from the Hugging Face 'transformers' package.
    As the architectures of the two models are identical, their trainable variables
    match one-to-one, although their names are different (more detail available 
    in the 'gpt2_model_from_research_papers' project).

    We create the following dictionary:
    - Each item is a trainable variable of the two models.
    - The key is the name of the variable in our model.
    - The value are the weights of the variable in the Hugging Face model.
    
    Then, the dictionary is saved in a .npz file.

    Using this name-based mechanism, OpenAI weights can be loaded in our models 
    that include LoRA layers (see function `load_openai_gpt2_weights()` 
    in model_utils.py).
    """

    model = create_gpt2_language_model(model_size, name='gpt2_lm')

    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    # Get Hugging Face model of the same size
    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    assert len(model.trainable_variables) == len(hf_model.trainable_variables)

    var_weights = {}
    for i in range(len(model.trainable_variables)):
        var = model.trainable_variables[i]
        hf_var = hf_model.trainable_variables[i]

        # Use var.path if it exists (in this case, var.name is just a leaf name)
        var_name = var.path if hasattr(var, 'path') else var.name

        # If the variable name does not have a ':0' suffix, add it.
        if var_name[-2:] != ':0':
            var_name += ':0'
        
        weights = hf_var.numpy()
        var_weights[var_name] = np.squeeze(weights)

    np.savez(filepath, **var_weights)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model_size',
        help='GPT-2 model size (124M, 355M, 744M, or 1.56B)',
        type=str,
        default='124M'
    )
    parser.add_argument(
        '--weights_filepath',
        help='OpenAI weights filepath (extension must be .npz)',
        type=str,
        default='./project/weights/openai_weights_124M.npz',
    )

    args = parser.parse_args()

    # Check that the directory where to write the weights file exists
    path = Path(args.weights_filepath)
    if not os.path.isdir(path.parent):
        raise FileNotFoundError(f'Unable to write weights file. Directory {path.parent} does not exist.')
    if path.suffix != '.npz':
        raise ValueError('The weights file extension must be .npz')

    save_gpt2_openai_weights(args.model_size, args.weights_filepath)

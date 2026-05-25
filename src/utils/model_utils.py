# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
from tabulate import tabulate
import numpy as np
import tensorflow as tf
from models.gpt2_language_model import GPT2LanguageModel


def get_gpt2_model_config(model_size):
    """
    Gets model configuration parameters for each of OpenAI's model sizes.
    """
    model_configs = {
         "124M": {"vocab_size": 50257,  "max_seq_len": 1024, "d_model": 768,  "n_layers": 12, "n_heads": 12},
         "355M": {"vocab_size": 50257,  "max_seq_len": 1024, "d_model": 1024, "n_layers": 24, "n_heads": 16},
         "774M": {"vocab_size": 50257,  "max_seq_len": 1024, "d_model": 1280, "n_layers": 36, "n_heads": 20},
        "1542M": {"vocab_size": 50257,  "max_seq_len": 1024, "d_model": 1600, "n_layers": 48, "n_heads": 25}
    }

    supported_sizes = list(model_configs.keys())
    if model_size not in supported_sizes:
        raise ValueError(f"Supported model sizes are {supported_sizes}. Received `{model_size}`")

    config = model_configs[model_size]
    config["size"] = model_size

    return config


def load_openai_gpt2_weights(model, filepath):
    """
    Loads OpenAI weights into a model using a .npz file created 
    by function save_openai_gpt2_weights() in model_utils.py.

    A .npz file contains a dictionary:
    - Each item is a trainable variable of the model.
    - The key is the name of the variable.
    - The value are the OpenAI weights of the variable.
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Unable to find OpenAI GPT-2 weights file `{filepath}`")
    
    with np.load(filepath) as loaded:
        # Convert to a dictionary
        var_weights = {key: loaded[key] for key in loaded.files}

    # Check that the number of non-LoRA variables is equal
    # to the number of variables (items) in the weights dict
    num_vars = 0
    for var in model.trainable_variables:
        # Use var.path if it exists (in this case, var.name is just a leaf name)
        var_name = var.path if hasattr(var, "path") else var.name

        # Exclude variables of LoRA layers
        if  not "lora" in var_name:
            num_vars += 1
    assert len(var_weights) == num_vars

    assigned = 0
    for var in model.trainable_variables:
        var_name = var.path if hasattr(var, "path") else var.name

        # Replace the model name by "gpt2_lm", which is the name 
        # that was used to create the weights dict.
        parts = var_name.split("/")
        parts[0] = "gpt2_lm"
        var_name = "/".join(parts)

        # Variable names in the weights dict have a ":0" suffix. This suffix
        # may or may not be present depending on Tensorflow versions.
        # Add it if needed to match the keys of the weights dict.
        if var_name[-2:] != ":0":
            var_name += ":0"

        # Assign OpenAI weights to the variable
        if var_name in var_weights:
            weights = var_weights[var_name]
            var.assign(weights)
            assigned += 1

    # Check that all the keys of the weights dict have been used
    assert assigned == len(var_weights)


def create_gpt2_language_model_from_config(
        model_config,
        lora_config=None,
        name="gpt2_lm"
    ):
    
    """
    Creates a GPT-2 language model (base GPT-2 model with a language modelling output layer).
    OpenAI's pretrained weights are loaded in the model.

    Arguments:
        model_size:
            '124M', '355M', '774M', or '1542M'.
        lora_config:
            Optional LoRA configuration, a dictionary.
            If present, keys must include 'rank' and 'alpha'.
            If not present, LoRA layers are inactive.
        dropout_rate:
            Dropout rate for dropout layers.
            Applied after embeddings and after each transformer sub-layer.
            Optional, defaults to 0.1

    Returns:
        A tf.keras.models.Model object.
        Pretrained GPT-2 language model of the specified size.
    """
        
    model = GPT2LanguageModel(
        model_config, 
        lora_config=lora_config,
        name=name
    )

    max_seq_len = model_config["max_seq_len"]
    vocab_size = model_config["vocab_size"]

    if lora_config is not None:
        # Cycle through all adapters
        num_adapters = lora_config["num_adapters"]
        for i in range(num_adapters):
            inputs = {
                "input_ids": tf.random.uniform((1, max_seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
                "attention_mask": tf.ones((1, max_seq_len), dtype=tf.int32),
                "adapter": tf.constant([i], dtype=tf.int32)
            }
            _ = model(inputs)
    else:
        inputs = {
            "input_ids": tf.random.uniform((1, max_seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
            "attention_mask": tf.random.uniform((1, max_seq_len), minval=0, maxval=2, dtype=tf.int32)
        }
        _ = model(inputs)

    return model


def create_gpt2_language_model(
    model_size,
    lora_config=None,
    name="gpt2_lm"
):
    model_config = get_gpt2_model_config(model_size)
    
    model = create_gpt2_language_model_from_config(
        model_config,
        lora_config=lora_config,
        name=name
    )

    return model


def load_gpt2_model(model_dir, model_name):
    """
    Reloads a GPT2 model that was saved using the save() method of the model.

    Arguments:
        model_dir:
            Path to the directory where the model files were saved.
        model_name:
            Name of the model.

    Two files must be present in the directory:
        <model_name>.json
            Model configuration file, including LoRA layers if any.
       <model_name>.weights.h5
            Model weights.

    The model is recreated using the JSON configuration file
    and the weights file is loaded into it.
    """
    config_fn = os.path.join(model_dir, f"{model_name}.json")
    if not os.path.isfile(config_fn):
        raise FileNotFoundError(f"Unable to find model config file {config_fn}")
    
    weights_fn = os.path.join(model_dir, f"{model_name}.weights.h5")
    if not os.path.isfile(config_fn):
        raise FileNotFoundError(f"Unable to find model weights file {weights_fn}")

    # Load the config file
    with open(config_fn, "r") as f:
        config = json.load(f)

    model_config = config["model_config"]
    lora_config = config.get("lora_config", None)

    if lora_config is not None:
        lora_config["rank"] = tuple(lora_config["rank"])
        lora_config["alpha"] = tuple(lora_config["alpha"])
        
    model = create_gpt2_language_model_from_config(
        model_config,
        lora_config=lora_config
    )
    model.load_weights(weights_fn)

    return model


def print_model_variables(model, verbose=False):
    """
    Prints the trainable/non-trainable variables of a model
    (names, shapes, number of parameters)
    """

    def print_vars(model_size, var_list, var_type):
        if verbose:
            print("\n" + "=" * 80)
            print(f"  {var_type} variables of model `{model_size}`")
            print("=" * 80 + "\n")

        total_params = 0
        if len(var_list) > 0:
            data = []
            total_params = 0
            for var in var_list:
                var_name = var.path if hasattr(var, "path") else var.name
                num_params = int(np.prod(var.shape))
                total_params += num_params
                data.append([f"{var_name}", f"{var.shape}", f"{num_params:,.0f}"])

            if verbose:
                headers = ["Variable", "Shape", "#Params"]
                print(tabulate(data, headers=headers, tablefmt="pipe", colalign=("left", "center", "right")))

        return total_params

    model_size = model.model_config["size"]
    num_params = print_vars(model_size, model.trainable_variables, "Trainable")
    print(f"Trainable parameters: {num_params:,}")

    num_params = print_vars(model_size, model.non_trainable_variables, "Non-trainable")
    print(f"Non-trainable parameters: {num_params:,}")

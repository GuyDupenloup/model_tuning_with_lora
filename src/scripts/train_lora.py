# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import tensorflow as tf

from utils.model_utils import create_gpt2_language_model, print_model_variables, load_openai_gpt2_weights
from utils.dataset_utils import create_data_loaders


def evaluate(model, test_ds):

    loss, accuracy, perplexity = model.evaluate(test_ds, verbose=1)
    print(f"Test set evaluation:")
    print(f"  loss: {loss:.4f}")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  perplexity: {perplexity:.4f}")


def train_squad_adapter(model, data_loaders):
    """
    Train the question answering adapter on squad dataset
    """

    model.set_dropout_rate(0.025)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    epochs = 2

    # optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    # model.compile(optimizer=optimizer)
    # epochs = 2

    train_ds, val_ds, test_ds = data_loaders

    # Train the model
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    evaluate(model, test_ds)


def train_wikilarge_adapter(model, data_loaders):
    """
    Train the text simplification adapter on wikilarge dataset
    """

    model.set_dropout_rate(0.1)

    steps_per_epoch = 7741
    epochs = 6
    total_steps = steps_per_epoch * epochs

    warmup_steps = 1000
    peak_lr = 1e-4
    final_lr = 1e-5

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps, 
        alpha=final_lr / peak_lr,
        warmup_target=peak_lr,
        warmup_steps=warmup_steps
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)

    train_ds, val_ds, test_ds = data_loaders

    # Train the model
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    evaluate(model, test_ds)


def train_ag_news_adapter(model, data_loaders):
    """
    Train the news classification adapter on ag_news dataset
    """

    model.set_dropout_rate(0.1)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    train_ds, val_ds, test_ds = data_loaders

    # Train the model
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2
    )

    evaluate(model, test_ds)


def train_model(project_root, model_size):
    """"
    Create a GPT-2 model with 3 LoRA adapters, load pretrained weights,
    and train each adapter.
    """

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")
    
    # Create the training directory if it does not exist
    train_dir = os.path.join(project_root, f"gpt2_{model_size}", "trained_models")
    os.makedirs(train_dir, exist_ok=True)

    # Get the path to pretrained weights
    openai_filepath = os.path.join(
        project_root,
        f"gpt2_{model_size}", 
        "pretrained_weights",
        f"openai_weights_gpt2_{model_size}.npz"
    )

    # Set LoRA adapters indices
    squad_adapter = 0
    wikilarge_adapter = 1
    ag_news_adapter = 2

    # Create data loaders
    data_loaders = {}
    dataset_root = os.path.join(project_root, "datasets")

    data_loaders["squad"], _ = create_data_loaders(
        os.path.join(dataset_root, "squad"),
        batch_size=16,
        adapter=squad_adapter
    )
    data_loaders["wikilarge"], _ = create_data_loaders(
        os.path.join(dataset_root, "wikilarge"),
        batch_size=16,
        adapter=wikilarge_adapter
    )
    data_loaders["ag_news"], _ = create_data_loaders(
        os.path.join(dataset_root, "ag_news"),
        batch_size=16,
        adapter=ag_news_adapter
    )
    
    # Create GPT-2 model and load pretrained weights
    print(f"\nCreating gpt-2 model `{model_size}`")
    lora_config = {
        "num_adapters": 3,
        "rank": (16, 8, 8),
        "alpha": (32, 16, 16),
        "tasks": ("answer question", "simplify text", "classify news")
    }
    model = create_gpt2_language_model(
        model_size,
        lora_config=lora_config
    )
    load_openai_gpt2_weights(model, openai_filepath)
    
    # Train squad adapter
    print(f"\nTraining LoRA adapter #{squad_adapter} on `squad` dataset")
    model.lora_freeze(squad_adapter)  # Freeze all layers but question answering adapter
    print_model_variables(model)
    train_squad_adapter(model, data_loaders["squad"])

    # Train wikilarge adapter
    print(f"\nTraining LoRA adapter #{wikilarge_adapter} on `wikilarge` dataset")
    model.lora_freeze(wikilarge_adapter)    # Freeze all layers but text simplification adapter
    print_model_variables(model)
    train_wikilarge_adapter(model, data_loaders["wikilarge"])

    # Train ag_news adapter
    print(f"\nTraining LoRA adapter #{ag_news_adapter} on `ag_news` dataset")
    model.lora_freeze(ag_news_adapter)    # Freeze all layers but news classification adapter
    print_model_variables(model)
    train_ag_news_adapter(model, data_loaders["ag_news"])

    # Save the model with trained adapters
    model.save(train_dir, "lora_adapters")


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

    train_model(args.project_root, args.model_size)

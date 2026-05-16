# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf

from utils.model_utils import create_gpt2_language_model, print_model_variables, load_openai_gpt2_weights
from utils.dataset_utils import create_data_loaders

def evaluate(model, test_ds):

    loss, accuracy, perplexity = model.evaluate(test_ds, verbose=1)
    print(f"Test set evaluation:")
    print(f"  loss: {loss:.4f}")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  perplexity: {perplexity:.4f}")


def train_alpaca_adapter(model, data_loaders):

    model.set_dropout_rate(0.1)

    steps_per_epoch = 2600
    epochs = 7
    total_steps = steps_per_epoch * epochs 

    warmup_steps = 1000
    peak_lr = 1e-3   # 5e-4
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
    start_time = timer()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    end_time = timer()

    train_run_time = int(end_time - start_time)
    print("Training runtime: " + str(timedelta(seconds=train_run_time))) 

    evaluate(model, test_ds)


def train_wikilarge_adapter(model, data_loaders):

    model.set_dropout_rate(0.1)

    steps_per_epoch = 7741
    epochs = 6
    total_steps = steps_per_epoch * epochs  # 30964

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
    start_time = timer()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    end_time = timer()

    train_run_time = int(end_time - start_time)
    print("Training runtime: " + str(timedelta(seconds=train_run_time))) 

    evaluate(model, test_ds)


def train_ag_news_adapter(model, data_loaders):

    model.set_dropout_rate(0.1)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    train_ds, val_ds, test_ds = data_loaders

    # Train the model
    start_time = timer()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2
    )
    end_time = timer()

    train_run_time = int(end_time - start_time)
    print("Training runtime: " + str(timedelta(seconds=train_run_time))) 

    evaluate(model, test_ds)


def train_model(project_root, model_size):

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")
    
    train_dir = os.path.join(project_root, f"trained_lora_{model_size}")
    os.makedirs(train_dir, exist_ok=True)

    openai_filepath = os.path.join(project_root, "openai_weights", f"openai_weights_gpt2_{model_size}.npz")
    
    data_loaders = {}
    dataset_root = os.path.join(project_root, "datasets")

    data_loaders["alpaca"], _ = create_data_loaders(
        os.path.join(dataset_root, "alpaca"),
        batch_size=16
    )
    data_loaders["wikilarge"], _ = create_data_loaders(
        os.path.join(dataset_root, "wikilarge"),
        batch_size=16
    )
    data_loaders["ag_news"], _ = create_data_loaders(
        os.path.join(dataset_root, "ag_news"),
        batch_size=16
    )

    lora_config = {
        "num_adapters": 3,
        "rank": (8, 8, 8),
        "alpha": (16, 16, 16),
        "tasks": {
            "answer questions": 0,
            "simplify text": 1,
            "classify news": 2
        }
    }

    print(f"\nCreating gpt-2 model `{model_size}`")
    model = create_gpt2_language_model(
        model_size,
        lora_config=lora_config
    )
    load_openai_gpt2_weights(model, openai_filepath)

    alpaca_adapter = 0
    wikilarge_adapter = 1
    ag_news_adapter = 2

    print(f"\nTraining LoRA adapter #{alpaca_adapter} on `alpaca`")
    model.select_adapter(alpaca_adapter)
    model.lora_freeze()

    print_model_variables(model)

    train_alpaca_adapter(
        model,
        data_loaders=data_loaders["alpaca"]
    )
    model.save(train_dir, "alpaca_lora_32")

    # print(f"\nTraining LoRA adapter #{wikilarge_adapter} on `wikilarge`")
    # model.select_adapter(wikilarge_adapter)
    # model.lora_freeze()

    # print_model_variables(model)

    # train_wikilarge_adapter(
    #     model,
    #     data_loaders=data_loaders["wikilarge"]
    # )

    # print(f"\nTraining LoRA adapter #{ag_news_adapter} on `ag_news`")
    # model.select_adapter(ag_news_adapter)
    # model.lora_freeze()

    # print_model_variables(model)

    # train_ag_news_adapter(
    #     model,
    #     data_loaders=data_loaders["ag_news"]
    # )

    # model.save(train_dir, "lora")
    
    
# project_root = "/content/drive/MyDrive/project"
project_root = "../../project"
model_size = "124M"

from google.colab import drive
drive.mount('/content/drive')

train_model(
    project_root,
    model_size
)

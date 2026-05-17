# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
from timeit import default_timer as timer
from datetime import timedelta
import argparse
import tensorflow as tf

from utils.dataset_utils import create_data_loaders
from utils.model_utils import (
    create_gpt2_language_model, load_openai_gpt2_weights, load_gpt2_model
)

def evaluate(model, test_ds):

    loss, accuracy, perplexity = model.evaluate(test_ds, verbose=1)
    print(f'Test set evaluation:')
    print(f'  loss: {loss:.4f}')
    print(f'  accuracy: {accuracy:.4f}')
    print(f'  perplexity: {perplexity:.4f}')


def train(
    model,
    dropout_rate=None,
    learning_rate=None,
    epochs=None,
    data_loaders=None,
    train_dir=None
):

    model.set_dropout_rate(dropout_rate)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)

    log_dir = os.path.join(train_dir, 'tensorboard_log')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    train_ds, val_ds, test_ds = data_loaders

    # Train the model
    start_time = timer()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        # callbacks=[tensorboard_callback]
    )
    end_time = timer()

    train_run_time = int(end_time - start_time)
    print('Training runtime: ' + str(timedelta(seconds=train_run_time))) 

    evaluate(model, test_ds)


def train_baseline(model_size, data_loaders, train_dir, openai_filepath):
    
    print(f'\nCreating gpt-2 model `{model_size}`')
    model = create_gpt2_language_model(model_size)
    load_openai_gpt2_weights(model, openai_filepath)
    model.compile()

    print('\nEvaluating model on `squad` before tuning')
    evaluate(model, data_loaders['squad'][2])

    print('\nTraining model on `squad`')
    train(
        model,
        dropout_rate=0.2,
        learning_rate=5e-5,
        epochs=3,
        data_loaders=data_loaders['squad'],
        train_dir=train_dir
    )
    model.save(train_dir, 'squad_baseline')

    # #---------------------------------------------------------------

    # print(f"\n{'-' * 50}\n{'-' * 50}\n")

    # print(f'\nCreating GPT2-2 model `{model_size}`')
    # model = create_gpt2_language_model(model_size)
    # load_openai_gpt2_weights(model, openai_filepath)
    # model.compile()

    # print('\nEvaluating model on `wikilarge` before tuning')
    # evaluate(model, data_loaders['wikilarge'][2])

    # print('Training model on `wikilarge`')
    # train(
    #     model,
    #     dropout_rate=0.1,
    #     learning_rate=5e-5,
    #     epochs=2,
    #     data_loaders=data_loaders['wikilarge'],
    #     train_dir=train_dir
    # )
    # model.save(train_dir, 'wikilarge_baseline')

    # #---------------------------------------------------------------

    # print(f"\n{'-' * 50}\n{'-' * 50}\n")

    # print(f'\nCreating gpt-2 model `{model_size}`')
    # model = create_gpt2_language_model(model_size)
    # load_openai_gpt2_weights(model, openai_filepath)
    # model.compile()

    # print('\nEvaluating model on `ag_news` before tuning')
    # evaluate(model, data_loaders['ag_news'][2])

    # print('Training model on `ag_news`')
    # train(
    #     model,
    #     dropout_rate=0.1,
    #     learning_rate=1e-4,
    #     epochs=2,
    #     data_loaders=data_loaders['ag_news'],
    #     train_dir=train_dir
    # )
    # model.save(train_dir, 'ag_news_baseline')


def train_sequential(train_dir, data_loaders):

    # print(f'\nLoading model `squad_baseline` from directory {train_dir}')
    # model = load_gpt2_model(train_dir, "squad_baseline")
    # model.compile()

    # evaluate(model, data_loaders["squad"][2])

    # print("\nTraining `squad` baseline on `wikilarge`")
    # train(
    #     model,
    #     dropout_rate=0.1,
    #     learning_rate=5e-5,
    #     epochs=2,
    #     data_loaders=data_loaders["wikilarge"],
    #     train_dir=train_dir
    # )

    # print("\nRe-evaluating model on `squad`")
    # evaluate(model, data_loaders["squad"][2])

    #---------------------------------------------------------------

    print(f"\nLoading model `wikilarge_baseline` from directory {train_dir}")
    model = load_gpt2_model(train_dir, "wikilarge_baseline")
    model.compile()

    evaluate(model, data_loaders["wikilarge"][2])

    print("\nTraining `wikilarge` baseline on `squad`")
    train(
        model,
        dropout_rate=0.1,
        learning_rate=5e-5,
        epochs=2,
        data_loaders=data_loaders["squad"],
        train_dir=train_dir
    )

    print("\nRe-evaluating model on `wikilarge`")
    evaluate(model, data_loaders["wikilarge"][2])

    #---------------------------------------------------------------

    # print(f"\n{"-" * 50}\n{"-" * 50}\n")

    # print(f"\nLoading model `squad_baseline` from directory {train_dir}")
    # model = load_gpt2_model(train_dir, "squad_baseline")
    # model.compile()

    # evaluate(model, data_loaders["squad"][2])

    # print("\nTraining `squad` baseline on `ag_news`")
    # train(model,
    #     dropout_rate=0.1,
    #     learning_rate=1e-4,
    #     epochs=2,
    #     data_loaders=data_loaders["ag_news"],
    #     train_dir=train_dir
    # )

    # print("\nRe-evaluating model on `squad`")
    # evaluate(model, data_loaders["squad"][2])

    #---------------------------------------------------------------

    print(f"\n{"-" * 50}\n{"-" * 50}\n")

    print(f"\nLoading model `ag_news_baseline` from directory {train_dir}")
    model = load_gpt2_model(train_dir, "ag_news_baseline")
    model.compile()

    evaluate(model, data_loaders["ag_news"][2])

    print("\nTraining `ag_news` baseline on `squad`")
    train(model,
        dropout_rate=0.1,
        learning_rate=5e-5,
        epochs=2,
        data_loaders=data_loaders["squad"],
        train_dir=train_dir
    )

    print("\nRe-evaluating model on `ag_news`")
    evaluate(model, data_loaders["ag_news"][2])

    #---------------------------------------------------------------

    # print(f"\n{"-" * 50}\n{"-" * 50}\n")

    # print(f"\nLoading model `wikilarge_baseline` from directory {train_dir}")
    # model = load_gpt2_model(train_dir, "wikilarge_baseline")

    # model.compile()
    # evaluate(model, data_loaders["wikilarge"][2])

    # print("\nTraining `wikilarge` baseline on `ag_news`")
    # train(model,
    #     dropout_rate=0.1,
    #     learning_rate=1e-4,
    #     epochs=2,
    #     data_loaders=data_loaders["ag_news"],
    #     train_dir=train_dir
    # )

    # print("\nRe-evaluating model on `wikilarge")
    # evaluate(model, data_loaders["wikilarge"][2])

   #---------------------------------------------------------------

    print(f"\n{"-" * 50}\n{"-" * 50}\n")

    print(f"\nLoading model `ag_news_baseline` from directory {train_dir}")
    model = load_gpt2_model(train_dir, "ag_news_baseline")

    model.compile()
    evaluate(model, data_loaders["ag_news"][2])

    print("\nTraining `ag_news` baseline on `wikilarge`")
    train(model,
        dropout_rate=0.1,
        learning_rate=5e-5,
        epochs=2,
        data_loaders=data_loaders["wikilarge"],
        train_dir=train_dir
    )

    print("\nRe-evaluating model on `ag_news")
    evaluate(model, data_loaders["ag_news"][2])


def train_model(project_root, model_size="124M", run="baseline"):

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Unable to find project root directory {project_root}")

    train_dir = os.path.join(project_root, f"gpt2_{model_size}", "trained_models")
    os.makedirs(train_dir, exist_ok=True)

    openai_filepath = os.path.join(
        project_root,
        f"gpt2_{model_size}",
        "pretrained_weights",
        f"openai_weights_gpt2_{model_size}.npz"
    )

    data_loaders = {}
    dataset_root = os.path.join(project_root, "datasets")

    data_loaders["squad"], _ = create_data_loaders(
        os.path.join(dataset_root, "squad"),
        batch_size=16
    )
    data_loaders["wikilarge"], _ = create_data_loaders(
        os.path.join(dataset_root, "wikilarge"),
        batch_size=32
    )
    data_loaders["ag_news"], _ = create_data_loaders(
        os.path.join(dataset_root, "ag_news"),
        batch_size=32
    )

    if run == "baseline":
        train_baseline(model_size, data_loaders, train_dir, openai_filepath)
    elif run == "sequential":
        train_sequential(train_dir, data_loaders)
    else:
        raise ValueError(f"Invalid run mode `{run}`")
    

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
        help="Model size, one of '124M', '355M', '774M', '1542M'"
        type=str,
        default="124M"
    )
    parser.add_argument(
        "--run",
        help="Training run, either 'baseline' or 'sequential'"
        type=str,
        default="baseline"
    )
 
    args = parser.parse_args()
    train_model(
        args.project_root,
        model_size=args.model_size,
        run=args.run
    )
 
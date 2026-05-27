# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import shutil
import json
import numpy as np
import tensorflow as tf


def write_dataset_tfrecords(output_dir, metadata, train_data, val_data, test_data):
    """
    Write in output_dir directory:
        - Training, validation, and test data to TFRecords
        - Dataset metadata to JSON file
    """

    def write_tfrecord(data, filepath):
        
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]

        def _int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        num_examples = input_ids.shape[0]
        with tf.io.TFRecordWriter(filepath) as writer:
            for i in range(num_examples):
                feat = {
                    "input_ids": _int_feature(input_ids[i].astype(np.int64)),
                    "attention_mask": _int_feature(attention_mask[i].astype(np.int64)),
                    "loss_mask": _int_feature(loss_mask[i].astype(np.int64))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(example.SerializeToString())

    # Create TFRecords directory if it does not exist
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Write metadata to JSON file
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write TFRecords files
    write_tfrecord(train_data, filepath=os.path.join(output_dir, "train.tfrecord"))
    write_tfrecord(val_data, filepath=os.path.join(output_dir, "val.tfrecord"))
    write_tfrecord(test_data, filepath=os.path.join(output_dir, "test.tfrecord"))


def build_ds_pipeline(
    ds, seq_len, batch_size, adapter=None, shuffle=False, cache=False, buffer_size=1000
):
    """
    Build a tf.data.Dataset pipeline
    """

    def parse_and_wrap(example_proto):
        feature_spec = {
            "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([seq_len], tf.int64),
            "loss_mask": tf.io.FixedLenFeature([seq_len], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        return {
            "input_ids": tf.cast(parsed["input_ids"], tf.int32),
            "attention_mask": tf.cast(parsed["attention_mask"], tf.int32),
            "loss_mask": tf.cast(parsed["loss_mask"], tf.int32)
        }

    ds = ds.map(parse_and_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)

    if adapter is not None:
        ds = ds.map(
            lambda batch: {**batch, "adapter": tf.fill((batch_size,), adapter)},  # CHANGE
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_data_loaders(dataset_dir, batch_size, adapter=None):
    """
    Create training, validation and test tf.data.Dataset data loaders
    from TFRecords files saved by the write_dataset_tfrecords() function.

    The optional `adapter` argument is used to provide the index of a LoRA adapter.
    The data loaders replicates it in every item of every batch.

    The function returns:
        - The dataset metadata (data sizes, sequence length, etc)
        - A training, validation and test tf.data.Dataset data loaders

    Each data loader returns a batch of dictionaries, each of them containing:
        - Token IDs of the prompt and annotation
        - Attention mask to use to hide the pad tokens from the attention heads
        - Loss mask to use to only include the model answer in the loss calculation
        - Adapter index if used
    """

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Unable to find dataset directory {dataset_dir}")
    
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata['batchsize'] = batch_size

    seq_len = metadata['seq_len']
    print(f"\nCreating data loaders for dataset `{metadata['dataset_name']}`:")
    print(f"  train size: {metadata['train_size']}")
    print(f"  val size: {metadata['val_size']}")
    print(f"  test size: {metadata['test_size']}")
    print(f"  sequence length: {seq_len}")

    train_tfr = tf.data.TFRecordDataset(os.path.join(dataset_dir, "train.tfrecord"))
    val_tfr = tf.data.TFRecordDataset(os.path.join(dataset_dir, "val.tfrecord"))
    test_tfr = tf.data.TFRecordDataset(os.path.join(dataset_dir, "test.tfrecord"))

    train_ds = build_ds_pipeline(train_tfr, seq_len, batch_size, adapter=adapter, shuffle=True)
    val_ds   = build_ds_pipeline(val_tfr,   seq_len, batch_size, adapter=adapter)
    test_ds  = build_ds_pipeline(test_tfr,  seq_len, batch_size, adapter=adapter)

    return (train_ds, val_ds, test_ds), metadata

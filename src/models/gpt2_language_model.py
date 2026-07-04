# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
import tensorflow as tf
from models.gpt2_model import GPT2Model


class GPT2LanguageModel(tf.keras.models.Model):
    """
    Implements OpenAI's GPT-2 model with language modelling head.

    The model calculates loss and metrics including perplexity and 
    exact-match accuracy. It can be trained using the fit() method 
    and evaluated  using the evaluate() method.

    Arguments:
        model_config:
            The model configuration parameters, a dictionary with 
            the following items:
                "vocab_size": vocabulary size.
                "max_seq_len": input sequence maximum length (context size).
                "d_model": hidden state size (embeddings size).
                "n_layers": number of transformer blocks.
                "n_heads": number of attention heads.
            These parameters for a given model size can be obtained using
            the get_gpt2_model_config() function in model_utils.py.

        lora_config:
            The LoRA layers configuration, a dictionary specifying the number
            of adapters, and the rank and alpha parameters of each adapter.
            Example:
                lora_config = {
                    "num_adapters": 3,     # Number of adapters
                    "rank": (16, 8, 8),    # rank parameter of each adapter
                    "alpha": (32, 16, 16)  # alpha parameter of each adapter (same order as in rank)
                }
            Present only when the model has LoRA adapters.
    """

    def __init__(self, model_config, lora_config=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.model_config = model_config
        self.lora_config = lora_config

        self.gpt2_model = GPT2Model(
            model_config,
            lora_config=lora_config,
            name="gpt2_model"
        )

        # Training metrics trackers
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy_tracker = tf.keras.metrics.Mean(name="train_accuracy")
        self.train_perplexity_tracker =  tf.keras.metrics.Mean(name="train_perplexity")

        # Test metrics trackers
        self.test_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.test_accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
        self.test_perplexity_tracker =  tf.keras.metrics.Mean(name="perplexity")


    def call(self, inputs, training=None):
        """
        Forward pass through the GPT-2 LM model.

        Arguments:
            inputs:
                A batch of dictionaries, each of them with the following items:
                    "token_ids": 
                        Input token sequences.
                        A tensor with shape (batch, seq_len).
                    "attention_mask":
                        Mask specifying which token positions to attend to.
                        A tensor with shape (batch, seq_len).
                    "adapter":
                        Indices of the active LoRA adapters, one for each input sequence.
                        A tensor with shape (batch,).
                        Present only when the model has LoRA adapters.
                        If an index is greater than or equal to num_adapters, no adapter 
                        will be selected at this sequence position and the network will 
                        behave as if there was no adapter (pretrained weights will be used).
            training:
                Training or evaluation mode.

        Returns:
            The logits, a tensor with shape (batch, seq_len, vocab_size).
        """

        if "adapter" in inputs:
            # Convert the adapter index to one-hot
            # If an index is >= num_adapters, it is encoded 
            # to tf.zeros((batch_size, num_adapters)).
            adapter_selector = tf.one_hot(
                inputs["adapter"],
                depth=self.lora_config["num_adapters"],
                dtype=tf.float32
            )
        else:
            adapter_selector = None
            
        gpt2_output = self.gpt2_model(
            inputs["input_ids"],
            inputs["attention_mask"],
            adapter_selector=adapter_selector,
            training=training
        )

        # Output linear layer that projects hidden state representations 
        # to vocabulary. Weights are shared with the token embedding matrix.
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits


    # Set the dropout rate of all dropout layers
    def set_dropout_rate(self, dropout_rate):
        self.gpt2_model.set_dropout_rate(dropout_rate)

    # Freeze all the weights of the model except the adapter with the index in argument
    def lora_freeze(self, adapter):
        self.gpt2_model.lora_freeze(adapter)

    # Save the configuration of the model (including the LoRA configuration
    # if any) in a JSON file and its weights in a Keras weights.h5 file.
    def save(self, model_dir, model_name):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Save model and LoRA config
        config = {"model_config": self.model_config}
        if self.lora_config is not None:
            config["lora_config"] = self.lora_config
        with open(os.path.join(model_dir, f"{model_name}.json"), "w") as f:
            json.dump(config, indent=2, fp=f)

        # Save model weights
        self.save_weights(os.path.join(model_dir, f"{model_name}.weights.h5"))


    def compute_loss(self, input_ids, y_pred, mask):
        """
        Calculates the categorical crossentropy loss.

        Arguments:
            input_ids: 
                Input token sequences.
                A tensor with shape (batch, seq_len).
            y_pred:
                Model predictions (logits over the vocabulary).
                A tensor with shape (batch, seq_len, vocab_size).
            mask:
                Mask specifying which token positions contribute to the loss.
                A tensor with shape (batch, seq_len).

        Returns:
            The mean cross-entropy loss over non-masked tokens.
            A scalar tensor.
        """

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate the mask to align with labels
        mask = mask[:, 1:]

        # Drop the last prediction (no next-token label for the final position)
        y_pred = y_pred[:, :-1, :]

        # Calculate cross-entropy loss per token (element wise)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Apply mask token-wise
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        
        # Return mean loss over non-masked tokens
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


    def compute_accuracy(self, input_ids, y_pred, mask):
        """
        Calculates the exact-match accuracy of model predictions
        at the sequence level.

        Arguments:
            input_ids: 
                Input token sequences.
                A tensor with shape (batch, seq_len).
            y_pred:
                Model predictions (logits over the vocabulary).
                A tensor with shape (batch, seq_len, vocab_size).
            mask:
                Mask specifying which token positions contribute to the loss.
                A tensor with shape (batch, seq_len).

        Returns:
            The exact-match accuracy at the sequence level, a scalar tensor.

        """
        y_true = input_ids[:, 1:]
        mask = mask[:, 1:]
        y_pred = y_pred[:, :-1, :]

        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        correct = tf.cast(tf.equal(predictions, y_true), dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        # A position is "wrong" if incorrect AND masked (i.e. part of the label)
        # A label sequence is fully correct only if all its tokens are correct
        incorrect = (1.0 - correct) * mask  # 1 where a label token is wrong
        
        # Per-sample: was there any incorrect label token?
        any_incorrect = tf.reduce_sum(incorrect, axis=1)       # (batch,)
        sample_correct = tf.cast(tf.equal(any_incorrect, 0.0), dtype=tf.float32)

        # Also check that the sample had at least one label token (mask sum > 0)
        has_label = tf.cast(tf.greater(tf.reduce_sum(mask, axis=1), 0.0), dtype=tf.float32)

        accuracy = tf.reduce_sum(sample_correct * has_label) / tf.maximum(tf.reduce_sum(has_label), 1.0)

        return accuracy


    def train_step(self, inputs):
        """
        Performs one training step.

        Arguments:
            inputs:
                A dictionary with the following items:
                    "input_ids":
                        Input token sequences.
                        A tensor with shape (batch, seq_len).
                    "attention_mask":
                        Mask specifying which token positions to attend to.
                        A tensor with shape (batch, seq_len).
                    "loss_mask":
                        Mask specifying which token positions contribute to the loss.
                        A tensor with shape (batch, seq_len).
                    "adapter":
                        Indices of the active LoRA adapters, one for each input sequence.
                        A tensor with shape (batch,).
                        Present only when the model has LoRA adapters.
        """

        input_ids = inputs["input_ids"]
        loss_mask = inputs["loss_mask"]

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compute_loss(input_ids, y_pred, loss_mask)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(gradients, self.trainable_variables)
        ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Calculate metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.exp(loss)

        # Update loss and metrics trackers
        self.train_loss_tracker.update_state(loss)
        self.train_accuracy_tracker.update_state(accuracy)
        self.train_perplexity_tracker.update_state(perplexity)

        return {m.name: m.result() for m in [
            self.train_loss_tracker,
            self.train_accuracy_tracker,
            self.train_perplexity_tracker
        ]}


    def test_step(self, inputs):
        """
        Performs one evaluation step.
        Same arguments as train_step().
        """

        input_ids = inputs["input_ids"]
        loss_mask = inputs["loss_mask"]
 
        y_pred = self(inputs, training=False)
        loss = self.compute_loss(input_ids, y_pred, loss_mask)
        
        # Compute metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.math.exp(loss)

        # Update loss and metrics trackers
        self.test_loss_tracker.update_state(loss)
        self.test_accuracy_tracker.update_state(accuracy)
        self.test_perplexity_tracker.update_state(perplexity)
        
        # Return metrics
        return {m.name: m.result() for m in [
            self.test_loss_tracker,
            self.test_accuracy_tracker,
            self.test_perplexity_tracker
        ]}


    # Register loss and metrics trackers
    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.train_accuracy_tracker,
            self.train_perplexity_tracker,
            self.test_loss_tracker,
            self.test_accuracy_tracker,
            self.test_perplexity_tracker
        ]

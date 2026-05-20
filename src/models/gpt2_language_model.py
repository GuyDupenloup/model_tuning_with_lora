# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
import tensorflow as tf
from models.gpt2_model import GPT2Model


@tf.keras.utils.register_keras_serializable()
class GPT2LanguageModel(tf.keras.models.Model):
    """
    Implements OpenAI's GPT-2 model with language modelling head.
    
    The model calculates loss, and metrics including perplexity and accuracy (exact token-to-token match).

    Model instantiation arguments:
    -----------------------------
        model_config:
            The model configuration, a dictionary.
            Keys must include:
                'vocab_size': vocabulary size
                'max_seq_len': input sequence maximum length (context size)
                'd_model': hidden state size (embeddings size)
                'n_layers': number of transformer blocks
                'n_heads': number of attention heads

        lora_config:
            Optional LoRA configuration, a dictionary.
            If present, keys must include 'num_adapters, 'rank' and 'alpha'. The values of 'alpha' and 'rank'
            must be tuples of positive integers with length equal to 'num_adapters'.
            If the argument is not present, no LoRA layers are added to the model.

    Model call() method:
    -------------------
        Arguments:
            inputs:
                A dictionary with the following items:
                    'input_ids':
                        Token IDs of the input sequence.
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32
                    'attention mask':
                        Attention mask used to remove padding tokens from consideration.
                        0: ignored, 1: considered
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32
                    'loss_mask':
                        Loss mask to exclude some of the tokens in the prompt from loss calculation.
                        0: excluded, 1: included
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32

        Returns:
            Hidden state logits over vocabulary
            A tf.Tensor of with shape (batch_size, seq_len, vocab_size) and data_type tf.float32
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

    # Set the dropout rate of all dropout layers
    def set_dropout_rate(self, dropout_rate):
        self.gpt2_model.set_dropout_rate(dropout_rate)

    # Activate a LoRA adapter
    def activate_adapter(self, adapter):
        self.gpt2_model.activate_adapter(adapter)

    # Deactivate all LoRA adapters
    def deactivate_adapters(self):
        self.gpt2_model.deactivate_adapters()

    # Freeze all the weights of the model except
    # for the layers of the active LoRA adapter
    def lora_freeze(self):
        self.gpt2_model.lora_freeze()

    # Save the configuration of the model in a JSON file
    # and its weights in a "weights.h5" file. The model
    # can be reloaded using these two files.
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

    def call(self, inputs, training=None):
        """
        Forward pass through language model.
        """
        gpt2_output = self.gpt2_model(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("adapter_selector", None),
            training=training
        )

        # Output linear layer that projects hidden state representations to vocabulary.
        # Weights of the projection matrix are shared with the token embedding matrix.
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits

    def compute_loss(self, input_ids, y_pred, mask):
        """
        Calculates the loss.

        Arguments:
            input_ids: 
                Token IDs of the input sequence.
                Shape: (batch_size, seq_len)
            y_pred:
                Model predictions (logits over the vocabulary).
                Shape: (batch_size, seq_len, vocab_size)
            mask:
                Mask specifying which token positions contribute to the loss.
                Shape: (batch_size, seq_len)
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
        any_incorrect = tf.reduce_sum(incorrect, axis=1)       # (batch_size,)
        sample_correct = tf.cast(tf.equal(any_incorrect, 0.0), dtype=tf.float32)

        # Also check that the sample had at least one label token (mask sum > 0)
        has_label = tf.cast(tf.greater(tf.reduce_sum(mask, axis=1), 0.0), dtype=tf.float32)

        accuracy = tf.reduce_sum(sample_correct * has_label) / tf.maximum(tf.reduce_sum(has_label), 1.0)

        return accuracy


    def train_step(self, inputs):
        """
        Performs one training step using next-token prediction.
        Computes the forward pass, loss, gradients, and updates model weights.
        Also updates accuracy and perplexity metrics based on the masked tokens.
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

        # Metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.exp(loss)

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
        Runs a forward pass without gradient updates and computes evaluation loss.
        Updates accuracy and perplexity using the same masked next-token objective.
        Returns the current values of all tracked evaluation metrics.
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


    # Register trackers
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
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "model_config": self.model_config,          # set as self.config in __init__
            "lora_config": self.lora_config
        })
        return config

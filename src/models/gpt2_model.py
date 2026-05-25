# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

def gelu_approximate(x):
    return tf.nn.gelu(x, approximate=True)


@tf.keras.utils.register_keras_serializable()
class LoRALayer(tf.keras.layers.Layer):
    """
    Low-Rank Adaptation layer.
    Matrices A and B are initialized as described in the original LoRA paper.
    """
    def __init__(self, output_size, rank, alpha, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.output_size = output_size
        self.rank = rank
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        self.scaling = tf.constant(alpha, dtype=tf.float32) / tf.constant(rank, dtype=tf.float32)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # A: projects from input to rank
        self.lora_A = tf.keras.layers.Dense(
            rank,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="lora_A"
        )
        
        # B: projects from rank to output_size
        self.lora_B = tf.keras.layers.Dense(
            output_size,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="lora_B"
        )
    
    def call(self, inputs, training=None):
        # Apply dropout before A
        x = self.dropout(inputs, training=training)
        return self.lora_B(self.lora_A(x)) * self.scaling

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_size": self.output_size,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout_rate": self.dropout_rate
        })
        return config


def _apply_lora_layers(lora_layers, inputs, adapter_selector, training=None):
    """
    Compute the weighted sum of all LoRA adapter outputs for each sample.

    Args:
        lora_layers:      list of LoRALayer, length = num_adapters
        inputs:           tf.Tensor (batch, seq, d)
        adapter_selector: tf.Tensor (batch, num_adapters), float32
        training:         bool or None

    Returns:
        tf.Tensor (batch, seq, d)
    """
    # (num_adapters, batch, seq, d)
    stacked = tf.stack([layer(inputs, training=training) for layer in lora_layers], axis=0)

    # (batch, num_adapters, seq, d)
    stacked = tf.transpose(stacked, perm=[1, 0, 2, 3])

    # Reshape selector for broadcasting: (batch, num_adapters, 1, 1)
    weights = tf.reshape(
        adapter_selector,
        (tf.shape(adapter_selector)[0], len(lora_layers), 1, 1)
    )

    # Weighted sum over adapter axis → (batch, seq, d)
    return tf.reduce_sum(stacked * weights, axis=1)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_model, n_heads, lora_config=None, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        assert d_model % n_heads == 0, f"d_model={d_model} is not divisible by n_heads={n_heads}"

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.lora_config = lora_config
        self.dropout_rate = dropout_rate

        self.d_head = d_model // n_heads
        add_lora_layers = lora_config is not None

        if add_lora_layers:
            num_adapters = lora_config["num_adapters"]
            rank = lora_config["rank"]
            alpha = lora_config["alpha"]

        self.epsilon = tf.constant(-1e9, dtype=tf.float32)

        # Combined QKV projection
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name="W_qkv")

        if add_lora_layers:
            self.W_q_lora_layers = [
                LoRALayer(d_model, rank=rank[i], alpha=alpha[i], name=f"W_q_lora_{i}")
                for i in range(num_adapters)
            ]
            self.W_k_lora_layers = [
                LoRALayer(d_model, rank=rank[i], alpha=alpha[i], name=f"W_k_lora_{i}")
                for i in range(num_adapters)
            ]
            self.W_v_lora_layers = [
                LoRALayer(d_model, rank=rank[i], alpha=alpha[i], name=f"W_v_lora_{i}")
                for i in range(num_adapters)
            ]

        self.output_proj = tf.keras.layers.Dense(d_model, name="out_proj")

        if add_lora_layers:
            self.c_proj_lora_layers = [
                LoRALayer(d_model, rank=rank[i], alpha=alpha[i], name=f"c_proj_lora_{i}")
                for i in range(num_adapters)
            ]

        self.attn_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.causal_mask_full = tf.linalg.band_part(
            tf.ones((max_seq_len, max_seq_len), dtype=tf.bool), -1, 0
        )


    def call(self, inputs, attention_mask, adapter_selector=None, training=None):

        batch = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Compute Q, K, V from combined projection then split
        QKV = self.W_qkv(inputs)
        Q, K, V = tf.split(QKV, 3, axis=-1)

        if adapter_selector is not None: 
            Q += _apply_lora_layers(self.W_q_lora_layers, inputs, adapter_selector, training)
            K += _apply_lora_layers(self.W_k_lora_layers, inputs, adapter_selector, training)
            V += _apply_lora_layers(self.W_v_lora_layers, inputs, adapter_selector, training)

        # Reshape
        Q = tf.reshape(Q, (batch, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, seq_len, self.n_heads, self.d_head))

        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Attention scores
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))
        scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        # Masks
        attn_mask = tf.cast(attention_mask, tf.bool)
        scores = tf.where(attn_mask[:, None, None, :], scores, self.epsilon)

        causal_mask = self.causal_mask_full[:seq_len, :seq_len]
        scores = tf.where(causal_mask[None, None, :, :], scores, self.epsilon)

        # Softmax
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Context
        context = tf.matmul(attn_weights, V)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, (batch, seq_len, self.d_model))

        # Output projection
        out = self.output_proj(context)

        if adapter_selector is not None: 
            out += _apply_lora_layers(self.c_proj_lora_layers, context, adapter_selector, training) 

        return out


    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "lora_config": self.lora_config,
            "dropout_rate": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network.
    Applies two linear transformations with GELU activation.
    """
    def __init__(self, d_model, lora_config=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
        self.lora_config = lora_config
        
        add_lora_layers = lora_config is not None

        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=gelu_approximate,
            name="ffn_inner",
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name="ffn_out")

        if add_lora_layers:
            num_adapters = lora_config["num_adapters"]
            rank = lora_config["rank"]
            alpha = lora_config["alpha"]

            self.ff_inner_lora_layers = [
                LoRALayer(4 * d_model, rank=rank[i], alpha=alpha[i], name=f"ff_inner_lora_{i}")
                for i in range(num_adapters)
            ]
            self.ff_out_lora_layers = [
                LoRALayer(d_model, rank=rank[i], alpha=alpha[i], name=f"ff_out_lora_{i}")
                for i in range(num_adapters)
            ]


    def call(self, inputs, adapter_selector=None, training=None):

        x = self.ff_inner(inputs)
        if adapter_selector is not None:
            x += _apply_lora_layers(self.ff_inner_lora_layers, inputs, adapter_selector, training)

        x = self.ff_out(x)
        if adapter_selector is not None:
            x += _apply_lora_layers(self.ff_out_lora_layers, x, adapter_selector, training)

        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "lora_config": self.lora_config
        })
        return config

@tf.keras.utils.register_keras_serializable()
class GPT2Transformer(tf.keras.layers.Layer):
    """
    GPT2 transformer block.
    Consists of multi-head attention and feed-forward network,
    each with layer normalization and residual connections.
    """
    def __init__(
            self, max_seq_len, d_model, n_heads, lora_config=None, dropout_rate=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.lora_config = lora_config
        self.dropout_rate = dropout_rate

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.attn_heads = MultiHeadAttention(
            max_seq_len,
            d_model,
            n_heads,
            lora_config=lora_config,
            dropout_rate=dropout_rate,
            name="attn_heads"
        )
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate, name="drop_1")
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_2")
        self.ffn = GPT2FeedForwardNetwork(d_model, lora_config=lora_config, name="ffn")
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate, name="drop_2")


    def call(self, inputs, attention_mask, adapter_selector=None, training=None):

        # First sub-layer
        x1 = self.layer_norm_1(inputs, training=training)
        x1 = self.attn_heads(x1, attention_mask, adapter_selector=adapter_selector, training=training)
        x1 = self.dropout_1(x1, training=training)

        # First residual connection
        x2 = x1 + inputs

        # Second sub-layer
        x3 = self.layer_norm_2(x2, training=training)
        x3 = self.ffn(x3, adapter_selector=adapter_selector, training=training)
        x3 = self.dropout_2(x3, training=training)

        # Second residual connection
        output = x2 + x3

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "lora_config": self.lora_config,
            "dropout_rate": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable()
class GPT2Model(tf.keras.models.Model):
    """
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
    """

    def __init__(self, model_config, lora_config=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.model_config = model_config
        self.lora_config = lora_config

        self.dropout_rate = 0.1   # Default value

        vocab_size, max_seq_len, d_model, n_layers, n_heads = (
            model_config[k] for k in ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads")
        )

        # Token and position embedding layers
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name="tkn_emb")
        self.position_embed_layer = tf.keras.layers.Embedding(max_seq_len, d_model, name="pos_emb")

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        # Transformer layers
        self.transformer_layers = [
            GPT2Transformer(
                max_seq_len,
                d_model,
                n_heads,
                lora_config=lora_config,
                dropout_rate=self.dropout_rate,
                name=f"transformer_{i}"
            ) 
            for i in range(n_layers)
        ]

        self.layer_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="lnorm_f")

        # Token indices for position embeddings
        self.positions = tf.range(start=0, limit=max_seq_len, delta=1)


    def call(self, inputs, attention_mask, adapter_selector=None, training=None):
        """
        Forward pass through the GPT-2 model.
        """
        
        # Token embeddings
        token_embed = self.token_embed_layer(inputs)

        # Position embeddings
        seq_length = tf.shape(inputs)[1]
        position_ids = tf.range(seq_length)
        position_embed = self.position_embed_layer(position_ids)

        # Embeddings
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for transformer in self.transformer_layers:
            x = transformer(x,
                attention_mask,
                adapter_selector=adapter_selector,
                training=training
            )

        output = self.layer_norm_final(x, training=training)

        return output


    def set_dropout_rate(self, dropout_rate):
        """
        Update the dropout rate for all dropout layers in the model.
        
        Arguments:
            dropout_rate: float between 0 and 1
        """
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.dropout_rate = dropout_rate

        for transformer in self.transformer_layers:
            transformer.dropout_1.rate = dropout_rate
            transformer.dropout_2.rate = dropout_rate
            transformer.attn_heads.attn_dropout.rate = dropout_rate

            # LoRA dropout layers
            if self.lora_config is not None:
                for lora_layers in (
                    transformer.attn_heads.W_q_lora_layers,
                    transformer.attn_heads.W_k_lora_layers,
                    transformer.attn_heads.W_v_lora_layers,
                    transformer.attn_heads.c_proj_lora_layers,
                    transformer.ffn.ff_inner_lora_layers,
                    transformer.ffn.ff_out_lora_layers,
                ):
                    for lora_layer in lora_layers:
                        lora_layer.dropout.rate = dropout_rate


    def lora_freeze(self, adapter_idx):
        """
        Freezes all model weights except the LoRA layers of the active adapter.
        Raises an error if the model has no LoRA layers or no LoRA adapter was activated.
        """

        if self.lora_config is None:
            raise ValueError("Unable to freeze layers. The model has no LoRA layers.")
        
        num_adapters = self.lora_config["num_adapters"]
        if adapter_idx < 0 or adapter_idx >= num_adapters:
            raise ValueError(
                f"LoRA adapter index should be in interval [0, {num_adapters - 1}]. "
                f"Received {adapter_idx}"
            )

        # Make the model trainable
        self.trainable = True

        # Freeze embeddings
        self.token_embed_layer.trainable = False
        self.position_embed_layer.trainable = False

        # Freeze final layer norm
        self.layer_norm_final.trainable = False

        for transformer in self.transformer_layers:
            # Freeze LayerNorms
            transformer.layer_norm_1.trainable = False
            transformer.layer_norm_2.trainable = False

            # Freeze FFN base weights
            transformer.ffn.ff_inner.trainable = False
            transformer.ffn.ff_out.trainable = False

            # Freeze multi-head attention base weights
            attn = transformer.attn_heads
            attn.W_qkv.trainable = False
            attn.output_proj.trainable = False

            if self.lora_config is not None:
                for i in range(self.lora_config["num_adapters"]):
                    if i != adapter_idx:
                        attn.W_q_lora_layers[i].trainable = False
                        attn.W_k_lora_layers[i].trainable = False
                        attn.W_v_lora_layers[i].trainable = False
                        attn.c_proj_lora_layers[i].trainable = False
                        transformer.ffn.ff_inner_lora_layers[i].trainable = False
                        transformer.ffn.ff_out_lora_layers[i].trainable = False
                    else:
                        attn.W_q_lora_layers[i].trainable = True
                        attn.W_k_lora_layers[i].trainable = True
                        attn.W_v_lora_layers[i].trainable = True
                        attn.c_proj_lora_layers[i].trainable = True
                        transformer.ffn.ff_inner_lora_layers[i].trainable = True
                        transformer.ffn.ff_out_lora_layers[i].trainable = True


    def get_config(self):
        config = super().get_config()
        config.update({
            "model_config": self.model_config,
            "lora_config": self.lora_config
        })
        return config


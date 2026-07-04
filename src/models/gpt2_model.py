# Copyright (c) 2026 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf

def gelu_approximate(x):
    return tf.nn.gelu(x, approximate=True)


class LoRALayer(tf.keras.layers.Layer):
    """
    Implements the Low-Rank Adaptation layer from the original LoRA paper. 
    Matrices A and B are initialized as described in the paper.
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
        """
        Forward pass through the LoRA layer.

        Arguments:
            inputs:
                Input tensor with shape (batch, seq_len, input_size).
            training:
                Training or evaluation mode.

        Returns:
            A low-rank correction tensor with the same shape as inputs,
            scaled by alpha / rank.

        """
        x = self.dropout(inputs, training=training)
        return self.lora_B(self.lora_A(x)) * self.scaling


def _apply_lora_layers(lora_layers, inputs, adapter_selector, training=None):
    """
    Given a list of LoRA layers and inputs to these layers, get the output
    of the layer designated by a one-hot encoded selector index.

    First, the outputs of all the LoRA layers are computed. Then, the adapter
    selector is used as a mask to get the output of the selected layer
    (computed as a weighted sum).

    The adapter selector is the one-hot-encoded index of the active adapter.

    All its bits can also be set to zero (instead of being a one-hot encoding). 
    In this case, the function outputs zeros. The LoRA layers don't contribute,
    and the model behaves like if no LoRA layer was present.

    Arguments:
        lora_layers:
            List of LoRA layers. Length: num_adapters.
        inputs:
            Inputs to the LoRA layers, a tensor with shape (batch, seq_len, d_model).
        adapter_selector:
            One-hot encoded index of the selected adapter. All bits set to 0 if no LoRA
            adapter is active.
            A tensor with shape (batch, num_adapters).
        training:
            Training or evaluation mode.

    Returns:
        The output of the selected LoRA layer. 
        A tensor with shape (batch, seq_len, d_model).
    """

    # Get the outputs of the LoRA layers and stack them
    # Shape: (num_adapters, batch, seq, d)
    stacked = tf.stack([layer(inputs, training=training) for layer in lora_layers], axis=0)

    # (batch, num_adapters, seq, d)
    stacked = tf.transpose(stacked, perm=[1, 0, 2, 3])    # (batch, num_adapters, seq, len)

    # Reshape selector for broadcasting: (batch, num_adapters, 1, 1)
    weights = tf.reshape(
        adapter_selector,
        (tf.shape(adapter_selector)[0], len(lora_layers), 1, 1)
    )

    # Weighted sum of all layer outputs
    return tf.reduce_sum(stacked * weights, axis=1)   # (batch, seq, d)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements the multi-head causal attention block from the Transformer
    and GPT/GPT-2 papers, with the optional addition of LoRA layers 
    as described in the original LoRA paper.
    """
    
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

        # Combined QKV projection matrices
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
        """
        Forward pass through the attention heads
        """

        batch = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Compute Q, K, V from combined projection then split
        QKV = self.W_qkv(inputs)              # (batch, seq_len, 3 * d_model)
        Q, K, V = tf.split(QKV, 3, axis=-1)   # 3 x (batch, seq_len, d_model)

        if adapter_selector is not None: 
            Q += _apply_lora_layers(self.W_q_lora_layers, inputs, adapter_selector, training)
            K += _apply_lora_layers(self.W_k_lora_layers, inputs, adapter_selector, training)
            V += _apply_lora_layers(self.W_v_lora_layers, inputs, adapter_selector, training)

        # Reshape
        Q = tf.reshape(Q, (batch, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, seq_len, self.n_heads, self.d_head)) 
        V = tf.reshape(V, (batch, seq_len, self.n_heads, self.d_head))

        Q = tf.transpose(Q, perm=(0, 2, 1, 3))   # (batch, n_heads, seq_len, d_head)
        K = tf.transpose(K, perm=(0, 2, 1, 3))   # (batch, n_heads, seq_len, d_head)
        V = tf.transpose(V, perm=(0, 2, 1, 3))   # (batch, n_heads, seq_len, d_head)

        # Calculate and scale the attention scores
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))
        scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        # Apply attention mask specifying which token positions 
        # to attend to (used to hide pad tokens)
        attn_mask = tf.cast(attention_mask, tf.bool)                           # (batch, seq_len)
        scores = tf.where(attn_mask[:, None, None, :], scores, self.epsilon)   # (batch, n_heads, seq_len, seq_len)

        # Apply causal attention mask
        causal_mask = self.causal_mask_full[:seq_len, :seq_len]
        scores = tf.where(causal_mask[None, None, :, :], scores, self.epsilon)    # (batch, n_heads, seq_len, seq_len)

        # Apply softmax to the scores to get the weights
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Context
        context = tf.matmul(attn_weights, V)                           # (batch, n_heads, seq_len, d_head)
        context = tf.transpose(context, perm=(0, 2, 1, 3))             # (batch, seq_len, n_heads, d_head)
        context = tf.reshape(context, (batch, seq_len, self.d_model)) 

        # Output projection
        out = self.output_proj(context)     # (batch, seq_len, d_model)

        if adapter_selector is not None: 
            out += _apply_lora_layers(self.c_proj_lora_layers, context, adapter_selector, training) 

        return out


class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Implements the FFN from the original Transformer and GPT/GPT-2 papers.
    """

    def __init__(self, d_model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model
    
        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=gelu_approximate,
            name="ffn_inner",
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name="ffn_out")


    def call(self, inputs, adapter_selector=None, training=None):
        """
        Forward pass through the FFN
        """

        x = self.ff_inner(inputs)
        out = self.ff_out(x)
        return out


class GPT2Transformer(tf.keras.layers.Layer):
    """
    Implements the transformer block from the original Transformer
    and GPT/GPT-2 papers.
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
        self.ffn = GPT2FeedForwardNetwork(d_model, name="ffn")
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


class GPT2Model(tf.keras.models.Model):
    """
    Implements the GPT-2 model from OpenAI's GPT-2 paper.
    
    This is the base model. A head needs to be added to it, 
    e.g. language modelling or classification head.

    Arguments:
        model_config:
            The model configuration parameters, a dictionary 
            with the following items:
                "vocab_size": vocabulary size.
                "max_seq_len": input sequence maximum length (context size).
                "d_model": hidden state size (embeddings size).
                "n_layers": number of transformer blocks.
                "n_heads": number of attention heads.
            These parameters for a given model size can be obtained using
            the get_gpt2_model_config() function in model_utils.py.

        lora_config:
            The LoRA layers configuration, an optional dictionary specifying the number
            of adapters, and the rank and alpha parameters of each of them.
            Example:
                lora_config = {
                    "num_adapters": 3,     # Number of adapters
                    "rank": (16, 8, 8),    # rank parameter of each adapter
                    "alpha": (32, 16, 16)  # alpha parameter of each adapter (same order as in rank)
                }
            The argument is present only when the model has LoRA adapters.
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

        Arguments:
            inputs:
                Input token sequences.
                A tensor with shape (batch, seq_len).

            attention_mask:
                Mask specifying which token positions to attend to.
                A tensor with shape (batch, seq_len).

            adapter_selector:
                One-hot encoded indices of the active LoRA adapters.
                All zeros at positions where no adapter is active.
                A tensor with shape (batch, num_adapters).
                Present only when the model has LoRA adapters.

            training:
                Training or evaluation mode.

        Returns:
            The hidden state output, a tensor with shape (batch, seq_len, d_model).

        All the bits of an index in the adapter selector can be set to zero,
        instead of being a one-hot encoding. In this case, no LoRA adapter is active
        at this position, and the model reverts back to pretrained weights
        with no adapter.

        For example, for a model with 3 LoRA adapters:
            adapter_selector = [
                [1, 0, 0],       # Activate adapter #0
                [0, 0, 1],       # Activate adapter #2
                [0, 0, 0],       # Don't activate any adapter
                [0, 1, 0]        # Activate adapter #1
            ]

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
        Set the dropout rate for all the dropout layers of the model.
        
        Arguments:
            dropout_rate: a float between 0 and 1.
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
                    transformer.attn_heads.c_proj_lora_layers
                ):
                    for lora_layer in lora_layers:
                        lora_layer.dropout.rate = dropout_rate


    def lora_freeze(self, adapter):
        """
        Makes the entire model trainable, then freezes all the model weights
        except for the layers of the LoRA adapter with index `adapter`.

        Raises an error if the model has no LoRA adapter.
        """

        if self.lora_config is None:
            raise ValueError("Unable to freeze. The model has no LoRA adapter.")
        
        num_adapters = self.lora_config["num_adapters"]
        if adapter < 0 or adapter >= num_adapters:
            raise ValueError(
                f"LoRA adapter index should be in interval [0, {num_adapters - 1}]. "
                f"Received {adapter}"
            )

        # Make the entire model trainable
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
                    if i != adapter:
                        attn.W_q_lora_layers[i].trainable = False
                        attn.W_k_lora_layers[i].trainable = False
                        attn.W_v_lora_layers[i].trainable = False
                        attn.c_proj_lora_layers[i].trainable = False
                    else:
                        attn.W_q_lora_layers[i].trainable = True
                        attn.W_k_lora_layers[i].trainable = True
                        attn.W_v_lora_layers[i].trainable = True
                        attn.c_proj_lora_layers[i].trainable = True

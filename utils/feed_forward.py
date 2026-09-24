"""
feed_forward.py - Position-wise Feed-Forward Network
Transformer Reproduction Project - Phase 7

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from feed_forward import FeedForwardNetwork
"""

import tensorflow as tf

class FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, x W1 + b1) W2 + b2

    Args:
        d_model : input and output dimension (e.g. 512)
        d_ff : inner dimension (e.g. 2048)
        dropout : dropout rate (paper: 0.1)
    """

    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.supports_masking = True
        self.dense_1 = tf.keras.layers.Dense(d_ff, activation="relu", name="ffn_dense_1")
        self.dense_2 = tf.keras.layers.Dense(d_model, name="ffn_dense_2")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # x shape: (batch, seq_len, d_model)
        x = self.dense_1(x)
        # x shape: (batch, seq_len, d_ff)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        # x shape: (batch, seq_len, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model" : self.d_model,
            "d_ff" : self.d_ff,
            "dropout" : self.dropout_rate,
        })
        return config

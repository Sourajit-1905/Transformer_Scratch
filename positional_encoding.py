"""
positional_encoding.py — Sinusoidal Positional Encoding

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from positional_encoding import PositionalEncoding
"""

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal positional encoding - Vaswani et al. 2017, Section 3.5.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model : embedding dimension (e.g. 512)
        max_len : maximum sequence length (e.g. 5000)
        dropout : dropout rate (paper: 0.1)
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        self.supports_masking = True
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.pe = self._build_pe(max_len, d_model)

    def _build_pe(self, max_len, d_model):
        positions = np.arange(max_len)[:, np.newaxis]
        i = np.arange(0, d_model, 2)
        div_term = np.exp(-np.log(10000.0) * i / d_model)
        angles = positions * div_term
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
        return tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model" : self.d_model,
            "max_len" : self.max_len,
            "dropout" : self.dropout_rate,
        })
        return config

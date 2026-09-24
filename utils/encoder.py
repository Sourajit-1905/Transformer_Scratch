"""
encoder.py - Transformer Encoder

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from encoder import EncoderLayer, Encoder
"""

import tensorflow as tf
from utils.multi_head_attention import MultiHeadAttention
from utils.feed_forward import FeedForwardNetwork
from utils.positional_encoding import PositionalEncoding


class EncoderLayer(tf.keras.layers.Layer):
    """
    Single encoder layer (POST-LN, original paper).

    x = LayerNorm(x + MultiHeadAttention(x, x, x, mask))
    x = LayerNorm(x + FFN(x))
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.supports_masking = True

        self.self_attention = MultiHeadAttention(d_model, num_heads,
                                                 name="encoder_self_attention")
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout,
                                                 name="encoder_ffn")
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                  name="encoder_norm_1")
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                  name="encoder_norm_2")
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        attn_out, attn_weights = self.self_attention(x, x, x,
                                                      mask=mask,
                                                      training=training)

        attn_out = self.dropout_1(attn_out, training=training)
        x = self.norm_1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        x = self.norm_2(x + ffn_out)
        return x, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate}
        )

        return config


class Encoder(tf.keras.layers.Layer):
    """
    Full encoder stack.
    src_ids -> Embedding -> PositionalEncoding -> [EncoderLayer x N]
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 vocab_size, max_len=5000, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.supports_masking = True
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                                                       name="encoder_embedding")
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout,
                                               name="encoder_pos_encoding")
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout,
                         name=f"encoder_layer_{i}")
            for i in range(num_layers)
        ]

    def call(self, src_ids, mask=None, training=False):
        x = self.embedding(src_ids)
        x = self.pos_encoding(x, training=training)
        attention_weights = {}
        for i, layer in enumerate(self.enc_layers):
            x, attn_w = layer(x, mask=mask, training=training)
            attention_weights[f"encoder_layer_{i}"] = attn_w
        return x, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "d_model": self.d_model})
        return config

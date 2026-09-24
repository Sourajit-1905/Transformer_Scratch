"""
decoder.py - Transformer Decoder
Transformer Reproduction Project - Phase 9

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from decoder import DecoderLayer, Decoder
"""

import tensorflow as tf
from utils.multi_head_attention import MultiHeadAttention
from utils.feed_forward import FeedForwardNetwork
from utils.positional_encoding import PositionalEncoding

class DecoderLayer(tf.keras.layers.Layer):
    """
    Single decoder layer (POST-LN, original paper).

    x = LayerNorm(x + MaskedMHA(x, x, x, dec_mask))
    x = LayerNorm(x + CrossMHA(x, enc_out, enc_out, enc_mask))
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
                                                   name="decoder_self_attention")
        self.cross_attention = MultiHeadAttention(d_model, num_heads,
                                                   name="decoder_cross_attention")
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout,
                                        name="decoder_ffn")
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                        name="decoder_norm_1")
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                        name="decoder_norm_2")
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                        name="decoder_norm_3")

        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
        self.dropout_3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, dec_mask=None, enc_mask=None, training=False):
        self_attn_out, self_attn_w = self.self_attention(
            x, x, x, mask=dec_mask, training=training)
        self_attn_out = self.dropout_1(self_attn_out, training=training)
        x = self.norm_1(x + self_attn_out)

        cross_attn_out, cross_attn_w = self.cross_attention(
            x, enc_output, enc_output, mask=enc_mask, training=training)
        cross_attn_out = self.dropout_2(cross_attn_out, training=training)
        x = self.norm_2(x + cross_attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout_3(ffn_out, training=training)
        x = self.norm_3(x + ffn_out)

        return x, self_attn_w, cross_attn_w

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads,
                        "d_ff": self.d_ff, "dropout": self.dropout_rate})
        return config


class Decoder(tf.keras.layers.Layer):
    """
    Full decoder stack.
    tgt_ids -> Embedding -> PositionalEncoding -> [DecoderLayer x N]
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 vocab_size, max_len=5000, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.supports_masking = True
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                                                    name="decoder_embedding")
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout,
                                               name="decoder_pos_encoding")
        self.dec_layers   = [
            DecoderLayer(d_model, num_heads, d_ff, dropout,
                         name=f"decoder_layer_{i}")
            for i in range(num_layers)
        ]

    def call(self, tgt_ids, enc_output, dec_mask=None,
             enc_mask=None, training=False):
        x = self.embedding(tgt_ids)
        x = self.pos_encoding(x, training=training)
        attention_weights = {}
        for i, layer in enumerate(self.dec_layers):
            x, self_attn_w, cross_attn_w = layer(
                x, enc_output,
                dec_mask=dec_mask,
                enc_mask=enc_mask,
                training=training,
            )

            attention_weights[f"decoder_layer_{i}_self"]  = self_attn_w
            attention_weights[f"decoder_layer_{i}_cross"] = cross_attn_w
            
        return x, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "d_model": self.d_model})
        return config

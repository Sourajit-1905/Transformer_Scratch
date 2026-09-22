"""
multi_head_attention.py — Multi-Head Attention
Transformer Reproduction Project — Phase 5

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from multi_head_attention import MultiHeadAttention
"""

import tensorflow as tf
from attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention from scratch.
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    Args:
        d_model: model dimension (e.g. 512)
        num_heads: number of attention heads (e.g. 8)
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_Q = tf.keras.layers.Dense(d_model, use_bias=False, name="W_Q")
        self.W_K = tf.keras.layers.Dense(d_model, use_bias=False, name="W_K")
        self.W_V = tf.keras.layers.Dense(d_model, use_bias=False, name="W_V")
        self.W_O = tf.keras.layers.Dense(d_model, use_bias=False, name="W_O")

    def split_heads(self, x, batch_size):
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None, training=False):
        batch_size = tf.shape(q)[0]
        Q_proj = self.W_Q(q)
        K_proj = self.W_K(k)
        V_proj = self.W_V(v)
        Q_heads = self.split_heads(Q_proj, batch_size)
        K_heads = self.split_heads(K_proj, batch_size)
        V_heads = self.split_heads(V_proj, batch_size)
        attn_out, attn_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask=mask
        )
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        seq_q = tf.shape(attn_out)[1]
        attn_out = tf.reshape(
            attn_out, (batch_size, seq_q, self.num_heads * self.d_v)
        )
        output = self.W_O(attn_out)
        return output, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

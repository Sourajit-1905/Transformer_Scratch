"""
attention.py — Scaled Dot-Product Attention

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from attention import scaled_dot_product_attention
"""

import tensorflow as tf

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q   : (batch, heads, seq_q, d_k)
        K   : (batch, heads, seq_k, d_k)
        V   : (batch, heads, seq_k, d_v)
        mask: float32, broadcastable to (batch, heads, seq_q, seq_k)
              1.0 = masked, 0.0 = attended

    Returns:
        output           : (batch, heads, seq_q, d_v)
        attention_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True)

    scores = scores / tf.math.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attention_weights = tf.nn.softmax(scores, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights

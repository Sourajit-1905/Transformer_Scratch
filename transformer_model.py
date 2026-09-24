"""
transformer_model.py - Complete Transformer Model
Transformer Reproduction Project - Phase 10

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from transformer_model import Transformer, TiedTransformer
"""

import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from data_pipeline import create_padding_mask, create_decoder_mask


class Transformer(tf.keras.Model):
    """
    Complete Transformer for sequence-to-sequence translation.
    Vaswani et al. 2017 "Attention Is All You Need"
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 src_vocab, tgt_vocab, max_len=5000, dropout=0.1,
                 tie_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tie_weights = tie_weights

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                               src_vocab, max_len, dropout, name="encoder")
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               tgt_vocab, max_len, dropout, name="decoder")
        self.output_projection = tf.keras.layers.Dense(tgt_vocab,
                                                        use_bias=False,
                                                        name="output_projection")

    def call(self, src_ids, tgt_ids, training=False,
             enc_mask=None, dec_mask=None):
        if enc_mask is None:
            enc_mask = create_padding_mask(src_ids)
        if dec_mask is None:
            dec_mask = create_decoder_mask(tgt_ids, tgt_len=tf.shape(tgt_ids)[1])
        enc_output, enc_attn = self.encoder(src_ids, mask=enc_mask, training=training)
        dec_output, dec_attn = self.decoder(tgt_ids, enc_output,
                                             dec_mask=dec_mask, enc_mask=enc_mask,
                                             training=training)
        logits = self.output_projection(dec_output)
        return logits, {**enc_attn, **dec_attn}

    def build_masks(self, src_ids, tgt_ids):
        enc_mask = create_padding_mask(src_ids)
        dec_mask = create_decoder_mask(tgt_ids, tgt_len=tf.shape(tgt_ids)[1])
        return enc_mask, dec_mask

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "d_model": self.d_model,
                        "num_heads": self.num_heads, "d_ff": self.d_ff,
                        "src_vocab": self.src_vocab, "tgt_vocab": self.tgt_vocab,
                        "tie_weights": self.tie_weights})
        return config


class TiedTransformer(Transformer):
    """
    Transformer with weight tying (paper Section 3.4).
    Output projection uses decoder embedding weights transposed.
    Requires src_vocab == tgt_vocab.
    """

    def call(self, src_ids, tgt_ids, training=False,
             enc_mask=None, dec_mask=None):
        if enc_mask is None:
            enc_mask = create_padding_mask(src_ids)
        if dec_mask is None:
            dec_mask = create_decoder_mask(tgt_ids, tgt_len=tf.shape(tgt_ids)[1])
        enc_output, enc_attn = self.encoder(src_ids, mask=enc_mask, training=training)
        dec_output, dec_attn = self.decoder(tgt_ids, enc_output,
                                             dec_mask=dec_mask, enc_mask=enc_mask,
                                             training=training)
        embedding_weights = self.decoder.embedding.embeddings
        logits = tf.matmul(dec_output, embedding_weights, transpose_b=True)
        return logits, {**enc_attn, **dec_attn}

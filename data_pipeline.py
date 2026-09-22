"""
data_pipeline.py — Transformer Reproduction Project
Phase 2: Data loading and batching utilities.

Usage in a new Colab session:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from data_pipeline import (
        load_raw_pairs, encode_and_wrap, pad_sequence,
        build_tf_dataset, create_padding_mask,
        create_look_ahead_mask, create_decoder_mask,
    )
"""

import numpy as np
import tensorflow as tf
from typing import Callable, List, Tuple

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def load_raw_pairs(src_path, tgt_path):
    with open(src_path, encoding="utf-8") as f:
        src = [l.strip() for l in f if l.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt = [l.strip() for l in f if l.strip()]
    assert len(src) == len(tgt)
    return src, tgt


def encode_and_wrap(text, encode_fn, bos_id=BOS_ID, eos_id=EOS_ID):
    return [bos_id] + encode_fn(text) + [eos_id]


def pad_sequence(ids, max_len, pad_id=PAD_ID):
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def build_tf_dataset(src_lines, tgt_lines, encode_fn, max_len,
                     batch_size, bos_id=BOS_ID, eos_id=EOS_ID,
                     pad_id=PAD_ID, shuffle=True, shuffle_seed=42):
    enc_inputs, dec_inputs, dec_targets = [], [], []
    for src, tgt in zip(src_lines, tgt_lines):
        src_ids = encode_and_wrap(src, encode_fn, bos_id, eos_id)
        src_ids = pad_sequence(src_ids, max_len, pad_id)
        tgt_full = encode_and_wrap(tgt, encode_fn, bos_id, eos_id)
        tgt_in = pad_sequence(tgt_full[:-1], max_len, pad_id)
        tgt_out= pad_sequence(tgt_full[1:],  max_len, pad_id)
        enc_inputs.append(src_ids)
        dec_inputs.append(tgt_in)
        dec_targets.append(tgt_out)
    enc_inputs = np.array(enc_inputs,  dtype=np.int32)
    dec_inputs = np.array(dec_inputs,  dtype=np.int32)
    dec_targets = np.array(dec_targets, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((enc_inputs, dec_inputs, dec_targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(src_lines), seed=shuffle_seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_padding_mask(seq):
    mask = tf.cast(tf.equal(seq, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    ones= tf.ones((size, size))
    lower_tri = tf.linalg.band_part(ones, -1, 0)
    mask= 1.0 - lower_tri
    return mask[tf.newaxis, tf.newaxis, :, :]


def create_decoder_mask(tgt_seq, tgt_len):
    look_ahead = create_look_ahead_mask(tgt_len)
    tgt_pad= create_padding_mask(tgt_seq)
    return tf.maximum(look_ahead, tgt_pad)

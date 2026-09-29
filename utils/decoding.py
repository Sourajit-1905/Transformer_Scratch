"""
decoding.py - Greedy and Beam Search Decoding
Transformer Reproduction Project - Phase 14

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from decoding import greedy_decode, beam_search
"""

import numpy as np
import tensorflow as tf


def encode_source(model, src_text, tokenizer, max_len=100):
    src_ids    = tokenizer.encode_with_special(src_text)[:max_len]
    src_ids    = src_ids + [tokenizer.pad_id] * (max_len - len(src_ids))
    src_tensor = tf.constant([src_ids], dtype=tf.int32)
    enc_mask   = tf.cast(
        tf.equal(src_tensor, tokenizer.pad_id), tf.float32
    )[:, tf.newaxis, tf.newaxis, :]
    enc_output, _ = model.encoder(src_tensor, mask=enc_mask, training=False)
    return enc_output, enc_mask, src_tensor


def decoder_mask_for(dec_ids, pad_id):
    tgt_len    = tf.shape(dec_ids)[1]
    look_ahead = 1.0 - tf.linalg.band_part(
        tf.ones((1, 1, tgt_len, tgt_len)), -1, 0
    )
    pad_mask = tf.cast(
        tf.equal(dec_ids, pad_id), tf.float32
    )[:, tf.newaxis, tf.newaxis, :]
    return tf.maximum(look_ahead, pad_mask)


def project_to_vocab(model, dec_output_last):
    weights = model.decoder.embedding.embeddings
    logits  = tf.matmul(dec_output_last, weights, transpose_b=True)
    return logits[:, 0, :]


def length_penalty(length, alpha=0.6):
    return ((5.0 + length) / 6.0) ** alpha


def greedy_decode(model, src_text, tokenizer,
                  max_len=50, repeat_penalty=2.0, repeat_window=5):
    enc_output, enc_mask, _ = encode_source(model, src_text, tokenizer)
    dec_ids   = tf.constant([[tokenizer.bos_id]], dtype=tf.int32)
    generated = []

    for _ in range(max_len):
        dec_mask   = decoder_mask_for(dec_ids, tokenizer.pad_id)
        dec_output, _ = model.decoder(
            dec_ids, enc_output,
            dec_mask=dec_mask, enc_mask=enc_mask, training=False,
        )
        logits = project_to_vocab(model, dec_output[:, -1:, :]).numpy()[0]
        for tok in set(generated[-repeat_window:]):
            logits[tok] -= repeat_penalty
        next_token = int(np.argmax(logits))
        if next_token == tokenizer.eos_id:
            break
        generated.append(next_token)
        dec_ids = tf.concat(
            [dec_ids, tf.constant([[next_token]], dtype=tf.int32)], axis=1
        )
    return tokenizer.decode(generated)


def beam_search(model, src_text, tokenizer,
                beam_size=4, max_len=50, alpha=0.6, min_len=1):
    enc_output, enc_mask, _ = encode_source(model, src_text, tokenizer)
    beams     = [([tokenizer.bos_id], 0.0)]
    completed = []

    for step in range(max_len):
        all_candidates = []
        for beam_ids, beam_score in beams:
            dec_ids  = tf.constant([beam_ids], dtype=tf.int32)
            dec_mask = decoder_mask_for(dec_ids, tokenizer.pad_id)
            dec_output, _ = model.decoder(
                dec_ids, enc_output,
                dec_mask=dec_mask, enc_mask=enc_mask, training=False,
            )
            logits = project_to_vocab(model, dec_output[:, -1:, :]).numpy()[0]
            log_probs = logits - (
                np.log(np.sum(np.exp(logits - logits.max()))) + logits.max()
            )
            for token in np.argsort(log_probs)[-beam_size:]:
                new_ids   = beam_ids + [int(token)]
                new_score = beam_score + float(log_probs[token])
                if int(token) == tokenizer.eos_id and len(new_ids) > min_len:
                    completed.append((new_ids, new_score / length_penalty(len(new_ids), alpha)))
                else:
                    all_candidates.append((new_ids, new_score))

        if not all_candidates:
            break
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]
        if len(completed) >= beam_size:
            break

    if not completed:
        best_ids = beams[0][0]
    else:
        completed.sort(key=lambda x: x[1], reverse=True)
        best_ids = completed[0][0]

    best_ids = [t for t in best_ids
                if t not in (tokenizer.bos_id, tokenizer.eos_id)]
    return tokenizer.decode(best_ids)

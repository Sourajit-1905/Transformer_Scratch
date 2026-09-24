"""
training.py - Training utilities
Transformer Reproduction Project - Phase 11

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from training import (
        label_smoothing_loss,
        TransformerLRSchedule,
        train_step,
        train,
        compute_validation_loss,
        create_checkpoint_manager,
        restore_checkpoint,
    )
"""

import os
import time
import csv
import numpy as np
import tensorflow as tf

PAD_ID = 0


def label_smoothing_loss(logits, targets, vocab_size, smoothing=0.1, pad_id=PAD_ID):
    one_hot = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
    smoothed = one_hot * (1.0 - smoothing) + (smoothing / vocab_size)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss_per_token = -tf.reduce_sum(smoothed * log_probs, axis=-1)
    non_pad_mask = tf.cast(tf.not_equal(targets, pad_id), dtype=tf.float32)
    loss_per_token = loss_per_token * non_pad_mask
    num_tokens = tf.reduce_sum(non_pad_mask)
    return tf.reduce_sum(loss_per_token) / num_tokens


class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.maximum(tf.cast(step, tf.float32), 1.0)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": int(self.d_model.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy())}


@tf.function
def train_step(model, optimizer, src_ids, dec_input, dec_target, vocab_size):
    with tf.GradientTape() as tape:
        logits, _ = model(src_ids, dec_input, training=True)
        loss = label_smoothing_loss(logits, dec_target, vocab_size)
    gradients, _ = tf.clip_by_global_norm(
        tape.gradient(loss, model.trainable_variables), 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def create_checkpoint_manager(model, optimizer, checkpoint_dir, max_to_keep=5):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir,
                                          max_to_keep=max_to_keep)
    return ckpt, manager


def restore_checkpoint(ckpt, manager):
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Restored: {manager.latest_checkpoint}")
        return True
    print("No checkpoint found. Starting from scratch.")
    return False


def compute_validation_loss(model, val_dataset, vocab_size, max_batches=20):
    total, n = 0.0, 0
    for enc_input, dec_input, dec_target in val_dataset.take(max_batches):
        logits, _ = model(enc_input, dec_input, training=False)
        total += label_smoothing_loss(logits, dec_target, vocab_size).numpy()
        n += 1
    return total / n if n > 0 else float("inf")


def train(model, optimizer, train_dataset, val_dataset, vocab_size,
          checkpoint_dir, total_steps, checkpoint_every=500,
          validate_every=500, log_every=50, initial_step=0):
    ckpt, manager = create_checkpoint_manager(model, optimizer, checkpoint_dir)
    restored = restore_checkpoint(ckpt, manager)
    current_step = optimizer.iterations.numpy() if restored else initial_step
    history = {"step": [], "train_loss": [], "val_loss": []}
    dataset_iter = iter(train_dataset.repeat())
    step, t_start = current_step, time.time()

    while step < total_steps:
        enc_input, dec_input, dec_target = next(dataset_iter)
        loss = train_step(model, optimizer, enc_input,
                          dec_input, dec_target, vocab_size)
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t_start
            steps_done = step - current_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            eta = (total_steps - step) / sps if sps > 0 else 0
            lr_now = optimizer.learning_rate(step).numpy()
            print(f"step {step:>6}  loss={loss.numpy():.4f}  "
                  f"lr={lr_now:.6f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
            history["step"].append(step)
            history["train_loss"].append(float(loss.numpy()))

        if step % validate_every == 0:
            val_loss = compute_validation_loss(model, val_dataset, vocab_size)
            print(f"val loss at step {step}: {val_loss:.4f}")
            history["val_loss"].append((step, float(val_loss)))

        if step % checkpoint_every == 0:
            print(f"checkpoint saved: {manager.save()}")

    print(f"Training complete. Final checkpoint: {manager.save()}")
    return history

# Phase 1: Colab Environment

## What Was Implemented

1. **Google Drive Integration**  
   Automatic mounting of Google Drive and instantiation of the persistent project directory structure.

2. **Environment & Hardware Telemetry**  
   A diagnostic script to read Python, TensorFlow, and Keras versions, along with system RAM, disk space, and exact GPU memory metrics.

3. **Reproducibility Anchors**  
   Global random seed initialization across Python, NumPy, and TensorFlow to ensure behavior is as repeatable as possible on the available hardware.

## Why It Is Required

Google Colab limits the available compute environment, often with constraints such as limited runtime, GPU VRAM, and transient local storage. If the instance disconnects, files stored only on the local runtime can be lost.

By mounting Google Drive immediately and establishing the project folder structure, important artifacts can persist across Colab sessions.

The telemetry script is important because attempting to train a **Transformer Base** model on a CPU or on a GPU with insufficient available memory can result in extremely slow execution or out-of-memory errors.

Setting deterministic seeds also helps minimize run-to-run variation, making it easier to debug complex tensor operations and compare experiments.

---

# Phase 2: Dataset Strategy

## What This Phase Does

Builds the complete, staged data pipeline — from raw text through tokenization, integer encoding, padding, batching, and mask generation.

The staged strategy prevents unnecessary GPU usage on the full **WMT14** dataset before the model and data pipeline have been proven to work correctly.

## Why a Staged Dataset Strategy?

The original *Attention Is All You Need* paper trained on **WMT14**, using approximately **4.5 million sentence pairs** and **8 P100 GPUs for 12 hours**.

Google Colab provides a single GPU in typical sessions, limited RAM, and session lifetimes that can interrupt long-running experiments.

If WMT14 is downloaded and fully preprocessed before verifying that the implementation works, a bug in the model or data pipeline could result in hours of wasted preprocessing and compute time.

The staged approach allows each part of the system to be validated progressively:

| Stage | Dataset | Purpose |
|---|---|---|
| **Stage A** | Tiny handmade dataset | Proves that the pipeline works end-to-end |
| **Stage B** | Small real dataset — OPUS/Tatoeba | Proves that the pipeline works on real parallel data |
| **Stage C** | Medium-sized subset | Validates training speed, memory usage, and stability |
| **Stage D** | WMT14 EN→DE | Serves as the actual reproduction target |

This progression ensures that problems are identified early, before committing significant GPU time to large-scale training.
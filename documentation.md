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
**File:** `02_data_preprocessing.ipynb`

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


# Phase 3 — Tokenization
**File:** `03_tokenization.ipynb`

## Overview
Implements a Byte Pair Encoding (BPE) tokenizer from scratch in pure Python.
No Hugging Face, no SentencePiece, no external tokenization libraries.

---

## What Was Done

### Text Normalisation
- Light cleaning applied before BPE training
- Punctuation spaced out so `"student."` → `"student ."`
- Whitespace collapsed
- German capitalisation preserved (nouns are capitalised in German)

### BPE Training
- Words represented as character sequences with `</w>` end-of-word marker
- Most frequent adjacent symbol pairs merged iteratively
- Repeated until target vocabulary size is reached
- Trained on **combined** source + target lines (shared vocabulary)

### BPE Encoding
- Applies learned merge rules to any new text at inference time
- Falls back to character-level for unseen words — no true `<unk>` unless character itself is unseen

### BPE Decoding
- Joins subword symbols and removes `</w>` markers to recover original words

### `BPETokenizer` Class
- Clean API: `encode()`, `decode()`, `encode_with_special()`, `save()`, `load()`
- Saved as `tokenizer.py` on Drive for reuse across sessions

---

## Two Tokenizers Trained

| Tokenizer | Data | Vocab Size | Purpose |
|---|---|---|---|
| Stage A | 20 handmade sentences | ~200 tokens | Overfitting smoke test (Phase 13) |
| Stage B | Tatoeba ~90k pairs | 8,000 tokens | Real small-scale experiments |

---

## Special Token IDs (Fixed)

| Token | ID |
|---|---|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

---

## Files Saved to Drive

```
transformer_reproduction/
├── vocab/
│   ├── stage_a/
│   │   ├── vocab.json
│   │   └── merges.json
│   └── stage_b/
│       ├── vocab.json
│       └── merges.json
└── tokenizer.py
```

---

## Integration with Phase 2
`build_tf_dataset()` from Phase 2 accepts any `encode_fn(text) → list[int]`.
Phase 3 passes `tokenizer.encode` as that function — no other changes needed.

---

## Verified
- Round-trip `encode → decode` produces original text
- Reload from disk produces identical IDs
- Unknown words segmented to characters, not `<unk>`
- BOS at `decoder_input` position 0 confirmed
- All token IDs within valid vocabulary range
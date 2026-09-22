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

# Phase 4 — Attention From Scratch
**File:** `04_attention.ipynb`

## Overview
Implements **Scaled Dot-Product Attention from scratch** — the core mathematical
building block of the entire Transformer. Every other component depends on this
being correct before proceeding.

---

## Formula Implemented

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
```

Five explicit steps:
1. **Dot product** — `scores = Q @ K^T` → shape `(batch, heads, seq_q, seq_k)`
2. **Scale** — `scores / sqrt(d_k)` → prevents softmax saturation at large d_k
3. **Mask** — `scores + (mask * -1e9)` → drives ignored positions to ~0 after softmax
4. **Softmax** — over key dimension (`axis=-1`) → probability distribution per query
5. **Weighted sum** — `weights @ V` → final output per query position

---

## Transformer Base Dimensions

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `num_heads` | 8 |
| `d_k = d_v` | 512 / 8 = **64** |

---

## What Was Done

### `scaled_dot_product_attention(Q, K, V, mask=None)`
- Implemented the exact formula from Section 3.2.1 of the paper
- Returns both `output` and `attention_weights` (weights saved for Phase 23 visualisation)
- `mask=1.0` at positions to ignore, `mask=0.0` at positions to attend to

### Shape Tests
- No mask: `(batch, heads, seq_q, d_v)` output confirmed
- With padding mask: shapes unchanged, mask broadcasts correctly
- Self-attention (`seq_q == seq_k`): verified

### Numerical Behavior Tests
- Attention weights sum to exactly `1.0` across key dimension
- Masked positions receive weight `< 1e-7`
- Scaling reduces score variance by factor of `d_k` (~64x)
- Output confirmed to be a convex combination of V rows

### Masking Tests
- **Padding mask** — PAD positions (ID=0) receive ~0 attention weight; non-PAD weights sum to 1.0
- **Causal mask** — no query attends to any future key position; all positions attend to self and past

### Gradient Flow Test
- Gradients confirmed non-None, non-NaN, non-zero for Q, K, and V
- Backpropagation flows correctly through the full attention operation

### Visualisations
- Attention weight heatmap (random Q/K/V — shape verification only)
- Causal mask matrix showing which positions each query can attend to

---

## Files Saved to Drive

```
transformer_reproduction/
├── attention.py
└── figures/
    ├── attention_weights_phase4.png
    └── causal_mask_phase4.png
```

---

## Integration with Previous Phases
- Uses `create_padding_mask()` and `create_look_ahead_mask()` from Phase 2
- Uses Stage A `BPETokenizer` from Phase 3 for the visualisation example
- `attention.py` saved to Drive — imported directly in Phase 5

---

## Verified
- All tensor shapes correct for Transformer Base dimensions
- Softmax weights sum to 1.0 (tolerance < 1e-5)
- Masked positions have weight < 1e-7
- Scaling reduces variance ~64x as expected
- Gradients flow through Q, K, and V
- No causal violations in look-ahead mask
- `attention.py` reloads and passes final shape check
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

# Phase 5 — Multi-Head Attention
**File:** `05_multi_head_attention.ipynb`

## Overview
Implements **Multi-Head Attention from scratch** as a `tf.keras.layers.Layer`.
Combines Phase 4's scaled dot-product attention with learned linear projections,
head splitting, and output projection. No `tf.keras.layers.MultiHeadAttention` used.

---

## Formula Implemented

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

Five explicit steps:
1. **Linear projections** — Q, K, V each projected through a Dense layer → `(batch, seq, d_model)`
2. **Split heads** — reshape + transpose → `(batch, num_heads, seq, d_k)`
3. **Scaled dot-product attention** — runs across all heads in parallel (Phase 4)
4. **Concatenate heads** — transpose + reshape → `(batch, seq, d_model)`
5. **Output projection** — final Dense layer W_O → `(batch, seq, d_model)`

---

## Transformer Base Dimensions

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `num_heads` | 8 |
| `d_k = d_v` | 512 / 8 = **64** |
| Parameters per layer | 4 × 512 × 512 = **1,048,576** |

---

## What Was Done

### `MultiHeadAttention` Class
- Four learned projection matrices: `W_Q`, `W_K`, `W_V`, `W_O`
- All Dense layers with `use_bias=False` (following the paper)
- `split_heads()` — reshape `(batch, seq, d_model)` → `(batch, num_heads, seq, d_k)`
- `get_config()` implemented for Keras model saving/loading

### Shape Tests
- No mask, self-attention, padding mask, causal mask — all verified at `d_model=512, num_heads=8`

### d_k Verification
- Explicit check: `512 / 8 = 64` 
- Parameter count confirmed: `4 × 512 × 512 = 1,048,576` 

### Numerical Behavior Tests
- Attention weights sum to `1.0` per head across key dimension
- Different heads produce different attention patterns (not identical)
- PAD positions receive `< 1e-7` weight across **all** heads
- Output confirmed to be a transformation of input (not identity)

### Causal Mask Test
- Verified zero future-token violations across all heads, all batch items

### Gradient Flow Test
- Gradients confirmed for all 3 inputs (Q, K, V)
- Gradients confirmed for all 4 weight matrices (W_Q, W_K, W_V, W_O)

### Three Attention Modes
All three modes used in the Transformer verified explicitly:

| Mode | Q | K | V | Mask |
|---|---|---|---|---|
| Encoder self-attention | encoder input | encoder input | encoder input | src padding |
| Decoder masked self-attention | decoder state | decoder state | decoder state | causal + padding |
| Decoder cross-attention | decoder state | encoder output | encoder output | src padding |

---

## Files Saved to Drive

```
transformer_reproduction/
└── multi_head_attention.py
```

---

## Integration with Previous Phases
- Imports `scaled_dot_product_attention` from `attention.py` (Phase 4)
- Uses `create_padding_mask`, `create_look_ahead_mask`, `create_decoder_mask` from `data_pipeline.py` (Phase 2)
- `multi_head_attention.py` saved to Drive — imported directly in Phases 6, 7, 8

---

## Verified
- All 4 shape tests pass at Transformer Base dimensions
- Parameter count matches expected `1,048,576`
- Heads produce distinct attention patterns
- No causal violations across any head
- All weight matrices receive gradients
- All three attention modes produce correct output shapes
- `multi_head_attention.py` reloads and passes final shape check

# Phase 6 — Positional Encoding
**File:** `06_positional_encoding.ipynb`

## Overview
Implements the **Sinusoidal Positional Encoding** exactly as described in
Section 3.5 of the paper. Since the Transformer has no recurrence or convolution,
this is the only mechanism that informs the model about token order.

---

## Formula Implemented

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

Where:
- `pos` — position in the sequence (0, 1, 2, ..., max_len-1)
- `i` — dimension index (0, 1, ..., d_model/2 - 1)
- Even dimensions → **sine**, Odd dimensions → **cosine**

---

## Key Properties Verified

| Property | Status |
|---|---|
| Fixed (no learned parameters) |
| Values bounded in `[-1, 1]` |
| Each position has a unique encoding vector | 
| PE(pos+k) is a linear function of PE(pos) |
| Generalises beyond training sequence lengths | (pre-computed to max_len=5000) |

---

## Transformer Base Configuration

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `max_len` | 5000 |
| `dropout` | 0.1 (applied after adding PE) |

---

## What Was Done

### `PositionalEncoding` Class
- PE matrix pre-computed **once** at `__init__` using numerically stable formula:
  `exp(-log(10000) * i / d_model)` instead of `10000^(2i/d_model)`
- Sliced to actual `seq_len` at call time — handles variable length sequences
- Embeddings scaled by `sqrt(d_model)` before adding PE (paper Section 3.4)
- Dropout applied after adding PE (paper Section 5.4)
- `supports_masking = True` set to suppress Keras implicit mask warning
- Zero trainable parameters

### Shape and Value Tests
- Output shape correct at Transformer Base dimensions
- PE matrix shape `(1, 5000, 512)` verified
- Even dims = `sin(0) = 0.0` and odd dims = `cos(0) = 1.0` at position 0
- Embedding scaling by `sqrt(512)` verified numerically

### Exact Formula Verification
- Manually computed 3 specific PE values at `d_model=8`
- All match implementation to `< 1e-6`

### Visualisation
- Heatmap of full PE matrix (200 positions × 512 dims)
- Frequency plot showing low dims oscillate fast, high dims oscillate slow
- Saved to `figures/positional_encoding.png`

### Unique Position Test
- 500 randomly sampled positions checked for duplicates — none found
- All adjacent positions confirmed distinct

### Relative Position Property
- Verified numerically that `PE(pos+k)` can be reconstructed as a linear
  function of `PE(pos)` using trigonometric identities
- Error `< 1e-5` for all 4 tested `(pos, k)` pairs

### Gradient Flow Test
- Gradient flows through PE layer back to embedding input
- Gradient value = `sqrt(d_model)` = `sqrt(512)` ≈ `22.627` everywhere (correct,
  since `d(x * sqrt(d_model) + pe) / dx = sqrt(d_model)`)

---

## Files Saved to Drive

```
transformer_reproduction/
├── positional_encoding.py
└── figures/
    └── positional_encoding.png
```

---

## Integration with Previous Phases
- `positional_encoding.py` saved to Drive — imported directly in Phase 8 (Encoder)
- No dependencies on Phase 4 or Phase 5 — standalone layer
- Will be placed immediately after the embedding layer in both encoder and decoder

---

## Verified
- All 6 shape/value sub-tests pass
- All 3 manual formula cross-checks match to `< 1e-6`
- Figure renders and saves with clearly visible sinusoidal pattern
- No duplicate position encodings in 500-position sample
- Relative position reconstruction error `< 1e-5`
- Gradient = `sqrt(512)` confirmed
- `positional_encoding.py` reloads and passes all assertions

# Phase 7 — Feed-Forward Network
**File:** `07_feed_forward.ipynb`

## Overview
Implements the **Position-wise Feed-Forward Network (FFN)** from Section 3.3
of the paper. This is the second sub-layer inside every encoder and decoder layer,
applied independently and identically to each position in the sequence.

---

## Formula Implemented

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

Two linear transformations with a ReLU activation in between.
No interaction between positions — each position is processed independently
using the same weights.

---

## Transformer Base Dimensions

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `d_ff` | 2048 (4x expansion) |
| `dropout` | 0.1 (applied after first dense layer) |
| Parameters per FFN layer | 2,099,712 |

---

## What Was Done

### `FeedForwardNetwork` Class
- Two `tf.keras.layers.Dense` layers: `dense_1` (d_ff, ReLU) and `dense_2` (d_model)
- Dropout applied after the first activation
- `supports_masking = True` set on the layer
- `get_config()` implemented for Keras model saving/loading

### Shape Tests
- Verified correct output shape `(batch, seq_len, d_model)` for variable sequence
  lengths: 1, 5, 50, 128 — all pass

### Parameter Count
- Verified against expected value: `2 × 512 × 2048 + 2048 + 512 = 2,099,712`

### Numerical Behavior Tests
- ReLU confirmed: no negative values after `dense_1`
- Position-wise independence: changing one position does not affect others
- Deterministic at inference (`training=False`): two passes produce identical output
- Stochastic during training (`training=True`): two passes differ due to dropout

### Gradient Flow Test
- Gradient flows through the layer back to the input
- Both weight matrices (`W1`, `W2`) and both biases receive non-zero, non-NaN gradients

### Paper Comparison
- No deviations from Section 3.3
- Formula, dimensions, activation, dropout placement all match exactly

---

## Files Saved to Drive

```
transformer_reproduction/
└── feed_forward.py
```

---

## Integration with Previous Phases
- No dependencies on Phase 4, 5, or 6 — standalone layer
- `feed_forward.py` saved to Drive — imported directly in Phase 8 (Encoder) and
  Phase 9 (Decoder)
- Used as the second sub-layer in every encoder and decoder layer

---

## Verified
- Output shape correct for all tested sequence lengths
- Parameter count matches expected `2,099,712`
- ReLU removes all negative intermediate activations
- Positions confirmed to be processed independently
- Inference is deterministic, training is stochastic
- All gradients non-None and non-zero
- `feed_forward.py` reloads and passes shape and parameter count assertions
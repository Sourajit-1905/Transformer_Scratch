# Transformer from Scratch

> Coded "Attention Is All You Need" (Vaswani et al. 2017) from scratch on a single Google Colab GPU.
> No pretrained models. No library attention layers.

---

## Links

| | |
|---|---|
| LinkedIn post | <!-- add link --> |
| Live dashboard | <!-- add link --> |
| GitHub | <!-- add link --> |
| Colab notebooks | <!-- add link --> |

---

## What was built

Complete Transformer pipeline across 17 phases — scaled dot-product attention,
multi-head attention, sinusoidal positional encoding, POST-LN encoder/decoder,
BPE tokenization, beam search, and BLEU-4 evaluation, all implemented manually in TensorFlow.

- 20/20 exact matches on toy overfit test
- Training loss 5.38 to 2.14 over 10,000 steps in 35 minutes
- 42.51 BLEU on Tatoeba EN-DE validation (greedy decoding)
- 12 reusable Python modules, 9.6M parameters

---

## Constraints vs original paper

The architecture matches the paper exactly.
The gap in results is compute, not implementation.

| | Paper | Ours |
|---|---|---|
| Dataset | WMT14 EN-DE (4.5M pairs) | Tatoeba EN-DE (80k pairs) |
| Training steps | 100,000 | 10,000 |
| Hardware | 8x NVIDIA P100 | 1x Colab GPU |
| BLEU eval set | newstest2014 | Tatoeba validation |
| BLEU score | 27.3 | 42.51 * |

\* Scores are not directly comparable. Tatoeba validation sentences are shorter
and simpler than WMT14 newstest2014.

---

## Tech stack

TensorFlow 2 / Keras / Python / Google Colab / NumPy / Matplotlib

---

## Phase documentation

| Phase | File | What was done |
|---|---|---|
| 2 | [Data Preprocessing](./documentation.md#phase-2-dataset-strategy) | Staged data pipeline, masking, tf.data |
| 3 | [Tokenization](./documentation.md#phase-3--tokenization) | BPE from scratch, shared vocab |
| 4 | [Attention](./documentation.md#phase-4--attention-from-scratch) | Scaled dot-product attention |
| 5 | [Multi-Head Attention](./documentation.md#phase-5--multi-head-attention) | 4 projections, split/merge heads |
| 6 | [Positional Encoding](./documentation.md#phase-6--positional-encoding) | Sinusoidal PE, exact paper formula |
| 7 | [Feed-Forward Network](./documentation.md#phase-7--feed-forward-network) | FFN(x) = ReLU(xW1+b1)W2+b2 |
| 8 | [Encoder](./documentation.md#phase-8--encoder) | POST-LN encoder stack |
| 9 | [Decoder](./documentation.md#phase-9--decoder) | POST-LN decoder, cross-attention |
| 10 | [Complete Transformer](./documentation.md#phase-10--complete-transformer) | Complete model, weight tying |
| 11 | [Training Loss and Loop](./documentation.md#phase-11--training-loss-and-loop) | Loss, LR schedule, training loop |
| 12 | [Toy Overfitting](./documentation.md#phase-12--toy-overfitting) | Pipeline verification, 20/20 matches |
| 13 | [Small Scale Experiment](./documentation.md#phase-13--small-scale-experiment) | Tatoeba training run |
| 14 | [Decoding](./documentation.md#phase-14--decoding) | Greedy and beam search |
| 15 | [BLEU Evaluation](./documentation.md#phase-15--bleu-evaluation) | BLEU-4 from scratch |
| 16 | [Attention Visualization](./documentation.md#phase-16--attention-visualization) | Heatmaps, all three attention types |

---

## Project structure

```
Transformer_Mine/
├── notebooks/          17 Colab notebooks, one per phase
├── checkpoints/        h5 weights + TF checkpoints
├── datasets/           Stage A (handmade) and Stage B (Tatoeba)
├── vocab/              BPE vocab and merge rules
├── figures/            Training curves, attention heatmaps
├── translations/       Greedy and beam search outputs
├── results/            BLEU JSON results
├── experiments/        results.csv experiment log
└── *.py                12 reusable Python modules
```

---

## Citation

```
@article{vaswani2017attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
             Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
             Kaiser, Lukasz and Polosukhin, Illia},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2017}
}
```
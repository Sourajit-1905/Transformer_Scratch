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
| 2 | [Data Preprocessing](docs/phase2_documentation.md) | Staged data pipeline, masking, tf.data |
| 3 | [Tokenization](docs/phase3_documentation.md) | BPE from scratch, shared vocab |
| 4 | [Attention](docs/phase4_documentation.md) | Scaled dot-product attention |
| 5 | [Multi-Head Attention](docs/phase5_documentation.md) | 4 projections, split/merge heads |
| 6 | [Positional Encoding](docs/phase6_documentation.md) | Sinusoidal PE, exact paper formula |
| 7 | [Feed-Forward Network](docs/phase7_documentation.md) | FFN(x) = ReLU(xW1+b1)W2+b2 |
| 8 | [Encoder](docs/phase8_documentation.md) | POST-LN encoder stack |
| 9 | [Decoder](docs/phase9_documentation.md) | POST-LN decoder, cross-attention |
| 10 | [Transformer](docs/phase10_documentation.md) | Complete model, weight tying |
| 11 | [Training](docs/phase11_documentation.md) | Loss, LR schedule, training loop |
| 12 | [Toy Overfit](docs/phase12_documentation.md) | Pipeline verification, 20/20 matches |
| 13 | [Small Scale Experiment](docs/phase13_documentation.md) | Tatoeba training run |
| 14 | [Decoding](docs/phase14_documentation.md) | Greedy and beam search |
| 15 | [BLEU Evaluation](docs/phase15_documentation.md) | BLEU-4 from scratch |
| 16 | [Attention Visualization](docs/phase16_documentation.md) | Heatmaps, all three attention types |
| 17 | Final Results | Reproduction assessment and dashboard |

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
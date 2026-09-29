"""
bleu.py - BLEU Evaluation from Scratch
Transformer Reproduction Project - Phase 15

Usage:
    import sys
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Transformer_Mine
    from bleu import corpus_bleu, sentence_bleu, normalise_for_bleu
"""

import re
import math
import collections


def normalise_for_bleu(text):
    """Space out punctuation to match BPE tokenizer output."""
    text = re.sub(r'([\.,!?;:\(\)\[\]{}\"\'\-/])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_ngrams(tokens, n):
    return collections.Counter(
        tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
    )


def modified_precision(hypotheses, references, n):
    clipped_count = 0
    total_count   = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_ngrams = get_ngrams(hyp, n)
        ref_ngrams = get_ngrams(ref, n)
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
        total_count += max(len(hyp) - n + 1, 0)
    if total_count == 0:
        return 0.0
    return clipped_count / total_count


def brevity_penalty(hypotheses, references):
    c = sum(len(h) for h in hypotheses)
    r = sum(len(r) for r in references)
    if c == 0:
        return 0.0
    if c >= r:
        return 1.0
    return math.exp(1 - r / c)


def corpus_bleu(hypotheses, references, max_n=4, weights=None):
    if weights is None:
        weights = [1.0 / max_n] * max_n
    hyp_tokens = [h.strip().split() for h in hypotheses]
    ref_tokens = [r.strip().split() for r in references]
    precisions = [modified_precision(hyp_tokens, ref_tokens, n)
                  for n in range(1, max_n + 1)]
    bp   = brevity_penalty(hyp_tokens, ref_tokens)
    if any(p == 0.0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * math.exp(
            sum(w * math.log(p) for w, p in zip(weights, precisions))
        ) * 100
    stats = {
        'BLEU': round(bleu, 2),
        'BP'  : round(bp, 4),
        'p1'  : round(precisions[0] * 100, 2),
        'p2'  : round(precisions[1] * 100, 2),
        'p3'  : round(precisions[2] * 100, 2),
        'p4'  : round(precisions[3] * 100, 2),
    }
    return bleu, stats


def sentence_bleu(hypothesis, reference, max_n=4):
    hyp_tokens = hypothesis.strip().split()
    ref_tokens = reference.strip().split()
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        clipped    = sum(min(c, ref_ngrams.get(ng, 0))
                         for ng, c in hyp_ngrams.items())
        total      = max(len(hyp_tokens) - n + 1, 0)
        precisions.append((clipped + 1) / (total + 1))
    bp      = brevity_penalty([hyp_tokens], [ref_tokens])
    log_avg = sum(0.25 * math.log(p) for p in precisions)
    return round(bp * math.exp(log_avg) * 100, 2)

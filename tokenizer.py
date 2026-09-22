"""
tokenizer.py — BPE Tokenizer for Transformer Reproduction Project
Phase 3: Byte Pair Encoding from scratch.

Usage:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/transformer_reproduction")
    from tokenizer import BPETokenizer, normalise
    
    tokenizer = BPETokenizer.load("vocab/stage_b")
    ids  = tokenizer.encode("I am a student .")
    text = tokenizer.decode(ids)
"""

import re, json, os, collections

END_OF_WORD    = "</w>"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def normalise(text, lowercase=False):
    if lowercase:
        text = text.lower()
    text = re.sub(r'([\.,!?;:\(\)\[\]{}\"\'\-/])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_to_word_freqs(lines, normalise_fn=normalise):
    freq = collections.Counter()
    for line in lines:
        for word in normalise_fn(line).split():
            chars = tuple(list(word[:-1]) + [word[-1] + END_OF_WORD])
            freq[chars] += 1
    return freq

def get_pair_freqs(word_freqs):
    pairs = collections.Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    new_freqs = collections.Counter()
    a, b = pair
    merged = a + b
    for word, freq in word_freqs.items():
        new_word, i = [], 0
        while i < len(word):
            if i < len(word)-1 and word[i] == a and word[i+1] == b:
                new_word.append(merged); i += 2
            else:
                new_word.append(word[i]); i += 1
        new_freqs[tuple(new_word)] += freq
    return new_freqs

def train_bpe(lines, vocab_size, normalise_fn=normalise, verbose=True):
    word_freqs  = text_to_word_freqs(lines, normalise_fn)
    base_vocab  = {s for word in word_freqs for s in word}
    merges, vocab = [], set(base_vocab)
    target = max(0, vocab_size - len(SPECIAL_TOKENS) - len(base_vocab))
    for i in range(target):
        pf = get_pair_freqs(word_freqs)
        if not pf or max(pf.values()) < 2: break
        best = max(pf, key=pf.get)
        merges.append(best); vocab.add(best[0]+best[1])
        word_freqs = merge_pair(best, word_freqs)
        if verbose and (i+1) % max(1, target//10) == 0:
            print(f"  Merge {i+1}/{target}: {best[0]}+{best[1]}")
    return merges, vocab, word_freqs

def encode_word(word, merge_rules):
    if not word: return []
    symbols = list(word[:-1]) + [word[-1] + END_OF_WORD]
    for (a, b) in merge_rules:
        merged, i, new = a+b, 0, []
        while i < len(symbols):
            if i < len(symbols)-1 and symbols[i]==a and symbols[i+1]==b:
                new.append(merged); i += 2
            else:
                new.append(symbols[i]); i += 1
        symbols = new
    return symbols

def encode_line(text, merge_rules, token2id, normalise_fn=normalise, unk_id=UNK_ID):
    return [token2id.get(sw, unk_id)
            for word in normalise_fn(text).split()
            for sw in encode_word(word, merge_rules)]

def decode_ids(ids, id2token, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID):
    skip = {pad_id, bos_id, eos_id}
    text = "".join(id2token.get(i, "<unk>") for i in ids if i not in skip)
    return text.replace(END_OF_WORD, " ").strip()

class BPETokenizer:
    SPECIAL_TOKENS = SPECIAL_TOKENS
    PAD_ID=0; BOS_ID=1; EOS_ID=2; UNK_ID=3
    END_OF_WORD = END_OF_WORD

    def __init__(self):
        self.merge_rules=[]; self.token2id={}; self.id2token={}
        self.vocab_size=0; self._trained=False

    def train(self, lines, vocab_size, normalise_fn=normalise, verbose=True):
        merges, sym_vocab, _ = train_bpe(lines, vocab_size, normalise_fn, verbose)
        self.merge_rules = merges
        all_syms = self.SPECIAL_TOKENS + sorted(sym_vocab)
        self.token2id = {s:i for i,s in enumerate(all_syms)}
        self.id2token = {i:s for s,i in self.token2id.items()}
        self.vocab_size = len(self.token2id); self._trained = True

    def encode(self, text, normalise_fn=normalise):
        return encode_line(text, self.merge_rules, self.token2id, normalise_fn, self.UNK_ID)

    def encode_with_special(self, text):
        return [self.BOS_ID] + self.encode(text) + [self.EOS_ID]

    def decode(self, ids):
        return decode_ids(ids, self.id2token, self.PAD_ID, self.BOS_ID, self.EOS_ID)

    @property
    def pad_id(self): return self.PAD_ID
    @property
    def bos_id(self): return self.BOS_ID
    @property
    def eos_id(self): return self.EOS_ID
    @property
    def unk_id(self): return self.UNK_ID

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory,"vocab.json"),"w",encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        with open(os.path.join(directory,"merges.json"),"w",encoding="utf-8") as f:
            json.dump(self.merge_rules, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory):
        t = cls()
        with open(os.path.join(directory,"vocab.json"),encoding="utf-8") as f:
            t.token2id = json.load(f)
        t.id2token = {int(i):s for s,i in t.token2id.items()}
        with open(os.path.join(directory,"merges.json"),encoding="utf-8") as f:
            t.merge_rules = [tuple(p) for p in json.load(f)]
        t.vocab_size = len(t.token2id); t._trained = True
        return t

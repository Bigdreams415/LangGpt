import re
import pandas as pd
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.bpe_codes = {}

    def get_vocab(self, corpus):
        vocab = Counter()
        for word in corpus:
            word = " ".join(list(word)) + " </w>"
            vocab[word] += 1
        return vocab
    
    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = re.sub(pattern, replacement, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, corpus):
        vocab = self.get_vocab(corpus)
        for i in range(self.vocab_size):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            vocab = self.merge_vocab(best, vocab)

    def encode_word(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            candidates = [pair for pair in pairs if pair in self.bpe_codes]
            if not candidates:
                break
            best = min(candidates, key=lambda pair: self.bpe_codes[pair])
            i = pairs.index(best)
            word = word[:i] + [''.join(best)] + word[i+2:]
        return word

    def encode_sentence(self, sentence):
        return [self.encode_word(word) for word in sentence.strip().split()]

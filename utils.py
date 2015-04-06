__author__ = 'Matias'

from collections import Counter
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class Encoder(object):
    """
    Encodes words as unique one-hot column vectors
    """
    def __init__(self, sentences):
        self.counter = Counter()
        for sentence in sentences:
            self.counter.update(sentence)
        self.word2id = {word[0]: i for i, word in enumerate(self.counter.items())}
        self.id2word = {i: word[0] for i, word in enumerate(self.counter.items())}
        self.encoding_length = len(self.word2id)

    def word2onehot(self, word):
        word = word.lower()
        idx = self.word2id[word]
        onehot = np.asmatrix(np.zeros(shape=(1, self.encoding_length)))
        onehot[0, idx] = 1.0
        return onehot.T

    def onehot2word(self, onehot):
        idx = np.argmax(onehot)
        return self.id2word[idx]


def sentence2contexts(tokens, window):
    context_pairs = []
    for i, t in enumerate(tokens):
        left_context = tokens[max(0, i-window):max(0, i)]
        right_context = tokens[(i+1):(i+window+1)]
        context = left_context + right_context
        context_pairs.append((t, context))
    return context_pairs



class SentenceStream(object):
    def __init__(self, folder="data/lovecraft", is_test=True):
        self.files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        self.is_test = is_test

    def __iter__(self):
        count = 0
        for f in self.files:
            with open(f, 'r') as textfile:
                text = textfile.read().lower()
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    count += 1
                    sentence_tokens = word_tokenize(sentence)
                    yield sentence_tokens
                    if self.is_test and count > 1000:
                        break


if __name__ == "__main__":
    sents = SentenceStream(is_test=True)
    encoder = Encoder(sents)
    print encoder.encoding_length
    print np.sum(encoder.word2onehot("is"))

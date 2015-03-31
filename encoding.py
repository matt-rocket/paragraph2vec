__author__ = 'Matias'

from collections import Counter
from huffman import huffman_codes
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class Encoder(object):
    def __init__(self, sentences):
        self.counter = Counter()
        for sentence in sentences:
            self.counter.update(sentence)
        self.word2id = {word[0]: i for i, word in enumerate(self.counter.items())}
        self.encoding_length = len(self.word2id)

    def word2onehot(self, word):
        word = word.lower()
        idx = self.word2id[word]
        onehot = np.zeros(shape=(1, self.encoding_length))
        onehot[0][idx] = 1.0
        return onehot.T


class SentenceStream(object):
    def __init__(self, folder="data/lovecraft"):
        self.files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    def __iter__(self):
        for f in self.files:
            with open(f, 'r') as textfile:
                text = textfile.read().lower()
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    sentence_tokens = word_tokenize(sentence)
                    yield sentence_tokens
            break

if __name__ == "__main__":
    sents = SentenceStream()
    encoder = Encoder(sents)
    print encoder.encoding_length
    print np.sum(encoder.word2onehot("are"))

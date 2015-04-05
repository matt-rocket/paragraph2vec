__author__ = 'Matias'

import numpy as np
from encoding import Encoder, SentenceStream
import logging

"""
class NeuralNetworkLanguageModel(object):

    def __init__(self, context=1, encoding_length=3, cbow=True, concat=True):
        self.cbow = cbow
        self.context = context
        self.encoding_length = encoding_length
        self.hidden_units = 5
        self.output_units = 1 if self.cbow else context
        self.input_units = context if self.cbow else 1
        self.input2hidden = np.random.rand(self.hidden_units, self.input_units*self.encoding_length)*0.1
        self.hidden2output = np.random.rand(self.output_units*self.encoding_length, self.hidden_units)*0.1

    def _forward_pass(self, x):
        h = 1.0/self.context * self.input2hidden*x if self.cbow else self.input2hidden*x
        return h, softmax(self.hidden2output*h)

    def _backprop(self, x, t, y, h, alpha):
        # e = dE/dz
        e = y-t
        # dE/dw'
        dEdw_prime = e * h.T
        # dE/dh
        dEdh = self.hidden2output.T*e
        dEdw = dEdh * x.T
        # update W' and W
        self.hidden2output = self.hidden2output - alpha*dEdw_prime
        self.input2hidden = self.input2hidden - 1.0/self.context*alpha*dEdw if self.cbow else self.input2hidden - alpha*dEdw

    def _train(self, x, t):
        h, y = self._forward_pass(x)
        alpha = 0.1
        self._backprop(x, t, y, h, alpha)
"""


class CBOW(object):

    def __init__(self, sentences, context=1, hidden=5, concat=False, limit=1000):
        logging.info(msg="starting CBOW training..")
        self.context = context
        self.encoder = Encoder(sentences=sentences)
        self.encoding_length = self.encoder.encoding_length
        self.hidden_units = hidden
        self.output_units = 1
        self.input_units = context if concat else 1
        self.input2hidden = np.random.rand(self.hidden_units, self.input_units*self.encoding_length)*0.1
        self.hidden2output = np.random.rand(self.output_units*self.encoding_length, self.hidden_units)*0.1

        # train model
        word_count = 0
        for sentence in sentences:
            context_pairs = sentence2contexts(sentence, self.context)
            for w, c in context_pairs:
                self._train(w, c)
                word_count += 1
                if word_count % 100 == 0:
                    logging.info(msg="trained on %s words" % (word_count,))
            if word_count > limit:
                break

    def _forward_pass(self, x):
        h = (1.0/self.context) * self.input2hidden * x
        return h, softmax(self.hidden2output*h)

    def _backprop(self, x, t, y, h, alpha):
        # e = dE/dz
        e = y-t
        # dE/dw'
        dEdw_prime = e * h.T
        # dE/dh
        dEdh = self.hidden2output.T*e
        dEdw = dEdh * x.T
        # update W' and W
        self.hidden2output -= alpha*dEdw_prime
        self.input2hidden -= 1.0/self.context*alpha*dEdw

    def _train(self, word, context):
        onehot_context = [self.encoder.word2onehot(w) for w in context]
        onehot_word = self.encoder.word2onehot(word)
        t = onehot_word
        x = np.zeros_like(onehot_word)
        for c in onehot_context:
            x += c
        h, y = self._forward_pass(x)
        alpha = 0.1
        self._backprop(x, t, y, h, alpha)

    def predict(self, context):
        onehot_context = [self.encoder.word2onehot(w) for w in context]
        x = np.zeros_like(onehot_context[0])
        for c in onehot_context:
            x += c
        _, y = self._forward_pass(x)
        return y

    def __getitem__(self, word):
        onehot_word = self.encoder.word2onehot(word)
        return self.input2hidden*onehot_word



def sentence2contexts(tokens, window):
    context_pairs = []
    for i, t in enumerate(tokens):
        left_context = tokens[max(0, i-window):max(0, i)]
        right_context = tokens[(i+1):(i+window+1)]
        context = left_context + right_context
        context_pairs.append((t, context))
    return context_pairs


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator, 0)
    return numerator/denominator


if __name__ == "__main__":
    # setup logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    stream = SentenceStream()
    cbow_p2v = CBOW(stream,  context=1, hidden=20)
__author__ = 'Matias'

from numba import jit
import numpy as np
from utils import Encoder, SentenceStream, sentence2contexts
from huffman import HuffmanEncoder
import logging
import time

class CBOW(object):

    def __init__(self, sentences, context=1, hidden=5, concat=False):
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
        last_time = time.time()
        for sentence in sentences:
            context_pairs = sentence2contexts(sentence, self.context)
            for w, c in context_pairs:
                self._train(w, c)
                word_count += 1
                if word_count % 100 == 0:
                    now = time.time()
                    time_spent = 1.0/(now - last_time)*100
                    logging.info(msg="trained on %s words. %s words/sec" % (word_count, time_spent))
                    last_time = time.time()


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


class HierarchicalSoftmaxCBOW(object):

    def __init__(self, sentences, context=1, hidden=5, concat=False):
        logging.info(msg="starting CBOW training..")
        self.context = context
        self.encoder = Encoder(sentences=sentences)
        self.huffman_encoder = HuffmanEncoder(self.encoder.counter)
        self.encoding_length = self.encoder.encoding_length
        self.hidden_units = hidden
        self.output_units = 1
        self.input_units = context if concat else 1
        self.input2hidden = np.random.rand(self.hidden_units, self.input_units*self.encoding_length)*0.1
        self.hidden2output = np.random.rand(self.output_units*self.encoding_length-1, self.hidden_units)*0.1

        # train model
        word_count = 0
        last_time = time.time()
        for sentence in sentences:
            context_pairs = sentence2contexts(sentence, self.context)
            for w, c in context_pairs:
                self._train(w, c)
                # break
                word_count += 1
                if word_count % 100 == 0:
                    now = time.time()
                    time_spent = 1.0/(now - last_time)*100
                    logging.info(msg="trained on %s words. %s words/sec" % (word_count, time_spent))
                    last_time = time.time()

    def _train(self, word, context):
        onehot_context = [self.encoder.word2onehot(w) for w in context]
        onehot_word = self.encoder.word2onehot(word)
        t = onehot_word
        x = np.zeros_like(onehot_word)
        for c in onehot_context:
            x += c
        # forward pass
        h = (1.0/self.context)*self.input2hidden * x
        # probability of target word
        node_ids = self.huffman_encoder.get_internal_node_ids(word)
        huffman_code = self.huffman_encoder.get_code(word)
        indicator_vec = np.matrix([1.0 if e == "1" else 0.0 for e in huffman_code])
        alpha = 0.1
        # dE/dw'h
        dEdw_prime_h = np.empty_like(indicator_vec)
        # dE/dh
        dEdh = np.zeros_like(h)
        for j, idx in enumerate(node_ids):
            dEdw_prime_h[:, j] = (logit(self.hidden2output[idx].T*h) - indicator_vec[:, j])
            # (equation 52 - 54)
            dEdh_component = np.multiply(dEdw_prime_h[:, j],self.hidden2output[idx]).T
            dEdh = dEdh + dEdh_component

            # update w_j_prime (Equation 51)
            self.hidden2output[idx] = self.hidden2output[idx] - np.asarray(alpha*dEdw_prime_h[:, j] * h.T)

        dEdw = dEdh * x.T

        # update W' and W
        # self.hidden2output -= alpha*dEdw_prime
        self.input2hidden -= 1.0/self.context*alpha*dEdw

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


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator, 0)
    return numerator/denominator


def logit(z):
    return 1.0/(1.0 + np.exp(-z))


if __name__ == "__main__":
    # setup logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    stream = SentenceStream(is_test=True)
    hscbow_p2v = HierarchicalSoftmaxCBOW(stream,  context=4, hidden=10)
    # cbow_p2v = CBOW(stream,  context=4, hidden=10)
__author__ = 'Matias'

import numpy as np


class Paragraph2Vec(object):

    def __init__(self, window=2):
        self.encoding_length = 3
        self.window_size = window
        self.hidden_units = 5
        self.input2hidden = np.random.rand(self.hidden_units, self.window_size*self.encoding_length)
        self.hidden2output = np.random.rand(self.encoding_length, self.hidden_units)

    def _forward_pass(self, x):
        return softmax(self.hidden2output*(self.input2hidden*x))

    def _backprop(self):
        pass


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator,0)
    return numerator/denominator

if __name__ == "__main__":
    p2v = Paragraph2Vec()

    x = np.mat([0,1,0,0,0,0]).T # 2-word input
    print p2v._forward_pass(x)

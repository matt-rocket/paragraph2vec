__author__ = 'Matias'

import numpy as np
import time


class Paragraph2Vec(object):

    def __init__(self, window=2):
        self.encoding_length = 3
        self.window_size = window
        self.hidden_units = 5
        self.input2hidden = np.random.rand(self.hidden_units, self.window_size*self.encoding_length)*0.1
        self.hidden2output = np.random.rand(self.encoding_length, self.hidden_units)*0.1

    def _forward_pass(self, x):
        h = self.input2hidden*x
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
        self.input2hidden = self.input2hidden - alpha*dEdw

    def _train(self, x, t):
        h, y = self._forward_pass(x)
        alpha = 0.1
        self._backprop(x, t, y, h, alpha)


def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator, 0)
    return numerator/denominator

if __name__ == "__main__":
    p2v = Paragraph2Vec()

    x = np.mat([0,1,0,0,0,0]).T # 2-word input
    t = np.mat([1,0,0]).T # 2-word input

    #print p2v.hidden2output
    start = time.time()
    for i in range(200):
        p2v._train(x, t)
        h, y = p2v._forward_pass(x)
        #print y.T
    #print p2v.hidden2output
    end = time.time()
    print (end - start)*1000

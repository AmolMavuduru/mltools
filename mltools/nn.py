import numpy as np

class NeuralNet(object):
    def __init__ (self, in_size, hid_size, out_size):
        self.in_size = in_size+1
        self.hid_size = hid_size
        self.out_size = out_size

        # Init weights
        # ------------

        # Inputs to outputs
        self.w_inp_out = np.random.rand(self.in_size, self.out_size) # plus 1 for bias

    def sigmoid (self, x):
        return np.divide(1., np.add(1, np.exp(-x)))

    def threshold(self, x):
        return np.array([ 1 if i > .5 else 0 for i in x])

    def activate (self, inputs):
        inputs = np.concatenate((inputs, np.array([1.])), axis=0)
        in_b = np.array( inputs ) # inputs and bias

        a = np.dot(in_b, self.w_inp_out)
        return self.threshold( a )

import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase:
            self.dropout = np.ones_like(input_tensor)
        else:
            # random mask 1 with probability,0 with 1-probability.multiply 1/p during training
            self.dropout = (np.random.rand(*input_tensor.shape)<self.probability)/self.probability

        return input_tensor * self.dropout

    def backward(self, error_tensor):
        return error_tensor * self.dropout
import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        #exp(x_i - max(x)) / sum(exp(x_i - max(x)))
        exp_x = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y_hat

    def backward(self, error_tensor):
        return self.y_hat*(error_tensor-np.sum(error_tensor*self.y_hat, axis=1, keepdims=True))

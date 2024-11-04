import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #ReLU(x) = max(0, x)
        return np.maximum(input_tensor,0)

    def backward(self, error_tensor):
        #e_(n-1) = e_n * (x > 0)
        return error_tensor*(self.input_tensor > 0)

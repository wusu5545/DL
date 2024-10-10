import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        # Flatten the input tensor to a 2D array
        # Preserve batch dimension, combine all other dimensions
        return np.reshape(input_tensor, (input_tensor.shape[0], -1))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)
import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):

    def forward(self, input_tensor):
        self.output = np.tanh(input_tensor)
        return self.output

    def backward(self, error_tensor):
        #gradient of tanh is 1 - tanh^2
        return (1 - np.square(self.output))*error_tensor
import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):

    def forward(self, input_tensor):
        self.output = 1/(1+np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        #derivative of sigmoid is sigmoid(x)*(1-sigmoid(x))
        return self.output*(1-self.output)*error_tensor
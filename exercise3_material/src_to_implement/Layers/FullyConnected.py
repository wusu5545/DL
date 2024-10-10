import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        #input_size + 1 for bias
        self.weights = np.random.uniform(0,1,(input_size + 1, output_size))
        self.input_tensor = None
        self.output_tensor = None
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        # Add bias term (column of ones) to the input tensor
        self.input_tensor = np.append(input_tensor,np.ones((input_tensor.shape[0],1)),axis = 1)
        #batch_size * m+1 dot m+1*n = batch_size * n
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        #X.T dot error_tensor, (batch_size * m+1).T dot batch_size * n = m+1 * n
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # W(t+1)= W(t) - learning_rate * gradient_weights
        if self._optimizer:#ensure that the optimizer has been correctly initialized and set before using it
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        #return error_tensor_(n-1) same shape as X，weights without bias
        return np.dot(error_tensor,self.weights[:-1,:].T)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,self.input_size,self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape,1,self.output_size)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,value):
        self._gradient_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self,weights):
        self._weights = weights
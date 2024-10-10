import numpy as np
from Layers.Base import BaseLayer
from Layers import Helpers
import copy

from scipy.misc import derivative
from sympy import false


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.initialize()
        self._optimizer = None
        self.decay = 0.8
        self.running_mean = None
        self.running_var = None

    def initialize(self, weight_initializer=None,bias_initializer=None):
        self.gamma = np.ones(self.channels) #weights
        self.beta = np.zeros(self.channels) #bias

    def forward(self, input_tensor):
        is_cov = false
        if input_tensor.ndim == 4:
            is_cov = True
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor

        if self.testing_phase:
            self.mean = self.running_mean
            self.var = self.running_var
        else:
            self.mean = np.mean(input_tensor, axis=0, keepdims=True)
            self.var = np.var(input_tensor, axis=0, keepdims=True)
            if self.running_mean is None:
                self.running_mean = self.mean
                self.running_var = self.var
            else:
                self.running_mean = self.decay * self.running_mean + (1 - self.decay) * self.mean
                self.running_var = self.decay * self.running_var + (1 - self.decay) * self.var

        self.normalized_input = (self.input_tensor - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
        output = self.gamma * self.normalized_input + self.beta

        if is_cov:
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        is_cov = False
        if error_tensor.ndim == 4:
            is_cov = True
            error_tensor = self.reformat(error_tensor)

        derivative_gamma = np.sum(error_tensor * self.normalized_input, axis=0)
        derivative_beta = np.sum(error_tensor, axis=0)

        if self._optimizer is not None:
            self._optimizer.weights.calculate_update(self.gamma, derivative_gamma)
            self._optimizer.bias.calculate_update(self.beta, derivative_beta)

        # gradient_input
        output = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.gamma, self.mean, self.var)

        self.gradient_weights = derivative_gamma
        self.gradient_bias = derivative_beta

        if is_cov:
            output = self.reformat(output)

        return  output

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.unreformat_shape = tensor.shape
            batch, channel, M, N = tensor.shape
            tensor = tensor.reshape(batch, channel, M * N)
            tensor = tensor.transpose(0, 2, 1) # batch, M*N, channel
            tensor = tensor.reshape(batch * M * N, channel)
        else:
            batch, channel, M, N = self.unreformat_shape
            tensor = tensor.reshape(batch, M * N, channel)
            tensor = tensor.transpose(0, 2, 1) # batch, channel, M*N
            tensor = tensor.reshape(batch, channel, M, N)

        return tensor

    @property
    def weights(self):
        return self.gamma
    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
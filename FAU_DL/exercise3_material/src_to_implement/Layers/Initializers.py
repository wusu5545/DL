import numpy as np

#For fully connected layers:
#fan_in: input dimension of the weights
#fan_out: output dimension of the weights
#For convolutional layers:
#fan_in: [ # input_channels × kernel_height × kernel_width]
#fan_out: [ # output_channels × kernel_height × kernel_width]

class Constant:
    def __init__(self,constant_value: float = 0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant_value)

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1, weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/(fan_in + fan_out)), weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2/fan_in), weights_shape)

import numpy as np
from Layers.Base import BaseLayer
import copy

from scipy.signal import correlate,convolve

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels

        #initialize stride and convolution shape
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else:
            if len(stride_shape) == 1:#(1,)
                self.stride_shape = (stride_shape[0], stride_shape[0])
            else:
                self.stride_shape = stride_shape

        if len(convolution_shape) == 3:
            self.is_2d = True
            self.convolution_shape = convolution_shape
        else:
            self.is_2d = False
            self.convolution_shape = (*convolution_shape, 1)#* Unpacks the elements

        self.weights = np.random.uniform(0, 1, (num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self.gradient_weights = None
        self.gradient_bias = None

        self.input_tensor = None
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        #padded_input_size = input_size + filter_size -1
        #output_size = (n+2p-f)/s+1, n is the input_size, p is pad_size, f is filter_size, s is stride
        #pad_size = filter_size // 2  if even, - (filter_size-1)%2 at start
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor,axis = -1)#(batch_size,channel_size,input_size,1)

        batch_size, channel_size, _, _ = input_tensor.shape
        input_size = np.array(input_tensor.shape[2:])
        filter_size = np.array(self.convolution_shape[1:])
        pad_size = filter_size // 2 # up,left
        pad_size_even = (filter_size - 1) % 2
        output_size = (input_size - 1) // np.array(self.stride_shape) + 1
        """
            top     =   p(0)-p_e(0)             bottom      =       p(0)
            left    =   p(1)-p_e(1)             right       =       p(1) 
        """
        top = pad_size[0] - pad_size_even[0]
        bottom = pad_size[0]
        left = pad_size[1] - pad_size_even[1]
        right = pad_size[1]
        padded_input = np.zeros((batch_size,channel_size,input_size[0] + filter_size[0] - 1, input_size[1] + filter_size[1] - 1))
        padded_input[:,:,top:top + input_size[0],left:left + input_size[1]] = input_tensor
        #padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant')
        output_tensor = np.zeros((batch_size, self.num_kernels, *output_size))

        self.output_shape = output_tensor.shape

        #correlate input and wights/filter
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(channel_size):
                    correlation = correlate(padded_input[b,c],self.weights[k,c],mode='valid')
                    output_tensor[b,k] += correlation[::self.stride_shape[0], ::self.stride_shape[1]] #slicing operation using as the step_size
                output_tensor[b,k] += self.bias[k]

        if not self.is_2d:
            output_tensor = output_tensor[:,:,:,0]
        return output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor.reshape(self.output_shape)
        if not self.is_2d:
            self.input_tensor = np.expand_dims(self.input_tensor, axis=-1)  # (batch_size,channel_size,input_size,1)

        batch_size, channel_size, _, _ = self.input_tensor.shape
        input_size = self.input_tensor.shape[2:]
        filter_size = np.array(self.convolution_shape[1:])
        pad_size = filter_size // 2 # up,left
        pad_size_even = (filter_size - 1) % 2
        top = pad_size[0] - pad_size_even[0]
        bottom = pad_size[1]
        left = pad_size[1] - pad_size_even[1]
        right = pad_size[0]
        unpadded_input = np.zeros((batch_size, channel_size, input_size[0] + filter_size[0] - 1, input_size[1] + filter_size[1] - 1))
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros(self.num_kernels)
        self.upsampling_error = np.zeros((batch_size, self.num_kernels, *input_size))
        output_tensor = np.zeros_like(self.input_tensor)

        unpadded_input[:, :, top:top + input_size[0], left:left + input_size[1]] = self.input_tensor
        for b in range(batch_size):
            for k in range(self.num_kernels):
                self.gradient_bias[k] += np.sum(self.error_tensor[b,k])
                self.upsampling_error[b,k,::self.stride_shape[0],::self.stride_shape[1]] = self.error_tensor[b,k]

                for c in range(channel_size):
                    # Uses 'same' mode to maintain dimensions and preserve edge information
                    output_tensor[b,c,:] += convolve(self.upsampling_error[b,k,:],self.weights[k,c,:],mode = 'same')
                    self.gradient_weights[k,c,:] += correlate(unpadded_input[b,c,:],self.upsampling_error[b,k,:],mode = 'valid')

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights,self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias,self.gradient_bias)

        if not self.is_2d:
            output_tensor = output_tensor[:,:,:,0]

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,np.prod(self.convolution_shape),self.num_kernels*np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape,1,self.num_kernels)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
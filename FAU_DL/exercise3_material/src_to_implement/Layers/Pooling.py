import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channel_size, _, _ = input_tensor.shape
        input_size = np.array(input_tensor.shape[2:])
        output_size = (input_size - np.array(self.pooling_shape)) // np.array(self.stride_shape) + 1

        #reshape input as (height_out, pool_height, width_out, pool_width)
        input_pooling_reshaped_strided = np.lib.stride_tricks.as_strided(input_tensor,
                                                                         shape=(batch_size, channel_size, *output_size, *self.pooling_shape),
                                                                         strides=(input_tensor.strides[0],
                                                                                  input_tensor.strides[1],
                                                                                  self.stride_shape[0]*input_tensor.strides[2],# Stride between pooling windows
                                                                                  self.stride_shape[1]*input_tensor.strides[3],
                                                                                  input_tensor.strides[2],# Stride within pooling window
                                                                                  input_tensor.strides[3]))
        output_tensor = np.max(input_pooling_reshaped_strided, axis=(4,5))
        # Store indices of maximum values in each pooling window
        # Output shape: (batch_size, channels, output_shape[0], output_shape[1])
        # Each element is an integer in range [0, pooling_shape[0] * pooling_shape[1] - 1]
        self.max_indices = np.argmax(input_pooling_reshaped_strided.reshape(batch_size,channel_size,*output_size,-1), axis=4)

        return output_tensor

    def backward(self, error_tensor):
        batch_size, channel_size, _, _ = error_tensor.shape
        input_size = error_tensor.shape[2:]
        output = np.zeros_like(self.input_tensor)

        i,j = np.meshgrid(np.arange(input_size[0]),np.arange(input_size[1]),indexing='ij')

        for b in range(batch_size):
            for c in range(channel_size):
                max_pos = self.max_indices[b,c]
                y_pos = i * self.stride_shape[0] + max_pos // self.pooling_shape[0]
                x_pos = j * self.stride_shape[1] + max_pos % self.pooling_shape[1]
                #Accumulate gradients at max value positions, handling potential index overlap
                np.add.at(output, (b,c,y_pos,x_pos), error_tensor[b,c,i,j])

        return output
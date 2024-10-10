import numpy as np
from Layers import Base, FullyConnected, TanH, Sigmoid
import copy

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self._optimizer = None

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_state = [np.zeros((1, hidden_size))]
        self.fc_hidden = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected.FullyConnected(hidden_size, output_size)
        self.tanh = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()

        self.state = {
            'fc_hidden'   :{},
            'fc_output'   :{},
            'sigmoid'     :{},
            'tanh'        :{}
        }

    def initialize(self,weight_initializer, bias_initializer):
        self.fc_hidden.initialize(weight_initializer, bias_initializer)
        self.fc_output.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0] #time
        output = np.zeros((batch_size, self.output_size))
        if not self._memorize:
            self.hidden_state = [np.zeros((1, self.hidden_size))]

        for time in range(batch_size):
            x_t = input_tensor[time,]
            concatenated_input = np.concatenate((self.hidden_state[-1].flatten(), x_t.flatten())).reshape(1,-1)

            self.hidden_state.append(self.tanh.forward(self.fc_hidden.forward(concatenated_input)))
            self.state['fc_hidden'][time] = self.fc_hidden.input_tensor.copy()
            self.state['tanh'][time] = self.tanh.output.copy()

            output[time] = self.sigmoid.forward(self.fc_output.forward(self.hidden_state[-1]))
            self.state['fc_output'][time] = self.fc_output.input_tensor.copy()
            self.state['sigmoid'][time] = self.sigmoid.output.copy()

        return output

    def backward(self, error_tensor):
        self.gradient_hidden_weights = np.zeros(self.fc_hidden.weights.shape)
        self.gradient_output_weights = np.zeros(self.fc_output.weights.shape)
        error_x_t = np.zeros((self.input_tensor.shape[0],self.input_size))
        error_h_t = np.zeros((1,self.hidden_size))
        batch_size = error_tensor.shape[0]

        for time in reversed(range(batch_size)):
            error_y_t = error_tensor[time][np.newaxis,:]
            self.sigmoid.output = self.state['sigmoid'][time].copy()
            self.fc_output.input_tensor = self.state['fc_output'][time].copy()
            self.tanh.output = self.state['tanh'][time].copy()
            # Gradient of a copy procedure is a sum
            tanh_gradient = self.tanh.backward(self.fc_output.backward(self.sigmoid.backward(error_y_t)) + error_h_t)
            self.gradient_output_weights += self.fc_output.gradient_weights

            self.fc_hidden.input_tensor = self.state['fc_hidden'][time].copy()
            gradient_concatenated_input = self.fc_hidden.backward(tanh_gradient)
            self.gradient_hidden_weights += self.fc_hidden.gradient_weights

            error_h_t = gradient_concatenated_input[:,0:self.hidden_size].copy()
            error_x_t[time] = gradient_concatenated_input[:,self.hidden_size:].copy()

        if self._optimizer:
            self.fc_hidden.weights = self._optimizer.calculate_update(self.fc_hidden.weights, self.gradient_hidden_weights)
            self.fc_output.weights = self._optimizer.calculate_update(self.fc_output.weights, self.gradient_output_weights)

        return error_x_t

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.weights)

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_hidden_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fc_hidden._gradient_weights = gradient_weights

#remember to check property spelling !!!
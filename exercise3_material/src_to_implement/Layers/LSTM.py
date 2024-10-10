import numpy as np
from Layers import Base, FullyConnected, TanH, Sigmoid
import copy

class LSTM(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.memorize = False
        self.optimizer = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = [np.zeros((1,hidden_size))]
        self.cell_state = [np.zeros((1,hidden_size))]

        self.fc_f = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_i = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_c_hat = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_o = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_output = FullyConnected.FullyConnected(hidden_size, output_size)

        self.sigmoid = Sigmoid.Sigmoid()
        self.tanh = TanH.TanH()

        self.state = {
            'fc_hidden'     :{},
            'f_sigmoid'     :{},
            'i_sigmoid'     :{},
            'c_hat_tanh'    :{},
            'o_sigmoid'     :{},
            'fc_output'     :{},
            'sigmoid'       :{},
            'tanh'          :{}
        }

    def initialize(self,weight_initializer, bias_initializer):
        self.fc_f.initialize(weight_initializer, bias_initializer)
        self.fc_i.initialize(weight_initializer, bias_initializer)
        self.fc_c_hat.initialize(weight_initializer, bias_initializer)
        self.fc_o.initialize(weight_initializer, bias_initializer)
        self.fc_output.initialize(weight_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        output = np.zeros((batch_size, self.output_size))
        if not self.memorize:
            self.hidden_state = [np.zeros((1, self.hidden_size))]
        self.cell_state = [np.zeros((1, self.hidden_size))]

        for time in range(batch_size):
            x_t = self.input_tensor[time]
            concatenated_input = np.concatenate((self.hidden_state[-1].flatten(), x_t.flatten())).reshape(1, -1)

            f = self.sigmoid.forward(self.fc_f.forward(concatenated_input))
            self.state['f_sigmoid'][time] = self.sigmoid.output.copy()

            i = self.sigmoid.forward(self.fc_i.forward(concatenated_input))
            self.state['i_sigmoid'][time] = self.sigmoid.output.copy()

            c_hat = self.tanh.forward(self.fc_c_hat.forward(concatenated_input))
            self.state['c_hat_tanh'][time] = self.tanh.output.copy()

            o = self.sigmoid.forward(self.fc_o.forward(concatenated_input))
            self.state['o_sigmoid'][time] = self.sigmoid.output.copy()
            self.state['fc_hidden'][time] = self.fc_f.input_tensor.copy()

            self.cell_state.append(self.cell_state[-1] * f + i * c_hat)

            self.hidden_state.append(o * self.tanh.forward(self.cell_state[-1]))
            self.state['tanh'][time] = self.tanh.output.copy()

            output[time] = self.sigmoid.forward(self.fc_output.forward(self.hidden_state[-1]))
            self.state['fc_output'][time] = self.fc_output.input_tensor.copy()
            self.state['sigmoid'][time] = self.sigmoid.output.copy()

        return output

    def backward(self, error_tensor):
        self.gradient_hidden_weights = np.zeros((4,*self.fc_o.weights.shape))
        self.gradient_output_weights = np.zeros(self.fc_output.weights.shape)
        error_x_t = np.zeros((self.input_tensor.shape[0], self.input_size))
        error_h_t = np.zeros((1, self.hidden_size))
        error_c_t = np.zeros((1, self.hidden_size))
        batch_size = error_tensor.shape[0]

        for time in reversed(range(batch_size)):
            error_y_t = error_tensor[time][np.newaxis, :]
            self.sigmoid.output = self.state['sigmoid'][time].copy()
            self.fc_output.input_tensor = self.state['fc_output'][time].copy()
            derivative_h = self.fc_output.backward(self.sigmoid.backward(error_y_t)) + error_h_t
            self.gradient_output_weights += self.fc_output.gradient_weights

            derivative_o = self.state['tanh'][time] * derivative_h

            self.tanh.output = self.state['tanh'][time].copy()
            derivative_c = error_c_t + self.tanh.backward(derivative_h * self.state['o_sigmoid'][time])

            derivative_f = derivative_c * self.cell_state[time]
            derivative_i = derivative_c * self.state['c_hat_tanh'][time]
            derivative_c_hat = derivative_c * self.state['i_sigmoid'][time]

            self.sigmoid.output = self.state['f_sigmoid'][time].copy()
            derivative_f = self.sigmoid.backward(derivative_f)
            self.sigmoid.output = self.state['i_sigmoid'][time].copy()
            derivative_i = self.sigmoid.backward(derivative_i)
            self.tanh.output = self.state['c_hat_tanh'][time].copy()
            derivative_c_hat = self.tanh.backward(derivative_c_hat)
            self.sigmoid.output = self.state['o_sigmoid'][time].copy()
            derivative_o = self.sigmoid.backward(derivative_o)

            self.fc_f.input_tensor = self.state['fc_hidden'][time].copy()
            self.fc_i.input_tensor = self.state['fc_hidden'][time].copy()
            self.fc_c_hat.input_tensor = self.state['fc_hidden'][time].copy()
            self.fc_o.input_tensor = self.state['fc_hidden'][time].copy()
            gradient_concatenated_input = (self.fc_f.backward(derivative_f)  +
                                           self.fc_i.backward(derivative_i) +
                                           self.fc_c_hat.backward(derivative_c_hat) +
                                           self.fc_o.backward(derivative_o))

            self.gradient_hidden_weights[0] += self.fc_f.gradient_weights
            self.gradient_hidden_weights[1] += self.fc_i.gradient_weights
            self.gradient_hidden_weights[2] += self.fc_c_hat.gradient_weights
            self.gradient_hidden_weights[3] += self.fc_o.gradient_weights

            error_h_t = gradient_concatenated_input[:, 0:self.hidden_size].copy()
            error_x_t[time] = gradient_concatenated_input[:, self.hidden_size:].copy()
            error_c_t = derivative_c * self.state['f_sigmoid'][time]

        if self.optimizer:
            self.fc_f.weights = self._optimizer.calculate_update(self.fc_f.weights, self.gradient_hidden_weights[0])
            self.fc_i.weights = self._optimizer.calculate_update(self.fc_i.weights, self.gradient_hidden_weights[1])
            self.fc_c_hat.weights = self._optimizer.calculate_update(self.fc_c_hat.weights, self.gradient_hidden_weights[2])
            self.fc_o.weights = self._optimizer.calculate_update(self.fc_o.weights, self.gradient_hidden_weights[3])
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
        return np.hstack((self.fc_f.weights, self.fc_i.weights, self.fc_c_hat.weights,self.fc_o.weights))

    @weights.setter
    def weights(self, weights):
        self.fc_f.weights = weights[:,0:self.hidden_size]
        self.fc_i.weights = weights[:,self.hidden_size:self.hidden_size*2]
        self.fc_c_hat.weights = weights[:,self.hidden_size*2:self.hidden_size*3]
        self.fc_o.weights = weights[:,self.hidden_size*3:self.hidden_size*4]

    @property
    def gradient_weights(self):
        return np.hstack(self.gradient_hidden_weights[:])
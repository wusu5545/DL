import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        #use weight_tensor to output same decimal precision
        if self.regularizer is None:
            weight_tensor -= self.learning_rate * gradient_tensor
        else:
            weight_tensor -= self.learning_rate * gradient_tensor + self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum_rate: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        # w(k+1)=w(k)+momentum*v(k)-learning_rate*gradient
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        if self.regularizer is None:
            weight_tensor = weight_tensor + self.v
        else:
            weight_tensor -= -self.v + self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, mu: float = 0.9, rho: float = 0.999):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu #beta_1
        self.rho = rho #beta_2
        self.v = None
        self.r= None
        self.k = 0 #iteration-index

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Adam update rule
        # v(k) = mu * v(k-1) + (1 - mu) * gradient
        # r(k) = rho * r(k-1) + (1 - rho) * gradient^2
        # w(k+1) = w(k) - learning_rate * v_hat(k) / (sqrt(r_hat(k)) + epsilon)

        # Initialize first and second moment vectors
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)

        self.k += 1
        # Update first moment
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)
        #Bias correction
        #v_hat = v / (1 - mu^k+1), r_hat = r / (1 - rho^k+1)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        # Update weights

        if self.regularizer is None:
            weight_tensor -= self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)
        else:
            weight_tensor -= self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps) + self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor
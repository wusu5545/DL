import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        # Avoid division by zero by adding a small epsilon
        #-sum(y*log(y_hat+epsilon)))
        return -np.sum(self.label_tensor * np.log(self.prediction_tensor + np.finfo(np.float64).eps))

    def backward(self,label_tensor):
        # Compute the gradient of the loss with respect to the prediction tensor
        #-y/(y_hat+epsilon),where y is the label tensor and y_hat is the prediction tensor
        return -(label_tensor / (self.prediction_tensor + np.finfo(np.float64).eps))
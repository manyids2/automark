import numpy as np


from .functions.mse import mse_forward, mse_grad_input
from .functions.hinge import hinge_forward, hinge_grad_input


class Loss(object):
    
    def __init__(self):
        self.output = 0.0
        
    def forward(self, target_pred, target_true):
        return self.output
    
    def backward(self, target_pred, target_true):
        return np.zeros_like(target_pred)


class Hinge(Loss):
    
    def forward(self, target_pred, target_true):
        self.output = hinge_forward(target_pred, target_true)
        return self.output
    
    def backward(self, target_pred, target_true):
        return hinge_grad_input(target_pred, target_true)


class MSE(Loss):
    
    def forward(self, target_pred, target_true):
        self.output = mse_forward(target_pred, target_true)
        return self.output
    
    def backward(self, target_pred, target_true):
        return mse_grad_input(target_pred, target_true)


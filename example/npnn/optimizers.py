import numpy as np


class Optimizer(object):
    '''This is a basic class. 
    All other optimizers will inherit it
    '''

    def __init__(self, model, lr=0.001, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def update_params(self):
        pass


class SGD(Optimizer):
    '''Stochastic gradient descent optimizer
    https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    '''

    def update_params(self):
        weights = self.model.get_params()
        grads = self.model.get_params_gradients()
        for w, dw in zip(weights, grads):
            update = self.lr * (dw + self.weight_decay * w)
            # it writes the result to the previous variable instead of copying
            np.subtract(w, update, out=w)

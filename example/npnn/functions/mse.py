import numpy as np



def mse_forward(target_pred, target_true):
    """Compute the value of MSE loss
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the value of MSE loss 
        for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    n_objects = target_pred.shape[0]
    output = np.sum((target_pred - target_true) ** 2) / n_objects / 2.0
    return output 


def mse_grad_input(target_pred, target_true):
    """Compute the partial derivative 
        of MSE loss with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the partial derivative 
        of MSE loss with respect to its input
        np.array of size `(n_objects,)`
    """
    #################
    ### YOUR CODE ###
    #################
    n_objects = target_pred.shape[0]
    grad_input = (target_pred - target_true) / n_objects   
    return grad_input




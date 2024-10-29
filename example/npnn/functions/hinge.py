import numpy as np



def hinge_forward(target_pred, target_true):
    """Compute the value of Hinge loss 
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the value of Hinge loss 
        for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    n_objects = target_pred.shape[0]
    diff = 1 - target_pred * target_true
    diff = diff * (diff > 0)
    output = np.sum(diff) / n_objects
    return output


def hinge_grad_input(target_pred, target_true):
    """Compute the partial derivative 
        of Hinge loss with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the partial derivative 
        of Hinge loss with respect to its input
        np.array of size `(n_objects,)`
    """
    #################
    ### YOUR CODE ###
    #################
    n_objects = target_pred.shape[0]
    diff = 1 - target_pred * target_true
    grad = -target_true
    grad_input = (diff > 0) * grad / n_objects  
    return grad_input

    

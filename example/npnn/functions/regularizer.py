import numpy as np



def l2_regularizer(weight_decay, weights):
    """Compute the L2 regularization term
    # Arguments
        weight_decay: float
        weights: list of arrays of different shapes
    # Output
        sum of the L2 norms of the input weights
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    L2 = np.sum([np.sum(w**2) for w in weights])
    return weight_decay * L2 / 2.0





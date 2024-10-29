import numpy as np



def dropout_generate_mask(shape, drop_rate):
    """Generate mask 
    # Arguments
        shape: shape of the input array 
            tuple 
        drop_rate: probability of the element 
            to be multiplied by 0
            scalar
    # Output
        binary mask 
    """
    #################
    ### YOUR CODE ###
    #################
    mask = np.random.binomial(1, p=1.0 - drop_rate, size=shape)
    return mask


def dropout_forward(x_input, mask, drop_rate, training_phase):
    """Perform the mapping of the input
    # Arguments
        x_input: input of the layer 
            np.array of size `(n_objects, n_in)`
        mask: binary mask
            np.array of size `(n_objects, n_in)`
        drop_rate: probability of the element to be multiplied by 0
            scalar
        training_phase: bool eiser `True` - training, or `False` - testing
    # Output
        the output of the dropout layer 
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    if training_phase:
        output = x_input * mask
    else:
        output = x_input * (1.0 - drop_rate)
    return output


def dropout_grad_input(x_input, grad_output, mask):
    """Calculate the partial derivative of 
        the loss with respect to the input of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dropout layer 
            np.array of size `(n_objects, n_in)`
        mask: binary mask
            np.array of size `(n_objects, n_in)`
    # Output
        the partial derivative of the loss with 
        respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_input = grad_output * mask
    return grad_input



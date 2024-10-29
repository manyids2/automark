import numpy as np



def flatten_forward(x_input):
    """Perform the reshaping of the tensor of size `(K, L, M, N)` 
        to the tensor of size `(K, L*M*N)`
    # Arguments
        x_input: np.array of size `(K, L, M, N)`
    # Output
        output: np.array of size `(K, L*M*N)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = x_input.reshape((len(x_input), -1))
    return output


def flatten_grad_input(x_input, grad_output):
    """Calculate partial derivative of the loss with respect to the input
    # Arguments
        x_input: partial derivative of the loss 
            with respect to the output
            np.array of size `(K, L*M*N)`
    # Output
        output: partial derivative of the loss 
            with respect to the input
            np.array of size `(K, L, M, N)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_input = grad_output.reshape(x_input.shape)
    return grad_input







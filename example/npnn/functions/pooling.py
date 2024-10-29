import numpy as np



def maxpool_forward(x_input):
    """Perform max pooling operation with 2x2 window
    # Arguments
        x_input: np.array of size (2 * W, 2 * H)
    # Output
        output: np.array of size (W, H)
    """
    #################
    ### YOUR CODE ###
    ################# 
    x = np.stack([x_input[::2, ::2], x_input[1::2, ::2], 
                  x_input[::2, 1::2], x_input[1::2, 1::2]])
    x = np.max(x, axis=0)
    output = x
    return output



def maxpool_grad_input(x_input, grad_output):
    """Calculate partial derivative of the loss with respect to the input
    # Arguments
        x_input: np.array of size (2 * W, 2 * H)
        grad_output: partial derivative of the loss 
            with respect to the output 
            np.array of size (W, H)
    # Output
        output: partial derivative of the loss 
            with respect to the input
            np.array of size (2 * W, 2 * H) 
    """
    height, width = x_input.shape
    # create the array of zeros of the required size
    grad_input = np.zeros(x_input.shape)
    
    # let's put 1 if the element with this position 
    # is maximal in the window
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            window = x_input[i:i+2, j:j+2]
            i_max, j_max = np.unravel_index(np.argmax(window), (2, 2))
            grad_input[i + i_max, j + j_max] = 1
            
    # put corresponding gradient instead of 1       
    grad_input = grad_input.ravel()
    grad_input[grad_input == 1] = grad_output.ravel()
    grad_input = grad_input.reshape(x_input.shape)
    return grad_input




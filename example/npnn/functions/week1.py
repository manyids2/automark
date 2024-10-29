import numpy as np


def w1_linear_forward(x_input, P):
    """Perform the Linear mapping of the input
    """
    #################
    ### YOUR CODE ###
    #################

    output = np.dot(x_input, P)

    return output


def w1_cal_pseudoinverse(x_input, y_input):
    """Calculate pseduoinverse"""
    #################
    ### YOUR CODE ###
    #################
    X = x_input

    inverse = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T)

    output = np.matmul(inverse, y_input)

    return output


def w1_L2_regression(x_input, y_input, factor=0.001):
    """Calculate pseduoinverse"""
    #################
    ### YOUR CODE ###
    #################

    X = x_input

    inverse = np.matmul(np.linalg.pinv(np.matmul(X.T, X) + factor * np.eye(X.shape[1])), X.T)
    output = np.matmul(inverse, y_input)

    return output

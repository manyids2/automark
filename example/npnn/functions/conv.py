import numpy as np


def conv_matrix(matrix, kernel):
    """Perform the convolution of the matrix 
        with the kernel using zero padding
    # Arguments
        matrix: input matrix np.array of size `(N, M)`
        kernel: kernel of the convolution 
            np.array of size `(2p + 1, 2q + 1)`
    # Output
        the result of the convolution
        np.array of size `(N, M)`
    """
    height, width = matrix.shape
    k_h, k_w = kernel.shape
    
    p, q = (k_h - 1) // 2, (k_w - 1) // 2
    
    matrix_pad = np.pad(matrix, [(p, p), (q, q)], 'constant', constant_values=(0, 0))
    output = np.zeros(matrix.shape)
    
    for i in range(height):
        for j in range(width):
            patch = matrix_pad[i: i+2*p + 1, j: j+2*q + 1]
            output[i, j] = np.sum(patch * kernel) 
    return output



def box_blur(image, box_size):
    """Perform the blur of the image
    # Arguments
        image: input matrix - np.array of size `(N, M)`
        box_size: the size of the blur kernel - int > 0  
            the kernel is of size `(box_size, box_size)`
    # Output
        the result of the blur
            np.array of size `(N, M)`
    """ 
    k = 1.0 * np.ones((box_size, box_size)) / (box_size ** 2)
    output = conv_matrix(image, k)
    return output





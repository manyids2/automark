import numpy as np


from .functions.conv import conv_matrix
from .functions.dense import dense_forward, dense_grad_input, dense_grad_W, dense_grad_b
from .functions.dropout import dropout_generate_mask, dropout_forward, dropout_grad_input
from .functions.flatten import flatten_forward, flatten_grad_input
from .functions.pooling import maxpool_forward, maxpool_grad_input
from .functions.relu import relu_forward, relu_grad_input


class Layer(object):

    def __init__(self):
        self.training_phase = True
        self.output = 0.0

    def forward(self, x_input):
        self.output = x_input
        return self.output

    def backward(self, x_input, grad_output):
        return grad_output

    def get_params(self):
        return []

    def get_params_gradients(self):
        return []


class Dense(Layer):

    def __init__(self, n_input, n_output):
        super(Dense, self).__init__()
        # Randomly initializing the weights from normal distribution
        self.W = np.random.normal(size=(n_input, n_output))
        self.grad_W = np.zeros_like(self.W)
        # initializing the bias with zero
        self.b = np.zeros(n_output)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x_input):
        self.output = dense_forward(x_input, self.W, self.b)
        return self.output

    def backward(self, x_input, grad_output):
        # get gradients of weights
        self.grad_W = dense_grad_W(x_input, grad_output, self.W, self.b)
        self.grad_b = dense_grad_b(x_input, grad_output, self.W, self.b)
        # propagate the gradient backwards
        return dense_grad_input(x_input, grad_output, self.W, self.b)

    def get_params(self):
        return [self.W, self.b]

    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]


class ReLU(Layer):

    def forward(self, x_input):
        self.output = relu_forward(x_input)
        return self.output

    def backward(self, x_input, grad_output):
        return relu_grad_input(x_input, grad_output)


class Dropout(Layer):

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.drop_rate = drop_rate
        self.mask = 1.0

    def forward(self, x_input):
        if self.training_phase:
            self.mask = dropout_generate_mask(x_input.shape, self.drop_rate)
        self.output = dropout_forward(x_input, self.mask,
                                      self.drop_rate, self.training_phase)
        return self.output

    def backward(self, x_input, grad_output):
        grad_input = dropout_grad_input(x_input, grad_output, self.mask)
        return grad_input


class ConvLayer(Layer):
    """
    Convolutional Layer. The implementation is based on 
        the representation of the convolution as matrix multiplication
    """

    def __init__(self, n_in, n_out, filter_size):
        super(ConvLayer, self).__init__()
        self.W = np.random.normal(size=(n_out, n_in, filter_size, filter_size))
        self.b = np.zeros(n_out)

    def forward(self, x_input):
        n_obj, n_in, h, w = x_input.shape
        n_out = len(self.W)

        self.output = []

        for image in x_input:
            output_image = []
            for i in range(n_out):
                out_channel = 0.0
                for j in range(n_in):
                    out_channel += conv_matrix(image[j], self.W[i, j])
                output_image.append(out_channel)
            self.output.append(np.stack(output_image, 0))

        self.output = np.stack(self.output, 0)
        return self.output

    def backward(self, x_input, grad_output):

        N, C, H, W = x_input.shape
        F, C, HH, WW = self.W.shape

        pad = int((HH - 1) / 2)

        self.grad_b = np.sum(grad_output, (0, 2, 3))

        # pad input array
        x_padded = np.pad(x_input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        # naive implementation of im2col
        x_cols = None
        for i in range(HH, H_padded + 1):
            for j in range(WW, W_padded+1):
                for n in range(N):
                    field = x_padded[n, :, i-HH:i, j-WW:j].reshape((1, -1))
                    if x_cols is None:
                        x_cols = field
                    else:
                        x_cols = np.vstack((x_cols, field))

        x_cols = x_cols.T

        d_out = grad_output.transpose(1, 2, 3, 0)
        dout_cols = d_out.reshape(F, -1)

        dw_cols = np.dot(dout_cols, x_cols.T)
        self.grad_W = dw_cols.reshape(F, C, HH, WW)

        w_cols = self.W.reshape(F, -1)
        dx_cols = np.dot(w_cols.T, dout_cols)

        dx_padded = np.zeros((N, C, H_padded, W_padded))
        idx = 0
        for i in range(HH, H_padded + 1):
            for j in range(WW, W_padded + 1):
                for n in range(N):
                    dx_padded[n:n+1, :, i-HH:i, j-WW:j] += dx_cols[:, idx].reshape((1, C, HH, WW))
                    idx += 1
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        grad_input = dx
        return grad_input

    def get_params(self):
        return [self.W, self.b]

    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]


class MaxPool2x2(Layer):

    def forward(self, x_input):
        n_obj, n_ch, h, w = x_input.shape
        self.output = np.zeros((n_obj, n_ch, h // 2, w // 2))
        for i in range(n_obj):
            for j in range(n_ch):
                self.output[i, j] = maxpool_forward(x_input[i, j])
        return self.output

    def backward(self, x_input, grad_output):
        n_obj, n_ch, _, _ = x_input.shape
        grad_input = np.zeros_like(x_input)
        for i in range(n_obj):
            for j in range(n_ch):
                grad_input[i, j] = maxpool_grad_input(x_input[i, j], grad_output[i, j])
        return grad_input


class FlattenLayer(Layer):

    def forward(self, x_input):
        self.output = flatten_forward(x_input)
        return self.output

    def backward(self, x_input, grad_output):
        output = flatten_grad_input(x_input, grad_output)
        return output

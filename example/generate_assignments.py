from npnn.functions.relu import *
from npnn.functions.dense import *
from npnn.functions.conv import *
from npnn.functions.dropout import *
from npnn.functions.flatten import *
from npnn.functions.hinge import *
from npnn.functions.nearest_neighbors import *
from npnn.functions.pooling import *
from npnn.functions.regularizer import *
from npnn.functions.mse import *
from npnn.functions.sigmoid import *
from npnn.functions.nll import *
from npnn.functions.week1 import *

from trees import *

import numpy as np
import pickle
import os


def crdict(data):
    data_type = type(data).__name__
    if data_type == 'ndarray':
        return {'data': data.tolist(), 'type': data_type}
    else:
        return {'data': data, 'type': data_type}


def create_dense_forward():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_forward(x_input, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_forward():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_forward(x_input, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_linear_forward_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:

        x_input = np.random.uniform(-10.0, 10.0, size=(3, 2))
        ones_column = np.ones((x_input.shape[0], 1))
        x_input = np.hstack((x_input, ones_column))

        P = np.random.uniform(-10.0, 10.0, size=(3, 1))

        result = w1_linear_forward(x_input, P)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'P': P,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_linear_forward():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:

        x_input = np.random.uniform(-10.0, 10.0, size=(3, 2))
        ones_column = np.ones((x_input.shape[0], 1))
        x_input = np.hstack((x_input, ones_column))

        P = np.random.uniform(-10.0, 10.0, size=(3, 1))

        result = w1_linear_forward(x_input, P)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'P': crdict(P),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_cal_pseudoinverse_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        x_input = np.random.uniform(-10.0, 10.0, size=(2, 3))
        y_input = np.random.uniform(-10.0, 10.0, size=(2, 1))

        result = w1_cal_pseudoinverse(x_input, y_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'y_input': y_input,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_cal_pseudoinverse():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        x_input = np.random.uniform(-10.0, 10.0, size=(2, 3))
        y_input = np.random.uniform(-10.0, 10.0, size=(2, 1))

        result = w1_cal_pseudoinverse(x_input, y_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'y_input': crdict(y_input),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_L2_regression_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        x_input = np.random.uniform(-10.0, 10.0, size=(2, 3))
        y_input = np.random.uniform(-10.0, 10.0, size=(2, 1))

        result = w1_L2_regression(x_input, y_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'y_input': y_input,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_week1_L2_regression():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        x_input = np.random.uniform(-10.0, 10.0, size=(2, 3))
        y_input = np.random.uniform(-10.0, 10.0, size=(2, 1))

        result = w1_L2_regression(x_input, y_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'y_input': crdict(y_input),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_input(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_W():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_W(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_grad_W():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_W(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_b():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_b(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_grad_b():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_b(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'W': crdict(W),
            'b': crdict(b)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_relu_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = relu_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_sigmoid_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_in = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = sigmoid_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_relu_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = relu_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_sigmoid_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_in = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = sigmoid_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_hinge_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.random(size=(n_obj,)) > 0.5
        t_true = t_true.astype('float') * 2 - 1
        result = hinge_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_hinge_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.random(size=(n_obj,)) > 0.5
        t_true = t_true.astype('float') * 2 - 1
        result = hinge_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nll_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(1e-5, 1 - 1e-5, size=(n_obj, 1))
        t_true = np.random.randint(0, 2, size=(n_obj, 1)).astype('float')
        # t_true = t_true.astype('float') * 2 - 1
        result = nll_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nll_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(1e-5, 1 - 1e-5, size=(n_obj, 1))
        t_true = np.random.randint(0, 2, size=(n_obj, 1)).astype('float')
        # t_true = t_true.astype('float') * 2 - 1
        result = nll_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dist_to_training_samples():
    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_in = np.random.randint(5, 10)
        n_obj = np.random.randint(5, 10)
        x_input = np.random.normal(0.0, 1.0, size=(n_in))
        training_set = np.random.normal(0.0, 1.0, size=(n_obj, n_in))
        result = dist_to_training_samples(x_input, training_set)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'training_set': crdict(training_set)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nearest_neighbors():
    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(5, 10)
        distances = np.random.uniform(0.0, 1.0, size=(n_obj))
        training_labels = np.random.randint(0, 2, size=(n_obj)).astype('float')
        result = nearest_neighbors(distances, training_labels)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'distances': crdict(distances),
            'training_labels': crdict(training_labels),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_mse_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        result = mse_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_mse_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        result = mse_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': crdict(t_pred),
            'target_true': crdict(t_true)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dropout_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        drop_rate = np.random.random()
        training_phase = np.random.random() > 0.5
        mask = dropout_generate_mask(x_input.shape, drop_rate)
        result = dropout_forward(x_input, mask, drop_rate, training_phase)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'mask': crdict(mask),
            'drop_rate': crdict(drop_rate),
            'training_phase': crdict(training_phase)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dropout_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        drop_rate = np.random.random()
        training_phase = np.random.random() > 0.5
        mask = dropout_generate_mask(x_input.shape, drop_rate)
        result = dropout_grad_input(x_input, grad_output, mask)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output),
            'mask': crdict(mask),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_l2_regularizer():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        shape = np.random.randint(1, 7, size=3)
        weights = np.random.uniform(-10.0, 10.0, shape)
        wd = np.random.random()
        result = l2_regularizer(wd, weights)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'weight_decay': crdict(wd),
            'weights': crdict(weights)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_conv_matrix():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        p, q = np.random.randint(0, 2, size=2)
        h = np.random.randint(2 * p + 1, 2 * p + 4)
        w = np.random.randint(2 * q + 1, 2 * q + 4)

        image = np.random.uniform(-10.0, 10.0, size=(h, w))
        k = np.random.uniform(-10.0, 10.0, (2 * p + 1, 2 * q + 1))

        result = conv_matrix(image, k)
        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'matrix': crdict(image),
            'kernel': crdict(k)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_box_blur():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n = np.random.randint(1, 7)
        size = np.random.randint(2 * n + 4, 2 * n + 6, size=2)
        image = np.random.uniform(-10.0, 10.0, size)
        result = box_blur(image, 2 * n + 1)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'image': crdict(image),
            'box_size': crdict(2 * n + 1)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_maxpool_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds[:50]:
        h, w = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(h * 2, w * 2))
        result = maxpool_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
        })
        week_one_dict['outputs'].append(result)

    for ipd in ipds[50:]:
        h, w = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 0.0, size=(h * 2, w * 2))
        result = maxpool_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_flatten_forward():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        shape = np.random.randint(1, 7, size=4)
        x = np.random.uniform(-10.0, 10.0, shape)
        result = flatten_forward(x)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_flatten_grad_input():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        shape = np.random.randint(1, 7, size=4)
        x_input = np.random.uniform(-10.0, 10.0, shape)
        x_output = flatten_forward(x_input)
        grad_output = np.random.uniform(-10.0, 10.0, x_output.shape)
        result = flatten_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': crdict(x_input),
            'grad_output': crdict(grad_output)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict

########### LOCAL TESTS ##############


def create_dense_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_forward(x_input, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_forward(x_input, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_input(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_W_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_W(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dense_grad_b_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_b(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_grad_W_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_W(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_linear_grad_b_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in, n_out = np.random.randint(1, 7, size=3)
        n_out = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_out))
        W = np.random.uniform(-10.0, 10.0, size=(n_in, n_out))
        b = np.random.uniform(-10.0, 10.0, size=(n_out,))
        result = dense_grad_b(x_input, grad_output, W, b)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'W': W,
            'b': b
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_relu_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = relu_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_relu_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = relu_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_hinge_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.random(size=(n_obj,)) > 0.5
        t_true = t_true.astype('float') * 2 - 1
        result = hinge_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_hinge_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.random(size=(n_obj,)) > 0.5
        t_true = t_true.astype('float') * 2 - 1
        result = hinge_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_sigmoid_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_in = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = sigmoid_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_sigmoid_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        n_in = 1
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        result = sigmoid_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nll_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(1e-5, 1 - 1e-5, size=(n_obj, 1))
        t_true = np.random.randint(0, 2, size=(n_obj, 1)).astype('float')
        result = nll_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nll_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(1e-5, 1 - 1e-5, size=(n_obj, 1))
        t_true = np.random.randint(0, 2, size=(n_obj, 1)).astype('float')
        result = nll_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dist_to_training_samples_loc():
    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        n_in = np.random.randint(5, 10)
        n_obj = np.random.randint(5, 10)
        x_input = np.random.normal(0.0, 1.0, size=(n_in))
        training_set = np.random.normal(0.0, 1.0, size=(n_obj, n_in))
        result = dist_to_training_samples(x_input, training_set)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'training_set': training_set
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_nearest_neighbors_loc():
    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        n_obj = np.random.randint(5, 10)
        distances = np.random.uniform(0.0, 1.0, size=(n_obj))
        training_labels = np.random.randint(0, 2, size=(n_obj)).astype('float')
        result = nearest_neighbors(distances, training_labels)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'distances': distances,
            'training_labels': training_labels,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_mse_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        result = mse_forward(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_mse_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj = np.random.randint(1, 7)
        t_pred = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        t_true = np.random.uniform(-10.0, 10.0, size=(n_obj,))
        result = mse_grad_input(t_pred, t_true)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'target_pred': t_pred,
            'target_true': t_true
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dropout_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        drop_rate = np.random.random()
        training_phase = np.random.random() > 0.5
        mask = dropout_generate_mask(x_input.shape, drop_rate)
        result = dropout_forward(x_input, mask, drop_rate, training_phase)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'mask': mask,
            'drop_rate': drop_rate,
            'training_phase': training_phase
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_dropout_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n_obj, n_in = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        grad_output = np.random.uniform(-10.0, 10.0, size=(n_obj, n_in))
        drop_rate = np.random.random()
        training_phase = np.random.random() > 0.5
        mask = dropout_generate_mask(x_input.shape, drop_rate)
        result = dropout_grad_input(x_input, grad_output, mask)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output,
            'mask': mask,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_l2_regularizer_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        shape = np.random.randint(1, 7, size=3)
        weights = np.random.uniform(-10.0, 10.0, shape)
        wd = np.random.random()
        result = l2_regularizer(wd, weights)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'weight_decay': wd,
            'weights': weights
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_conv_matrix_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        p, q = np.random.randint(0, 2, size=2)
        h = np.random.randint(2 * p + 1, 2 * p + 4)
        w = np.random.randint(2 * q + 1, 2 * q + 4)

        image = np.random.uniform(-10.0, 10.0, size=(h, w))
        k = np.random.uniform(-10.0, 10.0, (2 * p + 1, 2 * q + 1))

        result = conv_matrix(image, k)
        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'matrix': image,
            'kernel': k
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_box_blur_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        n = np.random.randint(1, 7)
        size = np.random.randint(2 * n + 4, 2 * n + 6, size=2)
        image = np.random.uniform(-10.0, 10.0, size)
        result = box_blur(image, 2 * n + 1)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'image': image,
            'box_size': 2 * n + 1
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_maxpool_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(50):
        h, w = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 10.0, size=(h * 2, w * 2))
        result = maxpool_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
        })
        week_one_dict['outputs'].append(result)

    for _ in range(50):
        h, w = np.random.randint(1, 7, size=2)
        x_input = np.random.uniform(-10.0, 0.0, size=(h * 2, w * 2))
        result = maxpool_forward(x_input)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_flatten_forward_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.

    for _ in range(100):
        shape = np.random.randint(1, 7, size=4)
        x = np.random.uniform(-10.0, 10.0, shape)
        result = flatten_forward(x)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_flatten_grad_input_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        shape = np.random.randint(1, 7, size=4)
        x_input = np.random.uniform(-10.0, 10.0, shape)
        x_output = flatten_forward(x_input)
        grad_output = np.random.uniform(-10.0, 10.0, x_output.shape)
        result = flatten_grad_input(x_input, grad_output)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'x_input': x_input,
            'grad_output': grad_output
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


####################### TREES ##############
def create_tree_gini_index():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_left, n_right = np.random.randint(5, 25, size=2)
        y_left = (np.random.randn(n_left, 1) > 0.0).astype('float')
        y_right = (np.random.randn(n_right, 1) > 0.0).astype('float')
        classes = [0.0, 1.0]
        result = tree_gini_index(y_left, y_right, classes)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y_left': crdict(y_left),
            'Y_right': crdict(y_right),
            'classes': crdict(classes)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_gini_index_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_left, n_right = np.random.randint(5, 25, size=2)
        y_left = (np.random.randn(n_left, 1) > 0.0).astype('float')
        y_right = (np.random.randn(n_right, 1) > 0.0).astype('float')
        classes = [0.0, 1.0]
        result = tree_gini_index(y_left, y_right, classes)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y_left': y_left,
            'Y_right': y_right,
            'classes': classes
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_weighted_entropy():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n_left, n_right = np.random.randint(5, 25, size=2)
        y_left = (np.random.randn(n_left, 1) > 0.0).astype('float')
        y_right = (np.random.randn(n_right, 1) > 0.0).astype('float')
        classes = [0.0, 1.0]
        result = tree_weighted_entropy(y_left, y_right, classes)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y_left': crdict(y_left),
            'Y_right': crdict(y_right),
            'classes': crdict(classes)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_weighted_entropy_loc():
    '''Example function for creating a problem. In this case, the function the
    students must implement is a simple addition function. Returns: a dict with
    a set of input values and corresponding set of correct results.'''

    # We define the dict with a list of inputs and a list of expected outputs
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    for _ in range(100):
        n_left, n_right = np.random.randint(5, 25, size=2)
        y_left = (np.random.randn(n_left, 1) > 0.0).astype('float')
        y_right = (np.random.randn(n_right, 1) > 0.0).astype('float')
        classes = [0.0, 1.0]
        result = tree_weighted_entropy(y_left, y_right, classes)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y_left': y_left,
            'Y_right': y_right,
            'classes': classes
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_split_data_left():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n = np.random.randint(5, 10)
        X = np.random.uniform(-10, 10, size=(n, 2))
        Y = np.random.randint(0, 2, size=(n, 1)).astype('float')
        feature_index = int(np.random.choice([0, 1]))
        split_value = float(np.random.choice(
            sorted(X[:, feature_index])[1:-1], 2, replace=False).mean())
        result = tree_split_data_left(X, Y, feature_index, split_value)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'X': crdict(X),
            'Y': crdict(Y),
            'feature_index': crdict(feature_index),
            'split_value': crdict(split_value)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_split_data_right():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n = np.random.randint(5, 10)
        X = np.random.uniform(-10, 10, size=(n, 2))
        Y = np.random.randint(0, 2, size=(n, 1)).astype('float')
        feature_index = int(np.random.choice([0, 1]))
        split_value = float(np.random.choice(
            sorted(X[:, feature_index])[1:-1], 2, replace=False).mean())
        result = tree_split_data_right(X, Y, feature_index, split_value)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'X': crdict(X),
            'Y': crdict(Y),
            'feature_index': crdict(feature_index),
            'split_value': crdict(split_value)
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_split_data_left_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        n = np.random.randint(5, 10)
        X = np.random.uniform(-10, 10, size=(n, 2))
        Y = np.random.randint(0, 2, size=(n, 1)).astype('float')
        feature_index = int(np.random.choice([0, 1]))
        split_value = float(np.random.choice(
            sorted(X[:, feature_index])[1:-1], 2, replace=False).mean())
        result = tree_split_data_left(X, Y, feature_index, split_value)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'X': X,
            'Y': Y,
            'feature_index': feature_index,
            'split_value': split_value
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_split_data_right_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        n = np.random.randint(5, 10)
        X = np.random.uniform(-10, 10, size=(n, 2))
        Y = np.random.randint(0, 2, size=(n, 1)).astype('float')
        feature_index = int(np.random.choice([0, 1]))
        split_value = float(np.random.choice(
            sorted(X[:, feature_index])[1:-1], 2, replace=False).mean())
        result = tree_split_data_right(X, Y, feature_index, split_value)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'X': X,
            'Y': Y,
            'feature_index': feature_index,
            'split_value': split_value
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_to_terminal():
    week_one_dict = {'inputs': [], 'outputs': []}

    # We also create a set of random ID numbers to match student submissions
    # with the appropriate result in the database.
    ipds = list(np.random.choice(10000, 100, replace=False))

    week_one_dict['ipd'] = ipds
    for ipd in ipds:
        n = np.random.randint(5, 25)
        n = 2 * (n // 2) + 1
        Y = (np.random.randn(n, 1) > 0.0).astype('float')
        result = tree_to_terminal(Y)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y': crdict(Y),
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict


def create_tree_to_terminal_loc():
    week_one_dict = {'inputs': [], 'outputs': []}

    for _ in range(100):
        n = np.random.randint(5, 25)
        n = 2 * (n // 2) + 1
        Y = (np.random.randn(n, 1) > 0.0).astype('float')
        result = tree_to_terminal(Y)

        # 'inputs' should contain the names of the function arguments (defined in notebook)
        # and the values should be passed through crdict()
        week_one_dict['inputs'].append({
            'Y': Y,
        })
        week_one_dict['outputs'].append(result)

    return week_one_dict

#########
# Main
#########


def make_data_week_1_2_3_loc(file_path):
    data_dict = dict([])

    # Week 1
    data_dict['w1_linear_forward'] = create_week1_linear_forward_loc()
    data_dict['w1_cal_pseudoinverse'] = create_week1_cal_pseudoinverse_loc()
    data_dict['w1_L2_regression'] = create_week1_L2_regression_loc()

    # Week 2
    data_dict['w2_linear_forward'] = create_linear_forward_loc()
    data_dict['w2_linear_grad_W'] = create_linear_grad_W_loc()
    data_dict['w2_linear_grad_b'] = create_linear_grad_b_loc()
    data_dict['w2_sigmoid_forward'] = create_sigmoid_forward_loc()
    data_dict['w2_sigmoid_grad_input'] = create_sigmoid_grad_input_loc()
    data_dict['w2_nll_forward'] = create_nll_forward_loc()
    data_dict['w2_nll_grad_input'] = create_nll_grad_input_loc()
    data_dict['w2_dist_to_training_samples'] = create_dist_to_training_samples_loc()
    data_dict['w2_nearest_neighbors'] = create_nearest_neighbors_loc()
    data_dict['w2_tree_weighted_entropy'] = create_tree_weighted_entropy_loc()
    data_dict['w2_tree_split_data_left'] = create_tree_split_data_left_loc()
    data_dict['w2_tree_split_data_right'] = create_tree_split_data_right_loc()
    data_dict['w2_tree_to_terminal'] = create_tree_to_terminal_loc()

    # Week 3
    data_dict['w3_dense_forward'] = create_dense_forward_loc()
    data_dict['w3_relu_forward'] = create_relu_forward_loc()
    data_dict['w3_l2_regularizer'] = create_l2_regularizer_loc()
    data_dict['w3_conv_matrix'] = create_conv_matrix_loc()
    data_dict['w3_box_blur'] = create_box_blur_loc()
    data_dict['w3_maxpool_forward'] = create_maxpool_forward_loc()
    data_dict['w3_flatten_forward'] = create_flatten_forward_loc()

    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=2)


def make_data_week_1_2_3_remote(file_path):
    data_dict = dict([])

    # Week 1
    data_dict['w1_linear_forward'] = create_week1_linear_forward()
    data_dict['w1_cal_pseudoinverse'] = create_week1_cal_pseudoinverse()
    data_dict['w1_L2_regression'] = create_week1_L2_regression()

    # Week 2
    data_dict['w2_linear_forward'] = create_linear_forward()
    data_dict['w2_linear_grad_W'] = create_linear_grad_W()
    data_dict['w2_linear_grad_b'] = create_linear_grad_b()
    data_dict['w2_sigmoid_forward'] = create_sigmoid_forward()
    data_dict['w2_sigmoid_grad_input'] = create_sigmoid_grad_input()
    data_dict['w2_nll_forward'] = create_nll_forward()
    data_dict['w2_nll_grad_input'] = create_nll_grad_input()
    data_dict['w2_dist_to_training_samples'] = create_dist_to_training_samples()
    data_dict['w2_nearest_neighbors'] = create_nearest_neighbors()
    data_dict['w2_tree_weighted_entropy'] = create_tree_weighted_entropy()
    data_dict['w2_tree_split_data_left'] = create_tree_split_data_left()
    data_dict['w2_tree_split_data_right'] = create_tree_split_data_right()
    data_dict['w2_tree_to_terminal'] = create_tree_to_terminal()

    # Week 3
    data_dict['w3_dense_forward'] = create_dense_forward()
    data_dict['w3_relu_forward'] = create_relu_forward()
    data_dict['w3_l2_regularizer'] = create_l2_regularizer()
    data_dict['w3_conv_matrix'] = create_conv_matrix()
    data_dict['w3_box_blur'] = create_box_blur()
    data_dict['w3_maxpool_forward'] = create_maxpool_forward()
    data_dict['w3_flatten_forward'] = create_flatten_forward()

    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=2)


if __name__ == '__main__':
    from argparse import ArgumentParser

    # parser = ArgumentParser()
    # parser.add_argument('--week2', type=bool, default=False, nargs='?', const=True)
    # args = parser.parse_args()

    # if args.week2:
    make_data_week_1_2_3_loc('../automark_server/assignments/local_tests.pickle')
    make_data_week_1_2_3_remote('../automark_server/assignments/remote_tests.pickle')
    # else:
    # make_data_week_1_loc('../automark_server/assignments/local_tests.pickle')
    # make_data_week_1_remote('../automark_server/assignments/remote_tests.pickle')

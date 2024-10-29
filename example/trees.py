import numpy as np


def tree_gini_index(Y_left, Y_right, classes):
    """Compute the Gini Index.
    # Arguments
        Y: class labels of the data set; np.array of size `(n_objects, 1)`
        split_left: `list` of indices of data points belonging to this split
        split_right: `list` of indices of data points belonging to this split
        classes: `list` of all class values
    # Output
        gini: scalar `float`
    """
    gini = 0.0
    #################
    ### YOUR CODE ###
    #################
    n_instances = float(Y_left.shape[0] + Y_right.shape[0])
    # sum weighted Gini index for each group
    for Y in [Y_left, Y_right]:
        size = float(Y.shape[0])
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = np.count_nonzero(Y == class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)

    return gini


def tree_weighted_entropy(Y_left, Y_right, classes):
    """Compute the weighted entropy.
    # Arguments
        Y_left: class labels of the data left set
            np.array of size `(n_objects, 1)`
        Y_right: class labels of the data right set
            np.array of size `(n_objects, 1)`
        classes: list of all class values
    # Output
        weighted_entropy: scalar `float`
    """
    S = Y_left.shape[0] + Y_right.shape[0]

    entropy_left = 0
    if Y_left.shape[0]:
        for c in classes:
            fraction = (Y_left == c).sum() / Y_left.shape[0]
            if fraction:
                entropy_left -= fraction * np.log2(fraction)

    entropy_right = 0
    if Y_right.shape[0]:
        for c in classes:
            fraction = (Y_right == c).sum() / Y_right.shape[0]
            if fraction:
                entropy_right -= fraction * np.log2(fraction)

    weighted_entropy = Y_left.shape[0] / S * entropy_left + Y_right.shape[0] / S * entropy_right
    
    return weighted_entropy


def tree_split_data_left(X, Y, feature_index, split_value):
    """Split the data `X` and `Y`, at the feature indexed by `feature_index`.
    If the value is less than `split_value` then return it as part of the left group.

    # Arguments
        X: np.array of size `(n_objects, n_in)`
        Y: np.array of size `(n_objects, 1)`
        feature_index: index of the feature to split at
        split_value: value to split between
    # Output
        (XY_left): np.array of size `(n_objects_left, n_in + 1)`
    """
    X_left, Y_left = None, None
    X_right, Y_right = None, None
    #################
    ### YOUR CODE ###
    #################

    left_ind = X[:, feature_index] < split_value

    XY_left = np.concatenate((X[left_ind], Y[left_ind]), axis=-1)

    return XY_left


def tree_split_data_right(X, Y, feature_index, split_value):
    """Split the data `X` and `Y`, at the feature indexed by `feature_index`.
    If the value is greater or equal than `split_value` then return it as part of the right group.

    # Arguments
        X: np.array of size `(n_objects, n_in)`
        Y: np.array of size `(n_objects, 1)`
        feature_index: index of the feature to split at
        split_value: value to split between
    # Output
        (XY_left): np.array of size `(n_objects_left, n_in + 1)`
    """
    X_left, Y_left = None, None
    X_right, Y_right = None, None
    #################
    ### YOUR CODE ###
    #################

    left_ind = X[:, feature_index] < split_value
    right_ind = ~left_ind

    XY_right = np.concatenate((X[right_ind], Y[right_ind]), axis=-1)

    return XY_right


def tree_best_split(X, Y):
    class_values = list(set(Y.flatten().tolist()))
    r_index, r_value, r_score = float("inf"),  float("inf"), float("inf")
    r_XY_left, r_XY_right = (X, Y), (X, Y)
    for feature_index in range(X.shape[1]):
        for row in X:
            XY_left = tree_split_data_left(X, Y, feature_index, row[feature_index])
            XY_right = tree_split_data_right(X, Y, feature_index, row[feature_index])
            XY_left, XY_right = (XY_left[:, :-1], XY_left[:, -1:]
                                 ), (XY_right[:, :-1], XY_right[:, -1:])
            gini = tree_gini_index(XY_left[1], XY_right[1], class_values)
            if gini < r_score:
                r_index, r_value, r_score = feature_index, row[feature_index], gini
                r_XY_left, r_XY_right = XY_left, XY_right
    return {'index': r_index, 'value': r_value, 'XY_left': r_XY_left, 'XY_right': r_XY_right}


def tree_to_terminal(Y):
    """The most frequent class label, out of the data points belonging to the leaf node,
    is selected as the predicted class.

    # Arguments
        Y: np.array of size `(n_objects)`

    # Output
        label: most frequent label of `Y.dtype`
    """
    label = None
    #################
    ### YOUR CODE ###
    #################

    values, counts = np.unique(Y, return_counts=True)
    label = values[np.argmax(counts)]

    return label


def tree_recursive_split(X, Y, node, max_depth, min_size, depth):
    XY_left, XY_right = node['XY_left'], node['XY_right']
    del(node['XY_left'])
    del(node['XY_right'])
    # check for a no split
    if XY_left[0].size <= 0 or XY_right[0].size <= 0:
        node['left_child'] = node['right_child'] = tree_to_terminal(
            np.concatenate((XY_left[1], XY_right[1])))
        return
    # check for max depth
    if depth >= max_depth:
        node['left_child'], node['right_child'] = tree_to_terminal(
            XY_left[1]), tree_to_terminal(XY_right[1])
        return
    # process left child
    if XY_left[0].shape[0] <= min_size:
        node['left_child'] = tree_to_terminal(XY_left[1])
    else:
        node['left_child'] = tree_best_split(*XY_left)
        tree_recursive_split(X, Y, node['left_child'], max_depth, min_size, depth+1)
    # process right child
    if XY_right[0].shape[0] <= min_size:
        node['right_child'] = tree_to_terminal(XY_right[1])
    else:
        node['right_child'] = tree_best_split(*XY_right)
        tree_recursive_split(X, Y, node['right_child'], max_depth, min_size, depth+1)


def build_tree(X, Y, max_depth, min_size):
    root = tree_best_split(X, Y)
    tree_recursive_split(X, Y, root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left_child'], depth+1)
        print_tree(node['right_child'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


def tree_predict_single(x, node):
    if isinstance(node, dict):
        if x[node['index']] < node['value']:
            return tree_predict_single(x, node['left_child'])
        else:
            return tree_predict_single(x, node['right_child'])

    return node


def tree_predict_multi(X, node):
    Y = np.array([tree_predict_single(row, node) for row in X])
    return Y[:, None]  # size: (n_object,) -> (n_object, 1)

import numpy as np


def dist_to_training_samples(x_input, training_set):
    """Calculate distance between an input sample and the N training samples.
    # Arguments
        x_input: samples for which we want to make a predicton
            np.array of size `(n_in,)`
        training_set: all our training samples
            np.array of size `(N, n_in)`
    # Output
        The distances between our input samples and training samples
        np.array of size `(N,)`
    """
    distances = ((training_set - x_input) ** 2).sum(axis=1) ** 0.5
    return distances


def nearest_neighbors(distances, training_labels):
    """Predict the label of the input sample given the distances of
        this sample to the training samples and the labels of the
        training samples.
    # Arguments
        distances: distances from the input sample to the N training samples
            np.array of size `(N,)`
        training_labels: true labels of the training samples
            np.array of size `(N,)`
    # Output
        prediction: 
    """
    prediction = training_labels[distances.argmin()]
    return prediction

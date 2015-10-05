"""
The Artificial Neural Network
"""

import numpy as np
import scipy


def sigmoid(X):
    return (1 + np.exp(-X)) ** -1


class ArtificialNeuralNetwork(object):

    def __init__(self, gamma, layer_sizes, num_hidden, epsilon=None,
                 max_iters=None, schema=None):
        """
        Construct an artificial neural network classifier

        @param gamma : weight decay coefficient
        @param layer_sizes:  Number of hidden layers
        @param num_hidden:  Number of hidden units in each hidden layer
        @param epsilon : cutoff for gradient descent
                         (need at least one of [epsilon, max_iters])
        @param max_iters : maximum number of iterations to run
                            gradient descent for
                            (need at least one of [epsilon, max_iters])
        """
        # Save parameters.
        self._gamma = gamma
        if layer_sizes != 1:
            raise ValueError('Assignment description specifies a single hidden'
                             ' layer.')
        self._num_layers = layer_sizes
        self._num_hidden = num_hidden
        self._epsilon = epsilon
        self._max_iters = max_iters
        self._schema = schema
        if self._schema is None:
            raise ValueError('Must provide schema for input data!')
        self._num_attrs = len(schema.feature_names)
        self._eta = 0.01

        # Create layer weights:
        self._shape = (self._num_hidden, self._num_attrs)
        self._weights = np.random.uniform(-0.1, 0.1, self._shape)
        self._out_weights = np.random.uniform(-0.1, 0.1, self._num_hidden)

    def fit(self, X, y, sample_weight=None):
        """
        Fit a neural network of layer_sizes * num_hidden hidden units using X,y
        """
        print(X.shape)
        # Create a 0-1 copy of y.
        y01 = y.copy()
        y01[y01 == -1] = 0

        for _ in range(self._max_iters):
            self.fit_iter(X, y01)


    def fit_iter(self, X, y):
        # Feed examples through network.
        hidden_ns = np.dot(self._weights, X.T)
        hidden_outs = sigmoid(hidden_ns)
        outer_ns = np.dot(hidden_outs.T, self._out_weights)
        outer_out = sigmoid(outer_ns)

        # Get partial derivatives for outer unit.
        outer_deriv = outer_out * (1 - outer_out) * (outer_out - y)
        outer_deriv = outer_deriv.reshape((len(outer_deriv), 1))
        outer_deriv = outer_deriv * hidden_outs.T
        # Add weight decay term
        outer_deriv = outer_deriv + 2 * self._gamma * self._out_weights.reshape((1, len(self._out_weights)))
        outer_deriv_total = outer_deriv.sum(axis=0)

        # Get partial derivatives for hidden units.
        for unit in range(self._num_hidden):
            dl = (hidden_outs[unit] * (1 - hidden_outs[unit])).reshape(hidden_outs.shape[1], 1)
            dl = dl * X * (outer_deriv[:, unit] * self._out_weights[unit] / hidden_outs[unit]).reshape(len(X), 1)
            # weight decay term:
            dl = dl + 2 * self._gamma * self._weights[unit].reshape(1, len(self._weights[unit]))
            # sum over all examples
            dl = dl.sum(axis=0)
            self._weights[unit] = self._weights[unit] - self._eta * dl
        # ho_shape = (hidden_outs.shape[0], 1, hidden_outs.shape[1])
        # x_shape = (1, X.shape[0], X.shape[1])
        # hidden_deriv = (hidden_outs * (1 - hidden_outs)).reshape(ho_shape)
        # hidden_deriv *= X.T.reshape(x_shape)
        # hidden_deriv *= outer_deriv.T.reshape(ho_shape)
        # hidden_deriv *= self._out_weights.reshape((self._out_weights, 1, 1))
        # hidden_deriv /= hidden_outs.

        self._out_weights = self._out_weights - self._eta * outer_deriv_total

    def predict(self, X):
        """ Predict -1/1 output """
        proba = self.predict_proba(X)
        proba[proba <= 0.5] = -1
        proba[proba != -1] = 1
        return proba

    def predict_proba(self, X):
        """ Predict probabilistic output """
        # Feed examples through network.
        hidden_ns = np.dot(self._weights, X.T)
        hidden_outs = sigmoid(hidden_ns)
        outer_ns = np.dot(hidden_outs.T, self._out_weights)
        outer_out = sigmoid(outer_ns)
        return outer_out

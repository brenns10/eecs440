"""
The Artificial Neural Network
"""

from __future__ import division
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
        # Convenience variables:
        # k - number of examples
        # m - number of hidden units
        # n - number of attributes
        k, m, n = len(X), self._num_hidden, self._num_attrs

        # Feed examples through network.
        hidden_ns = np.dot(self._weights, X.T)  # m*k
        hidden_outs = sigmoid(hidden_ns)  # m*k
        outer_ns = np.dot(hidden_outs.T, self._out_weights)  # k
        outer_out = sigmoid(outer_ns)  # k
        #print('%03f/%r' % (outer_out[0], y[0]))

        # Get partial derivatives for outer unit.
        outer_deriv = outer_out * (1 - outer_out) * (outer_out - y)  # k
        outer_deriv = outer_deriv.reshape((k, 1))  # k*1
        outer_deriv = outer_deriv * hidden_outs.T  # k*m
        # Add weight decay term
        outer_deriv_total = outer_deriv.sum(axis=0)  # m
        outer_deriv_total = outer_deriv_total + 2 * self._gamma * self._out_weights

        # Get partial derivatives for hidden units.
        dl = (hidden_outs * (1 - hidden_outs)).reshape(m, k, 1) # m*k
        dl = dl * X.reshape(1, k, n) * (outer_deriv.T * self._out_weights.reshape(m, 1) / hidden_outs).reshape(m, k, 1)  # m*k*n
        # sum over all examples
        dl = dl.sum(axis=1)
        # weight decay term:
        dl = dl + 2 * self._gamma * self._weights

        # Update outer and hidden weights.
        self._out_weights = self._out_weights - self._eta * outer_deriv_total
        self._weights = self._weights - self._eta * dl

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

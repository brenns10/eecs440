"""
The Artificial Neural Network
"""

import numpy as np
import scipy


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

        # Create layer weights:
        self._shape = (self._num_layers, self._num_hidden, self._num_attrs)
        self._weights = np.random.uniform(-0.1, 0.1, shape=self._shape)
        self._out_weights = np.random.uniform(-0.1, 0.1,
                                              shape=self._num_hidden)

    def fit(self, X, y, sample_weight=None):
        """
        Fit a neural network of layer_sizes * num_hidden hidden units using X,y
        """
        pass

    def predict(self, X):
        """ Predict -1/1 output """
        pass

    def predict_proba(self, X):
        """ Predict probabilistic output """
        pass

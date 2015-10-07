"""
The Artificial Neural Network
"""

from __future__ import division
import numpy as np
import sys


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
        self._layers = []
        prev_input = self._num_attrs
        for _ in range(self._num_layers):
            print('%d x ' % prev_input, end='')
            shape = (prev_input, self._num_hidden)
            weights = np.random.uniform(-0.1, 0.1, shape)
            self._layers.append(weights)
            prev_input = shape[1]
        print('%d x 1' % prev_input)
        out_weights = np.random.uniform(-0.1, 0.1, (prev_input, 1))
        self._layers.append(out_weights)

    def standardize_inputs(self, X):
        """
        Standardize the values of X that go into the ANN.

        As per the assignment description, nominal attributes should be encoded
        1 to k.  Continuous attributes are normalized.
        :param X: k by n matrix of input data
        """
        # Float64 can hold all uint8 values, and won't overflow so quickly when
        # you do stuff with it.
        X = X.astype(np.float64)

        # Iterate over each attribute.
        for i, name in enumerate(self._schema.feature_names):
            if self._schema.is_nominal(i):
                # Nominal values should be 1 to k.  They come 0 to k-1, so just
                # increase them by one.
                X[:, i] += 1
            else:
                # Normalize the continuous values.
                attr = X[:, i]
                mean = attr.mean()
                std = attr.std()
                X[:, i] = (attr - mean) / std
        return X

    def fit(self, X, y, sample_weight=None):
        """
        Fit a neural network of layer_sizes * num_hidden hidden units using X,y

        :param X: k by n matrix of inputs.
        :param y: k-length list of outputs.
        :param sample_weight: ignored right now <3
        """
        X = self.standardize_inputs(X)
        # Create a 0/1 copy of y, instead of -1/1.
        y01 = y.copy()
        y01[y01 == -1] = 0

        # Do training iterations.
        for i in range(self._max_iters):
            # Update the iteration count on the same line so we have status
            # info without flooding the console.
            print('\b' * 11, end='')
            print('iter % 6d' % (i + 1), end='')
            sys.stdout.flush()
            # Do a single training iteration.
            self.fit_iter(X, y01)

        # Newline after the iteration count line.
        print()

    def _feed_forward(self, X):
        """
        Feed an example or list of examples through the network.
        :param X: k by n matrix of examples.
        :returns: list of (k by m) matrices of layer outputs, one per layer
        """
        X = X.astype(np.float64)
        layer_outputs = []
        for weights in self._layers:
            X = sigmoid(np.dot(X, weights))
            layer_outputs.append(X)
        return layer_outputs

    def _gradient_outer(self, X, y_hat, y):
        """
        Return the gradient for an outer layer.
        :param X: inputs to this layer (k by n)
        :param y_hat: outputs (predictions) from this layer (k by m)
        :param y: expected output from this layer (k)
        :return: gradients of outer layer weights for each example (k*n*m)
        """
        k, n = X.shape
        m = y_hat.shape[1]  # m should be 1, but just in case
        y = y.reshape(y_hat.shape)  # make y  k by m (just in case)
        # Compute dl/dw
        dl = y_hat * (1 - y_hat) * (y_hat - y)  # k by m
        dl = dl.reshape(k, 1, m) * X.reshape(k, n, 1)
        return dl

    def _gradient_hidden(self, X, outs, dL_next, W_next):
        """
        Return the gradient for a hidden layer.
        :param X: inputs to this layer (k by n)
        :param outs: outputs from this layer (k by m)
        :param dL_next: gradients from the following layer (k by m by p)
        :param W_next: (m by p)
        :return: gradients of hidden layer weights for each example (k*n*m)
        """
        k, n, m, p = X.shape[0], X.shape[1], outs.shape[1], W_next.shape[1]
        # The formula is:
        # dl/dw = h(n_j) (1 - h(n_j)) x_{ji}
        #         \sum_{k\in Downstream(j)} (dl/dw_{jk}) (w_{kj}/x_{kj})
        # However, h(n_j) is the same as x_{kj}, so it cancels out!

        # Compute the summation portion of of the above formula.
        downstream = (W_next.reshape(1, m, p) *
                      dL_next).sum(axis=2).reshape(k, 1, m)
        # Finish computing dl/dw
        dl = (1 - outs).reshape(k, 1, m) * downstream
        dl = dl * X.reshape(k, n, 1)
        return dl  # k by n by m

    def fit_iter(self, X, y):
        # Feed examples through network.
        outputs = self._feed_forward(X)

        # The inputs to each layer are just the outputs shifted.
        inputs = [X] + outputs[:-1]
        # We'll also need the weights of the "next" layer.
        layers_shifted = self._layers[1:] + [self._layers[0]]

        # Do backpropagation, starting with output layer.
        prev_grad = None  # hold the gradient from the last layer for backprop
        gradients = []
        for x, y_hat, W_next in reversed(list(zip(inputs, outputs, layers_shifted))):
            if prev_grad is None:
                # For the output layer:
                prev_grad = self._gradient_outer(x, y_hat, y)
            else:
                # For all the other layers:
                prev_grad = self._gradient_hidden(x, y_hat, prev_grad, W_next)
            gradients.append(prev_grad)

        # Now we have a list of k by (n by m) gradients, one for each layer.
        gradients.reverse()

        # Do parameter updates!
        new_layers = []
        for i, layer in enumerate(self._layers):  # zipping gave errors :(
            gradient = gradients[i]
            # sum up gradients across k:
            gradient = gradient.sum(axis=0)
            # add in weight decay term
            gradient = gradient + 2 * self._gamma * layer
            # do layer weight update
            new_layers.append(layer - self._eta * gradient)
        self._layers = new_layers

    def predict(self, X):
        """ Predict -1/1 output """
        proba = self.predict_proba(X)
        proba[proba <= 0.5] = -1
        proba[proba != -1] = 1
        return proba

    def predict_proba(self, X):
        """ Predict probabilistic output """
        # Feed examples through network.
        results = self._feed_forward(X)
        # Output will expect k-length list, not k by 1 matrix.
        return results[-1].reshape(X.shape[0])

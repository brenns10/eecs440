"""
The Artificial Neural Network
"""
import numpy as np
import scipy

class ArtificialNeuralNetwork(object):


    def __init__(self, gamma, layer_sizes, num_hidden, epsilon=None, max_iters=None):
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
        pass

    def fit(self, X, y, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        pass

    def predict(self, X):
        """ Predict -1/1 output """
        pass

    def predict_proba(self, X):
        """ Predict probabilistic output """
        pass

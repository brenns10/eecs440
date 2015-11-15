"""
An implementation of boosting as a wrapper for a classifier
"""
import numpy as np
import scipy

from dtree import DecisionTree
from ann import ArtificialNeuralNetwork
from nbayes import NaiveBayes
from logistic_regression import LogisticRegression

CLASSIFIERS = {
    'dtree'                 : DecisionTree,
    'ann'                   : ArtificialNeuralNetwork,
    'nbayes'                : NaiveBayes,
    'logistic_regression'   : LogisticRegression,
}

class Booster(object):

    def __init__(self, algorithm, iters, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of boosting to do
        @param params : Parameters for the classification algorithm
        """
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

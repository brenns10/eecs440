"""
An implementation of bagging as a wrapper for a classifier
"""

from __future__ import division, print_function

import numpy as np

from dtree import DecisionTree
from ann import ArtificialNeuralNetwork
# from nbayes import NaiveBayes
# from logistic_regression import LogisticRegression

CLASSIFIERS = {
    'dtree': DecisionTree,
    'ann': ArtificialNeuralNetwork,
    # 'nbayes'                : NaiveBayes,
    # 'logistic_regression'   : LogisticRegression,
}


class Bagger(object):

    def __init__(self, algorithm, iters, flip_probability=0, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of bagging to do
        @param params : Parameters for the classification algorithm
        """
        self._algorithm = algorithm
        self._cls = CLASSIFIERS[self._algorithm]
        self._iters = iters
        self._params = params
        self._classifiers = []
        self._pflip = flip_probability

    def fit(self, X, y):
        y = y.copy()
        flips = np.random.uniform(0.0, 1.0, len(y)) < self._pflip
        y[flips & (y == -1)] = 1
        y[flips & (y == 1)] = -1
        for i in range(self._iters):
            print('Bagger: fit classifier %03d/%03d' % (i+1, self._iters))
            indexer = np.random.randint(0, len(y), len(y))
            new_X = X[indexer]
            new_y = y[indexer]
            classifier = self._cls(**self._params)
            classifier.fit(new_X, new_y)
            self._classifiers.append(classifier)

    def predict(self, X):
        predictions = self.predict_proba(X)
        predictions[predictions > 0.5] = 1
        predictions[predictions != 1] = -1
        return predictions

    def predict_proba(self, X):
        predictions = [c.predict(X) for c in self._classifiers]
        positives = [(p == 1) for p in predictions]
        return sum(positives) / self._iters

"""
An implementation of boosting as a wrapper for a classifier
"""

from __future__ import division, print_function

import numpy as np

from dtree import DecisionTree
from ann import ArtificialNeuralNetwork
# from nbayes import NaiveBayes
# from logistic_regression import LogisticRegression

CLASSIFIERS = {
    'dtree'                 : DecisionTree,
    'ann'                   : ArtificialNeuralNetwork,
    # 'nbayes'                : NaiveBayes,
    # 'logistic_regression'   : LogisticRegression,
}


class Booster(object):

    def __init__(self, algorithm, iters, flip_probability=0, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of boosting to do
        @param params : Parameters for the classification algorithm
        """
        self._algorithm = algorithm
        self._cls = CLASSIFIERS[self._algorithm]
        self._iters = iters
        self._params = params
        self._classifiers = []
        self._weights = []
        self._pflip = flip_probability

    def fit(self, X, y):
        y = y.copy()
        N = len(y)
        flips = np.random.uniform(0.0, 1.0, N) < self._pflip
        y[flips & (y == -1)] = 1
        y[flips & (y == 1)] = -1

        # Initialize weights to 1/N
        weights = np.zeros(N)
        weights[:] = 1/N

        for i in range(self._iters):
            print('Booster: fit classifier %03d/%03d' % (i+1, self._iters))
            # Train a classifier, and then classify the training data.
            classifier = self._cls(**self._params)
            classifier.fit(X, y, sample_weight=weights)
            predictions = classifier.predict(X)

            # Epsilon is the weighted training error.  Use it to calculate
            # alpha, the weight of this classifier in the vote.
            epsilon = weights[predictions != y].sum()
            alpha = 0.5 * np.log((1-epsilon)/epsilon)

            # Store this classifier and its weight.
            self._classifiers.append(classifier)
            self._weights.append(alpha)

            # Finally, update the weights of each example.
            new_weights = weights * np.exp(-alpha * y * predictions)
            weights = new_weights / new_weights.sum()  # normalize

    def predict(self, X):
        predictions = self.predict_proba(X)
        predictions[predictions > 0.5] = 1
        predictions[predictions != 1] = -1
        return predictions

    def predict_proba(self, X):
        # Sum of all alpha:
        total_weights = sum(self._weights)

        # Now we sum up the weighted vote for each classifier.
        output = np.zeros(X.shape[0])
        for classifier, alpha in zip(self._classifiers, self._weights):
            predictions = classifier.predict(X)
            weighted_predictions = alpha * predictions / total_weights
            output = output + weighted_predictions

        # Output should range from -1 to 1, which is cool, but probabilistic
        # output should be in [0,1], since it expresses (sorta) the probability
        # of the example being positive.  So, we transform it.
        output = output/2 + 0.5
        return output

"""
The Naive Bayes Classifier
"""

import numpy as np
import scipy


class NaiveBayes(object):

    def __init__(self, m=None, schema=None):
        """
        Constructs a Naive Bayes classifier

        @param m : Smoothing parameter (0 for no smoothing, None to determine
        via internal cross validation)
        """
        self._m = m
        if schema is None:
            raise ValueError('Must provide input data schema.')
        self._schema = schema

        # self._conditional_probabilities is a list of 2 by v matrices, each
        # one containing probabilities for each value of the attribute, given a
        # class label.
        self._conditional_probabilities = []

        # Here we initialize all conditional probabilities to 1/v.
        for i, _ in enumerate(len(schema.nominal_values)):
            if schema.is_nominal(i):
                # For nominal values, we make v=the number of nominal values.
                v = len(schema.nominal_values[i])
            else:
                # For continuous ones, we just use 10 bins...
                v = 10

            new_probs = np.zeros((2, v))
            new_probs[:, :] = 1/v
            self._conditional_probabilities.append(new_probs)

    def fit(self, X, y):
        pass  # add code here

    def predict(self, X):
        pass  # add code here

    def predict_proba(self, X):
        pass  # add code here

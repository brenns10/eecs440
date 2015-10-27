"""
The Logistic Regression Classifier
"""

from __future__ import division

import numpy as np
import scipy

from util import internal_cross_validation


class LogisticRegression(object):

    def __init__(self, schema=None, **kwargs):
        """
        Constructs a logistic regression classifier

        @param lambda : Regularisation constant parameter
        """
        # lambda is a keyword in Python... I have to use kwargs.get instead of
        # listing it as a keyword argument.
        self._lambda = kwargs.get('lambda', None)

        if schema is None:
            raise ValueError('Must provide input data schema.')
        self._schema = schema

        # Randomly initialize parameters from [-1, 1]
        self._w = np.random.uniform(-1, 1, size=(len(schema.feature_names), 1))
        self._b = np.random.uniform(-1, 1)

        # eta controls the rate of gradient descent
        self._eta = 0.1  # as in ann.py

        # cutoff - the value that the l1 norm of the gradient vector should
        # drop below as a termination condition for gradient descent
        self._cutoff = 1e-3  # this is arbitrary right now

    def _standardize_continuous(self, X, i):
        """
        Standardize continuous attributes, by normalizing them.
        :param X: example-by-features NumPy matrix
        :param i: index of feature to standardize
        :returns: vector of normalized values
        """
        return (X[:, i] - self._means[i]) / self._stds[i]

    def _standardize_nominal(self, X, i):
        """
        Standardize nominal attributes, mapping them to an int 1 ... v.
        :param X: example-by-features NumPy matrix
        :param i: index of feature to standardize
        :returns: 1...v values
        """
        nom_vals = np.array(self._schema.nominal_values[i], dtype=np.float64)
        sorter = np.argsort(nom_vals)
        new_vals = sorter[np.searchsorted(nom_vals, X[:, i], sorter=sorter)]
        return new_vals.astype(np.float64) + 1

    def _standardize_inputs(self, X):
        """
        Return updated inputs modified for the classifier.
        :param X: example-by-features NumPy matrix
        :returns: new, standardized feature matrix
        """
        X = X.copy()
        for i, name in enumerate(self._schema.feature_names):
            if self._schema.is_nominal(i):
                X[:, i] = self._standardize_nominal(X, i)
            else:
                X[:, i] = self._standardize_continuous(X, i)
        return X.astype(np.float64)

    def fit(self, X, y):
        """
        Fit the logistic regression classifier to data.
        :param X: example-by-features NumPy matrix
        :param y: example-length vector of class labels
        """
        self._means = np.mean(X, 0)
        self._stds = np.std(X, 0)
        X = self._standardize_inputs(X)
        ymat = y.reshape((y.shape[0], 1))

        if self._lambda is None:
            self._lambda = internal_cross_validation(
                LogisticRegression, {'schema': self._schema}, 'lambda',
                [0, 0.001, 0.01, 0.1, 1, 10, 100], 'accuracy', X, y
            )

        # Gradient formulae:
        # - For weights:
        #   \frac{d}{dw} = \frac{1}{1+e^{y_i(w \cdot x_i + b)}} (-y_i x_i)
        # - For b:
        #   \frac{d}{dw} = \frac{1}{1+e^{y_i(w \cdot x_i + b)}} (-y_i)
        # This is summed up for each example, multiplied by C, and then the
        # penalty term is added.

        # We will break out of this loop by checking termination condition at
        # bottom.
        while True:
            # (k by 1) = (k by n) dot (n by 1)
            dotprod = np.dot(X, self._w)
            # (k by 1) still
            fraction = 1/(1 + ymat * np.exp(dotprod + self._b)) * (-ymat)
            # GRADIENT FOR W: (k by n) again
            wgrad = fraction * X
            # sum across the k examples to get (n) length vector
            wgrad = wgrad.sum(axis=0)
            # bring it back to an (n by 1) matrix
            wgrad = wgrad.reshape(wgrad.shape[0], 1)
            # now weight decay term and lambda
            wgrad = wgrad + self._lambda * self._w
            # GRADIENT FOR B: (k by 1)
            bgrad = fraction
            # Sum over all k, multiply by lambda, and add in weight decay term
            bgrad = bgrad.sum() + self._lambda * self._b

            # Do the parameter update.
            self._w = self._w - self._eta * wgrad
            self._b = self._b - self._eta * bgrad

            # Check for termination condition.
            l1norm = np.abs(wgrad).sum() + np.abs(bgrad)
            #print(wgrad.reshape(wgrad.shape[0]))
            #print('%r, %r' % (l1norm, bgrad))
            if np.all(np.abs(wgrad) < self._cutoff) and \
               np.abs(bgrad) < self._cutoff:
                break

    def predict(self, X):
        rv = np.where(self.predict_proba(X) > 0.5, 1, -1)
        ## The following code checks the predict_proba output by computing wx +
        ## b > 0.  Just as a sanity check.
        # rv2 = np.where(np.dot(X, self._w) + self._b > 0, 1, -1)
        # rv2 = rv2.reshape(rv.shape[0])
        # if np.any(rv != rv2):
        #     print('badness!')
        #     print(rv2)
        return rv

    def predict_proba(self, X):
        X = self._standardize_inputs(X)
        p_pos = (1 / (1 + np.exp(-np.dot(X, self._w) - self._b)))
        p_neg = (1 / (1 + np.exp(np.dot(X, self._w) + self._b)))
        rv_mat = p_pos / (p_pos + p_neg)
        rv = rv_mat.reshape(rv_mat.shape[0])
        #print(rv)
        return rv

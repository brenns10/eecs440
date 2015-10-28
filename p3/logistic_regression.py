"""
The Logistic Regression Classifier
"""

from __future__ import division

import numpy as np
import scipy.optimize

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
        nfeatures = len(schema.feature_names)

        # Initialize parameters to 0.
        self._w = np.zeros((nfeatures, 1))
        self._b = 0

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

    def loss(self, w_and_b, X, ymat):
        """
        Return the value of the loss function.
        """
        w = w_and_b[:-1]
        w = w.reshape((np.product(w.shape), 1))
        b = w_and_b[-1]

        # (k by 1) = (k by n) dot (n by 1)
        dotprod = np.dot(X, w)
        # still (k by 1)
        log = np.log(1 + np.exp(- ymat * (dotprod + b)))
        # now a scalar
        log_sum = log.sum()
        # loss = (1/2) lambda ||w||^2 + sum log( 1 + exp(...))
        loss = 0.5 * self._lambda * np.dot(w.T, w) + log_sum
        return loss

    def jac(self, w_and_b, X, ymat):
        """
        Return the Jacobian (gradient) of the loss function.
        """
        w = w_and_b[:-1]
        w = w.reshape((w.shape[0], 1))
        b = w_and_b[-1]

        # Gradient formulae:
        # - For weights:
        #   \frac{dL}{dw} = \frac{1}{1+e^{y_i(w \cdot x_i + b)}} (-y_i x_i)
        # - For b:
        #   \frac{dL}{db} = \frac{1}{1+e^{y_i(w \cdot x_i + b)}} (-y_i)
        # This is summed up for each example, multiplied by C, and then the
        # penalty term is added.

        # (k by 1) = (k by n) dot (n by 1)
        dotprod = np.dot(X, w)
        # (k by 1) still
        fraction = 1/(1 + np.exp(ymat * (dotprod + b))) * (-ymat)
        # GRADIENT FOR W: (k by n) again
        wgrad = fraction * X
        # sum across the k examples to get (n) length vector
        wgrad = wgrad.sum(axis=0)
        # bring it back to an (n by 1) matrix
        wgrad = wgrad.reshape(wgrad.shape[0], 1)
        # now weight decay term and lambda
        wgrad = wgrad + self._lambda * w
        # GRADIENT FOR B: (k by 1)
        bgrad = fraction
        # Sum over all k, multiply by lambda
        bgrad = bgrad.sum()

        return np.append(wgrad.reshape(wgrad.shape[0]), [bgrad])

    def fit(self, X, y):
        """
        Fit the logistic regression classifier to data.
        :param X: example-by-features NumPy matrix
        :param y: example-length vector of class labels
        """
        self._means = np.mean(X, 0)
        self._stds = np.std(X, 0)

        Xold = X
        X = self._standardize_inputs(X)
        ymat = y.reshape((y.shape[0], 1))

        if self._lambda is None:
            self._lambda = internal_cross_validation(
                LogisticRegression, {'schema': self._schema}, 'lambda',
                [0, 0.001, 0.01, 0.1, 1, 10, 100], 'auc', Xold, y
            )

        optres = scipy.optimize.minimize(self.loss,
                                         np.append(self._w, [self._b]),
                                         args=(X, ymat), jac=self.jac,
                                         method='Newton-CG')

        self._w = optres.x[:-1]
        self._w = self._w.reshape((self._w.shape[0], 1))  # make it a matrix
        self._b = optres.x[-1]

    def predict(self, X):
        """Return a vector of -1/+1 predictions."""
        rv = np.where(self.predict_proba(X) > 0.5, 1, -1)
        return rv

    def predict_proba(self, X):
        """Return the probability of positive."""
        X = self._standardize_inputs(X)
        p_pos = (1 / (1 + np.exp(-np.dot(X, self._w) - self._b)))
        p_neg = (1 / (1 + np.exp(np.dot(X, self._w) + self._b)))
        rv_mat = p_pos / (p_pos + p_neg)
        rv = rv_mat.reshape(rv_mat.shape[0])
        return rv

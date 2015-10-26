"""
The Naive Bayes Classifier
"""

from __future__ import division
import numpy as np
import scipy

from util import internal_cross_validation

NUM_BINS = 10


class NaiveBayes(object):

    def __init__(self, m_value=None, schema=None):
        """
        Constructs a Naive Bayes classifier

        :param m: Smoothing parameter (0 for no smoothing, None to determine
          via internal cross validation)
        :param schema: Schema object generated from data.py.
        """
        # Save parameters.
        self._m = m_value
        if schema is None:
            raise ValueError('Must provide input data schema.')
        self._schema = schema

        # self._params is a list of 2 by v matrices, each one containing
        # probabilities for each value of the attribute, given a class label.
        self._params = []

        # Here we initialize all conditional probabilities to 1/v.
        for i, _ in enumerate(schema.feature_names):
            if schema.is_nominal(i):
                # For nominal values, we make v=the number of nominal values.
                v = len(schema.nominal_values[i])
            else:
                # For continuous ones, we just use 10 bins...
                v = NUM_BINS

            new_probs = np.zeros((2, v))
            new_probs[:, :] = 1/v
            self._params.append(new_probs)

        # This is the probability distirbution of y.  Initialize to None so we
        # get an error later if we don't fit it.
        self._yparam = None

        # These contain the min and max.  The step is (max - min) / NUM_BINS.
        self._mins = [None] * len(schema.feature_names)
        self._maxs = [None] * len(schema.feature_names)

    def _standardize_nominal(self, X, i):
        """
        Standardize nominal inputs by returning their indices in the Schema.
        :param X: examples-by-features NumPy matrix
        :param i: index of the feature to standardize
        :returns: the column's indices
        """
        nom_vals = np.array(self._schema.nominal_values[i], dtype=np.float64)
        sorter = np.argsort(nom_vals)
        new_vals = sorter[np.searchsorted(nom_vals, X[:, i], sorter=sorter)]
        return new_vals

    def _standardize_continuous(self, X, i):
        """
        Standardize continuous inputs by binning them.
        :param X: examples-by-features NumPy matrix
        :param i: index of the feature to standardize
        :returns: the column, binned up
        """
        col = X[:, i]
        # If we haven't gotten mins and maxs yet, save them.
        if self._maxs[i] is None:
            self._maxs[i] = col.max()
            self._mins[i] = col.min()
        # Now subtract out the mins and divide by the step to get which bin
        # each item is in.
        col = col - self._mins[i]
        col = col / (self._maxs[i] - self._mins[i]) * NUM_BINS
        col = np.floor(col)
        # Assign values that are above/below the max/min to the highest/lowest
        # bin (since there's no guarantee that training data will include the
        # true max/min).
        col[col < 0] = 0
        col[col >= NUM_BINS] = NUM_BINS - 1
        return col

    def _standardize_inputs(self, X):
        """
        Modify attribute inputs for this classifier.
        :param X: examples-by-features NumPy matrix.
        :returns: new, standardized feature matrix
        """
        X = X.copy()  # may not be necessary
        for i, name in enumerate(self._schema.feature_names):
            if self._schema.is_nominal(i):
                X[:, i] = self._standardize_nominal(X, i)
            else:
                X[:, i] = self._standardize_continuous(X, i)
        return X.astype(int)

    def fit(self, X, y):
        """
        Fit Naive Bayes classifier to the given data.

        This uses the current values of self._params as our "prior estimates".
        When smoothing is used, these will be used in the m-Estimate.  Since
        the params are initialized to 1/v, this will usually be LaPlace
        smoothing, but if you were to fit this class multiple times, it would
        use the previous model as a prior estimate, which may be interesting.

        :param X: An examples-by-features NumPy matrix.
        :param y: An array of +/- 1 class labels.
        """
        # Standardize everything.
        Xstd = self._standardize_inputs(X)
        ystd = y.copy()
        ystd[y == -1] = 0
        yvals = np.bincount(ystd)

        # Select parameter m by internal cross validation.
        if self._m is None:
            self._m = internal_cross_validation(
                NaiveBayes, {'schema': self._schema}, 'm_value',
                [0, 0.001, 0.001, 0.1, 1, 10, 100], 'accuracy', X, y
            )

        # Set the conditional probabilities using smoothing.
        for i, matrix in enumerate(self._params):
            mp = self._m * matrix
            for yval, count in enumerate(yvals):
                Xrel = Xstd[ystd == yval]
                colbincount = np.bincount(Xrel[:, i], minlength=mp.shape[1])
                matrix[yval] = (mp[yval] + colbincount)
                matrix[yval] = matrix[yval] / (count + self._m)
            self._params[i] = matrix

        # Set the probabilities of y.
        self._yparam = yvals / len(ystd)

    def predict(self, X):
        """
        Return the predicted class label for X.
        :param X: An examples-by-features NumPy matrix.
        :returns: predicted class label.
        """
        prob_positive_given_X = self.predict_proba(X)
        return np.where(prob_positive_given_X > 0.5, 1, -1)

    def predict_proba(self, X):
        """
        Return the probability that the class label is +1 given X.
        :param X: An examples-by-features NumPy matrix.
        :returns: Pr[Y=1|X]
        """
        Xstd = self._standardize_inputs(X)
        Pacc = np.zeros((2, X.shape[0]))
        for y, yprob in enumerate(self._yparam):
            for i, matrix in enumerate(self._params):
                # This is a v-length vector of probabilities conditioned on y
                probabilities = matrix[y]
                # The X values are indices into this vector!  We just sum up
                # logs (instead of multiplying).  We silence divide by zero
                # errors because that's what NumPy gives when it takes the log
                # of 0.  Thankfully, np.log(0) = -inf, and np.exp(-np.inf) = 0,
                # so we're safe here!
                with np.errstate(divide='ignore'):
                    Pacc[y] += np.log(probabilities[Xstd[:, i].astype(int)])
            Pacc[y] += np.log(yprob)
        # Now, return the probability Y is positive conditioned on X!
        probs = np.exp(Pacc)
        with np.errstate(invalid='ignore'):
            # We silence invalid value warnings here, because that's what we
            # get when we do 0/0.  In this case, we haven't seen any training
            # examples with those values, and m is set to 0, so we just go
            # 50/50 and assign 0.5!
            rv = probs[1] / (probs[0] + probs[1])
        rv[np.isnan(rv)] = 0.5
        return rv

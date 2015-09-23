"""
The Decision Tree Classifier
"""
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.stats import mode
from scipy.ndimage.interpolation import shift
import logging
log = logging.getLogger('dtree')
log.setLevel(logging.CRITICAL)


def entropy(l):
    """
    Return the entropy of a vector of nonnegative discrete integers.

    Shannon Entropy is a measure of the "information content" of a random
    variable.  The more widely dispersed the possible values of the RV, the
    higher its entropy.  Entropy is measured in bits (as in, the theoretical
    minimum amount of bits it take to represent the RV).  A RV which was 1 with
    probability 1 would have entropy of 0.  A RV which took on two values with
    equal probability would have entropy of 1 bit, etc.  The entropy function
    is denoted by H(X), and the definition is as follows:

        :math:`H(X) = - \sum_{x\in X} p(X=x) \log_2(p(X=x))`

    :param l: Array of nonnegative integers/bools.
    :type l: numpy.array or similar
    :returns: The entropy of the array.
    """

    probabilities = np.bincount(l) / len(l)
    with np.errstate(divide='ignore'):  # ignore log(0) errors, we'll handle
        log_probabilities = np.log2(probabilities)
        log_probabilities[~np.isfinite(log_probabilities)] = 0
        log_probabilities[np.isnan(log_probabilities)] = 0
    return -np.sum(probabilities * log_probabilities)


def joint_dataset(l1, l2):
    """
    Create a joint dataset for two non-negative integer (boolean) arrays.

    Works best for integer arrays with values [0,N) and [0,M) respectively.
    This function will create an array with values [0,N*M), each value
    representing a possible combination of values from l1 and l2.  Essentially,
    this is equivalent to zipping l1 and l2, but much faster by using the NumPy
    native implementations of elementwise addition and multiplication.

    :param l1: first integer vector (values within 0-n)
    :type l1: numpy.array or similar
    :param l2: second integer vector (values with 0-m)
    :type l2: numpy.array or similar
    :returns: integer vector expressing states of both l1 and l2
    """
    N = np.max(l1) + 1
    return l2 * N + l1


def mutual_info(l1, l2):
    """
    Return the mutual information of non-negative integer arrays.

    Again, will work best for arrays with values [0,N), where N is rather
    small.  This will compute the mutual information (a measure of "shared
    entropy") between two arrays.  The mutual information between two arrays is
    maximized when one is completely dependant on the other, and minimized if
    and only if they are independent.  (Note that this really applies to random
    variables.  Measurements of random variables obviously won't always
    evaluate to completely independent probabilities, and so they won't always
    have exactly 0 mutual information).  The mathematical definition is:

        :math:`I(X; Y) = H(X) + H(Y) - H(X,Y)`

    :param l1: first integer vector (X)
    :type l1: numpy.array or similar
    :param l2: first integer vector (Y)
    :type l2: numpy.array or similar
    :retuns: mutual information, as a float
    """
    return entropy(l1) + entropy(l2) - entropy(joint_dataset(l1, l2))


def mutual_info_fast(l1, l2, l1_entropy, l2_entropy):
    """
    Compute mutual info without recomputing the entropy of l1 and l2.

    This function is useful when you are going to be computing many mutual
    information values.  Instead of blindly recomputing the entropy of each
    vector again and again, you may do it once and supply it to this function
    in order to save on that computation.

    :param l1: first integer vector (X)
    :type l1: numpy.array or similar
    :param l2: first integer vector (Y)
    :type l2: numpy.array or similar
    :param float l1_entropy: entropy of ``l1`` (precomputed)
    :param float l2_entropy: entropy of ``l2`` (precomputed)
    :retuns: mutual information, as a float
    """
    return l1_entropy + l2_entropy - entropy(joint_dataset(l1, l2))


class DecisionTree(object):

    def __init__(self, depth=None, schema=None, used=None):
        """
        Constructs a Decision Tree Classifier

        :param depth: maximum depth of the tree, or None for no maximum depth
        :param schema: schema of the attributes, so we can tell nominal from
          continuous attributes
        :param used: array of bools, used[i] == True if attribute i has already
          been used.
        """
        if schema is None:
            raise ValueError('Must provide schema to DecisionTree!')
        # Schema will tell us whether each attribute is nominal or continuous.
        self._schema = schema
        # How many more levels down we can go.
        self._allowed_depth = depth
        self._depth = 1
        self._size = 1
        # A numpy.array of booleans, used[i]=True if attribute i has been used.
        self._used = used
        # None if internal node, otherwise the label.
        self._label = None
        # Attribute that we test.
        self._attribute = None
        # If we're testing a continuous attribute, this is the cutoff
        self._cutoff = None
        # Dictionary of children.
        self._children = {}

    def _gain_ratio(self, splits, y, H_y):
        """
        Return the gain ratio of some split of the examples.

        :param splits: Array of integers corresponding to which split each
          example is in.
        :param y: The labels of the examples.
        :param H_y: The entropy of the labels (so we don't recompute)
        """
        H_splits = entropy(splits)
        information_gain = mutual_info_fast(splits, y, H_splits, H_y)

        with np.errstate(divide='ignore', invalid='ignore'):
            gain_ratio = information_gain / H_splits

        # We treate 0/0 as 0 here
        if np.isnan(gain_ratio):
            gain_ratio = 0

        return gain_ratio

    def _find_cutoff(self, X, y, attr, H_y):
        """
        Return the cutoff, IG, and array for a continuous attribute.
        :param X: examples
        :param y: labels
        :param attr: which attribute
        :param H_y: entropy of y, to avoid recomputation
        :returns: A 3-tuple:
          [0]: cutoff
          [1]: IG of the split
          [2]: split array
        """
        # Get coordinately sorted copies of X and y.
        argsort = X[:, attr].argsort()
        Xs = X[argsort]
        ys = y[argsort]

        # Create a y array shifted one index up
        yshift = shift(ys, 1, cval=np.NaN)

        # Iterate over every index where the label isn't the same as the
        # previous one.
        max_ig = -1
        changes = np.where(ys != yshift)[0]
        for index in changes:
            cutoff = Xs[index][attr]
            split =X[:, attr] < cutoff
            ig = self._gain_ratio(split, y, H_y)
            if ig > max_ig:
                max_ig = ig
                max_cutoff = cutoff
                max_split = split
        return max_cutoff, max_ig, max_split


    def _max_gain_ratio_split(self, X, y):
        """
        Return the attribute and split that maximizes the gain ratio.

        :param X: examples
        :param y: labels
        :returns: a 3-tuple:
          [0]: IG of the split
          [1]: index of the attribute used to split
          [2]: array containing categories for each example
        """
        # Initialize our "used" array (if it wasn't already).  This is so we
        # don't reuse nominal attributes.
        if self._used is None:
            self._used = np.zeros(X.shape[1], dtype=np.bool)

        # Initial values for our return value.
        max_ig = -1
        cutoff = None

        # Precompute the entropy of the examples, since we'll reuse it.  Also,
        # I'm changing the label -1 to 0, mostly so that my entropy
        # implementation works properly.
        y[y == -1] = 0
        H_y = entropy(y)

        # Iterate over each attribute, keeping track of the best one.
        for attr in range(X.shape[1]):
            # Skip attributes we've already used.
            if self._used[attr]:
                continue

            if self._schema.is_nominal(attr):
                # For nominal attributes, just use the attribute itself as the
                # split.
                split = X[:, attr]
                ig = self._gain_ratio(split.astype(int), y, H_y)
            else:
                cutoff, ig, split = self._find_cutoff(X, y, attr, H_y)

            # Hold onto it if it's the best so far.
            if ig > max_ig:
                max_ig = ig
                max_idx = attr
                max_split = split
                max_cutoff = cutoff

        if self._schema.is_nominal(max_idx):
            # If we chose a nominal attribute, we can't use it again.  We could
            # reuse a continuous attribute.
            self._used[max_idx] = True
        else:
            self._cutoff = max_cutoff

        return max_ig, max_idx, max_split

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        #from smbio.util.repl import repl; repl()

        # If we have a pure tree, we're done.
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            self._label = unique_labels[0]
            return 1

        # If we are at the maximum depth, or if we've used all of our
        # attributes, choose the majority class.
        if self._allowed_depth == 1 or np.all(self._used):
            self._label = mode(unique_labels).mode[0]
            return 1

        # Otherwise, choose the attribute to split that maximizes the gain
        # ratio.
        ig, self._attribute, split = self._max_gain_ratio_split(X, y)

        # Fit the children!
        for value in np.unique(split):
            # Choose the examples that fit this branch.
            new_X = X[split == value]
            new_y = y[split == value]
            # Create a new child with adjusted parameters.
            child = DecisionTree(depth=self._allowed_depth-1,
                                 schema=self._schema,
                                 used=np.copy(self._used))
            d = child.fit(new_X, new_y, sample_weight=sample_weight)
            if d + 1 > self._depth:
                self._depth = d + 1
            self._size += child._size
            self._children[value] = child
        return self._depth

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        rv = np.array([self.predict_one(x) for x in X])
        rv[rv == 0] = -1
        return rv

    def predict_one(self, x):
        """Return the prediction for a single example."""
        # If this node has a label, return it.
        if self._label is not None:
            return self._label

        # Otherwise, get the corresponding child node for this example.
        if self._schema.is_nominal(self._attribute):
            subtree = self._children.get(x[self._attribute])
        else:
            subtree = self._children.get(int(x[self._attribute]<self._cutoff))

        # If there is no subtree, the algorithm must not have had an example
        # like this in training.  For now, predict negative.
        if subtree is None:
            log.error('No branch available. (x[%d]=%d), children=%r',
                      self._attribute, x[self._attribute],
                      self._children.keys())
            return 0

        # Return the subtree's prediction on this example.
        return subtree.predict_one(x)

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        return self._size

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        return self._depth

    def __str_recurse(self, depth, cond):
        """
        Return a string representation of this tree, for printouts.
        This is a recursive helper function for __str__.

        :param depth: How many levels deep is this node?
        :param cond: What is the string condition to go to this node?
        """
        if self._label is None:
            # For an internal node, show the condition, followed by the next
            # attribute you will test.
            s = ('--'*depth) + cond + ' => test x[%r]' % self._attribute + '\n'
            # Then print the children.
            for k,v in self._children.iteritems():
                s += v.__str_recurse(depth+1, 'x[%r]==%r' % (self._attribute, k))
        else:
            # For a leaf node, show the condition, followed by the label you
            # predict.
            s = ('--'*depth) + cond + ' => label %r'% self._label + '\n'
        return s

    def __str__(self):
        """Return a string representation of this tree."""
        return self.__str_recurse(0, 'root')

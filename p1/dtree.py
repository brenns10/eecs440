"""
The Decision Tree Classifier
"""
import numpy as np
from scipy.stats import mode


def entropy(l):
    """
    Return the entropy of any vector of discrete values.

    Shannon Entropy is a measure of the "information content" of a random
    variable.  The more widely dispersed the possible values of the RV, the
    higher its entropy.  Entropy is measured in bits (as in, the theoretical
    minimum amount of bits it take to represent the RV).  A RV which was 1 with
    probability 1 would have entropy of 0.  A RV which took on two values with
    equal probability would have entropy of 1 bit, etc.  The entropy function
    is denoted by H(X), and the definition is as follows:

        :math:`H(X) = - \sum_{x\in X} p(X=x) \log_2(p(X=x))`

    :param l: Array of integers/bools.
    :type l: numpy.array or similar
    :returns: The entropy of the array.
    """

    probabilities = np.bincount(l) / len(l)
    with np.errstate(divide='ignore'):  # ignore log(0) errors, we'll handle
        log_probabilities = np.log2(probabilities)
        log_probabilities[~np.isfinite(log_probabilities)] = 0
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

    def __init__(self, depth=None, schema=None):
        """
        Constructs a Decision Tree Classifier

        @param depth=None : maximum depth of the tree,
                            or None for no maximum depth
        """
        if schema is None:
            raise ValueError('Must provide schema to DecisionTree!')
        self._schema = schema
        self._allowed_depth = depth
        self._label = None
        self._attribute = None
        self._children = {}

    def _gain_ratio(self, splits, y):
        """Return the gain ratio of some split of the examples."""
        H_splits = entropy(splits)
        H_y = entropy(y)
        information_gain = mutual_info_fast(splits, y, H_splits, H_y)
        return float(information_gain) / H_splits

    def _max_gain_ratio_split(self, X, y):
        max_ig = 0
        max_idx = -1
        max_split = None
        for attr in X.shape[1]:
            if self._schema.is_nominal(attr):
                # For nominal attributes, just use the attribute itself as the
                # split.
                split = X[:, attr]
            else:
                pass  # TODO: implement for continuous!

            ig = self._gain_ratio(split, y)
            if ig > max_ig:
                max_ig = ig
                max_idx = attr
        return max_ig, max_idx, max_split

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        # If we have a pure tree, we're done.
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            self._label = unique_labels[0]
            return 1

        # If we are at the maximum depth, choose the majority class.
        if self._allowed_depth == 1:
            self._label = mode(unique_labels).mode[0]
            return 1

        # Otherwise, choose the attribute to split that maximizes the gain
        # ratio.
        ig, attr, split = self._max_gain_ratio_split(X, y)
        new_X = np.delete(X, attr, axis=1)

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        if self._label is not None:
            return self._label

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        pass

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        pass

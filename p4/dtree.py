"""
The Decision Tree Classifier
"""
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.stats import mode
import logging
log = logging.getLogger('dtree')
log.addHandler(logging.StreamHandler())
#log.setLevel(logging.DEBUG)


def entropy(l, weights):
    """
    Return the entropy of a vector and corresponding weights.
    """
    max_val = np.max(l)
    probs = np.zeros(l.shape)
    for i in range(max_val + 1):
        probs[i] = weights[l == i].sum()
    probs = probs[probs != 0]
    probs = probs / probs.sum()
    log_probs = np.log2(probs)
    # with np.errstate(divide='ignore'):  # ignore log(0) errors, we'll handle
    #     log_probs = np.log2(probs)
    #     log_probs[~np.isfinite(log_probs)] = 0
    #     log_probs[np.isnan(log_probs)] = 0
    return -np.sum(probs * log_probs)


def joint_dataset(l1, l2):
    """
    Create a joint dataset for two 1D arrays of non-negative integers
    """
    N = np.max(l1) + 1
    return l2 * N + l1


def mutual_info(l1, l2, weights):
    """
    Return the mutual information of 2 arrays of any type.
    """
    return (
        entropy(l1, weights) +
        entropy(l2, weights) -
        entropy(joint_dataset(l1, l2), weights)
    )


def mutual_info_fast(l1, l2, weights, l1_entropy, l2_entropy):
    """
    Compute mutual info without recomputing the entropy of l1 and l2.
    """
    return l1_entropy + l2_entropy - entropy(joint_dataset(l1, l2), weights)


def safelog(n):
    """For single numbers, return log(n) unless n=0, then return 0."""
    if n <= 0:
        return 0
    else:
        return np.log2(n)


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

    def _gain_ratio(self, splits, y, H_y, weights):
        """
        Return the gain ratio of some split of the examples.

        :param splits: Array of integers corresponding to which split each
          example is in.
        :param y: The labels of the examples.
        :param H_y: The entropy of the labels (so we don't recompute)
        :param weights: sample weights
        """
        H_splits = entropy(splits, weights)
        information_gain = mutual_info_fast(splits, y, weights, H_splits, H_y)

        with np.errstate(divide='ignore', invalid='ignore'):
            gain_ratio = information_gain / H_splits

        # We treate 0/0 as 0 here
        if np.isnan(gain_ratio):
            gain_ratio = 0

        return gain_ratio

    def _find_cutoff(self, X, y, attr, H_y, weights):
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
        xs = X[argsort, attr]
        ys = y[argsort]
        ws = weights[argsort]

        wsum = ws.sum()
        max_ig = -1
        max_idx = -1
        max_cutoff = -1

        last_label = ys[0]
        left = 0
        right = ws.sum()
        left_pos = 0
        left_neg = 0
        right_pos = ws[ys == 1].sum()
        right_neg = ws[ys != 1].sum()
        for i in range(len(ys)):
            # Update the max information gain.
            if ys[i] != last_label:
                ig = H_y
                ig -= left/wsum * safelog(left/wsum)
                ig -= right/wsum * safelog(right/wsum)
                ig += left_pos/wsum * safelog(left_pos/wsum)
                ig += left_neg/wsum * safelog(left_neg/wsum)
                ig += right_pos/wsum * safelog(right_pos/wsum)
                ig += right_neg/wsum * safelog(right_neg/wsum)
                if ig > max_ig:
                    max_ig = ig
                    max_idx = i
                    max_cutoff = xs[i]

            # And then do the bookkeeping for moving this example to the left.
            left += ws[i]
            right -= ws[i]
            if ys[i] == 1:
                left_pos += ws[i]
                right_pos -= ws[i]
            else:
                left_neg += ws[i]
                right_neg -= ws[i]

        # max_cutoff = X[argsort[max_idx], attr]
        max_split = X[:, attr] < max_cutoff
        ig = mutual_info(y, max_split, weights)
        return max_cutoff, ig, max_split


    def _max_gain_ratio_split(self, X, y, weights):
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
        H_y = entropy(y, weights)

        # Iterate over each attribute, keeping track of the best one.
        for attr in range(X.shape[1]):
            # Skip attributes we've already used.
            if self._used[attr]:
                continue

            if self._schema.is_nominal(attr):
                # For nominal attributes, just use the attribute itself as the
                # split.
                split = X[:, attr]
                ig = self._gain_ratio(split.astype(int), y, H_y, weights)
                log.debug('Consider %d (nominal), ig=%r', attr, ig)
            else:
                cutoff, ig, split = self._find_cutoff(X, y, attr, H_y, weights)
                log.debug('Consider %r (continueous), cutoff=%r, ig=%r', attr, cutoff, ig)

            # Hold onto it if it's thmax()e best so far.
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
        if sample_weight is None:
            sample_weight = np.ones(y.shape)
        log.debug('Starting fit()')
        # If we have a pure tree, we're done.
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            self._label = unique_labels[0]
            log.debug('Found pure node.  Choosing pure label %d', self._label)
            return 1

        # If we are at the maximum depth, or if we've used all of our
        # attributes, choose the majority class.
        if self._allowed_depth == 1 or np.all(self._used):
            self._label = mode(y).mode[0]
            log.debug('Maximum depth.  Choosing max label %d', self._label)
            return 1

        # Otherwise, choose the attribute to split that maximizes the gain
        # ratio.
        ig, self._attribute, split = self._max_gain_ratio_split(X, y, sample_weight)
        log.debug('Choose attribute %r', self._attribute)
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
        rv = self.predict_proba(X)
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
        return np.array([self.predict_one(x) for x in X])

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
            attr = self._schema.feature_names[self._attribute]
            s = ('--'*depth) + cond + ' => test %s' % attr + '\n'
            # Then print the children.
            for k, v in self._children.iteritems():
                if self._schema.is_nominal(self._attribute):
                    newcond = attr + '=' + str(k)
                elif k:
                    newcond = attr + '<' + str(k)
                else:
                    newcond = attr + '>=' + str(k)
                s += v.__str_recurse(depth+1, newcond)
        else:
            # For a leaf node, show the condition, followed by the label you
            # predict.
            s = ('--'*depth) + cond + ' => label %r'% self._label + '\n'
        return s

    def __str__(self):
        """Return a string representation of this tree."""
        return self.__str_recurse(0, 'root')

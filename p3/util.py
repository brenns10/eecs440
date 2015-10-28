"""
Functions that are applicable to more than just one type of classifier.
"""

from __future__ import print_function, division

import time
import logging

import numpy as np

# This controls the verbosity of output from internal cross validation.  Set it
# to DEBUG to see what the values for each particular statistic were for each
# parameter value.  Set it to INFO to see just what parameter value was chosen.
# Comment out the "log.setLevel" line completely to silence all logging from
# this file.
log = logging.getLogger('util')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


def internal_cross_validation(cls, kwargs, paramname, paramrange, statistic,
                              X, y):
    """
    Performs internal cross validation, returns the best parameter value.

    Varies a parameter over a range and returns the one that maximizes a
    statistic in 5-fold cross validation.

    :param cls: The classifier class to use for cross validation.
    :param kwargs: The options (other than the parameter we're varying) to use
      for the classifier's constructor.
    :param paramname: The name of the constructor parameter we'll be varying.
    :param paramrange: The range of values we will vary it through.
    :param statistic: Name of the statistic to use for decision.
    :param X: examples-by-features NumPy matrix
    :param y: vector of class labels
    """

    # Delay these imports so that we don't have circular imports!
    from main import get_folds
    from stats import StatisticsManager

    # Much of this code is sourced from main.py's template.  It simply creates
    # a StatisticsManager for each parameter value.  It does the cross
    # validation on the same folds and picks the best value of the parameter.
    stats_managers = [StatisticsManager() for _ in paramrange]
    folds = get_folds(X, y, 5)
    for train_X, train_y, test_X, test_y in folds:
        for value, stats_manager in zip(paramrange, stats_managers):
            kwargs[paramname] = value
            train_time = time.time()
            classifier = cls(**kwargs)
            classifier.fit(train_X, train_y)
            train_time = train_time - time.time()
            predictions = classifier.predict(test_X)
            scores = classifier.predict_proba(test_X)
            stats_manager.add_fold(test_y, predictions, scores, train_time)
        log.debug('internal-cv: fold completed')

    # Get values for our statistic of interest.
    stat_values = []
    for i, mgr in enumerate(stats_managers):
        # pooled might as well be True, since we don't want a std
        stat = mgr.get_statistic(statistic, pooled=True)
        stat_values.append(stat)
        log.debug('internal-cv gets %s=%r for param %s=%r' %
                  (statistic, stat, paramname, paramrange[i]))
    log.debug('internal-cv gets argmax=%d' % np.argmax(stat_values))
    # Get the parameter value that maximizes our statistic.
    selection = paramrange[np.argmax(stat_values)]
    log.info('internal-cv selects %s=%r' % (paramname, selection))
    return selection

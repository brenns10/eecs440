"""
Statistics Computations
"""

import numpy as np
import scipy


class StatisticsManager(object):

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.prediction_scores = []
        self.training_times = []
        self.statistics = {
            'accuracy' : (accuracy,  self.predicted_labels),
            'precision': (precision, self.predicted_labels),
            'recall'   : (recall,    self.predicted_labels),
            'auc'      : (auc,       self.prediction_scores),
        }

    def add_fold(self, true_labels, predicted_labels,
                 prediction_scores, training_time):
        """
        Add a fold of labels and predictions for later statistics computations

        @param true_labels : the actual labels
        @param predicted_labels : the predicted binary labels
        @param prediction_scores : the real-valued confidence values
        @param training_time : how long it took to train on the fold
        """
        self.true_labels.append(true_labels)
        self.predicted_labels.append(predicted_labels)
        self.prediction_scores.append(prediction_scores)
        self.training_times.append(training_time)

    def get_statistic(self, statistic_name, pooled=True):
        """
        Get a statistic by name, either pooled across folds or not

        @param statistic_name : one of {accuracy, precision, recall, auc}
        @param pooled=True : whether or not to "pool" predictions across folds
        @return statistic if pooled, or (avg, std) of statistic across folds
        """
        if statistic_name not in self.statistics:
            raise ValueError('"%s" not implemented' % statistic_name)

        statistic, predictions = self.statistics[statistic_name]

        if pooled:
            predictions = np.hstack(map(np.asarray, predictions))
            labels = np.hstack(map(np.asarray, self.true_labels))
            return statistic(labels, predictions)
        else:
            stats = []
            for l, p in zip(self.true_labels, predictions):
                stats.append(statistic(l, p))
            return np.average(stats), np.std(stats)

def accuracy(labels, predictions):
    """
    What fraction of the predictions are the same as the labels?
    """
    return sum(labels == predictions) / len(labels)


def precision(labels, predictions):
    """
    What fraction of the examples predicted positive are actually positive?
    """
    pos_pred = (predictions == 1)
    if pos_pred.sum() == 0:
        return 1.0
    #                  True              Positives / All positive predictions
    return ((labels == predictions) & pos_pred).sum() / pos_pred.sum()


def recall(labels, predictions):
    """
    What fraction of the positive examples were predicted positive?
    """
    pos_label = (labels == 1)
    if pos_label.sum() == 0:
        return 1.0
    #                 True               Positives  / All positive labels
    return ((labels == predictions) & pos_label).sum() / pos_label.sum()


def specificity(labels, predictions):
    """
    What fraction of the negative examples were predicted negative?
    """
    neg_label = (labels == -1)
    if neg_label.sum() == 0:
        return 1.0
    #                 True               Negatives  / All negative labels
    return ((labels == predictions) & neg_label).sum() / neg_label.sum()


def auc(labels, predictions):
    cutoffs = np.sort(np.unique(np.append(predictions, [0, 1])), )
    auc = 0
    prev_tpr = 1
    prev_fpr = 1
    tprs = []
    fprs = []
    for x in cutoffs:
        pred = np.where(predictions > x, 1, -1)
        tpr = recall(labels, pred)
        fpr = 1 - specificity(labels, pred)
        auc += (prev_fpr - fpr) * (tpr + prev_tpr) * 0.5
        prev_tpr = tpr
        prev_fpr = fpr
        tprs.append(tpr)
        fprs.append(fpr)
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(fprs, tprs)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    fig.savefig('roc.pdf')
    return auc

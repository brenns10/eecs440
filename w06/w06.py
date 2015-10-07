"""
Code to plot decision boundary of ANN.

Stephen Brennan, EECS 440 Written 6, 10/06/2015.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def sigmoid(X):
    return (1 + np.exp(-X)) ** -1


def output(X, weights, out_weights):
    hidden_outs = sigmoid(np.dot(weights, X.T))
    return sigmoid(np.dot(hidden_outs.T, out_weights))


def plot_decision_boundary(wmin, wmax):
    hidden_weights = np.random.uniform(wmin, wmax, (2, 2))
    output_weights = np.random.uniform(wmin, wmax, 2)

    x1 = np.arange(-5.0, 5.1, step=0.1)
    x2 = np.arange(-5.0, 5.1, step=0.1)
    X = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
    y = output(X, hidden_weights, output_weights)

    fig, ax = plt.subplots()
    pos_idx = y > 0.5
    pos = X[pos_idx]
    neg = X[~pos_idx]
    print(y)
    print('%r positives, %r negatives' % (len(pos), len(neg)))
    print(pos)
    print(neg)
    ax.scatter(*pos.T, color='b')
    ax.scatter(*neg.T, color='r')
    return ax


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Need weight bounds.')
    else:
        bound = float(sys.argv[1])
        ax = plot_decision_boundary(-bound, bound)
        ax.figure.savefig(sys.argv[1] + '.pdf')

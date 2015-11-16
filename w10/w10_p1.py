"""
Code for EECS 440 W10 Problem 1, by Stephen Brennan (smb196).
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
plt.use('ggplot')


C = 0
Y = 1


def generate_box(p_Y, boxsize=100):
    sample = np.random.random(size=boxsize)
    box = np.zeros(boxsize, dtype=np.uint)
    box[:] = C
    box[sample < p_Y] = Y
    return box


def conditional(box, hypothesis):
    return binom.pmf(np.cumsum(box), np.arange(1, len(box)+1), hypothesis)


def get_probabilities(p_Y, hypotheses, priors=None):
    if priors is None:
        priors = [1/len(hypotheses)] * len(hypotheses)
    box = generate_box(p_Y)
    conditionals = [conditional(box, h) for h in hypotheses]
    conditional_matrix = np.concatenate([c.reshape((1, len(c))) for c in conditionals])
    sums = conditional_matrix.sum(0)
    return [p * c / sums for p, c in zip(priors, conditionals)]


def plot_probabilities(probabilities):
    fig, ax = plt.subplots()
    for prob_list in probabilities:
        ax.plot(prob_list)
    return fig

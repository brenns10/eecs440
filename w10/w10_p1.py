"""
Code for EECS 440 W10 Problem 1, by Stephen Brennan (smb196).

Run this on Python 3 with Numpy, SciPy, and Matplotlib installed.  Examples:

# To generate graphs for part 1 and 2:
$ python3 w10_p1.py 1
# Creates graphs: p1_part[12]_0.(75|50|25).pdf

# To generate graphs for part 3 (that is, with adjusted priors):
$ python3 w10_p2.py 3
# Creates garphs: p1_part[34]_0.(75|50|25).pdf
# You can ignore the part 4 graphs.  They're just part 2 graphs applied to this
# situation.
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
plt.style.use('ggplot')


C = 0
Y = 1


def generate_box(p_Y, boxsize=100):
    """
    Generate a box of candies with a given size.

    The probability of a yummy candy is given by p_Y.
    """
    sample = np.random.random(size=boxsize)
    box = np.zeros(boxsize, dtype=np.uint)
    box[:] = C
    box[sample < p_Y] = Y
    return box


def conditional(box, hypothesis):
    """
    Return probability of candy observations given a hypothesis. P[c|T]

    I've assumed that this is binomial for most simplicity.  The parameter box
    should be an ordered list of candies in the box, and the hypothesis should
    be a probability of yummy candies.  This will return a list of conditional
    probabilities:

    return value[i] = Pr[c_1, ..., c_i | hypothesis]
    """
    return binom.pmf(np.cumsum(box), np.arange(1, len(box)+1), hypothesis)


def get_probabilities(p_Y, hypotheses, priors=None):
    """
    Return probabilities of each hypothesis for each candy.

    This function generates a box from the true distribution, and then computes
    Pr[T=h|c_1, ..., c_i], for each combination of hypothesis h and number of
    candies i.  There are two return values:

    - Pr[T=h|...] a list of numpy lists.  Each numpy list corresponds to a
      hypothesis.
    - Pr[next is crummy|...]: just a numpy list
    """
    if priors is None:
        priors = np.array([1/len(hypotheses)] * len(hypotheses))
        priors = priors.reshape((len(hypotheses), 1))
    box = generate_box(p_Y)
    conditionals = [p*conditional(box, h) for h, p in zip(hypotheses, priors)]
    sums = sum(conditionals)
    final_conditionals = [c / sums for c in conditionals]
    p_crummy = sum((1-h) * p for h, p in zip(hypotheses, final_conditionals))
    return final_conditionals, p_crummy


def plot_probabilities(probabilities, crummies, p_Y, hypotheses):
    """
    Plot the probability lists from get_probabilities().
    """
    fig1, ax1 = plt.subplots()
    for prob_list, hypothesis in zip(probabilities, hypotheses):
        ax1.plot(prob_list, label='p(Y)=%0.2f' % hypothesis)
    ax1.set_title('Probability of Box Type After i Candies (p(Y)=%0.2f)' % p_Y)
    ax1.set_xlabel('i')
    ax1.set_ylabel('Probability')
    ax1.legend()
    fig2, ax2 = plt.subplots()
    ax2.plot(crummies)
    ax2.set_title('Probability of Crummy After i Candies (p(Y)=%0.2f)' % p_Y)
    ax2.set_xlabel('i')
    ax2.set_ylabel('Probability of Crummy')
    return fig1, fig2


def part(n):
    if n > 1:
        priors = [0.8, 0.1, 0.1]
    else:
        priors = None
    hypotheses = [0.75, 0.50, 0.25]
    for h in hypotheses:
        p, c = get_probabilities(h, hypotheses, priors)
        fig1, fig2 = plot_probabilities(p, c, h, hypotheses)
        fig1.savefig('p1_part%d_%0.2f.pdf' % (n, h))
        fig2.savefig('p1_part%d_%0.2f.pdf' % (n+1, h))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Must provide part number.')
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except:
        print('Provide a number, please.')
        sys.exit(2)
    part(n)

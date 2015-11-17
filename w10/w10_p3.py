"""
Code for EECS W10 Problem 3, by Stephen Brennan (smb196).
"""

import numpy as np
import scipy.optimize


def f(x, y, z=1):
    return 1 - x**2 - y**2


def minbias():
    samples = 10
    xs = np.random.uniform(-1, 1, (samples, 1))
    ys = np.random.uniform(-1, 1, (samples, 1))
    zs = np.ones((samples, 1))
    fs = f(xs, ys)

    arr = np.concatenate([xs, ys, zs], axis=1)
    abc, *etc = np.linalg.lstsq(arr, fs)

    a = abc[0]
    b = abc[1]
    c = abc[2]  # for some reason I can't just unpack

    def h(x, y, z=1):
        return a*x + b*x + c

    def variance_error(xvec):
        x = xvec[0]
        y = xvec[1]
        return (h(x, y) - c) ** 2

    def grad_variance_error(xvec):
        x = xvec[0]
        y = xvec[1]
        shared = 2 * (a * x + b * y)
        return np.array([shared * a, shared * b])

    def bias_error(xvec):
        x = xvec[0]
        y = xvec[1]
        return (c - f(x, y)) ** 2

    def grad_bias_error(xvec):
        x = xvec[0]
        y = xvec[1]
        shared = 2 * (a * x + b * y + c + x**2 + y**2 - 1)
        del_x = shared * (2 * x)
        del_y = shared * (2 * y)
        return np.array([del_x, del_y])

    xy_maxbias = scipy.optimize.minimize(lambda x: -bias_error(x),
                                         np.array([0, 0]),
                                         jac=lambda x: -grad_bias_error(x),
                                         bounds=[(-1, 1), (-1, 1)])
    xy_maxvar = scipy.optimize.minimize(lambda x: -variance_error(x),
                                        np.array([0, 0]),
                                        jac=lambda x: -grad_variance_error(x),
                                        bounds=[(-1, 1), (-1, 1)])
    # print(xy_minbias)
    return xy_maxbias.x, xy_maxvar.x


def main():
    niters = 1000
    maxbias = np.zeros((niters, 2))
    maxvar = np.zeros((niters, 2))
    for i in range(niters):
        maxbias[i], maxvar[i] = minbias()

    print('maximize bias:')
    print('means: ', np.mean(maxbias, axis=0))
    print('stds:  ', np.std(maxbias, axis=0))

    print('maximize variance:')
    print('means: ', np.mean(maxvar, axis=0))
    print('stds:  ', np.std(maxvar, axis=0))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 06 11:44:28 2014

@author: Konstantin
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    mu = np.array([1, 10, 20])
    sigma = np.matrix([[20, 10, 10],
                       [10, 25, 1],
                       [10, 1, 50]])
    np.random.seed(100)
    data = np.random.multivariate_normal(mu, sigma, 1000)
    print(data.shape)
    values = data.T
    print(values.shape)

    kde = stats.gaussian_kde(values)
    density = kde(values)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x, y, z = values
    #print(x.shape)
    #print(y.shape)
    #print(z.shape)
    ax.scatter(x, y, z, c=density)
    plt.show()


if __name__ == '__main__':
    main()

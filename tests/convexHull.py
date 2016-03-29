# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:29:10 2014

@author: konkle
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Delaunay


def plot_in_hull(p, cloud):
    """
    plot relative to `in_hull` for 2d data
    """
    hull = Delaunay(cloud)

    # plot triangulation
    poly = PolyCollection(hull.points[hull.vertices], facecolors='grey',
                          edgecolors='grey', alpha=0.1)
    plt.clf()
    plt.title('in hull: green, out of hull: red')
    plt.gca().add_collection(poly)
    plt.plot(hull.points[:, 0], hull.points[:, 1], 'o', hold=1, color='grey',
             alpha=0.2)

    # plot tested points `p` - green are inside hull, red outside
    inside = hull.find_simplex(p) >= 0
    plt.plot(p[inside, 0], p[inside, 1], 'og')
    plt.plot(p[~inside, 0], p[~inside, 1], 'or')
    plt.show()


cloud = np.random.rand(100, 2)
test = np.random.rand(70, 2)*1.5-0.25

plot_in_hull(test, cloud)

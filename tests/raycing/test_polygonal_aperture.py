# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "18 Jun 2022"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
import matplotlib.patches as patches


def main(dx, dz, px, pz, nx, nz):
    """dx, dz are opening sizes, px, pz are pitch steps (periods),
    nx and nz are counted from the central hole in -ve and +ve directions,
    so (2*nx+1)*(2*nz+1) rectangular holes in total."""

    # 4 corners + the 1st corner to close the path + nan to disconnect patches
    cellx = np.array([dx, -dx, -dx, dx, dx, np.nan])*0.5
    cellz = np.array([dz, dz, -dz, -dz, dz, np.nan])*0.5

    xc = np.linspace(-1, 1, 2*nx+1) * px * nx
    zc = np.linspace(-1, 1, 2*nz+1) * pz * nz
    xm, zm = np.meshgrid(xc, zc)
    xi = (xm.ravel(order='F') + cellx[:, np.newaxis]).ravel(order='F')
    zi = (zm.ravel(order='F') + cellz[:, np.newaxis]).ravel(order='F')
    vertices = np.column_stack((xi, zi))

    ntest = 250
    xmax, zmax = (px*nx+dx*0.75), (pz*nz+dz*0.75)
    xtest = np.linspace(-xmax, xmax, ntest)
    ztest = np.linspace(-zmax, zmax, ntest)
    xtestm, ztestm = np.meshgrid(xtest, ztest)
    xtestm = xtestm.ravel()
    ztestm = ztestm.ravel()

    footprint = mplPath(vertices)
    badIndices = np.invert(footprint.contains_points(np.array(
        list(zip(xtestm, ztestm)))))
    color = ['r' if ind else 'g' for ind in badIndices]

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    # patch = patches.PathPatch(footprint, facecolor='orange', lw=2)
    # ax.add_patch(patch)
    ax.scatter(xtestm, ztestm, s=2, c=color)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-zmax, zmax)
    plt.show()


if __name__ == '__main__':
    main(0.1, 0.1, 0.15, 0.15, 7, 7)

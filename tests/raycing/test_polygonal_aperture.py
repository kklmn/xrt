# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "20 Jun 2022"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
import matplotlib.patches as patches

import sys
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.apertures as rapts

ap = rapts.GridAperture(dx=0.1, dz=0.1, px=0.15, pz=0.15, nx=7, nz=7)
# ap = rapts.SiemensStar(nSpokes=24, r=0.1)
# ap = rapts.SiemensStar(nSpokes=9, rx=0.1, rz=0.08, vortex=2, vortexNradial=9)


def main():
    vertices = ap.opening

    ntest = 75
    xtest = np.linspace(*ap.limOptX, ntest)
    ztest = np.linspace(*ap.limOptY, ntest)
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
    patch = patches.PathPatch(footprint, facecolor='orange', lw=2)
    ax.add_patch(patch)
    ax.scatter(xtestm, ztestm, s=2, c=color)
    ax.set_xlim(*ap.limOptX)
    ax.set_ylim(*ap.limOptY)
    plt.show()


if __name__ == '__main__':
    main()

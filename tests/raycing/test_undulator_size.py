# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 19:59:40 2018

@author: konst
"""
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt

import xrt.backends.raycing.sources as rs


def main():
    """Compare with
    Harry Westfahl Jr et al. J. Synchrotron Rad. (2017). 24, 566–575.
    In particular notice their non-equidistant (in energy) harmonics."""
    und = rs.Undulator(
        name='U19', eE=3.0, eI=0.35,  # Sirius
        eEpsilonX=0.245, eEpsilonZ=0.0024, eEspread=1e-3,
        betaX=1.5, betaZ=1.5,
        period=19, n=105,
        targetE=(10000, 7),
        targetOpenCL=None)

    E = np.linspace(1500., 20000., 851)
    sx, sz = und.get_SIGMA(E)
    sPx, sPz = und.get_SIGMAP(E)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title("{0} undulator: linear source size".format(und.name))
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'rms linear source size (µm)')
    ax.plot(E*1e-3, sx*1e3, '-', label=r'$\sigma_x$')
    ax.plot(E*1e-3, sz*1e3, '--', label=r'$\sigma_y$')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 25)
    ax.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax.grid()
    ax.legend(loc='lower left')

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.set_title("{0} undulator: angular source size".format(und.name))
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'rms angular source size (µrad)')
    ax.plot(E*1e-3, sPx*1e6, '-', label=r"$\sigma'_x$")
    ax.plot(E*1e-3, sPz*1e6, '--', label=r"$\sigma'_y$")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 25)
    ax.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax.grid()
    ax.legend(loc='lower left')

    plt.show()


if __name__ == '__main__':
    main()
    print("Done")

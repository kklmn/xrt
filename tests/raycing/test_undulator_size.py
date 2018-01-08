# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 19:59:40 2018

@author: konst
"""
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

import xrt.backends.raycing.sources as rs
#withXrtSampling = True

thetaMax, psiMax = 60e-6, 30e-6  # rad


class TwoLineObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,x0+width], [0.7*height,0.7*height],
                        linestyle=orig_handle[0].get_linestyle(),
                        color=orig_handle[0].get_color())
        l2 = plt.Line2D([x0,x0+width], [0.3*height,0.3*height], 
                        linestyle=orig_handle[1].get_linestyle(),
                        color=orig_handle[1].get_color())
        return [l1, l2]


class TwoScatterObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0+0.2*width], [0.5*height], marker='o', 
                        color='C0', markerfacecolor='none')
        l2 = plt.Line2D([x0+0.8*width], [0.5*height], marker='o', 
                        color='C1', markerfacecolor='none')
        return [l1, l2]


def main():
    und = rs.Undulator(
        name='MAX IV U19', eE=3.0, eI=0.5,
        eEpsilonX=0.263, eEpsilonZ=0.008,
        betaX=9., betaZ=2.,

#        """Compare with
#        Harry Westfahl Jr et al. J. Synchrotron Rad. (2017). 24, 566–575.
#        But notice their non-equidistant (in energy) harmonics."""
#        name='Sirius U19', eE=3.0, eI=0.35,
#        eEpsilonX=0.245, eEpsilonZ=0.0024,
#        betaX=1.5, betaZ=1.5,

        period=19, n=105,
        targetE=(10000, 7),
        xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3,
        xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
        targetOpenCL='CPU',
        precisionOpenCL='float32')

    E = np.linspace(1400., 20000., 1860+1)
#    und.eEspread = 0
    sx0, sz0 = und.get_SIGMA(E)
    sPx0, sPz0 = und.get_SIGMAP(E)
    und.eEspread = 1e-3
    sx, sz = und.get_SIGMA(E)
    sPx, sPz = und.get_SIGMAP(E)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.set_title("{0} undulator: linear source size".format(und.name))
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'rms linear source size (µm)')
    l1, = ax.plot(E*1e-3, sx0*1e3, '--C0')
    l2, = ax.plot(E*1e-3, sz0*1e3, '--C1')
    l3, = ax.plot(E*1e-3, sx*1e3, '-C0')
    l4, = ax.plot(E*1e-3, sz*1e3, '-C1')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, None)
    ax.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax.grid()
    leg1 = ax.legend([(l1, l3), (l2, l4)], [r"$\sigma_x$", r"$\sigma_y$"], 
                     handler_map={tuple: TwoLineObjectsHandler()},
                     loc='center left')
    leg2 = ax.legend([(l1, l2), (l3, l4)],
                     [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     title='at energy spread', loc='right')
    ax.add_artist(leg1)

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    ax.set_title("{0} undulator: angular source size".format(und.name))
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'rms angular source size (µrad)')
    ax.plot(E*1e-3, sPx0*1e6, '--C0')
    ax.plot(E*1e-3, sPz0*1e6, '--C1')
    ax.plot(E*1e-3, sPx*1e6, '-C0')
    ax.plot(E*1e-3, sPz*1e6, '-C1')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, None)
    ax.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax.grid()
    leg1 = ax.legend([(l1, l3), (l2, l4)], [r"$\sigma'_x$", r"$\sigma'_y$"], 
                     handler_map={tuple: TwoLineObjectsHandler()},
                     loc='upper center')
    leg2 = ax.legend([(l1, l2), (l3, l4)],
                     [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     title='at energy spread', loc='lower center')
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    if withXrtSampling:
        theta = np.linspace(-1, 1, 201) * thetaMax
        psi = np.linspace(-1, 1, 101) * psiMax
        Eh = (np.arange(1, 14, 2) * und.E1 + \
            20*np.arange(-5, 3.1)[:, np.newaxis])
        sh = Eh.shape
#        und.eEspread = 0
        I0 = und.intensities_on_mesh(Eh.ravel(), theta, psi)[0]
        flux = I0.sum(axis=(1, 2))
        fluxsh = flux.reshape(sh)
        fluxMax = fluxsh.max(axis=0)
        sigma2theta = \
            (I0*(theta**2)[np.newaxis, :, np.newaxis]).sum(axis=(1, 2)) / flux
        sigma2psi = \
            (I0*(psi**2)[np.newaxis, np.newaxis, :]).sum(axis=(1, 2)) / flux
        l5 = ax.scatter(Eh*1e-3, sigma2theta**0.5*1e6,
                         s=fluxsh/fluxMax[np.newaxis, :]*50,
                         facecolors='none', edgecolors='C0')
        l6 = ax.scatter(Eh*1e-3, sigma2psi**0.5*1e6, s=fluxsh/fluxMax*50,
                         facecolors='none', edgecolors='C1')
        leg3 = ax.legend([(l5, l6)],
                         [r'sampled by xrt'],
                         handler_map={tuple: TwoScatterObjectsHandler()},
                         loc='lower right')

    fig1.savefig("undulatorLinearSize.png")
    fig2.savefig("undulatorAngularSize.png")


if __name__ == '__main__':
    main()
    plt.show()
    print("Done")

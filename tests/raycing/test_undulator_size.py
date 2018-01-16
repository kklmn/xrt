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
withXrtSampling = True

thetaMax, psiMax = 60e-6, 30e-6  # rad


class TwoLineObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height],
                        linestyle=orig_handle[0].get_linestyle(),
                        color=orig_handle[0].get_color())
        l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],
                        linestyle=orig_handle[1].get_linestyle(),
                        color=orig_handle[1].get_color())
        return [l1, l2]


class TwoScatterObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0+0.2*width], [0.5*height],
                        marker=orig_handle[0].get_paths()[0],
                        color=orig_handle[0].get_edgecolors()[0],
                        markerfacecolor='none')
        l2 = plt.Line2D([x0+0.8*width], [0.5*height],
                        marker=orig_handle[1].get_paths()[0],
                        color=orig_handle[1].get_edgecolors()[0],
                        markerfacecolor='none')
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
        eEspread=1e-3,
        targetOpenCL='CPU',
        precisionOpenCL='float32')

    E = np.linspace(1400., 20000., 1860+1)
    sx0, sz0 = und.get_SIGMA(E, with0eSpread=True)
    sPx0, sPz0 = und.get_SIGMAP(E, with0eSpread=True)
    sx, sz = und.get_SIGMA(E)
    sPx, sPz = und.get_SIGMAP(E)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_title("{0} undulator: linear source size".format(und.name))
    ax1.set_xlabel(u'energy (keV)')
    ax1.set_ylabel(u'rms linear source size (µm)')
    l1, = ax1.plot(E*1e-3, sx0*1e3, '--C0')
    l2, = ax1.plot(E*1e-3, sz0*1e3, '--C1')
    l3, = ax1.plot(E*1e-3, sx*1e3, '-C0')
    l4, = ax1.plot(E*1e-3, sz*1e3, '-C1')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, None)
    ax1.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax1.grid()
    leg1 = ax1.legend([(l1, l3), (l2, l4)], [r"$\sigma_x$", r"$\sigma_y$"],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     loc='center left')
    leg2 = ax1.legend([(l1, l2), (l3, l4)],
                     [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     title='at energy spread', loc='right')
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.set_title("{0} undulator: angular source size".format(und.name))
    ax2.set_xlabel(u'energy (keV)')
    ax2.set_ylabel(u'rms angular source size (µrad)')
    ax2.plot(E*1e-3, sPx0*1e6, '--C0')
    ax2.plot(E*1e-3, sPz0*1e6, '--C1')
    ax2.plot(E*1e-3, sPx*1e6, '-C0')
    ax2.plot(E*1e-3, sPz*1e6, '-C1')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, None)
    ax2.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax2.grid()
    leg1 = ax2.legend([(l1, l3), (l2, l4)], [r"$\sigma'_x$", r"$\sigma'_y$"],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     loc='upper center')
    leg2 = ax2.legend([(l1, l2), (l3, l4)],
                     [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                     handler_map={tuple: TwoLineObjectsHandler()},
                     title='at energy spread', loc='lower center')
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)

    if withXrtSampling:
        theta = np.linspace(-1, 1, 201) * thetaMax
        psi = np.linspace(-1, 1, 101) * psiMax
        Eh = (np.arange(1, 14, 2) * und.E1 +
              10*np.arange(-10, 7)[:, np.newaxis])
        sh = Eh.shape

        flux, sigma2theta, sigma2psi, dx2, dz2 = und.real_photon_source_sizes(
            Eh.ravel(), theta, psi)
        sigma2theta += und.dxprime**2
        sigma2psi += und.dzprime**2
        dx2 += und.dx**2
        dz2 += und.dz**2

        fluxsh = flux.reshape(sh)
        fluxMax = fluxsh.max(axis=0)

        l5 = ax2.scatter(Eh*1e-3, sigma2theta**0.5*1e6,
                         s=fluxsh/fluxMax[np.newaxis, :]*50,
                         facecolors='none', edgecolors='C0')
        l6 = ax2.scatter(Eh*1e-3, sigma2psi**0.5*1e6, s=fluxsh/fluxMax*50,
                         facecolors='none', edgecolors='C1')
        leg3 = ax2.legend([(l5, l6)],
                          [r'sampled by xrt'],
                          handler_map={tuple: TwoScatterObjectsHandler()},
                          loc='lower right')

        l7 = ax1.scatter(Eh*1e-3, dx2**0.5*1e3,
                         s=fluxsh/fluxMax[np.newaxis, :]*50,
                         facecolors='none', edgecolors='C0')
        l8 = ax1.scatter(Eh*1e-3, dz2**0.5*1e3, s=fluxsh/fluxMax*50,
                         facecolors='none', edgecolors='C1')
        leg4 = ax1.legend([(l7, l8)],
                          [r'sampled by xrt'],
                          handler_map={tuple: TwoScatterObjectsHandler()},
                          bbox_to_anchor=(1.0, 0.38))
        # inset2
        axS2 = fig2.add_axes([0.17, 0.16, 0.22, 0.3])
        axS2.plot(E*1e-3, sPx*1e6, '-C0')
        axS2.plot(E*1e-3, sPz*1e6, '-C1')
        axS2.scatter(Eh*1e-3, sigma2theta**0.5*1e6,
                     s=fluxsh/fluxMax[np.newaxis, :]*50,
                     facecolors='none', edgecolors='C0')
        axS2.scatter(Eh*1e-3, sigma2psi**0.5*1e6, s=fluxsh/fluxMax*50,
                     facecolors='none', edgecolors='C1')
        axS2.set_xlim(10-0.1, 10+0.06)
        axS2.set_ylim(8, 16)
        ax2.arrow(7, 6.5, 2.3, 2.8, head_width=0.3, head_length=0.75)

        # inset1
        axS1 = fig1.add_axes([0.37, 0.36, 0.22, 0.3])
        axS1.plot(E*1e-3, sz0*1e3, '--C1')
        axS1.plot(E*1e-3, sz*1e3, '-C1')
        axS1.scatter(Eh*1e-3, dx2**0.5*1e3,
                     s=fluxsh/fluxMax[np.newaxis, :]*50,
                     facecolors='none', edgecolors='C0')
        axS1.scatter(Eh*1e-3, dz2**0.5*1e3, s=fluxsh/fluxMax*50,
                     facecolors='none', edgecolors='C1')
        axS1.set_xlim(10-0.1, 10+0.06)
        axS1.set_ylim(3, 10)
        ax1.arrow(8.5, 16., 1.1, -7.5, head_width=0.3, head_length=2.)

    fig1.savefig("undulatorLinearSize.png")
    fig2.savefig("undulatorAngularSize.png")


if __name__ == '__main__':
    main()
    plt.show()
    print("Done")

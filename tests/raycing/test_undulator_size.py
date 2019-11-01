# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "1 Nov 2019"

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

import xrt.backends.raycing.sources as rs
from xrt.backends.raycing.physconsts import CH

withXrtSampling = True
withCoisson = True

thetaMax, psiMax = 60e-6, 30e-6  # rad

if withCoisson:
    coissonD = np.linspace(-0.4, 1.2, 17)
    coissonA = np.array([0.45, 0.47, 0.50, 0.54, 0.58, 0.63, 0.69, 0.75, 0.82,
                         0.90, 0.98, 1.06, 1.14, 1.22, 1.30, 1.37, 1.44])
    coissonS = np.array([4.51, 3.82, 3.31, 2.96, 2.67, 2.46, 2.29, 2.15, 2.04,
                         1.95, 1.88, 1.84, 1.82, 1.82, 1.87, 1.97, 2.16])

    def coisson(Ec, L, N, n):
        E = Ec * (1 - coissonD/N/n)  # in eV
        lambdaC = CH / Ec * 1e-7  # in mm
        div2 = coissonA**2
        div2 *= (lambdaC / L)
        lin2 = coissonS**2
        lin2 *= (lambdaC * L) / (4*np.pi)**2
        return E, div2, lin2


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

    print(u"Electron beam linear sizes = {0:.3f} µm × {1:.3f} µm".format(
        und.dx*1e3, und.dz*1e3))
    print(u"Electron beam angular sizes = {0:.3f} µrad × {1:.3f} µrad".format(
        und.dxprime*1e6, und.dzprime*1e6))

    E = np.linspace(1400., 16000., 1460+1)
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
    ax1.set_ylim(0, None)
    leg1 = ax1.legend([(l1, l3), (l2, l4)], [r"$\sigma_x$", r"$\sigma_y$"],
                      handler_map={tuple: TwoLineObjectsHandler()},
                      loc=(0.86, 0.73))
    leg2 = ax1.legend([(l1, l2), (l3, l4)],
                      [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                      handler_map={tuple: TwoLineObjectsHandler()},
                      title='at energy spread', loc=(0.73, 0.5))
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
    ax2.set_ylim(0, None)
    leg1 = ax2.legend([(l1, l3), (l2, l4)], [r"$\sigma'_x$", r"$\sigma'_y$"],
                      handler_map={tuple: TwoLineObjectsHandler()},
                      loc=(0.56, 0.83))
    leg2 = ax2.legend([(l1, l2), (l3, l4)],
                      [r'$\sigma_E$ = 0', r'$\sigma_E$ = 0.1%'],
                      handler_map={tuple: TwoLineObjectsHandler()},
                      title='at energy spread', loc='upper right')
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)

    if withXrtSampling:
        theta = np.linspace(-1, 1, 401) * thetaMax * 2
        psi = np.linspace(-1, 1, 201) * psiMax * 2
        Eh = (np.arange(1, 12, 2) * und.E1 +
              np.linspace(-14, 10, 13)[:, np.newaxis])
        sh = Eh.shape

        Efine = np.linspace(10000-14., 10000+10., 1400+1)
        sx0fine, sz0fine = und.get_SIGMA(Efine, with0eSpread=True)
        sxfine, szfine = und.get_SIGMA(Efine)
        sPxfine, sPzfine = und.get_SIGMAP(Efine)

        und.eEspread = 0.
        flux, sigma2theta, sigma2psi, dx2, dz2 = und.real_photon_source_sizes(
            Eh.ravel(), theta, psi, method=0.39)
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

        l7 = ax1.scatter(Eh*1e-3, dx2**0.5*1e3,
                         s=fluxsh/fluxMax[np.newaxis, :]*50,
                         facecolors='none', edgecolors='C0')
        l8 = ax1.scatter(Eh*1e-3, dz2**0.5*1e3, s=fluxsh/fluxMax*50,
                         facecolors='none', edgecolors='C1')
        if withCoisson:
            cE, cA2, cS2 = coisson(und.E1*7, und.L0*und.Np, und.Np, 7)
            cXP = (und.dxprime**2 + cA2)**0.5
            cZP = (und.dzprime**2 + cA2)**0.5
            cX = (und.dx**2 + cS2)**0.5
            cZ = (und.dz**2 + cS2)**0.5

        # inset2
        axS2 = fig2.add_axes([0.17, 0.17, 0.22, 0.3])
        axS2.plot(Efine, sPxfine*1e6, '-C0')
        axS2.plot(Efine, sPzfine*1e6, '-C1')
        if withCoisson:
            l9 = axS2.scatter(cE, cXP*1e6, marker='+', s=40,
                              facecolors='C0', edgecolors='C0')
            l10 = axS2.scatter(cE, cZP*1e6, marker='+', s=40,
                               facecolors='C1', edgecolors='C1')
        axS2.scatter(Eh, sigma2theta**0.5*1e6,
                     s=fluxsh/fluxMax[np.newaxis, :]*50,
                     facecolors='none', edgecolors='C0')
        axS2.scatter(Eh, sigma2psi**0.5*1e6, s=fluxsh/fluxMax*50,
                     facecolors='none', edgecolors='C1')
        axS2.set_xlim(10000-14, 10000+10)
        axS2.set_xticklabels(
            ['', '-10 eV', r'$E_{\rm harm}$', '+10 eV'], minor=False)
        axS2.set_ylim(6, 14)
        ax2.arrow(6.3, 6.5, 3.0, 2.8, head_width=0.3, head_length=0.75)

        # inset1
        axS1 = fig1.add_axes([0.35, 0.36, 0.22, 0.3])
        axS1.plot(Efine, sz0fine*1e3, '--C1')
        axS1.plot(Efine, szfine*1e3, '-C1')
        if withCoisson:
            l11 = axS1.scatter(cE, cX*1e3, marker='+', s=40,
                               facecolors='C0', edgecolors='C0')
            l12 = axS1.scatter(cE, cZ*1e3, marker='+', s=40,
                               facecolors='C1', edgecolors='C1')
        axS1.scatter(Eh, dx2**0.5*1e3,
                     s=fluxsh/fluxMax[np.newaxis, :]*50,
                     facecolors='none', edgecolors='C0')
        axS1.scatter(Eh, dz2**0.5*1e3, s=fluxsh/fluxMax*50,
                     facecolors='none', edgecolors='C1')
        axS1.set_xlim(10000-14, 10000+10)
        axS1.set_xticklabels(
            ['', '-10 eV', r'$E_{\rm harm}$', '+10 eV'], minor=False)
        axS1.set_ylim(4.5, 6)
        ax1.arrow(8.5, 16., 1.1, -7.5, head_width=0.3, head_length=2.)

        lCur, lLab = [(l5, l6)], [r'xrt']
        if withCoisson:
            lCur.append((l9, l10))
            lLab.append('Coïsson, SPIE 88')
        leg3 = ax2.legend(lCur, lLab, title=r'sampled at $\sigma_E$ = 0 by:',
                          handler_map={tuple: TwoScatterObjectsHandler()},
                          loc=(0.37, 0.03))

        lCur, lLab = [(l7, l8)], [r'xrt']
        if withCoisson:
            lCur.append((l11, l12))
            lLab.append('Coïsson, SPIE 88')
        leg4 = ax1.legend(lCur, lLab, title=r'sampled at $\sigma_E$ = 0 by:',
                          handler_map={tuple: TwoScatterObjectsHandler()},
                          loc=(0.66, 0.25))

    fig1.savefig("undulatorLinearSize.png")
    fig2.savefig("undulatorAngularSize.png")


if __name__ == '__main__':
    main()
    plt.show()
    print("Done")

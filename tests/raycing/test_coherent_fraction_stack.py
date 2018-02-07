# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "11 Nov 2017"
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np
import time
import matplotlib.pyplot as plt

import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.coherence as rco

harmonics = 3, 7
energySpreads = 0.8e-3,
emittances = 280.,  # pmrad
bw = 1e-3
p = 40000.

Kmax = 1.92
betaX, betaZ = 2., 2.  # m
thetaMax, psiMax = 60e-6, 30e-6  # rad
binsx, binsz = 64+1, 32+1
theta = np.linspace(-thetaMax, thetaMax, binsx)
psi = np.linspace(-psiMax, psiMax, binsz)

repeats = 10  # "macro-electrons"

kw = dict(
    eE=3.0, eI=0.5,  # MAX IV
    name='IVU19', period=19, n=162,  # MAX IV
    targetE=(10000, 7),
    betaX=betaX, betaZ=betaZ,
    xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3,
#    targetOpenCL='CPU',
    xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False)


def main():
    und = rs.Undulator(**kw)

    for harmonic in harmonics:
        Eh = harmonic * und.E1
#        Eh = Eh * (1 - 1./harmonic/und.Np)  # detuned to harmonic's maximum
        und.eMin = Eh * (1 - bw/2)
        und.eMax = Eh * (1 + bw/2)
        und.reset()
        energy = np.ones(repeats) * Eh
        st = 'st' if harmonic == 1 else 'nd' if harmonic == 2 else 'rd'\
            if harmonic == 3 else 'th'
        for energySpread in energySpreads:
            und.eEspread = energySpread
            for emittancex in emittances:
                if energySpread > 1e-12:
                    ses = '{0:.1e}'.format(energySpread)
                    ses = ses[:-2] + ses[-1]  # removes "0" from power
                else:
                    ses = '0'
                txt = '{0}{1} harmonic ({2:.0f} eV){5}{3} energy spread{5}' +\
                    '$\\epsilon_x$ = {4:.0f} pmrad'
                cap = txt.format(harmonic, st, Eh, ses, emittancex, ', ')
                baseName = 'h{0:02d}-esp{1}-em{2:04.0f}'.format(
                    harmonic, energySpread, emittancex)

                emittancez = 10 if emittancex > 1e-12 else 0
                und.dx = np.sqrt(emittancex*betaX) * 1e-3  # in mm
                und.dz = np.sqrt(emittancez*betaZ) * 1e-3  # in mm
                und.dxprime = \
                    emittancex*1e-9 / und.dx if und.dx > 0 else 0.  # rad
                und.dzprime = \
                    emittancez*1e-9 / und.dz if und.dz > 0 else 0.  # rad

                Es, Ep = und.multi_electron_stack(energy, theta, psi)
                print("Es.shape", Es.shape)
                k = binsx * binsz

                D = np.array(Es).reshape((repeats, k), order='F').T
                J = np.dot(D, D.T.conjugate())  # / repeats
                print("solving eigenvalue problem...")
                start = time.time()
                wN, vN = rco.calc_eigen_modes_4D(J, eigenN=4)
                stop = time.time()
                print("the eigenvalue problem has taken {0:.4} s".format(
                    stop-start))
                print("vN.shape", vN.shape)
                print("Top 4 eigen values (4D) = {0}".format(wN))

                figE4 = rco.plot_eigen_modes(theta*p, psi*p, wN, vN,
                                             xlabel='x (mm)', ylabel='z (mm)')
                figE4.suptitle('Eigen modes of mutual intensity,\n'
                               + cap, fontsize=11)
                figE4.savefig('Modes-{0}-{1}.png'.format('s', baseName))

##                total4D = []
##                for i in range(repeats):
##                    total4D.append(Es[i])
#                Esl = np.concatenate(total4D).reshape((-1, binsx, binsz))
                print("solving PCA problem...")
                start = time.time()
                wPCA, vPCA = rco.calc_eigen_modes_PCA(Es, eigenN=4)
                stop = time.time()
                print("the PCA problem has taken {0:.4} s".format(stop-start))
                print("vPCA.shape", vPCA.shape)
                print("Top 4 eigen values (PCA) = {0}".format(wPCA))
                figEP = rco.plot_eigen_modes(theta*p, psi*p, wPCA, vPCA,
                                             xlabel='x (mm)', ylabel='z (mm)')
                figEP.suptitle('Principal components of one-electron images,\n'
                               + cap, fontsize=11)
                figEP.savefig('Components-{0}-{1}.png'.format('s', baseName))

                xdata = rco.calc_1D_coherent_fraction(Es, 'x', theta*p, p)
                ydata = rco.calc_1D_coherent_fraction(Es, 'z', psi*p, p)

                fig2D1, figXZ = rco.plot_1D_degree_of_coherence(
                    xdata, 'x', theta*p)
                fig2D2, figXZ = rco.plot_1D_degree_of_coherence(
                    ydata, 'z', psi*p, fig2=figXZ)

                fig2D1.suptitle('Mutual intensity for horizontal cut,\n '
                                + cap, size=11)
                fig2D2.suptitle('Mutual intensity for vertical cut,\n '
                                + cap, size=11)
                figXZ.suptitle('Intensity and Degree of Coherence,\n '
                               + cap, size=11)

                fig2D1.savefig('MutualI-x-{0}-{1}.png'.format('s', baseName))
                fig2D2.savefig('MutualI-z-{0}-{1}.png'.format('s', baseName))
                figXZ.savefig('DOC-{0}-{1}.png'.format('s', baseName))

    plt.show()
#    plt.close('all')


if __name__ == '__main__':
    main()
    print("Done")

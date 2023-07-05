# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "21 May 2020"
import numpy as np
import matplotlib.pyplot as plt
import time

# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rs


def intensity_in_transverse_plane(energy, theta, psi, I0):
    plt.imshow(I0[len(energy)//2, :, :],
               extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])


def main(case, icalc, plot):
    if case == 1:
        angleMaxX = 30e-6  # rad
        angleMaxZ = 30e-6  # rad
        energy = np.linspace(3800, 4150, 351)
        theta = np.linspace(-1, 1, 51) * angleMaxX
        psi = np.linspace(-1, 1, 51) * angleMaxZ
        kwargsCommon = dict(
            eE=3.0, eI=0.5, eEpsilonX=0.263, eEpsilonZ=0.008,
            period=18.5, n=108, K=0.52,
            xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3)
        kwargsXRT = dict(
            betaX=9.539, betaZ=1.982,
            xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
#            targetOpenCL='CPU',
            distE='BW')
    elif case == 2:
        angleMaxX = 8.7e-6  # rad
        angleMaxZ = 8.7e-6  # rad
        energy = np.linspace(7250, 7450, 201)
        theta = np.linspace(-1, 1, 801) * angleMaxX * 32
        psi = np.linspace(-1, 1, 101) * angleMaxZ * 4
        kwargsCommon = dict(
            eE=1.72, eI=0.3, eEpsilonX=11.371, eEpsilonZ=0.1,
            period=17, n=82, K=1.071)
        kwargsXRT = dict(
            betaX=1.65, betaZ=1,
            xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3,
            xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
            distE='BW')
    elif case == 3:
        angleMaxX = 8.7e-6  # rad
        angleMaxZ = 8.7e-6  # rad
        energy = np.linspace(500, 10000, 1901)
        theta = np.linspace(-1, 1, 801) * angleMaxX * 32
        psi = np.linspace(-1, 1, 101) * angleMaxZ * 4
        kwargsCommon = dict(
            eE=1.72, eI=0.3, eEpsilonX=11.371, eEpsilonZ=0.1,
            period=17, n=82, K=1.071)
        kwargsXRT = dict(
            betaX=1.65, betaZ=1,
            xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3,
            xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
            distE='BW')

    kwargsURGENT = dict(
        eSigmaX=(kwargsCommon['eEpsilonX']*kwargsXRT['betaX']*1e3)**0.5,
        eSigmaZ=(kwargsCommon['eEpsilonZ']*kwargsXRT['betaZ']*1e3)**0.5,
        eMin=energy[0], eMax=energy[-1],
        eN=len(energy)-1,
        icalc=icalc,
        xPrimeMax=angleMaxX*1e3, zPrimeMax=angleMaxZ*1e3,
        nx=12, nz=12)
    if kwargsURGENT['icalc'] == 3:
        kwargsCommon['eEpsilonX'] = 0.
        kwargsCommon['eEpsilonZ'] = 0.

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(u'energy (keV)')
    apX = angleMaxX * 2 * 1e6
    apZ = angleMaxZ * 2 * 1e6
    ax.set_ylabel(u'flux through {0:.1f}×{1:.1f} µrad² (ph/s/0.1%bw)'.format(
        apX, apZ))
    axplot = ax.semilogy if 'log-y' in plot else ax.plot

    # xrt Undulator
    kwargs = kwargsCommon.copy()
    kwargs.update(kwargsXRT)
    dtheta = theta[1] - theta[0]
    dpsi = psi[1] - psi[0]
    source = rs.Undulator(**kwargs)
    print('please wait...')
    I0x, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi,
                                                 eSpreadNSamples=13)
    whereTheta = np.argwhere(abs(theta) <= angleMaxX)
    wherePsi = np.argwhere(abs(psi) <= angleMaxZ)
    fluxX = I0x[:,
                slice(whereTheta.min(), whereTheta.max()+1),
                slice(wherePsi.min(), wherePsi.max()+1)]\
        .sum(axis=(1, 2)) * dtheta * dpsi
    axplot(energy*1e-3, fluxX, label='xrt')
    # end xrt Undulator

    if case == 3:
        eS, fS = np.loadtxt("misc/bessy.dc0.gz", skiprows=2, usecols=(0, 1),
                            unpack=True)
        axplot(eS*1e-3, fS, label='Spectra')

    # # UrgentUndulator
    # import xrt.backends.raycing as raycing
    # beamLine = raycing.BeamLine(azimuth=0, height=0)
    # kwargs = kwargsCommon.copy()
    # kwargs.update(kwargsURGENT)
    # sourceU = rs.UndulatorUrgent(beamLine, **kwargs)
    # I0u, l1, l2, l3 = sourceU.intensities_on_mesh()
    # # add the other 3 quadrants, except the x=0 and z=0 lines:
    # I0u = np.concatenate((I0u[:, :0:-1, :], I0u), axis=1)
    # I0u = np.concatenate((I0u[:, :, :0:-1], I0u), axis=2)
    # fluxU = I0u.sum(axis=(1, 2)) * sourceU.dx * sourceU.dz
    # urgentEnergy = sourceU.Es
    # axplot(urgentEnergy*1e-3, fluxU, label='Urgent')
    # # end UrgentUndulator

    # ax.set_ylim(1e9, 4e13)
    ax.legend()
    fig.savefig(u'flux_case{0}_xrt_UrgentICALC{1}.png'.format(case, icalc))

    # fig = plt.figure()
    # intensity_in_transverse_plane(energy, theta, psi, I0x)
    # fig = plt.figure()
    # intensity_in_transverse_plane(energy, theta, psi, I0u)


if __name__ == '__main__':
    t0 = time.time()
    # ICALC=1 non-zero emittance, finite N
    # ICALC=2 non-zero emittance, infinite N
    # ICALC=3 zero emittance, finite N

    # main(case=1, icalc=3, plot='lin-y')
    # main(case=1, icalc=2, plot='lin-y')
    # main(case=1, icalc=1, plot='lin-y')

    # main(case=2, icalc=3, plot='lin-y')
    # main(case=2, icalc=2, plot='lin-y')
    main(case=2, icalc=1, plot='lin-y')

    # main(case=3, icalc=3, plot='log-y')
    # main(case=3, icalc=2, plot='log-y')
    # main(case=3, icalc=1, plot='log-y')

    print("Done in {0} s".format(time.time()-t0))
    plt.show()

# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rs

xPrimeMax, zPrimeMax = 1., 0.3  # mrad
energy = np.linspace(1500, 37500, 601)
theta = np.linspace(-1., 1., 3) * xPrimeMax * 1e-3
psi = np.linspace(-1., 1., 51) * zPrimeMax * 1e-3
kwargs = dict(eE=3, eI=0.1, B0=1.7, distE='BW',
              xPrimeMax=xPrimeMax, zPrimeMax=zPrimeMax)

compareWithLegacyCode = True


def main():
    dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]

    source = rs.BendingMagnet(**kwargs)
    I0xrt = source.intensities_on_mesh(energy, theta, psi)[0]
    print I0xrt.shape, I0xrt.max()
    flux_xrt = I0xrt.sum(axis=(1, 2)) * dtheta * dpsi
    plt.plot(energy/1e3, flux_xrt, 'r', label='xrt', lw=5)

    if compareWithLegacyCode:
        del(kwargs['distE'])
        kwargs['eMin'] = energy[0]
        kwargs['eMax'] = energy[-1]
        kwargs['eN'] = len(energy)-1
        kwargs['nz'] = len(psi)//2
        source = rs.BendingMagnetWS(**kwargs)
        I0ws = source.intensities_on_mesh()[0]
        I0ws = np.concatenate((I0ws[:, :0:-1, :], I0ws), axis=1)
        I0ws = np.concatenate((I0ws[:, :, :0:-1], I0ws), axis=2)
        print I0ws.shape, I0ws.max()

        dtheta = (theta[-1] - theta[0]) / 2
        flux_ws = I0ws.sum(axis=(1, 2)) * dtheta * dpsi * 1e6
        plt.plot(energy/1e3, flux_ws, 'b', label='ws', lw=3)

    ax = plt.gca()
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'total flux through {0}×{1} µrad² (ph/s/0.1%bw)'.format(
        2*xPrimeMax, 2*zPrimeMax))
    plt.legend()

    plt.savefig('bm_flux.png')
    plt.show()

if __name__ == '__main__':
    main()

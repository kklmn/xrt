# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
# path to xrt:
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rs


def flux_through_aperture(energy, theta, psi, I0):
    dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
    flux = I0.sum(axis=(1, 2)) * dtheta * dpsi
    plt.plot(energy, flux)


def intensity_in_transverse_plane(energy, theta, psi, I0):
    plt.imshow(I0[len(energy)//2, :, :],
               extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])


def colored_intensity_in_transverse_plane(energy, theta, psi, I0):
    import matplotlib as mpl
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_axes([0.15*6/7, 0.15, 0.7*6/7, 0.7])
    ax.set_xlabel(u'$\\theta$ (µrad)')
    ax.set_ylabel(u'$\\psi$ (µrad)')
    axC = fig.add_axes([0.95*6/7, 0.15, 0.05, 0.7])
    plt.setp(axC.get_xticklabels(), visible=False)
    axC.yaxis.set_label_position("right")
    axC.yaxis.tick_right()
    axC.set_xlim([0, 1])
    axC.set_ylim([energy[0], energy[-1]])
    axC.set_ylabel('energy (eV)')

    hue, tmp, tmp = np.meshgrid(energy, theta, psi, indexing='ij')
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = I0 / I0.max()
    rgbI0 = mpl.colors.hsv_to_rgb(np.stack((hue, sat, val), axis=3))
    rgbI0sum = rgbI0.sum(axis=0)
    ax.imshow(rgbI0sum / rgbI0sum.max(),
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])

    hue = np.array(energy)[:, np.newaxis]
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    rgbC = mpl.colors.hsv_to_rgb(np.stack((hue, sat, val), axis=2))  # numpy1.10
    print(rgbC.shape)
    axC.imshow(rgbC / rgbC.max(), aspect='auto', origin='lower',
               extent=[0, 1, energy[0], energy[-1]])


source = rs.Undulator(eE=3.0, eI=0.5, eEpsilonX=0.263, eEpsilonZ=0.008,
                      betaX=9.539, betaZ=1.982, period=18.5, n=108, K=0.52,
                      distE='BW')
energy = np.linspace(3850, 4150, 601)
theta = np.linspace(-1, 1, 51) * 30e-6
psi = np.linspace(-1, 1, 51) * 30e-6
I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)

flux_through_aperture(energy, theta, psi, I0)
#intensity_in_transverse_plane(energy, theta, psi, I0)
#colored_intensity_in_transverse_plane(energy, theta, psi, I0)

plt.show()

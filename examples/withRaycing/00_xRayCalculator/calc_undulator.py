# -*- coding: utf-8 -*-
"""Select one of the 3 functions at the end of main()"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "22 Jan 2016"
import numpy as np
import matplotlib.pyplot as plt
import time
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
#    hsvI0 = np.stack((hue, sat, val), axis=3) # numpy1.10
    hsvI0 = np.zeros([d for d in hue.shape] + [3])
    hsvI0[:, :, :, 0] = hue
    hsvI0[:, :, :, 1] = sat
    hsvI0[:, :, :, 2] = val

    rgbI0 = mpl.colors.hsv_to_rgb(hsvI0)
    rgbI0sum = rgbI0.sum(axis=0)
    ax.imshow(rgbI0sum / rgbI0sum.max(),
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])

    hue = np.array(energy)[:, np.newaxis]
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
#    hsvC = np.stack((hue, sat, val), axis=2) # numpy1.10
    hsvC = np.zeros([d for d in hue.shape] + [3])
    hsvC[:, :, 0] = hue
    hsvC[:, :, 1] = sat
    hsvC[:, :, 2] = val
    rgbC = mpl.colors.hsv_to_rgb(hsvC)
    axC.imshow(rgbC / rgbC.max(), aspect='auto', origin='lower',
               extent=[0, 1, energy[0], energy[-1]])


def main():
    energy = np.linspace(3850, 4150, 601)
    theta = np.linspace(-1, 1, 51) * 30e-6
    psi = np.linspace(-1, 1, 51) * 30e-6
    kwargs = dict(eE=3.0, eI=0.5, eEpsilonX=0.263, eEpsilonZ=0.008,
                  betaX=9.539, betaZ=1.982, period=18.5, n=108, K=0.52,
#                  eEspread=1e-3,
                  xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3,
#                  targetOpenCL='CPU',
                  distE='BW')

#    energy = [200000., 200000.]
#    theta = np.linspace(-2./25000., 2./25000., 51)
#    psi = np.linspace(-2./25000., 2./25000., 51)
#    kwargs = dict(name='IVU18.5', eE=3.0, eI=0.5,
#                  eEpsilonX=0.263, eEpsilonZ=0.008,
#                  betaX=9., betaZ=2.,
#                  period=18.5, n=108, K=1.92,
#                  xPrimeMax=theta[-1]*1e3, zPrimeMax=psi[-1]*1e3, distE='BW')

    if True:  # xrt Undulator
        source = rs.Undulator(**kwargs)
        I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)
    else:  # Urgent
        kwargs['eSigmaX'] = (kwargs['eEpsilonX']*kwargs['betaX']*1e3)**0.5
        kwargs['eSigmaZ'] = (kwargs['eEpsilonZ']*kwargs['betaZ']*1e3)**0.5
        del(kwargs['distE'])
        del(kwargs['betaX'])
        del(kwargs['betaZ'])
        kwargs['eMin'] = energy[0]
        kwargs['eMax'] = energy[-1]
        kwargs['eN'] = len(energy)
        kwargs['icalc'] = 3
        kwargs['nx'] = len(theta)//2
        kwargs['nz'] = len(psi)//2
        import xrt.backends.raycing as raycing
        beamLine = raycing.BeamLine(azimuth=0, height=0)
        source = rs.UndulatorUrgent(beamLine, **kwargs)

        I0, l1, l2, l3 = source.intensities_on_mesh()
        I0 = np.concatenate((I0[:, :0:-1, :], I0), axis=1)
        I0 = np.concatenate((I0[:, :, :0:-1], I0), axis=2)

    flux_through_aperture(energy, theta, psi, I0)
#    intensity_in_transverse_plane(energy, theta, psi, I0)
#    colored_intensity_in_transverse_plane(energy, theta, psi, I0)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("Done in {0} s".format(time.time()-t0))
    plt.show()


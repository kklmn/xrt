# -*- coding: utf-8 -*-
r"""
.. _OAM-HelicalU:

Orbital Angular Momentum of helical undulator radiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The images below are produced by
``\examples\withRaycing\01_SynchrotronSources\undulatorVortex.py``.

This example calculates flux and Orbital Angular Momentum (OAM) of helical
undulator radiation. The calculation is done on a 3D (energy, theta, psi) or 4D
(energy, theta, psi, gamma) rectangular mesh for zero and true electron beam
energy spread, respectively. To calculate OAM projection along the propagation
direction, we first calculate *OAM intensity* :math:`I^l_s` for the field
:math:`E_s`

.. math::
    I^l_s = E_s^* i\left(\psi\frac{\partial}{\partial\theta}-
    \theta\frac{\partial}{\partial\psi}\right) E_s

and similarly for the field :math:`E_p`. OAM intensity is an incoherent
distribution, similarly to the "usual" intensity, as the phase information is
lost in it, and therefore it is suitable for incoherent accumulation of fields
originated from different electrons distributed within emittance and energy
spread distributions. After averaging intensity and OAM intensity over electron
beam energy spread distribution and convolving with electron beam angular
distributions, we obtain *vorticity* – the averaged normalized OAM value:

.. math::
    l_s = \int I^l_s d\theta d\psi \bigg/ \int I_s d\theta d\psi

and similarly for the field :math:`E_p`.

The images below were calculated at the maximum flux energies at the 1st to 4th
harmonics. As a reminder, the flux is maximized at energies slightly below the
formal harmonic energies; at these energies the transverse distribution is of a
donut shape. As a second reminder, the radiation from a helical undulator has
only the first harmonic on the undulator axis but also has higher harmonics at
finite observation angles.

The coloring of the transverse images is done by field phase. For this purpose,
the radiation field (here, :math:`E_s`) was *coherently* averaged over electron
beam energy spread distribution and electron beam angular distributions. This
operation is physically illegal but was used here only for the visual effect.
The measurable values – intensity, flux and vorticity – were determined by the
correct incoherent averaging. The image brightness represents intensity.

+--------------+------------------------+------------------------+
|              |   zero emittance,      |   true emittance,      |
|              |   zero energy spread   |   true energy spread   |
+==============+========================+========================+
|   |resTxt|   |       .. centered::  |U48results|               |
+--------------+------------------------+------------------------+
| 1st harmonic |      |U48Esh1-00|      |      |U48Esh1-ff|      |
+--------------+------------------------+------------------------+
| 2nd harmonic |      |U48Esh2-00|      |      |U48Esh2-ff|      |
+--------------+------------------------+------------------------+
| 3rd harmonic |      |U48Esh3-00|      |      |U48Esh3-ff|      |
+--------------+------------------------+------------------------+
| 4th harmonic |      |U48Esh4-00|      |      |U48Esh4-ff|      |
+--------------+------------------------+------------------------+

.. |resTxt| replace:: Flux and vorticity
.. |U48results| imagezoom:: _images/U48-flux-vorticity.png
.. |U48Esh1-00| imagezoom:: _images/U48-00-Es-OAM-transverse-image-140-352.727eV.png
.. |U48Esh1-ff| imagezoom:: _images/U48-ff-Es-OAM-transverse-image-140-352.727eV.png
   :loc: upper-right-corner
.. |U48Esh2-00| imagezoom:: _images/U48-00-Es-OAM-transverse-image-386-621.091eV.png
.. |U48Esh2-ff| imagezoom:: _images/U48-ff-Es-OAM-transverse-image-386-621.091eV.png
   :loc: upper-right-corner
.. |U48Esh3-00| imagezoom:: _images/U48-00-Es-OAM-transverse-image-660-920.000eV.png
.. |U48Esh3-ff| imagezoom:: _images/U48-ff-Es-OAM-transverse-image-660-920.000eV.png
   :loc: upper-right-corner
.. |U48Esh4-00| imagezoom:: _images/U48-00-Es-OAM-transverse-image-935-1220.000eV.png
.. |U48Esh4-ff| imagezoom:: _images/U48-ff-Es-OAM-transverse-image-935-1220.000eV.png
   :loc: upper-right-corner

As expected, vorticity approximately equals the harmonic number minus one in
the perfect case of zero emittance and zero energy spread. This pattern is
quite strongly affected by electron beam energy spread. Emittance plays a much
lesser role as the used angular acceptance is much bigger than the electron
beam angular distributions (360 µrad vs 5.8 and 2.0 µrad rms).
"""
__author__ = "Konstantin Klementiev"
__date__ = "22 Jan 2023"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs


# APPLE II period 48 mm, Kx=Ky=2, ID length=4m
period, n = 48., 83
Kx = Ky = 2  # E1 = 356.1 eV

prefix = 'U48-ff'
thetaMax, psiMax = 180e-6, 180e-6  # rad
energy = np.linspace(200., 1400., 1101)
imageEnergy = [352., 621., 919., 1220.]
# energy = np.array(imageEnergy)
imageIndex = np.searchsorted(energy, imageEnergy)

Source = rs.Undulator
undulatorKwargs = dict(
    eE=3., eI=0.3,
    betaX=9., betaZ=2.,  # m

    # eEpsilonX=0., eEpsilonZ=0.,
    eEpsilonX=0.300, eEpsilonZ=0.008,  # nmrad
    # eEspread=0.,
    eEspread=0.001,

    period=period, n=n, Ky=Ky, Kx=Kx,

    phaseDeg=90,
    xPrimeMax=1e3*thetaMax, zPrimeMax=1e3*psiMax,  # mrad
    xPrimeMaxAutoReduce=False,
    zPrimeMaxAutoReduce=False,

    # distE='BW',
    eMin=energy[0], eMax=energy[-1],

    targetOpenCL='GPU',
    )


def plot_flux(energy, flux, vorticity, distE='0.1%bw'):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Energy (eV)')
    if distE == '0.1%bw':
        ylabel = r'Flux (ph/s/0.1%bw)'
    elif distE == 'eV':
        ylabel = r'Flux (ph/s/eV)'
    ax.set_ylabel(ylabel)
    if distE == '0.1%bw':
        y = flux * energy * 1e-3
    elif distE == 'eV':
        y = flux
    ax.plot(energy, y)
    ax.set_ylim(0, None)
    # fig.savefig(prefix+"-spectral-flux.png")


def plot_results():
    outName = "U48-00-grid_method.pickle"
    with open(outName, 'rb') as f:
        energy, flux00, vEss00 = pickle.load(f)[0:3]
    outName = "U48-f0-grid_method.pickle"
    with open(outName, 'rb') as f:
        energy, fluxF0, vEssF0 = pickle.load(f)[0:3]
    outName = "U48-ff-grid_method.pickle"
    with open(outName, 'rb') as f:
        energy, fluxFF, vEssFF = pickle.load(f)[0:3]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Energy (eV)')
    ax.set_ylabel(r'Flux (ph/s/0.1%bw)')
    ax2 = ax.twinx()
    ax2.set_ylabel(u'Vorticity')

    y = flux00 * energy * 1e-3
    ax.plot(energy, y, label='flux, 0 emittance, 0 eSpread')
    y = fluxF0 * energy * 1e-3
    ax.plot(energy, y, label='flux, true emittance, 0 eSpread')
    y = fluxFF * energy * 1e-3
    ax.plot(energy, y, label='flux, true emittance, true eSpread')
    ax.set_ylim(0, None)
    ax.legend(loc=(0.22, 0.84))

    ax2.plot(energy, vEss00, '--', label='vort, 0 emittance, 0 eSpread')
    ax2.plot(energy, vEssF0, '--', label='vort, true emittance, 0 eSpread')
    ax2.plot(energy, vEssFF, '--', label='vort, true emittance, true eSpread')
    ax2.set_ylim(0, 3.7)
    ax2.legend(loc=(0.22, 0.68))

    fig.savefig(prefix+"-flux-vorticity.png")
    plt.show()


def intensity_in_transverse_plane(ie, theta, psi, I0):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'θ (µrad)')
    ax.set_ylabel(r'ψ (µrad)')
    e = energy[ie]
    ax.imshow(I0[:, :].T,
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])
    ax.text(0.98, 0.98, '@E={0:.1f}eV'.format(e),
            ha='right', va='top', color='white', transform=ax.transAxes)
    fig.savefig(prefix+"-transverse-image-{0}-{1:.3f}eV.png".format(ie, e))
    plt.close(fig)


def colored_intensity_in_transverse_plane(energy, theta, psi, I0):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_axes([0.15*6/7, 0.15, 0.7*6/7, 0.7])
    ax.set_xlabel(r'θ (µrad)')
    ax.set_ylabel(r'ψ (µrad)')
    axC = fig.add_axes([0.95*6/7, 0.15, 0.05, 0.7])
    plt.setp(axC.get_xticklabels(), visible=False)
    axC.yaxis.set_label_position("right")
    axC.yaxis.tick_right()
    axC.set_xlim([0, 1])
    axC.set_ylim([energy[0], energy[-1]])
    axC.set_ylabel('energy (eV)')

    hue, _, _ = np.meshgrid(energy, theta, psi, indexing='ij')
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = I0 / I0.max()
    hsvI0 = np.stack((hue, sat, val), axis=3)  # (len(energy), ntheta, npsi, 3)
    rgbI0 = mpl.colors.hsv_to_rgb(hsvI0)
    rgbI0sum = rgbI0.sum(axis=0)
    ax.imshow(np.swapaxes(rgbI0sum, 0, 1) / rgbI0sum.max(),
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])

    hue = np.array(energy)[:, np.newaxis]
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hsvC = np.stack((hue, sat, val), axis=2)  # numpy1.10
    rgbC = mpl.colors.hsv_to_rgb(hsvC)
    axC.imshow(rgbC / rgbC.max(), aspect='auto', origin='lower',
               extent=[0, 1, energy[0], energy[-1]])
    fig.savefig(prefix+"-spectral-transverse-image.png")


def colored_phase_in_transverse_plane(ie, theta, psi, I0, phase, vorticity,
                                      fieldLabel):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_axes([0.15*6/7, 0.15, 0.7*6/7, 0.7])
    ax.set_xlabel(r'θ (µrad)')
    ax.set_ylabel(r'ψ (µrad)')
    e = energy[ie]
    axC = fig.add_axes([0.95*6/7, 0.15, 0.05, 0.7])
    plt.setp(axC.get_xticklabels(), visible=False)
    axC.yaxis.set_label_position("right")
    axC.yaxis.tick_right()
    axC.set_xlim([0, 1])
    axC.set_ylim([-1, 1])  # in units of pi
    formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
    axC.yaxis.set_major_formatter(formatter)
    axC.set_ylabel('phase')

    val = I0 / I0.max()
    sat = np.ones_like(val)
    hue = (phase/np.pi + 1) * 0.5
    hsvI0 = np.stack((hue, sat, val), axis=2)
    rgbI0 = mpl.colors.hsv_to_rgb(hsvI0)  # (len(energy), ntheta, npsi, 3)
    ax.imshow(np.swapaxes(rgbI0, 0, 1) / rgbI0.max(),
              extent=[theta[0]*1e6, theta[-1]*1e6, psi[0]*1e6, psi[-1]*1e6])
    ax.text(0, 1.01, '@E={0:.1f}eV'.format(e),
            ha='left', va='bottom', color='k', transform=ax.transAxes)
    ax.text(1, 1.01, '{0} vorticity={1:.2f}'.format(fieldLabel, vorticity),
            ha='right', va='bottom', color='k', transform=ax.transAxes)

    hue = np.linspace(-1, 1, 101)[:, np.newaxis]
    hue -= hue.min()
    hue /= hue.max()
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hsvC = np.stack((hue, sat, val), axis=2)  # numpy1.10
    rgbC = mpl.colors.hsv_to_rgb(hsvC)
    axC.imshow(rgbC / rgbC.max(), aspect='auto', origin='lower',
               extent=[0, 1, -1, 1])

    fig.savefig(prefix+"-{0}-OAM-transverse-image-{1}-{2:.3f}eV.png".format(
        fieldLabel, ie, e))
    plt.close(fig)


def grid_method():
    if undulatorKwargs['eEspread'] == 0:
        ntheta, npsi = 801, 801
        eSpreadNSamples = None
    else:
        ntheta, npsi = 251, 251
        eSpreadNSamples = 41
    dtheta, dpsi = 2*thetaMax/(ntheta-1), 2*psiMax/(npsi-1)

    xprime = (undulatorKwargs['eEpsilonX']*1e-9/undulatorKwargs['betaX'])**0.5
    zprime = (undulatorKwargs['eEpsilonZ']*1e-9/undulatorKwargs['betaZ'])**0.5
    print(xprime, zprime)
    marginFactorX = 6*xprime/thetaMax + 1  # ±3σ
    marginFactorZ = 6*zprime/psiMax + 1  # ±3σ
    # print(marginFactorX, marginFactorZ)
    ntheta = int(ntheta*marginFactorX)
    npsi = int(npsi*marginFactorZ)

    thetaPlus = np.arange(ntheta)*dtheta
    psiPlus = np.arange(npsi)*dpsi
    thetaPlus -= thetaPlus[-1]*0.5
    psiPlus -= psiPlus[-1]*0.5
    whereTheta = (-thetaMax-1e-12 < thetaPlus) & (thetaPlus < thetaMax+1e-12)
    wherePsi = (-psiMax-1e-12 < psiPlus) & (psiPlus < psiMax+1e-12)
    theta = thetaPlus[whereTheta]
    psi = psiPlus[wherePsi]

    source = Source(**undulatorKwargs)
    print('E1 = ', source.E1)

    flux = np.zeros_like(energy)
    vEss = np.zeros_like(energy)
    vEps = np.zeros_like(energy)
    I0stack = np.zeros((len(energy), len(theta), len(psi)))
    for ie, e in enumerate(energy):
        print(u'energy {0:.1f} eV, {1} of {2}'.format(e, ie+1, len(energy)))
        Is, Ip, OAMs, OAMp, Es, Ep = source.intensities_on_mesh(
            [e], thetaPlus, psiPlus, eSpreadNSamples=eSpreadNSamples,
            resultKind='vortex')
        fluxIs = np.trapz(np.trapz(
            Is[0, whereTheta, :][:, wherePsi], psi), theta)
        fluxIp = np.trapz(np.trapz(
            Ip[0, whereTheta, :][:, wherePsi], psi), theta)
        flux[ie] = fluxIs + fluxIp

        lEs = OAMs[0, whereTheta, :][:, wherePsi] / fluxIs
        lEp = OAMp[0, whereTheta, :][:, wherePsi] / fluxIp
        vEs = np.trapz(np.trapz(lEs, psi), theta)
        vEp = np.trapz(np.trapz(lEp, psi), theta)
        vEss[ie] = vEs
        vEps[ie] = vEp
        print('vorticity: Es = {0}, Ep = {1}'.format(vEs, vEp))
        I0 = (Is + Ip)[0, whereTheta, :][:, wherePsi]
        I0stack[ie, :, :] = I0
        if ie in imageIndex:
            intensity_in_transverse_plane(ie, theta, psi, I0)
            phEs = np.angle(Es[0, whereTheta, :][:, wherePsi])
            phEp = np.angle(Ep[0, whereTheta, :][:, wherePsi])
            colored_phase_in_transverse_plane(
                ie, theta, psi, I0, phEs, vEs, 'Es')
            colored_phase_in_transverse_plane(
                ie, theta, psi, I0, phEp, vEp, 'Ep')

    integrand = flux/energy*1e3 if source.distE == 'BW' else flux
    totalPhPerS = np.trapz(integrand, energy)
    print('total flux = {0:.3g} ph/s'.format(totalPhPerS))

    dump = [energy, flux, vEss, vEps, totalPhPerS]
    outName = prefix+"-grid_method.pickle"
    with open(outName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)

    plot_flux(energy, flux, vEs)
    # colored_intensity_in_transverse_plane(energy, theta, psi, I0stack)
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    # raycing._VERBOSITY_ = 100
    grid_method()
    # plot_results()
    print("Done in {0} s".format(time.time()-t0))

# -*- coding: utf-8 -*-
r"""
.. _through-aperture:

Undulator radiation through rectangular aperture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The images below are produced by
\examples\withRaycing\01_SynchrotronSources\fluxThroughAperture.py.

This example illustrates the sampling concepts explained in Section
:ref:`Sampling strategies <sampling-strategies>`.

Grid sampling comes in two flavors: with a grid exactly matching the 100×100
µrad² aperture and with the same grid plus extra 30% margins on each side.
Adding margins is crucial for doing convolution with electron beam angular
spread. Notice that for the sharp field distributions at zero emittance and
zero energy spread, the grid has to be set with fine angular steps, otherwise
the flux density spectrum gets unreal ripples. One may play with *ntheta* and
*npsi* values to observe the effect.

In this particular example (MAX IV Linac), eEspread value is four times bigger
than a typical value for a storage ring and so the grid here needs a bigger
number of gamma (relative electron beam energy) samples to obtain a smooth
spectrum. One may play with *eSpreadNSamples* parameter to observe the effect.

Ray tracing examples illustrate two different ways of field sampling: uniform
reciprocal space sampling and sampling by intensity distribution. Energy can
also be sampled variously: uniformly within a given range and on an equidistant
mesh (energy scan). The latter case also delivers single energy field images,
which also has some calculation overhead.

The following cases are sorted in increased complexity (and calculation time)
from top to bottom.

.. list-table::
   :widths: 50 50

   * - |zeroEmittanceTxt|
     - |zeroEmittance|
   * - |fullEmittanceTxt|
     - |fullEmittance|
   * - |fullESpreadTxt|
     - |fullESpread|

.. |zeroEmittance| imagezoom:: _images/fluxThroughAperture-zeroEmittance.png
   :loc: upper-right-corner
.. |fullEmittance| imagezoom:: _images/fluxThroughAperture-fullEmittance.png
   :loc: upper-right-corner
.. |fullESpread| imagezoom:: _images/fluxThroughAperture-fulleEspread.png
   :loc: upper-right-corner

.. |zeroEmittanceTxt| replace:: Zero emittance, zero energy spread. Here,
   angular mesh has to be very dense due to very sharp field features.
.. |fullEmittanceTxt| replace:: Non-zero emittance, zero energy spread. Here,
   extra margins have to be added to the angular mesh in order to be able to
   convolve with electron beam divergence. In this example, adding 30% margins
   (the orange curve) was not enough.
.. |fullESpreadTxt| replace:: Non-zero emittance, non-zero energy spread. Here,
   the number of energy spread samples has to be adjusted for a large energy
   spread sigma otherwise the spectrum may have false sharp peaks.

Finally, the standard ray tracing approach ("rays-b") should be the method of
choice as it does not need any parameter adjustment (in contrast to the grid
methods). For not extreme cases (not too small aperture or not to big emittance
and not too big energy spread) grid methods can be most efficient but anyways
need a sanity check by ray tracing.

"""
__author__ = "Konstantin Klementiev"
__date__ = "15 Jan 2023"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.run as rr

slitSize = 2., 2.
slitPos = 20000.
thetaMax = slitSize[0]*0.5 / slitPos
psiMax = slitSize[1]*0.5 / slitPos

energy = np.linspace(11800, 12600, 401)
bins, ppb = len(energy), 1

undulatorKwargs = dict(
    eE=3.0, eI=0.1,
    betaX=9.0, betaZ=9.0,  # [m]
    # eEpsilonX=0.68, eEpsilonZ=0.51,  # [nmrad]
    eEpsilonX=0, eEpsilonZ=0,
    # eEspread=0.004,
    eEspread=0,

    period=15.0, n=328*2, K=2.1,
    # R0=20000.,  # near field, takes very long!

    xPrimeMax=2e3*thetaMax, zPrimeMax=2e3*psiMax,  # mrad
    xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
    eMin=energy[0], eMax=energy[-1],

    targetOpenCL='GPU',
    )

# xPrimeMax = (undulatorKwargs['eEpsilonX']*1e-9/undulatorKwargs['betaX'])**0.5
# undulatorKwargs['xPrimeMax'] = (2*xPrimeMax + thetaMax)*1e3
# zPrimeMax = (undulatorKwargs['eEpsilonZ']*1e-9/undulatorKwargs['betaZ'])**0.5
# undulatorKwargs['zPrimeMax'] = (2*zPrimeMax + psiMax)*1e3

if undulatorKwargs['eEpsilonX'] == 0:
    prefix = 'zeroEmittance-'
else:
    prefix = 'fullEmittance-'

if undulatorKwargs['eEspread'] != 0:
    prefix = 'fulleEspread-'

if 'R0' in undulatorKwargs:
    prefix = 'nearField-'


def build_beamline(nrays):
    beamLine = raycing.BeamLine()
    beamLine.nrays = nrays
    undulatorKwargs['distE'] = 'eV'
    beamLine.source = rs.Undulator(beamLine, nrays=nrays, **undulatorKwargs)
    opening = [-slitSize[0]*0.5, slitSize[0]*0.5,
               -slitSize[1]*0.5, slitSize[1]*0.5]
    beamLine.slit = ra.RectangularAperture(
        beamLine, 'FE slit', [0, slitPos, 0],
        ('left', 'right', 'bottom', 'top'), opening)
    return beamLine


def run_process(beamLine):
    if beamLine.sampling == 'a':
        waveSlit = beamLine.slit.prepare_wave(beamLine.source, beamLine.nrays)
        beamSource = beamLine.source.shine(wave=waveSlit)
        beamAp = waveSlit
    elif beamLine.sampling == 'b':
        beamSource = beamLine.source.shine()
        beamAp = beamLine.slit.propagate(beamSource)
    elif beamLine.sampling == 'c':
        beamSource = beamLine.source.shine(fixedEnergy=beamLine.fixedEnergy)
        beamAp = beamLine.slit.propagate(beamSource)

    outDict = {'beamSource': beamSource, 'beamAp': beamAp}
    return outDict
rr.run_process = run_process  # analysis:ignore


def define_plots(beamLine):
    plots = []

    caxis = xrtp.XYCAxis('energy', 'eV', bins=len(energy), ppb=1)
    plot = xrtp.XYCPlot(
        'beamAp', (1,),
        xaxis=xrtp.XYCAxis('x', 'mm', bins=bins, ppb=ppb),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=bins, ppb=ppb),
        caxis=caxis)
    size = max(slitSize[0], slitSize[1])
    plot.xaxis.limits = -size*0.6, size*0.6
    plot.yaxis.limits = -size*0.6, size*0.6
    plot.caxis.limits = energy[0], energy[-1]
    plot.xaxis.fwhmFormatStr = '%.2f'
    plot.yaxis.fwhmFormatStr = '%.2f'

    plots.append(plot)
    plot.baseName = prefix+beamLine.sampling+'-apertureIntensity'
    plot.saveName = [plot.baseName + '.png', ]

    return plots


def after_script(plots, outName):
    plot = plots[0]
    print('total flux = {0:.3g} ph/s'.format(plot.flux))
    dump = [plot.caxis.binCenters, plot.caxis.total1D, plot.flux]
    with open(outName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)
    print("Done")


def energy_scan(beamLine, plots, outName):
    flux = np.zeros_like(energy)
    plot = plots[0]
    for ie, e in enumerate(energy):
        print(u'energy {0:.1f} eV, {1} of {2}'.format(e, ie+1, len(energy)))
        beamLine.fixedEnergy = e
        beamLine.source.eMin = e - 0.5  # i.e. per 1eV band
        beamLine.source.eMax = e + 0.5
        for plot in plots:
            plot.saveName = [
                plot.baseName + '-{0}-{1:.1f}eV.png'.format(ie, e), ]
        yield
        flux[ie] = plot.flux

    totalPhPerS = np.trapz(flux, energy)
    print('total flux = {0:.3g} ph/s'.format(totalPhPerS))
    dump = [energy, flux, totalPhPerS]
    outName = prefix+"ray_tracing_c.pickle"
    with open(outName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)

    plt.plot(energy, flux, lw=0.7)
    plt.show()


def grid_method(kind):
    if undulatorKwargs['eEspread'] == 0:
        ntheta, npsi = 501, 501
        eSpreadNSamples = 36
    else:
        ntheta, npsi = 101, 101
        eSpreadNSamples = 401
    dtheta, dpsi = 2*thetaMax/(ntheta-1), 2*psiMax/(npsi-1)
    # theta = np.linspace(-1, 1, ntheta) * thetaMax
    # psi = np.linspace(-1, 1, npsi) * psiMax
    if kind == 'b':  # add margins but keep the grid spacings
        marginFactor = 1.3
        # marginFactor = max(
        #     undulatorKwargs['xPrimeMax']*1e-3/thetaMax,
        #     undulatorKwargs['zPrimeMax']*1e-3/psiMax)
        ntheta = int(ntheta*marginFactor)
        npsi = int(npsi*marginFactor)
    theta = np.arange(ntheta)*dtheta
    psi = np.arange(npsi)*dpsi
    theta -= theta[-1]*0.5
    psi -= psi[-1]*0.5
    source = rs.Undulator(**undulatorKwargs)
    flux = np.zeros_like(energy)
    for ie, e in enumerate(energy):
        print(u'energy {0:.1f} eV, {1} of {2}'.format(e, ie+1, len(energy)))
        I0, l1, l2, l3 = source.intensities_on_mesh(
            [e], theta, psi, eSpreadNSamples=eSpreadNSamples)
        whereTheta = (-thetaMax-1e-12 < theta) & (theta < thetaMax+1e-12)
        wherePsi = (-psiMax-1e-12 < psi) & (psi < psiMax+1e-12)
        # print(I0[:, :, wherePsi].shape, psi[wherePsi].shape)
        flux[ie] = np.trapz(np.trapz(I0[:, :, wherePsi], psi[wherePsi])
                            [:, whereTheta], theta[whereTheta])

    integrand = flux/energy*1e3 if source.distE == 'BW' else flux
    totalPhPerS = np.trapz(integrand, energy)
    print('total flux = {0:.3g} ph/s'.format(totalPhPerS))
    dump = [energy, integrand, totalPhPerS]
    outName = prefix+"grid_method_{}.pickle".format(kind)
    with open(outName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)

    plt.plot(energy, flux, lw=0.7)
    plt.show()


def ray_tracing(kind, nrays, repeats):
    outName = prefix+"ray_tracing_{0}.pickle".format(kind)
    beamLine = build_beamline(nrays)
    beamLine.sampling = kind
    plots = define_plots(beamLine)
    if kind in ['a', 'b']:
        xrtr.run_ray_tracing(
            plots, repeats=repeats, beamLine=beamLine,
            afterScript=after_script, afterScriptArgs=[plots, outName])
    if kind in ['c']:
        raycing._VERBOSITY_ = 0
        xrtr.run_ray_tracing(
            plots, repeats=repeats, beamLine=beamLine,
            # globalNorm=True,
            generator=energy_scan, generatorArgs=[beamLine, plots, outName])


def compare_methods(methods, distE='0.1%bw', wantOffset=False):
    """
    *distE* can be '0.1%bw' or 'eV', for calculating, correspondingly,
    logarithmic or linear flux density.
    """
    data = []
    for method in methods:
        fname = '{0}{1}.pickle'.format(prefix, method)
        with open(fname, 'rb') as f:
            res = pickle.load(f)
        if method == 'grid_method_a':
            label = 'grid-a: equidistant mesh sampling'
        elif method == 'grid_method_b':
            label = 'grid-b: equidistant mesh sampling with margins'
        elif method == 'ray_tracing_a':
            label = 'rays-a: uniform MC sampling in full energy range'
        elif method == 'ray_tracing_b':
            label = 'rays-b: MC sampling by intensity in full energy range'
        elif method == 'ray_tracing_c':
            label = 'rays-c: MC sampling by intensity at scanning energy'
        print(method, res[2])
        res.append(label)
        data.append(res)

    fig = plt.figure(figsize=(7, 5))
    rect = [0.12, 0.1, 0.85, 0.85]
    ax = fig.add_axes(rect, aspect='auto')
    title = 'Flux through aperture by xrt'
    if wantOffset:
        title += ', offsetted for clarity'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'Energy (eV)')
    if distE == '0.1%bw':
        ylabel = r'Flux (ph/s/0.1%bw)'
    elif distE == 'eV':
        ylabel = r'Flux (ph/s/eV)'
    ax.set_ylabel(ylabel)

    offset = 5e13 if wantOffset else 0
    for imethod, (e, flux, fluxPhPerS, label) in enumerate(data):
        flux_eV = flux / np.trapz(flux, e) * fluxPhPerS
        if distE == '0.1%bw':
            y = flux_eV * e * 1e-3
        elif distE == 'eV':
            y = flux_eV
        ax.plot(e, y+imethod*offset, label=label)
    ax.set_xlim(e[0], e[-1])
    ax.legend()

    fig.savefig(prefix+"FluxThroughAperture{0}.png".format(
        '-offsetted' if wantOffset else ''))
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    raycing._VERBOSITY_ = 100

    # select one at a time:
    # Note: grid_method can take very long when eEspread>0
    grid_method('a')  # intensities_on_mesh
    # grid_method('b')  # intensities_on_mesh with margins
    # ray_tracing('a', nrays=1e6, repeats=100)  # uniform ray sampling
    # ray_tracing('b', nrays=1e5, repeats=25)  # ray sampling by intensity
    # ray_tracing('c', nrays=2e4, repeats=1)  # energy scan

    # when the above methods are done, run the comparison functions:
    # compare_methods(
    #     ('grid_method_a', 'grid_method_b',
    #      'ray_tracing_a', 'ray_tracing_b', 'ray_tracing_c'), wantOffset=False)
    # compare_methods(
    #     ('grid_method_a', 'grid_method_b',
    #      'ray_tracing_a', 'ray_tracing_b', 'ray_tracing_c'), wantOffset=True)

    print("Done in {0} s".format(time.time()-t0))

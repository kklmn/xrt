# -*- coding: utf-8 -*-
r"""
.. !!! select one of the two functions to run at the very bottom !!!

.. _fzpDiffraction:

Diffraction from FZP
--------------------

.. imagezoom:: _images/1-LE-FZP_70nm-orders-r_E.*
   :align: right

This examples demonstrates diffraction from a Fresnel Zone Plate with variously
thick outer zone and at variable energy. The radial intensity distribution is
shown in the figure below for a 70-nm-outer-zone FZP. Notice that the 2nd order
was also calculated and together with other even orders indeed results in
vanishing intensity.

The energy dependence of efficiency for 3 different FZPs is shown below. The
horizontal bars mark the expected :math:`1/m^2\pi^2` levels for odd orders and
25% transmission for the 0th order. Watch how a zone plate becomes a band pass
filter as the outer zone size approaches the wavelength, here ~10 nm.

.. raw:: html

    <div class="clearer"> </div>

+----------+----------+----------+
| |FZP_70| | |FZP_50| | |FZP_30| |
+----------+----------+----------+

.. |FZP_70| imagezoom:: _images/1-LE-FZP_70nm-eff_E.*
.. |FZP_50| imagezoom:: _images/1-LE-FZP_50nm-eff_E.*
.. |FZP_30| imagezoom:: _images/1-LE-FZP_30nm-eff_E.*
   :loc: upper-right-corner

"""
#Set proper setting for the FZP and comment/uncomment one of the two main
#invoked functions (at the very bottom).
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.waves as rw

cwd = os.getcwd()

E_FZP = 120
mGold = rm.Material('Au', rho=19.3, kind='FZP')

p = 1.  # is not important
f = 2.
thinnestZone = 30e-6  # in mm
maxOrder = 3
maxDisplayOrder = 3
visualizeCrossSection = True
cmap = mpl.colormaps['jet']

prefix = '1-LE-FZP_{0:.0f}nm'.format(thinnestZone*1e6)
energies = np.linspace(50, 250, 101)
angles = np.linspace(0, 2e-3, 41)
#whatToScan = 'angle'
whatToScan = 'energy'
if whatToScan == 'energy':
    suffix = '_E'
else:
    suffix = '_pitch'

Nr = 101
nrays = 2e5


def build_beamline():  # for test=True
    beamLine = raycing.BeamLine()
    gsource = rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0), nrays=nrays,
        distx='annulus', distxprime=None, distzprime=None,
        distE='lines', energies=[E_FZP], polarization='h')

    beamLine.fzp = roe.NormalFZP(
        beamLine, 'FZP', [0, p, 0], pitch=np.pi/2, material=mGold, f=f,
        E=E_FZP, thinnestZone=thinnestZone, isCentralZoneBlack=True)
    gsource.dx = (0, beamLine.fzp.rn[-1])
    beamLine.fzp.area = np.pi * beamLine.fzp.rn[-1]**2 / 2
    print('FZP radius=', beamLine.fzp.rn[-1])

    r0max = beamLine.fzp.rn[-1] * 5
    rNmax = thinnestZone * 5
    beamLine.yglo = np.repeat(
        [10000 if i == 0 else f/i for i in range(maxOrder+1)], Nr)
    beamLine.dr = np.array([r0max/(Nr-1) if i == 0 else rNmax/(Nr-1)
                           for i in range(maxOrder+1)])
    beamLine.zglo = (np.arange(Nr) * beamLine.dr[:, np.newaxis]).flatten()
    beamLine.xglo = np.zeros_like(beamLine.yglo)

    return beamLine


def run_process(beamLine):
    ygloS = beamLine.yglo * (beamLine.E/E_FZP) + p
    wavelen = len(ygloS)
    wave3Dpoints = rs.Beam(nrays=wavelen, forceState=1, withAmplitudes=True)
    rw.prepare_wave(
        beamLine.fzp, wave3Dpoints, beamLine.xglo, ygloS, beamLine.zglo)
    wave3Dpoints.dS = 1

    wrepeats = 1
    for repeat in range(wrepeats):
        beamSource = beamLine.sources[0].shine(withAmplitudes=True)
        outDict = {'beamSource': beamSource}

        beamLine.fluxIn = (beamSource.Jss + beamSource.Jpp).sum()
        oeGlobal, oeLocal = beamLine.fzp.reflect(beamSource)
        oeLocal.area = beamLine.fzp.area
        rw.diffract(oeLocal, wave3Dpoints)
        if wrepeats > 1:
            print('wave repeats: {0} of {1} done'.format(repeat+1, wrepeats))

    beamLine.intensityDiffr = wave3Dpoints.Jss + wave3Dpoints.Jpp
    return outDict
rr.run_process = run_process


def plot_generator(plots, beamLine):
    ilen = Nr * (maxOrder+1)
    pickleName = os.path.join(cwd, prefix + suffix + '.pickle')
    if whatToScan.startswith('angle'):
        scanAxis = angles
        lenAngles = len(angles)
        eff = np.zeros((lenAngles, maxOrder+1))
        if visualizeCrossSection:
            extIntensityDiff = np.zeros((lenAngles, ilen))
    elif whatToScan.startswith('energy'):
        scanAxis = energies
        lenEnergies = len(energies)
        eff = np.zeros((lenEnergies, maxOrder+1))
        if visualizeCrossSection:
            extIntensityDiff = np.zeros((lenEnergies, ilen))

    for isa, sa in enumerate(scanAxis):
        if whatToScan.startswith('angle'):
            print('angle scan: {0}, {1} of {2}'.format(
                sa, isa+1, lenAngles))
            beamLine.E = E_FZP
            beamLine.fzp.pitch = np.pi/2 + sa
        elif whatToScan.startswith('energy'):
            E0 = sa
            print('energy scan: {0}eV, {1} of {2}'.format(
                E0, isa+1, lenEnergies))
            beamLine.E = E0
            beamLine.sources[0].energies = [E0]
#            beamLine.sources[0].energies = E0-dE/2, E0+dE/2
        yield
        flux = (beamLine.intensityDiffr * beamLine.zglo).reshape(
            maxOrder+1, Nr)
        intgl = flux.sum(axis=1) * beamLine.dr * 2*np.pi
        eff[isa, :] = intgl / beamLine.fluxIn
        print('efficiencies = {0}'.format(eff[isa, :]))
        if visualizeCrossSection:
            extIntensityDiff[isa, :] = beamLine.intensityDiffr
    dump = [0, maxOrder, scanAxis, eff, Nr, beamLine.zglo,
            visualizeCrossSection]
    if visualizeCrossSection:
        dump.append(extIntensityDiff)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)


def afterScript():
    print('Now run "visualize_efficiency()"')


def get_efficiency():
    beamLine = build_beamline()
    xrtr.run_ray_tracing([], repeats=1, beamLine=beamLine, processes=1,
                         generator=plot_generator, afterScript=afterScript)


def read_curves(fname):
    pickleName = os.path.join(cwd, fname)
    with open(pickleName, 'rb') as f:
        res = pickle.load(f)
    return res


def create_fig(rect2d, scanAxis, axisLabel, scanAxisFactor, maxOrder):
    fig1 = plt.figure(figsize=(12, 6), dpi=72)
    rect2dX = rect2d[2] / (maxOrder+1)
    ax = []
    sharey = None
    for o in range(maxOrder+1):
        dx = o*rect2dX + 0.03 if o > 0 else 0
        recti = [rect2d[0] + dx, rect2d[1], rect2dX - 0.002, rect2d[3]]
        axi = fig1.add_axes(recti, aspect='auto', sharey=sharey)
        sharey = axi if o > 0 else None
        axi.locator_params(axis='x', nbins=3)
        orderText = r'{0}$^{{\rm {1}}}$ order'.format(
            o, 'st' if o == 1 else 'nd' if o == 2 else 'rd' if o == 3
            else 'th')
        axi.text(0.98, 0.5, orderText, rotation='vertical',
                 transform=axi.transAxes, ha='right', va='center', fontsize=14)
        if o > 0:
            axi.set_xlabel(u'$r$ (µm)', fontsize=14)
        ax.append(axi)
    for axi in ax[1:-1]:
        xticks = axi.xaxis.get_major_ticks()
        xticks[-1].label1.set_visible(False)
    for axi in ax[2:]:
        plt.setp(axi.get_yticklabels(), visible=False)
    ax[0].set_xlabel(u'$r$ (mm)', fontsize=14)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[0].set_ylabel(r'normalized intensity (a.u.)', fontsize=14)

    rect2d = [0.95, 0.1, 0.02, 0.8]
    ax1c = fig1.add_axes(rect2d, aspect='auto')
    ax1c.set_ylabel(axisLabel, fontsize=14)
    plt.setp(ax1c, xticks=())
    yLim = scanAxis[0] * scanAxisFactor, scanAxis[-1] * scanAxisFactor
    ax1c.set_ylim(yLim[0], yLim[1])
    a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
    ax1c.imshow(a, aspect='auto', cmap=cmap, origin="lower",
                extent=[0, 1, yLim[0], yLim[1]])

    return fig1, ax


def visualize_efficiency():
    if whatToScan == 'energy':
        axisLabel = 'energy (eV)'
        scanAxisFactor = 1
    else:
        axisLabel = 'pitch (mrad)'
        scanAxisFactor = 1e3

    res = read_curves(prefix + suffix + '.pickle')
    minOrder, maxOrder, scanAxis, eff, Nr, zglo, pickleCrossSection = res[0:7]
    maxPlotOrder = min(maxDisplayOrder, maxOrder)

    figEff = plt.figure(figsize=(6, 6), dpi=72)
    rect2d = [0.15, 0.1, 0.8, 0.8]
    axEff = figEff.add_axes(rect2d, aspect='auto', xlabel=axisLabel,
                            ylabel='absolute efficiency')

    if whatToScan == 'energy':
        axEff.plot([120, 120], [0, 0.5], '--', color='gray', lw=0.5,
                   label=None)
        axEff.plot([110, 130], [0.25, 0.25], 'k', lw=1, label='ideal 0')
        ideal1 = np.pi**-2
        axEff.plot([110, 130], [ideal1, ideal1], 'r', lw=1, label='ideal 1')
        ideal3 = np.pi**-2 / 3**2
        axEff.plot([110, 130], [ideal3, ideal3], 'g', lw=1, label='ideal 3')
        startLegend = 1
#        locLegend = 0.7, 0.2
        locLegend = 'upper right'
    else:
        startLegend = 0
        locLegend = 'upper left'
    axEff.set_title(
        u'FZP with\n$f$ = 2 mm (at 120 eV) and {0:.0f} nm outer zone'.format(
            thinnestZone*1e6))
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 0], '.k', lw=2, label='xrt 0')
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 1], '.r', lw=2, label='xrt 1')
    axEff.plot(scanAxis*scanAxisFactor, eff[:, 3], '.g', lw=2, label='xrt 3')

    lines = axEff.lines
    labels = [l.get_label() for l in lines]
    axEff.legend(lines[startLegend:], labels[startLegend:], title='orders',
                 loc=locLegend)
    axEff.set_ylim(0, 0.5)

    figEff.savefig(prefix + '-eff{0}.png'.format(suffix))

    if pickleCrossSection:
        extIntensityDiff = res[7]
        elen = extIntensityDiff.shape[0]
        orders = maxOrder + 1 - minOrder
        extIntensityDiff = extIntensityDiff.reshape(elen, orders, Nr)
        z = zglo.reshape(orders, Nr)
        zFactor = 1e3 * np.ones(orders)
        zFactor[0] = 1
        rect2d = [0.05, 0.1, 0.8, 0.8]
        figr, axr = create_fig(rect2d, scanAxis, axisLabel, scanAxisFactor,
                               maxPlotOrder)

        for iE in range(elen):
            for o, iaxr in zip(range(maxPlotOrder+1), axr):
                f = zFactor[o]
                iaxr.plot(z[o, :]*f, extIntensityDiff[iE, o, :],
                          '-', lw=0.5, color=cmap(float(iE)/(elen-1)))
                iaxr.set_xlim(z[o, 0]*f, z[o, -1]*f)

        # figr.savefig(prefix + '-orders-r{0}.png'.format(suffix))

    plt.show()


if __name__ == '__main__':
    get_efficiency()
    # visualize_efficiency()

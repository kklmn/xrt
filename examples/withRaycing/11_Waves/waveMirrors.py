# -*- coding: utf-8 -*-
r"""
.. _mirrorDiffraction:

Diffraction from mirror surface
-------------------------------

This example shows wave diffraction from a geometric source onto a flat or
hemispheric screen. The source is rectangular with no divergence. The mirror
has no material properties (reflectivity is 1) for simplicity. Notice the
difference in the calculated flux between the rays and the waves.

+------------+------------+
|    rays    |    wave    |
+============+============+
| |mirrorWR| | |mirrorWW| |
+------------+------------+

.. |mirrorWR| imagezoom:: _images/mirror-256-01flat-01-beamFSMrays_f.*
.. |mirrorWW| imagezoom:: _images/mirror-256-01flat-02-beamFSMwave_f.*

The flux losses are not due to the integration errors, as was proven by
variously dense meshes. The losses are solely caused by cutting the tails, as
proven by a wider image shown below.

.. imagezoom:: _images/mirror-256wide-01flat-02-beamFSMwave_f.*

"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.apertures as ra
#import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.waves as rw

#coating = rm.Material('Au', rho=19.3)
#coating = rm.Material('Ni', rho=8.902)
coating = None

E0 = 150.
dE = 1.
dx = 1.
pitch = 4e-3
dp = 2000.
p = 10000.
q = 10000.
dPrime = 120e-6

xBins, xppb = 128, 2
yBins, yppb = 128, 2
#xBins, xppb = 256, 1
#yBins, yppb = 256, 1
eBins, eppb = 32, 8
xName = 'yaw'
yName = 'pitch'
unit, ufactor = u'µrad', 1e6

case = 'mirror'
#case = '2mirrors'
#case = 'aperture'
sourceType = 'flat'

#case = 'mirrorEll'
#case = '2mirrorsEll'
#sourceType = 'divergent'

prefix = case
nrays = 1e6

if sourceType == 'flat':
    kw = {'distx': 'flat', 'dx': dx, 'distz': 'flat', 'dz': dx,
          'distxprime': None, 'distzprime': None}
    prefix += '-01' + sourceType + '-'
elif sourceType == 'divergent':
    kw = {'distx': None, 'distz': None,
          'distxprime': 'flat', 'distzprime': 'flat',
          'dxprime': dPrime, 'dzprime': dPrime}
    prefix += '-03' + sourceType + '-'

kw['distE'] = 'lines'
kw['energies'] = [E0]
#kw['distE'] = 'flat'
#kw['energies'] = [E0-dE/2, E0+dE/2]

kwargs = dict(
    eE=6.08, eI=0.1,  # eEspread=0.001,
    #eEpsilonX=1., eEpsilonZ=0.01,
    eEpsilonX=0., eEpsilonZ=0.,
    betaX=20., betaZ=3.95,
    period=32.0, n=12,
    filamentBeam=True,
    uniformRayDensity=True,
    xPrimeMax=dPrime/2*1e3, zPrimeMax=dPrime/2*1e3,
    targetE=[E0, 1])


def build_beamline():
    beamLine = raycing.BeamLine()
#    rs.GeometricSource(
#        beamLine, 'source', nrays=nrays, polarization='horizontal', **kw)
    rs.Undulator(beamLine, nrays=nrays, **kwargs)

    if case == 'mirror':
        beamLine.diffoe = roe.OE(
            beamLine, 'PlaneMirror', (0, p, 0), pitch=pitch,
            material=coating)
        phiOffset = 2 * pitch
        zFlatScreen = q*np.sin(2*pitch)
        zHemScreen = 0
    elif case == 'mirrorEll':
        beamLine.diffoe = roe.EllipticalMirrorParam(
            beamLine, 'EllipticalMirror', (0, p, 0), p=p, q=q, pitch=pitch,
            material=coating)

        phiOffset = 2 * pitch
        zFlatScreen = q*np.sin(2*pitch)
        zHemScreen = 0
    elif case == '2mirrors':
        beamLine.oe0 = roe.OE(
            beamLine, 'PlaneMirror0', (0, p-dp, 0), pitch=pitch,
            material=coating)
        zFlatScreen = dp * np.tan(2*pitch)
        beamLine.diffoe = roe.OE(
            beamLine, 'PlaneMirror', (0, p, zFlatScreen), pitch=-pitch,
            positionRoll=np.pi, material=coating)
        phiOffset = 0
        zHemScreen = zFlatScreen
    elif case == '2mirrorsEll':
        beamLine.oe0 = roe.OE(
            beamLine, 'PlaneMirror0', (0, p-dp, 0), pitch=pitch,
            material=coating)
        zFlatScreen = dp * np.tan(2*pitch)
        beamLine.diffoe = roe.EllipticalMirrorParam(
            beamLine, 'EllipticalMirror', (0, p, zFlatScreen), p=p, q=q,
            pitch=-pitch, positionRoll=np.pi, material=coating)
        phiOffset = 0
        zHemScreen = zFlatScreen
    elif case == 'aperture':
        beamLine.diffoe = ra.RectangularAperture(
            beamLine, 'ra', (0, p, 0), ('left', 'right', 'bottom', 'top'))
        phiOffset = 0
        zFlatScreen = 0
        zHemScreen = 0

    if case.startswith('mir'):
        zvec = [0, -np.sin(2*pitch), np.cos(2*pitch)]
    else:
        zvec = [0, 0, 1]

    beamLine.fsmF = rsc.Screen(beamLine, 'FSM',
                               [0, p+q*np.cos(2*pitch), zFlatScreen], z=zvec)
    beamLine.fsmH = rsc.HemisphericScreen(
        beamLine, 'FSM', (0, p, zHemScreen), R=q, phiOffset=phiOffset)
    return beamLine


def run_process(beamLine):
    waveOnScreenF = beamLine.fsmF.prepare_wave(
        beamLine.diffoe, beamLine.xMeshF, beamLine.zMeshF)
    waveOnScreenH = beamLine.fsmH.prepare_wave(
        beamLine.diffoe, beamLine.xMeshH, beamLine.zMeshH)

    wrepeats = 2
    for repeat in range(wrepeats):
        beamSource = beamLine.sources[0].shine(
            withAmplitudes=True, fixedEnergy=E0)

        beamToOE = beamSource
        if case.startswith('2mirrors'):
            oe0Global, oe0Local = beamLine.oe0.reflect(beamSource)
            beamToOE = oe0Global

        if isinstance(beamLine.diffoe, roe.OE):
            oeGlobal, oeLocal = beamLine.diffoe.reflect(beamToOE)
        elif isinstance(beamLine.diffoe, ra.RectangularAperture):
            oeLocal = beamLine.diffoe.propagate(beamToOE)
            oeGlobal = beamToOE
        else:
            raise ValueError('unknown diffracting element')

        beamFSMraysF = beamLine.fsmF.expose(oeGlobal)
        beamFSMraysH = beamLine.fsmH.expose(oeGlobal)
        rw.diffract(oeLocal, waveOnScreenF)
        rw.diffract(oeLocal, waveOnScreenH)

        if wrepeats > 1:
            print('wave repeats: {0} of {1} done'.format(repeat+1, wrepeats))

    outDict = {'beamSource': beamSource,
               'beamFSMwave_f': waveOnScreenF,
               'beamFSMrays_f': beamFSMraysF,
               'beamFSMwave_h': waveOnScreenH,
               'beamFSMrays_h': beamFSMraysH,
               }
    return outDict
rr.run_process = run_process


def nodes(dmin, dmax, dbins):
    dd = (dmax - dmin) / dbins
    centers = np.linspace(dmin + dd/2, dmax - dd/2, dbins)
    return centers


def define_plots(beamLine):
    if sourceType == 'flat':
        mf = 0.55 * dx * 2
        ms = mf / q * ufactor
        ma = dx/2
    else:
#        mf = dPrime/2 * (p+q) * 4
        if case.endswith('Ell'):
            mf = 0.2 / (E0 / 150.)
        else:
            mf = 2. / (E0 / 150.)
        ms = mf / q * ufactor
        ma = dPrime/2 * p
    if case == 'aperture':
        beamLine.diffoe.opening = [-ma, ma, -ma, ma]

    beamLine.xMeshF = nodes(-mf, mf, xBins)
    beamLine.zMeshF = nodes(-mf, mf, yBins)
    beamLine.xMeshH = nodes(-ms, ms, xBins) / ufactor
    beamLine.zMeshH = nodes(-ms, ms, yBins) / ufactor

    dataPhi = raycing.get_phi
    dataTheta = raycing.get_theta

    plots = []
    plot = xrtp.XYCPlot(
        'beamFSMrays_f', aspect='auto',
        xaxis=xrtp.XYCAxis('x', 'mm', bins=xBins, ppb=xppb, limits=[-mf, mf]),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=yBins, ppb=yppb, limits=[-mf, mf]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '01-beamFSMrays_f'
    plot.caxis.limits = [E0-dE/2, E0+dE/2]
    plot.caxis.offset = E0
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_f', aspect='auto',
        xaxis=xrtp.XYCAxis('x', 'mm', bins=xBins, ppb=xppb, limits=[-mf, mf]),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=yBins, ppb=yppb, limits=[-mf, mf]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '02-beamFSMwave_f'
    plot.caxis.limits = [E0-dE/2, E0+dE/2]
    plot.caxis.offset = E0
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_f', aspect='auto',
        xaxis=xrtp.XYCAxis('x', 'mm', bins=xBins, ppb=xppb, limits=[-mf, mf]),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=yBins, ppb=yppb, limits=[-mf, mf]),
        caxis=xrtp.XYCAxis('Es phase', '', bins=yBins, ppb=yppb,
                           data=raycing.get_Es_phase))
    plot.baseName = prefix + '02Es-beamFSMwave_f'
#    plot.ax2dHist.locator_params(axis='x', nbins=5)
    plot.caxis.limits = [-np.pi, np.pi]
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMrays_h', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb, data=dataTheta,
                           limits=[-ms, ms]),
        yaxis=xrtp.XYCAxis(yName, unit, bins=yBins, ppb=yppb, data=dataPhi,
                           limits=[-ms, ms]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '01-beamFSMrays_h'
    plot.caxis.limits = [E0-dE/2, E0+dE/2]
    plot.caxis.offset = E0
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_h', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb, data=dataTheta,
                           limits=[-ms, ms]),
        yaxis=xrtp.XYCAxis(yName, unit, bins=yBins, ppb=yppb, data=dataPhi,
                           limits=[-ms, ms]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '02-beamFSMwave_h'
    plot.caxis.limits = [E0-dE/2, E0+dE/2]
    plot.caxis.offset = E0
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_h', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb, data=dataTheta,
                           limits=[-ms, ms]),
        yaxis=xrtp.XYCAxis(yName, unit, bins=yBins, ppb=yppb, data=dataPhi,
                           limits=[-ms, ms]),
        caxis=xrtp.XYCAxis('Es phase', '', bins=yBins, ppb=yppb,
                           data=raycing.get_Es_phase))
    plot.baseName = prefix + '02Es-beamFSMwave_h'
#    plot.ax2dHist.locator_params(axis='x', nbins=5)
    plot.caxis.limits = [-np.pi, np.pi]
    plots.append(plot)

    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.3f'
        plot.yaxis.fwhmFormatStr = '%.3f'
        plot.fluxFormatStr = '%.2p'
        if hasattr(plot, 'baseName'):
            plot.saveName = plot.baseName + '.png'
#            plot.persistentName = plot.baseName + '.pickle'
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)

# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()

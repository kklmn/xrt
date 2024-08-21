# -*- coding: utf-8 -*-
r"""
.. _YoungDiffraction:

Young's experiment with undulator source
----------------------------------------

This example shows double slit diffraction of an undulator beam. The single
slit width is 10 µm, the slit separation is variable (displayed is edge-to-edge
distance), the slit position is 90 m from the source and the screen is at 110
m.

.. animation:: _images/YoungRays
.. animation:: _images/YoungWave

"""
__author__ = "Roman Chernikov", "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
# import time
#import matplotlib as mpl
#mpl.use('Agg')
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.waves as rw
import numpy as np

R0 = 90000
mynrays = 2e6
divsZ = -8
divsX = -8
#zero divergence
slitDx = 0.5
slitDz = 0.05
dR = 16000.
SCRx = 0.7
SCRz = 10
dE = 1e-5

nrep = 512
#finite divergence

E0 = 12000
kwargs = dict(
    betaX=1.20, betaZ=3.95,
    period=29., n=172,
    eE=6.08, eI=0.1,  # eEspread=0.001,
    eEpsilonX=0., eEpsilonZ=0.0,
    #eEpsilonX=1., eEpsilonZ=0.01,
    filamentBeam=True,
    uniformRayDensity=True,
    xPrimeMax=np.arctan(1.5/R0)*1.e3, zPrimeMax=np.arctan(1.5/R0)*1e3,
    #xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
    targetE=[E0, 3])

#E0 = 1355
eMinRays = E0 - 0.5
eMaxRays = E0 + 0.5
kwargs['eMin'] = eMinRays
kwargs['eMax'] = eMaxRays
prefix = 'far{0:02.0f}m-E0{1:4.0f}-'.format(R0*1e-3, E0)
suffix = "_realdiv"

if False:  # zero source size:
    kwargs['eSigmaX'] = 1e-3
    kwargs['eSigmaZ'] = 1e-3
    kwargs['eEpsilonX'] = 0
    kwargs['eEpsilonZ'] = 0

xBins = 32
zBins = 512
eBins = 16
xppb = 4
zppb = 1
eppb = 16
xfactor = 1e3
zfactor = 1e3
isScreenHemispheric = False
if isScreenHemispheric:
    screenName = '-hemis'
    xlimits = [-4*slitDx/R0*1e6, 4*slitDx/R0*1e6]
    zlimits = [-4*slitDz/R0*1e6, 4*slitDz/R0*1e6]
    xName = '$\\theta$'
    zName = '$\\phi$'
    unit = u'µrad'
else:
    screenName = '-plane'
    xlimits = [-SCRx*slitDx*1e3, SCRx*slitDx*1e3]
    zlimits = [-SCRz*slitDz*1e3, SCRz*slitDz*1e3]
    xName = '$x$'
    zName = '$z$'
    unit = u'µm'

dx = (xlimits[1] - xlimits[0]) / xBins
xmesh = np.linspace((xlimits[0] + dx/2) / xfactor,
                    (xlimits[1] - dx/2) / xfactor, xBins)
dz = (zlimits[1] - zlimits[0]) / zBins
zmesh = np.linspace((zlimits[0] + dz/2) / zfactor,
                    (zlimits[1] - dz/2) / zfactor, zBins)


def build_beamline(nrays=mynrays):
    beamLine = raycing.BeamLine()
    rs.Undulator(beamLine, nrays=nrays, **kwargs)

    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0', (0, R0-1, 0))
    beamLine.slit = ra.DoubleSlit(
        beamLine, 'squareSlit', [0, R0, 0], ('left', 'right', 'bottom', 'top'),
        [-slitDx/2, slitDx/2, -slitDz/2, slitDz/2], shadeFraction=0.1)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', [0, R0+dR, 0])
    return beamLine


def run_process(beamLine):
    beamFSM1wave = beamLine.fsm1.prepare_wave(beamLine.slit, xmesh, zmesh)
    beamSource = None
    wrepeats = 1
    for repeat in range(wrepeats):
        beamSource = beamLine.sources[0].shine(accuBeam=beamSource)
        beamFSM0 = beamLine.fsm0.expose(beamSource)
        waveOnSlit = beamLine.slit.propagate(beamSource)
        beamFSM1 = beamLine.fsm1.expose(beamSource)
        waveOnSlit.area = beamLine.slit.area
        rw.diffract(waveOnSlit, beamFSM1wave)
        if wrepeats > 1:
            print('wave repeats: {0} of {1} done'.format(repeat+1, wrepeats))
    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamFSM1': beamFSM1,
               'beamFSM1wave': beamFSM1wave
               }
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSM0', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        title='1 - Source')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM1', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        title='2 - DS Propagation Rays')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM1wave', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        #fluxKind='wave',
        title='3 - DS Propagation Wave')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM1wave', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('Es phase', '',
                           data=raycing.get_Es_phase, limits=[-np.pi, np.pi],
                           bins=eBins, ppb=eppb),
        fluxKind='s',
        title='4 - DS Propagation Es phase')
    plots.append(plot)

    for plot in plots:
        plot.textPanel = plot.fig.text(
            0.82, 0.6, '', transform=plot.fig.transFigure, size=14, color='r',
            ha='center')
    return plots


def plot_generator(plots, beamLine):
    for slitZ in np.linspace(0.025, 0.2, 8):
#    for slitZ in [0.05]:
        #slitZ=0.06
        slitwidth = 0.01
        slit_pos = 0
        R0s = 90.
        SP = (slitZ - 2*slitwidth) / slitZ
        beamLine.slit.center[2] = slit_pos
        beamLine.slit.opening[2] = -slitZ/2.
        beamLine.slit.opening[3] = slitZ/2.
        beamLine.slit.shadeFraction = SP
        beamLine.slit.area = slitDx * 2 * slitwidth
#        print("area")
#        print(beamLine.slit.area, slitDx, slitDz, (1-SP))
        dX = 20.
        beamLine.slit.center[1] = R0s*1000.
        beamLine.fsm1.center[1] = (R0s+dX)*1000.
        str1 = '{0} - slit at {1:.0f}m, deltaSlit {2:03.0f}mum'
        str2 = ', slitWidth {3:.0f}mum, screen at {4:.0f}m.png'
        tt = u'Δ slit = {0:.0f} µm'.format((slitZ-slitwidth)*1e3)
        for plot in plots:
            plot.ax2dHist.locator_params(axis='x', nbins=4)
            plot.xaxis.limits = xlimits
            plot.yaxis.limits = zlimits
            plot.xaxis.fwhmFormatStr = None
            plot.yaxis.fwhmFormatStr = '%.2f'
            if plot.caxis.label == 'energy':
                plot.caxis.limits = [eMinRays, eMaxRays]
                plot.caxis.offset = (eMinRays + eMaxRays) / 2
            plot.fluxFormatStr = '%.2p'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(tt)
            plot.baseName = plot.title
            plot.saveName = (str1 + str2).format(
                plot.baseName, R0s, (slitZ-slitwidth)*1e3,
                slitwidth*1e3, R0s+dX)
#            plot.persistentName = plot.saveName + '.pickle'
        yield


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=nrep, beamLine=beamLine,
                         generator=plot_generator)

if __name__ == '__main__':
    main()

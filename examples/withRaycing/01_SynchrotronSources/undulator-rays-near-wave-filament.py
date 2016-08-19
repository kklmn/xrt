# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import time
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

#E0 = 4850.
E0 = 11350.
K = 1.91988

#what, repeats = '-rays', 100
what, repeats = '-wave', 1000

suffix = what

#R0, lim, case = 5000., 1., '05m'
R0, lim, case = 25000., 5., '25m'
xlimits = [-lim, lim]  # Horizontal limits of the plot [mm]
zlimits = [-lim, lim]  # Vertical limits of the plot [mm]
accMax = xlimits[-1] / R0 * 1e3

bins = 256  # Number of bins in the plot histogram
ppb = 1  # Number of pixels per histogram bin
eUnit = 'eV'

prefix = 'xrt-far' + case
#prefix = 'xrt-near' + case
filamentBeam = False

kwargs = dict(
    eE=3., eI=0.5, eEspread=0.001,
    eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9.539, betaZ=1.982,
    period=18.5, n=108, K=K,
    xPrimeMax=accMax, zPrimeMax=accMax,
    xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False)
if 'near' in prefix:
    kwargs['R0'] = R0
if filamentBeam:
    kwargs['filamentBeam'] = filamentBeam
    suffix += '-filament'

if False:  # force zero source size and energy spread:
    kwargs['eSigmaX'] = 0
    kwargs['eSigmaZ'] = 0
    kwargs['eEpsilonX'] = 0
    kwargs['eEpsilonZ'] = 0
    kwargs['eEspread'] = 0


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine()
    beamLine.source = rs.Undulator(beamLine, nrays=nrays, **kwargs)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, R0, 0))
    return beamLine


def run_process(beamLine):
    startTime = time.time()
    kw = dict(fixedEnergy=E0)
    if 'wave' in what:
        wave1 = beamLine.fsm1.prepare_wave(
            beamLine.source, beamLine.fsmExpX, beamLine.fsmExpZ)
        kw['wave'] = wave1
        beamSource = beamLine.source.shine(**kw)
        outDict = {'beamSource': beamSource, 'beamFSM1': wave1}
    else:
        beamSource = beamLine.source.shine(**kw)
        beamFSM1 = beamLine.fsm1.expose(beamSource)
        outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1}
    print('shine time = {0}s'.format(time.time() - startTime))
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('Es phase', '',
                         data=raycing.get_Es_phase, limits=[-np.pi, np.pi],
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='s', aspect='auto', title='horizontal polarization flux')
    plot.saveName = prefix + '1horFlux' + suffix + '.png'
    plots.append(plot)

    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('Ep phase', '',
                         data=raycing.get_Ep_phase, limits=[-np.pi, np.pi],
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='p', aspect='auto', title='vertical polarization flux')
    plot.saveName = prefix + '3verFlux' + suffix + '.png'
    plots.append(plot)

    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.3f'
        plot.yaxis.fwhmFormatStr = '%.3f'
        plot.caxis.fwhmFormatStr = None
        plot.fluxFormatStr = '%.2p'
        plot.ax1dHistE.set_yticks([l*np.pi for l in (-1, -0.5, 0, 0.5, 1)])
        plot.ax1dHistE.set_yticklabels(
            (r'$-\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'))
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc

mGold = rm.Material('Au', rho=19.3, kind='FZP')
E0, dE = 400, 5


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)
#    source=rs.GeometricSource(
#        beamLine, 'GeometricSource', (0, 0, 0),
#        nrays=nrays, distx='flat', dx=0.12, distz='flat', dz=0.12,
#        dxprime=0, dzprime=0,
#        distE='flat', energies=(E0-dE, E0+dE), polarization='horizontal')
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, distx='annulus', dx=(0, 0.056),
        dxprime=0, dzprime=0,
        distE='flat', energies=(E0-dE, E0+dE), polarization='horizontal')

    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0, 10., 0))
#    beamLine.fzp = roe.NormalFZP(beamLine, 'FZP', [0, 10., 0], pitch=np.pi/2,
#      material=mGold, f=2., E=E0, N=50)
    beamLine.fzp = roe.GeneralFZPin0YZ(
        beamLine, 'FZP', [0, 10., 0], pitch=np.pi/2,
        material=mGold, f1='inf', f2=(0, 0, 2.), E=E0, N=500, phaseShift=np.pi)

#    source.dx = 2 * beamLine.fzp.rn[-1]
#    source.dz = source.dx
    beamLine.fzp.order = 1
    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', (0, 12., 0))
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
#    beamLine.feFixedMask.propagate(beamSource)
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamFZPglobal, beamFZPlocal = beamLine.fzp.reflect(beamSource)
    beamFSM2 = beamLine.fsm2.expose(beamFZPglobal)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamFZPglobal': beamFZPglobal,
               'beamFZPlocal': beamFZPlocal,
               'beamFSM2': beamFSM2}
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []

#    plot = xrtp.XYCPlot(
#        'beamFSM1', (1,), xaxis=xrtp.XYCAxis(r'$x$', r'$\mu$m'),
#        yaxis=xrtp.XYCAxis(r'$z$', r'$\mu$m'), title='FSM1_E')
#    plot.caxis.fwhmFormatStr = None
#    plot.saveName = [plot.title + '.png', ]
#    plots.append(plot)
#
    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1, -1),
        xaxis=xrtp.XYCAxis(r'$x$', r'$\mu$m', bins=512, ppb=1,
                           limits=[-12, 12]),
        yaxis=xrtp.XYCAxis(r'$y$', r'$\mu$m', bins=512, ppb=1,
                           limits=[-12, 12]),
        caxis='category',
        title='localZ')
    plot.caxis.fwhmFormatStr = None
    plot.textPanel = plot.ax1dHistX.text(
        0.5, 0.02, '', size=14, color='w', transform=plot.ax1dHistX.transAxes,
        ha='center', va='bottom')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1, -1),
        xaxis=xrtp.XYCAxis(r'$x$', r'$\mu$m', bins=512, ppb=1),
        yaxis=xrtp.XYCAxis(r'$y$', r'$\mu$m', bins=512, ppb=1),
        caxis='category',
        title='localFull')
    plot.caxis.fwhmFormatStr = None
    plot.textPanel = plot.ax1dHistX.text(
        0.5, 0.02, '', size=14, color='w', transform=plot.ax1dHistX.transAxes,
        ha='center', va='bottom')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', r'nm', bins=256, ppb=1, limits=[-500, 500]),
        yaxis=xrtp.XYCAxis(r'$z$', r'nm', bins=256, ppb=1, limits=[-500, 500]),
        caxis='category',
        title='FSM2_Es')
    plot.caxis.fwhmFormatStr = fwhmFormatStrE
    plot.fluxFormatStr = '%.2e'
    plot.textPanel = plot.ax1dHistX.text(
        0.5, 0.02, '', size=14, color='w', transform=plot.ax1dHistX.transAxes,
        ha='center', va='bottom')
    plots.append(plot)
    return plots


def plot_generator(plots, beamLine):
    nShifts = 8
    phaseShifts = np.arange(0, nShifts, dtype=float) / nShifts * 2 * np.pi
    strPhaseShifts = ('0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$')

    for iPhaseShift, (phaseShift, strPhaseShift) in\
            enumerate(zip(phaseShifts, strPhaseShifts)):
        beamLine.fzp.set_phase_shift(phaseShift)
        for plot in plots:
            plot.saveName = ['FZP-{0}{1}.png'.format(
                plot.title, iPhaseShift)]
            try:
                plot.textPanel.set_text(u'phase shift = {0}'.format(
                    strPhaseShift))
            except AttributeError:
                pass
        yield


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=360, generator=plot_generator,
                         beamLine=beamLine, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

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

E0, dE = 9000., 2.,
p = 20000.


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)
#    rs.GeometricSource(beamLine, 'GeometricSource', (0, 0, 0),
#      nrays=nrays, dx=0, dz=0, distxprime='flat', dxprime=1e-4,
#      distzprime='flat', dzprime=1e-4,
#      distE='flat', energies=(E0-dE, E0+dE), polarization='horizontal')
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, distx='annulus', dx=(0, 1), dxprime=0, dzprime=0,
        distE='flat', energies=(E0-dE, E0+dE), polarization='horizontal')
    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0., p-100, 0.))
    siCryst = rm.CrystalSi(hkl=(1, 1, 1), geom='Bragg-Fresnel')
    pitch = \
        siCryst.get_Bragg_angle(E0) - siCryst.get_dtheta_symmetric_Bragg(E0)
#    pitch = np.pi/2
    f = 0, p * np.cos(pitch), p * np.sin(pitch)
    beamLine.fzp = roe.GeneralFZPin0YZ(
        beamLine, 'FZP', [0., p, 0.], pitch=pitch,
        material=siCryst, f1='inf', f2=f, E=E0, N=340)
    beamLine.fzp.order = 1
    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', [0, 0, 0],
                               z=(0, -np.sin(2*pitch), np.cos(2*pitch)))
    beamLine.fsm2RelPos = np.linspace(0, p, 21)
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
#    beamLine.feFixedMask.propagate(beamSource)
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamFZPglobal, beamFZPlocal = beamLine.fzp.reflect(beamSource)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamFZPglobal': beamFZPglobal,
               'beamFZPlocal': beamFZPlocal}
    for i, pos in enumerate(beamLine.fsm2RelPos):
        beamLine.fsm2.center[1] = beamLine.fzp.center[1] +\
            pos * np.cos(2 * beamLine.fzp.pitch)
        beamLine.fsm2.center[2] = pos * np.sin(2 * beamLine.fzp.pitch)
        beamFSM2 = beamLine.fsm2.expose(beamFZPglobal)
        outDict['beamFSM2-{0:02d}'.format(i+1)] = beamFSM2

    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []

#    plot = xrtp.XYCPlot('beamFSM1', (1,), xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
#      yaxis=xrtp.XYCAxis(r'$z$', u'µm'), title='FSM1_E')
#    plot.caxis.fwhmFormatStr = None
#    plot.baseName = plot.title
#    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1, -1),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=512, ppb=1, limits=[-250, 250]),
        yaxis=xrtp.XYCAxis(r'$y$', u'µm', bins=512, ppb=1, limits=[-250, 250]),
        caxis='category', title='BFZPlocal')
    plot.baseName = plot.title
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1, -1),  # aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=100, ppb=1, limits=[-1, 1]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=512, ppb=1),
        caxis='category', title='BFZPlocalFull')
    plot.xaxis.fwhmFormatStr = None
    plot.ax1dHistX.set_xticks([-1, 0, 1])
    plot.baseName = plot.title
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=512, ppb=1, limits=[-0.5, 0.5]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=512, ppb=1, limits=[-0.5, 0.5]),
        title='BFZPlocalE')
    plot.baseName = plot.title
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2-01', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis='category',
        title='BraggFresnelFSM2-01c')
    plot.xaxis.fwhmFormatStr = fwhmFormatStrE
    plot.yaxis.fwhmFormatStr = fwhmFormatStrE
    plot.fluxFormatStr = '%.2e'
    plot.baseName = plot.title
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2-01', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        title='BraggFresnelFSM2-01E')
    plot.xaxis.fwhmFormatStr = fwhmFormatStrE
    plot.yaxis.fwhmFormatStr = fwhmFormatStrE
    plot.fluxFormatStr = '%.2e'
    plot.baseName = plot.title
    plots.append(plot)

    beamFSM2 = 'beamFSM2-{0:d}'.format(len(beamLine.fsm2RelPos))
    plot = xrtp.XYCPlot(
        beamFSM2, (1,),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        caxis='category',
        title='BraggFresnelFSM2-02c')
    plot.xaxis.limits = [-600, 600]
    plot.yaxis.limits = [-600, 600]
    plot.fluxFormatStr = '%.2e'
    plot.baseName = plot.title
    plots.append(plot)

    plot = xrtp.XYCPlot(
        beamFSM2, (1,),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        title='BraggFresnelFSM2-02E')
    plot.xaxis.limits = [-600, 600]
    plot.yaxis.limits = [-600, 600]
    plot.fluxFormatStr = '%.2e'
    plot.baseName = plot.title
    plots.append(plot)

    plotPos = []
    for i, pos in enumerate(beamLine.fsm2RelPos):
        plot = xrtp.XYCPlot(
            'beamFSM2-{0:02d}'.format(i+1), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
            caxis='category',
            title='BraggFresnelFSM2-Nc{0:02d}'.format(i+1))
        plot.xaxis.limits = [-1, 1]
        plot.yaxis.limits = [-1, 1]
        plot.xaxis.fwhmFormatStr = fwhmFormatStrE
        plot.yaxis.fwhmFormatStr = fwhmFormatStrE
        plot.fluxFormatStr = '%.2e'
        plot.textPanel = plot.ax1dHistX.text(
            0.5, 0.02, '$q$ = {0:.0f} m'.format(pos*1e-3), size=14, color='w',
            transform=plot.ax1dHistX.transAxes, ha='center', va='bottom')
        plot.baseName = plot.title
        plots.append(plot)

        plot = xrtp.XYCPlot(
            'beamFSM2-{0:02d}'.format(i+1), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
            yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
            title='BraggFresnelFSM2-NE{0:02d}'.format(i+1))
        plot.xaxis.limits = [-1, 1]
        plot.yaxis.limits = [-1, 1]
        plot.xaxis.fwhmFormatStr = fwhmFormatStrE
        plot.yaxis.fwhmFormatStr = fwhmFormatStrE
        plot.fluxFormatStr = '%.2e'
        plot.textPanel = plot.ax1dHistX.text(
            0.5, 0.02, '$q$ = {0:.0f} m'.format(pos*1e-3), size=14, color='w',
            transform=plot.ax1dHistX.transAxes, ha='center', va='bottom')
        plot.baseName = plot.title
        plots.append(plot)
        plotPos.append(plot)

    for plot in plots:
        plot.invertColorMap = True
        plot.negative = True
        plot.saveName = plot.baseName + '.png'
#        plot.persistentName = plot.baseName + '.pickle'
    return plots, plotPos


def afterScript(plotPos):
    xrtr.normalize_sibling_plots(plotPos)


def main():
    beamLine = build_beamline()
    plots, plotPos = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=60, beamLine=beamLine,
                         processes='all', afterScript=afterScript,
                         afterScriptArgs=[plotPos])

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

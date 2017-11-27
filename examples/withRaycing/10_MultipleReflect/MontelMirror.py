# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
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

showIn3D = True

mGold = rm.Material('Au', rho=19.3)

E0 = 9000.
L = 1200.
W = 10.
gap = 0.2
pitchVFM = 4e-3
pVFM = 20000.
qVFM = 400000000.
pitchHFM = 4e-3
pHFM = 20000.
qHFM = 400000000.
p = pVFM
q = 2000

#Select a case:
#case = 'parabolic'
case = 'elliptical'


def build_beamline(nrays=raycing.nrays):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, dx=0., dz=0., dxprime=2e-4, dzprime=1e-4,
        distE='lines', energies=(E0,), polarization='horizontal')

    if case == 'parabolic':
        mirrorVFM = roe.BentFlatMirror
        mirrorHFM = roe.BentFlatMirror
        RVFM = 2 * p / np.sin(pitchVFM)
        kwargsVFM = {'R': RVFM}
        RHFM = 2 * p / np.sin(pitchHFM)
        kwargsHFM = {'R': RHFM}
    elif case == 'elliptical':
        mirrorVFM = roe.EllipticalMirrorParam
        mirrorHFM = roe.EllipticalMirrorParam
        kwargsVFM = {'p': pVFM, 'q': qVFM, 'isCylindrical': True}
        kwargsHFM = {'p': pHFM, 'q': qHFM, 'isCylindrical': True}
    else:
        raise

    beamLine.VFM = mirrorVFM(
        beamLine, 'VFM', [0, p, 0], material=mGold,
        limPhysX=[gap/2, W], limPhysY=[-L/2, L/2], rotationSequence='RxRyRz',
        pitch=pitchVFM, yaw=-pitchHFM, **kwargsVFM)
    beamLine.HFM = mirrorHFM(
        beamLine, 'HFM', [0, p, 0], material=mGold,
        limPhysX=[-W, -gap/2], limPhysY=[-L/2, L/2], rotationSequence='RyRzRx',
        positionRoll=np.pi/2, pitch=pitchHFM, yaw=pitchVFM, **kwargsHFM)
    beamLine.fsmMontel = rsc.Screen(beamLine, 'FSM-Montel', (0, p+q, 0))
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    beamSource.nRefl = np.ones_like(beamSource.x)
    beamFSM1 = beamLine.fsmMontel.expose(beamSource)

    beamVFMGlobal1, beamVFMLocal = beamLine.VFM.reflect(beamSource)
    beamVFMLocal.nRefl = (beamVFMLocal.state == 1).astype(int)
    beamHFMGlobal1, beamHFMLocal = beamLine.HFM.reflect(beamVFMGlobal1)
    beamHFMLocal.nRefl = (beamHFMLocal.state == 1).astype(int) * 2
    beamHFMGlobal1.nRefl = np.array(beamVFMLocal.nRefl)
    beamHFMGlobal1.nRefl[beamHFMLocal.nRefl > 0] = 2
    beamHFMGlobal1.state[beamHFMGlobal1.nRefl > 0] = 1

    beamHFMGlobal2, beamHFMLocal2 = beamLine.HFM.reflect(beamSource)
    beamHFMLocal2.nRefl = (beamHFMLocal2.state == 1).astype(int)
    beamVFMGlobal2, beamVFMLocal2 = beamLine.VFM.reflect(beamHFMGlobal2)
    beamVFMLocal2.nRefl = (beamVFMLocal2.state == 1).astype(int) * 2
    beamVFMGlobal2.nRefl = np.array(beamHFMLocal2.nRefl)
    beamVFMGlobal2.nRefl[beamVFMLocal2.nRefl > 0] = 2
    beamVFMGlobal2.state[beamVFMGlobal2.nRefl > 0] = 1

    beamVFMLocal.replace_by_index(beamVFMLocal2.state == 1, beamVFMLocal2)
    beamHFMLocal.replace_by_index(beamHFMLocal2.state == 1, beamHFMLocal2)

    beamMontelGlobal = beamHFMGlobal1
    beamMontelGlobal.replace_by_index(beamVFMGlobal2.nRefl > 0, beamVFMGlobal2)

    beamFSM2 = beamLine.fsmMontel.expose(beamMontelGlobal)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamVFMLocal': beamVFMLocal,
               'beamHFMLocal': beamHFMLocal,
               'beamMontelGlobal': beamMontelGlobal,
               'beamFSM2': beamFSM2}
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
#    fwhmFormatStrE = '%.2f'
    plots = []
    pAdd = case[:3]

    plot = xrtp.XYCPlot(
        'beamFSM1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=128, limits=[-10, 20]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=128, limits=[-10, 20]),
        title='FSM1_E')
    plot.saveName = ['Montel_{0}_exit_no_mirror.png'.format(pAdd), ]
    plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamVFMLocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=128, limits=[-2, W+2]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=128, limits='sym'),
        caxis=xrtp.XYCAxis('number of reflections', '', bins=32, ppb=8,
                           data=raycing.get_reflection_number))
    plot.caxis.limits = [-0.1, 2.1]
    plot.saveName = ['Montel_{0}_localVFM_n.png'.format(pAdd), ]
    plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamHFMLocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=128, limits=[-W-2, 2]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=128, limits='sym'),
        caxis=xrtp.XYCAxis('number of reflections', '', bins=32, ppb=8,
                           data=raycing.get_reflection_number))
    plot.caxis.limits = [-0.1, 2.1]
    plot.saveName = ['Montel_{0}_localHFM_n.png'.format(pAdd), ]
    plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamFSM2', (1, 3, -1, -2),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=128, limits=[-10, 20]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=128, limits=[-10, 20]),
        caxis='category', title='FSM2_Es')
    plot.saveName = ['Montel_{0}_exit_cat.png'.format(pAdd), ]
    plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamFSM2', (1, 3, -1),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=128, limits=[-10, 20]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=128, limits=[-10, 20]),
        caxis=xrtp.XYCAxis('number of reflections', '', bins=32, ppb=8,
                           data=raycing.get_reflection_number),
        title='FSM2_Es')
    plot.caxis.limits = [-0.1, 2.1]
    plot.saveName = ['Montel_{0}_exit_n.png'.format(pAdd), ]
    plots.append(plot)

    for plot in plots:
        plot.xaxis.fwhmFormatStr = None
        plot.yaxis.fwhmFormatStr = None
        plot.caxis.fwhmFormatStr = None
    return plots


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=[300, 3, 300], centerAt='VFM')
        return
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=40, beamLine=beamLine,
                         processes='half')


if __name__ == '__main__':
    main()

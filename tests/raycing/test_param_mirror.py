# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "20 Sep 2021"
import sys
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc

E0, dE = 9000., 5.,
p = 10000.
q = p/2.
pitch = 2e-3
lim = [-0.5, 0.5]
limZoom = [-1, 1]

inclination = 0  # pitch of the source
#inclination = 2.5e-3

globalRoll = 0
# globalRoll = np.pi/2
# globalRoll = np.pi/4

case = 'elliptical'
# case = 'parabolical'


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)
    sourceCenter = [0, 0, 0]
    mirrorCenter = [0, p, p*np.tan(inclination)]

    kw = dict(
        nrays=nrays, distE='flat', energies=(E0-dE, E0+dE),
        polarization='horizontal', pitch=inclination)
    if case == 'elliptical':  # point source
        kw.update(dict(
            dx=0, dz=0, distxprime='flat', dxprime=1e-4,
            distzprime='flat', dzprime=1e-4))
        Mirror = roe.EllipticalMirrorParam
        kwMirror = dict(f1=sourceCenter, q=q)
    elif case == 'parabolical':  # collimated source
        kw.update(dict(
            dx=1, dz=1, distx='flat', distz='flat',
            distxprime=None, distzprime=None))
        Mirror = roe.ParabolicalMirrorParam
        dqs = q * np.sin(2*pitch+inclination)
        dqc = q * np.cos(2*pitch+inclination)
        kwMirror = dict(f2=[mirrorCenter[0] + dqs*np.sin(globalRoll),
                            mirrorCenter[1] + dqc,
                            mirrorCenter[2] + dqs*np.cos(globalRoll)])

    rs.GeometricSource(
        beamLine, 'GeometricSource', sourceCenter, **kw)
    beamLine.fsm1 = rsc.Screen(beamLine, 'beforeMirror', mirrorCenter)
    beamLine.ellMirror = Mirror(
        beamLine, 'EllM', mirrorCenter, rotationSequence='RyRzRx',
        pitch=pitch+inclination*np.cos(globalRoll), positionRoll=globalRoll,
        yaw=inclination*np.sin(globalRoll), **kwMirror)

    beamLine.fsm2 = rsc.Screen(beamLine, '@focus', [0, 0, 0])
    beamLine.fsm2dY = np.linspace(-2, 2, 5) * q*0.1
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
    xprime = beamSource.a / beamSource.b
    zprime = beamSource.c / beamSource.b
    print(xprime.max()-xprime.min(), zprime.max()-zprime.min())
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamEMglobal, beamEMlocal = beamLine.ellMirror.reflect(beamSource)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamEMglobal': beamEMglobal,
               'beamEMlocal': beamEMlocal}
    for i, dy in enumerate(beamLine.fsm2dY):
        dqs = (q+dy) * np.sin(2*pitch+inclination)
        dqc = (q+dy) * np.cos(2*pitch+inclination)
        beamLine.fsm2.center[0] = beamLine.ellMirror.center[0] +\
            dqs*np.sin(globalRoll)
        beamLine.fsm2.center[1] = beamLine.ellMirror.center[1] + dqc
        beamLine.fsm2.center[2] = beamLine.ellMirror.center[2] +\
            dqs*np.cos(globalRoll)
        beamFSM2 = beamLine.fsm2.expose(beamEMglobal)
        outDict['beamFSM2-{0:d}'.format(i+1)] = beamFSM2

    return outDict
rr.run_process = run_process


def main():
    beamLine = build_beamline()
    fwhmFormatStrE = '%.2f'
    plots = []

    plot = xrtp.XYCPlot('beamFSM1', caxis='category')
    plots.append(plot)

    for i, dy in enumerate(beamLine.fsm2dY):
        if i == len(beamLine.fsm2dY) // 2:
            limits, unit = limZoom, 'fm'
        else:
            limits, unit = lim, u'mm'
        xaxis = xrtp.XYCAxis(r'$x$', unit, limits=limits)
        yaxis = xrtp.XYCAxis(r'$z$', unit, limits=limits)
        plot = xrtp.XYCPlot('beamFSM2-{0:d}'.format(i+1),
                            xaxis=xaxis, yaxis=yaxis, caxis='category')
        plots.append(plot)

    for plot in plots:
        plot.caxis.fwhmFormatStr = fwhmFormatStrE
        plot.xaxis.fwhmFormatStr = fwhmFormatStrE
        plot.yaxis.fwhmFormatStr = fwhmFormatStrE
        plot.fluxFormatStr = '%.2e'
        if globalRoll == 0:
            globalRollTxt = '0'
        elif globalRoll == np.pi/2:
            globalRollTxt = u'0.5π'
        elif globalRoll == np.pi/4:
            globalRollTxt = u'0.25π'
        elif globalRoll == np.pi:
            globalRollTxt = u'π'
        else:
            globalRollTxt = '{0}'.format(globalRoll)
        plot.saveName = \
            ['{0}-roll={1}-{2}.png'.format(case, globalRollTxt, plot.title)]

    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)


if __name__ == '__main__':
    main()

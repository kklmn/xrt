# -*- coding: utf-8 -*-
"""
Tests of parametric mirrors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parametric mirrors are exemplified here:

+---------------+---------------+
|  |param1txt|  |  |param1img|  |
+---------------+---------------+
|  |param2txt|  |  |param2img|  |
+---------------+---------------+
|  |param3txt|  |  |param3img|  |
+---------------+---------------+
|  |param4txt|  |  |param4img|  |
+---------------+---------------+

.. |param1txt| replace:: 1. Elliptical. A point source is at focus f1 and
   a screen at f2.
.. |param2txt| replace:: 2. Parabolical. A collimated (parallel) source of
   a finite square cross-section illuminates the mirror and a screen is at
   the paraboloid focus.
.. |param3txt| replace:: 3. Hyperbolic with outer reflection. A point source is
   at focus f1 (the farther of the two foci) and a screen at f2 that collects
   the imaginary (back-projected) beam reflected by the mirror.
.. |param4txt| replace:: 4. Hyperbolic with inner reflection. A point source is
   at focus f1 (the closer of the two foci) and a screen at f2 that collects
   the imaginary (back-projected) beam reflected by the mirror.

.. |param1img| imagezoom:: _images/parametric_test_elliptical
   :loc: upper-right-corner
.. |param2img| imagezoom:: _images/parametric_test_parabolical
   :loc: upper-right-corner
.. |param3img| imagezoom:: _images/parametric_test_hyperbolic
   :loc: upper-right-corner
.. |param4img| imagezoom:: _images/parametric_test_hyperbolic-inner
   :loc: lower-right-corner

The observation screen has several positions around the focus point along the
beam direction to demonstrate the focusing function. Note the femtometer (fm)
axis unit for the plot right at the focus position.
"""
__author__ = "Konstantin Klementiev"
__date__ = "24 Aug 2026"

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
raycing.zEps = 1e-16  # mm

showIn3D = False

E0, dE = 9000., 5.,
p = 10000.
pitch = 2e-3
lim = [-25, 25]
limZoom = [-1, 1]

inclination = 0  # pitch of the source
# inclination = 2.5e-3

globalRoll = 0
# globalRoll = np.pi/2
# globalRoll = np.pi/4

case = 'elliptical'
# case = 'parabolical'
# case = 'hyperbolic'
# case = 'hyperbolic-inner'


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)
    sourceCenter = [0, 0, 0]
    mirrorCenter = [0, p, p*np.tan(inclination)]

    kw = dict(
        nrays=nrays, distE='flat', energies=(E0-dE, E0+dE),
        polarization='horizontal', pitch=inclination)
    if case == 'elliptical':  # point source
        q = p/2.
        kw.update(dict(
            dx=0, dz=0, distxprime='flat', dxprime=1e-4,
            distzprime='flat', dzprime=1e-4))
        Mirror = roe.EllipticalMirrorParam
        kwMirror = dict(f1=sourceCenter, q=q)
    elif case == 'parabolical':  # collimated source
        q = p/2.
        kw.update(dict(
            dx=1, dz=1, distx='flat', distz='flat',
            distxprime=None, distzprime=None))
        Mirror = roe.ParabolicalMirrorParam
        dqs = q * np.sin(2*pitch+inclination)
        dqc = q * np.cos(2*pitch+inclination)
        kwMirror = dict(f2=[mirrorCenter[0] + dqs*np.sin(globalRoll),
                            mirrorCenter[1] + dqc,
                            mirrorCenter[2] + dqs*np.cos(globalRoll)])
    elif case.startswith('hyperbolic'):  # point source, imaginary focus
        q = p*2 if case == 'hyperbolic-inner' else p*0.5
        kw.update(dict(
            dx=0, dz=0, distxprime='flat', dxprime=1e-4,
            distzprime='flat', dzprime=1e-4))
        Mirror = roe.HyperbolicMirrorParam
        # the two definitions of kwMirror are equivalent:
        kwMirror = dict(f1=sourceCenter, q=q)
        # kwMirror = dict(f1=sourceCenter,
        #                 f2=[0, p-q*np.cos(2*pitch), -q*np.sin(2*pitch)])
    else:
        raise ValueError('Unknown mirror')

    rs.GeometricSource(
        beamLine, 'GeometricSource', sourceCenter, **kw)
    beamLine.fsm1 = rsc.Screen(beamLine, 'beforeMirror', mirrorCenter)
    beamLine.mirror = Mirror(
        beamLine, 'M1', mirrorCenter, rotationSequence='RyRzRx',
        pitch=pitch+inclination*np.cos(globalRoll), positionRoll=globalRoll,
        yaw=inclination*np.sin(globalRoll), **kwMirror)
    if case == 'hyperbolic-inner':
        beamLine.mirror.invertNormal = 1  # the inner surface is reflective

    if case == 'elliptical':
        print('ellipseA', beamLine.mirror.ellipseA,
              'ellipseB', beamLine.mirror.ellipseB)
    elif case.startswith('hyperbolic'):
        print('hyperbolaA', beamLine.mirror.hyperbolaA,
              'hyperbolaB', beamLine.mirror.hyperbolaB)

    # The screen beamLine.fsm2 will be placed at the focus of ellipse, parabola
    # or hyperbola plus a few positions up- and downstream
    screenName = 'ImaginaryFocus' if case.startswith('hyperbolic') else 'Focus'
    beamLine.fsm2 = rsc.Screen(beamLine, screenName, [0, 0, 0])
    if showIn3D:
        beamLine.screenDY = [0]
    else:
        beamLine.screenDY = np.linspace(-200, 200, 5)  # pos around focus
    qsign = -1 if case.startswith('hyperbolic') else 1
    beamLine.screen3D = []
    for i, dy in enumerate(beamLine.screenDY):
        dqs = qsign * (q+dy) * np.sin(2*pitch+inclination)
        dqc = qsign * (q+dy) * np.cos(2*pitch+inclination)
        beamLine.screen3D.append(
            [beamLine.mirror.center[0] + dqs*np.sin(globalRoll),
             beamLine.mirror.center[1] + dqc,
             beamLine.mirror.center[2] + dqs*np.cos(globalRoll)])

    if case.startswith('hyperbolic'):  # add a screen for the real reflection
        dqs = (q*0.1) * np.sin(2*pitch+inclination)
        dqc = (q*0.1) * np.cos(2*pitch+inclination)
        screenCenter = [beamLine.mirror.center[0] + dqs*np.sin(globalRoll),
                        beamLine.mirror.center[1] + dqc,
                        beamLine.mirror.center[2] + dqs*np.cos(globalRoll)]
        beamLine.fsm3 = rsc.Screen(
            beamLine, 'after M1 real reflection', screenCenter)

    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    # xprime = beamSource.a / beamSource.b
    # zprime = beamSource.c / beamSource.b
    # print(xprime.max()-xprime.min(), zprime.max()-zprime.min())
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamMglobal, beamMlocal = beamLine.mirror.reflect(beamSource)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamMglobal': beamMglobal, 'beamMlocal': beamMlocal}

    if case.startswith('hyperbolic'):
        beamFSM3 = beamLine.fsm3.expose(beamMglobal)
        outDict['beamFSM3'] = beamFSM3

    for i, pos in enumerate(beamLine.screen3D):
        beamLine.fsm2.center = pos
        beamFSM2 = beamLine.fsm2.expose(beamMglobal)
        outDict['beamFSM2-{0:d}'.format(i)] = beamFSM2

    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow()
        return

    fwhmFormatStrE = '%.2f'
    plots = []

    plot = xrtp.XYCPlot('beamFSM1', caxis='category')
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$')
    yaxis = xrtp.XYCAxis(r'$y$')
    plot = xrtp.XYCPlot('beamMlocal', aspect='auto', xaxis=xaxis, yaxis=yaxis)
    plots.append(plot)

    for i, (pos, dy) in enumerate(zip(beamLine.screen3D, beamLine.screenDY)):
        if i == len(beamLine.screen3D) // 2:
            limits, unit = limZoom, 'fm'
        else:
            limits, unit = lim, u'µm'
        xaxis = xrtp.XYCAxis(r'$x$', unit, limits=limits)
        yaxis = xrtp.XYCAxis(r'$z$', unit, limits=limits)
        plot = xrtp.XYCPlot('beamFSM2-{0:d}'.format(i),
                            xaxis=xaxis, yaxis=yaxis, caxis='category')
        plot.textPanel = plot.fig.text(
            0.72, 0.85, '', transform=plot.fig.transFigure, size=10, color='r',
            ha='left')
        plot.textPanel.set_text(f'dy = \n{dy:.0f} mm')
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

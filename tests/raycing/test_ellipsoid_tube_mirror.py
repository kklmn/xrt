# -*- coding: utf-8 -*-
u"""
This test script exemplifies two implementations of elliptical mirror:
EllipticalMirrorParam and EllipsoidCapillaryMirror (pick out the wanted class
below). The former one can be open or closed (a complete surface of revolution)
while the latter one is always assumed closed.
"""
__author__ = "Konstantin Klementiev"
__date__ = "2 Aug 2024"
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

showIn3D = False

E0, dE = 9000., 5.,
p = 20000.
q = p/2
tubeLengtgh = 10000.
tubeRadius = 50.

Mirror = roe.EllipticalMirrorParam
# Mirror = roe.EllipsoidCapillaryMirror


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)

    sourceCenter = [0., 0., 0.]
    thetaMin = tubeRadius / (p + tubeLengtgh*0.5)
    thetaMax = tubeRadius / (p - tubeLengtgh*0.5)
    rs.GeometricSource(
        beamLine, 'Geometric Source', sourceCenter, nrays=nrays,
        distE='flat', energies=(E0-dE, E0+dE), polarization='horizontal',
        dx=0, dz=0, dy=0, distxprime='annulus', dxprime=(thetaMin, thetaMax))

    focalPoint = [0, p+q, 0]
    if Mirror is roe.EllipticalMirrorParam:
        mirrorCenter = [0, p, -tubeRadius]
        theta1 = np.arctan2(tubeRadius, p)
        theta2 = np.arctan2(tubeRadius, q)
        theta = (theta2-theta1) / 2
        beamLine.ellMirror = Mirror(
            beamLine, 'Ellipsoid Tube Mirror', mirrorCenter, isClosed=True,
            f1=sourceCenter, f2=focalPoint, pitch=theta, isCylindrical=False,
            limPhysY=[-tubeLengtgh/2, tubeLengtgh/2])
    elif Mirror is roe.EllipsoidCapillaryMirror:
        mirrorCenter = [0, p, 0]
        a = ((p**2 + tubeRadius**2)**0.5 + (q**2 + tubeRadius**2)**0.5) / 2
        c = (p+q) / 2
        b = (a**2 - c**2)**0.5
        beamLine.ellMirror = Mirror(
            beamLine, 'Ellipsoid Capillary Mirror', mirrorCenter,
            ellipseA=a, ellipseB=b, workingDistance=q-tubeLengtgh/2,
            limPhysY=[-tubeLengtgh/2, tubeLengtgh/2])
    else:
        raise ValueError('Unknown mirror class')

    print('ellipse a', beamLine.ellMirror.ellipseA)
    print('ellipse b', beamLine.ellMirror.ellipseB)
    # ellipse a 15000.093749560554
    # ellipse b 53.032967157584295

    beamLine.screen = rsc.Screen(beamLine, 'Movable Screen', focalPoint)
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    if showIn3D:
        beamSource = beamLine.sources[0].shine()
        beamMirrorGlobal, beamMirrorlocal = beamLine.ellMirror.reflect(
            beamSource)
        beamScreenAfterReflection = beamLine.screen.expose(beamMirrorGlobal)
        beamScreenAtFocus = beamLine.screen.expose(beamMirrorGlobal)
        outDict = {
            'beamSource': beamSource,
            'beamMirrorlocal': beamMirrorlocal,
            'beamScreenAfterReflection': beamScreenAfterReflection,
            'beamScreenAtFocus': beamScreenAtFocus,
            }
        beamLine.prepare_flow()
    else:
        beamSource = beamLine.sources[0].shine()

        beamLine.screen.center = [0, p, 0]
        beamScreenBeforeReflection = beamLine.screen.expose(beamSource)

        beamMirrorGlobal, beamMirrorlocal = beamLine.ellMirror.reflect(
            beamSource)
        beamScreenAfterReflection = beamLine.screen.expose(beamMirrorGlobal)

        beamLine.screen.center = [0, p+q-10, 0]
        beamScreenBeforeFocus = beamLine.screen.expose(beamMirrorGlobal)

        beamLine.screen.center = [0, p+q, 0]
        beamScreenAtFocus = beamLine.screen.expose(beamMirrorGlobal)

        outDict = {
            'beamSource': beamSource,
            'beamScreenBeforeReflection': beamScreenBeforeReflection,
            'beamMirrorlocal': beamMirrorlocal,
            'beamScreenAfterReflection': beamScreenAfterReflection,
            'beamScreenBeforeFocus': beamScreenBeforeFocus,
            'beamScreenAtFocus': beamScreenAtFocus,
            }
    return outDict
rr.run_process = run_process


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow()
        return

    plots = []

    plot = xrtp.XYCPlot('beamScreenBeforeReflection')
    plots.append(plot)

    plot = xrtp.XYCPlot('beamScreenAfterReflection')
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$s$', 'mm')
    yaxis = xrtp.XYCAxis(r'$\phi$', 'rad')
    plot = xrtp.XYCPlot('beamMirrorlocal', aspect='auto',
                        xaxis=xaxis, yaxis=yaxis)
    plots.append(plot)

    plot = xrtp.XYCPlot('beamScreenBeforeFocus')
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'fm')
    yaxis = xrtp.XYCAxis(r'$z$', 'fm')
    plot = xrtp.XYCPlot('beamScreenAtFocus', xaxis=xaxis, yaxis=yaxis)
    plots.append(plot)

    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)


if __name__ == '__main__':
    main()

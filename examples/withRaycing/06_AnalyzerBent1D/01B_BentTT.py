# -*- coding: utf-8 -*-
r"""
This example is a study of a crystal analyzer. The inner loop in the generator
``plot_generator()`` changes the source type. After a flat-energy source has
been ray-traced (after ``yield``), the widths of *z* and energy distributions
are saved. Then a single line source is ray-traced and provides the width of
*z* distribution. From these 3 numbers we calculate energy resolution and, as a
check, ray-trace a third source with 7 energy lines with a spacing equal to the
previously calculated energy resolution. The source sizes, axis limits, the
number of iterations etc. were determined experimentally and are defined in the
upper part of the script."""
__author__ = "Konstantin Klementiev"
__date__ = "2 Jul 2023"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

crystalMaterial = 'Si'
R = 500.  # mm
isJohansson = True
if isJohansson:
    Rm = R
else:
    Rm = 2. * R
useTT = False  # use elastic deformation in crystal reflectivity
if showIn3D:
    useTT = False

# crystal = rm.CrystalSi(hkl=(8, 8, 0), t=0.35, useTT=useTT)
crystal = rm.CrystalSi(hkl=(10, 10, 0), t=0.35, useTT=useTT)
# crystal = rm.CrystalSi(hkl=(12, 12, 0), t=0.35, useTT=useTT)

beamV = 0.1/2.35
beamH = 0.1/2.35
dxCrystal = 40.
dyCrystal = 300.

if not isJohansson:  # Johann
    bentName = '1D-01b'
    Cylinder = roe.JohannCylinder
    analyzerName = 'JohannAnalyzer'
    eAxisFlatLimits = [-12, 3]
else:  # Johansson
    bentName = '1D-02gb'
    Cylinder = roe.JohanssonCylinder
    analyzerName = 'JohanssonAnalyzer'

    # eAxisFlatLimits = [-3, 9]  # 8, 8, 0
    eAxisFlatLimits = [-2, 6]  # 10, 10, 0
    # eAxisFlatLimits = [-1.5, 4.5]  # 12, 12, 0

yAxisFlatLimits = [-120, 120]
yAxisLineLimits = [-80, 80]

# thetaDegree = 54.
E0 = 20000.
alphaDegree = 0.

numiter = 100
nprocesses = 4


def build_beamline(nrays=1e6):
    beamLine = raycing.BeamLine(azimuth=0, height=0)
    beamLine.source = rs.GeometricSource(
        beamLine, 'GeometricSource', nrays=nrays, dx=beamH, dy=0,
        dz=0.05, distxprime='flat', distzprime='flat', polarization=None)
    beamLine.analyzer = Cylinder(
        beamLine, analyzerName, surface=('',), Rm=Rm,
        limPhysX=(-dxCrystal/2, dxCrystal/2),
        limPhysY=(-dyCrystal/2, dyCrystal/2),
        targetOpenCL='auto' if useTT else None, precisionOpenCL='float32',
        )
    beamLine.detector = rsc.Screen(beamLine, 'Detector')
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine()
    beamAnalyzerGlobal, beamAnalyzerLocal = \
        beamLine.analyzer.reflect(beamSource)
    beamDetector = beamLine.detector.expose(beamAnalyzerGlobal)
    outDict = {'beamSource': beamSource,
               'beamAnalyzerGlobal': beamAnalyzerGlobal,
               'beamAnalyzerLocal': beamAnalyzerLocal,
               'beamDetector': beamDetector
               }
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []
    plotsAnalyzer = []
    plotsDetector = []
    plotsE = []
    limXCrystal = -dxCrystal/2 - 10, dxCrystal/2 + 10
    limYCrystal = -dyCrystal/2 - 10, dyCrystal/2 + 10
    limXDetector = -(dxCrystal + 5), dxCrystal + 5

    plotAnE = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limXCrystal, bins=400, ppb=1),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limYCrystal, bins=400, ppb=1),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f',
                           bins=200, ppb=2),
        title='xtal_E', oe=beamLine.analyzer)
    plotAnE.caxis.fwhmFormatStr = fwhmFormatStrE
    # plotAnE.caxis.invertAxis = True
    plotAnE.textPanel = plotAnE.fig.text(
        0.88, 0.85, '', transform=plotAnE.fig.transFigure, size=12, color='r',
        ha='center')
    plotsAnalyzer.append(plotAnE)
    plotsE.append(plotAnE)

    plotDetE = xrtp.XYCPlot(
        'beamDetector', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limXDetector,
                           fwhmFormatStr='%.2f'),
        yaxis=xrtp.XYCAxis(r'$z$', 'µm', limits=[-100, 100],
                           fwhmFormatStr='%.2f'),
        title='det_E')
    plotDetE.caxis.fwhmFormatStr = fwhmFormatStrE
    # plotDetE.caxis.invertAxis = True
    plotDetE.textPanel = plotDetE.fig.text(
        0.88, 0.8, '', transform=plotDetE.fig.transFigure, size=12, color='r',
        ha='center')
    plotsDetector.append(plotDetE)
    plotsE.append(plotDetE)

    for plot in plotsAnalyzer:
        plots.append(plot)
    for plot in plotsDetector:
        plots.append(plot)
    return plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE, plotDetE


def plot_generator(beamLine, plots=[], plotsAnalyzer=[], plotsDetector=[],
                   plotsE=[], plotAnE=[], plotDetE=[]):
    beamLine.source.dz = beamV
    bsname = 'h={0:03.0f}mum'.format(beamV*1e3)

    if np.any(np.array(crystal.hkl) > 9):
        hklSeparator = ','
    else:
        hklSeparator = ''
    crystalLabel = '{0}{1[0]}{2}{1[1]}{2}{1[2]}'.format(
        crystalMaterial, crystal.hkl, hklSeparator)
    beamLine.analyzer.surface = crystalLabel,
    if plotAnE:
        plotAnE.draw_footprint_area()
        # plotAnC.draw_footprint_area()
    beamLine.analyzer.material = crystal

    # theta = np.radians(thetaDegree)
    # sinTheta = np.sin(theta)
    # E0raw = rm.ch / (2 * crystal.d * sinTheta)
    # dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
    # E0 = rm.ch / (2 * crystal.d * np.sin(theta + dTheta))
    theta = crystal.get_Bragg_angle(E0)-crystal.get_dtheta_symmetric_Bragg(E0)
    thetaDegree = np.degrees(theta)
    offsetE = round(E0, 2)
    for plot in plotsE:
        if plot is None:
            continue
        plot.caxis.offset = offsetE

    alpha = np.radians(alphaDegree)
    beamLine.analyzer.alpha = alpha
    p = 2. * R * np.sin(theta + alpha)
    q = 2. * R * np.sin(theta - alpha)
    sin2Theta = np.sin(2 * theta)
    cos2Theta = np.cos(2 * theta)
    yDet = p + q * cos2Theta
    zDet = q * sin2Theta
    pdp = 2. * R * np.sin(theta + alpha - dyCrystal/6/R)
    beamLine.source.dxprime = dxCrystal / pdp
    beamLine.source.dzprime = dyCrystal * np.sin(theta - alpha) / pdp
    beamLine.analyzer.center = 0, p, 0
    beamLine.analyzer.pitch = theta + alpha
    beamLine.detector.center = 0, yDet, zDet
    beamLine.detector.set_orientation(z=(0, -sin2Theta, cos2Theta))

    tt = r'{0}{1}$\theta = {2:.3f}^\circ$'.format(
        crystalLabel, '\n', thetaDegree)
    for plot in plots:
        if hasattr(plot, 'textPanel'):
            plot.textPanel.set_text(tt)

    dELine, dzLine, dEFlat, dzFlat = 0, 0, 0, 1
    for isource in [0, 1, 2]:
        xrtr.set_repeats(numiter)
        for plot in plotsDetector:
            plot.yaxis.limits = yAxisFlatLimits
        if isource == 0:  # flat
            eAxisMin = offsetE + eAxisFlatLimits[0]
            eAxisMax = offsetE + eAxisFlatLimits[1]
            for plot in plotsE:
                if plot is None:
                    continue
                plot.caxis.limits = [eAxisMin, eAxisMax]
            beamLine.source.distE = 'flat'
            beamLine.source.energies = eAxisMin, eAxisMax
            sourcename = 'flat'
        elif isource == 1:  # line
            beamLine.source.distE = 'lines'
            beamLine.source.energies = E0,
            sourcename = 'line'
            for plot in plotsDetector:
                plot.yaxis.limits = yAxisLineLimits
        else:
            tt = (r'{0}{1}$\theta = {2:.2f}^\circ${1}' +
                  r'$\delta E = ${3:.3f} eV').format(
                crystalLabel, '\n', thetaDegree, dELine)
            for plot in plots:
                if hasattr(plot, 'textPanel'):
                    plot.textPanel.set_text(tt)
            beamLine.source.distE = 'lines'
            sourcename = '7lin'
            # for plot in plotsDetector:
            #     plot.yaxis.limits = [-dzLine*4, dzLine*4]
            dEStep = dELine
            beamLine.source.energies = [E0 + dEStep*i for i in range(-3, 4)]
            eAxisMin = E0 - dEStep * 4
            eAxisMax = E0 + dEStep * 4
            for plot in plotsE:
                if plot is None:
                    continue
                plot.caxis.limits = [eAxisMin, eAxisMax]

        for plot in plots:
            filename = '{0}-{1}-{2:.0f}-{3}-{4}'.format(
                bentName, crystalLabel, thetaDegree,
                plot.title, sourcename)
            plot.saveName = filename + '{}.png'.format('_TT' if useTT else '')
            # plot.persistentName = filename + '.pickle'
        if showIn3D:
            beamLine.glowFrameName = \
                '{0}-{1}-{2:.0f}-{3}-{4}.jpg'.format(
                    bentName, crystalLabel, thetaDegree, isource, sourcename)
        yield

        if not showIn3D:
            if isource == 0:
                dzFlat = plotDetE.dy
                dEFlat = plotDetE.dE
                if (dzFlat == 0) or (dEFlat == 0)\
                        or np.isnan(dzFlat) or np.isnan(dEFlat):
                    raise
            elif isource == 1:
                dzLine = plotDetE.dy
                try:
                    dELine = dzLine * dEFlat / dzFlat
                except Exception as e:
                    print(e)
                    print('dzFlat={0}'.format(dzFlat))
                    dELine = 0
                outStr = ('{0}, {1}, {2}, {3}, {4}, {5}, ' +
                          '{6}, {7}, {8}, {9}\n').format(
                    bsname, crystalLabel, thetaDegree,
                    alpha, E0, dzFlat, dEFlat, dzLine,
                    dELine, dELine/E0)
                print(outStr)

    if showIn3D:
        return


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=4, centerAt=analyzerName,
                      generator=plot_generator, generatorArgs=[beamLine])
        return
    plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE, plotDetE =\
        define_plots(beamLine)
    args = [beamLine, plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE,
            plotDetE]
    xrtr.run_ray_tracing(plots, generator=plot_generator, generatorArgs=args,
                         beamLine=beamLine,
                         processes=1 if useTT else nprocesses)


if __name__ == '__main__':
    main()

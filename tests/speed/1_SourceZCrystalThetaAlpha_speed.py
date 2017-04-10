# -*- coding: utf-8 -*-
r"""
Multithreading and multiprocessing in ray tracing
--------------------------------------------------

| Script: ``\tests\speed\1_SourceZCrystalThetaAlpha_speed.py``.
| The test is based on the example ``\examples\withRaycing\07_AnalyzerBent2D\01BD_SourceZCrystalThetaAlpha.py``.

This script calculates diffraction of a 2D-bent diced crystal anayzer from
three types of geometric source.

+-----------------+-----------+-----------------------+-----------------------+
|                 |           |     multithreading    |    multiprocessing    |
|     system      |     1     +-----------+-----------+-----------+-----------+
|                 |           |     2     |     4     |     2     |     4     |
+====+============+===========+===========+===========+===========+===========+
|[1]_|   |win|    |   587.5,  |   296.4,  |   222.3,  |   441.5,  |   295.4,  |
|    |            |   587.8   |   296.6   |   222.5   |   460.3   |   368.4   |
|    +------------+-----------+-----------+-----------+-----------+-----------+
|    |   |lin|    |   294.3,  |   181.9,  |   148.8,  |   160.1,  |   107.9,  |
|    |            |   293.3   |   174.5   |   139.1   |   164.5   |   105.9   |
+----+------------+-----------+-----------+-----------+-----------+-----------+
|[2]_|   |win|    |   359.9,  |   290.8,  |   218.7,  |   362.2,  |   290.8,  |
|    |            |   546.9   |   292.0   |   219.7   |   436.4   |   363.6   |
|    +------------+-----------+-----------+-----------+-----------+-----------+
|    |   |lin|    |   293.3,  |   172.3,  |   169.5,  |   173.3,  |   156.1,  |
|    |            |   293.1   |   173.6   |   168.3   |   172.4   |   154.6   |
+----+------------+-----------+-----------+-----------+-----------+-----------+

.. |win| replace:: Windows 7 64 bit
.. |lin| replace:: Ubuntu 16.04 64 bit

"""

"""
Intel(R) Core(TM) i7-3930K CPU @ 3.20GHz:

Python 2.7.10 64 bit on Windows 7:
th=1: 587.5, th=2: 296.4, th=4: 222.3 pr=2: 441.5, pr=4: 295.4

Python 3.6.1 64 bit on Windows 7:
th=1: 587.8, th=2: 296.6, th=4: 222.5 pr=2: 460.3, pr=4: 368.4

Python 2.7.12 64 bit on Ubuntu 16.04:
th=1: 294.3, th=2: 181.9, th=4: 148.8 pr=2: 160.1, pr=4: 107.9

Python 3.5.2 64 bit on Ubuntu 16.04:
th=1: 293.3, th=2: 174.5, th=4: 139.1 pr=2: 164.5, pr=4: 105.9


Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz:

Python 2.7.10 64 bit on Windows 10:
th=1: 359.9, th=2: 290.8, th=4: 218.7 pr=2: 362.2, pr=4: 290.8

Python 3.6.1 64 bit on Windows 10:
th=1: 546.9, th=2: 292.0, th=4: 219.7 pr=2: 436.4, pr=4: 363.6

Python 2.7.12+ 64 bit on Ubuntu 16.10:
th=1: 293.3, th=2: 172.3, th=4: 169.5 pr=2: 173.3, pr=4: 156.1

Python 3.5.2+ 64 bit on Ubuntu 16.10:
th=1: 293.1, th=2: 173.6, th=4: 168.3 pr=2: 172.4, pr=4: 154.6
"""

r"""
This example is a complete study of a crystal analyzer. The generator
``plot_generator()`` is rather complex and therefore needs some explanation.
The inner loop changes the source type. After a flat-energy source has been
ray-traced (after ``yield``), the widths of *z* and energy distributions are
saved. Then a single line source is ray-traced and provides the width of *z*
distribution. From these 3 numbers we calculate energy resolution and, as a
check, ray-trace a third source with 7 energy lines with a spacing equal to the
previously calculated energy resolution. There are 3 outer loops for various
asymmetry angles, Bragg angles and crystal types. These loops are additionally
controlled by masks (``crystalsMask``, ``thetaMask``) having zeros or ones to
switch off/on the corresponding angles or crystals. The source sizes, axis
limits, number of iterations etc. were determined experimentally and are given
by lists in the upper part of the script. The outputs are the plots and a text
file with the resulted energy resolutions."""

__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import math
import numpy as np
#import matplotlib as mpl

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

crystalMaterial = 'Si'
if crystalMaterial == 'Si':
    d111 = 3.1354161  # Si
elif crystalMaterial == 'Ge':
    d111 = 3.2662725  # Ge
else:
    raise
orders = (1, 4, 8, 12)
crystals = [rm.CrystalDiamond((i, i, i), d111/i) for i in orders]
crystalsMask = (0, 1, 0, 0)  # 0=don't execute; 1=execute
#numiters = [24, 360, 1280, 5120]  # @crysals
numiters = [96, 96, 192, 480]  # @crysals

R = 500.  # mm
isDiced = True
isJohansson = True

if isJohansson:
    Rm = R
else:
    Rm = 2. * R

beamV = 0.07/2.35,
beamH = 0.2/2.35
dxCrystal = 100.
dyCrystal = 100.

if not isDiced:
    xAxisLim = 0.25
    yAxesLim = 0.1,  # @beamV
    facetKWargs = {}
    if not isJohansson:
        bentName = '2D-01b'
        Toroid = roe.JohannToroid
        analyzerName = 'JohannAnalyzer'
        eAxesFlat = [[6.0e-3, 3.2e-3, 3.2e-3, 3.2e-3],
                     [1.2e-3, 9.6e-4, 9.6e-4, 9.6e-4],
                     [8.0e-4, 5.0e-4, 5.0e-4, 5.0e-4],
                     [6.0e-4, 2.0e-4, 2.0e-4, 2.0e-4],
                     [2.0e-4, 5.0e-5, 5.0e-5, 5.0e-5]]
        yAxesLine = [0.08, 0.08, 0.015, 0.006]  # @crysals
    else:  # Johansson
        bentName = '2D-02gb'
        Toroid = roe.JohanssonToroid
        analyzerName = 'JohanssonAnalyzer'
        eAxesFlat = [[3.5e-3, 2.0e-3, 2.0e-3, 1.0e-3],
                     [6.0e-4, 2.8e-4, 3.0e-4, 3.0e-4],
                     [3.5e-4, 1.6e-4, 1.2e-4, 1.0e-4],
                     [2.5e-4, 1.0e-4, 8.0e-5, 6.0e-5],
                     [2.0e-4, 3.0e-5, 3.0e-5, 3.0e-5]]
        yAxesLine = [0.1, 0.05, 0.007, 0.004]  # @crysals
else:  # diced
    xAxisLim = 2.5
    yAxesLim = 1.5,  # @beamV
    facetKWargs = {'dxFacet': 2.1, 'dyFacet': 1.4, 'dxGap': 0.05,
                   'dyGap': 0.05}
    if not isJohansson:
        bentName = '03bd'
        Toroid = roe.DicedJohannToroid
        analyzerName = 'DicedJohannAnalyzer'
        eAxesFlat = [[2.0e-2, 1.9e-2, 1.8e-2, 1.6e-2],
                     [6.0e-3, 6.0e-3, 6.0e-3, 6.0e-3],
                     [3.0e-3, 2.4e-3, 2.4e-3, 2.4e-3],
                     [1.5e-3, 1.4e-3, 1.2e-3, 1.2e-3],
                     [4.0e-4, 3.0e-4, 3.0e-4, 3.0e-4]]
        yAxesLine = [1, 0.5, 0.5, 0.5]  # @crysals
    else:  # Johansson
        bentName = '04gbd'
        Toroid = roe.DicedJohanssonToroid
        analyzerName = 'DicedJohanssonAnalyzer'
        eAxesFlat = [[1.5e-2, 1.0e-2, 1.0e-2, 1.0e-2],
                     [4.0e-3, 3.0e-3, 2.4e-3, 1.6e-3],
                     [2.0e-3, 1.5e-3, 1.2e-3, 1.2e-3],
                     [1.2e-3, 8.0e-4, 6.0e-4, 1.0e-3],
                     [4.0e-4, 2.5e-4, 2.2e-4, 1.6e-4]]
        yAxesLine = [1, 0.1, 0.08, 0.06]  # @crysals

thetasDegree = 10., 30., 45., 60., 80.,  # degree
thetaMask = 0, 0, 0, 1, 0
alphasDegree = 0.,  # degree


def build_beamline(nrays=raycing.nrays):
    beamLine = raycing.BeamLine(azimuth=0, height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', nrays=nrays, dx=beamH, dy=0,
        dz=0.05, distxprime='flat', distzprime='flat', polarization=None)
    beamLine.analyzer = Toroid(
        beamLine, analyzerName, surface=('',),
        limPhysX=(-dxCrystal/2, dxCrystal/2),
        limPhysY=(-dyCrystal/2, dyCrystal/2),
        Rm=Rm, shape='rect', **facetKWargs)
    beamLine.detector = rsc.Screen(beamLine, 'Detector', x=(1, 0, 0))
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    beamAnalyzerGlobal, beamAnalyzerLocal = \
        beamLine.analyzer.reflect(beamSource)
    beamDetector = beamLine.detector.expose(beamAnalyzerGlobal)
    outDict = {'beamSource': beamSource,
               'beamAnalyzerGlobal': beamAnalyzerGlobal,
               'beamAnalyzerLocal': beamAnalyzerLocal,
               'beamDetector': beamDetector
               }
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []
    plotsAnalyzer = []
    plotsDetector = []
    plotsE = []

#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,2,3,-1),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', 1.0, limits=[-52, 52], bins=400,
#                         ppb=1),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', 1.0, limits=[-52, 52], bins=400,
#                         ppb=1),
#      caxis='category', title='xtal_footprint', oe=beamLine.analyzer)
#    plotsAnalyzer.append(plot)

    plotAnE = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-52, 52], bins=400, ppb=1),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-52, 52], bins=400, ppb=1),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f',
                           bins=200, ppb=2),
        title='xtal_E', oe=beamLine.analyzer)
    plotAnE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotAnE.caxis.invertAxis = True
    plotAnE.textPanel = plotAnE.fig.text(
        0.88, 0.85, '', transform=plotAnE.fig.transFigure, size=14, color='r',
        ha='center')
    plotsAnalyzer.append(plotAnE)
    plotsE.append(plotAnE)

#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      caxis=xrtp.XYCAxis('degree of polarization', '', bins=200, ppb=2,
#      data=raycing.get_polarization_degree, limits=[0, 1]),
#      title='xtal_DegOfPol', oe=beamLine.analyzer)
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      caxis=xrtp.XYCAxis('circular polarization rate', '', bins=200, ppb=2,
#      data=raycing.get_circular_polarization_rate, limits=[-1, 1]),
#      title='xtal_CircPol', oe=beamLine.analyzer)
#    plot.textPanel = plot.fig.text(0.88, 0.85, '',
#      transform=plot.fig.transFigure, size=14, color='r', ha='center')
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      caxis=xrtp.XYCAxis('ratio of ellipse axes', '', bins=200, ppb=2,
#      data=raycing.get_ratio_ellipse_axes, limits=[-1, 1]),
#      title='xtal_PolAxes', oe=beamLine.analyzer)
#    plot.textPanel = plot.fig.text(0.88, 0.85, '',
#      transform=plot.fig.transFigure, size=14, color='r', ha='center')
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-52, 52], bins=400, ppb=1),
#      caxis=xrtp.XYCAxis('phase shift', '', bins=200, ppb=2,
#      data=raycing.get_phase_shift, limits=[-1, 1]),#in units of pi!
#      title='xtal_Phase', oe=beamLine.analyzer)
#    plot.textPanel = plot.fig.text(0.88, 0.85, '',
#      transform=plot.fig.transFigure, size=14, color='r', ha='center')
#    formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
#    plot.ax1dHistE.yaxis.set_major_formatter(formatter)
#    plotsAnalyzer.append(plot)
#
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,-1),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
#      caxis='category', title='xtal_zoom', oe=beamLine.analyzer,
#      raycingParam=1000)
#    plotsAnalyzer.append(plot)
#
    plot = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f'),
        title='xtal_E_zoom', oe=beamLine.analyzer, raycingParam=1000)
    plot.caxis.fwhmFormatStr = fwhmFormatStrE
    plot.caxis.invertAxis = True
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plotsAnalyzer.append(plot)
    plotsE.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
#      caxis=xrtp.XYCAxis('degree of polarization', '',
#      data=raycing.get_polarization_degree, limits=[0, 1]),
#      title='xtal_DegOfPol_zoom', oe=beamLine.analyzer, raycingParam=1000)
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
#      caxis=xrtp.XYCAxis('circular polarization rate', '',
#      data=raycing.get_circular_polarization_rate, limits=[-1, 1]),
#      title='xtal_CircPol_zoom', oe=beamLine.analyzer, raycingParam=1000)
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
#      caxis=xrtp.XYCAxis('ratio of ellipse axes', '',
#      data=raycing.get_ratio_ellipse_axes, limits=[-1, 1]),
#      title='xtal_PolAxes_zoom', oe=beamLine.analyzer, raycingParam=1000)
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    plotsAnalyzer.append(plot)
#
#    plot = xrtp.XYCPlot('beamAnalyzerLocal', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1.6, 1.6]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-1.6, 1.6]),
#      caxis=xrtp.XYCAxis('phase shift', '',
#      data=raycing.get_phase_shift, limits=[-1, 1]),#in units of pi!
#      title='xtal_Phase_zoom', oe=beamLine.analyzer, raycingParam=1000)
#    plot.textPanel = plot.fig.text(
#    0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
#    plot.ax1dHistE.yaxis.set_major_formatter(formatter)
#    plotsAnalyzer.append(plot)

    plotDetE = xrtp.XYCPlot(
        'beamDetector', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-xAxisLim, xAxisLim],
                           fwhmFormatStr='%.3f'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.25, 0.25],
                           fwhmFormatStr='%.3f'),
        title='det_E')
    plotDetE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotDetE.caxis.invertAxis = True
    plotDetE.textPanel = plotDetE.fig.text(
        0.88, 0.8, '', transform=plotDetE.fig.transFigure, size=14, color='r',
        ha='center')
    plotsDetector.append(plotDetE)
    plotsE.append(plotDetE)

#    plot = xrtp.XYCPlot('beamDetector', (1,), aspect='auto',
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-xAxisLim, xAxisLim],
#                         fwhmFormatStr='%.3f'),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.25, 0.25],
#                         fwhmFormatStr='%.3f'),
#      caxis=xrtp.XYCAxis('degree of polarization', '',
#                         data=raycing.get_polarization_degree,
#                         limits=[0, 1]), title='det_DegOfPol')
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
#        ha='center')
#    plotsDetector.append(plot)
#
#    plot = xrtp.XYCPlot('beamDetector', (1,), aspect='auto',
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-xAxisLim, xAxisLim],
#                         fwhmFormatStr='%.3f'),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.25, 0.25],
#                         fwhmFormatStr='%.3f'),
#      caxis=xrtp.XYCAxis('circular polarization rate', '',
#                         data=raycing.get_circular_polarization_rate,
#                         limits=[-1, 1]),
#      title='det_CircPol')
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
#        ha='center')
#    plotsDetector.append(plot)
#
#    plot = xrtp.XYCPlot('beamDetector', (1,), aspect='auto',
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-xAxisLim, xAxisLim],
#                         fwhmFormatStr='%.3f'),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.25, 0.25],
#                         fwhmFormatStr='%.3f'),
#      caxis=xrtp.XYCAxis('ratio of ellipse axes', '',
#                         data=raycing.get_ratio_ellipse_axes,
#                         limits=[-1, 1]),
#      title='det_PolAxes')
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
#        ha='center')
#    plotsDetector.append(plot)
#
#    plot = xrtp.XYCPlot('beamDetector', (1,), aspect='auto',
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-xAxisLim, xAxisLim],
#                         fwhmFormatStr='%.3f'),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.25, 0.25],
#                         fwhmFormatStr='%.3f'),
#      caxis=xrtp.XYCAxis('phase shift', '',
#                         data=raycing.get_phase_shift,
#                         limits=[-1, 1]),#in units of pi!
#      title='det_Phase')
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
#        ha='center')
#    formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
#    plot.ax1dHistE.yaxis.set_major_formatter(formatter)
#    plotsDetector.append(plot)

    for plot in plotsAnalyzer:
        plots.append(plot)
    for plot in plotsDetector:
        plots.append(plot)
    return plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE, plotDetE


def plot_generator(plots, plotsAnalyzer, plotsDetector, plotsE,
                   plotAnE, plotDetE, beamLine):
    fOut1 = open(crystalMaterial + bentName + '_long.txt', 'w')
    E0table = [[0 for ic in crystals] for it in thetasDegree]
    dEtable = [[0 for ic in crystals] for it in thetasDegree]
    rEtable = [[0 for ic in crystals] for it in thetasDegree]

    for bvs, yAxisLim in zip(beamV, yAxesLim):
        beamLine.sources[0].dz = bvs
        bsname = 'h={0:03.0f}mum'.format(bvs*1e3)
        for icrystal, crystal in enumerate(crystals):
            if not crystalsMask[icrystal]:
                continue
            numiter = numiters[icrystal]
            yAxisLine = yAxesLine[icrystal]
            if np.any(np.array(crystal.hkl) > 10):
                hklSeparator = ','
            else:
                hklSeparator = ''
            crystalLabel = '{0}{1[0]}{2}{1[1]}{2}{1[2]}'.format(
                crystalMaterial, crystal.hkl, hklSeparator)
            beamLine.analyzer.surface = crystalLabel,
            plotAnE.draw_footprint_area()
            beamLine.analyzer.material = crystal
            for ithetaDegree, thetaDegree in enumerate(thetasDegree):
                if not thetaMask[ithetaDegree]:
                    continue
                theta = np.radians(thetaDegree)
                sinTheta = np.sin(theta)
                E0raw = rm.ch / (2 * crystal.d * sinTheta)
                dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
                E0 = rm.ch / (2 * crystal.d * math.sin(theta + dTheta))
                offsetE = round(E0, 3)
                for alphaDegree in alphasDegree:
                    alpha = np.radians(alphaDegree)
                    beamLine.analyzer.set_alpha(alpha)
#                    b = math.sin(theta - alpha) / math.sin(theta + alpha)
                    p = 2. * R * math.sin(theta + alpha)
                    q = 2. * R * math.sin(theta - alpha)
                    sin2Theta = math.sin(2 * theta)
                    cos2Theta = math.cos(2 * theta)
                    Rs = 2. * R * sinTheta**2
                    yDet = p + q * cos2Theta
                    zDet = q * sin2Theta
                    pdp = 2. * R * math.sin(theta + alpha - dyCrystal/6/R)
                    beamLine.sources[0].dxprime = dxCrystal / pdp
                    beamLine.sources[0].dzprime = dyCrystal * \
                        math.sin(theta + alpha) / pdp
                    beamLine.analyzer.center = 0, p, 0
                    beamLine.analyzer.pitch = theta + alpha
                    beamLine.analyzer.Rs = Rs
                    beamLine.detector.center = 0, yDet, zDet
                    beamLine.detector.set_orientation(
                        z=(0, -sin2Theta, cos2Theta))

                    dELine = 0
                    dzLine = 0
                    for isource in np.arange(3):
                        for plot in plotsDetector:
                            plot.yaxis.limits = [-yAxisLim, yAxisLim]
                        xrtr.runCardVals.repeats = numiter
                        if isource == 0:  # flat
#                            xrtr.runCardVals.repeats = 0
                            eAxisFlat = eAxesFlat[ithetaDegree][icrystal]
                            eAxisMin = E0 * (1. - eAxisFlat)
                            eAxisMax = E0 * (1. + eAxisFlat)
                            for plot in plotsE:
                                if plot is None:
                                    continue
                                plot.caxis.offset = offsetE
                                plot.caxis.limits = [eAxisMin, eAxisMax]
                            tt = r'{0}{1}$\theta = {2:.0f}^\circ$'.format(
                                crystalLabel, '\n', thetaDegree)
                            for plot in plots:
                                try:
                                    plot.textPanel.set_text(tt)
                                except AttributeError:
                                    pass
                            beamLine.sources[0].distE = 'flat'
                            beamLine.sources[0].energies = \
                                eAxisMin, eAxisMax
                            sourcename = 'flat'
                        elif isource == 1:  # line
#                            xrtr.runCardVals.repeats = 0
                            beamLine.sources[0].distE = 'lines'
                            beamLine.sources[0].energies = E0,
                            sourcename = 'line'
                            for plot in plotsDetector:
                                plot.yaxis.limits = [-yAxisLine, yAxisLine]
                        else:
#                            xrtr.runCardVals.repeats = 0
                            tt = (r'{0}{1}$\theta = {2:.0f}^\circ${1}$' +
                                  '\delta E = ${3:.3f} eV').format(
                                crystalLabel, '\n', thetaDegree, dELine)
                            for plot in plots:
                                try:
                                    plot.textPanel.set_text(tt)
                                except AttributeError:
                                    pass
                            beamLine.sources[0].distE = 'lines'
                            sourcename = '7lin'
                            for plot in plotsDetector:
                                plot.yaxis.limits = [-dzLine*4, dzLine*4]
                            dEStep = dELine
                            beamLine.sources[0].energies = \
                                [E0 + dEStep * i for i in range(-3, 4)]
                            eAxisMin = E0 - dEStep * 4
                            eAxisMax = E0 + dEStep * 4
                            for plot in plotsE:
                                if plot is None:
                                    continue
                                plot.caxis.limits = [eAxisMin, eAxisMax]

                        for plot in plots:
                            filename = '{0}-{1}-{2}-{3}-{4}'.format(
                                bentName, crystalLabel, thetaDegree,
                                plot.title, sourcename)
                            plot.saveName = filename + '.png'
#                            plot.persistentName = filename + '.pickle'
                        yield

                        if isource == 0:
                            dzFlat = plotDetE.dy
                            dEFlat = plotDetE.dE
                        elif isource == 1:
                            dzLine = plotDetE.dy
                            try:
                                dELine = dzLine * dEFlat / dzFlat
                            except:
                                print('dzFlat={0}'.format(dzFlat))
                                dELine = 0
                            outStr = ('{0}, {1}, {2}, {3}, {4}, {5}, ' +
                                      '{6}, {7}, {8}\n').format(
                                bsname, crystalLabel, theta, alpha, E0,
                                dzFlat, dEFlat, dzLine, dELine)
                            print(outStr)
                            E0table[ithetaDegree][icrystal] = E0
                            dEtable[ithetaDegree][icrystal] = dELine
                            rEtable[ithetaDegree][icrystal] = dELine/E0
                            fOut1.write(outStr)
    fOut1.close()
    fOut2 = open(crystalMaterial + bentName + '_table.txt', 'w')
    for ithetaDegree, thetaDegree in enumerate(thetasDegree):
        outStr = (r'| {0:2.0f} | *E* = {1[0]:.0f} eV       | *E* =' +
                  r' {1[1]:.0f} eV       | *E* = {1[2]:.0f} eV     ' +
                  r'  | *E* = {1[3]:.0f} eV      |{2}').format(
            thetaDegree, E0table[ithetaDegree], '\n')
        fOut2.write(outStr)
        outStr = (r'|    | {0}={1[0]:.4f}eV={2[0]:.1e} *E* | {0}=' +
                  r'{1[1]:.4f}eV={2[1]:.1e} *E* | {0}={1[2]:.4f}eV=' +
                  r'{2[2]:.1e} *E* | {0}={1[3]:.4f}eV={2[3]:.1e} *E* ' +
                  r'|{3}').format(bentName, dEtable[ithetaDegree],
                                  rEtable[ithetaDegree], '\n')
        fOut2.write(outStr)
    fOut2.close()


def main():
    beamLine = build_beamline()
    plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE, plotDetE =\
        define_plots(beamLine)
    args = [plots, plotsAnalyzer, plotsDetector, plotsE, plotAnE, plotDetE,
            beamLine]
    xrtr.run_ray_tracing(
        plots, generator=plot_generator, generatorArgs=args,
        beamLine=beamLine, updateEvery=4, threads=1, processes=4)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

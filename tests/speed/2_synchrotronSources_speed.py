# -*- coding: utf-8 -*-
r"""
OpenCL performance with Undulator source
----------------------------------------

| Script: ``\tests\speed\2_synchrotronSources_speed.py``.
| The test is based on the example ``\examples\withRaycing\01_SynchrotronSources\synchrotronSources.py``.

This script calculates characteristics of an undulator source at energies
around one harmonic.

+-----------------+---------------+---------------+---------------+
|     system      |   no OpenCL   | OpenCL on CPU | OpenCL on GPU |
+====+============+===============+===============+===============+
|[1]_|   |winW|   |     1471      |      36.0     |      25.7     |
|    |            |     1385      |      34.1     |      23.9     |
|    +------------+---------------+---------------+---------------+
|    |   |linW|   |      950      |      34.6     |      20.6     |
|    |            |      950      |      35.4     |      21.0     |
+----+------------+---------------+---------------+---------------+
|[2]_|   |winH|   |     1801      |      61.0     |      126      |
|    |            |     1909      |      60.3     |      123      |
|    +------------+---------------+---------------+---------------+
|    |   |linH|   |     1245      |      57.6     |      122      |
|    |            |     1255      |      60.2     |      127      |
+----+------------+---------------+---------------+---------------+
|[9]_|   local    |               |               |      30.0     |
|    +------------+---------------+---------------+---------------+
|    |  ZMQ 1Gb   |               |               |      182.9    |
+----+------------+---------------+---------------+---------------+

"""

"""
Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz:

Python 2.7.10 64 bit on Windows 10:
no OpenCL, 1 core of CPU: 1471 s.
OpenCL on CPU: shine =  36.0 s (Intel/Intel)
OpenCL on CPU: shine = 166.3 s (AMD/Intel)
OpenCL on AMD W9100 GPU: 25.7 s (total)

Python 3.6.1 64 bit on Windows 10:
no OpenCL, 1 core of CPU: 1385 s.
OpenCL on CPU: shine =  34.1 s (Intel/Intel)
OpenCL on CPU: shine =  159.7 s (AMD/Intel)
OpenCL on AMD W9100 GPU: 23.9 s (total)

Python 2.7.12 64 bit on Ubuntu 16.04:
no OpenCL, 1 core of CPU: 950.0 s.
OpenCL on CPU: shine = 34.6 s (Intel/Intel)
no (AMD/Intel)
OpenCL on AMD W9100 GPU: 20.6 s (total)

Python 3.5.2 64 bit on Ubuntu 16.04:
no OpenCL, 1 core of CPU: 950.4 s.
OpenCL on CPU: shine = 35.4 s (Intel/Intel)
no (AMD/Intel)
OpenCL on AMD W9100 GPU: 21.0 s (total)


Intel(R) Core(TM) i7-3930K CPU @ 3.20GHz:

Python 2.7.10 64 bit on Windows 7:
no OpenCL, 1 core of CPU: 1764 s.
OpenCL on CPU: shine = 197.5 s
OpenCL on AMD W9100 GPU: 36.1 s (total)

Python 3.6.1 64 bit on Windows 7:
no OpenCL, 1 core of CPU: 1894 s.
OpenCL on CPU: shine = 198.0 s
OpenCL on AMD W9100 GPU: 33.2 s (total)

Python 2.7.12 64 bit on Ubuntu 16.04:
no OpenCL, 1 core of CPU: 1869 s.
OpenCL on CPU: shine = 153.7 s
OpenCL on AMD W9100 GPU: 37.1 s (total)

Python 3.5.2 64 bit on Ubuntu 16.04:
no OpenCL, 1 core of CPU: 1893 s.
OpenCL on CPU: shine = 153.1 s
OpenCL on AMD W9100 GPU: 35.5 s (total)


Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz:

Python 2.7.10 64 bit on Windows 10:
no OpenCL, 1 core of CPU: 1801 s.
OpenCL on CPU: shine = 61.0 s
OpenCL on NVIDIA GeForce 940MX: 126 s (total)

Python 3.6.1 64 bit on Windows 10:
no OpenCL, 1 core of CPU: 1909 s.
OpenCL on CPU: shine = 60.3 s
OpenCL on NVIDIA GeForce 940MX: 123 s (total)

Python 2.7.12+ 64 bit on Ubuntu 16.10:
no OpenCL, 1 core of CPU: 1245 s.
OpenCL on CPU: shine = 57.6 s
OpenCL on NVIDIA GeForce 940MX: 122 s (total)

Python 3.5.2+ 64 bit on Ubuntu 16.10:
no OpenCL, 1 core of CPU: 1255 s.
OpenCL on CPU: shine = 60.2 s
OpenCL on NVIDIA GeForce 940MX: 127 s (total)

"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "06 Apr 2017"

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import time  # analysis:ignore
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import xrt.backends.raycing as raycing  # analysis:ignore
# raycing.targetOpenCL = "CPU"
# raycing.targetOpenCL = (0, 0)
# raycing.targetOpenCL = None
# raycing.precisionOpenCL = 'float32'

import xrt.backends.raycing.sources as rs  # analysis:ignore
import xrt.backends.raycing.screens as rsc  # analysis:ignore
import xrt.backends.raycing.run as rr  # analysis:ignore
import xrt.plotter as xrtp  # analysis:ignore
import xrt.runner as xrtr  # analysis:ignore

suffix = ''
E0 = 6900  # eV
R0 = 25000.  # Distance to the screen [mm]
bins = 256  # Number of bins in the plot histogram
ppb = 1  # Number of pixels per histogram bin

pprefix = '1_xrt'
Source = rs.Undulator
kwargs = dict(
    eE=3., eI=0.5,  # Parameters of the synchrotron ring [GeV], [Ampere]
    period=30., n=40,  # Parameters of the undulator, period in [mm]
    K=1.45,  # Deflection parameter (ignored if targetE is not None)
    eSigmaX=48.65, eSigmaZ=6.197,  # Size of the electron beam [mkm]
    eEpsilonX=0.263, eEpsilonZ=0.008,  # Emittance [nmrad]
    xPrimeMaxAutoReduce=False,
    zPrimeMaxAutoReduce=False)
xlimits = [-10, 10]  # Horizontal limits of the plot [mm]
zlimits = [-10, 10]  # Vertical limits of the plot [mm]
xlimitsZoom = [-2, 2]
zlimitsZoom = [-2, 2]
eEpsilonC = 'n'
kwargs['xPrimeMax'] = xlimits[-1] / R0 * 1e3
kwargs['zPrimeMax'] = zlimits[-1] / R0 * 1e3

prefix = pprefix+'-{0}-E-'.format(eEpsilonC)
eMinRays, eMaxRays = E0-300, E0+300
eUnit = 'eV'
kwargs['eMin'] = eMinRays
kwargs['eMax'] = eMaxRays


def build_beamline(nrays=100000):
    beamLine = raycing.BeamLine()
    beamLine.source = Source(
        beamLine, eN=1000, nx=40, nz=20, nrays=nrays, **kwargs)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, R0, 0))
    return beamLine


def run_process(beamLine):
    startTime = time.time()
    kw = {}

    beamSource = beamLine.source.shine(**kw)
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    outDict = {'beamSource': beamSource,
               'beamFSM1zoom': beamFSM1}
    print('shine time = {0}s'.format(time.time() - startTime))
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    plotsE = []

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='s', aspect='auto', title='horizontal polarization flux zoom')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '2horizFluxZoom' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    for plot in plotsE:
        f = plot.caxis.factor
        plot.caxis.limits = eMinRays*f, eMaxRays*f
    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.fluxFormatStr = '%.2p'
    return plots, plotsE


def main():
    beamLine = build_beamline()
    plots, plotsE = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)

# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()

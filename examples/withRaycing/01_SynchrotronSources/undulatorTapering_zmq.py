# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

import pickle
import numpy as np
import matplotlib.pyplot as plt

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

prefix = 'taper_'
xlimits = [-0.9, 0.9]
zlimits = [-0.9, 0.9]
eMin, eMax = 10200-800, 10200+800


def build_beamline(nrays=2e6):
    beamLine = raycing.BeamLine()
    rs.Undulator(
        beamLine, 'P06', nrays=nrays, eEspread=0.0011,
        eSigmaX=34.64, eSigmaZ=6.285, eEpsilonX=1., eEpsilonZ=0.01,
        period=31.4, K=2.1392-0.002, n=63, eE=6.08, eI=0.1, xPrimeMax=1.5e-2,
        zPrimeMax=1.5e-2, eMin=eMin, eMax=eMax, distE='BW',
        xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
#        targetOpenCL='CPU',
        targetOpenCL='SERVER_ADDRESS:15559',
        taper=(1.09, 11.254))
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, 90000, 0))
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    outDict = {'beamSource': beamSource,
               'beamFSM1': beamFSM1}
    if showIn3D:
        beamLine.prepare_flow()
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    plotsE = []

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=360, ppb=1)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=360, ppb=1)
    caxis = xrtp.XYCAxis('energy', 'keV', bins=360, ppb=1)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='total flux', ePos=1)
    plot.baseName = prefix + '1TotalFlux'
    plot.saveName = plot.baseName + '.png'
    plots.append(plot)
    plotsE.append(plot)

    for plot in plotsE:
        plot.caxis.limits = eMin*1e-3, eMax*1e-3
    for plot in plots:
        plot.fluxFormatStr = '%.2p'
    return plots, plotsE


def afterScript(plots):
    plot = plots[-1]
    flux = [plot.intensity, plot.nRaysAll, plot.nRaysAccepted,
            plot.nRaysSeeded]
    cwd = os.getcwd()
    pickleName = os.path.join(cwd, plot.baseName+'.pickle')
    with open(pickleName, 'wb') as f:
        pickle.dump((flux, plot.caxis.binEdges, plot.caxis.total1D), f,
                    protocol=2)
    plot_compare()


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow()
    else:
        plots, plotsE = define_plots(beamLine)
        xrtr.run_ray_tracing(plots, repeats=100, beamLine=beamLine,
                             afterScript=afterScript, afterScriptArgs=[plots])


def plot_compare():
    fig1 = plt.figure(1, figsize=(7, 5))
    ax = plt.subplot(111, label='1')
    ax.set_xlabel(u'energy (keV)')
    ax.set_ylabel(u'flux (a.u.)')

    cwd = os.getcwd()
    pickleName = os.path.join(cwd, 'taper_1TotalFlux.pickle')
    with open(pickleName, 'rb') as f:
        _f, binEdges, total1D = pickle.load(f)
    dE = binEdges[1] - binEdges[0]
    E = binEdges[:-1] + dE/2.
    ax.plot(E, total1D/max(total1D), 'r', label='calculated by xrt', lw=2)

    try:
        e, f = np.loadtxt('fluxUndulator1DtaperP06.dc0', skiprows=10,
                          usecols=[0, 1], unpack=True)
        ax.plot(e*1e-3, f/max(f), 'b', label='calculated by Spectra', lw=2)
    except:  # analysis:ignore
        pass

#    e, f = np.loadtxt('yaup-0.out', skiprows=32, usecols=[0, 1], unpack=True)
#    ax.plot(e*1e-3, f/max(f), 'g', label='calculated by YAUP/XOP', lw=2)

    theta, fl = np.loadtxt("thetaexafssc1an_zn_hgap_00002r2.fio.gz",
                           skiprows=113, usecols=(0, 5), unpack=True)
    si_1 = rm.CrystalSi(hkl=(1, 1, 1), tK=77)
    E = rm.ch / (2 * si_1.d * np.sin(np.radians(theta)))
    ax.plot(E*1e-3, fl/max(fl), 'k', lw=2, label='measured @ Petra3')

#    ax2.set_xlim(0, None)
#    ax2.set_ylim(1.400, 1.600)
    ax.legend(loc='lower center')

    fig1.savefig('compareTaper.png')
    plt.show()


if __name__ == '__main__':
    main()

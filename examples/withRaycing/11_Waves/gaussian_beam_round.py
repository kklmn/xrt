# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "28 May 2016"
import sys
sys.path.append(r"c:\Ray-tracing")
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.physconsts as phc
import xrt.backends.raycing.waves as rw

import xrt.plotter as xrtp
import xrt.runner as xrtr

nrays = 1e5
E0 = 9000.  # eV
w0 = 15e-3  # mm, waist size of the amplitude (not of intensity!)
maxFactor = 3.5  # factor that determines the screen limits as ±w*maxFactor
uniformRayDensity = True
ps = np.array([0.5, 1, 2, 4, 8]) * 10000.


def gaussian_beam_round(rSquare, y, w0, E):  # normalized to unit flux
    k = E / phc.CHBAR * 1e7  # mm^-1
    yR = k/2 * w0**2
    invR = y / (y**2 + yR**2)
    psi = np.arctan2(y, yR)
    w = w0 * (1 + (y/yR)**2)**0.5
    u = (2/np.pi)**0.5 / w *\
        np.exp(-rSquare/w**2 + 1j*k*(y + 0.5*rSquare*invR) - 1j*psi)
    return u, w


def build_beamline():
    beamLine = raycing.BeamLine(height=0)

    sig = w0 / 2  # gaussian beam is I~exp(-2r²/w²) but 'normal' I~exp(-r²/2σ²)
    beamLine.source = rs.GeometricSource(
        beamLine, 'Gaussian', nrays=nrays,
        uniformRayDensity=uniformRayDensity,
        distx='normal', dx=(sig, sig*maxFactor),
        distz='normal', dz=(sig, sig*maxFactor),
        distxprime=None, distzprime=None, energies=(E0,))

    beamLine.fsmFar = rsc.Screen(beamLine, 'FSM', [0, 0, 0])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine()
    beamSource.Es /= nrays**0.5
    beamSource.Jss /= nrays
    outDict = {'beamSource': beamSource}
    for ip, (p, (x, z)) in enumerate(zip(ps, beamLine.fsmXZmeshes)):
        beamLine.fsmFar.center[1] = p
        waveOnFSMg = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        gb = gaussian_beam_round(waveOnFSMg.x**2+waveOnFSMg.z**2, p, w0, E0)[0]
        dxdz = (x[1]-x[0]) * (z[1]-z[0])
        waveOnFSMg.Es[:] = gb * dxdz**0.5
        waveOnFSMg.Jss[:] = np.abs(gb)**2 * dxdz
        outDict['beamFSMg{0}'.format(ip)] = waveOnFSMg

        wrepeats = 1
        waveOnFSMk = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        for r in range(wrepeats):
            rw.diffract(beamSource, waveOnFSMk)
            if wrepeats > 1:
                print('wave repeats: {0} of {1} done'.format(r+1, wrepeats))
        outDict['beamFSMk{0}'.format(ip)] = waveOnFSMk
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamSource', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                           bins=256, ppb=1))
    lim = w0 / 2 * maxFactor * 1e3
    plot.xaxis.limits = [-lim, lim]
    plot.yaxis.limits = [-lim, lim]
    plot.saveName = '0-beamSource.png'
    plots.append(plot)

    beamLine.fsmXZmeshes = []
    for ip, p in enumerate(ps):
        lim = gaussian_beam_round(0, p, w0, E0)[1] / 2 * maxFactor * 1e3

        plot = xrtp.XYCPlot(
            'beamFSMg{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=256, ppb=1))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.saveName = '{0}-beamFSMg-at{1:02.0f}m.png'.format(ip+1, p*1e-3)
        plots.append(plot)

        plot = xrtp.XYCPlot(
            'beamFSMk{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=256, ppb=1))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.saveName = '{0}-beamFSMk-at{1:02.0f}m.png'.format(ip+1, p*1e-3)
        plots.append(plot)

        ax = plot.xaxis
        edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
        xCenters = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
        ax = plot.yaxis
        edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
        zCenters = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
        beamLine.fsmXZmeshes.append([xCenters, zCenters])

    for plot in plots:
        plot.caxis.limits = [-np.pi, np.pi]
        plot.caxis.fwhmFormatStr = None
        plot.ax1dHistE.set_yticks([l*np.pi for l in (-1, -0.5, 0, 0.5, 1)])
        plot.ax1dHistE.set_yticklabels(
            (r'$-\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'))

    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots, repeats=2, updateEvery=1, beamLine=beamLine, processes=1)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

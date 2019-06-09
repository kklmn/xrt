# -*- coding: utf-8 -*-

__author__ = "Konstantin Klementiev"
__date__ = "8 Jun 2019"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.waves as rw

import xrt.plotter as xrtp
xrtp.colorFactor = 1.
import xrt.runner as xrtr

#prefix = 'Laguerre-Gauss-'
#lVortex, pVortex = 1, 0
prefix = 'Gauss-'
lVortex, pVortex = 0, 0

E0 = 9000.  # eV
w0 = 15e-3  # mm, waist size of the amplitude (not of intensity!)
maxFactor = 2.  # factor that determines the screen limits as ±w*maxFactor
maxFactor *= (abs(lVortex)+pVortex+1)**0.25
# screen positions:
#ps = np.array([0, 0.5, 1, 2, 4, 8]) * 10000.
ps = np.array(list(range(10)) + list(range(1, 11)) +
              list(range(20, 101, 10))) * 1000.
ps[0:10] /= 10.
print("screen positions:", ps)

bins, ppb = 256, 1
wantKirchhoff = True
targetOpenCL='CPU'
#targetOpenCL='auto'

nSpokes = 12


def build_beamline():
    beamLine = raycing.BeamLine(height=0)
    beamLine.source = rs.LaguerreGaussianBeam(
        beamLine, 'Laguerre-Gaussian', w0=w0, vortex=(lVortex, pVortex),
        energies=(E0,))
    beamLine.slit = ra.SiemensStar(
        bl=beamLine, center=[0, 0.1, 0], nSpokes=nSpokes, rX=w0, rZ=w0,
        phi0=0.5*np.pi/nSpokes, vortex=0)
    beamLine.fsmFar = rsc.Screen(beamLine, 'FSM', [0, 0, 0])
    return beamLine


def run_process(beamLine):
    outDict = {}
    beamLine.fsmFar.center[1] = 0
    x, z = beamLine.fsmXZmeshes[0]
    waveOnFSMg = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
    beamLine.source.shine(wave=waveOnFSMg)
    beamLine.slit.propagate(waveOnFSMg)
    state = np.array(waveOnFSMg.state)
    for ip, (p, (x, z)) in enumerate(zip(ps, beamLine.fsmXZmeshes)):
        beamLine.fsmFar.center[1] = p
        waveOnFSMg = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        beamLine.source.shine(wave=waveOnFSMg)
        waveOnFSMg.state[:] = state
        if outDict == {}:  # or if ip == 0:
            beamSource = waveOnFSMg
            beamSource.area = 0.5 * np.pi*w0**2
        what = 'beamFSMg{0}'.format(ip)
        outDict[what] = waveOnFSMg

        if p > 100 and wantKirchhoff:
            print('p = {0}m, {1} of {2}'.format(p*1e-3, ip+1, len(ps)))
            wrepeats = 1
            waveOnFSMk = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
            for r in range(wrepeats):
                rw.diffract(beamSource, waveOnFSMk, targetOpenCL=targetOpenCL)
                if wrepeats > 1:
                    print('wave repeats: {0} of {1} done'.format(
                        r+1, wrepeats))
            what = 'beamFSMk{0}'.format(ip)
            outDict[what] = waveOnFSMk
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    beamLine.fsmXZmeshes = []
    for ip, p in enumerate(ps):
        lim = beamLine.source.w(p, E0) * maxFactor * 1e3

        plot = xrtp.XYCPlot(
            'beamFSMg{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=bins, ppb=ppb),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=bins, ppb=ppb),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=bins, ppb=ppb))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.title = '{0}{1:02d}-beamFSMg-at{2:03.1f}m'.format(
            prefix, ip, p*1e-3)
        tpf = '{0:2.1f} m' if p < 1000 else '{0:2.0f} m'
        plot.textPanel = plot.ax2dHist.text(
            0.02, 0.98, tpf.format(p*1e-3), size=14, color='w',
            transform=plot.ax2dHist.transAxes,
            ha='left', va='top')
        plot.saveName = plot.title + '.png'
        plots.append(plot)

        if p > 100 and wantKirchhoff:
            plot = xrtp.XYCPlot(
                'beamFSMk{0}'.format(ip), (1,),
                xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=bins, ppb=ppb),
                yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=bins, ppb=ppb),
                caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                                   bins=bins, ppb=ppb))
            plot.xaxis.limits = [-lim, lim]
            plot.yaxis.limits = [-lim, lim]
            plot.textPanel = plot.ax2dHist.text(
                0.02, 0.98, tpf.format(p*1e-3), size=14, color='w',
                transform=plot.ax2dHist.transAxes,
                ha='left', va='top')
            plot.title = '{0}{1:02d}-beamFSMk-at{2:03.1f}m'.format(
                prefix, ip, p*1e-3)
            plot.saveName = plot.title + '.png'
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
        plots, repeats=1, updateEvery=1, beamLine=beamLine, processes=1)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "25 Nov 2026"

import os
import sys
sys.path.append(os.path.join('..', '..'))  # analysis:ignore

import numpy as np
import matplotlib as mpl
# mpl.use('agg')

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.samples as rsam
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.materials.elemental as rme
# import xrt.backends.raycing.materials.compounds as rmc

import xrt.plotter as xrtp
import xrt.runner as xrtr

# case = '15keV'
# E0 = 15000.
case = '5keV'
E0 = 5000.

repeats, nrays = 1, 4e6
bins, ppb = 256, 1
binsS, ppbS = 512, 1
# binsE, ppbE = 2000, 1
binsE, ppbE = 512, 1

mVac = rm.EmptyMaterial(kind='lens')
mC = rme.C(kind='lens')
mSi = rme.Si(kind='lens')
mFe = rme.Fe(kind='lens')
mAu = rme.Au(kind='lens')


def get_xyprime(beam):
    return (beam.a**2 + beam.b**2)**0.5 / beam.c


def build_beamline():
    beamLine = raycing.BeamLine()
    beamLine.source = rs.GeometricSource(
        beamLine, 'CollimatedSource', nrays=nrays, uniformRayDensity=True,
        distx='flat', dx=6.2e-3, distz='flat', dz=6.2e-3,
        distxprime=None, distzprime=None, energies=(E0,))

    t = 5e-3
    # spheres = [dict(center=(1e-3, 1e-3), r=2e-3, material=mSi),
    #            dict(center=(-5e-4, -5e-4), r=1e-3, material=mAu)]
    spheres = [dict(center=(-2e-3, -2e-3), r=0.25e-3, material=mSi),
               dict(center=(0, -2e-3), r=0.5e-3, material=mSi),
               dict(center=(2e-3, -2e-3), r=1e-3, material=mSi),
               dict(center=(-2e-3, 0), r=1e-3, material=mFe),
               dict(center=(0, 0), r=0.5e-3, material=mFe),
               dict(center=(2e-3, 0), r=0.25e-3, material=mFe),
               dict(center=(-2e-3, 2e-3), r=0.25e-3, material=mAu),
               dict(center=(0, 2e-3), r=0.5e-3, material=mAu),
               dict(center=(2e-3, 2e-3), r=1e-3, material=mAu)]
    beamLine.spheresInPlate = rsam.SpheresInPlate(
        beamLine, 'spheresInPlate', center=[0, 10000, 0], pitch=np.pi/2,
        t=t, spheres=spheres,
        # material=mVac,
        material=mFe,  # the central row of spheres disappears, as expected
        # limPhysX=[-5e-3, 5e-3], limPhysY=[-5e-3, 5e-3]
        )

    beamLine.screenNear = rsc.Screen(beamLine, 'near', [0, 10500, 0])
    beamLine.screenFar = rsc.Screen(beamLine, 'far', [0, 30000, 0])
    return beamLine


def run_process_rays(beamLine):
    outDict = dict()
    beamSource = beamLine.source.shine()
    sphGlobal, sphLocal1, sphLocal2 = beamLine.spheresInPlate.transmit(
        beamSource)
    outDict['sphLocal1'] = sphLocal1
    outDict['sphLocal2'] = sphLocal2
    outDict['screenNear'] = beamLine.screenNear.expose(sphGlobal)
    outDict['screenFar'] = beamLine.screenFar.expose(sphGlobal)
    return outDict


rr.run_process = run_process_rays


def define_plots(beamLine):
    plots = []
    lims = -beamLine.source.dx*0.55e3, beamLine.source.dx*0.55e3

    plot = xrtp.XYCPlot(
        'sphLocal1', (1,), aspect=1,
        xaxis=xrtp.XYCAxis(r'$x$', 'µm', bins=binsS, ppb=ppbS),
        yaxis=xrtp.XYCAxis(r'$y$', 'µm', bins=binsS, ppb=ppbS),
        caxis=xrtp.XYCAxis(
            'Es phase', '', data=raycing.get_Es_phase,
            bins=binsS, ppb=ppbS),
        title='1a-sphLocal1')
    plot.xaxis.limits = lims
    plot.yaxis.limits = lims
    plot.textPanel = plot.fig.text(
        0.8, 0.9, '', transform=plot.fig.transFigure,
        size=12, color='r', ha='left')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'sphLocal2', (1,), aspect=1,
        xaxis=xrtp.XYCAxis(r'$x$', 'µm', bins=binsS, ppb=ppbS),
        yaxis=xrtp.XYCAxis(r'$y$', 'µm', bins=binsS, ppb=ppbS),
        caxis=xrtp.XYCAxis(
            'Es phase', '', data=raycing.get_Es_phase,
            bins=binsS, ppb=ppbS),
        title='1b-sphLocal2')
    plot.xaxis.limits = lims
    plot.yaxis.limits = lims
    plot.textPanel = plot.fig.text(
        0.8, 0.9, '', transform=plot.fig.transFigure,
        size=12, color='r', ha='left')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'sphLocal2', (1,), aspect=1,
        xaxis=xrtp.XYCAxis(r'$x$', 'µm', bins=binsS, ppb=ppbS),
        yaxis=xrtp.XYCAxis(r'$y$', 'µm', bins=binsS, ppb=ppbS),
        caxis=xrtp.XYCAxis(
            'radial divergence', 'nrad', data=get_xyprime,
            bins=binsS, ppb=ppbS),
        title='1c-sphLocal2')
    plot.xaxis.limits = lims
    plot.yaxis.limits = lims
    plot.caxis.limits = [0, 0.1]
    plot.textPanel = plot.fig.text(
        0.8, 0.9, '', transform=plot.fig.transFigure,
        size=12, color='r', ha='left')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'screenNear', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'µm', bins=binsS, ppb=ppbS),
        yaxis=xrtp.XYCAxis(r'$z$', 'µm', bins=binsS, ppb=ppbS),
        caxis=xrtp.XYCAxis(
            'Es phase', '', data=raycing.get_Es_phase,
            bins=binsS, ppb=ppbS),
        title='2-screenNear')
    plot.xaxis.limits = lims
    plot.yaxis.limits = lims
    plot.textPanel = plot.fig.text(
        0.8, 0.9, '', transform=plot.fig.transFigure,
        size=12, color='r', ha='left')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'screenFar', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'µm', bins=binsS, ppb=ppbS),
        yaxis=xrtp.XYCAxis(r'$z$', 'µm', bins=binsS, ppb=ppbS),
        caxis=xrtp.XYCAxis(
            'Es phase', '', data=raycing.get_Es_phase,
            bins=binsS, ppb=ppbS),
        title='3-screenFar')
    plot.xaxis.limits = lims
    plot.yaxis.limits = lims
    plot.textPanel = plot.fig.text(
        0.8, 0.9, '', transform=plot.fig.transFigure,
        size=12, color='r', ha='left')
    plots.append(plot)

    for iplot, plot in enumerate(plots):
        if not plot.caxis.useCategory:
            if plot.caxis.label == 'energy':
                plot.caxis.offset = E0
            if 'phase' in plot.caxis.label:
                plot.caxis.limits = [-np.pi, np.pi]
                plot.caxis.fwhmFormatStr = None
                plot.ax1dHistE.set_yticks(
                    [t*np.pi for t in (-1, -0.5, 0, 0.5, 1)])
                plot.ax1dHistE.set_yticklabels(
                    (r'$-\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$',
                     r'$\pi$'))
        plot.fluxFormatStr = '%.2p'
        plot.xaxis.fwhmFormatStr = '%.3f'
        plot.yaxis.fwhmFormatStr = '%.3f'
        plot.saveName = [f"{plot.title}.png"]
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine)


if __name__ == '__main__':
    main()

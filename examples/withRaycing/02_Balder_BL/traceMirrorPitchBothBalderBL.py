# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
#import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.run as rr
import BalderBL

showIn3D = False
BalderBL.showIn3D = showIn3D

stripe = 'Si'
E0 = 9000
dE = 2


def define_plots():
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMSample', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='Sample', ePos=0)
    plot.xaxis.limits = [-10, 10]
    plot.yaxis.limits = [42.79-10, 42.79+10]
#    plot.xaxis.fwhmFormatStr = '%.0f'
#    plot.yaxis.fwhmFormatStr = '%.2f'
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.ax2dHist.text(
        0.5, 0.9, '', transform=plot.ax2dHist.transAxes, size=14, color='r',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0
    return plots


def plot_generator(plots, beamLine):
    pitches = np.linspace(1., 4., 31)
    for pitch in pitches:
        BalderBL.align_beamline(beamLine, energy=E0, pitch=pitch*1e-3)
        for plot in plots:
            baseName = 'pitch-{0}{1:.1f}mrad'.format(plot.title, pitch)
            plot.saveName = baseName + '.png'
#            plot.persistentName = baseName + '.pickle'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(
                    r'$\theta$ = {0:.1f} mrad'.format(pitch))
        if showIn3D:
            beamLine.glowFrameName = 'pitch-{0:.1f}mrad.jpg'.format(pitch)
        yield


def main():
    myBalder = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    if showIn3D:
        myBalder.glow(centerAt='VFM', startFrom=2,
                      generator=plot_generator, generatorArgs=[[], myBalder])
        return
    plots = define_plots()
    xrtr.run_ray_tracing(
        plots, repeats=12, generator=plot_generator,
        beamLine=myBalder, globalNorm=True, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

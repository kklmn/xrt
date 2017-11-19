# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
from matplotlib.ticker import FixedLocator

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

mirror = 'vfm'
mirrorText = 'collimating' if mirror == 'vcm' else 'focusing'


def define_plots():
    plots = []
    if mirror == 'vcm':
        plot = xrtp.XYCPlot(
            'beamFSMDCM', (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
            caxis=xrtp.XYCAxis('energy', 'eV'), title='DCM')
        plot.xaxis.limits = [-7., 7.]
        plot.yaxis.limits = [38.1-7., 38.1+7.]
        plot.fluxFormatStr = '%.2p'
        plot.textPanel = plot.fig.text(
            0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
            ha='center')
        plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMSample', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xrtp.defaultBins/2),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='Sample')
    plot.xaxis.limits = [-300, 300]
    plot.yaxis.limits = [42.8-0.6, 42.8+0.6]
    plot.ax2dHist.xaxis.set_major_locator(FixedLocator([-200, 0, 200]))
    plot.xaxis.fwhmFormatStr = '%.0f'
    plot.yaxis.fwhmFormatStr = '%.2f'
    plot.fluxFormatStr = '%.2p'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0
    return plots


def plot_generator(plots, beamLine):
    Rs = np.linspace(0.6, 1.4, 21)
    if mirror == 'vcm':
        Rs *= beamLine.vcm.R
    else:
        Rs *= beamLine.vfm.R
    for R in Rs:
        if mirror == 'vcm':
            beamLine.vcm.R = R
        else:
            beamLine.vfm.R = R
        for plot in plots:
            baseName = '{0}R-{1}{2:.1f}km'.format(
                mirror, plot.title, R*1e-6)
            plot.saveName = baseName + '.png'
#            plot.persistentName = baseName + '.pickle'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(
                    '{0}\nmirror\n$R$ = {1:.1f} km'.format(
                        mirrorText, R*1e-6))
        if showIn3D:
            beamLine.glowFrameName = '{0}R-{1:.1f}km.jpg'.format(mirror, R*1e-6)
        yield


def main():
    myBalder = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    BalderBL.align_beamline(myBalder, energy=E0)
    if showIn3D:
        myBalder.glow(centerAt='VFM', startFrom=2,
                      generator=plot_generator, generatorArgs=[[], myBalder])
        return
    plots = define_plots()
    xrtr.run_ray_tracing(
        plots, repeats=16, generator=plot_generator,
        beamLine=myBalder, globalNorm=True, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

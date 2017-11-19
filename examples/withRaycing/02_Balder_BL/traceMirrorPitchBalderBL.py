# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.run as rr
import BalderBL

showIn3D = False
BalderBL.showIn3D = showIn3D

stripe = 'Ir'


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()

    beamFSM0 = beamLine.fsm0.expose(beamSource)
    beamTmp = beamLine.feFixedMask.propagate(beamSource)
    beamFSMFE = beamLine.fsmFE.expose(beamSource)
    beamFilter1global, beamFilter1local1, beamFilter1local2 =\
        beamLine.filter1.double_refract(beamSource)
    beamFilter1local2A = rs.Beam(copyFrom=beamFilter1local2)
    beamFilter1local2A.absorb_intensity(beamSource)
    beamFurtherDown = beamFilter1global

    beamVCMglobal, beamVCMlocal = beamLine.vcm.reflect(beamFurtherDown)
    beamVCMlocal.absorb_intensity(beamFurtherDown)
    beamFSMVCM = beamLine.fsmVCM.expose(beamVCMglobal)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamFSMFE': beamFSMFE,
               'beamFilter1global': beamFilter1global,
               'beamFilter1local1': beamFilter1local1,
               'beamFilter1local2': beamFilter1local2,
               'beamFilter1local2A': beamFilter1local2A,
               'beamVCMglobal': beamVCMglobal, 'beamVCMlocal': beamVCMlocal,
               'beamFSMVCM': beamFSMVCM}
    beamLine.beams = outDict
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

#    plot = xrtp.XYCPlot('beamFilter1local1', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
#      caxis=xrtp.XYCAxis('energy', 'keV'),
#      title='Footprint1_I')
#    plot.fluxFormatStr = '%.2e'
#    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamVCMlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-7, 7]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-700, 700]),
        caxis=xrtp.XYCAxis('energy', 'keV'),
        fluxKind='power', title='FootprintP',  # oe=beamLine.vcm,
        contourLevels=[0.9, ], contourColors=['r', ],
        contourFmt=r'%.3f W/mm$^2$')
    plot.fluxFormatStr = '%.0f'
    plot.xaxis.fwhmFormatStr = '%.1f'
    plot.yaxis.fwhmFormatStr = '%.0f'
    plot.textPanel = plot.fig.text(
        0.85, 0.8, '', transform=plot.fig.transFigure,
        size=14, color='r', ha='center')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMVCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV'), title='FSM')
    plot.xaxis.limits = [-7, 7]
    plot.yaxis.limits = [-2, 12]
    plot.fluxFormatStr = '%.2p'
    plot.textPanel = plot.fig.text(
        0.85, 0.8, '', transform=plot.fig.transFigure,
        size=14, color='r', ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [0, beamLine.sources[0].eMax*1e-3]
        plot.caxis.fwhmFormatStr = None
    return plots


def plot_generator(plots, beamLine):
    pitches = np.linspace(1., 4., 31)
    for pitch in pitches:
        beamLine.vcm.pitch = pitch * 1e-3
        for plot in plots:
            baseName = 'vcm{0}-{1}{2:.1f}mrad'.format(
                stripe, plot.title, pitch)
            plot.saveName = baseName + '.png'
#            plot.persistentName = baseName + '.pickle'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(
                    '{0}\n$\\theta$ = {1:.1f} mrad'.format(stripe, pitch))
        if showIn3D:
            beamLine.glowFrameName =\
                'vcm{0}-{1:.1f}mrad.jpg'.format(stripe, pitch)
        yield


def main():
    myBalder = BalderBL.build_beamline(stripe=stripe)
    BalderBL.align_beamline(myBalder)
    if showIn3D:
        myBalder.glow(centerAt='VCM', startFrom=2,
                      generator=plot_generator, generatorArgs=[[], myBalder])
        return
    plots = define_plots(myBalder)
    xrtr.run_ray_tracing(
        plots, repeats=16, generator=plot_generator,
        beamLine=myBalder, globalNorm=True, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

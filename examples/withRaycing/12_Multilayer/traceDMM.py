# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import BalderDMM
import xrt.plotter as xrtp
import xrt.runner as xrtr
# import xrt.backends.raycing.materials as rm

stripe = 'Si'
E0 = 8000
dE = 1200


def make_glow_scan(beamLine):
    """Align beamLine in place, then capture its settings as Glow frames."""
    # 1. Choose the scan inputs (eV).
    energies = np.linspace(E0 - 500, E0 + 500, 7)

    frames = {}
    for index, energy in enumerate(energies):
        # 2. Calculate all dependent element settings.
        BalderDMM.align_beamline(beamLine, energy=energy)
        thetaDeg = np.degrees(
            beamLine.dmm.bragg - 2*beamLine.vcm.pitch)
        frameName = 'DMM_{0:05.0f}.jpg'.format(thetaDeg*1e4)

        # 3. Capture the aligned values; copy mutable lists for this frame.
        frames['frame_{0:04d}'.format(index)] = {
            'objects': {
                beamLine.feFixedMask.name: {  # 'FEFixedMask'
                    'opening': list(beamLine.feFixedMask.opening)},
                beamLine.vcm.name: {  # 'VCM'
                    'pitch': beamLine.vcm.pitch, 'R': beamLine.vcm.R},
                beamLine.dmm.name: {  # 'DMM'
                    'center': list(beamLine.dmm.center),
                    'bragg': beamLine.dmm.bragg,
                    'cryst2perpTransl': beamLine.dmm.cryst2perpTransl},
                beamLine.slitAfterDCM.name: {  # 'SlitAfterDCM'
                    'opening': list(beamLine.slitAfterDCM.opening)},
                beamLine.vfm.name: {  # 'VFM'
                    'center': list(beamLine.vfm.center),
                    'pitch': beamLine.vfm.pitch, 'R': beamLine.vfm.R},
                beamLine.slitAfterVFM.name: {  # 'SlitAfterVFM'
                    'opening': list(beamLine.slitAfterVFM.opening)},
                beamLine.slitEH.name: {  # 'slitEH'
                    'opening': list(beamLine.slitEH.opening)},
                },
            'output': {'glowFrameName': frameName},
            }
    return {'version': 1, 'kind': 'timeline_recipe',
            'expandedFrames': frames}


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSM0', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'))
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = 'fsm0'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMVCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'))
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = 'fsmVVM'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamDCMlocal1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'),
        title='ML1 local')
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = '1stML'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamDCMlocal2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'),
        title='ML2 local')
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = '2ndML'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMDCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'), title='DCM')
    # plot.xaxis.limits = [-7., 7.]
    # plot.yaxis.limits = [20.3-7., 20.3+7.]
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = 'afterDMM'
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [(E0 - dE)*1e-3, (E0 + dE)*1e-3]
        plot.caxis.offset = E0*1e-3
    return plots


def plot_generator(plots, beamLine):
    """Classic runner generator for the multilayer energy scan.

    ``run_ray_tracing()`` still uses this yielding generator for plot labels
    and normalization. In xrtGlow the same energy loop is converted by
    ``make_glow_scan(beamLine)`` into explicit frames. We choose frames over
    compact loops here because each energy frame contains alignment side
    effects, not just the energy value itself.
    """
    # energies = np.linspace(E0 - 500, E0 + 500, 7)
    energies = E0,
    for energy in energies:
        BalderDMM.align_beamline(beamLine, energy=energy)
        thetaDeg = np.degrees(
            beamLine.dmm.bragg - 2*beamLine.vcm.pitch)
        for plot in plots:
            baseName = '{0}_{1:05.0f}'.format(plot.baseName, thetaDeg*1e4)
            plot.saveName = baseName + '.png'
            # plot.persistentName = baseName + '.pickle'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(
                    '$\\theta$ = {0:.3f}$^o$'.format(thetaDeg))
        if BalderDMM.showIn3D:
            beamLine.glowFrameName = 'DMM_{0:05.0f}.jpg'.format(thetaDeg*1e4)
        yield


def main():
    myBalder = BalderDMM.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    if BalderDMM.showIn3D:
        scan = make_glow_scan(myBalder)
        myBalder.glow(scale=[500, 10, 500], centerAt='VCM', startFrom=1,
                      scan=scan)
        return
    plots = define_plots(myBalder)
    xrtr.run_ray_tracing(plots, repeats=16, generator=plot_generator,
                         beamLine=myBalder,
                         globalNorm=True,
                         processes=1)


# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()

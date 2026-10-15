# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
# import xrt.backends.raycing.sources as rs
# import xrt.backends.raycing.run as rr
import BalderBL

showIn3D = True
BalderBL.showIn3D = showIn3D

stripe = 'Si'
E0 = 9000
dE = 2


def make_glow_scan():
    """Calculate each alignment, then capture its settings as a Glow frame."""
    # 1. Choose the scan inputs (mrad).
    pitchesMrad = np.linspace(1., 3., 21)

    # Preparing frames on a separate beamline preserves the initial Glow view.
    beamLine = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    frames = {}
    for index, pitchMrad in enumerate(pitchesMrad):
        # 2. Calculate all dependent element settings.
        BalderBL.align_beamline(
            beamLine, energy=E0, pitch=pitchMrad*1e-3)
        frameName = 'pitch-{0:.1f}mrad.jpg'.format(pitchMrad)

        # 3. Capture the aligned values; copy mutable lists for this frame.
        frames['frame_{0:04d}'.format(index)] = {
            'objects': {
                'FEFixedMask': {
                    'opening': list(beamLine.feFixedMask.opening)},
                'VCM': {
                    'pitch': beamLine.vcm.pitch, 'R': beamLine.vcm.R},
                'DCM': {
                    'center': list(beamLine.dcm.center),
                    'bragg': beamLine.dcm.bragg,
                    'cryst2perpTransl': beamLine.dcm.cryst2perpTransl},
                'SlitAfterDCM': {
                    'opening': list(beamLine.slitAfterDCM.opening)},
                'VFM': {
                    'center': list(beamLine.vfm.center),
                    'pitch': beamLine.vfm.pitch, 'R': beamLine.vfm.R},
                'SlitAfterVFM': {
                    'opening': list(beamLine.slitAfterVFM.opening)},
                'slitEH': {'opening': list(beamLine.slitEH.opening)},
                },
            'output': {'glowFrameName': frameName},
            }
    return {'version': 1, 'kind': 'timeline_recipe',
            'expandedFrames': frames}


def define_plots():
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMSample', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=256, ppb=1),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=256, ppb=1),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='Sample',
        ePos=0, showCornerLabels=False)
    plot.xaxis.limits = [-10, 10]
    plot.yaxis.limits = [-1, 1]
    plot.xaxis.fwhmFormatStr = '%.2f'
    plot.yaxis.fwhmFormatStr = '%.2f'
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.ax2dHist.text(
        0.5, 0.9, '', transform=plot.ax2dHist.transAxes, size=12, color='w',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0
    return plots


def plot_generator(plots, beamLine):
    """Classic runner generator for a coupled Balder pitch alignment scan.

    The generator is still used by ``run_ray_tracing()`` because it can update
    plots after each ``yield``. In xrtGlow the same idea is represented by
    ``make_glow_scan()``: every pitch value is first passed through the normal
    alignment routine, then the dependent beamline properties are written into
    explicit JSON-like frames. This is more verbose than ``linspace`` but it
    records the coupled geometry, not just the scanned pitch value.
    """
    pitches = np.linspace(1., 3., 21)
    for pitch in pitches:
        BalderBL.align_beamline(beamLine, energy=E0, pitch=pitch*1e-3)
        for plot in plots:
            baseName = 'pitch-{0}{1:.1f}mrad'.format(plot.title, pitch)
            plot.saveName = baseName + '.png'
            # plot.persistentName = baseName + '.pickle'
            if hasattr(plot, 'textPanel'):
                plot.textPanel.set_text(r'θ = {0:.1f} mrad'.format(pitch))
        if showIn3D:
            beamLine.glowFrameName = 'pitch-{0:.1f}mrad.jpg'.format(pitch)
        yield


def main():
    myBalder = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    if showIn3D:
        BalderBL.align_beamline(myBalder, energy=E0, pitch=2.5e-3)
        scan = make_glow_scan()
        myBalder.glow(centerAt='VFM', startFrom=2, scan=scan)
        return
    plots = define_plots()
    xrtr.run_ray_tracing(plots, repeats=16, generator=plot_generator,
                         beamLine=myBalder, globalNorm=True, processes=4)


# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()

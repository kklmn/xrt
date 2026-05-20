# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import copy
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

GLOW_ALIGNMENT_PROPS = [
    ('FEFixedMask', 'feFixedMask', ('opening',)),
    ('VCM', 'vcm', ('pitch', 'R')),
    ('DCM', 'dcm', ('center', 'bragg', 'cryst2perpTransl')),
    ('SlitAfterDCM', 'slitAfterDCM', ('opening',)),
    ('VFM', 'vfm', ('center', 'pitch', 'R')),
    ('SlitAfterVFM', 'slitAfterVFM', ('opening',)),
    ('slitEH', 'slitEH', ('opening',)),
    ]


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


def _copy_scan_value(value):
    if hasattr(value, 'uuid'):
        return value
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return tuple(_copy_scan_value(v) for v in value)
    if isinstance(value, list):
        return [_copy_scan_value(v) for v in value]
    return copy.deepcopy(value)


def _scan_state(beamLine):
    state = {}
    for _target, attr, props in GLOW_ALIGNMENT_PROPS:
        obj = getattr(beamLine, attr)
        state[attr] = {
            prop: _copy_scan_value(getattr(obj, prop)) for prop in props}
    return state


def _restore_scan_state(beamLine, state):
    for attr, props in state.items():
        obj = getattr(beamLine, attr)
        for prop, value in props.items():
            setattr(obj, prop, _copy_scan_value(value))


def _alignment_patch(beamLine):
    patch = {}
    for target, attr, props in GLOW_ALIGNMENT_PROPS:
        obj = getattr(beamLine, attr)
        patch[target] = {
            prop: _copy_scan_value(getattr(obj, prop)) for prop in props}
    return patch


def make_glow_scan(beamLine):
    """Builds explicit xrtGlow frames from the Balder alignment routine.

    This scan is not a plain one-property ``linspace``: changing the common
    mirror pitch also moves/retunes the DCM, VFM and slit openings. We
    therefore let ``BalderBL.align_beamline()`` calculate every frame and store
    the resulting absolute values as frame patches. The classic runner keeps
    the compact generator below; glow gets a frame dictionary that can be
    replayed by the standalone propagation process.
    """
    frames = {}
    state = _scan_state(beamLine)
    try:
        for index, pitch in enumerate(np.linspace(1., 4., 31)):
            BalderBL.align_beamline(
                beamLine, energy=E0, pitch=pitch*1e-3)
            frameName = 'pitch-{0:.1f}mrad.jpg'.format(pitch)
            frames['frame_{0:04d}'.format(index)] = {
                'objects': _alignment_patch(beamLine),
                'output': {'glowFrameName': frameName},
                }
    finally:
        _restore_scan_state(beamLine, state)
    return {'version': 1, 'kind': 'timeline_recipe',
            'expandedFrames': frames}


def plot_generator(plots, beamLine):
    """Classic runner generator for a coupled Balder pitch alignment scan.

    The generator is still used by ``run_ray_tracing()`` because it can update
    plots after each ``yield``. In xrtGlow the same idea is represented by
    ``make_glow_scan()``: every pitch value is first passed through the normal
    alignment routine, then the dependent beamline properties are written into
    explicit JSON-like frames. This is more verbose than ``linspace`` but it
    records the coupled geometry, not just the scanned pitch value.
    """
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
        scan = make_glow_scan(myBalder)
        myBalder.glow(centerAt='VFM', startFrom=2,
                      scan=scan)
        return
    plots = define_plots()
    xrtr.run_ray_tracing(
        plots, repeats=12, generator=plot_generator,
        beamLine=myBalder, globalNorm=True, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

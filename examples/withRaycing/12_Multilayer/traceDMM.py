# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

import copy
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import BalderDMM
import xrt.plotter as xrtp
import xrt.runner as xrtr
#import xrt.backends.raycing.materials as rm

stripe = 'Si'
E0 = 8000
dE = 1200

GLOW_ALIGNMENT_PROPS = [
    ('FEFixedMask', 'feFixedMask', ('opening',)),
    ('VCM', 'vcm', ('pitch', 'R')),
    ('DMM', 'dmm', ('center', 'bragg', 'cryst2perpTransl')),
    ('SlitAfterDCM', 'slitAfterDCM', ('opening',)),
    ('VFM', 'vfm', ('center', 'pitch', 'R')),
    ('SlitAfterVFM', 'slitAfterVFM', ('opening',)),
    ('slitEH', 'slitEH', ('opening',)),
    ]


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMDCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'), title='DCM')
    plot.xaxis.limits = [-7., 7.]
    plot.yaxis.limits = [20.3-7., 20.3+7.]
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = 'afterDMM'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamDCMlocal1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV', fwhmFormatStr='%.2f'),
        title='Xtal1 local')
    plot.xaxis.limits = [-86., 86.]
    plot.yaxis.limits = [-86., 86.]
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plot.baseName = '1stML'
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [(E0 - dE)*1e-3, (E0 + dE)*1e-3]
        plot.caxis.offset = E0*1e-3
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
    """Builds explicit xrtGlow frames for the DMM energy alignment scan.

    A simple JSON ``linspace`` can describe the input energies, but it cannot
    describe the dependent alignment updates that ``BalderDMM.align_beamline``
    calculates for the multilayer, mirrors and slits. For glow we therefore
    run the alignment loop once, record the absolute frame values, then restore
    the beamline before opening the viewer.
    """
    frames = {}
    state = _scan_state(beamLine)
    try:
        for index, energy in enumerate(np.linspace(E0 - 500, E0 + 500, 7)):
            BalderDMM.align_beamline(beamLine, energy=energy)
            thetaDeg = np.degrees(
                beamLine.dmm.bragg - 2*beamLine.vcm.pitch)
            frameName = 'DMM_{0:05.0f}.jpg'.format(thetaDeg*1e4)
            frames['frame_{0:04d}'.format(index)] = {
                'objects': _alignment_patch(beamLine),
                'output': {'glowFrameName': frameName},
                }
    finally:
        _restore_scan_state(beamLine, state)
    return {'version': 1, 'kind': 'timeline_recipe',
            'expandedFrames': frames}


def plot_generator(plots, beamLine):
    """Classic runner generator for the multilayer energy scan.

    ``run_ray_tracing()`` still uses this yielding generator for plot labels
    and normalization. In xrtGlow the same energy loop is converted by
    ``make_glow_scan()`` into explicit frames. We choose frames over compact
    loops here because each energy frame contains alignment side effects, not
    just the energy value itself.
    """
    energies = np.linspace(E0 - 500, E0 + 500, 7)
#    energies = E0,
    for energy in energies:
        BalderDMM.align_beamline(beamLine, energy=energy)
        thetaDeg = np.degrees(
            beamLine.dmm.bragg - 2*beamLine.vcm.pitch)
        for plot in plots:
            baseName = '{0}_{1:05.0f}'.format(plot.baseName, thetaDeg*1e4)
            plot.saveName = baseName + '.png'
#            plot.persistentName = baseName + '.pickle'
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
    xrtr.run_ray_tracing(plots, repeats=120, generator=plot_generator,
                         beamLine=myBalder,
                         globalNorm=True,
                         processes=1)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

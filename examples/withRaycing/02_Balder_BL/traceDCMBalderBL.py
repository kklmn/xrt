# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import copy
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.materials as rm
import BalderBL

showIn3D = False
BalderBL.showIn3D = showIn3D

stripe = 'Si'
E0 = 9000
dE = 4

si111_1 = rm.CrystalSi(hkl=(1, 1, 1), tK=-171+273.15)
si111_2 = rm.CrystalSi(hkl=(1, 1, 1), tK=-140+273.15)
si311_1 = rm.CrystalSi(hkl=(3, 1, 1), tK=-171+273.15)
si311_2 = rm.CrystalSi(hkl=(3, 1, 1), tK=-140+273.15)

GLOW_ALIGNMENT_PROPS = [
    ('FEFixedMask', 'feFixedMask', ('opening',)),
    ('VCM', 'vcm', ('pitch', 'R')),
    ('DCM', 'dcm', ('surface', 'material', 'material2', 'center',
                    'bragg', 'cryst2perpTransl')),
    ('SlitAfterDCM', 'slitAfterDCM', ('opening',)),
    ('VFM', 'vfm', ('center', 'pitch', 'R')),
    ('SlitAfterVFM', 'slitAfterVFM', ('opening',)),
    ('slitEH', 'slitEH', ('opening',)),
    ]


def define_plots():
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMDCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='DCM')
    plot.xaxis.limits = [-7., 7.]
    plot.yaxis.limits = [38.1-7., 38.1+7.]
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
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


def _material_ref(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return getattr(value, 'uuid', value)


def _scan_value(prop, value):
    if prop.startswith('material'):
        return _material_ref(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_scan_value(prop, v) for v in value]
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
            prop: _scan_value(prop, getattr(obj, prop)) for prop in props}
    return patch


def make_glow_scan(beamLine):
    """Builds explicit xrtGlow frames for the DCM energy scan.

    The DCM frame values are calculated by the same alignment code as the
    runner generator. We store frame patches rather than a compact ``linspace``
    because each energy changes several dependent properties: DCM Bragg angle,
    crystal translation, mirror positions and slit openings. Material
    references are kept as runtime UUIDs in this Python scan so the process
    updates the exact crystal objects created by the example.
    """
    frames = {}
    energies = np.linspace(E0 - dE*0.66, E0 + dE*0.66, 7)
#    crystals = 'Si111', 'Si311'
    crystals = 'Si111',
    state = _scan_state(beamLine)
    try:
        frameIndex = 0
        for crystal in crystals:
            if crystal == 'Si111':
                beamLine.dcm.surface = crystal,
                beamLine.dcm.material = si111_1,
                beamLine.dcm.material2 = si111_2,
            elif crystal == 'Si311':
                beamLine.dcm.surface = crystal,
                beamLine.dcm.material = si311_1,
                beamLine.dcm.material2 = si311_2,
            for energy in energies:
                BalderBL.align_beamline(beamLine, energy=energy)
                thetaDeg = np.degrees(
                    beamLine.dcm.bragg - 2*beamLine.vcm.pitch)
                frameName = '{0}_{1:.0f}.jpg'.format(
                    crystal, thetaDeg*1e4)
                frames['frame_{0:04d}'.format(frameIndex)] = {
                    'objects': _alignment_patch(beamLine),
                    'output': {'glowFrameName': frameName},
                    }
                frameIndex += 1
    finally:
        _restore_scan_state(beamLine, state)
    return {'version': 1, 'kind': 'timeline_recipe',
            'expandedFrames': frames}


def plot_generator(plots, beamLine):
    """Classic runner generator for DCM material/energy frames.

    For normal ray tracing this generator remains convenient because it can
    update plot labels and filenames around each ``yield``. For xrtGlow the
    helper ``make_glow_scan()`` performs the same nested loop up front and
    returns explicit frame dictionaries. We use frames, not a single
    ``linspace`` track, because every energy is accompanied by alignment
    changes across several elements.
    """
    energies = np.linspace(E0 - dE*0.66, E0 + dE*0.66, 7)
#    crystals = 'Si111', 'Si311'
    crystals = 'Si111',
    for crystal in crystals:
        if crystal == 'Si111':
            beamLine.dcm.surface = crystal,
            beamLine.dcm.material = si111_1,
            beamLine.dcm.material2 = si111_2,
        elif crystal == 'Si311':
            beamLine.dcm.surface = crystal,
            beamLine.dcm.material = si311_1,
            beamLine.dcm.material2 = si311_2,
        for energy in energies:
            BalderBL.align_beamline(beamLine, energy=energy)
            thetaDeg = np.degrees(
                beamLine.dcm.bragg - 2*beamLine.vcm.pitch)
            baseName = '{0}_{1:.0f}.png'.format(crystal, thetaDeg*1e4)
            for plot in plots:
                plot.saveName = baseName + '.png'
#                plot.persistentName = baseName + '.pickle'
                if hasattr(plot, 'textPanel'):
                    plot.textPanel.set_text(
                        '{0}\n$\\theta$ = {1:.3f}$^o$'.format(
                            crystal, thetaDeg))
            if showIn3D:
                beamLine.glowFrameName = baseName + '.jpg'
            yield


def main():
    myBalder = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    if showIn3D:
        scan = make_glow_scan(myBalder)
        myBalder.glow(centerAt='VFM', startFrom=7,
                      scan=scan)
        return
    plots = define_plots()
    xrtr.run_ray_tracing(plots, repeats=16, generator=plot_generator,
                         beamLine=myBalder, globalNorm=True, processes='half')


if __name__ == '__main__':
    main()

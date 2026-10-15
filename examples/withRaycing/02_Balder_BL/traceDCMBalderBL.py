# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.materials as rm
import BalderBL

showIn3D = True
BalderBL.showIn3D = showIn3D

stripe = 'Si'
E0 = 9000
dE = 4

si111_1 = rm.CrystalSi(hkl=(1, 1, 1), tK=-171+273.15)
si111_2 = rm.CrystalSi(hkl=(1, 1, 1), tK=-140+273.15)
si311_1 = rm.CrystalSi(hkl=(3, 1, 1), tK=-171+273.15)
si311_2 = rm.CrystalSi(hkl=(3, 1, 1), tK=-140+273.15)


def make_glow_scan(beamLine):
    """Align beamLine in place, then capture its settings as Glow frames."""
    # 1. Choose the energies (eV) and crystal pairs.
    energies = np.linspace(E0 - dE*0.66, E0 + dE*0.66, 7)
    crystals = [
        ('Si111', si111_1, si111_2),
        # ('Si311', si311_1, si311_2),
        ]

    frames = {}
    for crystal, material1, material2 in crystals:
        beamLine.dcm.surface = (crystal,)
        beamLine.dcm.material = (material1,)
        beamLine.dcm.material2 = (material2,)
        for energy in energies:
            # 2. Calculate all dependent element settings.
            BalderBL.align_beamline(beamLine, energy=energy)
            thetaDeg = np.degrees(
                beamLine.dcm.bragg - 2*beamLine.vcm.pitch)
            frameName = '{0}_{1:.0f}.jpg'.format(crystal, thetaDeg*1e4)

            # 3. Capture the aligned values. Copy lists changed by alignment;
            # material UUIDs refer to the crystal objects defined above.
            frames['frame_{0:04d}'.format(len(frames))] = {
                'objects': {
                    beamLine.feFixedMask.name: {  # 'FEFixedMask'
                        'opening': list(beamLine.feFixedMask.opening)},
                    beamLine.vcm.name: {  # 'VCM'
                        'pitch': beamLine.vcm.pitch, 'R': beamLine.vcm.R},
                    beamLine.dcm.name: {  # 'DCM'
                        'surface': [crystal],
                        'material': material1.uuid,
                        'material2': material2.uuid,
                        'center': list(beamLine.dcm.center),
                        'bragg': beamLine.dcm.bragg,
                        'cryst2perpTransl': beamLine.dcm.cryst2perpTransl},
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


def define_plots():
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMDCM', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='DCM')
    plot.xaxis.limits = [-7., 7.]
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.fig.text(0.88, 0.8, '',
                                   transform=plot.fig.transFigure, size=14,
                                   color='r', ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0
    return plots


def plot_generator(plots, beamLine):
    """Classic runner generator for DCM material/energy frames.

    For normal ray tracing this generator remains convenient because it can
    update plot labels and filenames around each ``yield``. For xrtGlow the
    helper ``make_glow_scan(beamLine)`` performs the same nested loop up front
    and returns explicit frame dictionaries. We use frames, not a single
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

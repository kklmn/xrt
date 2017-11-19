# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"

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


def plot_generator(plots, beamLine):
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
        myBalder.glow(centerAt='VFM', startFrom=7,
                      generator=plot_generator, generatorArgs=[[], myBalder])
        return
    plots = define_plots()
    xrtr.run_ray_tracing(plots, repeats=16, generator=plot_generator,
                         beamLine=myBalder, globalNorm=True, processes='half')


if __name__ == '__main__':
    main()

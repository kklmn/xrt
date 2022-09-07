# -*- coding: utf-8 -*-
u"""
Reflection from a thin mosaic crystal. Although the crystal is specified as
"Bragg reflected", the transmitted part is also calculated as those rays that
go deeper into the crystal than the crystal thickness when the impact points
sample the mean free path distribution. Two apertures are oriented around the
direct and the diffracted parts of the "reflected" beam.
"""
__author__ = "Konstantin Klementiev"
__date__ = "7 Sep 2022"

import sys
import os
import numpy as np

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
# import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

mosaicityFWHMdeg = 0.4
mosaicitySigma = np.radians(mosaicityFWHMdeg) / 2.355
xtalMosaicHOPG1mm = rmats.CrystalFromCell(
    'HOPG008_1mm',
    (0, 0, 8),
    a=2.456, c=6.696,
    gamma=120,
    atoms=['C']*4,
    atomsXYZ=[[0, 0, 0], [0, 0, 0.5], [1./3, 2./3, 0], [2./3, 1./3, 0.5]],
    geom="Bragg reflected",
    table='Chantler',
    mosaicity=mosaicitySigma,
    t=1.0  # try to comment it out
    )

E0 = 15000  # eV
dEh = 1.5  # eV
nrays = 1e5
xtalPos = 1000.
slitDist = 1000.


def build_beamline():
    beamLine = raycing.BeamLine()
    beamLine.source = rsources.GeometricSource(
        beamLine,
        center=[0, 0, 0],
        distx="normal", dx=.5,
        disty=None,
        distz="normal", dz=.5,
        distxprime=None, distzprime=None,
        distE='flat', energies=[E0-dEh, E0+dEh],
        polarization='h',
        nrays=nrays)

    beamLine.xtal = roes.OE(
        beamLine, center=[0., xtalPos, 0.], material=xtalMosaicHOPG1mm)

    beamLine.apertureDirect = rapts.RectangularAperture(
        beamLine, center=[0, xtalPos+slitDist, 0], opening=[-20, 20, -20, 20])
    beamLine.apertureDiff = rapts.RectangularAperture(
        beamLine, opening=[-20, 20, -20, 20])

    return beamLine


def align_beamline(beamLine, E):
    cr = beamLine.xtal.material
    bragg = cr.get_Bragg_angle(E) - cr.get_dtheta(E)
    beamLine.xtal.pitch = bragg

    cos2theta, sin2theta = np.cos(2*bragg), np.sin(2*bragg)
    beamLine.apertureDiff.center = [0, xtalPos + slitDist*cos2theta,
                                    slitDist*sin2theta]
    beamLine.apertureDiff.set_orientation('auto', [0, -sin2theta, cos2theta])


def run_process(beamLine):
    beamSource = beamLine.source.shine()

    xtalGlobal, xtalLocal = beamLine.xtal.reflect(beam=beamSource)
    directGlobal = rsources.Beam(copyFrom=xtalGlobal)

    directLocal = beamLine.apertureDirect.propagate(directGlobal)
    diffractedLocal = beamLine.apertureDiff.propagate(xtalGlobal)

    outDict = {'beamSource': beamSource,
               'xtalLocal': xtalLocal,
               'directBeam': directLocal,
               'diffractedBeam': diffractedLocal}
    return outDict
rrun.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot = xrtplot.XYCPlot(
        beam="beamSource",
        xaxis=xrtplot.XYCAxis('x', 'mm'),
        yaxis=xrtplot.XYCAxis('z', 'mm'))
    plot.saveName = ["0-beamSource.png"]
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        beam="xtalLocal",
        xaxis=xrtplot.XYCAxis('x', 'mm'),
        yaxis=xrtplot.XYCAxis('y', 'mm'))
    plot.saveName = ["1-xtalLocal.png"]
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        beam="directBeam",
        xaxis=xrtplot.XYCAxis('x', 'mm'),
        yaxis=xrtplot.XYCAxis('z', 'mm'))
    plot.saveName = ["2-directBeam.png"]
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        beam="diffractedBeam",
        xaxis=xrtplot.XYCAxis('x', 'mm'),
        yaxis=xrtplot.XYCAxis('z', 'mm'))
    plot.saveName = ["3-diffractedBeam.png"]
    plots.append(plot)

    for plot in plots:
        plot.caxis.offset = E0
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'

    return plots


def main():
    beamLine = build_beamline()
    align_beamline(beamLine, E0)
    plots = define_plots(beamLine)
    xrtrun.run_ray_tracing(plots=plots, repeats=1, beamLine=beamLine)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-08-13"

Created with xrtQook




"""

import sys
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.materials.compounds as rmatsco
import xrt.backends.raycing.materials.crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun


def build_beamline():
    bl = raycing.BeamLine(
        name=r"BeamLine")

#    bl.wiggler01 = rsources.Wiggler(
#        bl=bl,
#        name=r"wiggler01",
#        nrays=8e4,
#        n=10,
##        uniformRayDensity=True,
#        xPrimeMax=1.5e-2,
#        zPrimeMax=2.5e-2,
#        center=[0.0, -3000.0, 0.0])
#
#    bl.wiggler02 = rsources.Wiggler(
#        bl=bl,
#        name=r"wiggler02",
#        nrays=5e4,
#        n=20,
#        xPrimeMax=1.5e-2,
#        zPrimeMax=2.5e-2,
##        uniformRayDensity=True,
#        center=[0.0, 3000.0, 0.0])

    bl.wiggler01 = rsources.GeometricSource(
            bl=bl,
            nrays=8e4,
            totalFlux=3e7,
            disty='normal',
            dy=0.1,
            dx=0.1,
            dz=0.1,
            dxprime=0.001,
            dzprime=0.001,
            uniformRayDensity=True,
            center=[0.0, -3000.0, 0.0]
            )

    bl.wiggler02 = rsources.GeometricSource(
            bl=bl,
            nrays=1e4,
            totalFlux=5e6,
            disty='normal',
            dy=0.1,
            uniformRayDensity=True,
            dx=0.1,
            dz=0.1,
            dxprime=0.001,
            dzprime=0.001,
            center=[0.0, 3000.0, 0.0]
            )

    bl.screen01 = rscreens.Screen(
        bl=bl,
        name=r"screen01")

    return bl


def run_process(bl):
    wiggler01_global = bl.wiggler01.shine()

    wiggler01_global.Jss *= 0.5
    wiggler01_global.Jpp *= 0.5
    wiggler01_global.Jsp *= 0.5

    wiggler02_global = bl.wiggler02.shine()
    w2_raw = rsources.Beam(copyFrom=wiggler02_global)
#    w2_raw = wiggler02_global

    wiggler02_global.concatenate(wiggler01_global)

#    screen01_local = bl.screen01.expose(
#        beam=bendingMagnet01_global)

    outDict = {
        'wiggler01_global': wiggler01_global,
        'wiggler02_global': wiggler02_global,
        'w2_raw': w2_raw,
#        'screen01_local': screen01_local
        }
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"wiggler01_global",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01-wiggler01_global")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"w2_raw",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01-wiggler02_global")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"wiggler02_global",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot03-Sum")
    plots.append(plot03)

    return plots


def main():
    BeamLine = build_beamline()
#    E0 = 0.5 * (BeamLine.bendingMagnet01.eMin +
#                BeamLine.bendingMagnet01.eMax)
#    BeamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=1,
        backend=r"raycing",
        beamLine=BeamLine)
    i1 = plots[0].flux
    i2 = plots[1].flux
    i3 = plots[2].flux

    print(f"W1 flux: {i1:.4g}, W2 flux: {i2:.4g}")
    print(f"W1+W2 from plot: {i3:.4g}, W1+W2 sum: {i1+i2:.4g}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2021-09-18"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"c:\Ray-tracing")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        dx=0.1,
        dz=0.1,
        dzprime=0.001,
        energies=[1.0])

    beamLine.oe01 = roes.OE(
        bl=beamLine,
        name=None,
        center=[0, 1000, 0],
        pitch=r"45deg",
        positionRoll=r"90deg")

    beamLine.screen00 = rscreens.Screen(
        bl=beamLine,
        center=[500, 1000, 0],
        x=[0, -1, 0])

    beamLine.ellipticalMirrorParam01 = roes.EllipticalMirrorParam(
        bl=beamLine,
        name=None,
        center=[1000, 1000, 0],
        pitch=r"45deg",
        yaw=r"-89.9deg",
        limPhysX=[-20.0, 20.0],
        limPhysY=[-20.0, 20.0],
        p=2000,
        pAxis=[1, 0, 0],
        isCylindrical=True)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[1000, 1000, 1000],
        x=[0, -1, 0],
        z=[-1, 0, 0])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    oe01beamGlobal01, oe01beamLocal01 = beamLine.oe01.reflect(
        beam=geometricSource01beamGlobal01)

    screen02beamLocal01 = beamLine.screen00.expose(
        beam=oe01beamGlobal01)

    ellipticalMirrorParam01beamGlobal01, ellipticalMirrorParam01beamLocal01 = beamLine.ellipticalMirrorParam01.reflect(
        beam=oe01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=ellipticalMirrorParam01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'oe01beamGlobal01': oe01beamGlobal01,
        'oe01beamLocal01': oe01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01,
        'ellipticalMirrorParam01beamGlobal01': ellipticalMirrorParam01beamGlobal01,
        'ellipticalMirrorParam01beamLocal01': ellipticalMirrorParam01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"oe01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"ellipticalMirrorParam01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot02")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot03")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot04")
    plots.append(plot04)
    return plots


def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()

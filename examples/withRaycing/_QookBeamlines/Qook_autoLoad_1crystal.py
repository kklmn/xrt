# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-09-20"

Created with xrtQook






"""

import numpy as np
import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials_elemental as rmatsel
import xrt.backends.raycing.materials_compounds as rmatsco
import xrt.backends.raycing.materials_crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

crystalSi01 = rmats.CrystalSi(
    a=5.4307717932001225,
    hkl=[1, 1, 1],
    d=3.1354575567115175,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"crystalSi01",
    kind=r"crystal")


def build_beamline():
    myTestBeamline = raycing.BeamLine(
        name=r"myTestBeamline")

    myTestBeamline.bendingMagnet01 = raycing.sources_synchr.BendingMagnet(
        bl=myTestBeamline,
        name=r"bendingMagnet01",
        center=[0, 0, 0],
        eE=3.0,
        eI=0.5,
        eSigmaX=94.86832980505137,
        eSigmaZ=4.47213595499958,
        eMin=9990.0,
        eMax=10010.0,
        rho=10.00692285594456)

    myTestBeamline.oe01 = raycing.oes_base.OE(
        bl=myTestBeamline,
        name=r"oe01",
        center=[0, 20000, 0],
        pitch=r"auto",
        material=crystalSi01,
        order=1)

    myTestBeamline.screen01 = rscreens.Screen(
        bl=myTestBeamline,
        name=r"screen01",
        center=[0, 21000, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[-20.0, 20.0],
        limPhysY=[-10.0, 10.0],
        cLimits=[0.0, 0.0],
        histShape=[512.0, 256.0])

    return myTestBeamline


def run_process(myTestBeamline):
    bendingMagnet01_global = myTestBeamline.bendingMagnet01.shine(
        withAmplitudes=False)

    oe01_global, oe01_local = myTestBeamline.oe01.reflect(
        beam=bendingMagnet01_global)

    screen01_local = myTestBeamline.screen01.expose(
        beam=oe01_global)

    outDict = {
        'bendingMagnet01_global': bendingMagnet01_global,
        'oe01_global': oe01_global,
        'oe01_local': oe01_local,
        'screen01_local': screen01_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01",
        fluxFormatStr=r"%g")
    plots.append(plot01)
    return plots


def main():
    myTestBeamline = build_beamline()
    myTestBeamline.explore()
    E0 = 0.5 * (myTestBeamline.bendingMagnet01.eMin +
                myTestBeamline.bendingMagnet01.eMax)
    myTestBeamline.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=myTestBeamline)


if __name__ == '__main__':
    main()

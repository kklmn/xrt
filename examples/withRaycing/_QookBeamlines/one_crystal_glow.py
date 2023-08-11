# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2022-03-09"

Created with xrtQook






"""

import numpy as np
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

crystalSi01 = rmats.CrystalSi(
    name=None)


def build_beamline():
    myTestBeamline = raycing.BeamLine()

    myTestBeamline.bendingMagnet01 = rsources.BendingMagnet(
        bl=myTestBeamline,
        name=r"BM",
        center=[0, 0, 0],
        eE=3.0,
        eI=0.5,
        eMin=9990,
        eMax=10010,
        B0=1.0)

    myTestBeamline.oe01 = roes.OE(
        bl=myTestBeamline,
        name=None,
        center=[0, 20000, 0],
        limPhysX=[-20, 20],
        limPhysY=[-20, 20],
        pitch=r"auto",
        material=crystalSi01)

    myTestBeamline.screen01 = rscreens.Screen(
        bl=myTestBeamline,
        name=None,
        center=[0, 21000, r"auto"])

    return myTestBeamline


def run_process(myTestBeamline):
    bendingMagnet01beamGlobal01 = myTestBeamline.bendingMagnet01.shine(
        withAmplitudes=False)

    oe01beamGlobal01, oe01beamLocal01 = myTestBeamline.oe01.reflect(
        beam=bendingMagnet01beamGlobal01)

    screen01beamLocal01 = myTestBeamline.screen01.expose(
        beam=oe01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'oe01beamGlobal01': oe01beamGlobal01,
        'oe01beamLocal01': oe01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    myTestBeamline.prepare_flow()
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
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
    myTestBeamline.glow()
#    E0 = 0.5 * (myTestBeamline.bendingMagnet01.eMin +
#                myTestBeamline.bendingMagnet01.eMax)
#    myTestBeamline.alignE=E0
#    plots = define_plots()
#    xrtrun.run_ray_tracing(
#        plots=plots,
#        backend=r"raycing",
#        beamLine=myTestBeamline)


if __name__ == '__main__':
    main()

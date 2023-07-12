# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2021-11-16"

Created with xrtQook






"""

# import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

showIn3D = True

BK7 = rmats.Material(elements=('Si', 'O'), quantities=(1, 2), rho=6.19,
                     refractiveIndex='RefractiveIndexINFO_BK7.csv')


def build_beamline():
    myTestBeamline = raycing.BeamLine()

    myTestBeamline.bendingMagnet01 = rsources.GeometricSource(
        bl=myTestBeamline,
        dx=0.1, dz=0.1,
        dxprime=0, dzprime=0,
        distx='normal', distz='normal',
        distE='flat',
        energies=(1.77, 3.1))

    myTestBeamline.plate01 = roes.Plate(
        bl=myTestBeamline,
        center=[0, 20, 0],
        positionRoll='180deg',
        pitch=r"22.5deg",
        wedgeAngle='-45deg',
        t=1,
        limPhysX=[-2, 2],
        limPhysY=[-2, 1.4],
        limPhysY2=[-2, 1],
        material=BK7)

    myTestBeamline.screen01 = rscreens.Screen(
        bl=myTestBeamline,
        name=None,
        center=[0, 40, r"auto"])

    return myTestBeamline


def run_process(myTestBeamline):
    bendingMagnet01beamGlobal01 = myTestBeamline.bendingMagnet01.shine(
        withAmplitudes=False)

    plate01beamGlobal01, plate01beamLocal101, plate01beamLocal201 =\
        myTestBeamline.plate01.double_refract(
                beam=bendingMagnet01beamGlobal01)

    screen01beamLocal01 = myTestBeamline.screen01.expose(
        beam=plate01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'screen01beamLocal01': screen01beamLocal01}
    if showIn3D:
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
    E0 = 0.5*(1.77+3.1)
    myTestBeamline.alignE = E0

    if showIn3D:
        myTestBeamline.glow()
    else:
        plots = define_plots()
        xrtrun.run_ray_tracing(
            plots=plots,
            backend=r"raycing",
            beamLine=myTestBeamline)


if __name__ == '__main__':
    main()

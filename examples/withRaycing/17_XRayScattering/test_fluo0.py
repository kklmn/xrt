# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2019-03-14"

Created with xrtQook




"""

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun


NiCuZn = rmats.Material(
    elements=['Ni', 'Cu', 'Zn'],
    quantities=[1, 1, 0.1],
    kind=r"plate",
    rho=8,
    t=0.5,
    name=None)

Me = 'Se'

CopperInGlass = rmats.Material(
    elements=['Si', 'O', Me],
    quantities=[1, 2, 1e-4],
    kind=r"plate",
    rho=3,
    t=0.5,
    name=None)

Cu = rmats.Material(
    elements=['Cu'],
    quantities=[1],
    kind=r"plate",
    rho=8.96,
    t=0.5,
    name=None)

Si = rmats.Material(
    elements=['Si'],
    quantities=[1],
    kind=r"plate",
    rho=2.65,
    t=0.5,
    name=None)

E0 = 14000


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.bendingMagnet01 = rsources.BendingMagnet(
        bl=beamLine,
        nrays=1e5,
        center=[0, 0, 0],
        eMin=E0-2,
        eMax=E0+2,
        xPrimeMax=0.1,
        zPrimeMax=0.1)

#    beamLine.bendingMagnet01 = rsources.GeometricSource(
#        bl=beamLine,
#        nrays=1e5,
#        center=[0, 0, 0],
#        distE='lines', #'normal',
#        distx=None, distz=None,
#        distxprime=None, distzprime=None,
##        dxprime=1e-7,
##        dzprime=1e-3,
#        energies=(E0, )
#        )
##        xPrimeMax=0.1,
##        zPrimeMax=0.1)

    beamLine.plate01 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 5000, 0],
        pitch=r"45deg",
        positionRoll=r"90deg",
        material=CopperInGlass,
#        material=NiCuZn,
#        material=Cu,
        t=0.5)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 6000, 0])

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[50, 5000, 0],
        x=[0, 1, 0],
        z=[0, 0, 1])

    beamLine.SDD = roes.Plate(
        bl=beamLine,
        name='SDD',
        center=[50, 5000, 0],
        shape='round',
        limPhysX=[-5, 5],
        limPhysY=[-5, 5],
        pitch=0,
        positionRoll=r"-90deg",
        material=Si,
        fwhm=150,  # eV
        t=0.5)

    return beamLine


def run_process(beamLine):
    bendingMagnet01beamGlobal01 = beamLine.bendingMagnet01.shine()

    plate01beamGlobal01, plate01beamLocal101, plate01beamLocal201 =\
        beamLine.plate01.double_refract(beam=bendingMagnet01beamGlobal01)

    fluoBeam, fluoBeamLocal = beamLine.plate01.scatter(
            bendingMagnet01beamGlobal01, channels=[
                                                   'inelastic',
                                                   'elastic',
                                                   (Me, 'Ka1'),
                                                   (Me, 'Ka2'),
                                                   (Me, 'Kb1'),
                                                   (Me, 'Kb3'),
                                                  ],
            withSelfAbsorption=True)
    screen01beamLocal01 = beamLine.screen01.expose(
        beam=plate01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=fluoBeam)

    detectorGlobal, detectorLocal1, detectorLocal2 =\
        beamLine.SDD.double_refract(beam=fluoBeam, returnLocalAbsorbed=0)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'fluoBeam': fluoBeam,
        'detectorGlobal': detectorGlobal,
        'detectorLocal1': detectorLocal1,
        'detectorLocal2': detectorLocal2,
        'screen01beamLocal01': screen01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01}
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"bendingMagnet01beamGlobal01", ePos=2, aspect='auto',
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=512, ppb=1,),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=512, ppb=1,),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
#            limits=[18500, 19750],
#            limits=[E0-1600, E0],
#            limits=[7250, 9500],
            bins=512, ppb=1,
            unit=r"eV"),
        title=r"Source Beam")
    plots.append(plot01)

    plot01a = xrtplot.XYCPlot(
        beam=r"fluoBeam", ePos=2, aspect='equal',
        xaxis=xrtplot.XYCAxis(
            label=r"y", bins=512, ppb=1, limits=[4999.75, 5000.25]),
        yaxis=xrtplot.XYCAxis(
            label=r"x", bins=512, ppb=1, limits=[-0.25, 0.25]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
#            limits=[8847-160, 8847+160],
            limits=[E0-3500, E0+500],
            bins=512, ppb=1,
            unit=r"eV"),
        title=r"Sample top view")
    plots.append(plot01a)

    plot01 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01", ePos=2, aspect='auto',
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=512, ppb=1,),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=512, ppb=1,),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
#            limits=[18500, 19750],
#            limits=[E0-1600, E0],
            limits=[E0-3500, E0+500],
            bins=512, ppb=1,
            unit=r"eV"),
        title=r"Side screen")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Downstream Screen")
    plots.append(plot02)

    plot02 = xrtplot.XYCPlot(
        beam=r"detectorLocal2", ePos=2, aspect='equal',
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=512, ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"y", bins=512, ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy", bins=512, ppb=1,
            limits=[E0-3500, E0+500],
            unit=r"eV"),
        title=r"Absorption on SDD")
    plots.append(plot02)

    plot02 = xrtplot.XYCPlot(
        beam=r"detectorLocal2", aspect='equal',
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=512, ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"y", bins=512, ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy", bins=512, ppb=1,
            limits=[E0-3500, E0+500],
            unit=r"eV"),
        title=r"Absorption on SDD")
    plots.append(plot02)

    for iplot, plot in enumerate(plots):
        plot.saveName = "{0} - {1}.png".format(iplot, plot.title)

    return plots


def main():
    beamLine = build_beamline()
#    E0 = 0.5 * (beamLine.bendingMagnet01.eMin +
#                beamLine.bendingMagnet01.eMax)
#    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        repeats=100,
        beamLine=beamLine)


if __name__ == '__main__':
    main()

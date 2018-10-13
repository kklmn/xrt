# -*- coding: utf-8 -*-
"""
__author__ = "Roman Chernikov", "Konstantin Klementiev"
__date__ = "2018-10-01"

Created with xrtQook




Single Crystal Diffraction
------------------
Sample script to calculate the Single Crystal Laue Diffraction pattern.



"""

import sys
import os
sys.path.append(os.path.join('..', '..', '..'))
import matplotlib as mpl  # analysis:ignore
from matplotlib import pyplot as plt  # analysis:ignore

import xrt.backends.raycing.sources as rsources  # analysis:ignore
import xrt.backends.raycing.screens as rscreens  # analysis:ignore
import xrt.backends.raycing.materials as rmats  # analysis:ignore
import xrt.backends.raycing.oes as roes  # analysis:ignore
import xrt.backends.raycing.apertures as rapts  # analysis:ignore
import xrt.backends.raycing.run as rrun  # analysis:ignore
import xrt.backends.raycing as raycing  # analysis:ignore
import xrt.plotter as xrtplot  # analysis:ignore
import xrt.runner as xrtrun  # analysis:ignore

PowderSample = rmats.Powder(
    chi=[0, 6.283185307179586],
    name='CeO2 powder',
    hkl=[5, 5, 5],
    a=5.256,
    atoms=[58, 58, 58, 58, 8, 8, 8, 8, 8, 8, 8, 8],
    atomsXYZ=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
              [0.5, 0.5, 0.0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
              [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75],
              [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75]],
    t=1.0,
    table=r"Chantler total")

MonoCrystalSample = rmats.MonoCrystal(
    Nmax=5,  # from [-5, -5, -5] to [5, 5, 5]. Updated in plot_generator
    name='Silicon',
    geom='Laue reflected',
    hkl=[0, 0, 1],
    a=5.41949,
    t=0.1,
    table=r"Chantler total")

powder = False

dSize = 150
detLimits = [-dSize, dSize]


def build_beamline():
    P02_2 = raycing.BeamLine()

    P02_2.Undulator01 = rsources.GeometricSource(
        bl=P02_2,
        nrays=5e4,
        name='source',
        polarization='horizontal',
        dx=0.5,
        dz=0.5,
        dxprime=0.005e-3,
        dzprime=0.005e-3,
        distx='normal',
        distz='normal',
        energies=(60000, 60) if powder else (1000, 100000),
        distE='normal' if powder else 'flat')

    P02_2.FSM_Source = rscreens.Screen(
        bl=P02_2,
        name=r"FSM_Source",
        center=[0, 29001, 0])

    P02_2.Sample = roes.LauePlate(
        bl=P02_2,
        name=r"CeO2 Powder Sample" if powder else "Silicon 001 wafer",
        center=[0, 65000, 0],
        pitch='90deg',
        yaw=0 if powder else '45deg',
        rotationSequence='RxRyRz',
        material=PowderSample if powder else MonoCrystalSample,
        targetOpenCL=(0, 0),
        precisionOpenCL='float32')

    P02_2.FSM_Sample = rscreens.Screen(
        bl=P02_2,
        name=r"After Sample",
        center=[0, 65100, 0])

    P02_2.RoundBeamStop01 = rapts.RoundBeamStop(
        bl=P02_2,
        name=r"BeamStop",
        center=[0, 65149, 0],
        r=5)

    P02_2.Frame = rapts.RectangularAperture(
        bl=P02_2,
        name=r"Frame",
        center=[0, 65149.5, 0],
        opening=[-dSize, dSize, -dSize, dSize])

    P02_2.FSM_Detector = rscreens.Screen(
        bl=P02_2,
        name=r"Detector",
        center=[0, 65150, 0])

    return P02_2


def run_process(P02_2):
    Undulator01beamGlobal01 = P02_2.Undulator01.shine(
        withAmplitudes=False)

    FSM_SourcebeamLocal01 = P02_2.FSM_Source.expose(
        beam=Undulator01beamGlobal01)

    SamplebeamGlobal01, SamplebeamLocal01 = P02_2.Sample.reflect(
        beam=Undulator01beamGlobal01)

    RoundBeamStop01beamLocal01 = P02_2.RoundBeamStop01.propagate(
        beam=SamplebeamGlobal01)

    Frame01beamLocal01 = P02_2.Frame.propagate(  # analysis:ignore
        beam=SamplebeamGlobal01)

    FSM_DetectorbeamLocal01 = P02_2.FSM_Detector.expose(
        beam=SamplebeamGlobal01)

    outDict = {
        'Undulator01beamGlobal01': Undulator01beamGlobal01,
        'FSM_SourcebeamLocal01': FSM_SourcebeamLocal01,
        'SamplebeamGlobal01': SamplebeamGlobal01,
        'SamplebeamLocal01': SamplebeamLocal01,
        'RoundBeamStop01beamLocal01': RoundBeamStop01beamLocal01,
        'FSM_DetectorbeamLocal01': FSM_DetectorbeamLocal01}
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    Plot01 = xrtplot.XYCPlot(
        beam=r"FSM_SourcebeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"01 - Undulator Beam at 29m",
        fluxFormatStr=r"%g",
        saveName=r"01 - Undulator Beam at 29m.png")
    plots.append(Plot01)

    Plot03 = xrtplot.XYCPlot(
        beam=r"FSM_DetectorbeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=detLimits,
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=detLimits,
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"03 - Detector",
        fluxFormatStr=r"%g"
        )
    plots.append(Plot03)

    return plots


def plot_generator(beamLine, plots):
    for n in [5]:
        if powder:
            beamLine.Sample.material.hkl = [n, n, n]
        else:
            beamLine.Sample.material.Nmax = n + 1
        plots[-1].title = "03 - Detector. Nmax={}".format(n+1)
        plots[-1].saveName = "03 - Detector Nmax {}.png".format(n+1)
#        plots[-1].persistentName = "03 - Detector Nmax {} fp32.mat".format(n+1)  # analysis:ignore
        yield


def main():
    P02_2 = build_beamline()
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        generator=plot_generator,
        generatorArgs=[P02_2, plots],
        repeats=10,
#        pickleEvery=5,  # analysis:ignore
        backend=r"raycing",
        beamLine=P02_2,
        afterScript=plotJet,
        afterScriptArgs=[plots])


def plotJet(plots):
    plt.figure(100, figsize=(12, 12))
    extent = list(plots[-1].xaxis.limits)
    extent.extend(plots[-1].yaxis.limits)
    plt.imshow(plots[-1].total2D,
               extent=extent, aspect='equal',  origin='lower',
#               norm=mpl.colors.LogNorm(),  # uncomment this to plot in log scale  # analysis:ignore
               norm=mpl.colors.PowerNorm(0.22),  # reduce gamma (say, to 0.3) to increase contrast    # analysis:ignore
               cmap='binary')
    plt.xlabel(plots[-1].xaxis.displayLabel, fontsize=12)
    plt.ylabel(plots[-1].yaxis.displayLabel, fontsize=12)
    plt.savefig("Lauegram Intensity.png")
    plt.show()


if __name__ == '__main__':
    main()

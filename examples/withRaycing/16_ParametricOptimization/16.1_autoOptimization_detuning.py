# -*- coding: utf-8 -*-
"""

__author__ = "Roman Chernikov", "Konstantin Klementiev"
__date__ = "2018-06-03"

Automatic optimization of detuning parameter in DCM for the best energy
resolution.
Typical execution time: approx. 15 min.


"""

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

Si111 = rmats.CrystalSi(
    hkl=[1, 1, 1],
    name=r"Si111")

minimizationArray = []
counter = [0]


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=10000)

    beamLine.Wiggler = rsources.Wiggler(
        bl=beamLine,
        name=r"Flat-Top Wiggler",
        center=[0, 0, 0],
        nrays=500000,
        eE=2.9,
        eI=0.25,
        eEpsilonX=18.1,
        eEpsilonZ=0.0362,
        betaX=9.1,
        betaZ=2.8,
        xPrimeMax=0.3,
        zPrimeMax=0.005,
        eMin=9995,
        eMax=10005,
        K=35,
        period=150,
        n=11)

    beamLine.Generic_DCM = roes.DCM(
        bl=beamLine,
        name=r"Generic DCM",
        center=[0, 25300, 0],
        bragg=[10000],
        material=Si111,
        material2=Si111,
        cryst2perpTransl=6.5023)

    beamLine.Aperture = rapts.RectangularAperture(
        bl=beamLine,
        name=r"Aperture",
        center=[0, 28300, r"auto"],
        opening=[-10, 10, -0.1, 0.1])

    beamLine.FSM = rscreens.Screen(
        bl=beamLine,
        name=r"FSM",
        center=[0, 30650, r"auto"])

    return beamLine


def run_process(beamLine):
    WigglerbeamGlobal01 = beamLine.Wiggler.shine()

    Generic_DCMbeamGlobal01, Generic_DCMbeamLocal101, Generic_DCMbeamLocal201 =\
        beamLine.Generic_DCM.double_reflect(
                beam=WigglerbeamGlobal01)

    AperturebeamLocal01 = beamLine.Aperture.propagate(
        beam=Generic_DCMbeamGlobal01)

    FSMFootprint = beamLine.FSM.expose(
        beam=Generic_DCMbeamGlobal01)

    outDict = {'FSMFootprint': FSMFootprint}
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    plot02 = xrtplot.XYCPlot(
        beam=r"FSMFootprint",
        xaxis=xrtplot.XYCAxis(
            fwhmFormatStr='%.2f',
            label=r"x", bins=512, ppb=1),
        yaxis=xrtplot.XYCAxis(
            fwhmFormatStr='%.2f',
            label=r"z", bins=512, ppb=1),
        caxis=xrtplot.XYCAxis(
            fwhmFormatStr="%.2f",
            label=r"energy", bins=512, ppb=1,
            unit=r"eV",
            limits=[9995, 10005]),
        aspect=r"auto",
        title=r"FSM Footprint",
        fluxFormatStr=r"%g")
    plots.append(plot02)
    return plots


def propagation_function(dTheta):
    counter[0] += 1
    beamLine = build_beamline()
    beamLine.Generic_DCM.cryst2pitch = dTheta
    plots = define_plots()
    plots[-1].title += ', Iteration {0}, dTheta={1:.2f} urad'.format(
            counter[0], dTheta*1e6)
    plots[-1].saveName = plots[-1].title + ".png"
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        processes=1,
        backend=r"raycing",
        beamLine=beamLine,
        afterScript=closePlots)
    minimizationArray.append([dTheta, plots[-1].dE, plots[-1].flux])
    return plots[-1].dE


def closePlots():
    plt.close('all')


def main():
    res = optimize.brent(propagation_function,
                         brack=(0, 1e-5, 5e-5),
                         tol=1e-3,
                         full_output=True)
    print("Output:", res)
    plt.figure('dE vs dTheta')
    plt.plot(np.array(minimizationArray)[:, 0]*1e6,
             np.array(minimizationArray)[:, 1],
             'ro', ls='')
    plt.grid()
    axes = plt.gca()
    axes.set_xlabel(r"$d\Theta$, $\mu$rad"); axes.set_ylabel("$\Delta$E, eV")
    plt.savefig("dE_vs_dTheta.png")

    plt.figure('Flux vs dTheta')
    plt.plot(np.array(minimizationArray)[:, 0]*1e6,
             np.array(minimizationArray)[:, 2],
             'go', ls='')
    plt.grid()
    axes = plt.gca()
    axes.set_xlabel(r"$d\Theta$, $\mu$rad"); axes.set_ylabel("Flux, photons/s")
    plt.savefig("Flux_vs_dTheta.png")

    plt.figure('Convergence')
    plt.plot(np.arange(len(minimizationArray)),
             np.array(minimizationArray)[:, 1],
             '-bo')
    axes = plt.gca()
    axes.set_xlabel("Iteration Nr."); axes.set_ylabel("$\Delta$E, eV")
    plt.savefig("Convergence.png")
    plt.show()

if __name__ == '__main__':
    main()

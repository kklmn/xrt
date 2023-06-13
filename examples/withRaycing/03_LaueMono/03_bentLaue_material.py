# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2023-03-29"

Created with xrtQook




"""

import numpy as np
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

#showIn3D = False
showIn3D = True

nrays = 50000
bins = 256
ppb = 1
xtallims = [-5, 5]
plotlims = [-5, 5]
thickness = 0.5
sourceDx, sourceDz = 0.5, 0.5
sourceDxPrime, sourceDzPrime = 0, 0
#sourceDxPrime, sourceDzPrime = 0, 0
E0, dE = 30000, 50
eLims = [E0-dE, E0+dE]
eLimsSource = np.linspace(E0-dE, E0+dE, 7)
#eLims = [34850, 35100]
pLaueSCM, qLaueSCM  = 5000., 25000
#Rbend = 3.00e3  # Overbend
Rbend = 5.00e3  # Optimal bend
#Rbend = 60.0e3  # Underbend
#Rbend = 1e6  # (Almost) Plain crystal
#Rbend = [10e3, 20e3, 40e3, 1e6]
#Rbend = [3e3, 5e3, 10e3, 20e3, 40e3, 1e6]
precision='float64'
targetOpenCL=None
#targetOpenCL=(0, 0)

crystalSi01 = rmats.CrystalSi(
    t=thickness,
    geom=r"Laue reflected",
    name=r"Si111")

alpha = np.radians(-35)
thetaB = crystalSi01.get_Bragg_angle(E0)
pitch = np.pi/2+thetaB+alpha


crystalSi02 = rmats.CrystalSi(
    t=thickness,
    geom=r"Laue reflected",
    name=r"Si111tt",
    volumetricDiffraction=True,
    useTT=False)


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        nrays=nrays,
        center=[-10, 0, 0],
        dx=sourceDx,
        dz=sourceDz,
        distx='flat',
        distz='flat',
        dxprime=sourceDxPrime,
        dzprime=sourceDzPrime,
        energies=eLims,
        distE='flat',
#        energies=eLimsSource,
#        distE='lines',
        )

#    beamLine.lauePlate01 = roes.LauePlate(
#        bl=beamLine,
#        center=[-10, 5000, 0],
#        pitch=[35000],
#        alpha="15deg",
#        material=crystalSi01,
#        limPhysX=[-5.0, 5.0],
#        limPhysY=[-5.0, 5.0])

    beamLine.lauePlate01 = roes.BentLaueCylinder(
        bl=beamLine,
        R=Rbend,
#        R=(pLaueSCM, qLaueSCM),
        center=[-10, pLaueSCM, 0],
        pitch=pitch, # 1.8891209547313612,
        alpha=alpha,
        material=crystalSi01,
#        targetOpenCL='auto',
#        precisionOpenCL='float64',
        limPhysX=xtallims,
        limPhysY=xtallims)

    beamLine.Mirror01 = roes.OE(
        bl=beamLine,
        center=[-10, pLaueSCM + 500, 'auto'],
        pitch=-thetaB,
        positionRoll=np.pi,
        limPhysX=[-5.0, 5.0],
        limPhysY=[-40.0, 40.0])

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        center=[-10, pLaueSCM+qLaueSCM, 'auto'])

    beamLine.geometricSource02 = rsources.GeometricSource(
        bl=beamLine,
        nrays=nrays,
        center=[10, 0, 0],
        dx=sourceDx,
        dz=sourceDz,
        distx='flat',
        distz='flat',
        dxprime=sourceDxPrime,
        dzprime=sourceDzPrime,
        energies=eLims,
        distE='flat',
#        energies=eLimsSource,
#        distE='lines'
        )

#    beamLine.lauePlate02 = roes.BentLaueCylinder(
#        bl=beamLine,
#        R=Rbend,
    beamLine.lauePlate02 = roes.BentLaue2D(
        bl=beamLine,
        Rm=-28060,
#        Rs=None,
        Rs=370,

#        R=(pLaueSCM, qLaueSCM),
        center=[10, pLaueSCM, 0],
        pitch=pitch,
        alpha=alpha,
        material=crystalSi02,
        targetOpenCL=targetOpenCL,
        precisionOpenCL=precision,
        limPhysX=xtallims,
        limPhysY=xtallims)

    beamLine.Mirror02 = roes.OE(
        bl=beamLine,
        center=[10, pLaueSCM + 500, 'auto'],
        pitch=-thetaB,
        positionRoll=np.pi,
        limPhysX=[-5.0, 5.0],
        limPhysY=[-40.0, 40.0])

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        center=[10, pLaueSCM+qLaueSCM, 'auto'])


    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    lauePlate01beamGlobal01, lauePlate01beamLocal01 = beamLine.lauePlate01.reflect(
        beam=geometricSource01beamGlobal01)

    mirror01beamGlobal01, mirror01beamLocal01 = beamLine.Mirror01.reflect(
        beam=lauePlate01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=mirror01beamGlobal01)

    geometricSource02beamGlobal01 = beamLine.geometricSource02.shine()

    lauePlate02beamGlobal01, lauePlate02beamLocal01 = beamLine.lauePlate02.reflect(
        beam=geometricSource02beamGlobal01)

    mirror02beamGlobal01, mirror02beamLocal01 = beamLine.Mirror02.reflect(
        beam=lauePlate02beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=mirror02beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'lauePlate01beamGlobal01': lauePlate01beamGlobal01,
        'lauePlate01beamLocal01': lauePlate01beamLocal01,
        'mirror01beamGlobal01': mirror01beamGlobal01,
        'mirror01beamLocal01': mirror01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01,

        'geometricSource02beamGlobal01': geometricSource02beamGlobal01,
        'lauePlate02beamGlobal01': lauePlate02beamGlobal01,
        'lauePlate02beamLocal01': lauePlate02beamLocal01,
        'mirror02beamGlobal01': mirror02beamGlobal01,
        'mirror02beamLocal01': mirror02beamLocal01,
        'screen02beamLocal01': screen02beamLocal01}

    if showIn3D:
        beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    title01 = r"01 - Geometric Model"
    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title01,
#        saveName=title01+'_R{:.1f}m.png'.format(Rbend/1000)
        )
    plots.append(plot01)

    title02=r"02 - Takagi-Taupin Model Flux"
    plot02 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title02,
#        saveName=title02+'_R{:.1f}m.png'.format(Rbend/1000)
        )
    plots.append(plot02)

    return plots

def plot_generator(plots, beamLine):
    for radius in Rbend:
        beamLine.lauePlate01.R = radius
        beamLine.lauePlate02.R = radius
        for plot in plots:
            plot.saveName=plot.title+'_R{:.1f}m.png'.format(radius/1000)

        yield

def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE=E0
#    print(beamLine.lauePlate01.pitch)

    if showIn3D:
        beamLine.glow()

    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        repeats=4,
#        processes=10,
        generator=plot_generator,
        beamLine=beamLine)


if __name__ == '__main__':
    main()

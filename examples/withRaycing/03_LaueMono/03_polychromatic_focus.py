# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2023-03-29"

Bent Laue Crystal Polychromatic Focus




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

nrays = 250000
bins = 256
ppb = 1
xtallimsx = [-0.8, 0.8]
xtallimsy = [-2, 2]
plotlims = [-1, 1]
thickness = 2
sourceDx, sourceDz = 0.5, 0.5
sourceDxPrime, sourceDzPrime = 0, 0
E0, dE = 30000, 25
eLims = [E0-2*dE, E0+dE]
eLimsSource = np.linspace(E0-2*dE, E0+dE, 7)
#eLims = [34850, 35100]
pLaueSCM, qLaueSCM  = 200., 200
#Rbend = 3.00e3  # Overbend
#Rbend = 5.00e3  # Optimal bend
Rbend = 25.0e3  # Underbend
#Rbend = 1e6  # (Almost) Plain crystal
#Rbend = [10e3, 20e3, 40e3, 1e6]
#Rbend = [3e3, 5e3, 10e3, 20e3, 40e3, 1e6]
#precision='float64'
#targetOpenCL=None
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
    name=r"Si111vd",
    volumetricDiffraction=True,
    useTT=False)

tp_x = np.array([-1, -1, 1, 1])
tp_y = np.array([-1, 1, -1, 1])
tp_z = np.array([0, 0, 0, 0])

def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        nrays=nrays,
        center=[-1, 0, 0],
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

    beamLine.lauePlate01 = roes.BentLaueCylinder(
        name='LaueCylinder',
        bl=beamLine,
        R=Rbend,
        center=[-1, pLaueSCM, 0],
        pitch=[E0],
        alpha=alpha,
        material=crystalSi01,
        limPhysX=xtallimsx,
        limPhysY=xtallimsy)

    beamLine.Mirror01 = roes.OE(
        bl=beamLine,
        center=[-1, pLaueSCM + 20, 'auto'],
        pitch=-thetaB,
        positionRoll=np.pi,
        limPhysX=[-1.0, 1.0],
        limPhysY=[-15.0, 15.0])

    beamLine.screen01n = rscreens.Screen(
        bl=beamLine,
        center=[-1, pLaueSCM+qLaueSCM, 'auto'])

    beamLine.screen01f = rscreens.Screen(
        bl=beamLine,
        center=[-1, pLaueSCM+qLaueSCM*25, 'auto'])

    beamLine.geometricSource02 = rsources.GeometricSource(
        bl=beamLine,
        nrays=nrays,
        center=[1, 0, 0],
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



    beamLine.lauePlate02 = roes.BentLaueCylinder(
        name='LaueCylinderVD',
        bl=beamLine,
        R=Rbend,
        center=[1, pLaueSCM, 0],
        pitch=[E0],
        alpha=alpha,
        material=crystalSi02,
        limPhysX=xtallimsx,
        limPhysY=xtallimsy)

    print("local_z", beamLine.lauePlate02.local_z(tp_x, tp_y))
    print("local_n", beamLine.lauePlate02.local_n(tp_x, tp_y))
    print("local_n_depth", beamLine.lauePlate02.local_n_depth(tp_x, tp_y, tp_z))
    sys.exit()

    beamLine.Mirror02 = roes.OE(
        bl=beamLine,
        center=[1, pLaueSCM + 20, 'auto'],
        pitch=-thetaB*1.001,
        positionRoll=np.pi,
        limPhysX=[-1.0, 1.0],
        limPhysY=[-15.0, 15.0])

    beamLine.screen02n = rscreens.Screen(
        bl=beamLine,
        center=[1, pLaueSCM+qLaueSCM, 'auto'])

    beamLine.screen02f = rscreens.Screen(
        bl=beamLine,
        center=[1, pLaueSCM+qLaueSCM*25, 'auto'])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    lauePlate01beamGlobal01, lauePlate01beamLocal01 = beamLine.lauePlate01.reflect(
        beam=geometricSource01beamGlobal01)

    mirror01beamGlobal01, mirror01beamLocal01 = beamLine.Mirror01.reflect(
        beam=lauePlate01beamGlobal01)

    screen01beamLocal01n = beamLine.screen01n.expose(
        beam=mirror01beamGlobal01)

    screen01beamLocal01f = beamLine.screen01f.expose(
        beam=mirror01beamGlobal01)

    geometricSource02beamGlobal01 = beamLine.geometricSource02.shine()

    lauePlate02beamGlobal01, lauePlate02beamLocal01 = beamLine.lauePlate02.reflect(
        beam=geometricSource02beamGlobal01)

    mirror02beamGlobal01, mirror02beamLocal01 = beamLine.Mirror02.reflect(
        beam=lauePlate02beamGlobal01)

    screen02beamLocal01n = beamLine.screen02n.expose(
        beam=mirror02beamGlobal01)

    screen02beamLocal01f = beamLine.screen02f.expose(
        beam=mirror02beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'lauePlate01beamGlobal01': lauePlate01beamGlobal01,
        'lauePlate01beamLocal01': lauePlate01beamLocal01,
        'mirror01beamGlobal01': mirror01beamGlobal01,
        'mirror01beamLocal01': mirror01beamLocal01,
        'screen01beamLocal01n': screen01beamLocal01n,
        'screen01beamLocal01f': screen01beamLocal01f,

        'geometricSource02beamGlobal01': geometricSource02beamGlobal01,
        'lauePlate02beamGlobal01': lauePlate02beamGlobal01,
        'lauePlate02beamLocal01': lauePlate02beamLocal01,
        'mirror02beamGlobal01': mirror02beamGlobal01,
        'mirror02beamLocal01': mirror02beamLocal01,
        'screen02beamLocal01n': screen02beamLocal01n,
        'screen02beamLocal01f': screen02beamLocal01f
        }

    if showIn3D:
        beamLine.prepare_flow()
    return outDict


rrun.run_process = run_process


def define_plots():
    plots = []

    title01a = r"01a - Surface Diffraction - divergence"
    plot01a = xrtplot.XYCPlot(
        beam=r"lauePlate01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"y", bins=bins, ppb=ppb),
        yaxis=xrtplot.XYCAxis(
            label=r"y'", bins=bins, ppb=ppb),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title01a,
        )
    plots.append(plot01a)

    title01b = r"01b - Volumetric Diffraction - divergence"
    plot01b = xrtplot.XYCPlot(
        beam=r"lauePlate02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"y", bins=bins, ppb=ppb),
        yaxis=xrtplot.XYCAxis(
            label=r"y'", bins=bins, ppb=ppb),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title01b,
        )
    plots.append(plot01b)

    title02a = r"02a - Surface Diffraction - Screen Near"
    plot02a = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01n",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title02a,
        )
    plots.append(plot02a)

    title02b = r"02b - Volumetric Diffraction - Screen Near"
    plot02b = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01n",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title02b,
        )
    plots.append(plot02b)

    title03a = r"03a - Surface Diffraction - Screen Far"
    plot03a = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01f",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title03a,
        )
    plots.append(plot03a)

    title03b = r"03b - Volumetric Diffraction - Screen Far"
    plot03b = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01f",
        xaxis=xrtplot.XYCAxis(
            label=r"x", bins=bins, ppb=ppb, limits=plotlims),
        yaxis=xrtplot.XYCAxis(
            label=r"z", bins=bins, ppb=ppb, limits=plotlims),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV", bins=bins, ppb=ppb, limits=eLims),
        aspect='auto',
        title=title03b,
        )
    plots.append(plot03b)

    return plots

#def plot_generator(plots, beamLine):
#    for radius in Rbend:
#        beamLine.lauePlate01.R = radius
#        beamLine.lauePlate02.R = radius
#        for plot in plots:
#            plot.saveName=plot.title+'_R{:.1f}m.png'.format(radius/1000)
#
#        yield


def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE = E0

    if showIn3D:
        beamLine.glow()

    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        repeats=40,
        processes='half',
#        generator=plot_generator,
        beamLine=beamLine)


if __name__ == '__main__':
    main()

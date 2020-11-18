# -*- coding: utf-8 -*-
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2017-02-01"

Created with xrtQook
"""

import numpy as np
import sys
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
# import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

Emin, Emax = 8998, 9002

crystalSi01 = rmats.CrystalSi(
    t=0.1,
    hkl=[1, 1, 1],
    useTT=True,
    calcBorrmann='TT',
    geom=r"Laue reflected")


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        name=None,
        center=[0, 0, 0],
        nrays=1000,
        distE=r"flat",
        dx=0.001, dz=0.001,
        dxprime=0, dzprime=0,
        energies=[Emin, Emax])

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        center=[0, 10000, 0])

    beamLine.lauePlate01 = roes.BentLaueCylinder(
#    beamLine.lauePlate01 = roes.LauePlate(
        bl=beamLine,
        name=None,
        center=[0, 10000, 0],
        pitch=0.0,
        R=1e4,
        crossSection='parabolic',
        material=crystalSi01,
        targetOpenCL='CPU')

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0.0, 20000, 0.0])

    return beamLine


def run_process(beamLine):
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=geometricSource01beamGlobal01)

    lauePlate01beamGlobal01, lauePlate01beamLocal01 = beamLine.lauePlate01.reflect(
        beam=geometricSource01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=lauePlate01beamGlobal01)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'lauePlate01beamGlobal01': lauePlate01beamGlobal01,
        'lauePlate01beamLocal01': lauePlate01beamLocal01,
        'screen01beamLocal01': screen01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01}
    return outDict


rrun.run_process = run_process


def align_beamline(beamLine, energy):
    geometricSource01beamGlobal01 = rsources.Beam(nrays=2)
    geometricSource01beamGlobal01.a[:] = 0
    geometricSource01beamGlobal01.b[:] = 1
    geometricSource01beamGlobal01.c[:] = 0
    geometricSource01beamGlobal01.x[:] = 0
    geometricSource01beamGlobal01.y[:] = 0
    geometricSource01beamGlobal01.z[:] = 0
    geometricSource01beamGlobal01.E[:] = energy
    geometricSource01beamGlobal01.state[:] = 1

    tmpy = beamLine.screen02.center[1]
    newx = beamLine.screen02.center[0]
    newz = beamLine.screen02.center[2]
    beamLine.screen02.center = (newx, tmpy, newz)
    print("screen02.center:", beamLine.screen02.center)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=geometricSource01beamGlobal01)
    tmpy = beamLine.lauePlate01.center[1]
    newx = beamLine.lauePlate01.center[0]
    newz = beamLine.lauePlate01.center[2]
    beamLine.lauePlate01.center = (newx, tmpy, newz)
    print("lauePlate01.center:", beamLine.lauePlate01.center)

    braggT = crystalSi01.get_Bragg_angle(energy)
    alphaT = 0 if beamLine.lauePlate01.alpha is None else beamLine.lauePlate01.alpha
    lauePitch = 0
    print("bragg, alpha:", np.degrees(braggT), np.degrees(alphaT), "degrees")

    braggT += -crystalSi01.get_dtheta(energy, alphaT)
    if crystalSi01.geom.startswith('Laue'):
        lauePitch = 0.5 * np.pi
    print("braggT:", np.degrees(braggT), "degrees")

    loBeam = rsources.Beam(copyFrom=geometricSource01beamGlobal01)
    raycing.global_to_virgin_local(
        beamLine,
        geometricSource01beamGlobal01,
        loBeam,
        center=beamLine.lauePlate01.center)
    raycing.rotate_beam(
        loBeam,
        roll=-(beamLine.lauePlate01.positionRoll + beamLine.lauePlate01.roll),
        yaw=-beamLine.lauePlate01.yaw,
        pitch=0)
    theta0 = np.arctan2(-loBeam.c[0], loBeam.b[0])
    th2pitch = np.sqrt(1. - loBeam.a[0]**2)
    targetPitch = np.arcsin(np.sin(braggT) / th2pitch) -\
        theta0
    targetPitch += alphaT + lauePitch
    beamLine.lauePlate01.pitch = targetPitch
    print("lauePlate01.pitch:", np.degrees(beamLine.lauePlate01.pitch), "degrees")

    lauePlate01beamGlobal01, lauePlate01beamLocal01 = beamLine.lauePlate01.reflect(
        beam=geometricSource01beamGlobal01)
    print("Laue Plate exit point")
    print(lauePlate01beamGlobal01.x, lauePlate01beamGlobal01.y,
          lauePlate01beamGlobal01.z)
    tmpy = beamLine.screen01.center[1]
    newx = lauePlate01beamGlobal01.x[0] +\
        lauePlate01beamGlobal01.a[0] * (tmpy - lauePlate01beamGlobal01.y[0]) /\
        lauePlate01beamGlobal01.b[0]
    newz = lauePlate01beamGlobal01.z[0] +\
        lauePlate01beamGlobal01.c[0] * (tmpy - lauePlate01beamGlobal01.y[0]) /\
        lauePlate01beamGlobal01.b[0]
    beamLine.screen01.center = (newx, tmpy, newz)
    print("screen01.center:", beamLine.screen01.center)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=lauePlate01beamGlobal01)


def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"lauePlate01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            unit='$\mu$m',
            fwhmFormatStr="%.1f",
            bins=256,
            ppb=1,
#            factor=1000,
            limits=[-5, 5]),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            unit='$\mu$m',
            fwhmFormatStr="%.1f",
#            factor=1000,
            bins=256,
            ppb=1,
            limits=[-50, 50]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            limits=[Emin, Emax]),
        aspect=r"auto",
        title=r"01 - Laue crystal Fooprint")
    plots.append(plot01)

    plot01a = xrtplot.XYCPlot(
        beam=r"lauePlate01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"y",
            unit='$\mu$m',
            fwhmFormatStr="%.1f",
#            factor=1000,
            bins=256,
            ppb=1,
            limits=[-50, 50]),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            unit='$\mu$m',
            fwhmFormatStr="%.1f",
            bins=256,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            limits=[Emin, Emax]),
        aspect=r"auto",
        title=r"01a - Laue crystal depth profile")
    plots.append(plot01a)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            fwhmFormatStr="%.1f",
            bins=256,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            fwhmFormatStr="%.3f",
            bins=256,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            limits=[Emin, Emax],
            bins=256,
            ppb=1),
        aspect=r"auto",
        title=r"02 - screen")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1,
            unit='$\mu$m',
            fwhmFormatStr="%.1f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=256,
            ppb=1,
            unit='$\mu$m',
            fwhmFormatStr="%.1f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            bins=256,
            ppb=1,
            unit=r"eV",
            limits=[Emin, Emax]),
        aspect=r"auto",
        title=r"00 - Laue crystyal - incoming beam")
    plots.append(plot03)
    for plot in plots:
        plot.saveName = plot.title + "R{0}m_t{1}mm.png".format(10, 0.1)
    return plots


def main():
    beamLine = build_beamline()
    E0 = (Emin+Emax)*0.5
    align_beamline(beamLine, E0)
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()

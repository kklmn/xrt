# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2017-10-19"

Created with xrtQook






"""

import numpy as np
import sys
sys.path.append(r"D:\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

CVD = rmats.Material(
    elements=r"C",
    kind=r"plate",
    rho=2.3,
    table=r"Chantler total",
    name=None)

Rh = rmats.Material(
    elements=r"Rh",
    kind=r"mirror",
    rho=12.41,
    name=None)

Si220 = rmats.CrystalSi(
    hkl=[2, 2, 0],
    name=None)

CVDmirror = rmats.Material(
    elements=r"C",
    kind=r"mirror",
    rho=3.5,
    name=None)


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=9001,
        alignMode=True)

    beamLine.Wiggler = rsources.Wiggler(
        K=35,
        period=150,
        n=11,
        bl=beamLine,
        center=[0, 0, 0],
        eE=2.9,
        eI=0.25,
        eEpsilonX=18.1,
        eEpsilonZ=0.0362,
        betaX=9.1,
        betaZ=2.8,
        B0=2.5,
        eMin=8995,
        eMax=9005,
        xPrimeMax=1.,
        zPrimeMax=0.375)

    beamLine.FEMask = rapts.RectangularAperture(
        bl=beamLine,
        name=r"FEMask",
        center=[0, 12000, 0],
        opening=[-16, 16, -1.75, 1.75])

    beamLine.C_Filter = roes.Plate(
        t=0.02,
        bl=beamLine,
        name=r"C Filter",
        center=[0, 13600, 0],
        pitch=np.pi/2.,
        material=CVD)

    beamLine.WhiteBeamSlits = rapts.RectangularAperture(
        bl=beamLine,
        name=r"White Beam Slits",
        center=[0, 14000, 0],
        opening=[-15, 15, -15, 15])

    beamLine.M1 = roes.ToroidMirror(
        R=7e6,
        r=69.81,
        bl=beamLine,
        name=r"M1",
        center=[0, 14600, 0],
        pitch=np.radians(0.15),
        material=Rh,
        limOptX=[-12, 12],
        limOptY=[-495, 495])

    beamLine.CM_Slits = rapts.RectangularAperture(
        bl=beamLine,
        name=r"CM Slits",
        center=[0, 15600, r"auto"],
        opening=[-5, 5, -1, 2])

    beamLine.SSRL_DCM = roes.DCM(
        bragg=[8998],
        cryst2perpTransl=6.5023,
        limPhysX2=[-20, 20],
        limPhysY2=[-1.1951, 94.0549],
        limOptX2=[-20, 20],
        limOptY2=[-1.1951, 94.0549],
        material2=Si220,
        bl=beamLine,
        name=r"SSRL DCM",
        center=[0, 25300, r"auto"],
        material=Si220,
        limPhysX=[-20, 20],
        limOptX=[-20, 20],
        limPhysY=[-72.3913, 3.8087],
        limOptY=[-72.3913, 3.8087])

    beamLine.M2Paddle = rscreens.Screen(
        bl=beamLine,
        name=r"M2 Paddle",
        center=[0, 26000, r"auto"])

    beamLine.M2 = roes.ToroidMirror(
        R=3e6,
        r=35.9,
        bl=beamLine,
        name=r"M2",
        center=[0, 26900, r"auto"],
        pitch=np.radians(-0.2),
        positionRoll=np.pi,
        limOptX=[-12, 12],
        limOptY=[-550, 550])

    beamLine.PhotonShutter = rapts.RectangularAperture(
        bl=beamLine,
        name=r"PS1",
        center=[0, 28300, r"auto"],
        opening=[-10, 10, -5, 5])

    beamLine.DBHR1 = roes.OE(
        bl=beamLine,
        name=r"DBHR1",
        center=[0, 29900, r"auto"],
        pitch=np.radians(0.18),
        material=CVDmirror,
        limPhysX=[-10, 10],
        limOptX=[-10, 10],
        limPhysY=[-75, 75],
        limOptY=[-75, 75])

    beamLine.DBHR2 = roes.OE(
        bl=beamLine,
        name=r"DBHR2",
        center=[0, 30075, r"auto"],
        pitch=np.radians(-0.18),
        positionRoll=np.pi,
        material=CVDmirror,
        limPhysX=[-10, 10],
        limOptX=[-10, 10],
        limPhysY=[-75, 75],
        limOptY=[-75, 75])

    beamLine.JJslits = rapts.RectangularAperture(
        bl=beamLine,
        name=r"JJ slits",
        center=[0, 30350, r"auto"],
        opening=[-5, 5, 0.1, 0.5])

    beamLine.SampleScreen = rscreens.Screen(
        bl=beamLine,
        name=r"Sample",
        center=[0, 30400, r"auto"])

    return beamLine


def run_process(beamLine):
    WigglerbeamGlobal01 = beamLine.Wiggler.shine()

    FEMaskbeamGlobal, FEMaskWave = beamLine.FEMask.diffract(
        beam=WigglerbeamGlobal01)

    C_FilterbeamGlobal01, C_FilterbeamLocal101, C_FilterbeamLocal201 = beamLine.C_Filter.double_refract(
        beam=FEMaskbeamGlobal)

    WBSbeamGlobal, WBSWave = beamLine.WhiteBeamSlits.diffract(
        beam=C_FilterbeamGlobal01, wave=C_FilterbeamLocal201)

    M1beamGlobal01, M1wave = beamLine.M1.diffract(
        beam=WBSbeamGlobal, wave=WBSWave)

    CMSlitsbeamGlobal, CMSlitsWave = beamLine.CM_Slits.diffract(
        beam=M1beamGlobal01)

    SSRL_DCMbeamGlobal01, SSRL_DCMbeamLocal101, SSRL_DCMbeamLocal201 =\
        beamLine.SSRL_DCM.double_reflect(beam=CMSlitsbeamGlobal)

    screen01beamLocal01 = beamLine.M2Paddle.expose_wave(
        beam=SSRL_DCMbeamGlobal01, wave=SSRL_DCMbeamLocal201)

    M2beamGlobal01, M2wave = beamLine.M2.diffract(
        beam=SSRL_DCMbeamGlobal01, wave=SSRL_DCMbeamLocal201)

    PSbeamGlobal, PSWave  = beamLine.PhotonShutter.diffract(
        beam=M2beamGlobal01, wave=M2wave)

    DBHR1beamGlobal01, DBHR1beamLocal01 = beamLine.DBHR1.reflect(
        beam=PSbeamGlobal)

    DBHR2beamGlobal01, DBHR2beamLocal01 = beamLine.DBHR2.reflect(
        beam=DBHR1beamGlobal01)

    JJslitsbeamLocal01 = beamLine.JJslits.propagate(
        beam=DBHR2beamGlobal01)

    SampleScreenbeamLocal01 = beamLine.SampleScreen.expose_wave(
        beam=DBHR2beamGlobal01, wave=JJslitsbeamLocal01)

    outDict = {
        'WigglerbeamGlobal01': WigglerbeamGlobal01,
        'C_FilterbeamGlobal01': C_FilterbeamGlobal01,
        'C_FilterbeamLocal101': C_FilterbeamLocal101,
        'C_FilterbeamLocal201': C_FilterbeamLocal201,
        'M1beamGlobal01': M1beamGlobal01,
        'SSRL_DCMbeamGlobal01': SSRL_DCMbeamGlobal01,
        'SSRL_DCMbeamLocal101': SSRL_DCMbeamLocal101,
        'SSRL_DCMbeamLocal201': SSRL_DCMbeamLocal201,
        'M2beamGlobal01': M2beamGlobal01,
        'SampleScreenbeamLocal01': SampleScreenbeamLocal01,
        'screen01beamLocal01': screen01beamLocal01,
        'DBHR1beamGlobal01': DBHR1beamGlobal01,
        'DBHR1beamLocal01': DBHR1beamLocal01,
        'DBHR2beamGlobal01': DBHR2beamGlobal01,
        'DBHR2beamLocal01': DBHR2beamLocal01,
        'JJslitsbeamLocal01': JJslitsbeamLocal01,
        'FEMaskbeamGlobal': FEMaskbeamGlobal, 
        'FEMaskWave': FEMaskWave,
        'WBSbeamGlobal': WBSbeamGlobal, 
        'WBSWave': WBSWave,
        'M1wave': M1wave, 
        'CMSlitsbeamGlobal': CMSlitsbeamGlobal, 
        'CMSlitsWave': CMSlitsWave,
        'M2wave': M2wave,
        'PSbeamGlobal': PSbeamGlobal, 
        'PSWave': PSWave}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []
    outBeams = [
        'WigglerbeamGlobal01',
        'C_FilterbeamGlobal01',
        'C_FilterbeamLocal101',
        'C_FilterbeamLocal201',
        'M1beamGlobal01',
        'SSRL_DCMbeamGlobal01',
        'SSRL_DCMbeamLocal101',
        'SSRL_DCMbeamLocal201',
        'M2beamGlobal01',
        'SampleScreenbeamLocal01',
        'screen01beamLocal01',
        'DBHR1beamGlobal01',
        'DBHR1beamLocal01',
        'DBHR2beamGlobal01',
        'DBHR2beamLocal01',
        'JJslitsbeamLocal01',
        'FEMaskbeamGlobal', 
        'FEMaskWave',
        'WBSbeamGlobal', 
        'WBSWave',
        'M1wave', 
        'CMSlitsbeamGlobal', 
        'CMSlitsWave',
        'M2wave',
        'PSbeamGlobal', 
        'PSWave']    
    for beam in outBeams:
        plot = xrtplot.XYCPlot(
            beam=beam,
            xaxis=xrtplot.XYCAxis(
                label=r"x"),
            yaxis=xrtplot.XYCAxis(
                label=r"z" if 'lobal' in beam else r"y"),
            caxis=xrtplot.XYCAxis(
                label=r"energy",
                unit=r"eV",
                bins=256,
                ppb=1),
            aspect=r"auto",
            title=r"Sample")
        plots.append(plot)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.Wiggler.eMin +
                beamLine.Wiggler.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=1,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2017-06-28"

Created with xrtQook










"""

import numpy as np
import sys
sys.path.append(r"C:\GitHub\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
import xrt.xrtglow as xrtglow
from collections import OrderedDict
from PyQt4 import QtGui

CVD = rmats.Material(
    elements=r"C",
    rho=2.3,
    name=None)

Rh = rmats.Material(
    elements=r"Rh",
    rho=12.41,
    name=None)

Si220 = rmats.CrystalSi(
    hkl=[2, 2, 0],
    name=None)


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.Wiggler = rsources.Wiggler(
        K=35,
        period=150,
        n=11,
        bl=beamLine,
        center=[0, 0, 0],
        B0=2.5,
        eMin=8995,
        eMax=9005,
        xPrimeMax=1.,
        zPrimeMax=0.175)

    beamLine.FEMask = rapts.RectangularAperture(
        bl=beamLine,
        name=None,
        center=[0, 12000, 0],
        opening=[-10, 10, -5, 5])

    beamLine.C_Filter = roes.Plate(
        material2=None,
        t=0.02,
        bl=beamLine,
        name=None,
        center=[0, 13600, 0],
        pitch=np.pi/2.,
        material=CVD)

    beamLine.WhiteBeamSlits = rapts.RectangularAperture(
        bl=beamLine,
        name=None,
        center=[0, 14000, 0],
        opening=[-10, 10, -5, 5])

    beamLine.M1 = roes.ToroidMirror(
        bl=beamLine,
        name=None,
        center=[0, 14600, 0],
        pitch=np.radians(0.2),
        material=Rh)

    beamLine.CM_Slits = rapts.RectangularAperture(
        bl=beamLine,
        name=None,
        center=[0, 15600, 0.0],
        opening=[-10, 10, -5, 5])

    beamLine.SSRL_DCM = roes.DCM(
        bragg=0.0,
        cryst2perpTransl=7,
        material2=Si220,
        bl=beamLine,
        name=None,
        center=[0, 25300, 0.0],
        material=Si220)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 26000, 0.0])

    beamLine.M2 = roes.ToroidMirror(
        bl=beamLine,
        name=None,
        center=[0, 26900, 0.0],
        pitch=np.radians(-0.2),
        positionRoll=np.pi)

    beamLine.PhotonShutter = rapts.RectangularAperture(
        bl=beamLine,
        name=None,
        center=[0, 28300, 0.0],
        opening=[-10, 10, -5, 5])

    beamLine.SampleScreen = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 30400, 0.0])

    return beamLine


def run_process(beamLine):
    WigglerbeamGlobal01 = beamLine.Wiggler.shine()

    FEMaskbeamLocal01 = beamLine.FEMask.propagate(
        beam=WigglerbeamGlobal01)

    C_FilterbeamGlobal01, C_FilterbeamLocal101, C_FilterbeamLocal201 = beamLine.C_Filter.double_refract(
        beam=WigglerbeamGlobal01)

    WhiteBeamSlitsbeamLocal01 = beamLine.WhiteBeamSlits.propagate(
        beam=C_FilterbeamGlobal01)

    M1beamGlobal01, M1beamLocal01 = beamLine.M1.reflect(
        beam=C_FilterbeamGlobal01)

    CM_SlitsbeamLocal01 = beamLine.CM_Slits.propagate(
        beam=M1beamGlobal01)

    SSRL_DCMbeamGlobal01, SSRL_DCMbeamLocal101, SSRL_DCMbeamLocal201 = beamLine.SSRL_DCM.double_reflect(
        beam=M1beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=SSRL_DCMbeamGlobal01)

    M2beamGlobal01, M2beamLocal01 = beamLine.M2.reflect(
        beam=SSRL_DCMbeamGlobal01)

    PhotonShutterbeamLocal01 = beamLine.PhotonShutter.propagate(
        beam=M2beamGlobal01)

    SampleScreenbeamLocal01 = beamLine.SampleScreen.expose(
        beam=M2beamGlobal01)

    outDict = {
        'WigglerbeamGlobal01': WigglerbeamGlobal01,
        'FEMaskbeamLocal01': FEMaskbeamLocal01,
        'C_FilterbeamGlobal01': C_FilterbeamGlobal01,
        'C_FilterbeamLocal101': C_FilterbeamLocal101,
        'C_FilterbeamLocal201': C_FilterbeamLocal201,
        'WhiteBeamSlitsbeamLocal01': WhiteBeamSlitsbeamLocal01,
        'M1beamGlobal01': M1beamGlobal01,
        'M1beamLocal01': M1beamLocal01,
        'CM_SlitsbeamLocal01': CM_SlitsbeamLocal01,
        'SSRL_DCMbeamGlobal01': SSRL_DCMbeamGlobal01,
        'SSRL_DCMbeamLocal101': SSRL_DCMbeamLocal101,
        'SSRL_DCMbeamLocal201': SSRL_DCMbeamLocal201,
        'M2beamGlobal01': M2beamGlobal01,
        'M2beamLocal01': M2beamLocal01,
        'PhotonShutterbeamLocal01': PhotonShutterbeamLocal01,
        'SampleScreenbeamLocal01': SampleScreenbeamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    return outDict


rrun.run_process = run_process


def plot_layout(rayPath):
    app = QtGui.QApplication(sys.argv)
    blViewer = xrtglow.xrtGlow(rayPath)
    blViewer.setWindowTitle("xrtGlow")
    blViewer.show()
    app.exec_()


def align_beamline(beamLine, energy):

    rayPath = []
    oeDict = OrderedDict()
    beamDict = dict()
    WigglerbeamGlobal01 = beamLine.Wiggler.shine()

    WigglerbeamGlobal01.a[0] = 0
    WigglerbeamGlobal01.b[0] = 1
    WigglerbeamGlobal01.c[0] = 0
    WigglerbeamGlobal01.x[0] = 0
    WigglerbeamGlobal01.y[0] = 0
    WigglerbeamGlobal01.z[0] = 0
    WigglerbeamGlobal01.state[:] = 1
    WigglerbeamGlobal01.E[0] = energy

    oeDict['Wiggler'] = beamLine.Wiggler
    beamDict['WigglerbeamGlobal01'] = WigglerbeamGlobal01
    tmpy = beamLine.FEMask.center[1]
    newx = beamLine.FEMask.center[0]
    newz = beamLine.FEMask.center[2]
    beamLine.FEMask.center = (newx, tmpy, newz)
    oeDict['FEMask'] = beamLine.FEMask
    print("FEMask.center:", beamLine.FEMask.center)

    FEMaskbeamLocal01 = beamLine.FEMask.propagate(
        beam=WigglerbeamGlobal01)
    FEMaskbeamLocal01toGlobal = rsources.Beam(copyFrom=FEMaskbeamLocal01)
    beamLine.FEMask.local_to_global(FEMaskbeamLocal01toGlobal)
    beamDict['FEMaskbeamLocal01toGlobal'] = FEMaskbeamLocal01toGlobal
    rayPath.append(
        ['Wiggler', 'WigglerbeamGlobal01',
         'FEMask', 'FEMaskbeamLocal01toGlobal'])
    tmpy = beamLine.C_Filter.center[1]
    newx = beamLine.C_Filter.center[0]
    newz = beamLine.C_Filter.center[2]
    beamLine.C_Filter.center = (newx, tmpy, newz)
    oeDict['C_Filter'] = beamLine.C_Filter
    print("C_Filter.center:", beamLine.C_Filter.center)

    C_FilterbeamGlobal01, C_FilterbeamLocal101, C_FilterbeamLocal201 = beamLine.C_Filter.double_refract(
        beam=WigglerbeamGlobal01)
    C_FilterbeamLocal101toGlobal = rsources.Beam(copyFrom=C_FilterbeamLocal101)
    beamLine.C_Filter.local_to_global(C_FilterbeamLocal101toGlobal)
    beamDict['C_FilterbeamGlobal01'] = C_FilterbeamGlobal01
    rayPath.append(
        ['C_Filter', 'C_FilterbeamLocal101toGlobal',
         'C_Filter', 'C_FilterbeamGlobal01'])
    rayPath.append(
        ['Wiggler', 'WigglerbeamGlobal01',
         'C_Filter', 'C_FilterbeamLocal101toGlobal'])
    beamDict['C_FilterbeamLocal101toGlobal'] = C_FilterbeamLocal101toGlobal
    tmpy = beamLine.WhiteBeamSlits.center[1]
    newx = beamLine.WhiteBeamSlits.center[0]
    newz = beamLine.WhiteBeamSlits.center[2]
    beamLine.WhiteBeamSlits.center = (newx, tmpy, newz)
    oeDict['WhiteBeamSlits'] = beamLine.WhiteBeamSlits
    print("WhiteBeamSlits.center:", beamLine.WhiteBeamSlits.center)

    WhiteBeamSlitsbeamLocal01 = beamLine.WhiteBeamSlits.propagate(
        beam=C_FilterbeamGlobal01)
    WhiteBeamSlitsbeamLocal01toGlobal = rsources.Beam(copyFrom=WhiteBeamSlitsbeamLocal01)
    beamLine.WhiteBeamSlits.local_to_global(WhiteBeamSlitsbeamLocal01toGlobal)
    beamDict['WhiteBeamSlitsbeamLocal01toGlobal'] = WhiteBeamSlitsbeamLocal01toGlobal
    rayPath.append(
        ['C_Filter', 'C_FilterbeamGlobal01',
         'WhiteBeamSlits', 'WhiteBeamSlitsbeamLocal01toGlobal'])
    tmpy = beamLine.M1.center[1]
    newx = beamLine.M1.center[0]
    newz = beamLine.M1.center[2]
    beamLine.M1.center = (newx, tmpy, newz)
    oeDict['M1'] = beamLine.M1
    print("M1.center:", beamLine.M1.center)

    M1beamGlobal01, M1beamLocal01 = beamLine.M1.reflect(
        beam=C_FilterbeamGlobal01)
    rayPath.append(
        ['C_Filter', 'C_FilterbeamGlobal01',
         'M1', 'M1beamGlobal01'])
    beamDict['M1beamGlobal01'] = M1beamGlobal01
    tmpy = beamLine.CM_Slits.center[1]
    newx = beamLine.CM_Slits.center[0]
    newz = M1beamGlobal01.z[0] +\
        M1beamGlobal01.c[0] * (tmpy - M1beamGlobal01.y[0]) /\
        M1beamGlobal01.b[0]
    beamLine.CM_Slits.center = (newx, tmpy, newz)
    oeDict['CM_Slits'] = beamLine.CM_Slits
    print("CM_Slits.center:", beamLine.CM_Slits.center)

    CM_SlitsbeamLocal01 = beamLine.CM_Slits.propagate(
        beam=M1beamGlobal01)
    CM_SlitsbeamLocal01toGlobal = rsources.Beam(copyFrom=CM_SlitsbeamLocal01)
    beamLine.CM_Slits.local_to_global(CM_SlitsbeamLocal01toGlobal)
    beamDict['CM_SlitsbeamLocal01toGlobal'] = CM_SlitsbeamLocal01toGlobal
    rayPath.append(
        ['M1', 'M1beamGlobal01',
         'CM_Slits', 'CM_SlitsbeamLocal01toGlobal'])
    tmpy = beamLine.SSRL_DCM.center[1]
    newx = beamLine.SSRL_DCM.center[0]
    newz = M1beamGlobal01.z[0] +\
        M1beamGlobal01.c[0] * (tmpy - M1beamGlobal01.y[0]) /\
        M1beamGlobal01.b[0]
    beamLine.SSRL_DCM.center = (newx, tmpy, newz)
    oeDict['SSRL_DCM'] = beamLine.SSRL_DCM
    print("SSRL_DCM.center:", beamLine.SSRL_DCM.center)

    braggT = Si220.get_Bragg_angle(energy)
    alphaT = 0 if beamLine.SSRL_DCM.alpha is None else beamLine.SSRL_DCM.alpha
    lauePitch = 0
    print("bragg, alpha:", np.degrees(braggT), np.degrees(alphaT), "degrees")

    braggT += -Si220.get_dtheta(energy, alphaT)
    if Si220.geom.startswith('Laue'):
        lauePitch = 0.5 * np.pi
    print("braggT:", np.degrees(braggT), "degrees")

    loBeam = rsources.Beam(copyFrom=M1beamGlobal01)
    raycing.global_to_virgin_local(
        beamLine,
        M1beamGlobal01,
        loBeam,
        center=beamLine.SSRL_DCM.center)
    raycing.rotate_beam(
        loBeam,
        roll=-(beamLine.SSRL_DCM.positionRoll + beamLine.SSRL_DCM.roll),
        yaw=-beamLine.SSRL_DCM.yaw,
        pitch=0)
    theta0 = np.arctan2(-loBeam.c[0], loBeam.b[0])
    th2pitch = np.sqrt(1. - loBeam.a[0]**2)
    targetPitch = np.arcsin(np.sin(braggT) / th2pitch) -\
        theta0
    targetPitch += alphaT + lauePitch
    beamLine.SSRL_DCM.bragg = targetPitch-beamLine.SSRL_DCM.pitch
    print("SSRL_DCM.bragg:", np.degrees(beamLine.SSRL_DCM.bragg), "degrees")

    SSRL_DCMbeamGlobal01, SSRL_DCMbeamLocal101, SSRL_DCMbeamLocal201 = beamLine.SSRL_DCM.double_reflect(
        beam=M1beamGlobal01)
    SSRL_DCMbeamLocal101toGlobal = rsources.Beam(copyFrom=SSRL_DCMbeamLocal101)
    beamLine.SSRL_DCM.local_to_global(SSRL_DCMbeamLocal101toGlobal)
    beamDict['SSRL_DCMbeamGlobal01'] = SSRL_DCMbeamGlobal01
    rayPath.append(
        ['SSRL_DCM', 'SSRL_DCMbeamLocal101toGlobal',
         'SSRL_DCM', 'SSRL_DCMbeamGlobal01'])
    rayPath.append(
        ['M1', 'M1beamGlobal01',
         'SSRL_DCM', 'SSRL_DCMbeamLocal101toGlobal'])
    beamDict['SSRL_DCMbeamLocal101toGlobal'] = SSRL_DCMbeamLocal101toGlobal
    tmpy = beamLine.screen01.center[1]
    newx = beamLine.screen01.center[0]
    newz = SSRL_DCMbeamGlobal01.z[0] +\
        SSRL_DCMbeamGlobal01.c[0] * (tmpy - SSRL_DCMbeamGlobal01.y[0]) /\
        SSRL_DCMbeamGlobal01.b[0]
    beamLine.screen01.center = (newx, tmpy, newz)
    oeDict['screen01'] = beamLine.screen01
    print("screen01.center:", beamLine.screen01.center)

    screen01beamLocal01_global = beamLine.screen01.expose_global(
        beam=SSRL_DCMbeamGlobal01)
    rayPath.append(
        ['SSRL_DCM', 'SSRL_DCMbeamGlobal01',
         'screen01', 'screen01beamLocal01_global'])
    beamDict['screen01beamLocal01_global'] = screen01beamLocal01_global
    tmpy = beamLine.M2.center[1]
    newx = beamLine.M2.center[0]
    newz = SSRL_DCMbeamGlobal01.z[0] +\
        SSRL_DCMbeamGlobal01.c[0] * (tmpy - SSRL_DCMbeamGlobal01.y[0]) /\
        SSRL_DCMbeamGlobal01.b[0]
    beamLine.M2.center = (newx, tmpy, newz)
    oeDict['M2'] = beamLine.M2
    print("M2.center:", beamLine.M2.center)

    M2beamGlobal01, M2beamLocal01 = beamLine.M2.reflect(
        beam=SSRL_DCMbeamGlobal01)
    rayPath.append(
        ['SSRL_DCM', 'SSRL_DCMbeamGlobal01',
         'M2', 'M2beamGlobal01'])
    beamDict['M2beamGlobal01'] = M2beamGlobal01
    tmpy = beamLine.PhotonShutter.center[1]
    newx = beamLine.PhotonShutter.center[0]
    newz = M2beamGlobal01.z[0] +\
        M2beamGlobal01.c[0] * (tmpy - M2beamGlobal01.y[0]) /\
        M2beamGlobal01.b[0]
    beamLine.PhotonShutter.center = (newx, tmpy, newz)
    oeDict['PhotonShutter'] = beamLine.PhotonShutter
    print("PhotonShutter.center:", beamLine.PhotonShutter.center)

    PhotonShutterbeamLocal01 = beamLine.PhotonShutter.propagate(
        beam=M2beamGlobal01)
    PhotonShutterbeamLocal01toGlobal = rsources.Beam(copyFrom=PhotonShutterbeamLocal01)
    beamLine.PhotonShutter.local_to_global(PhotonShutterbeamLocal01toGlobal)
    beamDict['PhotonShutterbeamLocal01toGlobal'] = PhotonShutterbeamLocal01toGlobal
    rayPath.append(
        ['M2', 'M2beamGlobal01',
         'PhotonShutter', 'PhotonShutterbeamLocal01toGlobal'])
    tmpy = beamLine.SampleScreen.center[1]
    newx = beamLine.SampleScreen.center[0]
    newz = M2beamGlobal01.z[0] +\
        M2beamGlobal01.c[0] * (tmpy - M2beamGlobal01.y[0]) /\
        M2beamGlobal01.b[0]
    beamLine.SampleScreen.center = (newx, tmpy, newz)
    oeDict['SampleScreen'] = beamLine.SampleScreen
    print("SampleScreen.center:", beamLine.SampleScreen.center)

    SampleScreenbeamLocal01_global = beamLine.SampleScreen.expose_global(
        beam=M2beamGlobal01)
    rayPath.append(
        ['M2', 'M2beamGlobal01',
         'SampleScreen', 'SampleScreenbeamLocal01_global'])
    beamDict['SampleScreenbeamLocal01_global'] = SampleScreenbeamLocal01_global

    plot_layout([rayPath, beamDict, oeDict])

def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"SampleScreenbeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            limits=[8995, 9005],
            bins=256,
            ppb=1),
        title=r"Sample")
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
        title=r"Beam after DCM",
        saveName=r"Beam after DCM.png")
    plots.append(plot02)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.Wiggler.eMin +
                beamLine.Wiggler.eMax)
    align_beamline(beamLine, E0)


if __name__ == '__main__':
    main()

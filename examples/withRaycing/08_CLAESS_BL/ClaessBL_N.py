# -*- coding: utf-8 -*-
"""This module describes the beamline CLÆSS to be imported by
``traceClaessBL.py`` script."""
__author__ = "Konstantin Klementiev"
__date__ = "1 Mar 2012"

import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

showIn3D = False

#targetOpenCL = (0, 0)
targetOpenCL = None


def build_beamline(nrays=raycing.nrays, eMinRays=550, eMaxRays=30550):
    x0, y0 = -30095.04, -30102.65  # straight section center
    xFar, yFar = -59999.67, -198.017
    height = 1400.
    azimuth = np.arctan2(xFar - x0, yFar - y0)

    stripeSi = rm.Material('Si', rho=2.33)
    stripeRh = rm.Material('Rh', rho=12.41)
    stripePt = rm.Material('Pt', rho=21.45)
    si111_1 = rm.CrystalSi(hkl=(1, 1, 1), tK=-171+273.15)
    si311_1 = rm.CrystalSi(hkl=(3, 1, 1), tK=-171+273.15)
    si111_2 = rm.CrystalSi(hkl=(1, 1, 1), tK=-140+273.15)
    si311_2 = rm.CrystalSi(hkl=(3, 1, 1), tK=-140+273.15)
    filterDiamond = rm.Material('C', rho=3.52, kind='plate')
    beamLine = raycing.BeamLine(azimuth=azimuth, height=height)
    wigglerToStraightSection = 645.0
    xWiggler = x0 + wigglerToStraightSection * beamLine.sinAzimuth
    yWiggler = y0 + wigglerToStraightSection * beamLine.cosAzimuth

    rs.Wiggler(beamLine, name='MPW80', center=(
        xWiggler, yWiggler, height),
        nrays=nrays, period=80., K=13, n=12, eE=3., eI=0.4,
        eSigmaX=200, eSigmaZ=15, eEpsilonX=4.3, eEpsilonZ=0.043,
        eMin=eMinRays, eMax=eMaxRays,
        xPrimeMax=1.5, zPrimeMax=0.25)
#    rs.GeometricSource(beamLine, 'Source', (xWiggler, yWiggler, height),
#      nrays=nrays, distE='flat', energies=(2000, 25000),
#        polarization='horizontal')

    sampleToStraightSection = 36554.1
    beamLine.xSample = x0 + sampleToStraightSection * beamLine.sinAzimuth
    beamLine.ySample = y0 + sampleToStraightSection * beamLine.cosAzimuth

    beamLine.feFixedMask = ra.RectangularAperture(
        beamLine, 'FEFixedMask', (-36876.69, -23321.01, height),
        ('left', 'right', 'bottom', 'top'), [-8.3, 8.3, -2.35, 2.35])
    beamLine.fePhotonShutter = ra.RectangularAperture(
        beamLine, 'FEPhotonShutter', (-37133.37, -23064.33, height),
        ('left', 'right', 'bottom', 'top'), [-10.5, 10.5, -8.0, 8.0],
        alarmLevel=0.)
    beamLine.feMovableMaskLT = ra.RectangularAperture(
        beamLine, 'FEMovableMaskLT', (-38979.62, -21218.07, height),
        ('left', 'top'), [-10, 3.], alarmLevel=0.5)
    beamLine.feMovableMaskRB = ra.RectangularAperture(
        beamLine, 'FEMovableMaskRB', (-39262.47, -20935.23, height),
        ('right', 'bottom'), [10, -3.], alarmLevel=0.5)

    beamLine.filter1 = roe.Plate(
        beamLine, 'Filter1',
        (-42740.918, -17456.772, height), pitch=np.pi/2,
        limPhysX=(-20., 20.), limPhysY=(-9., 9.),
        surface=(r'diamond 90 $\mu$m',), material=(filterDiamond,), t=0.09,
        targetOpenCL=targetOpenCL,
        alarmLevel=0.)

    beamLine.fsm1 = rsc.Screen(
        beamLine, 'DiamondFSM1', (-42920.68, -17277.01, height),
        compressX=1./2.44)

    beamLine.vcm = roe.VCM(
        beamLine, 'VCM', [-43819.49, -16378.20, height],
        surface=('Rh', 'Si', 'Pt'), material=(stripeRh, stripeSi, stripePt),
        limPhysX=(-53., 53.), limPhysY=(-655., 655.),
        limOptX=((-47., -15.5, 16.), (-16., 15.5, 47.)),
        limOptY=((-650., -655., -650.), (650., 655., 650.)),
        R=5.0e6,
        jack1=[-43328.05, -16869.64, 973.0732],
        jack2=[-44403.38, -15964.02, 973.0732],
        jack3=[-44233.68, -15794.31, 973.0732],
        tx1=[0.0, -695.], tx2=[0.0, 705.75],
        targetOpenCL=targetOpenCL,
        alarmLevel=0.)
    beamLine.fsm2 = rsc.Screen(
        beamLine, 'NormalFSM2', (-44745.34, -15452.36, height),
        compressX=1./2.44)

    beamLine.dcm = roe.DCMOnTripodWithOneXStage(
        beamLine, 'DCM',
        [-45342.09, -14855.6, 1415.], surface=('Si311', 'Si111'),
        material=(si311_1, si111_1), material2=(si311_2, si111_2),
        limPhysX=((-51.1, 6.1), (-6.1, 51.1)), limPhysY=(-30., 30.),
        cryst2perpTransl=20., cryst2longTransl=95.,
        limPhysX2=((8.6, -48.6), (48.6, -8.6)), limPhysY2=(-90., 90.),
        jack1=[-45052.88, -15079.04, 702.4973],
        jack2=[-44987.82, -14490.02, 702.4973],
        jack3=[-45576.85, -14555.08, 702.4973],
        targetOpenCL=targetOpenCL,
        alarmLevel=0.)
    beamLine.fsm3 = rsc.Screen(
        beamLine, 'DiamondFSM3', (-46625.89, -13571.81, height),
        compressX=1./2.44)

    beamLine.BSBlock = ra.RectangularAperture(
        beamLine, 'BSBlock',
        (-45988.52, -14209.17, height), ('bottom',), (22,), alarmLevel=0.)
    beamLine.slitAfterDCM_LR = ra.RectangularAperture(
        beamLine, 'SlitAfterDCM_LR', (-46095.65, -14102.04, height),
        ('left', 'right'), [-25.0, 25.0], alarmLevel=0.5)
    beamLine.slitAfterDCM_BT = ra.RectangularAperture(
        beamLine, 'SlitAfterDCM_BT', (-46107.67, -14090.02, height),
        ('bottom', 'top'), [27.0, 77.0], alarmLevel=0.5)
    foilsZActuatorOffset = 0
    beamLine.xbpm4foils = ra.SetOfRectangularAperturesOnZActuator(
        beamLine, 'XBPM4foils', (-46137.73, -14059.97, height),
        (u'Cu5µm', u'Al7µm', u'Al0.8µm', 'top-edge'),
        (1344.607 + foilsZActuatorOffset, 1366.607 + foilsZActuatorOffset,
         1388.607 + foilsZActuatorOffset, 1400. + foilsZActuatorOffset),
        (45, 45, 45), (8, 8, 8), alarmLevel=0.)

    beamLine.vfm = roe.DualVFM(
        beamLine, 'VFM', [-47491.364, -12706.324, 1449.53],
        surface=('Rh', 'Pt'), material=(stripeRh, stripePt),
        limPhysX=(-56., 56.), limPhysY=(-714., 714.),
        limOptX=((1., -46.), (46., -4.)),
        limOptY=((-712., -712.), (712., 712.)),
        positionRoll=np.pi, R=5.0e6,
        r1=70., xCylinder1=23.5, hCylinder1=3.7035,
        r2=35.98, xCylinder2=-25.0, hCylinder2=6.9504,
        jack1=[-46987.20, -13210.49, 1272.88],
        jack2=[-48062.53, -12304.87, 1272.88],
        jack3=[-47892.83, -12135.16, 1272.88],
        tx1=[0.0, -713.], tx2=[0.0, 687.75],
        targetOpenCL=targetOpenCL,
        alarmLevel=0.2)
    beamLine.fsm4 = rsc.Screen(
        beamLine, 'DiamondFSM4', (-48350.17, -11847.53, height),
        compressX=1./2.44)

    beamLine.ohPSFront = ra.RectangularAperture(
        beamLine, 'OH-PS-FrontCollimator', (-48592.22, -11605.47, height),
        ('left', 'right', 'bottom', 'top'), (-23., 23., 30.48, 79.92),
        alarmLevel=0.2)
    beamLine.ohPSBack = ra.RectangularAperture(
        beamLine, 'OH-PS-BackCollimator', (-48708.19, -11489.51, height),
        ('left', 'right', 'bottom', 'top'), (-23., 23., 31.1, 81.1),
        alarmLevel=0.)
    beamLine.eh100To40Flange = ra.RoundAperture(
        beamLine, 'eh100To40Flange', [-53420.63, -6777.058, height], 19.,
        alarmLevel=0.)
    eh100To40FlangeToslit = 1159.
    slitX = beamLine.eh100To40Flange.center[0] +\
        eh100To40FlangeToslit * np.sin(azimuth)
    slitY = beamLine.eh100To40Flange.center[1] +\
        eh100To40FlangeToslit * np.cos(azimuth)
    beamLine.slitEH = ra.RectangularAperture(
        beamLine, 'slitEH', (slitX, slitY, height),
        ('left', 'right', 'bottom', 'top'), [-5, 5, -2.5, 2.5], alarmLevel=0.5)
    beamLine.fsmAtSample = rsc.Screen(
        beamLine, 'FocusAtSample',
        (beamLine.xSample, beamLine.ySample, height))

    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    for i in range(len(beamLine.sources)):
        curSource = beamLine.sources[i].shine()
        if i == 0:
            beamSource = curSource
        else:
            beamSource.concatenate(curSource)
        if shineOnly1stSource:
            break

    beamTmp1 = beamLine.feFixedMask.propagate(beamSource)
    beamTmp2 = beamLine.fePhotonShutter.propagate(beamSource)
    beamTmp3 = beamLine.feMovableMaskLT.propagate(beamSource)
    beamTmp4 = beamLine.feMovableMaskRB.propagate(beamSource)

    beamFilter1global, beamFilter1local1, beamFilter1local2 = \
        beamLine.filter1.double_refract(beamSource)

    beamFurtherDown = beamFilter1global
#    beamFurtherDown = beamSource
    beamFSM1 = beamLine.fsm1.expose(beamFurtherDown)
    beamVCMglobal, beamVCMlocal = beamLine.vcm.reflect(beamFurtherDown)
    beamFSM2 = beamLine.fsm2.expose(beamVCMglobal)
    beamDCMglobal, beamDCMlocal1, beamDCMlocal2 = \
        beamLine.dcm.double_reflect(beamVCMglobal)

    beamBSBlocklocal = beamLine.BSBlock.propagate(beamDCMglobal)
    beamTmp5 = beamLine.slitAfterDCM_LR.propagate(beamDCMglobal)
    beamTmp6 = beamLine.slitAfterDCM_BT.propagate(beamDCMglobal)
    beamXBPMlocal = beamLine.xbpm4foils.propagate(beamDCMglobal)
    beamFSM3 = beamLine.fsm3.expose(beamDCMglobal)

    beamVFMglobal, beamVFMlocal = beamLine.vfm.reflect(beamDCMglobal)
    beamFSM4 = beamLine.fsm4.expose(beamVFMglobal)

    beamPSFrontLocal = beamLine.ohPSFront.propagate(beamVFMglobal)
    beamTmp7 = beamLine.ohPSBack.propagate(beamVFMglobal)

    beam100To40FlangeLocal = beamLine.eh100To40Flange.propagate(beamVFMglobal)
    beamSlitEHLocal = beamLine.slitEH.propagate(beamVFMglobal)
    beamFSMSample = beamLine.fsmAtSample.expose(beamVFMglobal)

    outDict = {'beamSource': beamSource,
               'beamFilter1global': beamFilter1global,
               'beamFilter1local1': beamFilter1local1,
               'beamFilter1local2': beamFilter1local2,
               'beamFSM1': beamFSM1,
               'beamVCMglobal': beamVCMglobal, 'beamVCMlocal': beamVCMlocal,
               'beamFSM2': beamFSM2,
               'beamDCMglobal': beamDCMglobal,
               'beamDCMlocal1': beamDCMlocal1, 'beamDCMlocal2': beamDCMlocal2,
               'beamBSBlocklocal': beamBSBlocklocal,
               'beamXBPMlocal': beamXBPMlocal, 'beamFSM3': beamFSM3,
               'beamVFMglobal': beamVFMglobal, 'beamVFMlocal': beamVFMlocal,
               'beamFSM4': beamFSM4,
               'beamPSFrontLocal': beamPSFrontLocal,
               'beam100To40FlangeLocal': beam100To40FlangeLocal,
               'beamSlitEHLocal': beamSlitEHLocal,
               'beamFSMSample': beamFSMSample
               }
    beamLine.beams = outDict

    if showIn3D:
        beamLine.prepare_flow()
    return outDict

rr.run_process = run_process


def align_beamline(
    beamLine, hDiv=1.5e-3, vDiv=2.5e-4, nameVCMstripe='Rh', pitch=None,
    nameDCMcrystal='Si111', bragg=None, energy=9000., fixedExit=20.,
        nameDiagnFoil='top-edge', nameVFMcylinder='Rh', heightVFM=None):

    beamLine.feMovableMaskLT.set_divergence(
        beamLine.sources[0], [-hDiv/2, vDiv/2])
    beamLine.feMovableMaskRB.set_divergence(
        beamLine.sources[0], [hDiv/2, -vDiv/2])
    print('for {0:.2f}h x {1:.2f}v mrad^2 divergence:'.format(
          hDiv*1e3, vDiv*1e3))
    print('full feMovableMaskLT.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          beamLine.feMovableMaskLT.opening))
    print('full feMovableMaskRB.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          beamLine.feMovableMaskRB.opening))
    print('half feMovableMaskLT.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          [op/2 for op in beamLine.feMovableMaskLT.opening]))
    print('half feMovableMaskRB.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          [op/2 for op in beamLine.feMovableMaskRB.opening]))

    beamLine.vfm.select_surface(nameVFMcylinder)
    p = raycing.distance_xy(beamLine.vfm.center, beamLine.sources[0].center)
    if pitch is None:
        sinPitch = beamLine.vfm.r * 1.5 / p
        pitch = float(np.arcsin(sinPitch))
    else:
        sinPitch = np.sin(pitch)
    beamLine.vfm.R = p / sinPitch

    beamLine.vcm.select_surface(nameVCMstripe)
    beamLine.vcm.pitch = pitch
    beamLine.vcm.set_jacks()
    p = raycing.distance_xy(beamLine.vcm.center, beamLine.sources[0].center)
    beamLine.vcm.R = 2. * p / sinPitch
    beamLine.vcm.get_orientation()
    print('VCM.p = {0:.1f}'.format(p))
    print('VCM.pitch = {0:.6f} mrad'.format(beamLine.vcm.pitch*1e3))
    print('VCM.roll = {0:.6f} mrad'.format(beamLine.vcm.roll*1e3))
    print('VCM.yaw = {0:.6f} mrad'.format(beamLine.vcm.yaw*1e3))
    print('VCM.z = {0:.3f}'.format(beamLine.vcm.center[2]))
    print('VCM.R = {0:.0f}'.format(beamLine.vcm.R))
    print('VCM.dx = {0:.3f}'.format(beamLine.vcm.dx))
    print('VCM.jack1.zCalib = {0:.3f}'.format(beamLine.vcm.jack1Calib))
    print('VCM.jack2.zCalib = {0:.3f}'.format(beamLine.vcm.jack2Calib))
    print('VCM.jack3.zCalib = {0:.3f}'.format(beamLine.vcm.jack3Calib))
    print('VCM.tx1 = {0:.3f}'.format(beamLine.vcm.tx1[0]))
    print('VCM.tx2 = {0:.3f}'.format(beamLine.vcm.tx2[0]))

    p = raycing.distance_xy(beamLine.fsm2.center, beamLine.vcm.center)
    fsm2height = beamLine.height + p * np.tan(2 * pitch)

    beamLine.dcm.select_surface(nameDCMcrystal)
    p = raycing.distance_xy(beamLine.dcm.center, beamLine.vcm.center)
    beamLine.dcm.center[2] = beamLine.height + p * np.tan(2 * pitch)
    beamLine.dcm.set_jacks()
    aCrystal = beamLine.dcm.material[
        beamLine.dcm.surface.index(nameDCMcrystal)]
    dSpacing = aCrystal.d
    print('DCM.crystal = {0}'.format(nameDCMcrystal))
    print('DCM.dSpacing = {0:.6f} angsrom'.format(dSpacing))
    if bragg is None:
        theta = float(np.arcsin(rm.ch / (2 * dSpacing * energy)))
        bragg = theta + 2 * pitch
    else:
        theta = bragg - 2 * pitch
        energy = rm.ch / (2 * dSpacing * np.sin(theta))
    print('DCM.energy = {0:.3f} eV'.format(energy))
    print('DCM.bragg = {0:.6f} deg'.format(np.degrees(bragg)))
    print('DCM.realThetaAngle = DCM.bragg - 2*VCM.pitch = {0:.6f} deg'.format(
          np.degrees(theta)))
    beamLine.dcm.energy = energy
    beamLine.dcm.bragg = bragg
    p = raycing.distance_xy(beamLine.vfm.center, beamLine.vcm.center)
    if heightVFM is not None:
        fixedExit = (heightVFM - beamLine.height -
                     p * np.tan(2 * pitch)) * np.cos(2 * pitch)
    else:
        heightVFM = fixedExit / np.cos(2 * pitch) + \
            beamLine.height + p * np.tan(2 * pitch) + 0.5

    beamLine.heightVFM = heightVFM
    beamLine.dcm.fixedExit = fixedExit
    beamLine.dcm.cryst2perpTransl = \
        beamLine.dcm.fixedExit/2./np.cos(beamLine.dcm.bragg)
    beamLine.dcm.get_orientation()
    print('DCM.pitch = {0:.6f} mrad'.format(beamLine.dcm.pitch*1e3))
    print('DCM.roll = {0:.6f} mrad'.format(beamLine.dcm.roll*1e3))
    print('DCM.yaw = {0:.6f} mrad'.format(beamLine.dcm.yaw*1e3))
    print('DCM.z = {0:.3f}'.format(beamLine.dcm.center[2]))
    print('DCM.fixedExit = {0:.3f}'.format(fixedExit))
    print('DCM.cryst2perpTransl = {0:.3f}'.format(
          beamLine.dcm.cryst2perpTransl))
    print('DCM.dx = {0:.3f}'.format(beamLine.dcm.dx))
    print('DCM.jack1.zCalib = {0:.3f}'.format(beamLine.dcm.jack1Calib))
    print('DCM.jack2.zCalib = {0:.3f}'.format(beamLine.dcm.jack2Calib))
    print('DCM.jack3.zCalib = {0:.3f}'.format(beamLine.dcm.jack3Calib))

    p = raycing.distance_xy(beamLine.vfm.center, beamLine.fsm3.center)
    fsm3height = heightVFM - p * np.tan(2 * pitch)

    p = raycing.distance_xy(
        beamLine.vfm.center, (beamLine.xbpm4foils.center[0],
                              beamLine.xbpm4foils.center[1]))
    heightXBPM4 = heightVFM - p * np.tan(2 * pitch)
    beamLine.xbpm4foils.select_aperture(nameDiagnFoil, heightXBPM4)
    print('beamLine.xbpm4foils.zActuator = {0:.3f}'.format(
          beamLine.xbpm4foils.zActuator))

    beamLine.vfm.pitch = -pitch
    beamLine.vfm.center[2] = heightVFM - beamLine.vfm.hCylinder
    beamLine.vfm.set_jacks()
#    beamLine.vfm.yaw = 50e-6
    beamLine.vfm.set_x_stages()
    beamLine.vfm.get_orientation()
    print('VFM.pitch = {0:.6f} mrad'.format(beamLine.vfm.pitch*1e3))
    print('VFM.roll = {0:.6f} mrad'.format(beamLine.vfm.roll*1e3))
    print('VFM.yaw = {0:.6f} mrad'.format(beamLine.vfm.yaw*1e3))
    print('VFM.z = {0:.3f}'.format(beamLine.vfm.center[2]))
    print('VFM.R = {0:.0f}'.format(beamLine.vfm.R))
    print('VFM.r = {0:.3f}'.format(beamLine.vfm.r))
    print('VFM.dx = {0:.3f}'.format(beamLine.vfm.dx))
    print('VFM.jack1.zCalib = {0:.3f}'.format(beamLine.vfm.jack1Calib))
    print('VFM.jack2.zCalib = {0:.3f}'.format(beamLine.vfm.jack2Calib))
    print('VFM.jack3.zCalib = {0:.3f}'.format(beamLine.vfm.jack3Calib))
    print('VFM.tx1 = {0:.3f}'.format(beamLine.vfm.tx1[0]))
    print('VFM.tx2 = {0:.3f}'.format(beamLine.vfm.tx2[0]))

    beamLine.eh100To40Flange.center[2] = heightVFM
    print('eh100To40Flange.center[2] = {0:.3f}'.format(
          beamLine.eh100To40Flange.center[2]))

    beamLine.slitEH.opening[2] += heightVFM - beamLine.height
    beamLine.slitEH.opening[3] += heightVFM - beamLine.height
    beamLine.slitEH.set_optical_limits()

    print('fsm1.z = {0:.3f}'.format(beamLine.height))
    print('fsm2.z = {0:.3f}'.format(fsm2height))
    print('fsm3.z = {0:.3f}'.format(fsm3height))
    print('fsm4.z = {0:.3f}'.format(heightVFM))


def print_beamline_positions(
    hDiv=1.5e-3, vDiv=2.5e-4, nameVCMstripe='Rh', pitch=None,
    nameDCMcrystal='Si111', bragg=None, energy=9000., fixedExit=20.,
        nameDiagnFoil='top-edge', nameVFMcylinder='Rh', heightVFM=None):
    import time
    time_start = time.time()

    beamLine = build_beamline(nrays=25000)
    print(len(beamLine.slits), ' apertures')
    print(len(beamLine.oes), ' oes')
    print(len(beamLine.screens), ' screens')
    print()
    align_beamline(beamLine, hDiv, vDiv, nameVCMstripe, pitch, nameDCMcrystal,
                   bragg, energy, fixedExit, nameDiagnFoil, nameVFMcylinder,
                   heightVFM)

    run_process(beamLine)

    print('when slitAfterDCM touches the beam:')
    beamLine.slitAfterDCM_LR.touch_beam(beamLine.beams['beamDCMglobal'])
    print('beamLine.slitAfterDCM_LR.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          beamLine.slitAfterDCM_LR.opening))
    beamLine.slitAfterDCM_BT.touch_beam(beamLine.beams['beamDCMglobal'])
    print('beamLine.slitAfterDCM_BT.opening = [{0[0]:.3f}, {0[1]:.3f}]'.format(
          beamLine.slitAfterDCM_BT.opening))
    print('when slitEH touches the beam:')
    beamLine.slitEH.touch_beam(beamLine.beams['beamVFMglobal'])
    print("beamLine.slitEH.opening", beamLine.slitEH.opening)
    print(('beamLine.slitEH.opening (l, r, b, t) = [{0[0]:.3f}, {0[1]:.3f}' +
           ', {0[2]:.3f}, {0[3]:.3f}]').format(beamLine.slitEH.opening))

    time_finish = time.time()
    print('finished in {0:.1f} seconds'.format(time_finish-time_start))


if __name__ == '__main__':
    print_beamline_positions(
        hDiv=1.5e-3, vDiv=2.5e-4, nameVCMstripe='Rh',
        nameDCMcrystal='Si111', energy=9000., fixedExit=25.,
        nameDiagnFoil=u'Cu5µm', nameVFMcylinder='Rh')

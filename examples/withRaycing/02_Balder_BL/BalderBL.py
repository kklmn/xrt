# -*- coding: utf-8 -*-
"""This module describes the beamline CLÆSS to be imported by
``traceBalderBL.py`` script."""
#__name__ = "BalderBL"

import math
import numpy as np
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc


stripeSi = rm.Material('Si', rho=2.33)
stripeSiO2 = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)
stripeIr = rm.Material('Ir', rho=22.42)


def build_beamline(nrays=raycing.nrays, hkl=(1, 1, 1), stripe='Si',
                   eMinRays=None, eMaxRays=None):
    filterDiamond = rm.Material('C', rho=3.52, kind='plate')
    if stripe.startswith('S'):
        materialVCM = stripeSi
        materialVFM = stripeSiO2
    elif stripe.startswith('I'):
        materialVCM = stripeIr
        materialVFM = stripeIr
    else:
        raise ('Don''t know the mirror material')
    si_1 = rm.CrystalSi(hkl=hkl, tK=-171+273.15)
    si_2 = rm.CrystalSi(hkl=hkl, tK=-140+273.15)
    height = 0
    beamLine = raycing.BeamLine(azimuth=0, height=height)

    wigglerToStraightSection = 0
    xWiggler = wigglerToStraightSection * beamLine.sinAzimuth
    yWiggler = wigglerToStraightSection * beamLine.cosAzimuth
    if eMinRays is None:
        eMinRays = 50.
    if eMaxRays is None:
        eMaxRays = 60050.
#    rs.WigglerWS(
#        beamLine, name='SoleilW50', center=(xWiggler, yWiggler, height),
#        nrays=nrays, period=50., K=8.446, n=39, eE=3., eI=0.5,
#        eSigmaX=48.66, eSigmaZ=6.197, eEpsilonX=0.263, eEpsilonZ=0.008,
#        eMin=50, eMax=60050, eMinRays=eMinRays, eMaxRays=eMaxRays, eN=2000,
#        xPrimeMax=0.22, zPrimeMax=0.06, nx=40, nz=10)
    rs.Wiggler(
        beamLine, name='SoleilW50', center=(xWiggler, yWiggler, height),
        nrays=nrays, period=50., K=8.446, n=39, eE=3., eI=0.5,
        eSigmaX=48.66, eSigmaZ=6.197, eEpsilonX=0.263, eEpsilonZ=0.008,
        eMin=eMinRays, eMax=eMaxRays, xPrimeMax=0.22, zPrimeMax=0.06)

    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0', (0, 15000, height))
    beamLine.feFixedMask = ra.RectangularAperture(
        beamLine, 'FEFixedMask', (0, 15750, height),
        ('left', 'right', 'bottom', 'top'), [-3.15, 3.15, -0.7875, 0.7875])
    beamLine.fsmFE = rsc.Screen(beamLine, 'FSM-FE', (0, 16000, height))

    beamLine.filter1 = roe.Plate(
        beamLine, 'Filter1', (0, 23620, height),
        pitch=math.pi/2, limPhysX=(-9., 9.), limPhysY=(-4., 4.),
        surface='diamond 60 $\mu$m', material=filterDiamond, t=0.06,
        alarmLevel=0.)
    if stripe.startswith('I'):
        beamLine.filter2 = roe.Plate(
            beamLine, 'Filter2', (0, 23720, height),
            pitch=math.pi/2, limPhysX=(-9., 9.), limPhysY=(-4., 4.),
            surface='diamond 0.4 mm', material=filterDiamond, t=0.4,
            alarmLevel=0.)

    beamLine.vcm = roe.SimpleVCM(
        beamLine, 'VCM', [0, 25290, height],
        surface=('Si',), material=(materialVCM,),
        limPhysX=(-15., 15.), limPhysY=(-680., 680.), limOptX=(-6, 6),
        limOptY=(-670., 670.), R=5.0e6, pitch=2e-3, alarmLevel=0.)
    beamLine.fsmVCM = rsc.Screen(beamLine, 'FSM-VCM', (0, 26300, height))

    beamLine.dcm = roe.DCM(
        beamLine, 'DCM', [0, 27060, height], surface=('Si111',),
        material=(si_1,), material2=(si_2,),
        limPhysX=(-10, 10), limPhysY=(-30, 30),
        cryst2perpTransl=20, cryst2longTransl=65,
        limPhysX2=(-10, 10), limPhysY2=(-90, 90), alarmLevel=0.)

    beamLine.BSBlock = ra.RectangularAperture(
        beamLine, 'BSBlock', (0, 29100, height), ('bottom',),
        (22,), alarmLevel=0.)
    beamLine.slitAfterDCM = ra.RectangularAperture(
        beamLine, 'SlitAfterDCM', (0, 29200, height),
        ('left', 'right', 'bottom', 'top'), [-7, 7, -2, 2], alarmLevel=0.5)
    beamLine.fsmDCM = rsc.Screen(beamLine, 'FSM-DCM', (0, 29400, height))

    beamLine.vfm = roe.SimpleVFM(
        beamLine, 'VFM', [0, 30575, height],
        surface=('SiO2',), material=(materialVFM,),
        limPhysX=(-20., 20.), limPhysY=(-700., 700.),
        limOptX=(-10, 10), limOptY=(-700, 700),
        positionRoll=math.pi, R=5.0e6, r=40.77, alarmLevel=0.2)
    beamLine.slitAfterVFM = ra.RectangularAperture(
        beamLine, 'SlitAfterVFM', (0, 31720, height),
        ('left', 'right', 'bottom', 'top'), [-7, 7, -2, 2], alarmLevel=0.5)
    beamLine.fsmVFM = rsc.Screen(beamLine, 'FSM-VFM', (0, 32000, height))
    beamLine.ohPS = ra.RectangularAperture(
        beamLine, 'OH-PS', (0, 32070, height),
        ('left', 'right', 'bottom', 'top'), (-20, 20, 25, 55), alarmLevel=0.2)

    beamLine.slitEH = ra.RectangularAperture(
        beamLine, 'slitEH', (0, 43000, height),
        ('left', 'right', 'bottom', 'top'), [-20, 20, -7, 7], alarmLevel=0.5)
    beamLine.fsmSample = rsc.Screen(
        beamLine, 'FSM-Sample', (0, 45863, height))

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

    beamFSM0 = beamLine.fsm0.expose(beamSource)
    beamLine.feFixedMask.propagate(beamSource)
    beamFSMFE = beamLine.fsmFE.expose(beamSource)
    beamFilter1global, beamFilter1local1, beamFilter1local2 = \
        beamLine.filter1.double_refract(beamSource)
    beamFilter1local2A = rs.Beam(copyFrom=beamFilter1local2)
    beamFilter1local2A.absorb_intensity(beamSource)
    if hasattr(beamLine, 'filter2'):
        beamFilter2global, beamFilter2local1, beamFilter2local2 = \
            beamLine.filter2.double_refract(beamFilter1global)
        beamFilter2local2.absorb_intensity(beamFilter1global)
        beamFurtherDown = beamFilter2global
    else:
        beamFurtherDown = beamFilter1global
#        beamFurtherDown = beamSource
    beamVCMglobal, beamVCMlocal = beamLine.vcm.reflect(beamFurtherDown)
    beamVCMlocal.absorb_intensity(beamFurtherDown)
    beamFSMVCM = beamLine.fsmVCM.expose(beamVCMglobal)

    beamDCMglobal, beamDCMlocal1, beamDCMlocal2 = \
        beamLine.dcm.double_reflect(beamVCMglobal)
    beamDCMlocal1.absorb_intensity(beamVCMglobal)

    beamBSBlocklocal = beamLine.BSBlock.propagate(beamDCMglobal)
    beamSlitAfterDCMlocal = beamLine.slitAfterDCM.propagate(beamDCMglobal)
    beamFSMDCM = beamLine.fsmDCM.expose(beamDCMglobal)

    beamVFMglobal, beamVFMlocal = beamLine.vfm.reflect(beamDCMglobal)
    beamSlitAfterVFMlocal = beamLine.slitAfterVFM.propagate(beamVFMglobal)
    beamFSMVFM = beamLine.fsmVFM.expose(beamVFMglobal)
    beamPSLocal = beamLine.ohPS.propagate(beamVFMglobal)

    beamSlitEHLocal = beamLine.slitEH.propagate(beamVFMglobal)
    beamFSMSample = beamLine.fsmSample.expose(beamVFMglobal)

#               'beamFilter2global': beamFilter2global,
#               'beamFilter2local1': beamFilter2local1,
#               'beamFilter2local2': beamFilter2local2,
    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamFSMFE': beamFSMFE,
               'beamFilter1global': beamFilter1global,
               'beamFilter1local1': beamFilter1local1,
               'beamFilter1local2': beamFilter1local2,
               'beamFilter1local2A': beamFilter1local2A,
               'beamVCMglobal': beamVCMglobal, 'beamVCMlocal': beamVCMlocal,
               'beamFSMVCM': beamFSMVCM,
               'beamDCMglobal': beamDCMglobal,
               'beamDCMlocal1': beamDCMlocal1, 'beamDCMlocal2': beamDCMlocal2,
               'beamFSMDCM': beamFSMDCM,
               'beamBSBlocklocal': beamBSBlocklocal,
               'beamSlitAfterDCMlocal': beamSlitAfterDCMlocal,
               'beamFSMDCM': beamFSMDCM,
               'beamVFMglobal': beamVFMglobal, 'beamVFMlocal': beamVFMlocal,
               'beamSlitAfterVFMlocal': beamSlitAfterVFMlocal,
               'beamFSMVFM': beamFSMVFM,
               'beamPSLocal': beamPSLocal,
               'beamSlitEHLocal': beamSlitEHLocal,
               'beamFSMSample': beamFSMSample
               }
    if hasattr(beamLine, 'filter2'):
        outDict['beamFilter2global'] = beamFilter2global
        outDict['beamFilter2local1'] = beamFilter2local1
        outDict['beamFilter2local2'] = beamFilter2local2
    beamLine.beams = outDict
    return outDict

rr.run_process = run_process

aceptanceH = 4e-4
aceptanceV = 1e-4


def align_beamline(beamLine, pitch=None, bragg=None, energy=9000.,
                   fixedExit=None, heightVFM=42.79, vfmR='auto'):
    p = raycing.distance_xy(beamLine.vfm.center, beamLine.sources[0].center)
    if pitch is None:
        sinPitch = beamLine.vfm.r * 1.5 / p
        pitch = math.asin(sinPitch)
    else:
        sinPitch = math.sin(pitch)
    if vfmR == 'auto':
        beamLine.vfm.R = p / sinPitch
    else:
        beamLine.vfm.R = vfmR
    fefm = beamLine.feFixedMask
    op = fefm.center[1] * aceptanceV / 2 * min(1, pitch/2e-3)
    fefm.opening[2] = -op
    fefm.opening[3] = op

    beamLine.vcm.pitch = pitch
    p = raycing.distance_xy(beamLine.vcm.center, beamLine.sources[0].center)
    beamLine.vcm.R = 2. * p / sinPitch
    print('VCM.p = {0:.1f}'.format(p))
    print('VCM.pitch = {0:.6f} mrad'.format(beamLine.vcm.pitch*1e3))
    print('VCM.roll = {0:.6f} mrad'.format(beamLine.vcm.roll*1e3))
    print('VCM.yaw = {0:.6f} mrad'.format(beamLine.vcm.yaw*1e3))
    print('VCM.z = {0:.3f}'.format(beamLine.vcm.center[2]))
    print('VCM.R = {0:.0f}'.format(beamLine.vcm.R))

    p = raycing.distance_xy(beamLine.dcm.center, beamLine.vcm.center)
    beamLine.dcm.center[2] = beamLine.height + p*math.tan(2*pitch)
    aCrystal = beamLine.dcm.material[0]
    dSpacing = aCrystal.d
    print('DCM.dSpacing = {0:.6f} angsrom'.format(dSpacing))
    if bragg is None:
        theta = math.asin(rm.ch / (2*dSpacing*energy)) -\
            aCrystal.get_dtheta_symmetric_Bragg(energy)
        bragg = theta + 2*pitch
    else:
        theta = bragg - 2*pitch
        energy = rm.ch / (2*dSpacing*math.sin(theta))
    print('DCM.energy = {0:.3f} eV'.format(energy))
    print('DCM.bragg = {0:.6f} deg'.format(math.degrees(bragg)))
    print('DCM.realThetaAngle = DCM.bragg - 2*VCM.pitch = {0:.6f} deg'.format(
          math.degrees(theta)))
    beamLine.dcm.energy = energy
    beamLine.dcm.bragg = bragg
    p = raycing.distance_xy(beamLine.vfm.center, beamLine.vcm.center)
    if heightVFM is not None:
        fixedExit = (heightVFM - beamLine.height - p * math.tan(2 * pitch)) * \
            math.cos(2 * pitch)
    else:
        heightVFM = fixedExit / math.cos(2 * pitch) + \
            beamLine.height + p * math.tan(2 * pitch) + 0.2

    beamLine.heightVFM = heightVFM
    beamLine.dcm.fixedExit = fixedExit
    beamLine.dcm.cryst2perpTransl =\
        beamLine.dcm.fixedExit/2./math.cos(beamLine.dcm.bragg)
    print('DCM.pitch = {0:.6f} mrad'.format(beamLine.dcm.pitch*1e3))
    print('DCM.roll = {0:.6f} mrad'.format(beamLine.dcm.roll*1e3))
    print('DCM.yaw = {0:.6f} mrad'.format(beamLine.dcm.yaw*1e3))
    print('DCM.z = {0:.3f}'.format(beamLine.dcm.center[2]))
    print('DCM.fixedExit = {0:.3f}'.format(fixedExit))
    print('DCM.cryst2perpTransl = {0:.3f}'.format(
          beamLine.dcm.cryst2perpTransl))

    p = raycing.distance_xy(
        beamLine.vfm.center,
        (beamLine.slitAfterDCM.center[0], beamLine.slitAfterDCM.center[1]))
    slitHeight = heightVFM - p * math.tan(2 * pitch)
    dz = beamLine.slitAfterDCM.opening[3] - beamLine.slitAfterDCM.opening[2]
    beamLine.slitAfterDCM.opening[2] = slitHeight - beamLine.height - dz/2
    beamLine.slitAfterDCM.opening[3] = slitHeight - beamLine.height + dz/2
    beamLine.slitAfterDCM.set_optical_limits()

    p = raycing.distance_xy(beamLine.vfm.center, beamLine.fsmDCM.center)
    fsmHeight = heightVFM - p * math.tan(2 * pitch)
    print('fsmDCM.z = {0:.3f}'.format(fsmHeight))

    beamLine.vfm.pitch = -pitch
    beamLine.vfm.center[2] = heightVFM  # - beamLine.vfm.hCylinder
    print('VFM.pitch = {0:.6f} mrad'.format(beamLine.vfm.pitch*1e3))
    print('VFM.roll = {0:.6f} mrad'.format(beamLine.vfm.roll*1e3))
    print('VFM.yaw = {0:.6f} mrad'.format(beamLine.vfm.yaw*1e3))
    print('VFM.z = {0:.3f}'.format(beamLine.vfm.center[2]))
    print('VFM.R = {0:.0f}'.format(beamLine.vfm.R))
    print('VFM.r = {0:.3f}'.format(beamLine.vfm.r))

    dz = beamLine.slitAfterVFM.opening[3] - beamLine.slitAfterVFM.opening[2]
    beamLine.slitAfterVFM.opening[2] = heightVFM - beamLine.height - dz/2
    beamLine.slitAfterVFM.opening[3] = heightVFM - beamLine.height + dz/2
    beamLine.slitAfterVFM.set_optical_limits()

    p = raycing.distance_xy(beamLine.vfm.center, beamLine.sources[0].center)
    q = 1./(2 * np.sin(pitch)/beamLine.vfm.r - 1./p)

    qr = raycing.distance_xy(beamLine.fsmSample.center, beamLine.vfm.center)
    beamLine.spotSizeH = abs(1. - qr / q) * p * aceptanceH

    qr = raycing.distance_xy(
        (beamLine.slitEH.center[0], beamLine.slitEH.center[1]),
        beamLine.vfm.center)
    s = abs(1. - qr / q) * p * aceptanceH / 2
    beamLine.slitEH.opening[0] = -s * 1.2
    beamLine.slitEH.opening[1] = s * 1.2
    dz = beamLine.slitEH.opening[3] - beamLine.slitEH.opening[2]
    beamLine.slitEH.opening[2] = heightVFM - beamLine.height - dz/2
    beamLine.slitEH.opening[3] = heightVFM - beamLine.height + dz/2
    beamLine.slitEH.set_optical_limits()


if __name__ == '__main__':
    myBalder = build_beamline(nrays=25000)
    align_beamline(myBalder, pitch=2e-3, energy=9000., fixedExit=20.86)
    print('finished')

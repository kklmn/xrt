# -*- coding: utf-8 -*-
"""
!!! select one of the four functions to run at the very bottom !!!
!!! select 'rays', 'hybrid' or 'wave' below !!!
!!! select a desired cut and emittance below !!!

Described in the __init__ file.
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "07 Feb 2018"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc
#import xrt.backends.raycing.waves as rw
import xrt.backends.raycing.coherence as rco

showIn3D = False

mRhodium = rm.Material('Rh', rho=12.41)
mRhodiumGrating = rm.Material('Rh', rho=12.41, kind='grating')
mGolden = rm.Material('Au', rho=19.32)
mGoldenGrating = rm.Material('Au', rho=19.32, kind='grating')

acceptanceHor = 2.2e-4  # FE acceptance, full angle, rad
acceptanceVer = 4.2e-4  # FE acceptance

pFE = 19250.
pM1 = 24000.
pPG = 2000.
pM3 = 2800.  # distance to PG
qM3mer = 12000.
qM3sag = 12000.
dM4ES = 2200.
dM45 = 3200.
pExp = 1800.
#pFZP = 5000.

E0 = 280.
dE = 0.125
targetHarmonic = 1
gratingMaterial = mGoldenGrating
material = mGolden
#material = None
expSize = 50.

#E0 = 2400.
#dE = 1.
#targetHarmonic = 3
#gratingMaterial = mRhodiumGrating
#material = mRhodium
#expSize = 25.

dFocus = np.linspace(-100, 100, 9) if not showIn3D else [0]

pitch = np.radians(1)

cff = 1.6        # mono parameters
fixedExit = 20.  # mm
rho = 300.       # lines/mm
blaze = np.radians(0.6)

ESdX = 2.  # in mm EXIT SLIT SIZE
ESdZ = 0.1  # in mm EXIT SLIT SIZE

repeats = 1
nrays = 1e5

# what = 'rays'
what = 'hybrid'
# what = 'wave'

if what == 'rays':
    prefix = 'cxi_1D-1-rays-'
elif what == 'hybrid':
    prefix = 'cxi_1D-2-hybr-'
elif what == 'wave':
    prefix = 'cxi_1D-3-wave-'
if E0 > 280.1:
    prefix += '{0:.0f}eV-'.format(E0)

#==============================================================================
# horizontal cut
#==============================================================================
vShrink = 1e-3
hShrink = 1.
cut = 'hor-'
prefix += cut
xbins, xppb = 128, 2
zbins, zppb = 1, 8
xebins, xeppb = 128, 2  # at exp screen and wavefront
zebins, zeppb = 1, 8
cbins, cppb = xbins, xppb
cebins, ceppb = xebins, xeppb
abins = xebins

#==============================================================================
# vertical cut
#==============================================================================
#vShrink = 1.
#hShrink = 1e-3
#cut = 'ver-'
#prefix += cut
#xbins, xppb = 1, 8
#zbins, zppb = 128, 2
#xebins, xeppb = 1, 8
#zebins, zeppb = 128, 2  # at exp screen and wavefront
#cbins, cppb = zbins, zppb
#cebins, ceppb = zebins, zeppb
#abins = zebins


is0emittance = True
if is0emittance:
    emittanceFactor = 0.
    prefix += '0emit-'
else:
    emittanceFactor = 1.
    prefix += 'non0e-'
    nrays = 1e5
    repeats = 160

is0energySpread = True
if is0energySpread:
    energySpreadFactor = 0.
    prefix += '0enSpread-'
else:
    energySpreadFactor = 1.
#    prefix += 'non0enSpread-'

isMono = True
if isMono:
    prefix += 'monoE-'
else:
    prefix += 'wideE-'

vFactor = 1.
#prefix += '050%V-'
#hFactor = 0.25/4
#prefix += '025over04%H-'
hFactor = 1.
#hFactor = 0.25
#prefix += '025%HPG-'


class Grating(roe.OE):
    def local_g(self, x, y, rho=rho):
        return 0, -rho, 0  # constant line spacing along y


def build_beamline(azimuth=0):
    beamLine = raycing.BeamLine(azimuth=azimuth, height=0)

    beamLine.source = rs.Undulator(
        beamLine, 'Softi53', nrays=nrays,
        eE=3.0, eI=0.5,  eEspread=0.0008*energySpreadFactor,
        eEpsilonX=0.263*emittanceFactor, eEpsilonZ=0.008*emittanceFactor,
        betaX=9., betaZ=2.,
        period=48., n=77, targetE=(E0, targetHarmonic),
        eMin=E0-dE*vFactor, eMax=E0+dE*vFactor,
        xPrimeMax=acceptanceHor/2*1e3*hShrink,
        zPrimeMax=acceptanceVer/2*1e3*vShrink,
        xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
#        targetOpenCL='CPU',
        uniformRayDensity=True,
        filamentBeam=(what != 'rays'))

    opening = [-acceptanceHor*pFE/2*hShrink,
               acceptanceHor*pFE/2*hShrink,
               -acceptanceVer*pFE/2*vShrink,
               acceptanceVer*pFE/2*vShrink]
#    opening = [-acceptanceHor*pFE/2*hShrink*hFactor,
#               acceptanceHor*pFE/2*hShrink*hFactor,
#               -acceptanceVer*pFE/2*vShrink,
#               acceptanceVer*pFE/2*vShrink]
    beamLine.slitFE = ra.RectangularAperture(
        beamLine, 'FE slit', kind=['left', 'right', 'bottom', 'top'],
        opening=opening)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM-M1')

    xShrink = vShrink if what == 'wave' else 1
    yShrink = hShrink if what == 'wave' else 1
    beamLine.m1 = roe.ToroidMirror(
        beamLine, 'M1', surface=('Au',), material=material,
        limPhysX=(-5*xShrink, 5*xShrink), limPhysY=(-150*yShrink, 150*yShrink),
        positionRoll=np.pi/2, R=1e22,
        alarmLevel=0.1)
#    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM-M1')

    xShrink = hShrink if what == 'wave' else 1
    yShrink = vShrink if what == 'wave' else 1
    beamLine.m2 = roe.OE(
        beamLine, 'M2', surface=('Au',), material=material,
        limPhysX=(-5*xShrink, 5*xShrink), limPhysY=(-225*yShrink, 225*yShrink),
        alarmLevel=0.1)

    gratingKW = dict(
        positionRoll=np.pi,
        limPhysX=(-2*xShrink, 2*xShrink),
        #limPhysX=(-2*xShrink*hFactor, 2*xShrink*hFactor),
        limPhysY=(-40*yShrink, 40*yShrink), alarmLevel=0.1)
    if what == 'rays' or showIn3D:
        beamLine.pg = Grating(beamLine, 'PlaneGrating',
                              material=gratingMaterial, **gratingKW)
        beamLine.pg.material.efficiency = [(1, 0.4)]
    else:
        beamLine.pg = roe.BlazedGrating(
            beamLine, 'BlazedGrating', material=material, blaze=blaze,
            rho=rho, **gratingKW)
    beamLine.pg.order = 1
#    beamLine.fsmPG = rsc.Screen(beamLine, 'FSM-PG')

    beamLine.m3 = roe.ToroidMirror(
        beamLine, 'M3', surface=('Au',), material=material,
        positionRoll=-np.pi/2, limPhysX=(-10.*vShrink, 10.*vShrink),
        limPhysY=(-100.*hShrink, 100.*hShrink),
        #limPhysY=(-100.*hShrink*hFactor, 100.*hShrink*hFactor),
        alarmLevel=0.1)
    beamLine.fsm3 = rsc.Screen(beamLine, 'FSM-M3')

#    beamLine.exitSlit = ra.RoundAperture(
#         beamLine, 'ExitSlit', r=ESradius, alarmLevel=None)
    beamLine.exitSlit = ra.RectangularAperture(
         beamLine, 'ExitSlit',
#         opening=[-ESdX*hFactor/2, ESdX*hFactor/2,
#                  -ESdZ*vFactor/2, ESdZ*vFactor/2])
         opening=[-ESdX/2, ESdX/2,
                  -ESdZ/2, ESdZ/2])

    beamLine.m4 = roe.EllipticalMirrorParam(
        beamLine, 'M4', surface=('Au',), material=material,
        positionRoll=np.pi/2, pitch=pitch, isCylindrical=True,
        p=43000., q=dM45+pExp, limPhysX=(-0.5*vShrink, 0.5*vShrink),
        limPhysY=(-70.*hShrink, 70.*hShrink),
        #limPhysY=(-70.*hShrink*hFactor, 70.*hShrink*hFactor),
        alarmLevel=0.2)
    beamLine.fsm4 = rsc.Screen(beamLine, 'FSM-M4')

    beamLine.m5 = roe.EllipticalMirrorParam(
        beamLine, 'M5', surface=('Au',), material=material,
        yaw=-2*pitch, pitch=pitch, isCylindrical=True,
        p=dM4ES+dM45, q=pExp,
        limPhysX=(-0.5*hShrink, 0.5*hShrink),
        limPhysY=(-40.*vShrink, 40.*vShrink), alarmLevel=0.2)
    beamLine.fsm5 = rsc.Screen(beamLine, 'FSM-M5')

    beamLine.fsmExp = rsc.Screen(beamLine, 'FSM-Exp')

    beamLine.waveOnSampleA = [np.zeros(abins) for sP in dFocus]
    beamLine.waveOnSampleB = [np.zeros(abins) for sP in dFocus]
    beamLine.waveOnSampleC = [np.zeros(abins) for sP in dFocus]

    return beamLine


def align_grating(grating, E, m, cff):
    order = abs(m) if cff > 1 else -abs(m)
    f1 = cff**2 + 1
    f2 = cff**2 - 1
    if abs(f2) < 1e-5:
        raise ValueError('cff is not allowed to be close to 1!')

    ml_d = order * rho * rm.ch / E * 1e-7
    cosAlpha = np.sqrt(-ml_d**2 * f1 + 2*abs(ml_d) *
                       np.sqrt(f2**2 + cff**2 * ml_d**2)) / abs(f2)
    cosBeta = cff * cosAlpha
    alpha = np.arccos(cosAlpha)
    beta = -np.arccos(cosBeta)
    return alpha, beta


def align_beamline(beamLine, E0=E0, pitchM1=pitch, pitchM3=pitch,
                   pitchM4=pitch, pitchM5=pitch):
    beamLine.source.center = pM1 * np.sin(2*pitch), \
        -pM1 * np.cos(2*pitch), 0     # centre of undu when M1 is (0,0,0)
    beamLine.slitFE.center = (pM1-pFE) * np.sin(2*pitch), \
        -(pM1-pFE) * np.cos(2*pitch), 0
    beamLine.fsm0.center = beamLine.slitFE.center

    rM1 = 2. * pM1 * np.sin(pitch)
    print('M1: r = {0} mm'.format(rM1))
    beamLine.m1.center = 0, 0, 0   # THIS IS THE ORIGIN!, y-direction = M1-> PG
    beamLine.m1.pitch = pitchM1
    beamLine.m1.r = rM1
#    beamLine.fsm1.center = beamLine.m1.center
#    beamLine.fsm1.x = -np.sin(beamLine.m1.pitch), np.cos(beamLine.m1.pitch), 0

    if isinstance(beamLine.pg.order, int):
        m = beamLine.pg.order
    else:
        m = beamLine.pg.order[0]
    print('grating order = {0}'.format(m))
    alpha, beta = align_grating(beamLine.pg, E0, m=m, cff=cff)
    includedAngle = alpha - beta
    print('alpha = {0} deg'.format(np.degrees(alpha)))
    print('beta = {0} deg'.format(np.degrees(beta)))
    print('M2 theta angle = {0} deg'.format(np.degrees(includedAngle / 2)))
    print('cos(beta)/cos(alpha) = {0}'.format(np.cos(beta)/np.cos(alpha)))
    t = -fixedExit / np.tan(includedAngle)
    print('t = {0} mm'.format(t))
#    print('N = {0}'.format(Nzone)

    beamLine.m2.pitch = (np.pi - includedAngle) / 2.
    print('M2 pitch = {0} deg'.format(np.degrees(beamLine.m2.pitch)))
    beamLine.m2.pitch = (np.pi - includedAngle) / 2.
    beamLine.m2.center = 0, pPG - t, 0
    beamLine.m2.yaw = -2 * beamLine.m1.pitch

    beamLine.pg.pitch = -(beta + np.pi/2)
    print('PG pitch = {0} deg'.format(np.degrees(beamLine.pg.pitch)))
    beamLine.pg.center = 0, pPG, fixedExit
    beamLine.pg.yaw = -2 * beamLine.m1.pitch
#    beamLine.fsmPG.center = beamLine.pg.center
    print('rho = {0}'.format(rho))
    if what != 'rays' and not showIn3D:
        drho = beamLine.pg.get_grating_area_fraction()
        beamLine.pg.areaFraction = drho
        print(u'PG areaFraction = {0}'.format(beamLine.pg.areaFraction))

#    pM3mer = pM1 + pPG + pM3  # pM3sag = infinity
    sinPitchM3 = np.sin(pitchM3)
    rM3 = 2. * sinPitchM3 * qM3sag   # focusing
    print('M3: r = {0} mm'.format(rM3))
    beamLine.m3.center = [0, pPG + pM3, fixedExit]
    beamLine.m3.pitch = -pitchM3  # opposite angles to M1
    beamLine.m3.r = rM3
    beamLine.m3.R = 1e22  # no hor focusing: M3 cylindrical
    beamLine.fsm3.center = beamLine.m3.center

    beamLine.exitSlit.center = -qM3sag * np.sin(2*pitch),\
        beamLine.m3.center[1] + qM3sag * np.cos(2*pitch), fixedExit

    beamLine.m4.center = -(qM3sag+dM4ES) * np.sin(2*pitchM3),\
        beamLine.m3.center[1] + (qM3sag+dM4ES) * np.cos(2*pitchM3), fixedExit
    print('M4: p={0}, q={1}'.format(beamLine.m4.p, beamLine.m4.q))

    beamLine.fsm4.center = beamLine.m4.center
    beamLine.fsm4.x = np.cos(2*pitch), np.sin(2*pitch), 0

    beamLine.m5.center = beamLine.m4.center[0],\
        beamLine.m4.center[1] + dM45, fixedExit
    print('M5: p={0}, q={1}'.format(beamLine.m5.p, beamLine.m5.q))

    beamLine.fsm5.center = beamLine.m5.center

#    beamLine.fsmExp.center = \
#        beamLine.m4.center[0] + (dM45+pExp) * np.sin(pitchM3-pitchM4),\
#        beamLine.m4.center[1] + (pExp+dM45) * np.cos(pitchM3-pitchM4),\
#        fixedExit + pExp*np.tan(2*pitchM5)

#    beamLine.fsmExp.x = np.cos(2*pitchM4), np.sin(2*pitchM4), 0
    # beamLine.fsmExp.x = np.cos(2.0004*pitchM4), np.sin(2.0004*pitchM4), 0
    # beamLine.fsmExp.z = 0, -np.sin(2*pitchM5), np.cos(2*pitchM5)

    beamLine.fsmExpCenters = []
    for d in dFocus:
        p = pExp + d
        beamLine.fsmExpCenters.append(
            [beamLine.m4.center[0] + (dM45+p) * np.sin(pitchM3-pitchM4),
             beamLine.m4.center[1] + (dM45+p) * np.cos(pitchM3-pitchM4),
             beamLine.m4.center[2] + p * np.tan(2*pitchM5)])


def run_process_rays(beamLine, shineOnly1stSource=False):
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False
    if True:  # source to wave
        waveOnSlit = beamLine.slitFE.prepare_wave(beamLine.source, nrays)
        beamSource = beamLine.source.shine(wave=waveOnSlit,
                                           fixedEnergy=fixedEnergy)
        beamFSM0 = waveOnSlit
    else:
        beamSource = beamLine.source.shine(fixedEnergy=fixedEnergy)
        beamFSM0 = beamLine.fsm0.expose(beamSource)

    beamM1global, beamM1local = beamLine.m1.reflect(beamSource)
    beamM2global, beamM2local = beamLine.m2.reflect(beamM1global)
    beamPGglobal, beamPGlocal = beamLine.pg.reflect(beamM2global)
    beamM3global, beamM3local = beamLine.m3.reflect(beamPGglobal)
    beamExitSlit = beamLine.exitSlit.propagate(beamM3global)
    beamFSM4 = beamLine.fsm4.expose(beamM3global)
    beamM4global, beamM4local = beamLine.m4.reflect(beamM3global)
    beamFSM5 = beamLine.fsm5.expose(beamM4global)
    beamM5global, beamM5local = beamLine.m5.reflect(beamM4global)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               'beamM3global': beamM3global, 'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamFSM4': beamFSM4,
               'beamM4global': beamM4global, 'beamM4local': beamM4local,
               'beamFSM5': beamFSM5,
               'beamM5global': beamM5global, 'beamM5local': beamM5local
               }

    for ic, fsmExpCenter in enumerate(beamLine.fsmExpCenters):
        beamLine.fsmExp.center = fsmExpCenter
        beamFSMExp = beamLine.fsmExp.expose(beamM5global)
        outDict['beamFSMExp{0:02d}'.format(ic)] = beamFSMExp

    if showIn3D:
        beamLine.prepare_flow()
    return outDict


def run_process_hybr(beamLine, shineOnly1stSource=False):
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False
    if True:  # source to wave
        waveOnSlit = beamLine.slitFE.prepare_wave(beamLine.source, nrays)
        beamSource = beamLine.source.shine(wave=waveOnSlit,
                                           fixedEnergy=fixedEnergy)
        beamFSM0 = waveOnSlit
    else:
        beamSource = beamLine.source.shine(fixedEnergy=fixedEnergy)
        beamFSM0 = beamLine.fsm0.expose(beamSource)

    beamM1global, beamM1local = beamLine.m1.reflect(beamSource)
    beamM2global, beamM2local = beamLine.m2.reflect(beamM1global)
    beamPGglobal, beamPGlocal = beamLine.pg.reflect(beamM2global)

    beamPGlocal.area = 0
    beamPGlocal.areaFraction = beamLine.pg.areaFraction

    waveOnSamples = []
    waveOnFronts = []
    for fsmExpCenter in beamLine.fsmExpCenters:
        beamLine.fsmExp.center = fsmExpCenter
        waveOnSample = beamLine.fsmExp.prepare_wave(
                beamLine.m5, beamLine.fsmExpX, beamLine.fsmExpZ)
        waveOnSamples.append(waveOnSample)

    wrepeats = 1
    for repeat in range(wrepeats):
        if wrepeats > 1:
            print('wave repeats: {0} of {1} ...'.format(repeat+1, wrepeats))

        waveOnm3 = beamLine.m3.prepare_wave(beamLine.pg, nrays)
        beamTom3 = beamPGlocal.diffract(waveOnm3)
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        beamM3local.diffract(waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnm4 = beamLine.m4.prepare_wave(beamLine.exitSlit, nrays)
        beamTom4 = waveOnExitSlit.diffract(waveOnm4)
        beamM4global, beamM4local = beamLine.m4.reflect(
            beamTom4, noIntersectionSearch=True)

        waveOnm5 = beamLine.m5.prepare_wave(beamLine.m4, nrays)
        beamTom5 = beamM4local.diffract(waveOnm5)
        beamM5global, beamM5local = beamLine.m5.reflect(
            beamTom5, noIntersectionSearch=True)

#        for waveOnSample in waveOnSamples:
#            rw.diffract(beamM5local, waveOnSample)
        for fsmExpCenter, waveOnSample, a, b, c in zip(
                beamLine.fsmExpCenters, waveOnSamples,
                beamLine.waveOnSampleA, beamLine.waveOnSampleB,
                beamLine.waveOnSampleC):
            beamM5local.diffract(waveOnSample)
            a += waveOnSample.a
            b += waveOnSample.b
            c += waveOnSample.c

            if 'hor' in cut:
                d = a
                dvar = waveOnSample.x[1] - waveOnSample.x[0]
            elif 'ver' in cut:
                d = c - c[abins//2]
                dvar = waveOnSample.z[1] - waveOnSample.z[0]
            dy = np.cumsum(-np.abs(np.arctan2(d, b))) * dvar
            dy -= dy[abins//2]
            beamLine.fsmExp.center = fsmExpCenter
            waveOnFront = beamLine.fsmExp.prepare_wave(
                beamLine.m5, beamLine.fsmExpX, beamLine.fsmExpZ, dy=dy)
            beamM5local.diffract(waveOnFront)
            waveOnFronts.append(waveOnFront)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamM4local': beamM4local,
               'beamM5local': beamM5local,
               }

    for ic, waveOnSample in enumerate(waveOnSamples):
        outDict['beamFSMExp{0:02d}'.format(ic)] = waveOnSample
    for iw, waveOnFront in enumerate(waveOnFronts):
        outDict['beamFSMExpFront{0:02d}'.format(iw)] = waveOnFront

    return outDict


def run_process_wave(beamLine, shineOnly1stSource=False):
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False

    waveOnSamples = []
    waveOnFronts = []
    for fsmExpCenter in beamLine.fsmExpCenters:
        beamLine.fsmExp.center = fsmExpCenter
        waveOnSample = beamLine.fsmExp.prepare_wave(
                beamLine.m5, beamLine.fsmExpX, beamLine.fsmExpZ)
        waveOnSamples.append(waveOnSample)

    wrepeats = 1
    for repeat in range(wrepeats):
        if wrepeats > 1:
            print('wave repeats: {0} of {1} ...'.format(repeat+1, wrepeats))

        waveOnSlit = beamLine.slitFE.prepare_wave(beamLine.source, nrays)
        beamSource = beamLine.source.shine(wave=waveOnSlit,
                                           fixedEnergy=fixedEnergy)
        beamFSM0 = waveOnSlit

        waveOnm1 = beamLine.m1.prepare_wave(beamLine.slitFE, nrays)
        beamTom1 = waveOnSlit.diffract(waveOnm1)
        beamM1global, beamM1local = beamLine.m1.reflect(
            beamTom1, noIntersectionSearch=True)

        waveOnm2 = beamLine.m2.prepare_wave(beamLine.m1, nrays)
        beamTom2 = beamM1local.diffract(waveOnm2)
        beamM2global, beamM2local = beamLine.m2.reflect(
            beamTom2, noIntersectionSearch=True)

        waveOnPG = beamLine.pg.prepare_wave(beamLine.m2, nrays)
        beamToPG = beamM2local.diffract(waveOnPG)
        beamPGglobal, beamPGlocal = beamLine.pg.reflect(
            beamToPG, noIntersectionSearch=True)

        beamPGlocal.area = 0
        beamPGlocal.areaFraction = beamLine.pg.areaFraction

        waveOnm3 = beamLine.m3.prepare_wave(beamLine.pg, nrays)
        beamTom3 = beamPGlocal.diffract(waveOnm3)
#        beamM3local = waveOnm3
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        beamM3local.diffract(waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnm4 = beamLine.m4.prepare_wave(beamLine.exitSlit, nrays)
        beamTom4 = waveOnExitSlit.diffract(waveOnm4)
        beamM4global, beamM4local = beamLine.m4.reflect(
            beamTom4, noIntersectionSearch=True)

        waveOnm5 = beamLine.m5.prepare_wave(beamLine.m4, nrays)
        beamTom5 = beamM4local.diffract(waveOnm5)
        beamM5global, beamM5local = beamLine.m5.reflect(
            beamTom5, noIntersectionSearch=True)

#        for waveOnSample in waveOnSamples:
#            rw.diffract(beamM5local, waveOnSample)
        for fsmExpCenter, waveOnSample, a, b, c in zip(
                beamLine.fsmExpCenters, waveOnSamples,
                beamLine.waveOnSampleA, beamLine.waveOnSampleB,
                beamLine.waveOnSampleC):
            beamM5local.diffract(waveOnSample)
            a += waveOnSample.a
            b += waveOnSample.b
            c += waveOnSample.c

            if 'hor' in cut:
                d = a
                dvar = waveOnSample.x[1] - waveOnSample.x[0]
            elif 'ver' in cut:
                d = c - c[abins//2]
                dvar = waveOnSample.z[1] - waveOnSample.z[0]
            dy = np.cumsum(-np.abs(np.arctan2(d, b))) * dvar
            dy -= dy[abins//2]
            beamLine.fsmExp.center = fsmExpCenter
            waveOnFront = beamLine.fsmExp.prepare_wave(
                beamLine.m5, beamLine.fsmExpX, beamLine.fsmExpZ, dy=dy)
            beamM5local.diffract(waveOnFront)
            waveOnFronts.append(waveOnFront)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamM4local': beamM4local,
               'beamM5local': beamM5local,
               }

    for ic, waveOnSample in enumerate(waveOnSamples):
        outDict['beamFSMExp{0:02d}'.format(ic)] = waveOnSample
    for iw, waveOnFront in enumerate(waveOnFronts):
        outDict['beamFSMExpFront{0:02d}'.format(iw)] = waveOnFront

    return outDict


if (what == 'rays') or showIn3D:
    rr.run_process = run_process_rays
elif what.startswith('hybr'):
    rr.run_process = run_process_hybr
elif what == 'wave':
    rr.run_process = run_process_wave


def define_plots(beamLine):
    plots = []

    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamFSM0', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=zbins, ppb=zppb),
        caxis=caxis, ePos=ePos, title='00-FE')
    plot.xaxis.fwhmFormatStr = '%.3f'
    plot.yaxis.fwhmFormatStr = '%.3f'
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamM1local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m1.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m1.limPhysY),
        caxis=caxis, ePos=ePos, title='01-M1local')
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamM2local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m2.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m2.limPhysY),
        caxis=caxis, ePos=ePos, title='02-M2local')
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamPGlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.pg.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.pg.limPhysY),
        caxis=caxis, ePos=ePos, title='02-PGlocal')
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamM3local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m3.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m3.limPhysY),
        caxis=caxis, ePos=ePos, title='03-M3local')
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    caxis = xrtp.XYCAxis('Ep phase', '', bins=cbins, ppb=cppb,
                         data=raycing.get_Ep_phase, limits=[-np.pi, np.pi])
    plot = xrtp.XYCPlot(
        'beamM3local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m3.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m3.limPhysY),
        caxis=caxis, ePos=ePos, title='03s-M3local')
    plots.append(plot)

    limMaxX = ESdX*0.6e3
    limMaxZ = ESdZ*0.6e3
    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1, beamLine.exitSlit.lostNum), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
        caxis=caxis, ePos=ePos, title='04-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('Es phase', '', bins=cbins, ppb=cppb,
                         data=raycing.get_Es_phase, limits=[-np.pi, np.pi])
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
        caxis=caxis, ePos=ePos, title='04s-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamM4local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m4.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m4.limPhysY),
        caxis=caxis, ePos=ePos, title='05-M4local')
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    caxis = xrtp.XYCAxis('energy', 'eV', bins=cbins, ppb=cppb)
    plot = xrtp.XYCPlot(
        'beamM5local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m5.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m5.limPhysY),
        caxis=caxis, ePos=ePos, title='06-M5local')
    plots.append(plot)

    complexPlotsIs = []
    complexPlotsPCAs = []
    for ic, (fsmExpCenter, d) in enumerate(
            zip(beamLine.fsmExpCenters, dFocus)):
        ePos = 1 if xebins < 4 else 2
        caxis = xrtp.XYCAxis('energy', 'eV', bins=cebins, ppb=ceppb)
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xebins, ppb=xeppb,
                               limits=[-expSize, expSize]),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zebins, ppb=zeppb,
                               limits=[-expSize, expSize]),
            caxis=caxis,
            fluxKind='s', ePos=ePos, title='08-ExpFocus{0:02d}'.format(ic))
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plots.append(plot)
        complexPlotsIs.append(plot)
        fsmExpPlot = plot

        ePos = 1 if xebins < 4 else 2
        caxis = xrtp.XYCAxis('energy', 'eV', bins=cebins, ppb=ceppb)
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xebins, ppb=xeppb,
                               limits=[-expSize, expSize]),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zebins, ppb=zeppb,
                               limits=[-expSize, expSize]),
            caxis=caxis,
            fluxKind='EsPCA', ePos=ePos,
            title='08e-ExpFocus{0:02d}'.format(ic))
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plots.append(plot)
        complexPlotsPCAs.append(plot)

        ePos = 1 if xebins < 4 else 2
        caxis = xrtp.XYCAxis('Es phase', '', bins=cebins, ppb=ceppb,
                             data=raycing.get_Es_phase,
                             #density='kde',
                             limits=[-np.pi, np.pi])
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xebins, ppb=xeppb,
                               limits=[-expSize, expSize]),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zebins, ppb=zeppb,
                               limits=[-expSize, expSize]),
            caxis=caxis, ePos=ePos,
            title='08p-ExpFocusPhase{0:02d}'.format(ic))
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plot.caxis.fwhmFormatStr = None
        plots.append(plot)

        if what != 'rays':
            ePos = 1 if xebins < 4 else 2
            caxis = xrtp.XYCAxis('Es phase', '', bins=cebins, ppb=ceppb,
                                 data=raycing.get_Es_phase,
                                 #density='kde',
                                 limits=[-np.pi, np.pi])
            plot = xrtp.XYCPlot(
                'beamFSMExpFront{0:02d}'.format(ic), (1,), aspect='auto',
                xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xebins, ppb=xeppb,
                                   limits=[-expSize, expSize]),
                yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zebins, ppb=zeppb,
                                   limits=[-expSize, expSize]),
                caxis=caxis, ePos=ePos,
                title='08pf-ExpFocusPhaseFront{0:02d}'.format(ic))
            plot.textPanel = plot.fig.text(
                0.88, 0.8, u'f{0:+.0f} mm'.format(d),
                transform=plot.fig.transFigure, size=12, color='r',
                ha='center')
#            plot.caxis.fwhmFormatStr = '%.3f'
            plot.caxis.fwhmFormatStr = None
            plots.append(plot)

    ax = fsmExpPlot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
#    print('d beamLine.fsmExpX', beamLine.fsmExpX[1]-beamLine.fsmExpX[0])
    ax = fsmExpPlot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
#    print('d beamLine.fsmExpZ', beamLine.fsmExpZ[1]-beamLine.fsmExpZ[0])

    for plot in plots:
        plot.saveName = [prefix + plot.title + '.png', ]
        if plot.caxis.label.startswith('energy'):
            plot.caxis.limits = E0-dE*vFactor, E0+dE*vFactor
            plot.caxis.offset = E0
        if plot.fluxKind.startswith('power'):
            plot.fluxFormatStr = '%.0f'
        else:
            plot.fluxFormatStr = '%.1p'
        if plot.xaxis.bins < 4:
            plot.ax2dHist.xaxis.set_ticks([0])
            plot.xaxis.fwhmFormatStr = None
            plot.ax1dHistX.set_visible(False)
        if plot.yaxis.bins < 4:
            plot.ax2dHist.yaxis.set_ticks([0])
            plot.yaxis.fwhmFormatStr = None
            plot.ax1dHistY.set_visible(False)

    toSave = [complexPlotsIs, complexPlotsPCAs,
              beamLine.waveOnSampleA,
              beamLine.waveOnSampleB,
              beamLine.waveOnSampleC]
    lents = len(toSave[0])
    for ts in toSave[1:]:
        if len(ts) != lents:
            raise ValueError("cannot save the output!")
    return plots, toSave


def afterScript(toSave):
    dump = []
    for ic, (complexPlotIs, complexPlotPCAs, a, b, c) in\
            enumerate(zip(*toSave)):
        x = complexPlotIs.xaxis.binCenters
        y = complexPlotIs.yaxis.binCenters
        Ixy = complexPlotIs.total2D
        Es = complexPlotPCAs.field3D
        dump.append([x, y, Ixy, Es, a, b, c])

    pickleName = '{0}-res.pickle'.format(prefix)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)


def main():
    beamLine = build_beamline(azimuth=-2*pitch)
    align_beamline(beamLine)
    if showIn3D:
#        beamLine.orient_along_global_Y()
        beamLine.glow(scale=[100, 10, 1000], centerAt='M2')
        return
    plots, toSave = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine,
                         afterScript=afterScript, afterScriptArgs=[toSave],
                         processes=1)


def plotFocus():
    pickleName = '{0}-res.pickle'.format(prefix)
    with open(pickleName, 'rb') as f:
        dump = pickle.load(f)
    #  normalize over all images:
    norm = 0.
    for ic, d in enumerate(dFocus):
        Es = dump[ic][3]
        dmax = ((np.abs(Es)**2).sum(axis=0) / Es.shape[0]).max()
        if norm < dmax:
            norm = dmax
    norm = norm**0.5

    NN = len(dFocus)
    for ic, d in enumerate(dFocus):
        x, y, Ixy, Es = dump[ic][:4]
        scrStr = 'screen at f+{0:.0f} mm'.format(d)

        print("solving PCA problem, {0} of {1}...".format(ic+1, NN))
        start = time.time()
        wPCA, vPCA = rco.calc_eigen_modes_PCA(Es)
        stop = time.time()
        print("the PCA problem has taken {0} s".format(stop-start))
        figP = rco.plot_eigen_modes(x, y, wPCA, vPCA,
                                    xlabel='x (µm)', ylabel='z (µm)')
        figP.suptitle('Principal components of one-electron images, ' + scrStr,
                      fontsize=11)
        figP.savefig('PCA-{0}-{1:02d}.png'.format(prefix, ic))

        xarr, xnam = (x, 'x') if cut.startswith('hor') else (y, 'z')
        xdata = rco.calc_1D_coherent_fraction(Es/norm, xnam, xarr)
        fig2D, figXZ = rco.plot_1D_degree_of_coherence(
            xdata, xnam, xarr, "nm", isIntensityNormalized=True,
            locLegend='center left')
        fig2D.suptitle('Mutual intensity, ' + scrStr, size=11)
        figXZ.suptitle('Intensity and Degree of Coherence, ' + scrStr, size=11)
        fig2D.savefig('MutualI-{0}-{1:02d}.png'.format(prefix, ic))
        figXZ.savefig('IDOC-{0}-{1:02d}.png'.format(prefix, ic))

    print("Done")
    plt.show()


def plotFront():
    pickleName = '{0}-res.pickle'.format(prefix)
    with open(pickleName, 'rb') as f:
        dump = pickle.load(f)

    figABC = plt.figure(figsize=(5, 5))
    tit = r'wave fronts'
    if 'hor' in cut:
        tit += ', horizontal cuts'
    elif 'ver' in cut:
        tit += ', vertical cuts'
    figABC.suptitle(tit, fontsize=11)
    rect2d = [0.15, 0.1, 0.82, 0.83]
    ax = figABC.add_axes(rect2d, aspect='auto')
    ax.set_ylabel(u'y (Å)')
    if 'hor' in cut:
        ax.set_xlabel(r"$\psi$ " + u"(µrad)")
    elif 'ver' in cut:
        ax.set_xlabel(r"$\theta$ " + u"(µrad)")

    colors = ['r', 'm', 'b', 'c']
    colors = colors + ['g'] + colors[::-1]
    for ic, (df, color) in enumerate(zip(dFocus, colors)):
        if ic % 2 == 1:
            continue
        x, z = dump[ic][0:2]
        a, b, c = dump[ic][4:7]
        if 'hor' in cut:
            d = a - a[abins//2]
            var = x
        elif 'ver' in cut:
            d = c - c[abins//2]
            var = z
        dvar = (var[1] - var[0]) * 1e-3
        dy = np.cumsum(np.sign(d)*np.abs(np.arctan2(d, b))) * dvar * 1e7
        dy -= dy[abins//2]
        xx = np.arctan2(var, abs(df)) * 1e3
        st = ' (focus)' if df == 0 else ''
        line = '-' if df >= 0 else '--'
        style = color + line
        l, = ax.plot(xx, dy, style, lw=2, label='{0:.0f}{1}'.format(df, st))

    ax.legend(title='screen at (mm)', loc='lower center', fontsize=10)
    xm = var[-1] / dFocus[-1] * 1e3
    ax.set_xlim(-xm, xm)
#    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-40, 40)
    figABC.savefig('{0}-waveFront.png'.format(prefix))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()
    # plotFocus()
    # plotFront()

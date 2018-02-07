# -*- coding: utf-8 -*-
"""
!!! select one of the two functions to run at the very bottom !!!
!!! select 'rays', 'hybrid' or 'wave' below !!!
!!! select a desired emittance case below !!!

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
import xrt.backends.raycing.waves as rw
import xrt.backends.raycing.coherence as rco

showIn3D = False

mAu = rm.Material('Au', rho=19.32)
mRh = rm.Material('Rh', rho=12.41)
mGolden = rm.Material('Au', rho=19.32)
mGoldenGrating = rm.Material('Au', rho=19.32, kind='grating')
mRhodiumGrating = rm.Material('Rh', rho=12.41, kind='grating')
mAuFZP = rm.Material('Au', rho=19.3, kind='FZP')
mRhFZP = rm.Material('Rh', rho=12.41, kind='FZP')

E0 = 280.
dE = 0.5
targetHarmonic = 1
acceptanceHor = 2.2e-4  # FE acceptance, full angle, mrad
acceptanceVer = 4.2e-4  # FE acceptance

dFocus = np.linspace(-100, 100, 9) if not showIn3D else [0]

imageExtent = [-50, 50, -50, 50]

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

pitch = np.radians(1)

cff = 1.6        # mono parameters
fixedExit = 20.  # mm
rho = 300.       # lines/mm
blaze = np.radians(0.6)

#ESradius = 0.06  # in mm EXIT SLIT RADIUS
ESdX = 2.  # in mm EXIT SLIT RADIUS
ESdZ = 0.1  # in mm EXIT SLIT RADIUS

#ZPdiam = 300.                 # diameter of ZP, microns
#outerzone = 30.               # diameter of outermost zone of a ZP, nm
#wavelength = 1239.84187 / E0  # nanometers
#
#focus = ZPdiam*(1e-3)*outerzone/wavelength  # focal distance, mm
#
#dFocus = np.linspace(-0., 0., 1)
#
#Nzone = ZPdiam/(4*outerzone*1e-3)

#print('f_ZP: = {0} mm'.format(focus))
#print('N_ZP: = {0}'.format(Nzone))

repeats = 10
nrays = 1e5

#what = 'rays'
what = 'hybrid'
#what = 'wave'

if what == 'rays':
    prefix = 'cxi_2D-1-rays-'
elif what == 'hybrid':
    prefix = 'cxi_2D-2-hybr-'
elif what == 'wave':
    prefix = 'cxi_2D-3-wave-'

xbins, xppb = 64, 4
zbins, zppb = 64, 4

is0emittance = True
if is0emittance:
    emittanceFactor = 0.
    prefix += '0emit-'
else:
    emittanceFactor = 1.
    prefix += 'non0e-'
    nrays = 1e5
    repeats = 256

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
hFactor = 1.
#prefix += '025%H-'


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
        xPrimeMax=acceptanceHor/2*1e3,
        zPrimeMax=acceptanceVer/2*1e3,
        xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
#        targetOpenCL='CPU',
        uniformRayDensity=True,
        filamentBeam=(what != 'rays'))

    opening = [-acceptanceHor*pFE/2, acceptanceHor*pFE/2,
               -acceptanceVer*pFE/2, acceptanceVer*pFE/2]
    beamLine.slitFE = ra.RectangularAperture(
        beamLine, 'FE slit', kind=['left', 'right', 'bottom', 'top'],
        opening=opening)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM-M1')

    beamLine.m1 = roe.ToroidMirror(
        beamLine, 'M1', surface=('Au',), material=(mAu,),
        limPhysX=(-5, 5), limPhysY=(-150, 150),
        positionRoll=np.pi/2, R=1e22, alarmLevel=0.1)
#    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM-M1')

    beamLine.m2 = roe.OE(
        beamLine, 'M2', surface=('Au',), material=(mAu,),
        limPhysX=(-5, 5), limPhysY=(-225, 225), alarmLevel=0.1)

    gratingKW = dict(
        positionRoll=np.pi, limPhysX=(-2, 2),
        limPhysY=(-40, 40), alarmLevel=0.1)
    if what == 'rays':
        beamLine.pg = Grating(beamLine, 'PlaneGrating',
                              material=mGoldenGrating, **gratingKW)
        beamLine.pg.material.efficiency = [(1, 0.4)]
    else:
        beamLine.pg = roe.BlazedGrating(
            beamLine, 'BlazedGrating', material=mGolden, blaze=blaze,
            rho=rho, **gratingKW)
    beamLine.pg.order = 1
#    beamLine.fsmPG = rsc.Screen(beamLine, 'FSM-PG')

    beamLine.m3 = roe.ToroidMirror(
        beamLine, 'M3', surface=('Au',), material=(mAu,),
        positionRoll=-np.pi/2, limPhysX=(-10., 10.), limPhysY=(-100., 100.),
        alarmLevel=0.1)
    beamLine.fsm3 = rsc.Screen(beamLine, 'FSM-M3')

#    beamLine.exitSlit = ra.RoundAperture(
#         beamLine, 'ExitSlit', r=ESradius, alarmLevel=None)
    beamLine.exitSlit = ra.RectangularAperture(
         beamLine, 'ExitSlit',
         opening=[-ESdX*hFactor/2, ESdX*hFactor/2,
                  -ESdZ*vFactor/2, ESdZ*vFactor/2])

    beamLine.m4 = roe.EllipticalMirrorParam(
        beamLine, 'M4', surface=('Au',), material=(mAu,),
        positionRoll=np.pi/2, pitch=pitch, isCylindrical=True,
        p=43000., q=dM45+pExp, limPhysX=(-0.5, 0.5),
        limPhysY=(-70., 70.), alarmLevel=0.2)

    beamLine.m5 = roe.EllipticalMirrorParam(
        beamLine, 'M5', surface=('Au',), material=(mAu,),
        yaw=-2*pitch, pitch=pitch, isCylindrical=True,
        p=dM4ES+dM45, q=pExp,
        limPhysX=(-0.5, 0.5), limPhysY=(-40., 40.), alarmLevel=0.2)

    beamLine.fsmExp = rsc.Screen(beamLine, 'FSM-Exp')

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
    beamLine.m1.pitch = pitch
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
#    print('N = {0}'.format(Nzone))

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
    if what != 'rays':  # this is here because it needs pitch value
        drho = beamLine.pg.get_grating_area_fraction()
        beamLine.pg.areaFraction = drho
        print(u'PG areaFraction = {0}'.format(beamLine.pg.areaFraction))

#    pM3mer = pM1 + pPG + pM3  # pM3sag = infinity
    sinPitchM3 = np.sin(pitch)
    rM3 = 2. * sinPitchM3 * qM3sag   # focusing
    print('M3: r = {0} mm'.format(rM3))
    beamLine.m3.center = [0, pPG + pM3, fixedExit]
    beamLine.m3.pitch = -pitch  # opposite angles to M1
    beamLine.m3.r = rM3
    beamLine.m3.R = 1e22  # no hor focusing: M3 cylindrical
    beamLine.fsm3.center = beamLine.m3.center

    beamLine.exitSlit.center = -qM3sag * np.sin(2*pitch),\
        beamLine.m3.center[1] + qM3sag * np.cos(2*pitch), fixedExit

    beamLine.m4.center = -(qM3sag+dM4ES) * np.sin(2*pitchM3),\
        beamLine.m3.center[1] + (qM3sag+dM4ES) * np.cos(2*pitchM3), fixedExit
    print('M4: p={0}, q={1}'.format(beamLine.m4.p, beamLine.m4.q))

    beamLine.m5.center = beamLine.m4.center[0],\
        beamLine.m4.center[1] + dM45, fixedExit
    print('M5: p={0}, q={1}'.format(beamLine.m5.p, beamLine.m5.q))

#    beamLine.fsmExp.center = \
#        beamLine.m4.center[0] + (dM45+pExp) * np.sin(pitchM3-pitchM4),\
#        beamLine.m4.center[1] + (pExp+dM45) * np.cos(pitchM3-pitchM4),\
#        fixedExit + pExp*np.tan(2*pitchM5)
#    beamLine.fsmExp.z = 0, -np.sin(2*pitchM5), np.cos(2*pitchM5)

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
    if True:  # to wave
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
    beamM4global, beamM4local = beamLine.m4.reflect(beamM3global)
    beamM5global, beamM5local = beamLine.m5.reflect(beamM4global)
    beamFSMExp = beamLine.fsmExp.expose(beamM5global)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamM4global': beamM4global, 'beamM4local': beamM4local,
               'beamM5global': beamM5global, 'beamM5local': beamM5local,
               'beamFSMExp': beamFSMExp
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
    if True:  # to wave
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
        beamTom3 = rw.diffract(beamPGlocal, waveOnm3)
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        rw.diffract(beamM3local, waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnm4 = beamLine.m4.prepare_wave(beamLine.exitSlit, nrays)
        beamTom4 = rw.diffract(waveOnExitSlit, waveOnm4)
        beamM4global, beamM4local = beamLine.m4.reflect(
            beamTom4, noIntersectionSearch=True)

        waveOnm5 = beamLine.m5.prepare_wave(beamLine.m4, nrays)
        beamTom5 = rw.diffract(beamM4local, waveOnm5)
        beamM5global, beamM5local = beamLine.m5.reflect(
            beamTom5, noIntersectionSearch=True)

        for waveOnSample in waveOnSamples:
            rw.diffract(beamM5local, waveOnSample)

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

    return outDict


def run_process_wave(beamLine, shineOnly1stSource=False):
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False

    waveOnSamples = []
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
        beamTom1 = rw.diffract(waveOnSlit, waveOnm1)
        beamM1global, beamM1local = beamLine.m1.reflect(
            beamTom1, noIntersectionSearch=True)

        waveOnm2 = beamLine.m2.prepare_wave(beamLine.m1, nrays)
        beamTom2 = rw.diffract(beamM1local, waveOnm2)
        beamM2global, beamM2local = beamLine.m2.reflect(
            beamTom2, noIntersectionSearch=True)

        waveOnPG = beamLine.pg.prepare_wave(beamLine.m2, nrays)
        beamToPG = rw.diffract(beamM2local, waveOnPG)
        beamPGglobal, beamPGlocal = beamLine.pg.reflect(
            beamToPG, noIntersectionSearch=True)

        beamPGlocal.area = 0
        beamPGlocal.areaFraction = beamLine.pg.areaFraction

        waveOnm3 = beamLine.m3.prepare_wave(beamLine.pg, nrays)
        beamTom3 = rw.diffract(beamPGlocal, waveOnm3)
#        beamM3local = waveOnm3
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        rw.diffract(beamM3local, waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnm4 = beamLine.m4.prepare_wave(beamLine.exitSlit, nrays)
        beamTom4 = rw.diffract(waveOnExitSlit, waveOnm4)
        beamM4global, beamM4local = beamLine.m4.reflect(
            beamTom4, noIntersectionSearch=True)

        waveOnm5 = beamLine.m5.prepare_wave(beamLine.m4, nrays)
        beamTom5 = rw.diffract(beamM4local, waveOnm5)
        beamM5global, beamM5local = beamLine.m5.reflect(
            beamTom5, noIntersectionSearch=True)

        for waveOnSample in waveOnSamples:
            rw.diffract(beamM5local, waveOnSample)

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

    return outDict


if what == 'rays':
    rr.run_process = run_process_rays
elif what.startswith('hybr'):
    rr.run_process = run_process_hybr
elif what == 'wave':
    rr.run_process = run_process_wave


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSM0', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        title='00-FE')
    plot.xaxis.fwhmFormatStr = '%.3f'
    plot.yaxis.fwhmFormatStr = '%.3f'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamM1local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m1.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m1.limPhysY),
        title='01-M1local')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamM2local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m2.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m2.limPhysY),
        title='02-M2local')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamPGlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.pg.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.pg.limPhysY),
        title='02-PGlocal')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamM3local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m3.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m3.limPhysY),
        title='03-M3local')
    plots.append(plot)

    caxis = xrtp.XYCAxis('Ep phase', '',
                         data=raycing.get_Ep_phase, limits=[-np.pi, np.pi])
    plot = xrtp.XYCPlot(
        'beamM3local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m3.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m3.limPhysY),
        caxis=caxis, title='03s-M3local')
    plots.append(plot)

    limMaxX = ESdX*0.6e3
    limMaxZ = ESdZ*0.6e3
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1, beamLine.exitSlit.lostNum),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        oe=beamLine.exitSlit,
        title='04-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    caxis = xrtp.XYCAxis('Es phase', '',
                         data=raycing.get_Es_phase, limits=[-np.pi, np.pi])
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        caxis=caxis, oe=beamLine.exitSlit, title='04s-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamM4local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m4.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m4.limPhysY),
        title='05-M4local')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamM5local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=beamLine.m5.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=beamLine.m5.limPhysY),
        title='06-M5local')
    plots.append(plot)

    complexPlotsIs = []
    complexPlotsEs = []
    complexPlotsPCAs = []
    for ic, (fsmExpCenter, d) in enumerate(
            zip(beamLine.fsmExpCenters, dFocus)):
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
            fluxKind='s', title='08i-ExpFocus-Is{0:02d}'.format(ic))
        plot.xaxis.limits = [imageExtent[0], imageExtent[1]]
        plot.yaxis.limits = [imageExtent[2], imageExtent[3]]
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plots.append(plot)
        complexPlotsIs.append(plot)

#        plot = xrtp.XYCPlot(
#            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
#            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
#            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
#            fluxKind='Es4D', title='08e-ExpFocus-Es{0:02d}'.format(ic))
#        plot.xaxis.limits = [imageExtent[0], imageExtent[1]]
#        plot.yaxis.limits = [imageExtent[2], imageExtent[3]]
#        plot.textPanel = plot.fig.text(
#            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
#            transform=plot.fig.transFigure, size=12, color='r', ha='center')
#        plots.append(plot)
#        complexPlotsEs.append(plot)
        complexPlotsEs.append(None)  # the above consumes too much

        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
            fluxKind='EsPCA', title='08pca-ExpFocus-Es{0:02d}'.format(ic))
        plot.xaxis.limits = [imageExtent[0], imageExtent[1]]
        plot.yaxis.limits = [imageExtent[2], imageExtent[3]]
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.0f} mm'.format(d),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plots.append(plot)
        complexPlotsPCAs.append(plot)
    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    for plot in plots:
        plot.saveName = [prefix + plot.title + '.png', ]
        if plot.caxis.label.startswith('energy'):
            plot.caxis.limits = E0-dE*vFactor, E0+dE*vFactor
            plot.caxis.offset = E0
        if plot.fluxKind.startswith('power'):
            plot.fluxFormatStr = '%.0f'
        else:
            plot.fluxFormatStr = '%.1p'
    return plots, complexPlotsIs, complexPlotsEs, complexPlotsPCAs


def afterScript(complexPlotsIs, complexPlotsEs, complexPlotsPCAs):
    if what == 'rays':
        return

    dump = []
    NN = len(complexPlotsEs)
    for ic, (complexPlotIs, complexPlotEs, complexPlotPCAs) in enumerate(
            zip(complexPlotsIs, complexPlotsEs, complexPlotsPCAs)):
        x = complexPlotIs.xaxis.binCenters
        y = complexPlotIs.yaxis.binCenters
        Ixy = complexPlotIs.total2D

        if complexPlotEs is None:
            wN, vN = None, None
        else:
            Exy4D = complexPlotEs.total4D
            print("solving eigenvalue problem, {0} of {1}...".format(ic+1, NN))
            start = time.time()
            wN, vN = rco.calc_eigen_modes_4D(Exy4D)
            stop = time.time()
            print("the eigenvalue problem has taken {0} s".format(stop-start))

        print("solving PCA problem, {0} of {1}...".format(ic+1, NN))
        Es = complexPlotPCAs.field3D
        start = time.time()
        wPCA, vPCA = rco.calc_eigen_modes_PCA(Es)
        stop = time.time()
        print("the PCA problem has taken {0} s".format(stop-start))

        dump.append([x, y, Ixy, wN, vN, Es, wPCA, vPCA])

    pickleName = '{0}-res.pickle'.format(prefix)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)
    print("Done")


def main():
    beamLine = build_beamline(azimuth=-2*pitch)
    align_beamline(beamLine)
    if showIn3D:
#        beamLine.orient_along_global_Y()
        beamLine.glow(scale=[100, 10, 1000], centerAt='M2')
        return
    plots, complexPlotsIs, complexPlotsEs, complexPlotsPCAs = \
        define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine,
                         afterScript=afterScript,
                         afterScriptArgs=[
                             complexPlotsIs, complexPlotsEs, complexPlotsPCAs],
                         processes=1)


def plotFocus():
    pickleName = '{0}-res.pickle'.format(prefix)
    with open(pickleName, 'rb') as f:
        dump = pickle.load(f)

    for ic, d in enumerate(dFocus):
        x, y, Ixy, wN, vN, Es, wPCA, vPCA = dump[ic]
        scrStr = 'screen at f+{0:.0f} mm'.format(d)

        if wN is not None:
            figE = rco.plot_eigen_modes(x, y, wN, vN,
                                        xlabel='x (µm)', ylabel='z (µm)')
            figE.suptitle('Eigen modes of mutual intensity, ' + scrStr,
                          fontsize=11)
            figE.savefig('Modes-{0}-{1:02d}.png'.format(prefix, ic))

        figP = rco.plot_eigen_modes(x, y, wPCA, vPCA,
                                    xlabel='x (µm)', ylabel='z (µm)')
        figP.suptitle('Principal components of one-electron images, ' + scrStr,
                      fontsize=11)
        figP.savefig('PCA-{0}-{1:02d}.png'.format(prefix, ic))

        xdata = rco.calc_1D_coherent_fraction(Es, 'x', x)
        ydata = rco.calc_1D_coherent_fraction(Es, 'z', y)
        fig21, figXZ = rco.plot_1D_degree_of_coherence(xdata, 'x', x)
        fig22, figXZ = rco.plot_1D_degree_of_coherence(ydata, 'z', y,
                                                       fig2=figXZ)
        fig21.suptitle('Mutual intensity for horizontal cut,\n '+scrStr,
                       size=11)
        fig22.suptitle('Mutual intensity for vertical cut,\n '+scrStr, size=11)
        figXZ.suptitle('Intensity and Degree of Coherence,\n '+scrStr, size=11)

        fig21.savefig('MutualI-x-{0}-{1:02d}.png'.format(prefix, ic))
        fig22.savefig('MutualI-z-{0}-{1:02d}.png'.format(prefix, ic))
        figXZ.savefig('DOC-{0}-{1:02d}.png'.format(prefix, ic))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()
#    plotFocus()

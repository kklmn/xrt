# -*- coding: utf-8 -*-
"""
!!! select one of the two functions to run at the very bottom !!!
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
import xrt.backends.raycing.waves as rw
import xrt.backends.raycing.coherence as rco

showIn3D = False

mRhodium = rm.Material('Rh', rho=12.41)
mGolden = rm.Material('Au', rho=19.32)
mGoldenGrating = rm.Material('Au', rho=19.32, kind='grating')
mRhodiumGrating = rm.Material('Rh', rho=12.41, kind='grating')
mAuFZP = rm.Material('Au', rho=19.3, kind='FZP')
mRhFZP = rm.Material('Rh', rho=12.41, kind='FZP')

acceptanceHor = 1e-4  # FE acceptance
acceptanceVer = 1e-4  # FE acceptance

pFE = 21000.
pM1 = 24000.
pPG = 2000.
pM3 = 4000.  # distance to PG
qM3mer = 7500.
qM3sag = 7500.
pFZP = qM3mer / 5

E0 = 280.
dE = 0.1
targetHarmonic = 1
dFocus = np.linspace(0.0005, 0.002, 16) if not showIn3D else [0]
gratingMaterial = mGoldenGrating
material = mGolden

#E0 = 2400.
#dE = 1.
#targetHarmonic = 3
#dFocus = np.linspace(0.016, 0.046, 16)
##dFocus = np.linspace(-0.03, 0.03, 13)
#pFZP *= 3
#gratingMaterial = mRhodiumGrating
#material = mRhodium

pitch = np.radians(1)

cff = 1.6        # mono parameters
fixedExit = 30.  # mm
rho = 300.       # lines/mm
blaze = np.radians(0.5)

#ESradius = 0.06  # in mm EXIT SLIT RADIUS
ESdX = 0.08  # in mm EXIT SLIT width
ESdZ = 0.12  # in mm EXIT SLIT height

ZPdiam = 300.                 # diameter of ZP, microns
outerzone = 20.               # diameter of outermost zone of a ZP, nm
wavelength = 1239.84187 / E0  # nanometers

focus = ZPdiam*(1e-3)*outerzone/wavelength  # focal distance, mm
Nzone = ZPdiam/(4*outerzone*1e-3)

print('f_ZP: = {0} mm'.format(focus))
print('N_ZP: = {0}'.format(Nzone))

repeats = 10
nrays = 1e5

what = 'rays'
#what = 'hybrid'
#what = 'wave'

if what == 'rays':
    prefix = 'stxm-1D-1-rays-'
elif what == 'hybrid':
    prefix = 'stxm-1D-2-hybr-'
elif what == 'wave':
    prefix = 'stxm-1D-3-wave-'
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

#==============================================================================
# vertical cut
#==============================================================================
#vShrink = 1.
#hShrink = 1e-3
#cut = 'ver-'
#prefix += cut
#xbins, xppb = 1, 8
#zbins, zppb = 128, 2

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
hFactor = 1
#prefix += '050%H-'
#prefix += 'x4dist-'
#pFZP *= 4
#dFocus = np.linspace(0.000, 0.0015, 16)
#dFocus = np.linspace(-0.0005, 0.001, 16)

hExtraFactor = 1
#prefix += '001%Hm3-'


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
        period=48., n=81, targetE=(E0, targetHarmonic),
        eMin=E0-dE*vFactor, eMax=E0+dE*vFactor,
        xPrimeMax=acceptanceHor/2*1e3*hShrink,
        zPrimeMax=acceptanceVer/2*1e3*vShrink,
        xPrimeMaxAutoReduce=False, zPrimeMaxAutoReduce=False,
        targetOpenCL='CPU',
        uniformRayDensity=True,
        filamentBeam=(what != 'rays'))

    opening = [-acceptanceHor*pFE/2*hShrink, acceptanceHor*pFE/2*hShrink,
               -acceptanceVer*pFE/2*vShrink, acceptanceVer*pFE/2*vShrink]
    beamLine.slitFE = ra.RectangularAperture(
        beamLine, 'FE slit', kind=['left', 'right', 'bottom', 'top'],
        opening=opening)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM-M1')

    xShrink = vShrink if what == 'wave' else 1
    yShrink = hShrink if what == 'wave' else 1
    beamLine.m1 = roe.ToroidMirror(
        beamLine, 'M1', surface=('Au',), material=(material,),
        limPhysX=(-5*xShrink, 5*xShrink), limPhysY=(-150*yShrink, 150*yShrink),
        positionRoll=np.pi/2, R=1e22,
        alarmLevel=0.1)
#    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM-M1')

    xShrink = hShrink if what == 'wave' else 1
    yShrink = vShrink if what == 'wave' else 1
    beamLine.m2 = roe.OE(
        beamLine, 'M2', surface=('Au',), material=(material,),
        limPhysX=(-5*xShrink, 5*xShrink), limPhysY=(-225*yShrink, 225*yShrink),
        alarmLevel=0.1)

    gratingKW = dict(
        positionRoll=np.pi, limPhysX=(-2*xShrink, 2*xShrink),
        limPhysY=(-40*yShrink, 40*yShrink), alarmLevel=0.1)
    if what == 'rays':
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
        beamLine, 'M3', surface=('Au',), material=(material,),
        positionRoll=-np.pi/2, limPhysX=(-2*vShrink, 2*vShrink),
        limPhysY=(-80*hShrink*hExtraFactor, 80*hShrink*hExtraFactor),
        #limPhysY=(-80*hShrink, 80*hShrink),
        alarmLevel=0.1)
    beamLine.fsm3 = rsc.Screen(beamLine, 'FSM-M3')

#    beamLine.exitSlit = ra.RoundAperture(
#         beamLine, 'ExitSlit', r=ESradius, alarmLevel=None)
    beamLine.exitSlit = ra.RectangularAperture(
         beamLine, 'ExitSlit',
         opening=[-ESdX/2*hFactor*hShrink, ESdX/2*hFactor*hShrink,
                  -ESdZ/2*vFactor*vShrink, ESdZ/2*vFactor*vShrink])

    beamLine.fzp = roe.NormalFZP(
        beamLine, 'FZP', pitch=np.pi/2, material=mAuFZP, f=focus, E=E0,
        N=Nzone)
    beamLine.fzp.order = 1
    beamLine.fzp.material.efficiency = [(1, 0.1)]  # used with rays
    print('outerzone = {0} mm'.format(beamLine.fzp.rn[-1]-beamLine.fzp.rn[-2]))
    beamLine.fzp.area = np.pi * beamLine.fzp.rn[-1]**2 / 2
    beamLine.fsmFZP = rsc.Screen(beamLine, 'FSM-FZP')

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


def align_beamline(beamLine, E0=E0):
    beamLine.source.center = pM1 * np.sin(2*pitch), \
        -pM1 * np.cos(2*pitch), 0     # centre of undu when M1 is (0,0,0)
    beamLine.slitFE.center = (pM1-pFE) * np.sin(2*pitch), \
        -(pM1-pFE) * np.cos(2*pitch), 0
    beamLine.fsm0.center = beamLine.slitFE.center

    print('FE opening {0}×{1}'.format(
        beamLine.slitFE.opening[1] - beamLine.slitFE.opening[0],
        beamLine.slitFE.opening[3] - beamLine.slitFE.opening[2]))

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
    print('N = {0}'.format(Nzone))

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

    pM3mer = pM1 + pPG + pM3  # pM3sag = infinity
    sinPitchM3 = np.sin(pitch)
    #rM3 = 2e22     # collimating
    rM3 = 2. * sinPitchM3 * qM3sag   # focusing
    #RM3 = 2*pM3mer/sinPitchM3   # collimating
    RM3 = 2. / sinPitchM3 * (pM3mer*qM3mer) / (pM3mer+qM3mer)  # focusing
    print('M3: r = {0} mm, R = {1} m'.format(rM3, RM3*1e-3))
    beamLine.m3.center = [0, pPG + pM3, fixedExit]
    beamLine.m3.pitch = -pitch  # opposite angles to M1
    beamLine.m3.r = rM3
    beamLine.m3.R = RM3
    beamLine.fsm3.center = beamLine.m3.center

    beamLine.exitSlit.center = -qM3sag * np.sin(2*pitch),\
        beamLine.m3.center[1] + qM3sag * np.cos(2*pitch), fixedExit

    beamLine.fzp.center = -(qM3sag+pFZP) * np.sin(2*pitch),\
        beamLine.m3.center[1] + (qM3sag+pFZP) * np.cos(2*pitch), fixedExit
    beamLine.fsmFZP.center = beamLine.fzp.center

    beamLine.fsmExpCenters = []
    for d in dFocus:
        beamLine.fsmExpCenters.append(
            [beamLine.fzp.center[0] - (focus+d) * np.sin(2*pitch),
             beamLine.fzp.center[1] + (focus+d) * np.cos(2*pitch), fixedExit])


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
    beamFSMFZP = beamLine.fsmFZP.expose(beamM3global)
    beamFZPglobal, beamFZPlocal = beamLine.fzp.reflect(beamM3global)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamFSMFZP': beamFSMFZP,
               'beamFZPlocal': beamFZPlocal,
               }

    for ic, fsmExpCenter in enumerate(beamLine.fsmExpCenters):
        beamLine.fsmExp.center = fsmExpCenter
        beamFSMExp = beamLine.fsmExp.expose(beamFZPglobal)
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
            beamLine.fzp, beamLine.fsmExpX, beamLine.fsmExpZ)
        waveOnSamples.append(waveOnSample)

    repeats = 1
    for repeat in range(repeats):
        if repeats > 1:
            print('wave repeats: {0} of {1} ...'.format(repeat+1, repeats))

#        waveOnFSM3 = beamLine.fsm3.prepare_wave(
#            beamLine.pg, beamLine.fsm3X, beamLine.fsm3Z)
#        rw.diffract(beamPGlocal, waveOnFSM3)

        waveOnm3 = beamLine.m3.prepare_wave(beamLine.pg, nrays)
        beamTom3 = rw.diffract(beamPGlocal, waveOnm3)
#        beamM3local = waveOnm3
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        rw.diffract(beamM3local, waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnFZP = beamLine.fzp.prepare_wave(
            beamLine.exitSlit, nrays, shape='round')
        rw.diffract(waveOnExitSlit, waveOnFZP)
        beamFZPlocal = waveOnFZP

        for waveOnSample in waveOnSamples:
            rw.diffract(waveOnFZP, waveOnSample)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               #'waveOnFSM3': waveOnFSM3
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamFZPlocal': beamFZPlocal,
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
            beamLine.fzp, beamLine.fsmExpX, beamLine.fsmExpZ)
        waveOnSamples.append(waveOnSample)

    repeats = 1
    for repeat in range(repeats):
        if repeats > 1:
            print('wave repeats: {0} of {1} ...'.format(repeat+1, repeats))

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

#        waveOnFSM3 = beamLine.fsm3.prepare_wave(
#            beamLine.pg, beamLine.fsm3X, beamLine.fsm3Z)
#        rw.diffract(beamPGlocal, waveOnFSM3)

        waveOnm3 = beamLine.m3.prepare_wave(beamLine.pg, nrays)
        beamTom3 = rw.diffract(beamPGlocal, waveOnm3)
#        beamM3local = waveOnm3
        beamM3global, beamM3local = beamLine.m3.reflect(
            beamTom3, noIntersectionSearch=True)

        waveOnExitSlit = beamLine.exitSlit.prepare_wave(beamLine.m3, nrays)
        rw.diffract(beamM3local, waveOnExitSlit)
        beamExitSlit = waveOnExitSlit

        waveOnFZP = beamLine.fzp.prepare_wave(
            beamLine.exitSlit, nrays, shape='round')

        rw.diffract(waveOnExitSlit, waveOnFZP)
        beamFZPlocal = waveOnFZP

        for waveOnSample in waveOnSamples:
            rw.diffract(waveOnFZP, waveOnSample)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1local': beamM1local,
               'beamM2local': beamM2local,
               'beamPGlocal': beamPGlocal,
               #'waveOnFSM3': waveOnFSM3
               'beamM3local': beamM3local,
               'beamExitSlit': beamExitSlit,
               'beamFZPlocal': beamFZPlocal,
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

    ePos = 1 if xbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamFSM0', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=zbins, ppb=zppb),
        ePos=ePos, title='00-FE')
    plot.xaxis.fwhmFormatStr = '%.3f'
    plot.yaxis.fwhmFormatStr = '%.3f'
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamM1local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m1.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m1.limPhysY),
        ePos=ePos, title='01-M1local')
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamM2local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m2.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m2.limPhysY),
        ePos=ePos, title='02-M2local')
    plots.append(plot)

    ePos = 1 if xbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamPGlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.pg.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.pg.limPhysY),
        ePos=ePos, title='02-PGlocal')
    plots.append(plot)

    ePos = 1 if zbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamM3local', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=zbins, ppb=zppb,
                           limits=beamLine.m3.limPhysX),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', bins=xbins, ppb=xppb,
                           limits=beamLine.m3.limPhysY),
        ePos=ePos, title='03-M3local')
    plots.append(plot)

    caxis = xrtp.XYCAxis('Ep phase', '',
                         data=raycing.get_Ep_phase, limits=[-np.pi, np.pi])
    ePos = 1 if zbins < 4 else 2
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
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1, beamLine.exitSlit.lostNum), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
        ePos=ePos, title='04-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    caxis = xrtp.XYCAxis('Es phase', '',
                         data=raycing.get_Es_phase, limits=[-np.pi, np.pi])
    ePos = 1 if xbins < 4 else 2
    plot = xrtp.XYCPlot(
        'beamExitSlit', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xbins, ppb=xppb),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=zbins, ppb=zppb),
        caxis=caxis, ePos=ePos, title='04s-ExitSlit')
    plot.xaxis.limits = -limMaxX, limMaxX
    plot.yaxis.limits = -limMaxZ, limMaxZ
    plots.append(plot)

    if what == 'rays':
        plot = xrtp.XYCPlot(
            'beamFSMFZP', (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(
                r'$x$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
            yaxis=xrtp.XYCAxis(
                r'$z$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
            title='05l-FZP')
        plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(
            r'$x$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
        yaxis=xrtp.XYCAxis(
            r'$y$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
        title='05-FZP')
    plots.append(plot)

    caxis = xrtp.XYCAxis('Es phase', '',
                         data=raycing.get_Es_phase, limits=[-np.pi, np.pi])
    plot = xrtp.XYCPlot(
        'beamFZPlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(
            r'$x$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
        yaxis=xrtp.XYCAxis(
            r'$y$', u'µm', bins=256, ppb=1, limits=[-ZPdiam/2, ZPdiam/2]),
        caxis=caxis, title='05s-FZP')
    plots.append(plot)

    complexPlotsIs = []
    complexPlotsPCAs = []
    for ic, (fsmExpCenter, d) in enumerate(
            zip(beamLine.fsmExpCenters, dFocus)):
        ePos = 1 if xbins < 4 else 2
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', r'nm', bins=xbins, ppb=xppb,
                               limits=[-160, 160]),
            yaxis=xrtp.XYCAxis(r'$z$', r'nm', bins=zbins, ppb=zppb,
                               limits=[-160, 160]),
            fluxKind='s', ePos=ePos, title='06-ExpFocus{0:02d}'.format(ic))
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.1f} µm'.format(d*1e3),
            transform=plot.fig.transFigure, size=12, color='r', ha='center')
        plots.append(plot)
        complexPlotsIs.append(plot)

        ePos = 1 if xbins < 4 else 2
        plot = xrtp.XYCPlot(
            'beamFSMExp{0:02d}'.format(ic), (1,), aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', r'nm', bins=xbins, ppb=xppb,
                               limits=[-160, 160]),
            yaxis=xrtp.XYCAxis(r'$z$', r'nm', bins=zbins, ppb=zppb,
                               limits=[-160, 160]),
            fluxKind='EsPCA', ePos=ePos,
            title='06e-ExpFocus{0:02d}'.format(ic))
        plot.textPanel = plot.fig.text(
            0.88, 0.8, u'f{0:+.1f} µm'.format(d*1e3),
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
        if plot.xaxis.bins < 4:
            plot.ax2dHist.xaxis.set_ticks([0])
            plot.xaxis.fwhmFormatStr = None
            plot.ax1dHistX.set_visible(False)
        if plot.yaxis.bins < 4:
            plot.ax2dHist.yaxis.set_ticks([0])
            plot.yaxis.fwhmFormatStr = None
            plot.ax1dHistY.set_visible(False)
    return plots, complexPlotsIs, complexPlotsPCAs


def afterScript(complexPlotsIs, complexPlotsPCAs):
    dump = []
    for ic, (complexPlotIs, complexPlotPCAs) in enumerate(
            zip(complexPlotsIs, complexPlotsPCAs)):
        x = complexPlotIs.xaxis.binCenters
        y = complexPlotIs.yaxis.binCenters
        Ixy = complexPlotIs.total2D
        Es = complexPlotPCAs.field3D
        dump.append([x, y, Ixy, Es])
    print(Ixy.shape, Es.shape)

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
    plots, complexPlotsIs, complexPlotsDs = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=repeats, beamLine=beamLine,
                         afterScript=afterScript,
                         afterScriptArgs=[complexPlotsIs, complexPlotsDs],
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
        x, y, Ixy, Es = dump[ic]
        scrStr = 'screen at f+{0:.1f} µm'.format(d*1000)

        print("solving PCA problem, {0} of {1}...".format(ic+1, NN))
        start = time.time()
        wPCA, vPCA = rco.calc_eigen_modes_PCA(Es)
        stop = time.time()
        print("the PCA problem has taken {0} s".format(stop-start))
        print("Top 4 eigen values (PCA) = {0}".format(wPCA))

        figP = rco.plot_eigen_modes(x, y, wPCA, vPCA,
                                    xlabel='x (µm)', ylabel='z (µm)')
        figP.suptitle('Principal components of one-electron images, ' + scrStr,
                      fontsize=11)
        figP.savefig('PCA-{0}-{1:02d}.png'.format(prefix, ic))

        xarr, xnam = (x, 'x') if cut.startswith('hor') else (y, 'z')
        xdata = rco.calc_1D_coherent_fraction(Es/norm, xnam, xarr)
        fig2D, figXZ = rco.plot_1D_degree_of_coherence(
            xdata, xnam, xarr, "nm", isIntensityNormalized=True,
            locLegend='lower left')
        fig2D.suptitle('Mutual intensity, ' + scrStr, size=11)
        figXZ.suptitle('Intensity and Degree of Coherence, ' + scrStr, size=11)
        fig2D.savefig('MutualI-{0}-{1:02d}.png'.format(prefix, ic))
        figXZ.savefig('IDOC-{0}-{1:02d}.png'.format(prefix, ic))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()
#    plotFocus()

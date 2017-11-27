# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import copy
#import matplotlib as mpl
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

showIn3D = False

mGold = rm.Material('Au', rho=19.3)
mGoldenGrating = rm.Material(
    'Au', rho=19.3, kind='grating',
    efficiency=[(1, 2)], efficiencyFile='efficiency1-LEG.txt')
#    efficiency=[(1, 1)], efficiencyFile='efficiency1-LEG.pickle')
#    efficiency=[(1, 0.3)])
#    )
E0 = 80.
dE = 0.01

#distE = 'lines'
#energies = np.linspace(E0-dE, E0+dE, 5)

distE = 'flat'
energies = E0-dE, E0+dE

#=============================================================================
# Do not put many scanEnergies AND s1openings together or else you'll have
# MemoryError due to the numerous plots. You should fix one of the two and scan
# the other.
#=============================================================================
#scanEnergies = np.linspace(E0 - dE*0.75, E0 + dE*0.75, 7)
scanEnergies = E0,
#s1openings = np.linspace(0.01, 0.25, 25)
s1openings = 0.03,

cff = 2.25
pitch = np.radians(2)
fixedExit = 30.
rho = 1221.


class Grating(roe.OE):
    def local_g(self, x, y, rho=rho):
        return 0, -rho, 0  # constant line spacing along y


def build_beamline(azimuth=0, nrays=raycing.nrays):
    beamLine = raycing.BeamLine(azimuth=azimuth, height=0)
    rs.GeometricSource(
        beamLine, 'MAX-IV',
        nrays=nrays, dx=0.187, dz=0.032, dxprime=77e-6, dzprime=70e-6,
        distE=distE, energies=energies, polarization='horizontal')
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0')
    beamLine.m1 = roe.ToroidMirror(
        beamLine, 'M1', surface=('Au',), material=(mGold,),
        limPhysX=(-10., 10.), limPhysY=(-150., 150.), positionRoll=np.pi/2,
        R=1e12, alarmLevel=0.2)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM-M1')
    beamLine.m2 = roe.OE(
        beamLine, 'M2', surface=('Au',), material=(mGold,),
        limPhysX=(-10., 10.), limPhysY=(-150., 150.), alarmLevel=0.2)
    beamLine.pg = Grating(
        beamLine, 'PlaneGrating', material=mGoldenGrating,
        positionRoll=np.pi, limPhysX=(-15., 15.), limPhysY=(-55., 55.),
        alarmLevel=0.2)
#    beamLine.pg.order = -2,-1,0,1,2,3
    beamLine.pg.order = 1
    beamLine.fsmPG = rsc.Screen(beamLine, 'FSM-PG')
    beamLine.m3 = roe.ToroidMirror(
        beamLine, 'M3', material=(mGold,),
        positionRoll=-np.pi/2, limPhysX=(-15., 15.), limPhysY=(-150., 150.),
        alarmLevel=0.2)
    beamLine.fsm3hf = rsc.Screen(beamLine, 'FSM-M3hf')
    beamLine.fsm3vf = rsc.Screen(beamLine, 'FSM-M3vf')

    beamLine.s1s = [
        ra.RectangularAperture(
            beamLine, 'vert. slit', [0, 0, 0],
            ('bottom', 'top'), [fixedExit-opening/2., fixedExit+opening/2.])
        for opening in s1openings]

    beamLine.m4 = roe.ToroidMirror(
        beamLine, 'M4', material=(mGold,),
        positionRoll=np.pi/2, limPhysX=(-15., 15.), limPhysY=(-150., 150.),
        alarmLevel=0.2)
    beamLine.fsmExp1 = rsc.Screen(beamLine, 'FSM-Exp1')
    beamLine.fsmExp2 = rsc.Screen(beamLine, 'FSM-Exp2')
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
#    beamLine.feFixedMask.propagate(beamSource)
    beamFSM0 = beamLine.fsm0.expose(beamSource)
    beamM1global, beamM1local = beamLine.m1.reflect(beamSource)
    beamFSM1 = beamLine.fsm1.expose(beamM1global)
    beamM2global, beamM2local = beamLine.m2.reflect(beamM1global)
    beamPGglobal, beamPGlocal = beamLine.pg.reflect(beamM2global)
    beamFSMPG = beamLine.fsmPG.expose(beamPGglobal)

    beamM3global, beamM3local = beamLine.m3.reflect(beamPGglobal)
    beamFSM3hf = beamLine.fsm3hf.expose(beamM3global)
    beamFSM3vf = beamLine.fsm3vf.expose(beamM3global)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamM1global': beamM1global, 'beamM1local': beamM1local,
               'beamFSM1': beamFSM1,
               'beamM2global': beamM2global, 'beamM2local': beamM2local,
               'beamPGglobal': beamPGglobal, 'beamPGlocal': beamPGlocal,
               'beamFSMPG': beamFSMPG,
               'beamM3global': beamM3global, 'beamM3local': beamM3local,
               'beamFSM3hf': beamFSM3hf, 'beamFSM3vf': beamFSM3vf,
               }
    for iopening, s1 in enumerate(beamLine.s1s):
        if showIn3D:
            beamM3globalCopy = beamM3global
        else:
            beamM3globalCopy = copy.deepcopy(beamM3global)
        beamTemp1 = s1.propagate(beamM3globalCopy)
        beamFSM3vs = beamLine.fsm3vf.expose(beamM3globalCopy)
        beamM4global, beamM4local = beamLine.m4.reflect(beamM3globalCopy)
        beamFSMExp1 = beamLine.fsmExp1.expose(beamM4global)
        beamFSMExp2 = beamLine.fsmExp2.expose(beamM4global)
        sti = '{0:02d}'.format(iopening)
        outDict['beamFSM3vsOp'+sti] = beamFSM3vs
        outDict['beamM4globalOp'+sti] = beamM4global
        outDict['beamM4localOp'+sti] = beamM4local
        outDict['beamFSMExp1Op'+sti] = beamFSMExp1
        outDict['beamFSMExp2Op'+sti] = beamFSMExp2
        if showIn3D:
            break

    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process

eps = 1e-5


def align_grating(grating, E, m, cff):
    g = grating.local_g(0, 0)
    rho = np.dot(g, g)**0.5

    order = abs(m) if cff > 1 else -abs(m)
    f1 = cff**2 + 1
    f2 = cff**2 - 1
    if abs(f2) < eps:
        raise ValueError('cff is not allowed to be close to 1!')

    ml_d = order * rho * rm.ch / E * 1e-7
    cosAlpha = np.sqrt(-ml_d**2 * f1 + 2*abs(ml_d) *
                       np.sqrt(f2**2 + cff**2 * ml_d**2)) / abs(f2)
    cosBeta = cff * cosAlpha
    alpha = np.arccos(cosAlpha)
    beta = -np.arccos(cosBeta)
    return alpha, beta


def align_beamline(
    beamLine, E0=E0, pitchM1=pitch, cff=cff, fixedExit=fixedExit,
        pitchM3=pitch, pitchM4=pitch):
    pM1 = 12000.
    beamLine.sources[0].center = pM1 * np.sin(2*pitchM1), \
        -pM1 * np.cos(2*pitchM1), 0
    beamLine.fsm0.center = beamLine.sources[0].center

    rM1 = 2. * pM1 * np.sin(pitchM1)
    print('M1: r = {0} mm'.format(rM1))
    beamLine.m1.center = 0, 0, 0
    beamLine.m1.pitch = pitchM1
    beamLine.m1.r = rM1

    beamLine.fsm1.center = beamLine.m1.center

    if isinstance(beamLine.pg.order, int):
        m = beamLine.pg.order
    else:
        m = beamLine.pg.order[0]
    alpha, beta = align_grating(beamLine.pg, E0, m=m, cff=cff)
    includedAngle = alpha - beta
    print('alpha = {0} deg'.format(np.degrees(alpha)))
    print('beta = {0} deg'.format(np.degrees(beta)))
    print('included angle = {0} deg'.format(np.degrees(includedAngle)))
    print('cos(beta)/cos(alpha) = {0}'.format(np.cos(beta)/np.cos(alpha)))
    t = -fixedExit / np.tan(includedAngle)
    print('t = {0} mm'.format(t))
    pPG = 3000.
    beamLine.m2.center = 0, pPG - t, 0
    beamLine.m2.pitch = (np.pi - includedAngle) / 2.
    print('M2 pitch = {0} deg'.format(np.degrees(beamLine.m2.pitch)))
    beamLine.m2.yaw = -2 * pitchM1
    beamLine.pg.pitch = -(beta + np.pi/2)
    print('PG pitch = {0} deg'.format(np.degrees(beamLine.pg.pitch)))
    beamLine.pg.center = 0, pPG, fixedExit
    beamLine.pg.yaw = -2 * pitchM1

    beamLine.fsmPG.center = 0, beamLine.pg.center[1]+1000, 0

    pM3 = 1000.
    pM3mer = pM1 + pPG + pM3
    qM3mer = 5000.
    qM3sag = 7000.
    sinPitchM3 = np.sin(pitchM3)
    rM3 = 2. * sinPitchM3 * qM3sag
    RM3 = 2. / sinPitchM3 * (pM3mer*qM3mer) / (pM3mer+qM3mer)
    print('M3: r = {0} mm, R = {1} m'.format(rM3, RM3*1e-3))
    beamLine.m3.center = 0, pPG + pM3, fixedExit
    beamLine.m3.pitch = -2*pitchM1 + pitchM3
    beamLine.m3.r = rM3
    beamLine.m3.R = RM3

    beamLine.fsm3hf.center = -qM3mer * np.sin(2*pitchM3),\
        beamLine.m3.center[1] + qM3mer * np.cos(2*pitchM3), 0
    beamLine.fsm3vf.center = -qM3sag * np.sin(2*pitchM3),\
        beamLine.m3.center[1] + qM3sag * np.cos(2*pitchM3), 0
    for s1 in beamLine.s1s:
        s1.center = beamLine.fsm3vf.center

    pM4mer = 5000.
    pM4sag = 3000.
    qM4 = 3500.
    sinPitchM4 = np.sin(pitchM4)
    rM4 = 2. * sinPitchM4 * (pM4sag*qM4) / (pM4sag+qM4)
    RM4 = 2. / sinPitchM4 * (pM4mer*qM4) / (pM4mer+qM4)
    print('M4: r = {0} mm, R = {1} m'.format(rM4, RM4*1e-3))
    dM34 = qM3mer + pM4mer  # = qM3sag + pM4sag
    beamLine.m4.center = -dM34 * np.sin(2*pitchM3),\
        beamLine.m3.center[1] + dM34 * np.cos(2*pitchM3), fixedExit
    beamLine.m4.pitch = 2*pitchM1 - 2*pitchM3 + pitchM4
    beamLine.m4.r = rM4
    beamLine.m4.R = RM4

    qFSMExp1 = 1500.  # upstream of the focus
    beamLine.fsmExp1.center = beamLine.m4.center[0],\
        beamLine.m4.center[1] + qM4 - qFSMExp1, 0
    beamLine.fsmExp2.center = beamLine.m4.center[0],\
        beamLine.m4.center[1] + qM4, 0


def define_plots(beamLine):
    plots = []
    plotsMono = []
    plotsFocus = []

#    plot = xrtp.XYCPlot('beamSource', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm'))
#    plot.xaxis.fwhmFormatStr = '%.3f'
#    plot.yaxis.fwhmFormatStr = '%.3f'
#    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM0', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        title='00-FSM0')
    plot.xaxis.fwhmFormatStr = '%.3f'
    plot.yaxis.fwhmFormatStr = '%.3f'
    plots.append(plot)
#
#    plot = xrtp.XYCPlot(
#        'beamM1local', (1,), aspect='auto',
#        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
#        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-150, 150]),
#        title='01-M1local')
#    plots.append(plot)
#
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-4, 4]),
        title='02-FSM1')
    plots.append(plot)
#
#    plot = xrtp.XYCPlot(
#        'beamM2local', (1,), aspect='auto',
#        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
#        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-55, 55]),
#        title='03-M2local')
#    plots.append(plot)
#
    plot = xrtp.XYCPlot(
        'beamPGlocal', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-55, 55]),
        title='04-PGlocal')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMPG', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[fixedExit-4, fixedExit+4]),
        title='05-FSMPG')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMPG', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[fixedExit-4, fixedExit+4]),
        caxis=xrtp.XYCAxis('path', 'mm'),
        title='05-FSMPG-P')
    plot.caxis.offset = 16000
    plots.append(plot)

#    plot = xrtp.XYCPlot(
#        'beamM3local', (1,), aspect='auto',
#        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
#        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-150, 150]),
#        title='06-M3local')
#    plots.append(plot)
#
#    plot = xrtp.XYCPlot(
#        'beamFSM3hf', (1,),
#        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.5, 0.5]),
#        yaxis=xrtp.XYCAxis(r'$z$', 'mm',
#                           limits=[fixedExit-0.5, fixedExit+0.5]),
#        title='07-FSM3hf')
#    plots.append(plot)
#
#    for is1, (s1, op) in enumerate(zip(beamLine.s1s, s1openings)):
#        sti = '{0:02d}'.format(is1)
#
#        plot = xrtp.XYCPlot(
#            'beamFSM3vf', (1,),
#            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.5, 0.5]),
#            yaxis=xrtp.XYCAxis(r'$z$', 'mm',
#                               limits=[fixedExit-0.5, fixedExit+0.5]),
#            title='08-FSM3vfOp'+sti, oe=s1)
#        plots.append(plot)
#
#        plot = xrtp.XYCPlot(
#            'beamFSM3vsOp'+sti, (1,),
#            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.5, 0.5]),
#            yaxis=xrtp.XYCAxis(r'$z$', 'mm',
#                               limits=[fixedExit-0.5, fixedExit+0.5]),
#            title='09-FSM3vsOp'+sti)
#        plots.append(plot)
#        plotsMono.append(plot)
#
#        plot = xrtp.XYCPlot(
#            'beamM4localOp'+sti, (1,), aspect='auto',
#            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-4, 4]),
#            yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-150, 150]),
#            title='10-M4localOp'+sti)
#        plots.append(plot)
#        plotsMono.append(plot)
#
#        plot = xrtp.XYCPlot(
#            'beamFSMExp1Op'+sti, (1,),
#            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-1, 1]),
#            yaxis=xrtp.XYCAxis(r'$z$', 'mm',
#                               limits=[fixedExit-1, fixedExit+1]),
#            title='11-FSMExp1Op'+sti)
#        plot.xaxis.fwhmFormatStr = '%.3f'
#        plot.yaxis.fwhmFormatStr = '%.3f'
#        plots.append(plot)
#        plotsMono.append(plot)
#
#        plot = xrtp.XYCPlot(
#            'beamFSMExp2Op'+sti, (1,),
#            xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.25, 0.25]),
#            yaxis=xrtp.XYCAxis(r'$z$', 'mm',
#                               limits=[fixedExit-0.25, fixedExit+0.25]),
#            caxis=xrtp.XYCAxis('energy', 'eV', bins=256, ppb=1),
#            title='12-FSMExp2Op'+sti)
#        plot.xaxis.fwhmFormatStr = '%.3f'
#        plot.yaxis.fwhmFormatStr = '%.3f'
#        if len(s1openings) > 1:
#            plot.textPanel = plot.fig.text(
#                0.8, 0.8, u'slit opening\n{0:.0f} µm'.format(op*1e3),
#                transform=plot.fig.transFigure, size=14, color='r', ha='left')
#        plots.append(plot)
#        plotsMono.append(plot)
#        plotsFocus.append(plot)

    for plot in plots:
        if "energy" in plot.caxis.label:
            plot.caxis.limits = [E0-dE, E0+dE]
            plot.caxis.offset = E0
        if plot in plotsMono:
            plot.caxis.fwhmFormatStr = '%.4f'
        else:
            plot.caxis.fwhmFormatStr = None
    return plots, plotsMono, plotsFocus


def plot_generator(plots, plotsMono, plotsFocus, beamLine):
    for ienergy, energy in enumerate(scanEnergies):
        align_beamline(beamLine, E0=energy)
        for plot in plots:
            plot.saveName = 'FlexPES-{0}-{1}.png'.format(
                plot.title, ienergy)
        yield
    if len(s1openings) > 1:
        flux = np.array([plot.intensity for plot in plotsFocus])
        dE = np.array([E0/plot.dE*1e-4 for plot in plotsFocus])
        op = np.array(s1openings)
        fig = plt.figure(figsize=(5, 4), dpi=72)
        fig.subplots_adjust(right=0.88, bottom=0.12)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_title('At E = 40 eV')
        ax1.plot(op*1e3, dE, '-r', lw=2)
        ax2.plot(op*1e3, flux/max(flux), '-b', lw=2)
        ax1.set_xlabel(u'slit opening (µm)')
        ax1.set_ylabel(r'energy resolution $E/dE\times10^{4}$', color='r')
        ax2.set_ylabel('relative flux', color='b')
        fig.savefig('FlexPES-dE.png')


def main():
    beamLine = build_beamline(azimuth=-2*pitch, nrays=10000)
    align_beamline(beamLine)
    if showIn3D:
        beamLine.glow(scale=[100, 10, 1000], centerAt='M2')
        return
    plots, plotsMono, plotsFocus = define_plots(beamLine)
    args = [plots, plotsMono, plotsFocus, beamLine]
    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine,
                         generator=plot_generator, generatorArgs=args,
                         processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

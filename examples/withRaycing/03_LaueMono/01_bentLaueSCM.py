# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
#import matplotlib as mpl

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

prefix = '01_BentLaueSCM'

energies = [9e3, 1.6e4, 2.5e4, 3.6e4]
radii = [1e3, 5e3, 25e3, 125e3, np.inf]
dEOverE = [8e-2, 1.5e-2, 0.4e-2, 1.2e-3, 4e-4]  # @ radii
ddEOverE = [1, 1.25, 2.5, 3.]  # @ energies
#polarization = ['hor', 'vert', '+45', '-45', 'right', 'left', None]
polarization = 'hor',

#crystalDiamond = rm.CrystalDiamond((1,1,1), 2.0592872, elements='C',
siCrystal = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue', t=0.2)
pLaueSCM = 1000.
qLaueSCM = 100.
limitsFSM = -8, 8


def build_beamline():
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(beamLine, 'GeometricSource', dx=3., dz=3.,
                       dxprime=1.6e-4, distzprime=None)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, pLaueSCM - 100, 0))
    beamLine.laueSCM = roe.BentLaueCylinder(
        beamLine, 'LaueSCM', (0, pLaueSCM, 0), material=(siCrystal,))
    beamLine.fsm2 = rsc.Screen(beamLine, 'FSM2', [0, pLaueSCM + qLaueSCM, 0])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamLaueSCMglobal, beamLaueSCMlocal = beamLine.laueSCM.reflect(beamSource)
    beamFSM2 = beamLine.fsm2.expose(beamLaueSCMglobal)
    outDict = {'beamSource': beamSource,
               'beamFSM1': beamFSM1,
               'beamLaueSCMglobal': beamLaueSCMglobal,
               'beamLaueSCMlocal': beamLaueSCMlocal,
               'beamFSM2': beamFSM2}
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot1 = xrtp.XYCPlot(
        'beamFSM1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.fsm1.name+'_E')
    plot1.caxis.invertAxis = True
    plots.append(plot1)

    plot = xrtp.XYCPlot(
        'beamLaueSCMlocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.laueSCM.name+'_E')
    plot.caxis.invertAxis = True
    plots.append(plot)

    plot2 = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        ePos=1, title=beamLine.fsm2.name+'_E')
    plot2.caxis.invertAxis = True
    plot2.textPanel = plot2.fig.text(
        0.88, 0.8, '', transform=plot2.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot2)

    return plots


def plot_generator(plots, beamLine):
    for polar in polarization:
        beamLine.sources[0].polarization = polar
        suffix = polar
        if suffix is None:
            suffix = 'none'
        for iradius in range(5):
            radius = radii[iradius]
            beamLine.laueSCM.R = radius
            if radius == np.inf:
                radiusStr1 = 'inf'
                radiusStr2 = r'$\infty$'
            elif radius == -np.inf:
                radiusStr1 = '-inf'
                radiusStr2 = r'$-\infty$'
            else:
                radiusStr1 = '{0:05.1f} m'.format(radius * 1e-3)
                radiusStr2 = '{0:1.1f} m'.format(radius * 1e-3)
            for ienergy, energy in enumerate(energies):
                theta0 = np.arcsin(rm.ch / (2*siCrystal.d*energy))
#                   - siCrystal.get_dtheta_symmetric_Bragg(energy)
                dEE = dEOverE[iradius] * ddEOverE[ienergy]
                eAxisMin = energy * (1 - dEE/2.)
                eAxisMax = energy * (1 + dEE/2.)
                fsm2z = qLaueSCM * np.tan(2 * theta0)
                if plots:
                    plots[-1].yaxis.limits = fsm2z + limitsFSM[0], \
                        fsm2z + limitsFSM[1]
                alpha = 0  # asymmetry angle:
                pitch = np.pi/2 + theta0 + alpha
                beamLine.laueSCM.pitch = pitch
                beamLine.laueSCM.set_alpha(alpha)
                for distE in 'flat', :  # 'lines':
                    if distE == 'flat':
                        beamLine.sources[0].energies = eAxisMin, eAxisMax
                        sourcename = 'flat'
                    elif distE == 'lines':
                        beamLine.sources[0].energies = energy,
                        sourcename = 'line'
                    beamLine.sources[0].distE = distE
                    for plot in plots:
                        plot.caxis.offset = energy
                        plot.caxis.limits = [eAxisMin, eAxisMax]
                        fileName = '{0}_{1}_{2}_R={3}_{4:02.0f}keV_{5}'.\
                            format(prefix, plot.title, suffix, radiusStr1,
                                   energy * 1e-3, sourcename)
                        plot.saveName = fileName + '.png'
#                        plot.persistentName = fileName + '.pickle'
                        try:
                            plot.textPanel.set_text(
                                u'{0}\n$R=${1}\n'.format('', radiusStr2))
                        except:
                            pass
                    if showIn3D:
                        beamLine.glowFrameName =\
                            '{0}_{1}_R={2}_{3:02.0f}keV_{4}.jpg'.format(
                                prefix, suffix, radiusStr1, energy*1e-3,
                                sourcename)
                    yield


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=[30, 3, 30], centerAt='LaueSCM', startFrom=1,
                      generator=plot_generator, generatorArgs=[[], beamLine])
        return
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots, repeats=360, generator=plot_generator,
        beamLine=beamLine, processes='half')


if __name__ == '__main__':
    main()

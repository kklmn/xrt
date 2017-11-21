# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import math
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

prefix = '02_bentLaueDCM'

energies = [9e3, 1.6e4, 2.5e4, 3.6e4]
radii = [1e3, 5e3, 25e3, 125e3, np.inf]
dEOverE = [1.6e-1, 3e-2, 8e-3, 2.4e-3, 8e-4]  # @ radii
ddEOverE = [1, 1.25, 2.5, 3.]  # @ energies
dthetaMax = [8., 2., 0.4, 0.4, 0.4]  # mrad, @ radii
ddthetaMax = [1., 0.5, 0.25, 0.125]  # @ energies
#polarization = ['hor', 'vert', '+45', '-45', 'right', 'left', None]
polarization = 'hor',

nThetas = 51

#crystalDiamond = rm.CrystalDiamond((1,1,1), 2.0592872, elements='C',
si111 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue', t=0.2)
fixedExit = 51.
pLaueDCM = 1000.
qLaueDCM = 100.


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(beamLine, 'GeometricSource', nrays=nrays,
                       dx=3., dz=3., dxprime=1.6e-4, distzprime=None)
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, pLaueDCM - 100, 0))
    beamLine.laueDCM1 = roe.BentLaueCylinder(
        beamLine, 'LaueDCM1', (0, pLaueDCM, 0), material=(si111,))
    beamLine.laueDCM2 = roe.BentLaueCylinder(
        beamLine, 'LaueDCM2', [0, 0, fixedExit], material=(si111,))
    beamLine.fsm2 = rsc.Screen(
        beamLine, 'FSM2', [0, pLaueDCM + qLaueDCM, fixedExit])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamLaueDCMglobal1, beamLaueDCMlocal1 = beamLine.laueDCM1.reflect(
        beamSource)
    beamLaueDCMglobal2, beamLaueDCMlocal2 = beamLine.laueDCM2.reflect(
        beamLaueDCMglobal1)
    beamFSM2 = beamLine.fsm2.expose(beamLaueDCMglobal2)
    outDict = {'beamSource': beamSource,
               'beamFSM1': beamFSM1,
               'beamLaueDCMglobal1': beamLaueDCMglobal1,
               'beamLaueDCMlocal1': beamLaueDCMlocal1,
               'beamLaueDCMglobal2': beamLaueDCMglobal2,
               'beamLaueDCMlocal2': beamLaueDCMlocal2,
               'beamFSM2': beamFSM2}
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    limitsFSM = -15, 15
    plot1 = xrtp.XYCPlot(
        'beamFSM1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.fsm1.name+'_E')
    plot1.caxis.invertAxis = True
    plot1.textPanel = plot1.fig.text(
        0.86, 0.8, '', transform=plot1.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot1)

    plot = xrtp.XYCPlot(
        'beamLaueDCMlocal1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.laueDCM1.name+'_E')
    plot.caxis.invertAxis = True
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamLaueDCMlocal2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.laueDCM2.name+'_E')
    plot.caxis.invertAxis = True
    plots.append(plot)

    plot2 = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=limitsFSM),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=limitsFSM),
        ePos=1, title=beamLine.fsm2.name+'_E')
    plot2.caxis.invertAxis = True
    plot2.textPanel = plot2.fig.text(
        0.86, 0.8, '', transform=plot2.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot2)
    
    return plots


def plot_generator(plots, beamLine):
    for polar in polarization:
        beamLine.sources[0].polarization = polar
        suffix = polar
        if suffix is None:
            suffix = 'none'
        for iradius in [0, 1]:
#        for iradius in [2, 3, 4]:  # range(5):
            radius = radii[iradius]
            beamLine.laueDCM1.R = radius
            beamLine.laueDCM2.R = radius
            if radius == np.inf:
                radiusStr1 = 'inf'
                radiusStr2 = r'$\infty$'
            elif radius == -np.inf:
                radiusStr1 = '-inf'
                radiusStr2 = r'$-\infty$'
            else:
                radiusStr1 = '{0:03.0f}m'.format(radius * 1e-3)
                radiusStr2 = '{0:.0f} m'.format(radius * 1e-3)
            for ienergy, energy in enumerate(energies):
                theta0 = math.asin(rm.ch / (2 * si111.d * energy))
                dEE = dEOverE[iradius] * ddEOverE[ienergy]
                eAxisMin = energy * (1 - dEE / 2.)
                eAxisMax = energy * (1 + dEE / 2.)
                alpha = 0  # -theta0
                pitch = math.pi/2 + theta0 + alpha
                beamLine.laueDCM1.pitch = pitch
                beamLine.laueDCM1.set_alpha(alpha)
                beamLine.laueDCM2.set_alpha(alpha)
                beamLine.laueDCM2.center[1] = pLaueDCM + fixedExit *\
                    math.cos(theta0 - alpha) / math.tan(2. * theta0)
                for distE in 'flat', :  # 'lines':
                    if distE == 'flat':
                        beamLine.sources[0].energies = eAxisMin, eAxisMax
                        sourcename = 'flat'
                    elif distE == 'lines':
                        beamLine.sources[0].energies = energy,
                        sourcename = 'line'
                    beamLine.sources[0].distE = distE
                    dtM = dthetaMax[iradius] * ddthetaMax[ienergy]
                    dthetas = np.linspace(-dtM, dtM, nThetas)
                    rcIntensity = []
                    for dtheta in dthetas:
                        beamLine.laueDCM2.pitch = pitch + dtheta * 1e-3
                        for plot in plots:
                            plot.fluxFormatStr = '%.2e'
                            plot.caxis.offset = energy
                            plot.caxis.limits = [eAxisMin, eAxisMax]
                            fileName = ('{0}_{1}_{2}_R={3}_{4:02.0f}keV' +
                                        '_{5}{6:.3f}mrad').format(
                                prefix, plot.title, suffix, radiusStr1,
                                energy * 1e-3, sourcename, dtheta)
                            plot.saveName = fileName + '.png'
#                                plot.persistentName = fileName + '.pickle'
                            try:
                                plot.textPanel.set_text(
                                    (u'{0}\n$R=${1}\n' +
                                     r'$d\theta=${2} mrad').format(
                                        '', radiusStr2,
                                        repr(round(dtheta, 3))))
                            except:
                                pass
                        if showIn3D:
                            beamLine.glowFrameName =\
                                '{0}_{1}_R={2}_{3:02.0f}keV_{4}{5:.3f}'\
                                'mrad.jpg'.format(prefix, suffix, radiusStr1,
                                                  energy*1e-3, sourcename,
                                                  dtheta)
                        yield
                        if not showIn3D:
                            rcIntensity.append(plots[-1].intensity)

                    if showIn3D:
                        return

                    fig3 = plt.figure(figsize=(7, 5), dpi=72)
                    ax1 = plt.subplot(111)
                    ax1.set_title(r'Rocking curve @$E=${0}keV and $R=${1}'.
                                  format(energy * 1e-3, radiusStr2))
                    ax1.set_xlabel(r'$d\theta$ (mrad)', fontsize=14)
                    ax1.set_ylabel(r'flux (a.u.)', fontsize=14)
                    rc = np.array(rcIntensity) / max(rcIntensity)
                    dt = dthetas
                    ax1.plot(dt, rc, 'r', lw=2)
                    ax1.set_xlim(dt[0], dt[-1])
                    ax1.set_ylim(0, 1)
                    fileName = '{0}_R{1}_{2:02.0f}keV_{3}.png'.\
                        format('rc', radiusStr1, energy * 1e-3, sourcename)
                    fig3.savefig(fileName)
                    plt.close(fig3)


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=[30, 3, 30], centerAt='LaueDCM1', startFrom=1,
                      generator=plot_generator, generatorArgs=[[], beamLine])
        return
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots, repeats=36, generator=plot_generator,
        beamLine=beamLine, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

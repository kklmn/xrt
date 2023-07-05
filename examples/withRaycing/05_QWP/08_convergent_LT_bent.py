# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "01 Jul 2023"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import math
import numpy as np
import matplotlib as mpl

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False
useTT = False  # elastically distorted crystal reflectivity or ideal crystal
tOCL = "auto"

E0 = 9000.
eLimits = E0, E0+2.5
prefix = '08_conv_LT_bent_{}'.format("withTT" if useTT else "geom")

crystalDiamond = rm.CrystalDiamond((1, 1, 1), 2.0592872, elements='C',
                                   geom='Laue transmitted', t=0.3,
                                   useTT=useTT)
if useTT:
    crystalDiamond.auto_PyTTE_Limits = False
theta0 = math.asin(rm.ch / (2 * crystalDiamond.d * (E0+1.2)))

yDCM = 21000.
yVFM = 24000.
ySample = 1.5 * yVFM
yQWP = ySample - 3000.


def build_beamline(nrays=raycing.nrays):
    fixedExit = 15.

    beamLine = raycing.BeamLine(azimuth=0, height=0)
    hDiv = 1.5e-3
    vDiv = 2.5e-4
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, dx=0.1, dy=0, dz=2., dxprime=hDiv/2, dzprime=0,
        distE='flat', energies=eLimits, polarization='horizontal')

    beamLine.feMovableMaskLT = ra.RectangularAperture(
        beamLine, 'FEMovableMaskLT', [0, 10000, 0], ('left', 'top'), [-10, 3.])
    beamLine.feMovableMaskRB = ra.RectangularAperture(
        beamLine, 'FEMovableMaskRB', [0, 10500, 0], ('right', 'bottom'),
        [10, -3.])
    beamLine.feMovableMaskLT.set_divergence(
        beamLine.sources[0], [-hDiv/2, vDiv/2])
    beamLine.feMovableMaskRB.set_divergence(
        beamLine.sources[0], [hDiv/2, -vDiv/2])

    si111 = rm.CrystalSi(hkl=(1, 1, 1), tK=-171+273.15)
    beamLine.dcm = roe.DCM(
        beamLine, 'DCM', (0, yDCM, 0), surface=('Si111',), material=(si111,))
    beamLine.dcm.bragg = math.asin(rm.ch / (2 * si111.d * E0))
    beamLine.dcm.cryst2perpTransl = fixedExit/2./math.cos(beamLine.dcm.bragg)

    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, yDCM + 700, 0))

    beamLine.vfm = roe.ToroidMirror(beamLine, 'VFM', (0, yVFM, fixedExit),
                                    pitch=4e-3)
    beamLine.vfm.R = yVFM / beamLine.vfm.pitch
    beamLine.vfm.r = 2. / 3. * yVFM * beamLine.vfm.pitch
    yFlatMirror = yVFM + 2000.
    zFlatMirror = (yFlatMirror - yVFM) * 2. * beamLine.vfm.pitch + fixedExit
    beamLine.vdm = roe.OE(
        beamLine, 'FlatMirror', (0, yFlatMirror, zFlatMirror),
        pitch=-beamLine.vfm.pitch, positionRoll=math.pi)

    beamLine.qwp = roe.BentLaueCylinder(
        beamLine, 'QWP', (0, yQWP, zFlatMirror), roll=math.pi/4,
        material=(crystalDiamond,), R=3000.,
        # Note: with precisionOpenCL='float32' may not converge!
        targetOpenCL=tOCL if useTT else None, precisionOpenCL='float64'
        )
    beamLine.qwp.pitch = theta0 + math.pi/2

    beamLine.fsm2 = rsc.Screen(beamLine, 'FSM2', (0, ySample, zFlatMirror))

    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()
    plotTemp1 = beamLine.feMovableMaskLT.propagate(beamSource)
    plotTemp2 = beamLine.feMovableMaskRB.propagate(beamSource)

    beamDCMglobal, beamDCMlocal1, beamDCMlocal2 = \
        beamLine.dcm.double_reflect(beamSource)
    beamFSM1 = beamLine.fsm1.expose(beamDCMglobal)

    beamVFMglobal, beamVFMlocal = beamLine.vfm.reflect(beamDCMglobal)
    beamVDMglobal, beamVDMlocal = beamLine.vdm.reflect(beamVFMglobal)

    beamQWPglobal, beamQWPlocal = beamLine.qwp.reflect(beamVDMglobal)
    beamFSM2 = beamLine.fsm2.expose(beamQWPglobal)

    outDict = {'beamSource': beamSource,
               'beamDCMglobal': beamDCMglobal,
               'beamDCMlocal1': beamDCMlocal1,
               'beamDCMlocal2': beamDCMlocal2,
               'beamFSM1': beamFSM1,
               'beamVFMglobal': beamVFMglobal,
               'beamVFMlocal': beamVFMlocal,
               'beamVDMglobal': beamVDMglobal,
               'beamVDMlocal': beamVDMlocal,
               'beamQWPglobal': beamQWPglobal,
               'beamQWPlocal': beamQWPlocal,
               'beamFSM2': beamFSM2
               }
    if showIn3D:
        beamLine.prepare_flow()
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    dE = beamLine.sources[0].energies[-1] - beamLine.sources[0].energies[0]
    midE = \
        (beamLine.sources[0].energies[-1] + beamLine.sources[0].energies[0])/2
    if dE < midE / 20.:
        fwhmFormatStrE = '%.2f'
        offsetE = E0
    else:
        fwhmFormatStrE = None
        offsetE = 0
    plots = []

#    plot = xrtp.XYCPlot('beamFSM1', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-20, 20]),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-20, 20]),
#      ePos=1, title=beamLine.fsm1.name+'_E')
#    plot.caxis.fwhmFormatStr = fwhmFormatStrE
#    plot.caxis.limits = eLimits
#    plot.caxis.offset = offsetE
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    plots.append(plot)

#    plot = xrtp.XYCPlot('beamVFMlocal', (1,2), aspect='auto',
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-20, 20]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
#      title='VFM_footprint')
#    plots.append(plot)

#    plot = xrtp.XYCPlot('beamQWPlocal', (1,2),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-10, 10]),
#      yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
#      title='QWP_footprint')
#    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
        ePos=1, title=beamLine.fsm2.name+'_E')
    plot.caxis.fwhmFormatStr = fwhmFormatStrE
    plot.caxis.limits = eLimits
    plot.caxis.offset = offsetE
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

    # plot = xrtp.XYCPlot(
    #     'beamFSM2', (1,),
    #     xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
    #     yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
    #     caxis=xrtp.XYCAxis('degree of polarization', '',
    #                         data=raycing.get_polarization_degree,
    #                         limits=[0, 1]),
    #     ePos=1, title=beamLine.fsm2.name+'_DegreeOfPol')
    # plot.textPanel = plot.fig.text(
    #     0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
    #     ha='center')
    # plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
        caxis=xrtp.XYCAxis('circular polarization rate', '',
                           data=raycing.get_circular_polarization_rate,
                           limits=[-1, 1]),
        ePos=1, title=beamLine.fsm2.name+'_CircPolRate')
    plot.caxis.fwhmFormatStr = '%.2f'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
        caxis=xrtp.XYCAxis('ratio of ellipse axes', '',
                           data=raycing.get_ratio_ellipse_axes,
                           limits=[-1, 1]),
        ePos=1, title=beamLine.fsm2.name+'_PolAxesRatio')
    plot.caxis.fwhmFormatStr = '%.2f'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

#    plot = xrtp.XYCPlot('beamFSM2', (1,),
#      xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
#      yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
#      caxis=xrtp.XYCAxis('angle of polarization ellipse', 'rad',
#      data=raycing.get_polarization_psi, limits=[-math.pi/2, math.pi/2]),
#      ePos=1, title=beamLine.fsm2.name+'_PolPsi')
#    plot.ax1dHistE.set_yticks(
#        (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2))
#    plot.ax1dHistE.set_yticklabels((r'-$\frac{\pi}{2}$', r'-$\frac{\pi}{4}$',
#      '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'))
#    plot.textPanel = plot.fig.text(
#        0.88, 0.8, '', transform=plot.fig.transFigure,
#        size=14, color='r', ha='center')
#    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-0.15, 0.15]),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-0.15, 0.15]),
        caxis=xrtp.XYCAxis('phase shift', '',
                           data=raycing.get_phase_shift,
                           limits=[-1, 1]),  # limits are in units of pi!
        ePos=1, title=beamLine.fsm2.name+'_PhaseShift')
    plot.caxis.fwhmFormatStr = '%.2f'
    formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
    plot.ax1dHistE.yaxis.set_major_formatter(formatter)
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)
    return plots


def plot_generator(plots, beamLine):
#    polarization = ['horiz', 'vert', '+45', '-45', 'right', 'left', None]
    polarization = 'horiz',

    crystalDiamond.t = 0.3  # in mm
    if useTT:
        posTheta = np.logspace(0, 10, 11, base=2)
    else:
        posTheta = np.logspace(0, 13, 14, base=2)
    departureTheta = np.hstack((-posTheta[::-1], 0, posTheta))
    Rnominal = ySample - yQWP

    for polar in polarization:
        beamLine.sources[0].polarization = polar
        suffix = polar
        if suffix is None:
            suffix = 'none'
#        sq2 = math.sqrt(2)
#        for radiusFactor in [2, sq2, 1, 1/sq2, 0.5, np.inf]:
#        for radiusFactor in (-1, 1, np.inf):
        for radiusFactor in -1, :
            radius = radiusFactor * Rnominal
            beamLine.qwp.R = radius
            for iTheta, dTheta in enumerate(departureTheta):
                beamLine.qwp.pitch = theta0 + math.pi/2 + \
                    math.radians(dTheta / 3600.)
                for plot in plots:
                    plot.xaxis.fwhmFormatStr = '%.1f'
                    plot.yaxis.fwhmFormatStr = '%.1f'
                    if radiusFactor is np.inf:
                        fileName = '{0}_{1}_{2}_R=inf_{4:02d}'.\
                            format(prefix, plot.title, suffix, radius,
                                   iTheta)
                    else:
                        fileName = '{0}_{1}_{2}_R={3:05.0f}mm_{4:02d}'.\
                            format(prefix, plot.title, suffix, radius,
                                   iTheta)
                    plot.saveName = fileName + '.png'
#                    plot.persistentName = fileName + '.pickle'
                    try:
                        if radiusFactor is np.inf:
                            plot.textPanel.set_text(
                                u'{0}\nR = {1}\n{2:+4.0f} arcsec'.
                                format(suffix, r'$\infty$', dTheta))
                        else:
                            plot.textPanel.set_text(
                                u'{0}\nR = {1:5.0f} mm\n{2:+4.0f} arcsec'.
                                format(suffix, radius, dTheta))
                    except AttributeError:
                        pass
                if showIn3D:
                    if radiusFactor is np.inf:
                        beamLine.glowFrameName = '{0}_{1}_R=inf_{3:02d}.jpg'.\
                            format(prefix, suffix, radius, iTheta)
                    else:
                        beamLine.glowFrameName = \
                            '{0}_{1}_R={2:05.0f}mm_{3:02d}.jpg'.\
                            format(prefix, suffix, radius, iTheta)
                yield


def main():
    beamLine = build_beamline()
    if showIn3D:
        beamLine.glow(scale=[3e2, 3, 3e2], centerAt='QWP', startFrom=-2,
                      generator=plot_generator, generatorArgs=[[], beamLine],
                      colorAxis='circular_polarization_rate',
                      colorAxisLimits=[-1, 1])
        return
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots, repeats=24, generator=plot_generator,
        beamLine=beamLine, globalNorm=True,
        processes=1 if useTT else 'half'
        )


if __name__ == '__main__':
    main()

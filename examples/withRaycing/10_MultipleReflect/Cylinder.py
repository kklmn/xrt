# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib as mpl

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
#import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc

mGold = None  # rm.Material('Au', rho=19.3)


class Cylinder(roe.OE):
    def __init__(self, *args, **kwargs):
        self.Rm = kwargs.pop('Rm', 10000.)  # R meridional
        roe.OE.__init__(self, *args, **kwargs)

    def local_z(self, x, y):
        return self.Rm - np.sqrt(self.Rm**2 - y**2)

    def local_n(self, x, y):
        a = np.zeros_like(x)  # -dz/dx
        b = -y * (self.Rm**2 - y**2)**(-0.5)  # -dz/dy
        c = 1.
        norm = (b**2 + 1)**0.5
        b /= norm
        c /= norm
        return [a, b, c]


class CylinderP(Cylinder):
    def local_r(self, s, phi):
        return self.Rm

    def local_n(self, s, phi):
        a = np.zeros_like(phi)  # -dz/dx
        b = -np.sin(phi)  # -dz/dy
        c = np.cos(phi)
        return [a, b, c]

    def xyz_to_param(self, x, y, z):  # for flat mirror as example
        return x, np.arctan(y / (Rm - z)), np.sqrt(y**2 + (Rm - z)**2)

    def param_to_xyz(self, s, phi, r):  # for flat mirror as example
        return s, r * np.sin(phi), Rm - r * np.cos(phi)  # x, y, z

E0 = 2000.
L = 190.
Rm = 5000.
isFlat = False
isParametric = True


def build_beamline(nrays=raycing.nrays):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, dx=0., dz=0., dxprime=5e-4, dzprime=1e-5,
        distE='lines', energies=(E0,), polarization='horizontal')
    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0, 100, 0))

    if isFlat:
        fName = 'Flat'
        beamLine.cylinder = roe.OE(
            beamLine, 'FlatP', [0, 1000, -0.01],
            pitch=3e-3, material=mGold, isParametric=isParametric)
    else:
        fName = 'Cylinder'
        limPhysX = [-5, 5]
        limPhysY = [0, L]
        if isParametric:
            CylinderClass = CylinderP
        else:
            CylinderClass = Cylinder
        beamLine.cylinder = CylinderClass(
            beamLine, fName, [0, 1000, -0.05],
            pitch=3e-3, material=mGold, limPhysX=limPhysX, limPhysY=limPhysY,
            Rm=Rm, isParametric=isParametric)

    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', (0, 2000, 0))
    return beamLine, fName


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamCylinderGlobal, beamCylinderLocalN = \
        beamLine.cylinder.multiple_reflect(beamSource, maxReflections=100)
#      beamLine.cylinder.reflect(beamSource)
    beamFSM2 = beamLine.fsm2.expose(beamCylinderGlobal)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamCylinderGlobal': beamCylinderGlobal,
               'beamCylinderLocalN': beamCylinderLocalN,
               'beamFSM2': beamFSM2}
    return outDict
rr.run_process = run_process


def define_plots(beamLine, fName):
#    fwhmFormatStrE = '%.2f'
    plots = []
    pAdd = 'P' if isParametric else ''

    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xrtp.XYCAxis(r'$x$', r'$\mu$m'),
        yaxis=xrtp.XYCAxis(r'$z$', r'$\mu$m'), title='FSM1_E')
    plot.caxis.fwhmFormatStr = None
#    plot.caxis.limits = [70, 140]
    plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamCylinderLocalN', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm'), aspect='auto',
        caxis=xrtp.XYCAxis('number of reflections', '', bins=32, ppb=8,
                           data=raycing.get_reflection_number), title='local')
    plot.caxis.fwhmFormatStr = None
    plot.xaxis.limits = [-2, 2]
    plot.saveName = ['{0}Local{1}.png'.format(fName, pAdd), ]
    plots.append(plot)

    if isParametric:
        plot = xrtp.XYCPlotWithNumerOfReflections(
            'beamCylinderLocalN', (1,),
            xaxis=xrtp.XYCAxis(r'$s$', 'mm'), aspect='auto',
            yaxis=xrtp.XYCAxis(r'$\phi$', 'mrad'),
            caxis=xrtp.XYCAxis('number of reflections', '', bins=32, ppb=8,
                               data=raycing.get_reflection_number),
            title='local (s, phi)')
#        plot.yaxis.fwhmFormatStr = '%.2f' + r'$ \pi$'
        plot.caxis.fwhmFormatStr = None
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        plot.ax2dHist.xaxis.set_major_formatter(formatter)
        plot.saveName = ['{0}LocalPP.png'.format(fName), ]
        plots.append(plot)

    plot = xrtp.XYCPlotWithNumerOfReflections(
        'beamFSM2', (1, 2),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('number of reflections', '',  bins=32, ppb=8,
                           data=raycing.get_reflection_number),
        title='FSM2_Es')
    plot.xaxis.limits = [-4, 4]
    plot.caxis.fwhmFormatStr = None
    plot.fluxFormatStr = '%.2e'
    plot.saveName = ['{0}Out{1}.png'.format(fName, pAdd), ]
    plots.append(plot)
    return plots


def main():
    beamLine, fName = build_beamline()
    plots = define_plots(beamLine, fName)
    xrtr.run_ray_tracing(plots, repeats=40, updateEvery=1, beamLine=beamLine,
                         processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

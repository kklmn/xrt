# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "08 Mar 2016"
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.screens as rsc

mGold = rm.Material('Au', rho=19.3, kind='grating')

gratingKind = 'y-grating'
if gratingKind == 'x-grating':
    rho, fsm2pos = 3000., 15880.
elif gratingKind == 'y-grating':
    rho, fsm2pos = -100., 25000.
else:
    raise


class Grating(roe.OE):
    def local_g(self, x, y, rho=rho):
        """Must be directed toward the positive oreder direction!"""
        if gratingKind == 'x-grating':
            return rho, 0, 0  # constant line spacing along x
        if gratingKind == 'y-grating':
            return 0, rho, 0


def build_beamline(nrays=raycing.nrays):
    beamLine = raycing.BeamLine(height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0),
        nrays=nrays, dx=0., dz=0., dxprime=2e-4, dzprime=1e-4,
        distE='lines', energies=(75, 100, 125), polarization='horizontal')
#    beamLine.feFixedMask = ra.RectangularAperture(beamLine,
#      'FEFixedMask', 0, 10000,
#      ('left', 'right', 'bottom', 'top'), [-3., 3., -1., 1.])
    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0., 10001., 0.))
    beamLine.grating = Grating(beamLine, 'PlaneGrating', [0., 15000., 0.],
                               pitch=np.radians(10), material=mGold)
    beamLine.grating.order = np.arange(-2, 3)
    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', (0., fsm2pos, 0.))
    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    beamSource = beamLine.sources[0].shine()
#    beamLine.feFixedMask.propagate(beamSource)
    beamFSM1 = beamLine.fsm1.expose(beamSource)
    beamGratingGlobal, beamGratingLocal = beamLine.grating.reflect(beamSource)
    beamFSM2 = beamLine.fsm2.expose(beamGratingGlobal)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1,
               'beamGratingGlobal': beamGratingGlobal,
               'beamGratingLocal': beamGratingLocal,
               'beamFSM2': beamFSM2}
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = None
    plots = []

    plot1 = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'), title='FSM1_E')
    plot1.caxis.fwhmFormatStr = None
    plot1.caxis.limits = [70, 130]
    plots.append(plot1)

    plot2 = xrtp.XYCPlot(
        'beamFSM2', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits='symmetric'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=64, ppb=4, limits=[70, 130]),
        title='FSM2_Es')
    plot2.caxis.fwhmFormatStr = fwhmFormatStrE
#    plot2.fluxFormatStr = '%.2e'
    plot2.saveName = gratingKind + 'E.png'
    plots.append(plot2)

    plot3 = xrtp.XYCPlot(
        'beamFSM2', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits='symmetric'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('diffraction order', '', bins=32, ppb=8),
        title='FSM2_Es')
    plot3.caxis.fwhmFormatStr = None
    plot3.caxis.limits = [-2.1, 2.1]
#    plot2.fluxFormatStr = '%.2e'
    plot3.saveName = gratingKind + 'Order.png'
    plots.append(plot3)
    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

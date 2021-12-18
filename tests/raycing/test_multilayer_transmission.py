# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "17 Dec 2021"
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm

import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

E0 = 398.
theta = np.radians(37.75)
# try with two polarizations:
# polarization = 'hor'
polarization = 'ver'
p = 100.

table = 'Chantler'
# table = 'Henke'
mSc = rm.Material('Sc', rho=2.98, table=table)
mCr = rm.Material('Cr', rho=7.18, table=table)

mL = rm.Multilayer(mSc, 12.8, mCr, 12.8, 200, mSc, substThickness=13.0,
                   geom='transmitted')


def build_beamline(nrays=1e6):
    bl = raycing.BeamLine(azimuth=0, height=0)
    rs.GeometricSource(
        bl, 'GeometricSource', (0, 0, 0),
        nrays=nrays, distx='flat', dx=5, dy=0, dz=0, distzprime='flat',
        dxprime=0, dzprime=0.03,
        distE='lines', energies=(E0,), polarization=polarization)
    bl.plate = roe.OE(bl, 'plate1', (0, p, 0), material=mL,
                      pitch=theta, targetOpenCL='GPU')
    return bl


def run_process(bl):
    beamSource = bl.sources[0].shine()
    beamPlateGlobal, beamPlateLocal = bl.plate.reflect(beamSource)
    outDict = {'beamSource': beamSource,
               'beamPlateGlobal': beamPlateGlobal,
               'beamPlateLocal': beamPlateLocal}
    return outDict
rr.run_process = run_process


def define_plots(bl):
    plots = []
    title = '{0}_{1}'.format(bl.plate.name, polarization)
    plot = xrtp.XYCPlot(
        'beamPlateLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis(r"$z'$", 'deg'),
        ePos=1, title=title)
    plot.baseName = title
    plot.saveName = [plot.baseName + '.png', ]
    plots.append(plot)
    return(plots)


def main():
    bl = build_beamline()
    plots = define_plots(bl)
    xrtr.run_ray_tracing(plots, repeats=1, beamLine=bl)


if __name__ == '__main__':
    main()

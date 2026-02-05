# -*- coding: utf-8 -*-
"""
Infrared edge radiation
~~~~~~~~~~~~~~~~~~~~~~~

The images below are produced by
``examples/withRaycing/01_SynchrotronSources/edge_radiation.py``.

This example calculates an infrared source consisting of two bending magnets
defined by a big table of magnetic field. The magnetic field has four narrow
ranges of rapid field change, so called field edges, where edge radiation is
born. The two inner regions are aligned along the straight section, and their
radiation enters the beamline front end. The outer two edges produce radiation
that is directed to extreme negative and positive angles relative to the
straight section.

+--------------------------+--------------------------+
|      magnetic field      | electron beam trajectory |
+==========================+==========================+
|        |edgefield|       |        |edgetraj|        |
+--------------------------+--------------------------+

.. |edgefield| imagezoom:: _images/edge-field.png
.. |edgetraj| imagezoom:: _images/2bm-traj-xyz.png

The infrared wavelength in this example is 10 µm. The equatorial gap mimics a
3 mm gap in the extracting mirror that reflects the beam sideways or upwards.
The calculated field also containes bending magnet radiation that is about two
orders of magnitude weaker than edge radiation at this energy and therefore is
not visible on the pictures below.

Edge radiation has a peculiar polarization pattern: it is *radially* polarized
[Bosch]_. This property is demonstrated by the polarization selective images
in the table below: the p-polarized image is suppressed in the equatorial
region whereas the s-polarized image is suppressed in the vertical meridional
region. The lowest images, colored by the polarization ellipse angle,
additionally demonstrate the radial polarization property.

This example is an extreme case of a difference between far field and near
field calculation expressions, compare the two columns.

.. [Bosch] *Focusing of infrared edge and synchrotron radiation*,
   NIMA **431** (1999) 320--333

+-------------+---------------------+---------------------+
|             |     far field       |     near field      |
+=============+=====================+=====================+
|    s-pol    |      |farhor|       |      |nearhor|      |
+-------------+---------------------+---------------------+
|    p-pol    |      |farver|       |      |nearver|      |
+-------------+---------------------+---------------------+
| pol ellipse |      |farpsi|       |      |nearpsi|      |
| angle       |                     |                     |
+-------------+---------------------+---------------------+

.. |farhor| imagezoom:: _images/1-far-field-1-hor-pol.png
.. |farver| imagezoom:: _images/1-far-field-2-ver-pol.png
.. |farpsi| imagezoom:: _images/1-far-field-3-polPsi.png
.. |nearhor| imagezoom:: _images/2-near-field-1-hor-pol.png
   :loc: upper-right-corner
.. |nearver| imagezoom:: _images/2-near-field-2-ver-pol.png
   :loc: upper-right-corner
.. |nearpsi| imagezoom:: _images/2-near-field-3-polPsi.png
   :loc: upper-right-corner

"""

__author__ = "Konstantin Klementiev"
__date__ = "19 Jan 2026"

import numpy as np
import sys
from matplotlib import pyplot as plt

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
xrtplot.colorFactor = 1.

# binsx, binsy = 896, 128
# ppbx, ppby = 1, 1  # pixel per bin
# limx, limy = [-84, 84], [-12, 12]
binsx, binsy = 256, 256
ppbx, ppby = 1, 1  # pixel per bin
limx, limy = [-20, 20], [-20, 20]

nrays = 4e6

wlmin, wl0, wlmax = 2, 10, 1000  # in µm: min, w0, max
E0 = rmats.CH / wl0 * 1e-4
Emin = rmats.CH / wlmax * 1e-4
Emax = rmats.CH / wlmin * 1e-4
elim = Emin, Emax

sourcey = 6180
m1y = sourcey + 1324.45

upperBottom = 1.5
upperTop = limy[1]
lowerBottom = -upperTop
lowerTop = -upperBottom
openingVDouble = [
    (limx[0], upperTop), (limx[1], upperTop),
    (limx[1], upperBottom), (limx[0], upperBottom),
    (limx[0], lowerTop), (limx[1], lowerTop),
    (limx[1], lowerBottom), (limx[0], lowerBottom)]


case = '1-far-field'
# case = '2-near-field'
fieldFileName = 'edge_radiation.xlsx'

if 'far-field' in case:
    R0 = None
elif 'near-field' in case:
    R0 = m1y
else:
    raise ValueError('Unknown case')


def build_beamline(nrays=nrays):
    beamLine = raycing.BeamLine()
    beamLine.nrays = nrays

    beamLine.source = rsources.SourceFromField(
        bl=beamLine,
        name=r"bendingMagnet",
        customField=[fieldFileName, dict(sheet_name='Sheet1', skiprows=0)],
        eE=2.75, eI=0.5,
        eEpsilonX=0, eEpsilonZ=0, eEspread=0,
        # eEspread=1.025E-3,
        # eEpsilonX=180, eEpsilonZ=8,  # emittance [nm.rad]
        # eSigmaX=3.9, eSigmaZ=0.043,  # beam size [um]
        nrays=nrays,
        targetOpenCL='auto',
        precisionOpenCL='float64',
        eMin=Emin, eMax=Emax,
        xPrimeMax=[lim/m1y*1e3 for lim in limx],
        zPrimeMax=[lim/m1y*1e3 for lim in limy],
        xPrimeMaxAutoReduce=False,
        zPrimeMaxAutoReduce=False,
        filamentBeam=True,  # important
        uniformRayDensity=True,  # important
        R0=R0,
        center=[0, 0, 0])

    beamLine.slit = rapts.PolygonalAperture(
        beamLine, 'doubleSlit', [0, m1y-upperTop, 0], opening=openingVDouble)

    pitch = np.pi/4
    beamLine.limPhysYM1 = [lim/np.sin(pitch) for lim in limy]
    beamLine.m1 = roes.OE(
        beamLine, 'm1', [0, m1y, 0], pitch=pitch,
        limPhysX=limx, limPhysY=beamLine.limPhysYM1)

    beamLine.screenAboveM1 = rscreens.Screen(
        beamLine, "screenAboveM1", [0, m1y, upperTop], z=[0, -1, 0])

    return beamLine


def run_process(beamLine):
    waveSlit = beamLine.slit.prepare_wave(beamLine.source, beamLine.nrays)
    beamSource = beamLine.source.shine(fixedEnergy=E0, wave=waveSlit)

    m1Global, m1Local = beamLine.m1.reflect(beamSource)
    screenAboveM1Local = beamLine.screenAboveM1.expose(m1Global)

    outDict = {
        'waveSlit': waveSlit, 'm1Local': m1Local,
        'screenAboveM1Local': screenAboveM1Local,
        }
    return outDict

rrun.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot = xrtplot.XYCPlot(
        "waveSlit", aspect='auto',
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx, limits=limx),
        yaxis=xrtplot.XYCAxis("z", bins=binsy, ppb=ppby, limits=limy),
        caxis=xrtplot.XYCAxis(
            "energy", "eV", bins=binsy, ppb=ppby, limits=elim),
        title="total")
    plots.append(plot)
    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.screenX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.screenZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    plotEs = xrtplot.XYCPlot(
        "waveSlit", aspect=1,
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx, limits=limx),
        yaxis=xrtplot.XYCAxis("z", bins=binsy, ppb=ppby, limits=limy),
        caxis=xrtplot.XYCAxis(
            "energy", "eV", bins=binsy, ppb=ppby, limits=elim),
        fluxKind="s", title="hor-pol")
    plots.append(plotEs)

    plotEp = xrtplot.XYCPlot(
        "waveSlit", aspect=1,
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx, limits=limx),
        yaxis=xrtplot.XYCAxis("z", bins=binsy, ppb=ppby, limits=limy),
        caxis=xrtplot.XYCAxis(
            "energy", "eV", bins=binsy, ppb=ppby, limits=elim),
        fluxKind="p", title="ver-pol")
    plots.append(plotEp)

    plot = xrtplot.XYCPlot(
        "waveSlit", aspect=1,
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx, limits=limx),
        yaxis=xrtplot.XYCAxis("z", bins=binsy, ppb=ppby, limits=limy),
        caxis=xrtplot.XYCAxis(
            'angle of polarization ellipse', 'rad', bins=binsy, ppb=ppby,
            data=raycing.get_polarization_psi,
            limits=[-np.pi/2, np.pi/2]),
        title='polPsi')
    plot.ax1dHistE.set_yticks((-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2))
    plot.ax1dHistE.set_yticklabels(
        (r'-$\frac{\pi}{2}$', r'-$\frac{\pi}{4}$',
         '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'))
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        "m1Local", aspect='auto',
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx),
        yaxis=xrtplot.XYCAxis("y", bins=binsy, ppb=ppby),
        caxis=xrtplot.XYCAxis(
            "energy", "eV", bins=binsy, ppb=ppby, limits=elim),
        title="footprint-m1")
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        "screenAboveM1Local", aspect=1,
        xaxis=xrtplot.XYCAxis("x", bins=binsx, ppb=ppbx, limits=limx),
        yaxis=xrtplot.XYCAxis("z", bins=binsy, ppb=ppby, limits=limy),
        caxis=xrtplot.XYCAxis(
            "energy", "eV", bins=binsy, ppb=ppby, limits=elim),
        title="screen-above-m1")
    plots.append(plot)

    for iplot, plot in enumerate(plots):
        plot.caxis.fwhmFormatStr = '%.2f'
        plot.baseName = '{0}-{1}-{2}'.format(case, iplot, plot.title)
        plot.saveName = [plot.baseName + '.png']

    return plots


def get_vorticity():
    beamLine = build_beamline()
    theta = np.linspace(limx[0]/m1y, limx[1]/m1y, binsx)
    psi = np.linspace(limy[0]/m1y, limy[1]/m1y, binsy)
    Is, Ip, OAMs, OAMp, Es, Ep = beamLine.source.intensities_on_mesh(
        [E0], theta, psi, resultKind='vortex')
    fluxIs = np.trapezoid(np.trapezoid(Is, psi), theta)
    fluxIp = np.trapezoid(np.trapezoid(Ip, psi), theta)
    # flux = fluxIs + fluxIp
    lEs = OAMs / fluxIs
    lEp = OAMp / fluxIp
    vEs = np.trapezoid(np.trapezoid(lEs, psi), theta)
    vEp = np.trapezoid(np.trapezoid(lEp, psi), theta)
    print('vorticity: Es = {0}, Ep = {1}'.format(vEs, vEp))


def plot_field():
    from pandas import read_excel
    fname = 'edge_radiation.xlsx'
    data = read_excel(fname).values
    y = data[:, 0]
    Bz = data[:, 2]

    fig1 = plt.figure(figsize=(6., 4))
    ax = fig1.add_axes([0.12, 0.15, 0.55, 0.83])
    ax.set_xlabel('y (m)')
    ax.set_ylabel('$B_z$ (T)')
    ax.plot(y*1e-3, Bz, color='C1')

    ax2 = fig1.add_axes([0.7, 0.15, 0.28, 0.83])
    ax2.set_xlabel('y (m)')
    ax2.plot(y*1e-3, Bz, color='C1')
    ax2.set_xlim(6.175, 6.225)
    ax2.yaxis.set_ticklabels([])

    fig1.savefig("edge-field.png")
    plt.show()


def see_trajectory(source=None):
    if source is None:
        beamLine = build_beamline(nrays=1)
        source = beamLine.source
        source.shine(fixedEnergy=E0)

    traj = source.trajectory
    beta = source.beta

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(projection='3d')
    ax1.set_title("trajectory 3D")
    ax1.plot(traj[0], traj[2]*1e-3, traj[1], label='parametric curve')
    ax1.set_xlabel("x, mm")
    ax1.set_ylabel("y, m")
    ax1.set_zlabel("z, mm")
    ax1.set_box_aspect((5, 10, 5))
    ax1.view_init(23, -48, 0)
    fig1.savefig(case+'-traj-xyz.png')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set_title("trajectory in xy plane")
    ax2.plot(traj[0], traj[2])
    ax2.set_xlabel("x, mm")
    ax2.set_ylabel("y, mm")
    fig2.savefig(case+'-traj-xy.png')

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot()
    ax3.set_title(r"$\beta_x$ vs y")
    ax3.plot(beta[0], traj[-1])
    ax3.set_xlabel(r"$\beta_x$")
    ax3.set_ylabel("y, mm")
    fig3.savefig(case+'-traj-betax.png')

    plt.show()


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtrun.run_ray_tracing(plots=plots, repeats=1, beamLine=beamLine)


if __name__ == '__main__':
    # plot_field()
    # see_trajectory()
    # get_vorticity()
    main()  # needs a decent GPU
    print("Done")

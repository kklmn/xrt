# -*- coding: utf-8 -*-
u"""
.. start
.. _tests_mosaic:

Reflectivity of mosaic crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These tests implement the diffraction setup from [SanchezDelRioMosaic]_, Fig.
4. In our case, the source has a finite energy band to demonstrate the energy
dispersion effect in parafocusing (cf. Figs. 5 and 6 ibid).

+----------+----------+
|  |mosA|  |  |mosB|  |
+----------+----------+

.. |mosA| imagezoom:: _images/MosaicGraphite002-screenA.*
   :align: center
.. |mosB| imagezoom:: _images/MosaicGraphite002-screenB.*
   :align: center

The penetration depth distribution should be compared with Fig 7 ibid.

.. imagezoom:: _images/MosaicGraphite002-Z.*
   :align: center

The reflectivity curves are compared with those by XCrystal/XOP [XOP]_. The
small differences are primarily due to small differences in the tabulations of
the scattering factors. We use the one by Chantler [Chantler]_.

.. imagezoom:: _images/MosaicGraphite002-ReflectivityS.*
   :align: center

.. end
"""
__author__ = "Konstantin Klementiev"
__date__ = "2018/08/03"
import sys
import os
import numpy as np
import pickle

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

hkl = (0, 0, 2)
# hkl = (0, 0, 6)
shlk = ''.join(['{0}'.format(i) for i in hkl])
kw = dict(hkl=hkl, a=2.456, c=6.696, gamma=120,
          atoms=[6]*4, atomsXYZ=[[0., 0., 0.], [0., 0., 0.5],
                                 [1./3, 2./3, 0.], [2./3, 1./3, 0.5]],
          table='Chantler')

xtalPerfect = rmats.CrystalFromCell('perfect', **kw)

mosaicityFWHMdeg = 0.4  # = 0.4° fwhm
mosaicityFWHM = np.deg2rad(mosaicityFWHMdeg)
mosaicity = mosaicityFWHM/2.355
xtalMosaic = rmats.CrystalFromCell('mosaic', **kw, mosaicity=mosaicity)

# Ec = 3000.
Ec = 8000.
# Ec = 17000.

dE = 2e-4*Ec
nrays = 1e6

p = 1e4
# screenPoss = np.array([1.5, 2, 2.5, 3]) * p
screenPoss = np.array([2, 3]) * p

xBins = 160
zBins = 100


def build_beamline():
    beamLine = raycing.BeamLine()
    bragg = xtalPerfect.get_Bragg_angle(Ec) - xtalPerfect.get_dtheta(Ec)
    print('theta={0}deg'.format(np.degrees(bragg)))
    print('dtheta', xtalPerfect.get_dtheta(Ec), 'mosaicity',
          mosaicityFWHM/2.355)
    beamLine.bragg = bragg
    print("mosaic divergence hor", 2*mosaicityFWHM*np.sin(bragg))

    beamLine.source = rsources.GeometricSource(
        beamLine, center=[0, 0, 0],
        distx=None, disty=None, distz=None,

        # for reflectivity calculations select larger dzprime:
        # distxprime=None, distzprime='flat', dzprime=0.022,
        # distE='lines', energies=[Ec], polarization='h',
        # for getting diffracted images select this one:
        distxprime='flat', distzprime='flat', dxprime=1e-3, dzprime=2e-4,
        distE='flat', energies=(Ec-dE/2, Ec+dE/2), polarization='h',

        nrays=nrays,
        pitch=-bragg)

    beamLine.hX = p * np.tan(bragg)
    kwOE = dict(bl=beamLine, center=[0, p, -beamLine.hX], limPhysY=[-1e4, 1e4])
#    beamLine.xtalP = roes.OE(**kwOE, material=xtalPerfect)
    beamLine.xtalM = roes.OE(**kwOE, material=xtalMosaic)

    beamLine.screen = rscreens.Screen(beamLine, center=[0, 2*p, 0])

    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine()
#    beamXtalGlobal, beamXtalLocal = beamLine.xtalP.reflect(beamSource)
    beamXtalGlobal, beamXtalLocal = beamLine.xtalM.reflect(beamSource)
    raycing.rotate_beam(beamSource, pitch=beamLine.bragg)
    beamSource.dtheta = beamSource.c/beamSource.b
    beamXtalLocal.dtheta = beamSource.dtheta
    outD = {'beamSource': beamSource,
            'beamXtalGlobal': beamXtalGlobal, 'beamXtalLocal': beamXtalLocal}
    for iscr, screenPos in enumerate(screenPoss):
        hs = beamLine.hX * (-1 + (screenPos-p)/p)
        beamLine.screen.center = [0, screenPos, hs]
        beamScreen = beamLine.screen.expose(beamXtalGlobal)
        outD['beamScreen'+'{0:d}'.format(iscr)] = beamScreen
    return outD
rrun.run_process = run_process


def get_dtheta(beam):
    return beam.dtheta


def define_plots(beamLine):
    plots = []

    plot = xrtplot.XYCPlot(
        beam=r"beamSource", aspect='auto',
        xaxis=xrtplot.XYCAxis(r"x'", 'mrad'),
        yaxis=xrtplot.XYCAxis(r"z'", 'mrad', data=get_dtheta))
    plot.saveName = ["0-source.png"]
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        beam=r"beamXtalLocal", aspect='auto',
        xaxis=xrtplot.XYCAxis(r"x'", 'mrad'),
        yaxis=xrtplot.XYCAxis(r"z'", 'mrad', data=get_dtheta))
    plot.saveName = ["1-localXtal.png"]
    plots.append(plot)

    plot = xrtplot.XYCPlot(
        beam=r"beamXtalLocal", aspect='auto',
        xaxis=xrtplot.XYCAxis(r"y", 'mm'),
        yaxis=xrtplot.XYCAxis(r"z", 'µm', limits=[-400, 10]))
    plot.saveName = ["1-localXtalDepth.png"]
    plots.append(plot)

    for iscr, screenPos in enumerate(screenPoss):
        plot = xrtplot.XYCPlot(
            beam='beamScreen'+'{0:d}'.format(iscr), aspect='auto',
            xaxis=xrtplot.XYCAxis(r"x", limits=[-100, 100], bins=xBins),
            yaxis=xrtplot.XYCAxis(r"z", limits=[-3.6, 3.6], bins=zBins))
        plot.textPanel = plot.fig.text(
            0.88, 0.9, 'p:q = 1:{0:.0f}'.format((screenPos-p)/p),
            transform=plot.fig.transFigure, size=14, color='r',
            ha='center')
        plot.saveName = ["2-screen{0:d}.png".format(iscr+1)]
        plots.append(plot)

    for plot in plots:
        plot.caxis.offset = Ec
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'

    return plots


def afterScript(plots, beamLine):
    sax = plots[0].yaxis
    rax = plots[1].yaxis
    dax = plots[2].yaxis
    fn = 'reflMosaicGraphite{0}-{1:02.0f}keV-{2}deg.pickle'.format(
        shlk, Ec*1e-3, mosaicityFWHMdeg)
    with open(fn, 'wb') as f:
        pickle.dump([sax.binCenters, sax.total1D, rax.total1D,
                     dax.binCenters, dax.total1D, plots[0].nRaysGood], f)
    print("Saved")


def calc_refl(E):
    delta = np.linspace(-0.011, 0.011, 221)
    thetaB = xtalMosaic.get_Bragg_angle(E) - xtalMosaic.get_dtheta(E)
    beamInDotNormal = -np.sin(thetaB + delta)
    rs, rp = xtalMosaic.get_amplitude_mosaic(E, beamInDotNormal)
    return delta, rs, rp


def plot_reflectivity(fromRayTracing=False):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerBase

    class NLineObjectsHandler(HandlerBase):
        def create_artists(self, legend, orig_handles,
                           x0, y0, width, height, fontsize, trans):
            return [plt.Line2D([x0, x0+width], [p*height, p*height],
                               linestyle=handle.get_linestyle(),
                               color=handle.get_color()) for handle, p in
                    zip(orig_handles, self.get_line_vpos(len(orig_handles)))]

        def get_line_vpos(self, n):
            if n == 1:
                pos = [0.5]
            elif n == 2:
                pos = [0.7, 0.3]
            elif n == 3:
                pos = [0.9, 0.5, 0.1]
            elif n == 4:
                pos = [0.95, 0.65, 0.35, 0.05]
            else:
                raise ValueError('too many lines')
            return pos

    ms = "{0}°".format(mosaicityFWHMdeg)

    fig = plt.figure(1)
    fig.suptitle("\nRocking curves for graphite "
                 "({0}) with mosaicity FWHM ".format(shlk)+ms, fontsize=12)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\theta-\theta_B$ (mrad)')
    ax.set_ylabel('reflectivity s')

    if fromRayTracing:
        fig2 = plt.figure(2)
        fig2.suptitle("\nDepth distribution for graphite "
                      "({0}) with mosaicity FWHM ".format(shlk)+ms,
                      fontsize=12)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel(r'z (µm)')
        ax2.set_ylabel('relative weight (a.u.)')

    lxrt, lxc, labels = [], [], []
    for E0, color in zip([3, 8, 17], ['C0', 'C1', 'C2']):
        if fromRayTracing:
            fn = 'reflMosaicGraphite{0}-{1:02.0f}keV-{2}deg.pickle'.format(
                shlk, E0, mosaicityFWHMdeg)
            with open(fn, 'rb') as f:
                delta, source, refl, z, nz, raysGood = pickle.load(f)
            source[source == 0] = 1.
            refl /= source
        else:
            delta, rs, rp = calc_refl(E0*1e3)
            delta *= 1e3
            refl = rs**2
        label = '{0:.0f} keV'.format(E0)
        labels.append(label)
        l, = ax.plot(delta, refl, '--', color=color, lw=1.5)
        lxrt.append(l)

        fnx = 'graphite{0}_mosaic{1:02.0f}_{2:02.0f}kev.xc.gz'.format(
            shlk, mosaicityFWHMdeg*10, E0)
        path = os.path.join('', 'XOP-RockingCurves', fnx)
        x, reflXOP = np.loadtxt(path, unpack=True)
        l, = ax.plot(x*1e3, reflXOP, '-', color=color, lw=1.5)
        lxc.append(l)

        if fromRayTracing:
            ax2.plot(-z, nz/nz.sum(), '-', color=color, lw=1.5, label=label)

    ax.set_xlim([delta.min(), delta.max()])
    ax.set_ylim([0, None])

    leg1 = ax.legend([(l1, l2) for l1, l2 in zip(lxrt, lxc)], labels,
                     handler_map={tuple: NLineObjectsHandler()},
                     loc='upper right')
    leg2 = ax.legend([tuple(lxrt), tuple(lxc)],
                     [r'xrt', r'XCrystal/XOP'],
                     handler_map={tuple: NLineObjectsHandler()},
                     loc='upper left')
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    if fromRayTracing:
        ax2.set_xlim([0, abs(z).max()])
        ax2.set_ylim([0, None])
        ax2.legend(loc='upper right')

    fname = 'MosaicGraphite{0}-ReflectivityS'.format(shlk)
    fig.savefig(fname + '.png')
    if fromRayTracing:
        fname2 = 'MosaicGraphite{0}-Z'.format(shlk)
        fig2.savefig(fname2 + '.png')

    plt.show()


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtrun.run_ray_tracing(plots=plots, repeats=1, beamLine=beamLine,
                           afterScript=afterScript,
                           afterScriptArgs=[plots, beamLine])


if __name__ == '__main__':
    # main()
    plot_reflectivity()

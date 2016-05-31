# -*- coding: utf-8 -*-
r"""
Tests of Kirchhoff integral with Gaussian and Laguerre-Gaussian beams
---------------------------------------------------------------------

Find the test module test_wave.py, as well as several other tests for raycing
backend, in ``\tests\raycing``.

Gaussian beam
~~~~~~~~~~~~~

Gaussian beam at the waist (at :math:`z=0`) is defined as

.. math::
    u_0 = \sqrt{\frac2\pi}\frac1{w_0}\exp\left(-\frac{r^2}{w_0^2}\right),

where the pre-exponent factor provides normalization:
:math:`\int{|u|^2dS} = 2\pi\int_0^\infty {|u|^2 rdr} = 1`.

Define

:math:`z_R \buildrel \text{def}\over = \frac{\pi w_0^2}\lambda`,

:math:`R \buildrel \text{def}\over = z\left(1+\frac{z_R^2}{z^2}\right)`,

:math:`w \buildrel \text{def}\over = w_0\sqrt{1+\frac{z^2}{z_R^2}}`.

Gaussian beam at arbitrary z is expressed as:

.. math::
    u(z) = \sqrt{\frac2\pi}\frac1{w_0}\frac1{1-i\frac{z}{z_R}}\exp
    \left(-\frac{r^2}{w^2}-\frac{i\pi r^2}{\lambda R}\right),

where the exponent can also be rewritten as:  :math:`\exp\left(-\frac{r^2}
{w_0^2(1-i\frac{z}{z_R})}\right)`. The pre-exponent factor can also be factored
as :math:`\frac1{1-i\frac{z}{z_R}} = \frac{w_0}w\exp(i\psi)` with :math:`\psi=
\arctan{\frac{z}{z_R}}=\frac{1}{2i}\log{\frac{1+i\frac{z}{z_R}}{1-i\frac{z}
{z_R}}}`.

:math:`U = u(z)\exp\left(-i(kz-\omega t)\right)` satisfies the wave equation

.. math::
    \nabla^2U = \frac{1}{c^2}\frac{\partial^2U}{\partial t^2}.

:math:`u(z)` can also be obtained by integrating the Gaussian beam waist in a
diffraction integral (in our implementation, the Kirchhoff integral):

.. math::
    u(x, y, z) = -\frac{ik}{4\pi} \iint_{-\infty}^\infty u_0(x', y')
    K(x, y, x', y') dx'dy'.

The table below compares Kirchhoff diffraction integrals of a Gaussian waist
with analytical solutions. The coloring is by wave phase. Notice equal shape
and (almost) equal total flux. The Gaussian waist was calculated as
GeometricSource with :math:`w_0` = 15 µm:

+------------------------+
|  Gaussian waist (z=0)  |
+========================+
|        |g00m|          |
+------------------------+

.. |g00m| image:: _images/Gauss-0-beamSource.png
   :scale: 50 %

+-------+--------------+-----------------------+
|       |  analytical  | Kirchhoff diffraction |
+=======+==============+=======================+
| z=5m  |    |g05m|    |         |k05m|        |
+-------+--------------+-----------------------+
| z=10m |    |g10m|    |         |k10m|        |
+-------+--------------+-----------------------+
| z=20m |    |g20m|    |         |k20m|        |
+-------+--------------+-----------------------+
| z=40m |    |g40m|    |         |k40m|        |
+-------+--------------+-----------------------+
| z=80m |    |g80m|    |         |k80m|        |
+-------+--------------+-----------------------+

.. |g05m| image:: _images/Gauss-1-beamFSMg-at05m.png
   :scale: 50 %
.. |k05m| image:: _images/Gauss-1-beamFSMk-at05m.png
   :scale: 50 %
.. |g10m| image:: _images/Gauss-2-beamFSMg-at10m.png
   :scale: 50 %
.. |k10m| image:: _images/Gauss-2-beamFSMk-at10m.png
   :scale: 50 %
.. |g20m| image:: _images/Gauss-3-beamFSMg-at20m.png
   :scale: 50 %
.. |k20m| image:: _images/Gauss-3-beamFSMk-at20m.png
   :scale: 50 %
.. |g40m| image:: _images/Gauss-4-beamFSMg-at40m.png
   :scale: 50 %
.. |k40m| image:: _images/Gauss-4-beamFSMk-at40m.png
   :scale: 50 %
.. |g80m| image:: _images/Gauss-5-beamFSMg-at80m.png
   :scale: 50 %
.. |k80m| image:: _images/Gauss-5-beamFSMk-at80m.png
   :scale: 50 %

Laguerre-Gaussian beam
~~~~~~~~~~~~~~~~~~~~~~

Vortex beams are given by Laguerre-Gaussian modes:

.. math::
    u^l_p = u(z) \sqrt{\frac{p!}{(p+|l|)!}}\left(\frac{r\sqrt{2}}{w}\right)
    ^{|l|}L^{|l|}_p\left(\frac{2r^2}{w^2}\right)\exp\left(i(|l|+2p)\psi
    \right)\exp(il\phi).

The flux is again normalized to unity.

The table below compares Kirchhoff diffraction integrals of a Laguerre-Gaussian
waist with analytical solutions. The coloring is by wave phase. Notice equal
shape and (almost) equal total flux. The Laguerre-Gaussian waist was calculated
as GeometricSource with :math:`w_0` = 15 µm, :math:`l` = 1 and :math:`p` = 1:

+---------------------------------+
|  Laguerre-Gaussian waist (z=0)  |
+=================================+
|            |lg00m|              |
+---------------------------------+

.. |lg00m| image:: _images/Laguerre-Gauss-0-beamSource.png
   :scale: 50 %

+-------+--------------+-----------------------+
|       |  analytical  | Kirchhoff diffraction |
+=======+==============+=======================+
| z=5m  |    |lg05m|   |        |lk05m|        |
+-------+--------------+-----------------------+
| z=10m |    |lg10m|   |        |lk10m|        |
+-------+--------------+-----------------------+
| z=20m |    |lg20m|   |        |lk20m|        |
+-------+--------------+-----------------------+
| z=40m |    |lg40m|   |        |lk40m|        |
+-------+--------------+-----------------------+
| z=80m |    |lg80m|   |        |lk80m|        |
+-------+--------------+-----------------------+

.. |lg05m| image:: _images/Laguerre-Gauss-1-beamFSMg-at05m.png
   :scale: 50 %
.. |lk05m| image:: _images/Laguerre-Gauss-1-beamFSMk-at05m.png
   :scale: 50 %
.. |lg10m| image:: _images/Laguerre-Gauss-2-beamFSMg-at10m.png
   :scale: 50 %
.. |lk10m| image:: _images/Laguerre-Gauss-2-beamFSMk-at10m.png
   :scale: 50 %
.. |lg20m| image:: _images/Laguerre-Gauss-3-beamFSMg-at20m.png
   :scale: 50 %
.. |lk20m| image:: _images/Laguerre-Gauss-3-beamFSMk-at20m.png
   :scale: 50 %
.. |lg40m| image:: _images/Laguerre-Gauss-4-beamFSMg-at40m.png
   :scale: 50 %
.. |lk40m| image:: _images/Laguerre-Gauss-4-beamFSMk-at40m.png
   :scale: 50 %
.. |lg80m| image:: _images/Laguerre-Gauss-5-beamFSMg-at80m.png
   :scale: 50 %
.. |lk80m| image:: _images/Laguerre-Gauss-5-beamFSMk-at80m.png
   :scale: 50 %

"""
__author__ = "Konstantin Klementiev"
__date__ = "28 May 2016"
import sys
sys.path.append(r"c:\Ray-tracing")
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.waves as rw

import xrt.plotter as xrtp
import xrt.runner as xrtr

prefix = 'Laguerre-Gauss-'

nrays = 1e6
E0 = 9000.  # eV
w0 = 15e-3  # mm, waist size of the amplitude (not of intensity!)
lVortex = 1
pVortex = 1
maxFactor = 4.  # factor that determines the screen limits as ±w*maxFactor
maxFactor *= (abs(lVortex)+pVortex+1)**0.25
uniformRayDensity = True
ps = np.array([0.5, 1, 2, 4, 8]) * 10000.


def build_beamline():
    beamLine = raycing.BeamLine(height=0)

    sig = w0 / 2  # gaussian beam is I~exp(-2r²/w²) but 'normal' I~exp(-r²/2σ²)
    beamLine.source = rs.GeometricSource(
        beamLine, 'Gaussian', nrays=nrays,
        uniformRayDensity=uniformRayDensity, vortex=(lVortex, pVortex),
        distx='normal', dx=(sig, sig*maxFactor),
        distz='normal', dz=(sig, sig*maxFactor),
        distxprime=None, distzprime=None, energies=(E0,))

    beamLine.fsmFar = rsc.Screen(beamLine, 'FSM', [0, 0, 0])
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine()
    beamSource.Es /= nrays**0.5
    beamSource.Jss /= nrays
    outDict = {'beamSource': beamSource}
    for ip, (p, (x, z)) in enumerate(zip(ps, beamLine.fsmXZmeshes)):
        beamLine.fsmFar.center[1] = p
        waveOnFSMg = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        phi = np.arctan2(waveOnFSMg.z, waveOnFSMg.x)
        gb = rs.laguerre_gaussian_beam(
            waveOnFSMg.x**2+waveOnFSMg.z**2, phi, p, w0, E0,
            lVortex, pVortex)[0]
        dxdz = (x[1]-x[0]) * (z[1]-z[0])
        waveOnFSMg.Es[:] = gb * dxdz**0.5
        waveOnFSMg.Jss[:] = np.abs(gb)**2 * dxdz
        outDict['beamFSMg{0}'.format(ip)] = waveOnFSMg

        wrepeats = 1
        waveOnFSMk = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        for r in range(wrepeats):
            rw.diffract(beamSource, waveOnFSMk)
            if wrepeats > 1:
                print('wave repeats: {0} of {1} done'.format(r+1, wrepeats))
        outDict['beamFSMk{0}'.format(ip)] = waveOnFSMk
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []

    plot = xrtp.XYCPlot(
        'beamSource', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
        yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
        caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                           bins=256, ppb=1))
    lim = w0 / 2 * maxFactor * 1e3
    plot.xaxis.limits = [-lim, lim]
    plot.yaxis.limits = [-lim, lim]
    plot.saveName = prefix + '0-beamSource.png'
    plots.append(plot)

    beamLine.fsmXZmeshes = []
    for ip, p in enumerate(ps):
        lim = rs.laguerre_gaussian_beam(
            0, 0, p, w0, E0, lVortex, pVortex)[1] / 2 * maxFactor * 1e3

        plot = xrtp.XYCPlot(
            'beamFSMg{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=256, ppb=1))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.title = '{0}{1}-beamFSMg-at{2:02.0f}m'.format(
            prefix, ip+1, p*1e-3)
        plot.saveName = plot.title + '.png'
        plots.append(plot)

        plot = xrtp.XYCPlot(
            'beamFSMk{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm'),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm'),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=256, ppb=1))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.title = '{0}{1}-beamFSMk-at{2:02.0f}m'.format(
            prefix, ip+1, p*1e-3)
        plot.saveName = plot.title + '.png'
        plots.append(plot)

        ax = plot.xaxis
        edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
        xCenters = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
        ax = plot.yaxis
        edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
        zCenters = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
        beamLine.fsmXZmeshes.append([xCenters, zCenters])

    for plot in plots:
        plot.caxis.limits = [-np.pi, np.pi]
        plot.caxis.fwhmFormatStr = None
        plot.ax1dHistE.set_yticks([l*np.pi for l in (-1, -0.5, 0, 0.5, 1)])
        plot.ax1dHistE.set_yticklabels(
            (r'$-\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'))

    return plots


def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots, repeats=1, updateEvery=1, beamLine=beamLine, processes=1)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
r"""
.. _test-Laguerre-Gaussian:

Tests of Kirchhoff integral with various Gaussian beams
-------------------------------------------------------

Find the test module laguerre_gaussian_beam.py, as well as several other tests
for raycing backend, in ``\tests\raycing``.

.. note::

    In this section we consider z-axis to be directed along the beamline in
    order to be compatible with the formulas in their standard notations.
    Elsewhere in xrt z-axis looks upwards.


Gaussian beam
~~~~~~~~~~~~~

Gaussian beam at the waist (at :math:`z=0`) is defined as

.. math::
    u_0(r) = \sqrt{\frac2\pi}\frac1{w_0}\exp\left(-\frac{r^2}{w_0^2}\right),

where the pre-exponent factor provides normalization:
:math:`\int{|u|^2dS} = 2\pi\int_0^\infty {|u|^2 rdr} = 1`.

Define

:math:`z_R \buildrel \text{def}\over = \frac{\pi w_0^2}\lambda`,

:math:`R \buildrel \text{def}\over = z\left(1+\frac{z_R^2}{z^2}\right)`,

:math:`w \buildrel \text{def}\over = w_0\sqrt{1+\frac{z^2}{z_R^2}}`.

Gaussian beam at arbitrary z is expressed as:

.. math::
    u(r, z) = \sqrt{\frac2\pi}\frac1{w_0}\frac1{1-i\frac{z}{z_R}}\exp
    \left(-\frac{r^2}{w^2}-\frac{i\pi r^2}{\lambda R}\right),

where the exponent can also be rewritten as:  :math:`\exp\left(-\frac{r^2}
{w_0^2(1-i\frac{z}{z_R})}\right)`. The pre-exponent factor can also be factored
as :math:`\frac1{1-i\frac{z}{z_R}} = \frac{w_0}w\exp(i\psi)` with :math:`\psi=
\arctan{\frac{z}{z_R}}=\frac{1}{2i}\log{\frac{1+i\frac{z}{z_R}}{1-i\frac{z}
{z_R}}}`.

:math:`U = u(r, z)\exp\left(-i(kz-\omega t)\right)` satisfies the wave equation

.. math::
    \nabla^2U = \frac{1}{c^2}\frac{\partial^2U}{\partial t^2}.

:math:`u` can also be obtained by integrating the Gaussian beam waist in a
diffraction integral (in our implementation, the Kirchhoff integral):

.. math::
    u(x, y, z) = -\frac{ik}{4\pi} \iint_{-\infty}^\infty u_0(x', y')
    K(x, y, x', y') dx'dy'.

The table below compares Kirchhoff diffraction integrals of a Gaussian waist
with analytical solutions. The coloring is by wave phase. Notice equal shape
and unity total flux. The Gaussian waist was calculated as GaussianBeam with
:math:`w_0` = 15 µm:

+------------------------+
|  Gaussian waist (z=0)  |
+========================+
|        |g00m|          |
+------------------------+

.. |g00m| imagezoom:: _images/Gauss-0-beamFSMg-at00m.png

.. note::
    The resulting unity flux is *not* obtained by an ad hoc normalization after
    the diffraction. This flux is obtained from the diffraction field as is,
    which demonstrates the correctness of the field amplitude in our
    implementation of the Kirchhoff integral.

+-------+--------------------------+---------------------------------+
|       | analytical Gaussian beam | numerical Kirchhoff diffraction |
+=======+==========================+=================================+
| z=5m  |          |g05m|          |              |k05m|             |
+-------+--------------------------+---------------------------------+
| z=10m |          |g10m|          |              |k10m|             |
+-------+--------------------------+---------------------------------+
| z=20m |          |g20m|          |              |k20m|             |
+-------+--------------------------+---------------------------------+
| z=40m |          |g40m|          |              |k40m|             |
+-------+--------------------------+---------------------------------+
| z=80m |          |g80m|          |              |k80m|             |
+-------+--------------------------+---------------------------------+

.. |g05m| imagezoom:: _images/Gauss-1-beamFSMg-at05m.png
.. |k05m| imagezoom:: _images/Gauss-1-beamFSMk-at05m.png
   :loc: upper-right-corner
.. |g10m| imagezoom:: _images/Gauss-2-beamFSMg-at10m.png
.. |k10m| imagezoom:: _images/Gauss-2-beamFSMk-at10m.png
   :loc: upper-right-corner
.. |g20m| imagezoom:: _images/Gauss-3-beamFSMg-at20m.png
.. |k20m| imagezoom:: _images/Gauss-3-beamFSMk-at20m.png
   :loc: upper-right-corner
.. |g40m| imagezoom:: _images/Gauss-4-beamFSMg-at40m.png
.. |k40m| imagezoom:: _images/Gauss-4-beamFSMk-at40m.png
   :loc: upper-right-corner
.. |g80m| imagezoom:: _images/Gauss-5-beamFSMg-at80m.png
.. |k80m| imagezoom:: _images/Gauss-5-beamFSMk-at80m.png
   :loc: upper-right-corner

Laguerre-Gaussian beam
~~~~~~~~~~~~~~~~~~~~~~

Vortex beams are given by Laguerre-Gaussian modes:

.. math::
    u^l_p(r, \phi, z) = u(r, z) \sqrt{\frac{p!}{(p+|l|)!}}
    \left(\frac{r\sqrt{2}}{w}\right)^{|l|}
    L^{|l|}_p\left(\frac{2r^2}{w^2}\right)
    \exp\left(i(|l|+2p)\psi\right)\exp(il\phi).

The flux is again normalized to unity.

The table below compares Kirchhoff diffraction integrals of a Laguerre-Gaussian
waist with analytical solutions. The coloring is by wave phase. Notice equal
shape and unity total flux. The Laguerre-Gaussian waist was calculated as
LaguerreGaussianBeam with :math:`w_0` = 15 µm, :math:`l` = 1 and :math:`p` = 1:

+---------------------------------+
|  Laguerre-Gaussian waist (z=0)  |
+=================================+
|            |lg00m|              |
+---------------------------------+

.. |lg00m| imagezoom:: _images/Laguerre-Gauss-0-beamFSMg-at00m.png

.. note::
    The resulting unity flux is *not* obtained by an ad hoc normalization after
    the diffraction. This flux is obtained from the diffraction field as is,
    which demonstrates the correctness of the field amplitude in our
    implementation of the Kirchhoff integral.

+-------+-----------------------------------+---------------------------------+
|       | analytical Laguerre-Gaussian beam | numerical Kirchhoff diffraction |
+=======+===================================+=================================+
| z=5m  |              |lg05m|              |             |lk05m|             |
+-------+-----------------------------------+---------------------------------+
| z=10m |              |lg10m|              |             |lk10m|             |
+-------+-----------------------------------+---------------------------------+
| z=20m |              |lg20m|              |             |lk20m|             |
+-------+-----------------------------------+---------------------------------+
| z=40m |              |lg40m|              |             |lk40m|             |
+-------+-----------------------------------+---------------------------------+
| z=80m |              |lg80m|              |             |lk80m|             |
+-------+-----------------------------------+---------------------------------+

.. |lg05m| imagezoom:: _images/Laguerre-Gauss-1-beamFSMg-at05m.png
.. |lk05m| imagezoom:: _images/Laguerre-Gauss-1-beamFSMk-at05m.png
   :loc: upper-right-corner
.. |lg10m| imagezoom:: _images/Laguerre-Gauss-2-beamFSMg-at10m.png
.. |lk10m| imagezoom:: _images/Laguerre-Gauss-2-beamFSMk-at10m.png
   :loc: upper-right-corner
.. |lg20m| imagezoom:: _images/Laguerre-Gauss-3-beamFSMg-at20m.png
.. |lk20m| imagezoom:: _images/Laguerre-Gauss-3-beamFSMk-at20m.png
   :loc: upper-right-corner
.. |lg40m| imagezoom:: _images/Laguerre-Gauss-4-beamFSMg-at40m.png
.. |lk40m| imagezoom:: _images/Laguerre-Gauss-4-beamFSMk-at40m.png
   :loc: upper-right-corner
.. |lg80m| imagezoom:: _images/Laguerre-Gauss-5-beamFSMg-at80m.png
   :loc: lower-left-corner
.. |lk80m| imagezoom:: _images/Laguerre-Gauss-5-beamFSMk-at80m.png
   :loc: lower-right-corner

Hermite-Gaussian beam
~~~~~~~~~~~~~~~~~~~~~

Higher order modes in rectangular coordinates are given by Hermite-Gaussian
modes:

.. math::
    u_{mn}(x, y, z) = u(r, z) \frac{1}{\sqrt{2^{m+n}m!n!}}
    H_m\left(\frac{\sqrt2x}{w}\right) H_n\left(\frac{\sqrt2y}{w}\right)
    \exp\left(i(m+n)\psi\right),

where :math:`r^2 = x^2 + y^2`. The flux is again normalized to unity.

The table below compares Kirchhoff diffraction integrals of a Hermite-Gaussian
waist with analytical solutions. The coloring is by wave phase. Notice equal
shape and unity total flux. The Hermite-Gaussian waist was calculated as
HermiteGaussianBeam with :math:`w_0` = 15 µm, :math:`m` = 3 and :math:`n` = 2:

+---------------------------------+
|  Hermite-Gaussian waist (z=0)   |
+=================================+
|            |hg00m|              |
+---------------------------------+

.. |hg00m| imagezoom:: _images/Hermite-Gauss-0-beamFSMg-at00m.png

.. note::
    The resulting unity flux is *not* obtained by an ad hoc normalization after
    the diffraction. This flux is obtained from the diffraction field as is,
    which demonstrates the correctness of the field amplitude in our
    implementation of the Kirchhoff integral.

+-------+-----------------------------------+---------------------------------+
|       | analytical Hermite-Gaussian beam  | numerical Kirchhoff diffraction |
+=======+===================================+=================================+
| z=5m  |              |hg05m|              |             |hk05m|             |
+-------+-----------------------------------+---------------------------------+
| z=10m |              |hg10m|              |             |hk10m|             |
+-------+-----------------------------------+---------------------------------+
| z=20m |              |hg20m|              |             |hk20m|             |
+-------+-----------------------------------+---------------------------------+
| z=40m |              |hg40m|              |             |hk40m|             |
+-------+-----------------------------------+---------------------------------+
| z=80m |              |hg80m|              |             |hk80m|             |
+-------+-----------------------------------+---------------------------------+

.. |hg05m| imagezoom:: _images/Hermite-Gauss-1-beamFSMg-at05m.png
.. |hk05m| imagezoom:: _images/Hermite-Gauss-1-beamFSMk-at05m.png
   :loc: upper-right-corner
.. |hg10m| imagezoom:: _images/Hermite-Gauss-2-beamFSMg-at10m.png
.. |hk10m| imagezoom:: _images/Hermite-Gauss-2-beamFSMk-at10m.png
   :loc: upper-right-corner
.. |hg20m| imagezoom:: _images/Hermite-Gauss-3-beamFSMg-at20m.png
.. |hk20m| imagezoom:: _images/Hermite-Gauss-3-beamFSMk-at20m.png
   :loc: upper-right-corner
.. |hg40m| imagezoom:: _images/Hermite-Gauss-4-beamFSMg-at40m.png
.. |hk40m| imagezoom:: _images/Hermite-Gauss-4-beamFSMk-at40m.png
   :loc: upper-right-corner
.. |hg80m| imagezoom:: _images/Hermite-Gauss-5-beamFSMg-at80m.png
   :loc: lower-left-corner
.. |hk80m| imagezoom:: _images/Hermite-Gauss-5-beamFSMk-at80m.png
   :loc: lower-right-corner


"""
__author__ = "Konstantin Klementiev"
__date__ = "28 May 2016"
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.waves as rw

import xrt.plotter as xrtp
xrtp.colorFactor = 1.
import xrt.runner as xrtr

case = 2
wantKirchhoff = True

if case == 0:
    prefix = 'Gauss'
    maxFactor = 2.  # factor that determines the screen limits as ±w*maxFactor
elif case == 1:
    prefix = 'Laguerre-Gauss'
    lVortex, pVortex = 1, 1
    maxFactor = 2*(abs(lVortex)+pVortex+1)**0.25
elif case == 2:
    prefix = 'Hermite-Gauss'
    m, n = 3, 2
    maxFactor = 2 * (m+n)**0.25
else:
    raise ValueError("unknown case")

E0 = 9000.  # eV
w0 = 15e-3  # mm, waist size of the amplitude (not of intensity!)

# screen positions:
if True:  # short test
    ps = np.array([0, 0.5, 1, 2, 4, 8]) * 10000.
else:  # long test
    ps = np.array(list(range(10)) + list(range(1, 11)) +
                  list(range(20, 101, 10))) * 1000.
    ps[0:10] /= 10.
#print(ps)

bins, ppb = 256, 1


def build_beamline():
    beamLine = raycing.BeamLine(height=0)
    if case == 0:
        beamLine.source = rs.GaussianBeam(
            beamLine, prefix, w0=w0, energies=(E0,))
    elif case == 1:
        beamLine.source = rs.LaguerreGaussianBeam(
            beamLine, prefix, w0=w0, vortex=(lVortex, pVortex), energies=(E0,))
    elif case == 2:
        beamLine.source = rs.HermiteGaussianBeam(
            beamLine, prefix, w0=w0, TEM=(m, n), energies=(E0,))
    beamLine.fsmFar = rsc.Screen(beamLine, 'FSM', [0, 0, 0])
    return beamLine


def run_process(beamLine):
    outDict = {}
    for ip, (p, (x, z)) in enumerate(zip(ps, beamLine.fsmXZmeshes)):
        print('screen position {0} of {1}'.format(ip+1, len(ps)))
        beamLine.fsmFar.center[1] = p
        waveOnFSMg = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
        beamLine.source.shine(wave=waveOnFSMg)
#        mult = np.exp(0.5j * waveOnFSMg.x / x.max())
#        waveOnFSMg.Es *= mult
#        waveOnFSMg.Ep *= mult
#        mult = np.exp(0.5j * waveOnFSMg.z / z.max())
#        waveOnFSMg.Es *= mult
#        waveOnFSMg.Ep *= mult
        if outDict == {}:
            beamSource = waveOnFSMg
        what = 'beamFSMg{0}'.format(ip)
        outDict[what] = waveOnFSMg

        if p > 100 and wantKirchhoff:
            wrepeats = 1
            waveOnFSMk = beamLine.fsmFar.prepare_wave(beamLine.source, x, z)
            for r in range(wrepeats):
                rw.diffract(beamSource, waveOnFSMk)
                if wrepeats > 1:
                    print('wave repeats: {0} of {1} done'.format(
                        r+1, wrepeats))
            what = 'beamFSMk{0}'.format(ip)
            outDict[what] = waveOnFSMk
    return outDict
rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    beamLine.fsmXZmeshes = []
    for ip, p in enumerate(ps):
        lim = beamLine.source.w(p, E0) * maxFactor * 1e3

        plot = xrtp.XYCPlot(
            'beamFSMg{0}'.format(ip), (1,),
            xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=bins, ppb=ppb),
            yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=bins, ppb=ppb),
            caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                               bins=bins, ppb=ppb))
        plot.xaxis.limits = [-lim, lim]
        plot.yaxis.limits = [-lim, lim]
        plot.title = '{0}-{1:02d}-beamFSMg-at{2:03.1f}m'.format(
            prefix, ip, p*1e-3)
        tpf = '{0:2.1f} m' if p < 1000 else '{0:2.0f} m'
        plot.textPanel = plot.ax2dHist.text(
            0.02, 0.98, tpf.format(p*1e-3), size=14, color='w',
            transform=plot.ax2dHist.transAxes,
            ha='left', va='top')
        plot.saveName = plot.title + '.png'
        plots.append(plot)

        if p > 100 and wantKirchhoff:
            plot = xrtp.XYCPlot(
                'beamFSMk{0}'.format(ip), (1,),
                xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=bins, ppb=ppb),
                yaxis=xrtp.XYCAxis(r'$z$', u'µm', bins=bins, ppb=ppb),
                caxis=xrtp.XYCAxis('Es phase', '', data=raycing.get_Es_phase,
                                   bins=bins, ppb=ppb))
            plot.xaxis.limits = [-lim, lim]
            plot.yaxis.limits = [-lim, lim]
            plot.textPanel = plot.ax2dHist.text(
                0.02, 0.98, tpf.format(p*1e-3), size=14, color='w',
                transform=plot.ax2dHist.transAxes,
                ha='left', va='top')
            plot.title = '{0}-{1:02d}-beamFSMk-at{2:03.1f}m'.format(
                prefix, ip, p*1e-3)
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

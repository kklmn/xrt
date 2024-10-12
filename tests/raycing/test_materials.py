# -*- coding: utf-8 -*-
"""
Tests for Materials
-------------------

The module compares reflectivity, transmittivity, refraction index,
absorption coefficient etc. with those calculated by XOP.

Various material properties are compared with those calculated by XCrystal/XOP
(also used by shadow for modeling crystal optics), XInpro/XOP and others.

Find the corresponding scripts in `tests/raycing` directory.

Reflectivity of Bragg crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The small amplitude differences with XOP results are due
to slight differences in Debye-Waller factor and/or tabulated values of the
atomic scattering factors. The phase difference between s- and p-polarized
rays (calculated by xrt, cyan line, right Y axis) is not calculated by the
XOP programs and therefore is given without comparison.

+-------+--------------------+------------------+-------------------+
|       |      α = -5°       |    symmetric     |       α = 5°      |
+=======+====================+==================+===================+
| thick | |bSi111_thick_-5|  | |bSi111_thick_0| | |bSi111_thick_5|  |
|       | |bSi333_thick_-5|  | |bSi333_thick_0| | |bSi333_thick_5|  |
+-------+--------------------+------------------+-------------------+
| 100 µm|   |bSi111_100_-5|  |  |bSi111_100_0|  |  |bSi111_100_5|   |
|       |   |bSi333_100_-5|  |  |bSi333_100_0|  |  |bSi333_100_5|   |
+-------+--------------------+------------------+-------------------+
| 7 µm  |   |bSi111_007_-5|  |  |bSi111_007_0|  |  |bSi111_007_5|   |
|       |   |bSi333_007_-5|  |  |bSi333_007_0|  |  |bSi333_007_5|   |
+-------+--------------------+------------------+-------------------+

.. |bSi111_thick_-5| imagezoom:: _images/bSi111_thick_-5.*
.. |bSi111_thick_0| imagezoom:: _images/bSi111_thick_0.*
.. |bSi111_thick_5| imagezoom:: _images/bSi111_thick_5.*
   :loc: upper-right-corner
.. |bSi111_100_-5| imagezoom:: _images/bSi111_100mum_-5.*
.. |bSi111_100_0| imagezoom:: _images/bSi111_100mum_0.*
.. |bSi111_100_5| imagezoom:: _images/bSi111_100mum_5.*
   :loc: upper-right-corner
.. |bSi111_007_-5| imagezoom:: _images/bSi111_007mum_-5.*
.. |bSi111_007_0| imagezoom:: _images/bSi111_007mum_0.*
.. |bSi111_007_5| imagezoom:: _images/bSi111_007mum_5.*
   :loc: upper-right-corner

.. |bSi333_thick_-5| imagezoom:: _images/bSi333_thick_-5.*
.. |bSi333_thick_0| imagezoom:: _images/bSi333_thick_0.*
.. |bSi333_thick_5| imagezoom:: _images/bSi333_thick_5.*
   :loc: upper-right-corner
.. |bSi333_100_-5| imagezoom:: _images/bSi333_100mum_-5.*
.. |bSi333_100_0| imagezoom:: _images/bSi333_100mum_0.*
.. |bSi333_100_5| imagezoom:: _images/bSi333_100mum_5.*
   :loc: upper-right-corner
.. |bSi333_007_-5| imagezoom:: _images/bSi333_007mum_-5.*
.. |bSi333_007_0| imagezoom:: _images/bSi333_007mum_0.*
.. |bSi333_007_5| imagezoom:: _images/bSi333_007mum_5.*
   :loc: upper-right-corner

Reflectivity of Laue crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The small amplitude differences with XOP results are due
to slight differences in Debye-Waller factor and/or tabulated values of the
atomic scattering factors. The phase difference between s- and p-polarized
rays (calculated by xrt, cyan line, right Y axis) is not calculated by the
XOP programs and therefore is given without comparison.

+-------+--------------------+------------------+-------------------+
|       |      α = -5°       |    symmetric     |       α = 5°      |
+=======+====================+==================+===================+
| 100 µm|   |lSi111_100_-5|  |  |lSi111_100_0|  |  |lSi111_100_5|   |
|       |   |lSi333_100_-5|  |  |lSi333_100_0|  |  |lSi333_100_5|   |
+-------+--------------------+------------------+-------------------+
| 7 µm  |   |lSi111_007_-5|  |  |lSi111_007_0|  |  |lSi111_007_5|   |
|       |   |lSi333_007_-5|  |  |lSi333_007_0|  |  |lSi333_007_5|   |
+-------+--------------------+------------------+-------------------+

.. |lSi111_100_-5| imagezoom:: _images/lSi111_100mum_-5.*
.. |lSi111_100_0| imagezoom:: _images/lSi111_100mum_0.*
.. |lSi111_100_5| imagezoom:: _images/lSi111_100mum_5.*
   :loc: upper-right-corner
.. |lSi111_007_-5| imagezoom:: _images/lSi111_007mum_-5.*
.. |lSi111_007_0| imagezoom:: _images/lSi111_007mum_0.*
.. |lSi111_007_5| imagezoom:: _images/lSi111_007mum_5.*
   :loc: upper-right-corner

.. |lSi333_100_-5| imagezoom:: _images/lSi333_100mum_-5.*
.. |lSi333_100_0| imagezoom:: _images/lSi333_100mum_0.*
.. |lSi333_100_5| imagezoom:: _images/lSi333_100mum_5.*
   :loc: upper-right-corner
.. |lSi333_007_-5| imagezoom:: _images/lSi333_007mum_-5.*
.. |lSi333_007_0| imagezoom:: _images/lSi333_007mum_0.*
.. |lSi333_007_5| imagezoom:: _images/lSi333_007mum_5.*
   :loc: upper-right-corner

.. _transmittivity_Bragg:

Transmittivity of Bragg crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The curves are basically equal only for the symmetric case. Both
XCrystal/XOP and XInpro/XOP are different for asymmetric crystals, when they
give too low or too high (>1) transmittivity. This is because they refer to
intencities, not flux and therefore cannot be directly applied to rays in
ray tracing.

+-------+--------------------+-------------------+--------------------+
|       |      α = -5°       |     symmetric     |        α = 5°      |
+=======+====================+===================+====================+
| 100 µm|  |btSi111_100_-5|  |  |btSi111_100_0|  |  |btSi111_100_5|   |
|       |  |btSi333_100_-5|  |  |btSi333_100_0|  |  |btSi333_100_5|   |
+-------+--------------------+-------------------+--------------------+
| 7 µm  |  |btSi111_007_-5|  |  |btSi111_007_0|  |  |btSi111_007_5|   |
|       |  |btSi333_007_-5|  |  |btSi333_007_0|  |  |btSi333_007_5|   |
+-------+--------------------+-------------------+--------------------+

.. |btSi111_100_-5| imagezoom:: _images/btSi111_100mum_-5.*
.. |btSi111_100_0| imagezoom:: _images/btSi111_100mum_0.*
.. |btSi111_100_5| imagezoom:: _images/btSi111_100mum_5.*
   :loc: upper-right-corner
.. |btSi111_007_-5| imagezoom:: _images/btSi111_007mum_-5.*
.. |btSi111_007_0| imagezoom:: _images/btSi111_007mum_0.*
.. |btSi111_007_5| imagezoom:: _images/btSi111_007mum_5.*
   :loc: upper-right-corner

.. |btSi333_100_-5| imagezoom:: _images/btSi333_100mum_-5.*
.. |btSi333_100_0| imagezoom:: _images/btSi333_100mum_0.*
.. |btSi333_100_5| imagezoom:: _images/btSi333_100mum_5.*
   :loc: upper-right-corner
.. |btSi333_007_-5| imagezoom:: _images/btSi333_007mum_-5.*
.. |btSi333_007_0| imagezoom:: _images/btSi333_007mum_0.*
.. |btSi333_007_5| imagezoom:: _images/btSi333_007mum_5.*
   :loc: upper-right-corner

Transmittivity of Laue crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The curves are basically equal only for the symmetric case and only with
XInpro/XOP results. XInpro/XOP is wrong for asymmetric crystals, when it
gives too low or too high (>1) transmittivity. As seen, XCrystal/XOP is
essentially different and wrong with Laue transmittivity.

+-------+--------------------+-------------------+--------------------+
|       |      α = -5°       |    symmetric      |        α = 5°      |
+=======+====================+===================+====================+
| 100 µm|  |ltSi111_100_-5|  |  |ltSi111_100_0|  |  |ltSi111_100_5|   |
|       |  |ltSi333_100_-5|  |  |ltSi333_100_0|  |  |ltSi333_100_5|   |
+-------+--------------------+-------------------+--------------------+
| 7 µm  |  |ltSi111_007_-5|  |  |ltSi111_007_0|  |  |ltSi111_007_5|   |
|       |  |ltSi333_007_-5|  |  |ltSi333_007_0|  |  |ltSi333_007_5|   |
+-------+--------------------+-------------------+--------------------+

.. |ltSi111_100_-5| imagezoom:: _images/ltSi111_100mum_-5.*
.. |ltSi111_100_0| imagezoom:: _images/ltSi111_100mum_0.*
.. |ltSi111_100_5| imagezoom:: _images/ltSi111_100mum_5.*
   :loc: upper-right-corner
.. |ltSi111_007_-5| imagezoom:: _images/ltSi111_007mum_-5.*
.. |ltSi111_007_0| imagezoom:: _images/ltSi111_007mum_0.*
.. |ltSi111_007_5| imagezoom:: _images/ltSi111_007mum_5.*
   :loc: upper-right-corner

.. |ltSi333_100_-5| imagezoom:: _images/ltSi333_100mum_-5.*
.. |ltSi333_100_0| imagezoom:: _images/ltSi333_100mum_0.*
.. |ltSi333_100_5| imagezoom:: _images/ltSi333_100mum_5.*
   :loc: upper-right-corner
.. |ltSi333_007_-5| imagezoom:: _images/ltSi333_007mum_-5.*
.. |ltSi333_007_0| imagezoom:: _images/ltSi333_007mum_0.*
.. |ltSi333_007_5| imagezoom:: _images/ltSi333_007mum_5.*
   :loc: upper-right-corner

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

Mirror reflectivity
~~~~~~~~~~~~~~~~~~~

The small amplitude differences with XOP are due
to slight differences in tabulated values of the atomic scattering factors.
The phase difference between s- and p-polarized rays (calculated by xrt,
cyan line, right Y axis) is not calculated by the XOP programs and
therefore is given without comparison.

+---------+---------+
|  |mSi|  | |mSiO2| |
+---------+---------+
|  |mRh|  |  |mPt|  |
+---------+---------+

.. |mSi| imagezoom:: _images/MirrorReflSi@0.5deg.*
.. |mSiO2| imagezoom:: _images/MirrorReflSiO2@0.5deg.*
   :loc: upper-right-corner
.. |mRh| imagezoom:: _images/MirrorReflRh@2mrad.*
.. |mPt| imagezoom:: _images/MirrorReflPt@4mrad.*
   :loc: upper-right-corner

.. _multilayer_reflectivity:

Slab, multilayer and coating reflectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, the phase difference between s- and p-polarized rays is given without
comparison.

+---------------------+---------------------+
|        |mlW|        |       |mlSiW|       |
+---------------------+---------------------+
|      |mlSiWg|       |   |mlSiWCXRO_id0|   |
+---------------------+---------------------+
|   |mlSiWCXRO_id6|   |      |mlRhOnSi|     |
+---------------------+---------------------+
|  |DiamondOnQuartz|  |                     |
+---------------------+---------------------+

.. |mlW| imagezoom:: _images/SlabReflW.*
.. |mlSiW| imagezoom:: _images/MultilayerSiW.*
   :loc: upper-right-corner
.. |mlSiWg| imagezoom:: _images/MultilayerSiW-graded.*
.. |mlSiWCXRO_id0| imagezoom:: _images/MultilayerSiWCXRO_id0.*
   :loc: upper-right-corner
.. |mlSiWCXRO_id6| imagezoom:: _images/MultilayerSiWCXRO_id6.*
.. |mlRhOnSi| imagezoom:: _images/MirrorRefl30nmRhOnSi_4mrad_RMSroughness2nm.*
   :loc: upper-right-corner
.. |DiamondOnQuartz| imagezoom:: _images/MirrorRefl20nmDiamondOnQuartz_0.2deg_RMSroughness1nm.*

.. _tests_ml_tran:

.. note::

    At low energy, the result strongly depends on the used tabulation.
    'xrt-Henke' below overplots the curves calculated by Mlayer and REFLEC that
    use the tabulation by Henke. 'xrt-Chantler' is significantly different, and
    it is more trustworthy.

.. note::

    Mlayer/XOP does not calculate multilayers in transmission.

+-------------+-------------+
|  |mlScCr|   |  |mlScCrt|  |
+-------------+-------------+

.. |mlScCr| imagezoom:: _images/MultilayerScCr.*
.. |mlScCrt| imagezoom:: _images/MultilayerScCr-transmitted.*
   :loc: upper-right-corner

Transmittivity of materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The small amplitude differences with XOP are due
to slight differences in tabulated values of the atomic scattering factors.

.. imagezoom:: _images/TransmDiamond.*

.. raw:: html

    <div class="clearer"> </div>

Absorption of materials
~~~~~~~~~~~~~~~~~~~~~~~

The deviations at low energies due to differences in tabulated values of the
atomic scattering factors.

+---------+---------+---------+
|  |aBe|  |  |aNi|  |  |aAu|  |
+---------+---------+---------+

.. |aBe| imagezoom:: _images/AbsorptionBe.*
   :loc: lower-left-corner
.. |aNi| imagezoom:: _images/AbsorptionNi.*
   :loc: lower-left-corner
.. |aAu| imagezoom:: _images/AbsorptionAu.*
   :loc: lower-right-corner

"""
__author__ = "Konstantin Klementiev"
__date__ = "12 Mar 2014"

import math
#import cmath
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import pickle

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing as raycing
from xrt.backends.raycing.pyTTE_x import TakagiTaupin, TTcrystal, TTscan, Quantity
raycing._VERBOSITY_ = 1000


class BlackLineObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, x0+width], [0.5*height, 0.5*height],
                        linestyle=orig_handle[0].get_linestyle(), color='k')
        return [l1]


class TwoLineObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, x0+width], [0.7*height, 0.7*height],
                        linestyle=orig_handle[0].get_linestyle(),
                        color=orig_handle[0].get_color())
        if orig_handle[1]:
            l2 = plt.Line2D([x0, x0+width], [0.3*height, 0.3*height],
                            linestyle=orig_handle[1].get_linestyle(),
                            color=orig_handle[1].get_color())
            return [l1, l2]
        else:
            return [l1]


class TwoScatterObjectsHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0+0.2*width], [0.5*height],
                        marker=orig_handle[0].get_paths()[0],
                        color=orig_handle[0].get_edgecolors()[0],
                        markerfacecolor='none')
        l2 = plt.Line2D([x0+0.8*width], [0.5*height],
                        marker=orig_handle[1].get_paths()[0],
                        color=orig_handle[1].get_edgecolors()[0],
                        markerfacecolor='none')
        return [l1, l2]


def compare_rocking_curves(hkl, t=None, geom='Bragg reflected', factDW=1.,
                           legendPos1=4, legendPos2=1):
    """A comparison subroutine used in the module test suit."""
    def for_one_alpha(crystal, alphaDeg, hkl):
        alpha = math.radians(alphaDeg)
        s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
        sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
        if geom.startswith('Bragg'):
            n = (0, 0, 1)  # outward surface normal
        else:
            n = (0, -1, 0)  # outward surface normal
        hn = (0, math.sin(alpha), math.cos(alpha))  # outward Bragg normal
        gamma0 = sum(i*j for i, j in zip(n, s0))
        gammah = sum(i*j for i, j in zip(n, sh))
        hns0 = sum(i*j for i, j in zip(hn, s0))

        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.88)
        ax = fig.add_subplot(111)

#        curS, curP = crystal.get_amplitude_Authie(E, gamma0, gammah, hns0)
#        p5, = ax.plot((theta - thetaCenter) * convFactor, abs(curS)**2, '-g')
#        p6, = ax.plot((theta - thetaCenter) * convFactor, abs(curP)**2, '--g')
        curS, curP = crystal.get_amplitude(E, gamma0, gammah, hns0)
# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(curS * curP.conj()))
        p9, = ax2.plot((theta-thetaCenter) * convFactor, phi, 'c', lw=1,
                       yunits=math.pi, zorder=0)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')

        if t is not None:
            tt = u', t={0:.0f}µm'.format(t * 1e3)
            tname = '{0:03d}mum'.format(int(t * 1e3))
        else:
            tt = ' thick'
            tname = 'thick'
        if geom.startswith('Bragg'):
            geomPrefix = 'b'
        else:
            geomPrefix = 'l'
        if geom.endswith('transmitted'):
            geomPrefix += 't'
        fig.suptitle(r'{0} Si{1}, $\alpha={2:.1f}^\circ${3}'.format(geom,
                     hkl, alphaDeg, tt), fontsize=16)

        path = os.path.join('', 'XOP-RockingCurves') + os.sep
        x, R2s = np.loadtxt("{0}{1}Si{2}_{3}_{4:-.0f}_s.xc.gz".format(path,
                            geomPrefix, hkl, tname, alphaDeg), unpack=True)
        p1, = ax.plot(x, R2s, '-k', label='s XCrystal')
        x, R2p = np.loadtxt("{0}{1}Si{2}_{3}_{4:-.0f}_p.xc.gz".format(path,
                            geomPrefix, hkl, tname, alphaDeg), unpack=True)
        p2, = ax.plot(x, R2p, '--k', label='p XCrystal')

        x, R2s = np.loadtxt("{0}{1}Si{2}_{3}_{4:-.0f}_s.xin.gz".format(path,
                            geomPrefix, hkl, tname, alphaDeg), unpack=True)
        p3, = ax.plot(x, R2s, '-b', label='s XInpro')
        x, R2p = np.loadtxt("{0}{1}Si{2}_{3}_{4:-.0f}_p.xin.gz".format(path,
                            geomPrefix, hkl, tname, alphaDeg), unpack=True)
        p4, = ax.plot(x, R2p, '--b', label='p XInpro')

        p7, = ax.plot((theta - thetaCenter) * convFactor, abs(curS)**2, '-r')
        p8, = ax.plot((theta - thetaCenter) * convFactor, abs(curP)**2, '--r')
        ax.set_xlabel(r'$\theta-\theta_B$ (arcsec)')
        if geom.endswith('transmitted'):
            ax.set_ylabel('transmittivity')
        else:
            ax.set_ylabel('reflectivity')
        ax.set_xlim([dtheta[0] * convFactor, dtheta[-1] * convFactor])
#upper right	1
#upper left	2
#lower left	3
#lower right	4
#right   	5
#center left	6
#center right	7
#lower center	8
#upper center	9
#center         10
        l1 = ax2.legend([p1, p2], ['s', 'p'], loc=legendPos1)
#        ax.legend([p1, p3, p5, p7], ['XCrystal/XOP', 'XInpro/XOP',
#        'pxrt-Authier', 'pxrt-Bel&Dm'], loc=1)
        ax2.legend([p1, p3, p7], ['XCrystal/XOP', 'XInpro/XOP', 'xrt'],
                   loc=legendPos2)
        ax2.add_artist(l1)

        fname = '{0}Si{1}_{2}_{3:-.0f}'.format(
            geomPrefix, hkl, tname, alphaDeg)
        fig.savefig(fname + '.png')

    E0 = 10000.
    convFactor = 180 / math.pi * 3600.  # arcsec
    if hkl == '111':  # Si111
        if geom.startswith('Bragg'):
            dtheta = np.linspace(0, 100, 400) * 1e-6
        else:
            dtheta = np.linspace(-50, 50, 400) * 1e-6
        dSpacing = 3.13562
        hklInd = 1, 1, 1
    elif hkl == '333':  # Si333
        if geom.startswith('Bragg'):
            dtheta = np.linspace(0, 30, 400) * 1e-6
        else:
            dtheta = np.linspace(-15, 15, 400) * 1e-6
        dSpacing = 3.13562 / 3
        hklInd = 3, 3, 3

    siCrystal = rm.CrystalDiamond(hklInd, dSpacing, t=t, geom=geom,
                                  factDW=factDW)
    thetaCenter = math.asin(rm.ch / (2*siCrystal.d*E0))

    E = np.ones_like(dtheta) * E0
    theta = dtheta + thetaCenter
    for_one_alpha(siCrystal, 0., hkl)
    for_one_alpha(siCrystal, -5., hkl)
    for_one_alpha(siCrystal, 5., hkl)


def compare_rocking_curves_bent(hkl, t=None, geom='Laue reflected', factDW=1.,
                                legendPos1=4, legendPos2=1, Rcurvmm=None,
                                alphas=[0]):
    try:
        import pyopencl as cl
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        isOpenCL = True
    except ImportError:
        isOpenCL = False

    if isOpenCL:
        import xrt.backends.raycing.myopencl as mcl
        matCL = mcl.XRT_CL(r'materials.cl',
#                           precisionOpenCL='float32',
                           precisionOpenCL='float64',
                           # targetOpenCL='CPU'
                           )
    else:
        matCL = None

    """A comparison subroutine used in the module test suit."""
    def for_one_alpha(crystal, alphaDeg, hkl, t):
        xcb_tt = True
        xcb_pp = True

        Rcurv=np.inf if Rcurvmm is None else Rcurvmm
        if Rcurv==0:
            Rcurv=np.inf
        alpha = np.radians(alphaDeg)
        s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
        sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
        if geom.startswith('Bragg'):
            n = (0, 0, 1)  # outward surface normal
        else:
            n = (0, -1, 0)  # outward surface normal
        hn = (0, np.sin(alpha), np.cos(alpha))  # outward Bragg normal
        gamma0 = sum(i*j for i, j in zip(n, s0))
        gammah = sum(i*j for i, j in zip(n, sh))
        hns0 = sum(i*j for i, j in zip(hn, s0))

        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.88)
        ax = fig.add_subplot(111)

        curS, curP = crystal.get_amplitude(E, gamma0, gammah, hns0)
        if geom.startswith('L'):
            curSD, curPD = crystal.get_amplitude_TT(E, gamma0, gammah, hns0,
                                             ucl=matCL, alphaAsym=alpha,
                                             Rcurvmm=Rcurv)
        curSDpt, curPDpt = crystal.get_amplitude_pytte(E, gamma0, gammah, hns0,
                                                       ucl=matCL, alphaAsym=alpha,
                                                       Ry=Rcurv,
                                                       tolerance=1e-6)

# phases:
#        ax2 = ax.twinx()
#        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
#        phi = np.unwrap(np.angle(curS * curP.conj()))
#        phiD = np.unwrap(np.angle(curSD * curPD.conj()))
#        p9, = ax2.plot((theta-thetaCenter) * convFactor, phi, 'c', lw=1,
#                       yunits=math.pi, zorder=0)
#        p10, = ax2.plot((theta-thetaCenter) * convFactor, -phiD, 'm', lw=1,
#                        yunits=math.pi, zorder=0)
#        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
#        ax2.yaxis.set_major_formatter(formatter)
#        for tl in ax2.get_yticklabels():
#            tl.set_color('c')

        if t is not None:
            tt = u', t={0:.0f}µm'.format(t * 1e3)
            tname = '{0:03d}mkm'.format(int(t * 1e3))
            thickness = t
        else:
            tt = ' thick'
            tname = 'thick'
            thickness = 1.
        if geom.startswith('Bragg'):
            geomPrefix = 'b'
        else:
            geomPrefix = 'l'
        if geom.endswith('transmitted'):
            geomPrefix += 't'
        if np.isinf(Rcurv):
            fig.suptitle(r'{0} Si{1}, $\alpha={2:.1f}^\circ$, bending R=$\infty$, E={3:.0f}keV'.format(geom, # analysis:ignore
                         hkl, alphaDeg, E0/1e3), fontsize=16)
        else:
            fig.suptitle(r'{0} Si{1}, $\alpha={2:.1f}^\circ$, bending R={3:.1f}m, E={4:.0f}keV'.format(geom, # analysis:ignore
                         hkl, alphaDeg, Rcurv/1e3, E0/1e3), fontsize=16)

        path = os.path.join('', 'XOP-RockingCurves-Bent') + os.sep
        try:
            if np.isinf(Rcurv) or alphaDeg == 0:
                x, R2p, R2s = np.loadtxt(
                    "{0}{1}Si{2}_E{3:-.0f}keV_t{4}_R{5:-.0f}m_a{6:-.0f}deg.xc.gz".format( # analysis:ignore
                        path, geomPrefix, hkl, E0/1e3, tname, Rcurv/1e3, alphaDeg),
                    unpack=True, usecols=(0, 2, 3))
                p1, = ax.plot(x, R2s, '-k', label='s XCrystal', linewidth=2)
    #            p2, = ax.plot(x, R2p, '--k', label='p XCrystal')
    
            if xcb_tt:
                x, R2s = np.loadtxt(
                    "{0}{1}Si{2}_E{3:-.0f}keV_t{4}_R{5:-.0f}m_a{6:-.0f}deg_tt_sigma.xcb.gz".format( # analysis:ignore
                        path, geomPrefix, hkl, E0/1e3, tname, Rcurv/1e3, alphaDeg),
                    unpack=True, usecols=(0, 3))
                p3t, = ax.plot(x*1e3, R2s, '-b', label='s XCrystal_Bent TT')
                x, R2p = np.loadtxt(
                    "{0}{1}Si{2}_E{3:-.0f}keV_t{4}_R{5:-.0f}m_a{6:-.0f}deg_tt_pi.xcb.gz".format( # analysis:ignore
                        path, geomPrefix, hkl, E0/1e3, tname, Rcurv/1e3, alphaDeg),
                    unpack=True, usecols=(0, 3))
        #        p4, = ax.plot(x*1e3, R2p, '--b', label='p XCrystal_Bent TT')
            if xcb_pp:
                x, R2s = np.loadtxt(
                    "{0}{1}Si{2}_E{3:-.0f}keV_t{4}_R{5:-.0f}m_a{6:-.0f}deg_pp_sigma.xcb.gz".format( # analysis:ignore
                        path, geomPrefix, hkl, E0/1e3, tname, Rcurv/1e3, alphaDeg),
                    unpack=True, usecols=(0, 7))
                p3p, = ax.plot(x*1e3, R2s, '-c', label='s XCrystal_Bent PP')
                x, R2p = np.loadtxt(
                    "{0}{1}Si{2}_E{3:-.0f}keV_t{4}_R{5:-.0f}m_a{6:-.0f}deg_pp_pi.xcb.gz".format( # analysis:ignore
                        path, geomPrefix, hkl, E0/1e3, tname, Rcurv/1e3, alphaDeg),
                    unpack=True, usecols=(0, 7))
        #        p4, = ax.plot(x*1e3, R2p, '--b', label='p XCrystal_Bent TT')
        except:
            pass
#        if np.isinf(Rcurv) or alphaDeg == 0:
        p7, = ax.plot((theta - thetaCenter) * convFactor,
                      abs(curS)**2, '-r', linewidth=2)
#            p8, = ax.plot((theta - thetaCenter) * convFactor,
#                          abs(curP)**2, '-c', linewidth=2)
        if geom.startswith('L'):
            p11, = ax.plot((theta - thetaCenter) * convFactor,
                           abs(curSD)**2, '--g', linewidth=2)
    #        p12, = ax.plot((theta - thetaCenter) * convFactor,
    #                       abs(curPD)**2, '--r')

        p13, = ax.plot((theta - thetaCenter) * convFactor,
                       abs(curSDpt)**2, 'm', linewidth=2)
#        p14, = ax.plot((theta - thetaCenter) * convFactor,
#                       abs(curPDpt)**2, '--r')

        ax.set_xlabel(r'$\theta-\theta_B$ ($\mu$rad)')
        if geom.endswith('transmitted'):
            ax.set_ylabel('transmittivity')
        else:
            ax.set_ylabel('reflectivity')
        ax.set_xlim([dtheta[0] * convFactor, dtheta[-1] * convFactor])

        plotList = [p7, p13]
        curveList = ['xrt perfect', 'xrt pyTTE']
        if geom.startswith('L'):
            plotList.append(p11)
            curveList.append('xrt TT')

        try:
            if np.isinf(Rcurv) or alphaDeg == 0:
                plotList.append(p1)
                curveList.append('XCrystal')
            if xcb_tt:
                plotList.append(p3t)
                curveList.append('XCrystal_Bent TT')
            if xcb_pp:
                plotList.append(p3p)
                curveList.append('XCrystal_Bent PP')
        except:
            pass
        legendPostmp = legendPos2 if alphaDeg > 0 else legendPos2+1
        ax.legend(plotList, curveList, loc=legendPostmp)
#        ax2.add_artist(l1)

        fname = '{0}Si{1}_{2}_a{3:-.0f}_{4:-.2f}'.format(
            geomPrefix, hkl, tname, alphaDeg, Rcurv/1e3)

        fig.savefig(fname + '.png')

    E0 = 50000. if geom.startswith('L') else 7000.
#    convFactor = 180 / np.pi * 3600.  # arcsec
    convFactor = 1e6
    for alpha in alphas:
        if hkl == '111':  # Si111
            if geom.startswith('Bragg'):
                dtheta = np.linspace(-200, 200, 1000) * 1e-6
            else:
                dtheta = np.linspace(-5-np.abs(alpha)*2*50e3/Rcurvmm,
                                     5+np.abs(alpha)*2*50e3/Rcurvmm,
                                     1000) * 1e-6
            dSpacing = 3.13562
            hklInd = 1, 1, 1
        elif hkl == '333':  # Si333
            if geom.startswith('Bragg'):
                dtheta = np.linspace(-100, 100, 1000) * 1e-6
            else:
                dtheta = np.linspace(-2-np.abs(alpha)**1.2*50e3/Rcurvmm,
                                     2+np.abs(alpha)**1.2*50e3/Rcurvmm,
                                     1000) * 1e-6
            dSpacing = 3.13562 / 3.
            hklInd = 3, 3, 3

#        siCrystal = rm.CrystalDiamond(hklInd, dSpacing, t=t, geom=geom,
#                                      factDW=factDW)
        siCrystal = rm.CrystalSi(hkl=hklInd, t=t, geom=geom, factDW=factDW)
        thetaCenter = math.asin(rm.ch / (2*siCrystal.d*E0))

        E = np.ones_like(dtheta) * E0
        theta = dtheta + thetaCenter
        for_one_alpha(siCrystal, alpha, hkl, t)

def compare_crystal_bending(hkl, t=None, geom='Laue reflected', factDW=1.,
                                legendPos1=4, legendPos2=1, Rcurvmm=[np.inf],
                                alphas=[0]):
    try:
        import pyopencl as cl
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        isOpenCL = True
    except ImportError:
        isOpenCL = False

    if isOpenCL:
        import xrt.backends.raycing.myopencl as mcl
        matCL = mcl.XRT_CL(r'materials.cl',
                           precisionOpenCL='float32',
#                           precisionOpenCL='float64',
                           # targetOpenCL='CPU'
                           )
    else:
        matCL = None
        
        
    """A comparison subroutine used in the module test suit."""
    def for_one_R(crystal, alphaDeg, hkl, t, R):

        Rcurv=np.inf if R is None else R
        if Rcurv==0:
            Rcurv=np.inf
        alpha = np.radians(alphaDeg)
        s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
        sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))
        if geom.startswith('Bragg'):
            n = (0, 0, 1)  # outward surface normal
        else:
            n = (0, -1, 0)  # outward surface normal
        hn = (0, np.sin(alpha), np.cos(alpha))  # outward Bragg normal
        gamma0 = sum(i*j for i, j in zip(n, s0))
        gammah = sum(i*j for i, j in zip(n, sh))
        hns0 = sum(i*j for i, j in zip(hn, s0))

        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.88)
        ax = fig.add_subplot(111)

        curS, curP = crystal.get_amplitude(E, gamma0, gammah, hns0)

        curSDpt, curPDpt = crystal.get_amplitude_pytte(E, gamma0, gammah, hns0,
                                                       ucl=matCL, alphaAsym=alpha,
                                                       Ry=Rcurv)

        pyTTE = False
        if pyTTE:
            geotag = 0 if geom.startswith('B') else np.pi*0.5
#            print(Rcurv)
            ttx = TTcrystal(crystal = 'Si', hkl=crystal.hkl, thickness = Quantity(t*1e3,'um'), 
                            debye_waller = 1, xrt_crystal=crystal, Rx=Quantity(Rcurv, 'mm'),
                            asymmetry = Quantity(alpha+geotag, 'rad'))
            tts = TTscan(constant=Quantity(E[0],'eV'),
                         scan=Quantity(theta-thetaCenter, 'rad'),
                         polarization='sigma')
#            ttp = TTscan(constant=Quantity(E[0],'eV'),
#                         scan=Quantity(theta-thetaCenter, 'rad'),
#                         polarization='pi')
            
            scan_tt_s=TakagiTaupin(ttx, tts)
#            scan_tt_p=TakagiTaupin(ttx, ttp)        
    
            if __name__ == '__main__':
                scan_vector, R, T, curSD = scan_tt_s.run()
#                scan_vector, R, T, curPD = scan_tt_p.run()

# phases:
#        ax2 = ax.twinx()
#        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
#        phi = np.unwrap(np.angle(curS * curP.conj()))
#        phiD = np.unwrap(np.angle(curSD * curPD.conj()))
#        p9, = ax2.plot((theta-thetaCenter) * convFactor, phi, 'c', lw=1,
#                       yunits=math.pi, zorder=0)
#        p10, = ax2.plot((theta-thetaCenter) * convFactor, -phiD, 'm', lw=1,
#                        yunits=math.pi, zorder=0)
#        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
#        ax2.yaxis.set_major_formatter(formatter)
#        for tl in ax2.get_yticklabels():
#            tl.set_color('c')

        if t is not None:
            tt = u', t={0:.0f}µm'.format(t * 1e3)
            tname = '{0:03d}mkm'.format(int(t * 1e3))
            thickness = t
        else:
            tt = ' thick'
            tname = 'thick'
            thickness = 1.
        if geom.startswith('Bragg'):
            geomPrefix = 'b'
        else:
            geomPrefix = 'l'

        if np.isinf(Rcurv):
            fig.suptitle(r'{0} Si{1}, $\alpha={2:.1f}^\circ$, bending R=$\infty$, E={3:.0f}keV'.format(geom, # analysis:ignore
                         hkl, alphaDeg, E0/1e3), fontsize=16)
        else:
            fig.suptitle(r'{0} Si{1}, $\alpha={2:.1f}^\circ$, bending R={3:.1f}m, E={4:.0f}keV'.format(geom, # analysis:ignore
                         hkl, alphaDeg, Rcurv/1e3, E0/1e3), fontsize=16)

        p7, = ax.plot((theta - thetaCenter) * convFactor,
                      abs(curS)**2, '-r', linewidth=2)
#            p8, = ax.plot((theta - thetaCenter) * convFactor,
#                          abs(curP)**2, '-c', linewidth=2)

        if pyTTE:
            p11, = ax.plot((theta - thetaCenter) * convFactor,
                           abs(curSD)**2, 'g', linewidth=2)
    #        p12, = ax.plot((theta - thetaCenter) * convFactor,
    #                       abs(curPD)**2, '--r')

        p13, = ax.plot((theta - thetaCenter) * convFactor,
                       abs(curSDpt)**2, 'b', linewidth=2)
#        p14, = ax.plot((theta - thetaCenter) * convFactor,
#                       abs(curPDpt)**2, '--r')

        ax.set_xlabel(r'$\theta-\theta_B$ ($\mu$rad)')
        if geom.endswith('transmitted'):
            ax.set_ylabel('transmittivity')
        else:
            ax.set_ylabel('reflectivity')
        ax.set_xlim([dtheta[0] * convFactor, dtheta[-1] * convFactor])

        plotList = [p7, p13]
        curveList = ['xrt perfect', 'xrt pyTTE']
        if pyTTE:
            plotList.append(p11)
            curveList.append('pyTTE')


        legendPostmp = legendPos2 if alphaDeg > 0 else legendPos2+1
        ax.legend(plotList, curveList, loc=legendPostmp)
#        ax2.add_artist(l1)

        fname = '{0}Si{1}_{2}_a{3:-.0f}_{4:-.2f}'.format(
            geomPrefix, hkl, tname, alphaDeg, Rcurv/1e3)

        fig.savefig(fname + '.png')

    E0 = 30000. if geom.startswith('L') or hkl =='333' else 15000.
#    convFactor = 180 / np.pi * 3600.  # arcsec
    convFactor = 1e6
    for radius in Rcurvmm:
        if hkl == '111':  # Si111
            if geom.startswith('Bragg'):
                dtheta = np.linspace(min(-2e6/radius, -100), 100, 2000) * 1e-6
            else:
                dtheta = np.linspace(-150, 100, 1000) * 1e-6
                #                dtheta = np.linspace(-5-np.abs(alphas[0])*2*50e3/radius,
#                                     5+np.abs(alphas[0])*2*50e3/radius,
#                                     1000) * 1e-6
#            dSpacing = 3.13562
            hklInd = 1, 1, 1
        elif hkl == '333':  # Si333
            if geom.startswith('Bragg'):
                dtheta = np.linspace(-100, 100, 1000) * 1e-6
            else:
                dtheta = np.linspace(-2-np.abs(alphas[0])**1.2*50e3/radius,
                                     2+np.abs(alphas[0])**1.2*50e3/radius,
                                     1000) * 1e-6
#            dSpacing = 3.13562 / 3.
            hklInd = 3, 3, 3

#        siCrystal = rm.CrystalDiamond(hklInd, dSpacing, t=t, geom=geom,
#                                      factDW=factDW)
        siCrystal = rm.CrystalSi(hkl=hklInd, t=t, geom=geom, factDW=factDW)
        thetaCenter = math.asin(rm.ch / (2*siCrystal.d*E0))

        E = np.ones_like(dtheta) * E0
        theta = dtheta + thetaCenter
        for_one_R(siCrystal, alphas[0], hkl, t, radius)


def compare_Bragg_Laue(hkl, beamPath, factDW=1.):
    """A comparison subroutine used in the module test suit."""
    def for_one_alpha(alphaDeg, hkl):
        alpha = math.radians(alphaDeg)
        s0 = (np.zeros_like(theta), np.cos(theta+alpha), -np.sin(theta+alpha))
        sh = (np.zeros_like(theta), np.cos(theta-alpha), np.sin(theta-alpha))

#'Bragg':
        n = (0, 0, 1)  # outward surface normal
        hn = (0, math.sin(alpha), math.cos(alpha))  # outward Bragg normal
        gamma0 = sum(i*j for i, j in zip(n, s0))
        gammah = sum(i*j for i, j in zip(n, sh))
        hns0 = sum(i*j for i, j in zip(hn, s0))
        braggS, braggP = siBraggCrystal.get_amplitude(E, gamma0, gammah, hns0)
#'Laue':
        n = (0, -1, 0)  # outward surface normal
        hn = (0, math.sin(alpha), math.cos(alpha))  # outward Bragg normal
        gamma0 = sum(i*j for i, j in zip(n, s0))
        gammah = sum(i*j for i, j in zip(n, sh))
        hns0 = sum(i*j for i, j in zip(hn, s0))
        laueS, laueP = siLaueCrystal.get_amplitude(E, gamma0, gammah, hns0)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(braggS * braggP.conj()))
        p5, = ax2.plot((theta-thetaCenter) * convFactor, phi, '-c', lw=1,
                       yunits=math.pi, zorder=0)
        phi = np.unwrap(np.angle(laueS * laueP.conj()))
        p6, = ax2.plot((theta-thetaCenter) * convFactor, phi, '-c.', lw=1,
                       yunits=math.pi, zorder=0)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')

        fig.suptitle(r'Comparison of Bragg and Laue transmittivity for Si{0}'.
                     format(hkl), fontsize=16)
        p1, = ax.plot((theta-thetaCenter) * convFactor, abs(braggS)**2, '-r')
        p2, = ax.plot((theta-thetaCenter) * convFactor, abs(braggP)**2, '-b')
        p3, = ax.plot((theta-thetaCenter) * convFactor, abs(laueS)**2, '-r.')
        p4, = ax.plot((theta-thetaCenter) * convFactor, abs(laueP)**2, '-b.')
        ax.set_xlabel(r'$\theta-\theta_B$ (arcsec)')
        ax.set_ylabel('transmittivity')

#upper right	1
#upper left	2
#lower left	3
#lower right	4
#right   	5
#center left	6
#center right	7
#lower center	8
#upper center	9
#center         10
        l1 = ax.legend([p1, p2], ['s', 'p'], loc=3)
        ax.legend([p1, p3], [u'Bragg t={0:.1f} µm'.format(
            siBraggCrystal.t * 1e3), u'Laue t={0:.1f} µm'.format(
            siLaueCrystal.t * 1e3)], loc=2)
        ax.add_artist(l1)
        ax.set_xlim([dtheta[0] * convFactor, dtheta[-1] * convFactor])

        fname = r'BraggLaueTrSi{0}'.format(hkl)
        fig.savefig(fname + '.png')

    E0 = 10000.
    convFactor = 180 / math.pi * 3600.  # arcsec
    if hkl == '111':  # Si111
        dtheta = np.linspace(-100, 100, 400) * 1e-6
        dSpacing = 3.13562
        hklInd = 1, 1, 1
    elif hkl == '333':  # Si333
        dtheta = np.linspace(-30, 30, 400) * 1e-6
        dSpacing = 3.13562 / 3
        hklInd = 3, 3, 3

    thetaCenter = math.asin(rm.ch / (2*dSpacing*E0))
    t = beamPath * math.sin(thetaCenter)
    siBraggCrystal = rm.CrystalDiamond(hklInd, dSpacing, t=t,
                                       geom='Bragg transmitted', factDW=factDW)
    t = beamPath * math.cos(thetaCenter)
    siLaueCrystal = rm.CrystalDiamond(hklInd, dSpacing, t=t,
                                      geom='Laue transmitted', factDW=factDW)

    E = np.ones_like(dtheta) * E0
    theta = dtheta + thetaCenter
    for_one_alpha(0., hkl)
#    for_one_alpha(siCrystal, -5., hkl)
#    for_one_alpha(siCrystal, 5., hkl)


def compare_reflectivity():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(stripe, refs, refp, theta, reprAngle):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.86)
        ax = fig.add_subplot(111)
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('reflectivity')
        ax.set_xlim(30, 5e4)
        fig.suptitle(stripe.name + ' ' + reprAngle, fontsize=16)
        x, R2s = np.loadtxt(refs, unpack=True)
        p1, = ax.plot(x, R2s, '-k', label='s xf1f2')
        x, R2s = np.loadtxt(refp, unpack=True)
        p2, = ax.plot(x, R2s, '--k', label='p xf1f2')
        refl = stripe.get_amplitude(E, math.sin(theta))
        rs, rp = refl[0], refl[1]
        p3, = ax.semilogx(E, abs(rs)**2, '-r')
        p4, = ax.semilogx(E, abs(rp)**2, '--r')
        l1 = ax.legend([p1, p2], ['s', 'p'], loc=3)
        ax.legend([p1, p3], ['Xf1f2/XOP', 'xrt'], loc=6)
        ax.add_artist(l1)
# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(rs * rp.conj()))
        p9, = ax2.plot(E, phi, 'c', lw=2, yunits=math.pi, zorder=0)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')

        fname = 'MirrorRefl' + stripe.name + "".join(reprAngle.split())
        fig.savefig(fname + '.png')

    dataDir = os.path.join('', 'XOP-Reflectivities')
    E = np.logspace(1.+math.log10(3.), 4.+math.log10(5.), 500)
    stripeSi = rm.Material('Si', rho=2.33)
    for_one_material(stripeSi,
                     os.path.join(dataDir, "Si05deg_s.xf1f2.gz"),
                     os.path.join(dataDir, "Si05deg_p.xf1f2.gz"),
                     math.radians(0.5), '@ 0.5 deg')
    stripePt = rm.Material('Pt', rho=21.45)
    for_one_material(stripePt,
                     os.path.join(dataDir, "Pt4mrad_s.xf1f2.gz"),
                     os.path.join(dataDir, "Pt4mrad_p.xf1f2.gz"),
                     4e-3, '@ 4 mrad')
    stripeSiO2 = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)
    for_one_material(stripeSiO2,
                     os.path.join(dataDir, "SiO205deg_s.xf1f2.gz"),
                     os.path.join(dataDir, "SiO205deg_p.xf1f2.gz"),
                     math.radians(0.5), '@ 0.5 deg')
    stripeRh = rm.Material('Rh', rho=12.41)
    for_one_material(stripeRh,
                     os.path.join(dataDir, "Rh2mrad_s.xf1f2.gz"),
                     os.path.join(dataDir, "Rh2mrad_p.xf1f2.gz"),
                     2e-3, '@ 2 mrad')


def compare_reflectivity_coated():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(stripe, strOnly, refs, refp, theta, reprAngle):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.86)
        ax = fig.add_subplot(111)
        ax.set_xlabel('energy (keV)')
        ax.set_ylabel('reflectivity')
        # ax.set_xlim(100, 4e4)
        ax.set_ylim(1e-7, 2)
        fig.suptitle(stripe.name + ' ' + reprAngle, fontsize=16)
        x, R2s = np.loadtxt(refs, unpack=True, skiprows=2, usecols=(0, 1))
        p1, = ax.semilogy(x*1e-3, R2s, '-k', label='s CXRO')
        x, R2s = np.loadtxt(refp, unpack=True, skiprows=2, usecols=(0, 1))
        p2, = ax.semilogy(x*1e-3, R2s, '--k', label='p CXRO')
        refl = stripe.get_amplitude(E, math.sin(theta))
        rs, rp = refl[0], refl[1]
        rs0, rp0 = strOnly.get_amplitude(E, math.sin(theta))[0:2]
        p3, = ax.semilogy(E*1e-3, abs(rs)**2, '-r')
        p4, = ax.semilogy(E*1e-3, abs(rp)**2, '--r')
        p5, = ax.semilogy(E*1e-3, abs(rs0)**2, '-m')
        p6, = ax.semilogy(E*1e-3, abs(rp0)**2, '--c')
        l1 = ax.legend([p1, p2], ['s', 'p'], loc=3)
        ax.legend([p1, p3, p5], ['CXRO', 'xrt', 'bulk coating'], loc=1)
        ax.add_artist(l1)
# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(rs * rp.conj()))
        p9, = ax2.plot(E*1e-3, phi, 'c', lw=2, yunits=math.pi, zorder=0)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')

        ax2.set_zorder(-1)
        ax.patch.set_visible(False)  # hide the 'canvas'

        fname = 'MirrorRefl' + stripe.name + "".join(reprAngle.split())
        fig.savefig(fname + '.png')

    dataDir = os.path.join('', 'CXRO-Reflectivities')
    E = np.logspace(2., 4.+math.log10(4.), 1000)
    mSi = rm.Material('Si', rho=2.33)
    mSiO2 = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.65)
#    mB4C = rm.Material(('B', 'C'), quantities=(4, 1), rho=2.52)
    mRh = rm.Material('Rh', rho=12.41, kind='mirror')
    mC = rm.Material('C', rho=3.5, kind='mirror')
    cRhSi = rm.Coated(coating=mRh, cThickness=300,
                      substrate=mSi, surfaceRoughness=20,
                      substRoughness=20, name='30 nm Rh on Si')
    cCSiO2 = rm.Coated(coating=mC, cThickness=200,
                        substrate=mSiO2, surfaceRoughness=10,
                        substRoughness=10, name='20 nm Diamond on Quartz')
    for_one_material(cRhSi, mRh,
                     os.path.join(dataDir, "RhSi_s_rough2.CXRO.gz"),
                     os.path.join(dataDir, "RhSi_p_rough2.CXRO.gz"),
                     4e-3, '@ 4 mrad,\nRMS roughness 2 nm')

    for_one_material(cCSiO2, mC,
                      os.path.join(dataDir, "CSiO2_s_rough1.CXRO.gz"),
                      os.path.join(dataDir, "CSiO2_p_rough1.CXRO.gz"),
                      np.radians(0.2), '@ 0.2 deg,\nRMS roughness 1 nm')


def compare_reflectivity_slab():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(stripe, refs, refp, E, reprEnergy):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.86)
        ax = fig.add_subplot(111)
        ax.set_xlabel('grazing angle (deg)')
        ax.set_ylabel('reflectivity')
        ax.set_xlim(0, 10)
        fig.suptitle(stripe.name + ' ' + reprEnergy, fontsize=16)
        x, R2s = np.loadtxt(refs, unpack=True)
        p1, = ax.plot(x, R2s, '-k', label='s Mlayer')
        x, R2s = np.loadtxt(refp, unpack=True)
        p2, = ax.plot(x, R2s, '--k', label='p Mlayer')
        refl = stripe.get_amplitude(E, np.sin(np.deg2rad(theta)))
        rs, rp = refl[0], refl[1]
        p3, = ax.semilogy(theta, abs(rs)**2, '-r')
        p4, = ax.semilogy(theta, abs(rp)**2, '--r')
        l1 = ax.legend([p1, p2], ['s', 'p'], loc=3)
        ax.legend([p1, p3], ['Mlayer/XOP', 'xrt'], loc=1)
        ax.add_artist(l1)
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 1])
# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(rs * rp.conj()))
        p9, = ax2.plot(theta, phi, 'c', lw=2, yunits=math.pi, zorder=0)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')

        ax2.set_zorder(-1)
        ax.patch.set_visible(False)  # hide the 'canvas'

        fname = 'SlabRefl' + stripe.name + ' ' + reprEnergy
        fig.savefig(fname + '.png')

    dataDir = os.path.join('', 'XOP-Reflectivities')
    theta = np.linspace(0, 10, 500)  # degrees
    layerW = rm.Material('W', kind='thin mirror', rho=19.3, t=2.5e-6)
    for_one_material(layerW,
                     os.path.join(dataDir, "W25A_10kev_s.mlayer.gz"),
                     os.path.join(dataDir, "W25A_10kev_p.mlayer.gz"), 1e4,
                     u'slab, t = 25 Å, @ 10 keV')


def compare_multilayer():
    """A comparison subroutine used in the module test suit."""
#    cl_list = None
    # wantOpenCL = True
    wantOpenCL = False

    if wantOpenCL:
        try:
            import pyopencl as cl
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
            isOpenCL = True
        except ImportError:
            isOpenCL = False

        if isOpenCL:
            import xrt.backends.raycing.myopencl as mcl
            matCL = mcl.XRT_CL(r'materials.cl')
    #        cl_list = matCL.cl_ctx[0], matCL.cl_queue[0], matCL.cl_program[0]
        else:
            matCL = None
    else:
        matCL = None

    def compare(xrtMLs, refDatas, E, title, flabel=''):
        refxs, refss, refps, refphs = [], [], [], []
        refLabels = [rd[1] for rd in refDatas]
        for refData in refDatas:
            refFile = refData[0]
            if refFile.endswith('.npy'):
                data = np.load(refFile, allow_pickle=True)
                # print([key for key in data.item()])
                refx = data.item()['Angle']
                refs = data.item()['Ang, TRANSMITTANCE s-pol']
                refp = data.item()['Ang, TRANSMITTANCE p-pol']
                refph = np.radians(data.item()[
                    'Ang, PHASE DIFFERENCE DELTA-(S-P)'])
            else:
                res = np.loadtxt(refFile, unpack=True)
                refx, refs = res[:2]
                refp = res[2] if len(res) > 2 else None
                if len(res) > 4:
                    refph = (res[3] - res[4]) * np.pi
                else:
                    refph = None
            refxs.append(refx)
            refss.append(refs)
            refps.append(refp)
            refphs.append(refph)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.86)
        ax = fig.add_subplot(111)
        ax.set_xlabel('grazing angle (deg)')
        if 'trans' in xrtMLs[0][0].geom:
            ax.set_ylabel('transmittivity')
        else:
            ax.set_ylabel('reflectivity')
        ax.set_xlim(theta[0], theta[-1])
        fig.suptitle(title, fontsize=12)

        xrtData = []
        xrtLines = []
        xrtLabels = [xd[1] for xd in xrtMLs]
        for iml, ml in enumerate(xrtMLs):
            refl = ml[0].get_amplitude(E, np.sin(np.radians(theta)), ucl=matCL)
            rs, rp = refl[0], refl[1]
            xrtData.append((rs, rp))
            c = 'C{0}'.format(iml)
            p1, = ax.plot(theta, abs(rs)**2, '-', color=c, lw=2, zorder=100)
            p2, = ax.plot(theta, abs(rp)**2, '--', color=c, lw=2, zorder=100)
            xrtLines.append((p1, p2))

        refLines = []
        for iref, (refx, refs, refp) in enumerate(zip(refxs, refss, refps)):
            c = 'C{0}'.format(iref+len(xrtMLs))
            pS, = ax.plot(refx, refs, '-', lw=2, color=c)
            if refp is not None:
                pP, = ax.plot(refx, refp, '--', lw=2, color=c)
                refLines.append((pS, pP))
            else:
                refLines.append((pS, None))

        l1 = ax.legend([(p1,), (p2,)], ['s', 'p'], loc='upper right',
                       handler_map={tuple: BlackLineObjectsHandler()})

        ax.legend(xrtLines+refLines, xrtLabels+refLabels, loc='lower left',
                  title=ax.get_ylabel(),
                  handler_map={tuple: TwoLineObjectsHandler()})
        ax.add_artist(l1)
        ax.set_ylim([0, None])

# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$')  # , color='c')
        xrtLinesPh = []
        for iml, (rs, rp) in enumerate(xrtData):
            c = 'C{0}'.format(iml)
            phi = np.unwrap(np.angle(rs * rp.conj()))
            p9, = ax2.plot(theta, phi, color=c, yunits=math.pi, alpha=0.5,
                           zorder=90)
            xrtLinesPh.append(p9)
        refLinesPh = []
        for iref, (refx, refph) in enumerate(zip(refxs, refphs)):
            c = 'C{0}'.format(iref+len(xrtMLs))
            if refph is not None:
                pH, = ax2.plot(refx, refph, color=c, yunits=math.pi, alpha=0.5)
                refLinesPh.append(pH)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        ax2.set_xlim(theta[0], theta[-1])

        ax2.set_zorder(-1)
        ax.patch.set_visible(False)  # hide the 'canvas'
        ax2.legend(xrtLinesPh+refLinesPh, xrtLabels+refLabels,
                   loc='lower right', title='phase difference')

        ml = xrtMLs[0][0]
        fname = 'Multilayer{0}{1}'.format(ml.tLayer.name, ml.bLayer.name)
        fig.savefig(fname + flabel)

    dataDir = os.path.join('', 'XOP-Reflectivities')

    theta = np.linspace(35, 40, 511)  # degrees
    E0 = 398.0
    mScH = rm.Material('Sc', rho=2.98, table='Henke')
    mCrH = rm.Material('Cr', rho=7.18, table='Henke')
    mScC = rm.Material('Sc', rho=2.98, table='Chantler')
    mCrC = rm.Material('Cr', rho=7.18, table='Chantler')

    # mLH = rm.Multilayer(mScH, 12.8, mCrH, 12.8, 200, mScH)
    # mLC = rm.Multilayer(mScC, 12.8, mCrC, 12.8, 200, mScC)
    # compare(
    #     ((mLH, 'xrt-Henke'), (mLC, 'xrt-Chantler')),
    #     [(os.path.join(dataDir, "ScCr_ML_reflection_REFLEC.npy"), 'REFLEC'),
    #      (os.path.join(dataDir, "mLScCr-spph.mlayer.gz"), 'Mlayer/XOP')],
    #     E0, u'200 × [12.8 Å Sc + 12.8 Å Cr] / Sc multilayer @ 398 eV')

    mLH = rm.Multilayer(mScH, 12.8, mCrH, 12.8, 200, mScH, substThickness=13.0,
                        geom='transmitted')
    mLC = rm.Multilayer(mScC, 12.8, mCrC, 12.8, 200, mScC, substThickness=13.0,
                        geom='transmitted')
    compare(
        ((mLH, 'xrt-Henke'), (mLC, 'xrt-Chantler')),
        [(os.path.join(dataDir, "ScCr_ML_transmission_REFLEC.npy"), 'REFLEC')],
        E0, u'200 × [12.8 Å Sc + 12.8 Å Cr] / 13 Å Sc multilayer @ 398 eV',
        '-transmitted')

    # theta = np.linspace(0, 1.6, 801)  # degrees
    # mSi = rm.Material('Si', rho=2.33)
    # mW = rm.Material('W', rho=19.3)

    # mL = rm.Multilayer(mSi, 27, mW, 18, 40, mSi)
    # compare([(mL, 'xrt')],
    #         [(os.path.join(dataDir, "WSi45A04.mlayer.gz"), 'Mlayer/XOP')],
    #         8050, u'40 × [27 Å Si + 18 Å W] multilayer @ 8.05 keV')

    # mL = rm.Multilayer(mSi, 27*2, mW, 18*2, 40, mSi, 27, 18, 2)
    # compare([(mL, 'xrt')],
    #         [(os.path.join(dataDir, "WSi45_100A40.mlayer.gz"), 'Mlayer/XOP')],
    #         8050, u'Depth graded multilayer \n 40 × [54 Å Si + 36 Å W]'
    #         u' to [27 Å Si + 18 Å W] @ 8.05 keV', '-graded')

    # mL = rm.Multilayer(mSi, 27, mW, 18, 40, mSi, 27*2, 18*2, 2)
    # compare([(mL, 'xrt')],
    #         [(os.path.join(dataDir, "WSi100_45A40.mlayer.gz"), 'Mlayer/XOP')],
    #         8050, u'Depth graded multilayer \n 40 × [27 Å Si + 18 Å W]'
    #         u' to [54 Å Si + 36 Å W] multilayer @ 8.05 keV', '-antigraded')


def compare_multilayer_interdiffusion():
    """A comparison subroutine used in the module test suit."""
#    cl_list = None
    try:
        import pyopencl as cl
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        isOpenCL = True
    except ImportError:
        isOpenCL = False

    if isOpenCL:
        import xrt.backends.raycing.myopencl as mcl
        matCL = mcl.XRT_CL(r'materials.cl')
#        cl_list = matCL.cl_ctx[0], matCL.cl_queue[0], matCL.cl_program[0]
    else:
        matCL = None

    def for_one_material(ml, refs, E, label, flabel=''):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.subplots_adjust(right=0.86)
        ax = fig.add_subplot(111)
        ax.set_xlabel('grazing angle (deg)')
        ax.set_ylabel('reflectivity')
        ax.set_xlim(theta[0], theta[-1])
        fig.suptitle(label, fontsize=14)

        x, R2s = np.loadtxt(refs, unpack=True, skiprows=2, usecols=(0, 1))
        refl = ml.get_amplitude(E, np.sin(np.deg2rad(theta)), ucl=matCL)
        rs, rp = refl[0], refl[1]

# amplitudes:
        p1, = ax.plot(x, R2s, '-k', label='s CXRO')
        p3, = ax.plot(theta, abs(rs)**2, '-r', lw=1)
        p4, = ax.plot(theta, abs(rp)**2, '--r')
        l1 = ax.legend([p3, p4], ['s', 'p'], loc=3)
        ax.legend([p1, p3], ['CXRO-Multilayer', 'xrt'], loc=1)
        ax.add_artist(l1)
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 1])

# phases:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$\phi_s - \phi_p$', color='c')
        phi = np.unwrap(np.angle(rs * rp.conj()))
        p9, = ax2.plot(theta, phi, 'c', lw=1, yunits=math.pi)
        ax2.set_ylim(-0.001, 0.001)
        formatter = mpl.ticker.FormatStrFormatter('%g' + r'$ \pi$')
        ax2.yaxis.set_major_formatter(formatter)
        for tl in ax2.get_yticklabels():
            tl.set_color('c')
        ax2.set_xlim(theta[0], theta[-1])

        ax2.set_zorder(-1)
        ax.patch.set_visible(False)  # hide the 'canvas'

        fname = 'Multilayer' + ml.tLayer.name + ml.bLayer.name
        fig.savefig(fname + flabel)

    dataDir = os.path.join('', 'CXRO-Reflectivities')
    theta = np.linspace(0, 1.6, 1001)  # degrees
    mSi = rm.Material('Si', rho=2.33)
    mW = rm.Material('W', rho=19.3)

    mL = rm.Multilayer(mSi, 17.82, mW, 11.88, 300, mSi, idThickness=0)
    for_one_material(mL, os.path.join(dataDir, "WSi300id0.CXRO.gz"), 24210,
                     u'300 × [17.82 Å Si + 11.88 Å W] multilayer @ 24.21 keV\nInterdiffusion RMS 0 Å',
                     'CXRO_id0')

    mL = rm.Multilayer(mSi, 17.82, mW, 11.88, 300, mSi, idThickness=6)
    for_one_material(mL, os.path.join(dataDir, "WSi300id6.CXRO.gz"), 24210,
                     u'300 × [17.82 Å Si + 11.88 Å W] multilayer @ 24.21 keV\nInterdiffusion RMS 6 Å',
                     'CXRO_id6')


def compare_dTheta():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(mat, ref1, title):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\theta_{B}-\alpha$, deg')
        ax.set_ylabel(r'$\delta\theta$, deg')
        fig.suptitle(title, fontsize=16)
        thetaB = np.degrees(mat.get_Bragg_angle(E[0]))
        calc_dt_new = np.degrees(
            np.abs(mat.get_dtheta(E, np.radians(alpha))))
        p2, = ax.semilogy(
            alpha+thetaB, calc_dt_new, '-b',
            label=r'with curvature')
        calc_dt_reg = np.degrees(
            np.abs(mat.get_dtheta_regular(E, np.radians(alpha))))
        p3, = ax.semilogy(
            alpha+thetaB, calc_dt_reg, '-r',
            label=r'without curvature')
        ax.legend(loc=0)
        ax.set_xlim(0, 90)
        ax.set_ylim(6e-5, 0.3)
        fname = title
        fig.savefig(fname + '.png')

    E = np.ones(500) * 10000.
    alpha = np.linspace(-90., 90., 500)
    mat = rm.CrystalSi(tK=80)
    titleStr = u'Deviation from the Bragg angle, Si(111), {0:d}keV'.format(
        int(E[0]/1000))
    for_one_material(mat, None, titleStr)


def compare_absorption_coeff():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(m):
        rf = os.path.join(dataDir, "{0}_absCoeff.xcrosssec.gz".format(m.name))
        title = u'Absorption in {0}'.format(mat.name)
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel(r'$\mu_0$ (cm$^{-1}$)')
        fig.suptitle(title, fontsize=16)
        x, mu0 = np.loadtxt(rf, unpack=True)
        p1, = ax.loglog(x, mu0, '-k', lw=2, label='XCrossSec')
        calcmu0 = m.get_absorption_coefficient(E)
        p3, = ax.loglog(E, calcmu0, '-r', label='xrt')
        ax.legend(loc=1)
        ax.set_xlim(E[0], E[-1])
        fname = title
        fig.savefig(fname + '.png')

    dataDir = os.path.join('', 'XOP-Reflectivities')
    E = np.logspace(1+math.log10(2.), 4.+math.log10(3.), 500)

#    mat = rm.Material('Be', rho=1.848)
#    for_one_material(mat)
#    mat = rm.Material('Ag', rho=10.50)
#    mat = rm.Material('Ag', rho=10.50, table='Henke')
#    for_one_material(mat)
    mat = rm.Material('Al', rho=2.6989)
    for_one_material(mat)
#    mat = rm.Material('Au', rho=19.3)
#    for_one_material(mat)
#    mat = rm.Material('Ni', rho=8.902)
#    for_one_material(mat)
    E = 30000
    print("abs coeff at {0} keV = {1} 1/cm".format(
        E*1e-3, mat.get_absorption_coefficient(E)))
    print("refr index at {0} keV = {1}".format(
        E*1e-3, mat.get_refractive_index(E)))
#    el = rm.Element('Ag')
#    print("f1, f2", el.get_f1f2(30000))


def compare_transmittivity():
    """A comparison subroutine used in the module test suit."""
    def for_one_material(mat, thickness, ref1, title, sname):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('transmittivity')
        fig.suptitle(title, fontsize=16)
        x, tr = np.loadtxt(ref1, unpack=True)
        p1, = ax.semilogx(x, tr, '-k', lw=2, label='XPower/XOP')
        calcmu0 = mat.get_absorption_coefficient(E)
        transm = np.exp(-calcmu0 * thickness)
        p3, = ax.semilogx(E, transm, '-r', label='xrt')
        ax.legend(loc=2)
        ax.set_xlim(E[0], E[-1])
        fname = 'Transm' + sname
        fig.savefig(fname + '.png')

    dataDir = os.path.join('', 'XOP-Reflectivities')
    E = np.logspace(2.+math.log10(3.), 4.+math.log10(3.), 500)
    matDiamond = rm.Material('C', rho=3.52)
    for_one_material(matDiamond, 60*1e-4,
                     os.path.join(dataDir, "Diamond60mum.xpower.gz"),
                     r'Transmittivity of 60-$\mu$m-thick diamond', 'Diamond')


def run_tests():
    """The body of the module test suit. Uncomment the tests you want."""

#Compare the calculated rocking curves of Si crystals with those calculated by
#XCrystal and XInpro (parts of XOP):
    # compare_rocking_curves('111')
    # compare_rocking_curves('333')
    # compare_rocking_curves('111', t=0.007)  # t is thickness in mm
    # compare_rocking_curves('333', t=0.007)
    # compare_rocking_curves('111', t=0.100)
    # compare_rocking_curves('333', t=0.100)
    # compare_rocking_curves('111', t=0.007, geom='Bragg transmitted')
    # compare_rocking_curves('333', t=0.007, geom='Bragg transmitted')
    # compare_rocking_curves('111', t=0.100, geom='Bragg transmitted')
    # compare_rocking_curves('333', t=0.100, geom='Bragg transmitted')
    # compare_rocking_curves('111', t=0.007, geom='Laue reflected')
    # compare_rocking_curves('333', t=0.007, geom='Laue reflected')
    # compare_rocking_curves('111', t=0.100, geom='Laue reflected')
    # compare_rocking_curves('333', t=0.100, geom='Laue reflected')
    # compare_rocking_curves('111', t=0.007, geom='Laue transmitted')
    # compare_rocking_curves('333', t=0.007, geom='Laue transmitted')
    # compare_rocking_curves('111', t=0.100, geom='Laue transmitted')
    # compare_rocking_curves('333', t=0.100, geom='Laue transmitted')

# Compare rocking curves for bent Si crystals with those calculated by
#XCrystal_Bent (Takagi-Taupin):
#     compare_rocking_curves_bent('111', t=1.0, geom='Laue reflected',
#                           Rcurvmm=np.inf, alphas=[30])
    # compare_rocking_curves_bent('333', t=1.0, geom='Laue reflected',
    #                       Rcurvmm=np.inf, alphas=[30])
#    compare_rocking_curves_bent('111', t=1.0, geom='Laue reflected',
#                           Rcurvmm=50*1e3,
#                           alphas=[-60, -45, -30, -15, 0, 15, 30, 45, 60])
#    compare_rocking_curves_bent('111', t=1.0, geom='Bragg reflected',
#                           Rcurvmm=1e3,
#                           alphas=[0, 4, 8, 12])

#    compare_rocking_curves_bent('333', t=1.0, geom='Laue reflected',
#                           Rcurvmm=50*1e3,
#                           alphas=[-60, -45, -30, -15, 0, 15, 30, 45, 60])
    compare_crystal_bending('111', t=0.3, geom='Bragg reflected',
#                            Rcurvmm=[np.inf, 1e5, 1e4, 1e3])
                            Rcurvmm=[np.inf]) #, 1e4, 1e3])

#    compare_crystal_bending('333', t=1.5, geom='Bragg reflected',
#                            Rcurvmm=[np.inf, 1e5, 1e4, 1e3])

#    compare_crystal_bending('111', t=0.5, geom='Laue reflected',
#                            Rcurvmm=[np.inf, 1e5, 1e4, 1e3],
#                            alphas=[5.])

#check that Bragg transmitted and Laue transmitted give the same results if the
#beam path is equal:
    # beamPath = 0.1  # mm
    # compare_Bragg_Laue('111', beamPath=beamPath)
    # compare_Bragg_Laue('333', beamPath=beamPath)

#Compare the calculated reflectivities of Si, Pt, SiO_2 with those by Xf1f2
#(part of XOP):
    # compare_reflectivity()

    # compare_reflectivity_coated()

#Compare the calculated reflectivities of W slab with those by Mlayer
#(part of XOP):
    # compare_reflectivity_slab()

#Compare the calculated reflectivities of W slab with those by Mlayer
#(part of XOP):
#    compare_multilayer()
    # compare_multilayer_interdiffusion()
    # compare_dTheta()

#Compare the calculated absorption coefficient with that by XCrossSec
#(part of XOP):
    # compare_absorption_coeff()

#Compare the calculated transmittivity with that by XPower
#(part of XOP):
    # compare_transmittivity()

#Play with Si crystal:
    # crystalSi = rm.CrystalSi(hkl=(1, 1, 1), tK=100.)
    # print(2 * crystalSi.get_a()/math.sqrt(3.))  # 2dSi111
    # print('Si111 d-spacing = {0:.6f}'.format(crystalSi.d))
    # print(crystalSi.get_Bragg_offset(8600, 8979))

    # crystalDiamond = rm.CrystalDiamond((1, 1, 1), 2.0592872, elements='C')
    # E = 9000.
    # print(u'Darwin width at E={0:.0f} eV is {1:.5f} µrad for s-polarization'.
    #       format(E, crystalDiamond.get_Darwin_width(E) * 1e6))
    # print(u'Darwin width at E={0:.0f} eV is {1:.5f} µrad for p-polarization'.
    #       format(E, crystalDiamond.get_Darwin_width(E, polarization='p')*1e6))

    plt.show()
    print("finished")


if __name__ == '__main__':
    run_tests()

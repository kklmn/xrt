# -*- coding: utf-8 -*-
r"""
.. _synchrotron-sources:

Synchrotron sources
-------------------

The images below are produced by
``\tests\raycing\test_sources.py`` and by
``\examples\withRaycing\01_SynchrotronSources\synchrotronSources.py``.

Bending magnet
~~~~~~~~~~~~~~

On a transversal screen the image is unlimited horizontally (practically
limited by the front end). The energy distribution is red-shifted for off-plane
photons. The polarization is primarily horizontal. The off-plane radiation has
non-zero projection to the vertical polarization plane.

+----------+---------------------+---------------------+---------------------+
| source   |     total flux      |   horiz. pol. flux  |   vert. pol. flux   |
+==========+=====================+=====================+=====================+
| using WS |    |bmTotalWS|      |    |bmHorizWS|      |     |bmVertWS|      |
+----------+---------------------+---------------------+---------------------+
| internal |                     |                     |                     |
| xrt      |    |bmTotalXRT|     |    |bmHorizXRT|     |     |bmVertXRT|     |
+----------+---------------------+---------------------+---------------------+

.. |bmTotalWS| imagezoom:: _images/3bm_ws1-n-wideE-1TotalFlux.png
.. |bmHorizWS| imagezoom:: _images/3bm_ws1-n-wideE-2horizFlux.png
.. |bmVertWS| imagezoom:: _images/3bm_ws1-n-wideE-3vertFlux.png
   :loc: upper-right-corner
.. |bmTotalXRT| imagezoom:: _images/3bm_xrt1-n-wideE-1TotalFlux.png
.. |bmHorizXRT| imagezoom:: _images/3bm_xrt1-n-wideE-2horizFlux.png
.. |bmVertXRT| imagezoom:: _images/3bm_xrt1-n-wideE-3vertFlux.png
   :loc: upper-right-corner

The off-plane radiation is in fact left and right polarized:

+----------+-----------------------------+
| source   | circular polarization rate  |
+==========+=============================+
| using    |                             |
| WS       |     |bmCircPolRateWS|       |
+----------+-----------------------------+
| internal |                             |
| xrt      |     |bmCircPolRateXRT|      |
+----------+-----------------------------+

.. |bmCircPolRateWS| imagezoom:: _images/3bm_ws1-n-wideE-5CircPolRate.png
.. |bmCircPolRateXRT| imagezoom:: _images/3bm_xrt1-n-wideE-5CircPolRate.png

The horizontal phase space projected to a transversal plane at the origin is
parabolic:

+-------------------------+---------------------+
| zero electron beam size | σ\ :sub:`x` = 49 µm |
+=========================+=====================+
|         |bmPhaseSp0|    |     |bmPhaseSpN|    |
+-------------------------+---------------------+

.. |bmPhaseSp0| imagezoom:: _images/3bm_xrt1-0-wideE-horPhaseSpace.png
.. |bmPhaseSpN| imagezoom:: _images/3bm_xrt1-n-wideE-horPhaseSpace.png

Multipole wiggler
~~~~~~~~~~~~~~~~~

The horizontal image size is determined by the parameter K. The energy
distribution is red-shifted for off-plane photons. The polarization is
primarily horizontal. The off-plane radiation has non-zero projection to the
vertical polarization plane.

+----------+------------------+------------------+------------------+
| source   | total flux       | horiz. pol. flux | vert. pol. flux  |
+==========+==================+==================+==================+
| using    |                  |                  |                  |
| WS       |    |wTotalWS|    |    |wHorizWS|    |    |wVertWS|     |
+----------+------------------+------------------+------------------+
| internal |                  |                  |                  |
| xrt      |    |wTotalXRT|   |    |wHorizXRT|   |    |wVertXRT|    |
+----------+------------------+------------------+------------------+

.. |wTotalWS| imagezoom:: _images/2w_ws1-n-wideE-1TotalFlux.png
.. |wHorizWS| imagezoom:: _images/2w_ws1-n-wideE-2horizFlux.png
.. |wVertWS| imagezoom:: _images/2w_ws1-n-wideE-3vertFlux.png
   :loc: upper-right-corner
.. |wTotalXRT| imagezoom:: _images/2w_xrt1-n-wideE-1TotalFlux.png
.. |wHorizXRT| imagezoom:: _images/2w_xrt1-n-wideE-2horizFlux.png
.. |wVertXRT| imagezoom:: _images/2w_xrt1-n-wideE-3vertFlux.png
   :loc: upper-right-corner

The horizontal longitudinal cross-section reveals a sinusoidal shape of the
source. The horizontal phase space projected to the transversal plane at the
origin has individual branches for each pole.

+-------------------------+-------------------------+
| zero electron beam size |   σ\ :sub:`x` = 49 µm   |
+=========================+=========================+
|          |wYX0|         |          |wYXN|         |
+-------------------------+-------------------------+
|       |wPhaseSp0|       |       |wPhaseSpN|       |
+-------------------------+-------------------------+

.. |wYX0| imagezoom:: _images/2w_xrt1-0-wideE-crossectionYX.png
.. |wYXN| imagezoom:: _images/2w_xrt1-n-wideE-crossectionYX.png
   :loc: upper-right-corner
.. |wPhaseSp0| imagezoom:: _images/2w_xrt1-0-wideE-horPhaseSpace.png
.. |wPhaseSpN| imagezoom:: _images/2w_xrt1-n-wideE-horPhaseSpace.png
   :loc: upper-right-corner

Undulator
~~~~~~~~~

The module :mod:`~tests.raycing.test_sources` has functions for
visualization of the angular and energy distributions of the implemented
sources in 2D and 3D. This is especially useful for undulators because they
have sharp peaks, which requires a proper selection of angular and energy
meshes (not important in the new versions of xrt (>0.9), where there are no
angular and energy meshes and the intensity is calculated *for each* ray).

.. imagezoom:: _images/I0_x'E_mode4-1-und-urgent.png
.. imagezoom:: _images/I0_z'E_mode4-1-und-urgent.png
   :loc: upper-right-corner

.. imagezoom:: _images/IpPol

The ray traced images of an undulator source are feature-rich. The polarization
is primarily horizontal. The off-plane radiation has non-zero projection to the
vertical polarization plane.

+--------+----------------+----------------+----------------+----------------+
| source |   total flux   | hor. pol. flux | ver. pol. flux |  deg. of pol.  |
+========+================+================+================+================+
| using  |                |                |                |                |
| Urgent | |uTotalUr|     |   |uHorizUr|   |    |uVertUr|   | |uDegPolUr|    |
+--------+----------------+----------------+----------------+----------------+
|internal|                |                |                |                |
|xrt     | |uTotalXRT|    |  |uHorizXRT|   |    |uVertXRT|  | |uDegPolXRT|   |
+--------+----------------+----------------+----------------+----------------+

.. |uTotalUr| imagezoom:: _images/1u_urgent3-n-monoE-1TotalFlux.png
.. |uHorizUr| imagezoom:: _images/1u_urgent3-n-monoE-2horizFlux.png
.. |uVertUr| imagezoom:: _images/1u_urgent3-n-monoE-3vertFlux.png
   :loc: upper-right-corner
.. |uDegPolUr| imagezoom:: _images/1u_urgent3-n-monoE-4DegPol.png
   :loc: upper-right-corner
.. |uTotalXRT| imagezoom:: _images/1u_xrt3-n-monoE-1TotalFlux-Espread.png
.. |uHorizXRT| imagezoom:: _images/1u_xrt3-n-monoE-2horizFlux-Espread.png
.. |uVertXRT| imagezoom:: _images/1u_xrt3-n-monoE-3vertFlux-Espread.png
   :loc: upper-right-corner
.. |uDegPolXRT| imagezoom:: _images/1u_xrt3-n-monoE-4DegPol-Espread.png
   :loc: upper-right-corner

Elliptical undulator
~~~~~~~~~~~~~~~~~~~~

An elliptical undulator gives circular images with a higher circular
polarization rate in the inner rings:

+----------+--------------------+--------------------+--------------------+
| source   |    total flux      |   hor. pol. flux   |   ver. pol. flux   |
+==========+====================+====================+====================+
| using    |                    |                    |                    |
| Urgent   |     |euTotalUr|    |     |euHorizUr|    |     |euVertUr|     |
+----------+--------------------+--------------------+--------------------+
| internal |                    |                    |                    |
| xrt      |    |euTotalXRT|    |    |euHorizXRT|    |     |euVertXRT|    |
+----------+--------------------+--------------------+--------------------+

.. |euTotalUr| imagezoom:: _images/4eu_urgent3-n-monoE-1TotalFlux.png
.. |euHorizUr| imagezoom:: _images/4eu_urgent3-n-monoE-2horizFlux.png
.. |euVertUr| imagezoom:: _images/4eu_urgent3-n-monoE-3vertFlux.png
   :loc: upper-right-corner
.. |euTotalXRT| imagezoom:: _images/4eu_xrt3-n-monoE-1TotalFlux.png
.. |euHorizXRT| imagezoom:: _images/4eu_xrt3-n-monoE-2horizFlux.png
.. |euVertXRT| imagezoom:: _images/4eu_xrt3-n-monoE-3vertFlux.png
   :loc: upper-right-corner

+----------+----------------------------+----------------------------+
| source   |     deg. of pol.           | circular polarization rate |
+==========+============================+============================+
| using    |                            |                            |
| Urgent   |       |euDegPolUr|         |     |euCircPolRateUr|      |
+----------+----------------------------+----------------------------+
| internal |                            |                            |
| xrt      |        |euDegPolXRT|       |    |euCircPolRateXRT|      |
+----------+----------------------------+----------------------------+

.. |euDegPolUr| imagezoom:: _images/4eu_urgent3-n-monoE-4DegPol.png
.. |euCircPolRateUr| imagezoom:: _images/4eu_urgent3-n-monoE-5CircPolRate.png
   :loc: upper-right-corner
.. |euDegPolXRT| imagezoom:: _images/4eu_xrt3-n-monoE-4DegPol.png
.. |euCircPolRateXRT| imagezoom:: _images/4eu_xrt3-n-monoE-5CircPolRate.png
   :loc: upper-right-corner

.. _undulator_custom:

Custom field undulator
~~~~~~~~~~~~~~~~~~~~~~

A custom magnetic field can be specified by an Excel file or a column text
file. The example below is based on a table supplied by Hamed Tarawneh
[Tarawneh]_. The idea of introducing quasi-periodicity is to shift the n-th
harmonics down in energy relative to the exact n-fold multiple of the 1st
harmonic energy. This trick eliminates higher *monochromator* harmonics that
are situated at the exact n-fold energies, which is a safer solution compared
to a gas absorption filter.

Compare the harmonic energies (half-maximum position at the higher energy side)
of the 3rd harmonic with the triple energy of the 1st harmonic.

.. [Tarawneh] Quasi-periodic undulator field for ARPES beamline at
   MAX IV 1.5 GeV ring, unpublished.

.. note::

    The definition of xyz coordinate system differs for the tabulated field and
    for xrt screens: z is along the beam direction in the tabulation and as a
    vertical axis in xrt.

+--------------------+--------------------------+--------------------------+
|                    |         periodic         |       quasi-periodic     |
+====================+==========================+==========================+
| tabulated field    |       |EPU_field|        |      |QEPU_field|        |
+--------------------+--------------------------+--------------------------+
| trajectory         |                          |                          |
| top view           |        |EPU_traj|        |       |QEPU_traj|        |
+--------------------+--------------------------+--------------------------+
| wide band          |                          |                          |
| image and spectrum |        |EPU_wide|        |       |QEPU_wide|        |
+--------------------+--------------------------+--------------------------+
| 1st harmonic       |                          |                          |
| image and spectrum |        |EPU_1sth|        |       |QEPU_1sth|        |
+--------------------+--------------------------+--------------------------+
| 3rd harmonic       |                          |                          |
| image and spectrum |        |EPU_3rdh|        |       |QEPU_3rdh|        |
+--------------------+--------------------------+--------------------------+

.. |EPU_field| imagezoom:: _images/1-EPU_HP_field.png
.. |EPU_traj| imagezoom:: _images/1-EPU_HP_mode-x_average.png
.. |EPU_wide| imagezoom:: _images/1EPU_HP_mode-1-band1totalFlux.png
.. |EPU_1sth| imagezoom:: _images/1EPU_HP_mode-2-1stHarmonic1totalFlux.png
.. |EPU_3rdh| imagezoom:: _images/1EPU_HP_mode-6-3rdHarmonic1totalFlux.png
.. |QEPU_field| imagezoom:: _images/2-QEPU_HP_field.png
   :loc: upper-right-corner
.. |QEPU_traj| imagezoom:: _images/2-QEPU_HP_mode-x_average.png
   :loc: upper-right-corner
.. |QEPU_wide| imagezoom:: _images/2QEPU_HP_mode-1-band1totalFlux.png
   :loc: upper-right-corner
.. |QEPU_1sth| imagezoom:: _images/2QEPU_HP_mode-2-1stHarmonic1totalFlux.png
   :loc: upper-right-corner
.. |QEPU_3rdh| imagezoom:: _images/2QEPU_HP_mode-6-3rdHarmonic1totalFlux.png
   :loc: upper-right-corner

For validation, our calculations are compared here with those by Spectra for a
particular case — quasi-periodic undulator defined by the same tabulated field,
the 3rd harmonic, at E=20.5 eV. Notice again that Spectra provides either a
spectrum *or* a transverse image while xrt can combine both by using colors and
brightness. Notice also that on the following pictures the p-polarized flux is
only ~3% of the total flux.

+------------+------------+
|  SPECTRA   |    xrt     |
+============+============+
|  |CFspeT|  |  |CFxrtT|  |
+------------+------------+
|  |CFspeP|  |  |CFxrtP|  |
+------------+------------+

.. |CFspeT| imagezoom:: _images/spectra-custom.png
.. |CFspeP| imagezoom:: _images/spectra-custom_p.png
   :loc: lower-left-corner
.. |CFxrtT| imagezoom:: _images/2QEPU_mono3rdHarmonic.png
.. |CFxrtP| imagezoom:: _images/2QEPU_mono3rdHarmonic_p.png
   :loc: lower-left-corner

"""

"""
.. _THz:

Terahertz undulator
~~~~~~~~~~~~~~~~~~~

This is a computationally difficult example that is hardly possible to
calculate with other commonly used codes. To achieve good convergence in a
reasonable calculation time, we have introduced a new undulator parameter
*gIntervals* that controls the underlying Gauss-Legendre mesh. The user may
want to modify it in extreme cases (wigglers, near field, wide angles etc.).


+------------------+-----------------+
| horiz. pol. flux | vert. pol. flux |
+==================+=================+
|     |uTHzH|      |     |uTHzV|     |
+------------------+-----------------+

.. |uTHzH| image:: _images/2THzU_xrt1-n-narrowE-2horizFlux-U_code.png
   :scale: 50 %
.. |uTHzV| image:: _images/2THzU_xrt1-n-narrowE-3vertFlux-U_code.png
   :scale: 50 %
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "08 Mar 2016"

import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import time
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

# one of 'u', 'w', 'bm', 'eu', 'wu':
sourceType = 'u'
# one of 'mono', '1harmonic', 'smaller', 'wide'
energyRange = '1harmonic'
#energyRange = 'mono'
#energyRange = 'wide'
what = 'rays'
#what = 'wave'  # only for rs.Undulator
#suffix = '_fullLength_emittance_NF'
suffix = ''
isInternalSource = True  # xrt source or (Urgent or WS)
limitsFSM0X = 'symmetric'
limitsFSM0Z = 'symmetric'
E0 = 6900  # eV
R0 = 25000.  # Distance to the screen [mm]
bins = 256  # Number of bins in the plot histogram
ppb = 1  # Number of pixels per histogram bin

if sourceType == 'u':
    whose = '_xrt' if isInternalSource else '_urgent'
    pprefix = '1'+sourceType+whose
    Source = rs.Undulator if isInternalSource else rs.UndulatorUrgent
    Kmax = 1.92
    kwargs = dict(
        eE=3., eI=0.5,  # Parameters of the synchrotron ring [GeV], [Ampere]
        #eEspread=0.001,  # Energy spread of the electrons in the ring
        period=30., n=40,  # Parameters of the undulator, period in [mm]
        K=1.45,  # Deflection parameter (ignored if targetE is not None)
        # targetE=[6940, 5, False],  # [energy [eV], harmonic]
        eSigmaX=48.65, eSigmaZ=6.197,  # Size of the electron beam [mkm]
        # customField=0.0,  # Longitudinal magnetic field. If not None,
        # trajectory of the electron is calculated numerically.
        # eSigmaX=0., eSigmaZ=0.,  # Zero size electron beam
        # uniformRayDensity=True,
        # filamentBeam=True,  # Single wavefront
        # R0=R0,   # Near Field.
        # gIntervals=5,  # Number of the integration intervals. Should be
        # increased for the near field and custom magnetic field cases.
        # gp=1e-2,  # Precision of the integration.
        # targetOpenCL=None,
        #taper = [0 ,10],
        #eEpsilonX=0.0, eEpsilonZ=0.0)  # Emittance [nmrad]
        eEpsilonX=0.263, eEpsilonZ=0.008)  # Emittance [nmrad]
    xlimits = [-10, 10]  # Horizontal limits of the plot [mm]
    zlimits = [-10, 10]  # Vertical limits of the plot [mm]
    xlimitsZoom = [-2, 2]
    zlimitsZoom = [-2, 2]
    xPrimelimits = [-0.3, 0.3]  # Angular limits of the plot [mrad]
    if isInternalSource:
        kwargs['xPrimeMaxAutoReduce'] = False
        kwargs['zPrimeMaxAutoReduce'] = False
    else:
        kwargs['icalc'] = 3
elif sourceType == 'w':
    whose = '_xrt' if isInternalSource else '_ws'
    pprefix = '3'+sourceType+whose
    Source = rs.Wiggler if isInternalSource else rs.WigglerWS
    kwargs = dict(period=80., K=13., n=10, eE=3.)
    xlimits = [-40, 40]
    zlimits = [-20, 20]
    xlimitsZoom = [-8, 8]
    zlimitsZoom = [-4, 4]
    xPrimelimits = [-2.3, 2.3]
    limitsFSM0X = [-1000, 1000]
    limitsFSM0Z = [-100, 100]
elif sourceType == 'bm':
    whose = '_xrt' if isInternalSource else '_ws'
    pprefix = '4'+sourceType+whose
    Source = rs.BendingMagnet if isInternalSource else rs.BendingMagnetWS
    kwargs = dict(B0=1.7, eI=0.1, eE=3.)
#    kwargs['uniformRayDensity'] = True
    xlimits = [-40, 40]
    zlimits = [-20, 20]
    xlimitsZoom = [-8, 8]
    zlimitsZoom = [-4, 4]
    xPrimelimits = 'symmetric'
    limitsFSM0X = [-500, 500]
    limitsFSM0Z = [-20, 20]
elif sourceType == 'eu':
    whose = '_xrt' if isInternalSource else '_urgent'
    pprefix = '5'+sourceType+whose
    Source = rs.Undulator if isInternalSource else rs.UndulatorUrgent
    kwargs = dict(
        period=30., Ky=1.45, Kx=1.45, n=40,
        eE=3., eI=0.5,
        eSigmaX=48.65, eSigmaZ=6.197,
        eEpsilonX=0.263, eEpsilonZ=0.008)
    if isInternalSource:
        kwargs['phaseDeg'] = 90
        kwargs['xPrimeMaxAutoReduce'] = False
        kwargs['zPrimeMaxAutoReduce'] = False
    xlimits = [-12, 12]
    zlimits = [-12, 12]
    xlimitsZoom = [-5, 5]
    zlimitsZoom = [-5, 5]
    xPrimelimits = [-0.55, 0.55]
elif sourceType == 'wu':  # wiggler by undulator code
    whose = '_xrt' if isInternalSource else '_ws'
    pprefix = '6'+sourceType+whose
    Source = rs.Undulator if isInternalSource else rs.WigglerWS
    kwargs = dict(period=80., K=13., n=10, eE=3., eI=0.5)
    xlimits = [-40, 40]
    zlimits = [-20, 20]
    xlimitsZoom = [-5, 5]
    zlimitsZoom = [-5, 5]
#    kwargs['uniformRayDensity'] = True
#    kwargs['gIntervals'] = 99
    kwargs['zPrimeMaxAutoReduce'] = False
    xPrimelimits = [-2.3, 2.3]
    limitsFSM0X = [-1000, 1000]
    limitsFSM0Z = [-100, 100]
else:
    raise ValueError('Unknown source type!')

if False:  # force zero source size:
    kwargs['eSigmaX'] = 0
    kwargs['eSigmaZ'] = 0
    kwargs['eEpsilonX'] = 0
    kwargs['eEpsilonZ'] = 0
    kwargs['eEspread'] = 0

    eEpsilonC = '0'
else:
    eEpsilonC = 'n'

kwargs['xPrimeMax'] = xlimits[-1] / R0 * 1e3
kwargs['zPrimeMax'] = zlimits[-1] / R0 * 1e3

prefix = pprefix+'-{0}-{1}E-'.format(eEpsilonC, energyRange)
if energyRange == 'mono':
    eMinRays, eMaxRays = E0-1, E0+1
    eUnit = 'eV'
elif energyRange == '1harmonic':
    eMinRays, eMaxRays = E0-300, E0+300
    eUnit = 'eV'
elif energyRange == 'smaller':
    eMinRays, eMaxRays = 1500, 7500
    eUnit = 'keV'
elif energyRange == 'wide':
    eMinRays, eMaxRays = 1500, 37500
    eUnit = 'keV'

kwargs['eMin'] = eMinRays
kwargs['eMax'] = eMaxRays
if Source == rs.UndulatorUrgent:
    kwargs['processes'] = 'half'


def build_beamline(nrays=1e5):
    beamLine = raycing.BeamLine()
    beamLine.source = Source(
        beamLine, eN=1000, nx=40, nz=20, nrays=nrays, **kwargs)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0', (0, 0, 0))
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', (0, R0, 0))
    return beamLine


def run_process(beamLine):
    startTime = time.time()
    kw = {}
    if 'mono' in prefix and isInternalSource:
        kw['fixedEnergy'] = E0

    if 'wave' in what and Source == rs.Undulator:
        wave1 = beamLine.fsm1.prepare_wave(
            beamLine.source, beamLine.fsmExpX, beamLine.fsmExpZ)
        wave1zoom = beamLine.fsm1.prepare_wave(
            beamLine.source, beamLine.fsmExpXzoom, beamLine.fsmExpZzoom)

        kw['wave'] = wave1
        beamSource = beamLine.source.shine(**kw)
        beamFSM0 = beamLine.fsm0.expose(beamSource)
        kw['wave'] = wave1zoom
        beamSourceZ = beamLine.source.shine(**kw)
        outDict = {'beamSource': beamSourceZ,
                   'beamFSM0': beamFSM0,
                   'beamFSM1': wave1, 'beamFSM1zoom': wave1zoom}
    else:
        beamSource = beamLine.source.shine(**kw)
        beamFSM0 = beamLine.fsm0.expose(beamSource)
        beamFSM1 = beamLine.fsm1.expose(beamSource)
        outDict = {'beamSource': beamSource,
                   'beamFSM0': beamFSM0,
                   'beamFSM1': beamFSM1, 'beamFSM1zoom': beamFSM1}
    print('shine time = {0}s'.format(time.time() - startTime))
    return outDict

rr.run_process = run_process


def define_plots(beamLine):
    plots = []
    plotsE = []

    xaxis = xrtp.XYCAxis(r'$y$', 'mm', bins=256)
    yaxis = xrtp.XYCAxis(r'$x$', '$\mu$m', limits='symmetric')
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None)
    plot = xrtp.XYCPlot(
        'beamSource', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='YX source cross-section')
    plot.saveName = prefix + 'crossectionYX' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$y$', 'mm', bins=256)
    yaxis = xrtp.XYCAxis(r'$z$', '$\mu$m', limits='symmetric')
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None)
    plot = xrtp.XYCPlot(
        'beamSource', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='YZ source cross-section')
    plot.saveName = prefix + 'crossectionYZ' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', '$\mu$m', limits=limitsFSM0X,
                         bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', '$\mu$m', limits=limitsFSM0Z,
                         bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamSource', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='image at 0')
    plot.saveName = prefix + 'fsm0' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)
    plotsE.append(plot)

    beam = 'beamFSM0'
    xaxis = xrtp.XYCAxis(r'$x$', '$\mu$m', limits=limitsFSM0X)
    yaxis = xrtp.XYCAxis(r"$x'$", 'mrad', limits=xPrimelimits)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None)
    plot = xrtp.XYCPlot(
        beam, (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='horizontal phase space')
    plot.saveName = prefix + 'horPhaseSpace' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', '$\mu$m', limits=[-80, 80])
    yaxis = xrtp.XYCAxis(r"$x'$", 'mrad', limits=[-0.15, 0.15])
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None)
    plot = xrtp.XYCPlot(
        beam, (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='horizontal phase space zoomed')
    plot.saveName = prefix + 'horPhaseSpaceZoom' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='total flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '1TotalFlux' + suffix + '.png'
    #plot.persistentName = prefix + '1TotalFlux' + suffix + '.mat'
    plots.append(plot)
    plotsE.append(plot)
    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        aspect='auto', title='total flux zoom')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '1TotalFluxZoom' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)
    ax = plot.xaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpXzoom = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
    ax = plot.yaxis
    edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins+1)
    beamLine.fsmExpZzoom = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='s', aspect='auto', title='horizontal polarization flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '2horizFlux' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='s', aspect='auto', title='horizontal polarization flux zoom')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '2horizFluxZoom' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='p', aspect='auto', title='vertical polarization flux')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '3vertFlux' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    caxis = xrtp.XYCAxis('energy', eUnit, fwhmFormatStr=None,
                         bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis, caxis=caxis,
        fluxKind='p', aspect='auto', title='vertical polarization flux zoom')
    plot.caxis.fwhmFormatStr = None
    plot.saveName = prefix + '3vertFluxZoom' + suffix + '.png'
    plots.append(plot)
    plotsE.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis,
        caxis=xrtp.XYCAxis('degree of polarization', '',
                           data=raycing.get_polarization_degree,
                           limits=[0.9, 1.01], bins=bins, ppb=ppb),
        aspect='auto', title='degree of polarization')
    plot.saveName = prefix + '4DegPol' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis,
        caxis=xrtp.XYCAxis('degree of polarization', '',
                           data=raycing.get_polarization_degree,
                           limits=[0.9, 1.01], bins=bins, ppb=ppb),
        aspect='auto', title='degree of polarization zoom')
    plot.saveName = prefix + '4DegPolZoom' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimits, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimits, bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1', (1,), xaxis=xaxis, yaxis=yaxis,
        caxis=xrtp.XYCAxis('circular polarization rate', '',
                           data=raycing.get_circular_polarization_rate,
                           limits=[-1, 1], bins=bins, ppb=ppb),
        aspect='auto', title='circular polarization rate')
    plot.saveName = prefix + '5CircPolRate' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)

    xaxis = xrtp.XYCAxis(r'$x$', 'mm', limits=xlimitsZoom, bins=bins, ppb=ppb)
    yaxis = xrtp.XYCAxis(r'$z$', 'mm', limits=zlimitsZoom, bins=bins, ppb=ppb)
    plot = xrtp.XYCPlot(
        'beamFSM1zoom', (1,), xaxis=xaxis, yaxis=yaxis,
        caxis=xrtp.XYCAxis('circular polarization rate', '',
                           data=raycing.get_circular_polarization_rate,
                           limits=[-1, 1], bins=bins, ppb=ppb),
        aspect='auto', title='circular polarization rate zoom')
    plot.saveName = prefix + '5CircPolRateZoom' + suffix + '.png'
    plot.caxis.fwhmFormatStr = None
    plots.append(plot)

    for plot in plotsE:
        f = plot.caxis.factor
        plot.caxis.limits = eMinRays*f, eMaxRays*f
    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.fluxFormatStr = '%.2p'
    return plots, plotsE


def afterScript(*plots):
    import os
    import pickle
    plot = plots[-1]
    flux = [plot.intensity, plot.nRaysAll, plot.nRaysSeeded]
    cwd = os.getcwd()
    pickleName = os.path.join(cwd, prefix+'.pickle')
    with open(pickleName, 'wb') as f:
        pickle.dump((flux, plot.caxis.binEdges, plot.caxis.total1D), f,
                    protocol=2)


def main():
    beamLine = build_beamline()
    plots, plotsE = define_plots(beamLine)
    xrtr.run_ray_tracing(plots, repeats=1,
                         # afterScript=afterScript, afterScriptArgs=plots,
                         beamLine=beamLine)

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Package :mod:`~xrt.backends.raycing` provides the internal backend of xrt. It
defines beam sources in the module :mod:`~xrt.backends.raycing.sources`,
rectangular and round apertures in :mod:`~xrt.backends.raycing.apertures`,
optical elements in :mod:`~xrt.backends.raycing.oes`, material properties
(essentially reflectivity, transmittivity and absorption coefficient) for
interfaces and crystals in :mod:`~xrt.backends.raycing.materials` and screens
in :mod:`~xrt.backends.raycing.screens`.

Coordinate systems
------------------

.. imagezoom:: _images/axes.png
   :align: right

The following coordinate systems are considered (always right-handed):

1) *The global coordinate system*. It is arbitrary (user-defined) with one
   requirement driven by code simplification: Z-axis is vertical. For example,
   the system origin of Alba synchrotron is in the center of the ring at the
   ground level with Y-axis northward, Z upright and the units in mm.

   .. note::
       The positions of all optical elements, sources, screens etc. are given
       in the global coordinate system. This feature simplifies the beamline
       alignment when 3D CAD models are available.

2) *The local systems*.

   a) *of the beamline*. The local Y direction (the direction of the source)
      is determined by *azimuth* parameter of
      :class:`~xrt.backends.raycing.BeamLine` -- the angle measured cw from the
      global Y axis. The local beamline Z is also vertical and upward. The
      local beamline X is to the right. At *azimuth* = 0 the global system and
      the local beamline system are parallel to each other. In most of the
      supplied examples the global system and the local beamline system
      coincide.

   b) *of an optical element*. The origin is on the optical surface. Z is
      out-of-surface. At pitch, roll and yaw all zeros the local oe system
      and the local beamline system are parallel to each other.

      .. note::
          Pitch, roll and yaw rotations (correspondingly: Rx, Ry and Rz) are
          defined relative to the local axes of the optical element. The local
          axes rotate together with the optical element!

      .. note::
          The rotations are done in the following default sequence: yaw, roll,
          pitch. It can be changed by the user for any particular optical
          element. Sometimes it is necessary to define misalignment angles in
          addition to the positional angles. Because rotations do not commute,
          an extra set of angles may become unavoidable, which are applied
          after the positional rotations.
          See :class:`~xrt.backends.raycing.oes.OE`.

      The user-supplied functions for the surface height (z) and the normal as
      functions of (x, y) are defined in the local oe system.

   c) *of other beamline elements: sources, apertures, screens*. Z is upward
      and Y is along the beam line. The origin is given by the user. Usually it
      is on the original beam line.

xrt sequentially transforms beams (instances of
:class:`~xrt.backends.raycing.sources.Beam`) -- containers of arrays which hold
beam properties for each ray. Geometrical beam properties such as *x, y, z*
(ray origins) and *a, b, c* (directional cosines) as well as polarization
characteristics depend on the above coordinate systems. Therefore, beams are
usually represented by two different objects: one in the global and one in a
local system.

Units
-----

For the internal calculations, lengths are assumed to be in mm, although for
reflection geometries and simple Bragg cases (thick crystals) this convention
is not used. Angles are unitless (radians). Energy is in eV.

For plotting, the user may select units and conversion factors. The latter are
usually automatically deduced from the units.

Beam categories
---------------

xrt discriminates rays by several categories:

a) ``good``: reflected within the working optical surface;
b) ``out``: reflected outside of the working optical surface, i.e. outside of
   a metal stripe on a mirror;
c) ``over``: propagated over the surface without intersection;
d) ``dead``: arrived below the optical surface and thus absorbed by the OE.

This distinction simplifies the adjustment of entrance and exit slits. The
user supplies `physical` and `optical` limits, where the latter is used to
define the ``out`` category (for rays between `physical` and `optical` limits).
An alarm is triggered if the fraction of dead rays exceeds a specified level.

.. _scriptingRaycing:

Scripting in python
-------------------

The user of :mod:`~xrt.backends.raycing` must do the following:

1) Instantiate class :class:`~xrt.backends.raycing.BeamLine` and fill it with
   sources, optical elements, screens etc.
2) Create a module-level function that returns a dictionary of beams -- the
   instances of :class:`~xrt.backends.raycing.sources.Beam`. Assign this
   function to the module variable `xrt.backends.raycing.run.run_process`.
   The beams should be obtained by the methods shine() of a source, expose() of
   a screen, reflect() or multiple_reflect() of an optical element, propagate()
   of an aperture.
3) Use the keys in this dictionary for creating the plots (instances of
   :class:`~xrt.plotter.XYCPlot`). Note that at the time of instantiation the
   plots are just empty placeholders for the future 2D and 1D histograms.
4) Run :func:`~xrt.runner.run_ray_tracing()` function for the created plots.

Additionally, the user may define a generator that will run a loop of ray
tracing jobs for changing geometry settings (mimics a real scan) or for
different material properties etc. The generator should modify the beamline
elements and output file names of the plots before *yield*. After the *yield*
the plots are ready and the generator may use their fields, e.g. *intensity* or
*dE* or *dy* or others to prepare a scan plot. Typically, this sequence is
contained within a loop; after the loop the user may prepare the final scan
plot using matplotlib functionality. The generator is passed to
:func:`~xrt.runner.run_ray_tracing()` as a parameter.

Consider an example of a generator::

    def energy_scan(beamLine, plots, energies):
        flux = np.zeros_like(energies)
        for ie, e in enumerate(energies):
            print(f'energy {e:.1f} eV, {ie+1} of {len(energies)}')
            beamLine.fixedEnergy = e
            beamLine.source.eMin = e - 0.5  # defines 1 eV energy band
            beamLine.source.eMax = e + 0.5
            for plot in plots:
                plot.saveName = [plot.baseName + f'-{ie}-{e:.1f}eV.png']

            yield
            # now all plots for this scan point are ready
            flux[ie] = plot.flux

        # now the whole scan is complete
        integratedFlux = np.trapz(flux, energies)
        print(f'total flux = {integratedFlux:.3g} ph/s')

        with open("ray_tracing_c.pickle", 'wb') as f:
            pickle.dump([energies, flux, integratedFlux], f)

        plt.plot(energies, flux)
        plt.show()

... and an example of passing this generator to
:func:`~xrt.runner.run_ray_tracing()`::

    def ray_study(nrays, repeats):
        beamLine = build_beamline(nrays)
        plots = define_plots(beamLine)
        energies = np.linspace(11800, 12600, 401)
        xrtr.run_ray_tracing(
            plots, repeats=repeats, beamLine=beamLine,
            generator=energy_scan, generatorArgs=[beamLine, plots, energies])

Find more generators in the supplied examples.
"""

__module__ = "raycing"
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "20 Nov 2025"

import sys
import types
import numpy as np
import copy
import json
import inspect
import uuid  # analysis:ignore
from itertools import islice  # analysis:ignore
from collections import OrderedDict

if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec

from .singletons import (
    colorPrint, colors, unicode, basestring, is_sequence, _VERBOSITY_)

from .physconsts import SIE0, CH  # analysis:ignore

from ._rotate import (
    rotate_x, rotate_y, rotate_z, rotate_beam, rotate_xyz, rotate_point)

from ._beam_props import (
    get_energy, get_x, get_y, get_z, get_s, get_phi, get_r, get_a, get_b,
    get_xprime, get_zprime, get_xzprime, get_path, get_order,
    get_reflection_number, get_elevation_d,
    get_elevation_x, get_elevation_y, get_elevation_z,
    get_Es_amp, get_Ep_amp, get_Es_phase, get_Ep_phase,
    get_polarization_degree, get_ratio_ellipse_axes,
    get_circular_polarization_rate, get_polarization_psi, get_phase_shift,
    get_incidence_angle, get_theta)

from ._sets_units import (
    allBeamFields, orientationArgSet, shapeArgSet, derivedArgSet,
    renderOnlyArgSet, compoundArgs, dependentArgs, diagnosticArgs, allUnitsAng,
    allUnitsAngStr, allUnitsLen, allUnitsLenStr, allUnitsEnergy,
    allUnitsEnergyStr, allUnitsEmittance, allUnitsEmittanceStr,
    allUnitsCurrent, allUnitsCurrentStr, lengthUnitParams)

from ._epics import to_valid_var_name, EpicsDevice

from ._named_arrays import NamedArrayFactory, Center, Limits, Opening, Image2D

from ._flow_utils import (
    auto_units_angle, append_to_flow, append_to_flow_decorator, set_name,
    vec_to_quat, multiply_quats, quat_vec_rotate, get_init_val, get_params,
    parametrize, create_paramdict_oe, create_paramdict_mat, get_obj_str,
    get_init_kwargs, is_valid_uuid, run_process_from_file, build_hist)
from ._flow import propagationProcess, MessageHandler

from .beamline import (
    distance_xy, distance_xyz, global_to_virgin_local, virgin_local_to_global,
    xyz_from_xz, is_auto_align_required, AlignmentBeam, BeamLine)

_DEBUG_ = True  # If False, exceptions inside the module are ignored

stateGood, stateOut, stateOver = 1, 2, 3

zEps = 1e-12  # mm: target accuracy in z while searching for intersection
misalignmentTolerated = 0.1  # for automatic checking of oe center position
accuracyInPosition = 0.1  # accuracy for positioning of oe
dt = 1e-5  # mm: margin around OE within which the intersection is searched
ds = 0.  # mm: margin used in multiple reflections
nrays = 100000
maxIteration = 100  # max number of iterations while searching for intersection
maxHalfSizeOfOE = 1000.
maxDepthOfOE = 100.
# maxZDeviationAtOE = 100.

# colors of the rays in a 0-10 range (red-violet)
hueGood = 3.
hueOut = 8.
hueOver = 1.6
hueDead = 0.2
hueMin = 0.
hueMax = 10.

targetOpenCL = 'auto'
precisionOpenCL = 'auto'
# targetOpenCL = (0, 0)
# precisionOpenCL = 'float32'

msg_start = {"command": "start"}
msg_stop = {"command": "stop"}
msg_exit = {"command": "exit"}


def center_property():
    def getter(self):
        return self._center if self._centerVal is None else self._centerVal

    def setter(self, center):
        centerInit = copy.deepcopy(center)
        if isinstance(center, str):
            center = [x.strip().lower() for x in center.strip('[]').split(",")]
            tmp = []
            for value in center:
                try:
                    value = float(value)
                except ValueError:
                    pass
                tmp.append(value)

        if any([isinstance(x, str) for x in center]):
            self._centerInit = centerInit
            self._centerVal = None
#            self._center = copy.deepcopy(center)
        else:
            self._centerVal = Center(center)

        self._center = copy.deepcopy(center)

    return property(getter, setter)


def check_alarm(self, incoming, beam):
    """Appends an alarm string to the list of beamline alarms if the alarm
    condition is fulfilled."""
    incomingSum = incoming.sum()
    try:
        objName = self.name
    except AttributeError:
        objName = self.__class__.__name__
    if incomingSum > 0:
        badState = beam.state == self.lostNum
        badSum = badState.sum()
        badFlux = (beam.Jss[badState] + beam.Jpp[badState]).sum()
        allFlux = (beam.Jss + beam.Jpp).sum()
        ratio = float(badSum)/incomingSum
        ratioFlux = badFlux / allFlux
        if ratio > self.alarmLevel:
            alarmStr = ('{0}{1} absorbes {2:.2%} of rays or {3:.2%} of flux ' +
                        'at {4:.0%} alarm level!').format(
                'Alarm! ', objName, ratio, ratioFlux, self.alarmLevel)
            self.bl.alarms.append(alarmStr)
    else:
        self.bl.alarms.append('no incident rays to {0}!'.format(objName))


def get_output(plot, beamsReturnedBy_run_process):
    """Used by :mod:`multipro` for creating images of *plot* - instance of
    :class:`XYCPlot`. *beamsReturnedBy_run_process* is a dictionary of
    :class:`Beam` instances returned by user-defined :func:`run_process`.

    :func:`get_output` creates an indexing array corresponding to the requested
    properties of rays in *plot*. It also calculates the number of rays with
    various properties defined in `raycing` backend.
     """
    beam = beamsReturnedBy_run_process[plot.beam]
    if plot.beamState is None:
        beamState = beam.state
    else:
        beamState = beamsReturnedBy_run_process[plot.beamState].state
    nrays = len(beam.x)

    locAlive = (beamState > 0).sum()
    part = np.zeros(nrays, dtype=bool)
    locGood = 0
    locOut = 0
    locOver = 0
    locDead = 0
    for rayFlag in plot.rayFlag:
        locPart = beamState == rayFlag
        if rayFlag == 1:
            locGood = locPart.sum()
        if rayFlag == 2:
            locOut = locPart.sum()
        if rayFlag == 3:
            locOver = locPart.sum()
        if rayFlag < 0:
            locDead += locPart.sum()
        part = part | locPart
    if hasattr(beam, 'accepted'):
        locAccepted = beam.accepted
        locAcceptedE = beam.acceptedE
        locSeeded = beam.seeded
        locSeededI = beam.seededI
    else:
        locAccepted = 0
        locAcceptedE = 0
        locSeeded = 0
        locSeededI = 0

    if hasattr(beam, 'displayAsAbsorbedPower'):
        plot.displayAsAbsorbedPower = True

    if isinstance(plot.xaxis.data, types.FunctionType):
        x = plot.xaxis.data(beam) * plot.xaxis.factor
    elif isinstance(plot.xaxis.data, np.ndarray):
        x = plot.xaxis.data * plot.xaxis.factor
    else:
        raise ValueError('cannot find x data for plot {0}'.format(plot.beam))

    if isinstance(plot.yaxis.data, types.FunctionType):
        y = plot.yaxis.data(beam) * plot.yaxis.factor
    elif isinstance(plot.yaxis.data, np.ndarray):
        y = plot.yaxis.data * plot.yaxis.factor
    else:
        raise ValueError('cannot find y data for plot {0}'.format(plot.beam))

    if plot.caxis.useCategory:
        cData = np.zeros_like(beamState)
        cData[beamState == stateGood] = hueGood
        cData[beamState == stateOut] = hueOut
        cData[beamState == stateOver] = hueOver
        cData[beamState < 0] = hueDead
        intensity = np.ones_like(x)
        flux = intensity
    else:
        if plot.beamC is None:
            beamC = beam
        else:
            beamC = beamsReturnedBy_run_process[plot.beamC]
        if isinstance(plot.caxis.data, types.FunctionType):
            cData = plot.caxis.data(beamC) * plot.caxis.factor
        elif isinstance(plot.caxis.data, np.ndarray):
            cData = plot.caxis.data * plot.caxis.factor
        else:
            raise ValueError('cannot find data for cData!')

        if plot.fluxKind.startswith('power'):
            intensity = ((beam.Jss + beam.Jpp) *
                         beam.E * beam.accepted / beam.seeded * SIE0)
        elif plot.fluxKind.startswith('s'):
            intensity = beam.Jss
        elif plot.fluxKind.startswith('p'):
            intensity = beam.Jpp
        elif plot.fluxKind.startswith('+-45'):
            intensity = 2*beam.Jsp.real
        elif plot.fluxKind.startswith('left-right'):
            intensity = 2*beam.Jsp.imag
        elif plot.fluxKind.startswith('E'):
            if plot.fluxKind.startswith('Es'):
                intensity = beam.Es
                flux = beam.Jss
            elif plot.fluxKind.startswith('Ep'):
                intensity = beam.Ep
                flux = beam.Jpp
            else:
                intensity = beam.Es + beam.Ep
                flux = beam.Jss + beam.Jpp
        else:
            intensity = beam.Jss + beam.Jpp

        if not plot.fluxKind.startswith('E'):
            flux = intensity

    return x[part], y[part], intensity[part], flux[part], cData[part], nrays, \
        locAlive, locGood, locOut, locOver, locDead, \
        locAccepted, locAcceptedE, locSeeded, locSeededI

# -*- coding: utf-8 -*-
"""
Package :mod:`~xrt.backends.raycing` provides the internal backend of xrt. It
defines beam sources in the module :mod:`~xrt.backends.raycing.sources`,
rectangular and round apertures in :mod:`~xrt.backends.raycing.apertures`,
optical elements in :mod:`~xrt.backends.raycing.oes`, material properties
(essentially reflectivity, transmittivity and absorption coefficient) for
interfaces and crystals in :mod:`~xrt.backends.raycing.materials` and screens
in :mod:`~xrt.backends.raycing.screens`.
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
    colorama, colorPrint, colors, unicode, basestring, is_sequence,
    _VERBOSITY_)

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

#from .sources.beams import Beam

from ._sets_units import (
    allBeamFields, orientationArgSet, shapeArgSet, derivedArgSet,
    renderOnlyArgSet, compoundArgs, dependentArgs, diagnosticArgs, allUnitsAng,
    allUnitsAngStr, allUnitsLen, allUnitsLenStr, allUnitsEnergy,
    allUnitsEnergyStr, allUnitsEmittance, allUnitsEmittanceStr,
    allUnitsCurrent, allUnitsCurrentStr, lengthUnitParams, auto_unit)

from .epics import to_valid_var_name, EpicsDevice, DynamicBeamline

from ._named_arrays import NamedArrayFactory, Center, Limits, Opening, Image2D

from ._flow_utils import (
    auto_units_angle, auto_units_angle_with_energy, append_to_flow,
    append_to_flow_decorator, set_name, vec_to_quat, multiply_quats,
    quat_vec_rotate, get_init_val, get_params, parametrize,
    normalize_ref, ref_kind_for_arg,
    get_argument_editor_hint, parse_editor_mapping, format_editor_scalar,
    serialize_editor_value,
    create_paramdict_oe, create_paramdict_mat, get_obj_str, get_init_kwargs,
    is_valid_uuid, run_process_from_file, build_hist, parse_energy_string,
    is_auto_align_value, get_auto_align_energy, format_energy_input,
    warn_deprecated_list_auto_align, warn_deprecated_glow_v2)
from ._flow import propagationProcess, MessageHandler

from .beamline import (
    distance_xy, distance_xyz, global_to_virgin_local, virgin_local_to_global,
    xyz_from_xz, is_auto_align_required, AlignmentBeam, BeamLine,
    get_layout_beamline_name, get_layout_scan_description,
    set_layout_scan_description)

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
            center = tmp
        elif isinstance(center, tuple):
            center = list(center)

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

    if plot.showAbsorbed:
        absorbBeamKey = getattr(plot, 'beamAbsorb', None)
        if absorbBeamKey is not None:
            ab = beamsReturnedBy_run_process.get(absorbBeamKey)
            if ab is not None:
                absorbedLb = copy.deepcopy(beam)
                absorbedLb.absorb_intensity(ab)
                beam = absorbedLb

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
        if rayFlag == 4:
            locPart = beamState > 0
        else:
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

        srcWt = nrays * beam.sourceWeight if hasattr(beam, 'sourceWeight')\
            else 1.

        if plot.fluxKind.startswith('power'):
            intensity = (beam.Jss + beam.Jpp) * beam.E * SIE0 * srcWt
#            intensity = ((beam.Jss + beam.Jpp) *
#                         beam.E * beam.seededI / beam.seeded * SIE0)
#            intensity = ((beam.Jss + beam.Jpp) *
#                         beam.E * beam.accepted / beam.seeded * SIE0)
        elif plot.fluxKind.startswith('s'):
            intensity = beam.Jss * srcWt
        elif plot.fluxKind.startswith('p'):
            intensity = beam.Jpp * srcWt
        elif plot.fluxKind.startswith('+/-45'):
            intensity = 2*beam.Jsp.real * srcWt
        elif plot.fluxKind.startswith('left-right'):
            intensity = 2*beam.Jsp.imag * srcWt
        elif plot.fluxKind.startswith('E'):
            sqrtWt = np.sqrt(srcWt)
            if plot.fluxKind.startswith('Es'):
                intensity = beam.Es * sqrtWt
                flux = beam.Jss * srcWt
            elif plot.fluxKind.startswith('Ep'):
                intensity = beam.Ep* sqrtWt
                flux = beam.Jpp * srcWt
            else:
                intensity = (beam.Es + beam.Ep) * sqrtWt
                flux = (beam.Jss + beam.Jpp) * srcWt
        else:
            intensity = (beam.Jss + beam.Jpp) * srcWt

        if not plot.fluxKind.startswith('E'):
            flux = intensity

    return x[part], y[part], intensity[part], flux[part], cData[part], nrays, \
        locAlive, locGood, locOut, locOver, locDead, \
        locAccepted, locAcceptedE, locSeeded, locSeededI

# -*- coding: utf-8 -*-
"""
Apertures
---------

Module :mod:`apertures` defines rectangular and round apertures and a set of
coplanar rectangular apertures. Rectangular apertures may have one or more
defining edges. For example, a simple obstacle, like a beam stop block would
have one edge, a block of front-end slits would have two edges at 90 degrees to
each other, and a collimator would have all four edges.

The classes have useful methods for getting divergence from the aperture size,
for setting divergence (calculating the aperture size given the divergence) and
for touching the beam with the aperture, i.e. calculating the minimum aperture
size that lets the whole beam through.
"""

import numpy as np
import inspect
from matplotlib.path import Path as mplPath
import copy
from .. import raycing
from . import sources as rs
from .physconsts import CHBAR

__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"
__all__ = 'RectangularAperture', 'RoundAperture', 'RoundBeamStop', 'DoubleSlit'

allArguments = ('bl', 'name', 'center', 'kind', 'opening', 'alarmLevel', 'r',
                'shadeFraction')


class RectangularAperture(object):
    """Implements an aperture or an obstacle with a combination of horizontal
    and/or vertical edge(s)."""
    def __init__(self, bl=None, name='', center=[0, 0, 0],
                 kind=['left', 'right', 'bottom', 'top'],
                 opening=[-10, 10, -10, 10], alarmLevel=None):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Optical elements are added to its
            `slits` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: 3-sequence of floats
            3D point in global system. The aperture is assumed to be a vertical
            plane perpendicular to the beam line.

        *kind*: sequence
            Any combination of 'top', 'bottom', 'left', 'right'.

        *opening*: sequence
            Distances (with sign according to the local coordinate system) from
            the blade edges to the initial beam line with the length
            corresponding to *kind*.

        *alarmLevel*: float or None.
            Allowed fraction of number of rays absorbed at the aperture
            relative to the number of incident rays. If exceeded, an alarm
            output is printed in the console.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.slits:
                bl.slits.append(self)
                self.ordinalNum = len(bl.slits)
                self.lostNum = -self.ordinalNum - 1000
        raycing.set_name(self, name)
#        if name not in [None, 'None', '']:
#            self.name = name
#        elif not hasattr(self, 'name'):
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 1]

        self.center = center
        if any([x == 'auto' for x in self.center]):
            self._center = copy.copy(self.center)
        if isinstance(kind, str):
            self.kind = (kind,)
            self.opening = [opening, ]
        else:
            self.kind = kind
            self.opening = opening
        self.alarmLevel = alarmLevel
# For plotting footprint images with the envelope aperture:
        self.surface = name,
        self.limOptX = [-500, 500]
        self.limOptY = [-500, 500]
        self.limPhysX = self.limOptX
        self.limPhysY = self.limOptY
        if opening is not None:
            self.set_optical_limits()
        self.shape = 'rect'
        self.spotLimits = []

    def set_optical_limits(self):
        """For plotting footprint images with the envelope aperture."""
        for akind, d in zip(self.kind, self.opening):
            td = float(d)  # otherwise is of type 'numpy.float64' and is
# raycing.is_sequence(d) returns True which is not expected.
            if akind.startswith('l'):
                self.limOptX[0] = td
            elif akind.startswith('r'):
                self.limOptX[1] = td
            elif akind.startswith('b'):
                self.limOptY[0] = td
            elif akind.startswith('t'):
                self.limOptY[1] = td

    def get_divergence(self, source):
        """Gets divergences given the blade openings."""
        sourceToAperture = ((self.center[0]-source.center[0])**2 +
                            (self.center[1]-source.center[1])**2 +
                            (self.center[2]-source.center[2])**2)**0.5
        divergence = []
        for d in self.opening:
            divergence.append(d / sourceToAperture)
        return divergence

    def set_divergence(self, source, divergence):
        """Gets the blade openings given divergences.
        *divergence* is a sequence corresponding to *kind*"""
        sourceToAperture = ((self.center[0]-source.center[0])**2 +
                            (self.center[1]-source.center[1])**2 +
                            (self.center[2]-source.center[2])**2)**0.5
        d = []
        for div in divergence:
            if div > 0:
                sgn = 1
            else:
                sgn = -1
            d.append(div*sourceToAperture + sgn*raycing.accuracyInPosition)
        self.opening = d

    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``

        .. Returned values: beamLocal
        """
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        path = -lo.y[good] / lo.b[good]
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.path[good] += path
        badIndices = np.zeros(len(beam.x), dtype=np.bool)
        for akind, d in zip(self.kind, self.opening):
            if akind.startswith('l'):
                badIndices[good] = badIndices[good] | (lo.x[good] < d)
            elif akind.startswith('r'):
                badIndices[good] = badIndices[good] | (lo.x[good] > d)
            elif akind.startswith('b'):
                badIndices[good] = badIndices[good] | (lo.z[good] < d)
            elif akind.startswith('t'):
                badIndices[good] = badIndices[good] | (lo.z[good] > d)
        beam.state[badIndices] = self.lostNum

        lo.state[:] = beam.state
        lo.y[good] = 0.

        if hasattr(lo, 'Es'):
            propPhase = np.exp(1e7j * (lo.E[good]/CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        goodN = lo.state > 0
        try:
            if self.spotLimits:
                self.spotLimits[0] = min(self.spotLimits[0], lo.x[goodN].min())
                self.spotLimits[1] = max(self.spotLimits[1], lo.x[goodN].max())
                self.spotLimits[2] = min(self.spotLimits[2], lo.z[goodN].min())
                self.spotLimits[3] = max(self.spotLimits[3], lo.z[goodN].max())
            else:
                self.spotLimits = [lo.x[goodN].min(), lo.x[goodN].max(),
                                   lo.z[goodN].min(), lo.z[goodN].max()]
        except ValueError:
            self.spotLimits = []

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo],
                                   inspect.currentframe())
            return lo

    def touch_beam(self, beam):
        """Adjusts the aperture (i.e. sets self.opening) so that it touches the
        *beam*."""
        good = (beam.state == 1) | (beam.state == 2)
#        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        lo.y[good] /= lo.b[good]
        if ('left' in self.kind) or ('right' in self.kind):
            lo.x[good] -= lo.a[good] * lo.y[good]
        if ('top' in self.kind) or ('bottom' in self.kind):
            lo.z[good] -= lo.c[good] * lo.y[good]
        locOpening = []
        if good.sum() > 0:
            for akind, d in zip(self.kind, self.opening):
                if akind.startswith('l'):
                    locOpening.append(lo.x[good].min())
                elif akind.startswith('r'):
                    locOpening.append(lo.x[good].max())
                elif akind.startswith('t'):
                    locOpening.append(lo.z[good].max())
                elif akind.startswith('b'):
                    locOpening.append(lo.z[good].min())
                else:
                    continue
        self.opening = locOpening
        self.set_optical_limits()

    def local_to_global(self, glo, returnBeam=False, **kwargs):
        if returnBeam:
            retGlo = rs.Beam(copyFrom=glo)
            raycing.virgin_local_to_global(self.bl, retGlo,
                                           self.center, **kwargs)
            return retGlo
        else:
            raycing.virgin_local_to_global(self.bl, glo, self.center, **kwargs)

    def prepare_wave(self, prevOE, nrays, rw=None):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *nrays* of samples are randomly distributed over the slit area.
        """
        if rw is None:
            from . import waves as rw

        nrays = int(nrays)
        wave = rs.Beam(nrays=nrays, forceState=1, withAmplitudes=True)
        xy = np.random.rand(nrays, 2)
        dX = self.limOptX[1] - self.limOptX[0]
        dZ = self.limOptY[1] - self.limOptY[0]
        wave.x[:] = xy[:, 0] * dX + self.limOptX[0]
        wave.z[:] = xy[:, 1] * dZ + self.limOptY[0]
        wave.area = dX * dZ
        wave.dS = wave.area / nrays
        wave.toOE = self

        glo = rs.Beam(copyFrom=wave)
        self.local_to_global(glo)
        rw.prepare_wave(prevOE, wave, glo.x, glo.y, glo.z)
        return wave

    def propagate_wave(self, wave=None, beam=None, nrays='auto'):
        """
        Propagates the incoming *wave* through an aperture using the
        Kirchhoff diffraction theorem. Returned global and local beams can be
        used correspondingly for the consequent ray and wave propagation
        calculations.

        *wave*: Beam object
            Local beam on the surface of the previous optical element.

        *beam*: Beam object
            Incident global beam, only used for alignment purpose.

        *nrays*: 'auto' or int
            Dimension of the created wave. If 'auto' - the same as the incoming
            wave.


        .. Returned values: beamGlobal, beamLocal
        """
        from . import waves as rw
        waveSize = len(wave.x) if nrays == 'auto' else int(nrays)
        prevOE = wave.parent
        if raycing._VERBOSITY_ > 10:
            print("Diffract", self.name, " Prev OE:", prevOE.name)
        if self.bl is not None:
            if raycing.is_auto_align_required(self):
                if beam is not None:
                    self.bl.auto_align(self, beam)
                elif 'source' in str(type(prevOE)):
                    self.bl.auto_align(self, wave)
                else:
                    self.bl.auto_align(self, prevOE.local_to_global(
                        wave, returnBeam=True))
        waveOnSelf = self.prepare_wave(prevOE, waveSize, rw=rw)
        if 'source' in str(type(prevOE)):
            retGlo = prevOE.shine(wave=waveOnSelf)
        else:
            retGlo = rw.diffract(wave, waveOnSelf)
        waveOnSelf.parent = self
        return retGlo, waveOnSelf


class SetOfRectangularAperturesOnZActuator(RectangularAperture):
    """Implements a set of coplanar apertures with a Z actuator."""
    def __init__(self, bl, name, center, apertures, centerZs, dXs, dZs,
                 alarmLevel=None):
        """
        *apertures*: sequence of str
            Names of apertures. The last one must be one of 'bottom-edge' or
            'top-edge'.

        *centerZs*: sequence of float
            Z coordinates of the aperture centers relative to center[2].
            The last one specifies the edge.

        *dXs* and *dZs*: sequence of float
            Openings in x and z local axes which correspond to
            *apertures[:-1]*.


        """
        self.bl = bl
        if bl is not None:
            if self not in bl.slits:
                bl.slits.append(self)
                self.ordinalNum = len(bl.slits)
                self.lostNum = -self.ordinalNum - 1000
        raycing.set_name(self, name)
#        if name not in [None, 'None', '']:
#            self.name = name
#        elif not hasattr(self, 'name'):
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 1]

        self.center = center
        if any([x == 'auto' for x in self.center]):
            self._center = copy.copy(self.center)
        self.zActuator = center[2]
        self.z0 = center[2]
        self.apertures = apertures
        self.centerZs = centerZs
        self.dXs = dXs
        self.dZs = dZs
        self.zlims = None
        self.alarmLevel = alarmLevel
# For plotting footprint images:
        self.surface = self.apertures
        self.limOptX = [0, 0]
        self.limOptY = [0, 0]
        self.limOptX[0] = [-dx*0.5 for dx in dXs]
        self.limOptX[1] = [dx*0.5 for dx in dXs]
        self.limOptX[0].append(-500)
        self.limOptX[1].append(500)
        self.limPhysX = self.limOptX
        self.limPhysY = self.limOptY
        self.shape = 'rect'
        self.spotLimits = []

    def select_aperture(self, apertureName, targetZ):
        """Updates self.curAperture index and finds dz offset corresponding to
        the requested aperture."""
        ca = self.apertures.index(apertureName)
        self.curAperture = ca
        if ca < len(self.apertures) - 1:
            self.kind = 'left', 'right', 'bottom', 'top'
            dx = self.dXs[ca] * 0.5
            dz = self.dZs[ca] * 0.5
            cz = targetZ - self.bl.height
            self.opening = -dx, dx, cz-dz, cz+dz
            self.zActuator = self.z0 + targetZ - self.centerZs[ca]
        else:
            if self.apertures[-1] == 'top-edge':
                self.kind = 'bottom',
            elif self.apertures[-1] == 'bottom-edge':
                self.kind = 'top',
            else:
                raise ValueError('not "top-edge" nor "bottom-edge"!')
            self.opening = self.centerZs[-1] - self.bl.height,
            self.zActuator = self.z0
        maxHalfdZ = max(self.dZs) * 0.5
        minZ = min(self.centerZs) + self.zActuator - self.z0
        maxZ = max(self.centerZs) + self.zActuator - self.z0
        self.zlims = [min(minZ, targetZ) - self.bl.height - maxHalfdZ,
                      max(maxZ, targetZ) - self.bl.height + maxHalfdZ]
        self.set_optical_limits()

    def set_optical_limits(self):
        """For plotting footprint images with the envelope apertures."""
        addToCz = -self.bl.height + self.zActuator - self.z0
        self.limOptY[0] = \
            [cz + addToCz - dz*0.5 for cz, dz in zip(self.centerZs, self.dZs)]
        self.limOptY[1] = \
            [cz + addToCz + dz*0.5 for cz, dz in zip(self.centerZs, self.dZs)]
        self.limOptY[0].append(self.centerZs[-1] + addToCz)
        self.limOptY[1].append(200)


class RoundAperture(object):
    """Implements a round aperture meant to represent a pipe or a flange."""
    def __init__(self, bl=None, name='',
                 center=[0, 0, 0], r=1, alarmLevel=None):
        """ The aperture is assumed to be a vertical plane perpendicular
        to the beam line. *r* is the radius.
        """
        self.bl = bl
        if bl is not None:
            if self not in bl.slits:
                bl.slits.append(self)
                self.ordinalNum = len(bl.slits)
                self.lostNum = -self.ordinalNum - 1000
        raycing.set_name(self, name)
#        if name not in [None, 'None', '']:
#            self.name = name
#        elif not hasattr(self, 'name'):
#            self.name = '{0}{1}'.format(self.__class__.__name__,
#                                        self.ordinalNum)

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 1]

        self.center = center
        if any([x == 'auto' for x in self.center]):
            self._center = copy.copy(self.center)
        self.r = r
        self.alarmLevel = alarmLevel
# For plotting footprint images with the envelope aperture:
        self.surface = name,
        self.limOptX = [-r, r]
        self.limOptY = [-r, r]
        self.limPhysX = self.limOptX
        self.limPhysY = self.limOptY
        self.shape = 'round'
        self.spotLimits = []

    def get_divergence(self, source):
        """Gets the full divergence given the aperture radius."""
        ss = [a - b for a, b in zip(self.center - source.center)]
        return self.r * 2 * (np.dot(ss, ss) ** -0.5)

    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``

        .. Returned values: beamLocal
        """
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        path = -lo.y[good] / lo.b[good]
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.r = (lo.x[good]**2 + lo.z[good]**2)**0.5
        lo.path[good] += path

        badIndices = np.zeros(len(beam.x), dtype=np.bool)
        badIndices[good] = lo.r > self.r
        beam.state[badIndices] = self.lostNum
        lo.state[good] = beam.state[good]
        lo.y[good] = 0.

        if hasattr(lo, 'Es'):
            propPhase = np.exp(1e7j * (lo.E[good]/CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo],
                                   inspect.currentframe())
            return lo

    def local_to_global(self, glo, **kwargs):
        raycing.virgin_local_to_global(self.bl, glo, self.center, **kwargs)

    def prepare_wave(self, prevOE, nrays, rw=None):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *nrays* of samples are randomly distributed over the slit area.
        """
        if rw is None:
            from . import waves as rw

        nrays = int(nrays)
        wave = rs.Beam(nrays=nrays, forceState=1, withAmplitudes=True)
        xy = np.random.rand(nrays, 2)
        r = xy[:, 0]**0.5 * self.r
        phi = xy[:, 1] * 2*np.pi
        wave.x[:] = r * np.cos(phi)
        wave.z[:] = r * np.sin(phi)
        wave.area = np.pi * self.r**2
        wave.dS = wave.area / nrays
        wave.toOE = self

        glo = rs.Beam(copyFrom=wave)
        self.local_to_global(glo)
        rw.prepare_wave(prevOE, wave, glo.x, glo.y, glo.z)
        return wave

    def propagate_wave(self, wave=None, beam=None, nrays='auto'):
        """
        Propagates the incoming *wave* through an aperture using the
        Kirchhoff diffraction theorem. Returned global and local beams can be
        used correspondingly for the consequent ray and wave propagation
        calculations.

        *wave*: Beam object
            Local beam on the surface of the previous optical element.

        *beam*: Beam object
            Incident global beam, only used for alignment purpose.

        *nrays*: 'auto' or int
            Dimension of the created wave. If 'auto' - the same as the incoming
            wave.


        .. Returned values: beamLocal
        """
        from . import waves as rw
        waveSize = len(wave.x) if nrays == 'auto' else int(nrays)
        prevOE = wave.parent
        if self.bl is not None:
            if raycing.is_auto_align_required(self):
                if beam is not None:
                    self.bl.auto_align(self, beam)
                elif 'source' in str(type(prevOE)):
                    self.bl.auto_align(self, wave)
                else:
                    self.bl.auto_align(self, prevOE.local_to_global(
                        wave, returnBeam=True))
        waveOnSelf = self.prepare_wave(prevOE, waveSize, rw=rw)
        if 'source' in str(type(prevOE)):
            prevOE.shine(wave=waveOnSelf)
        else:
            rw.diffract(wave, waveOnSelf)
        waveOnSelf.parent = self
        return self.local_to_global(waveOnSelf, returnBeam=True), waveOnSelf


class RoundBeamStop(RoundAperture):
    """Implements a round beamstop."""

#    def _reportNaN(self, x, strName):
#        nanSum = np.isnan(x).sum()
#        if nanSum > 0:
#            print("{0} NaN rays in {1}!".format(nanSum, strName))

    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``

        .. Returned values: beamLocal
        """
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        with np.errstate(divide='ignore'):
            path = -lo.y[good] / lo.b[good]
        indBad = np.where(np.isnan(path))
        path[indBad] = 0.
        indBad = np.where(np.isinf(path))
        path[indBad] = 0.
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.r = (lo.x[good]**2 + lo.z[good]**2)**0.5
        lo.path[good] += path

        badIndices = np.zeros(len(beam.x), dtype=np.bool)
        badIndices[good] = lo.r < self.r
        beam.state[badIndices] = self.lostNum
        lo.state[good] = beam.state[good]
        lo.y[good] = 0.

        if hasattr(lo, 'Es'):
            propPhase = np.exp(1e7j * (lo.E[good]/CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            glo.path[good] += beam.path[good]
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo],
                                   inspect.currentframe())
            return lo


class DoubleSlit(RectangularAperture):
    """Implements an aperture or an obstacle with a combination of horizontal
    and/or vertical edge(s)."""
    def __init__(self, *args, **kwargs):
        self.shadeFraction = kwargs.pop('shadeFraction', 0.5)
        super(DoubleSlit, self).__init__(*args, **kwargs)

    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``

        .. Returned values: beamLocal
        """
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        shadeMin = (1 - self.shadeFraction) * 0.5
        shadeMax = shadeMin + self.shadeFraction
        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        path = -lo.y[good] / lo.b[good]
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.path[good] += path
        badIndices = np.zeros(len(beam.x), dtype=np.bool)
        for akind, d in zip(self.kind, self.opening):
            if akind.startswith('l'):
                badIndices[good] = badIndices[good] | (lo.x[good] < d)
            elif akind.startswith('r'):
                badIndices[good] = badIndices[good] | (lo.x[good] > d)
            elif akind.startswith('b'):
                badIndices[good] = badIndices[good] | (lo.z[good] < d)
                dsb = d
            elif akind.startswith('t'):
                badIndices[good] = badIndices[good] | (lo.z[good] > d)
                dst = d

        sb = dsb + (dst - dsb) * shadeMin
        st = dsb + (dst - dsb) * shadeMax
        badIndices[good] = \
            badIndices[good] | ((lo.z[good] > sb) & (lo.z[good] < st))
        beam.state[badIndices] = self.lostNum

        lo.state[good] = beam.state[good]
        lo.y[good] = 0.

        if hasattr(lo, 'Es'):
            propPhase = np.exp(1e7j * (lo.E[good]/CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            glo.path[good] += beam.path[good]
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo],
                                   inspect.currentframe())
            return lo


class PolygonalAperture(object):
    """Implements an aperture or an obstacle defined as a set of polygon
    vertices."""
    def __init__(self, bl=None, name='', center=[0, 0, 0],
                 opening=None, alarmLevel=None):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`
            Container for beamline elements. Optical elements are added to its
            `slits` list.

        *name*: str
            User-specified name, can be used for diagnostics output.

        *center*: 3-sequence of floats
            3D point in global system. The aperture is assumed to be a vertical
            plane perpendicular to the beam line.

        *opening*: sequence
            Coordinates [(x0, y0),...(xN, yN)] of the polygon vertices.

        *alarmLevel*: float or None.
            Allowed fraction of number of rays absorbed at the aperture
            relative to the number of incident rays. If exceeded, an alarm
            output is printed in the console.


        """
        self.bl = bl
        if bl is not None:
            bl.slits.append(self)
            self.ordinalNum = len(bl.slits)
            self.lostNum = -self.ordinalNum - 1000
        if name in [None, 'None', '']:
            self.name = '{0}{1}'.format(self.__class__.__name__,
                                        self.ordinalNum)
        else:
            self.name = name

        if bl is not None:
            if self.bl.flowSource != 'Qook':
                bl.oesDict[self.name] = [self, 1]

        self.center = center
        if any([x == 'auto' for x in self.center]):
            self._center = self.center

        self.opening = opening
        self.vertices = np.array(self.opening)
        self.alarmLevel = alarmLevel
# For plotting footprint images with the envelope aperture:
        self.surface = name,
        self.limOptX = [-500, 500]
        self.limOptY = [-500, 500]
        self.limPhysX = self.limOptX
        self.limPhysY = self.limOptY
        if opening is not None:
            self.set_optical_limits()
        self.shape = 'polygon'

    def set_optical_limits(self):
        """For plotting footprint images with the envelope aperture."""
        self.limOptX = [np.min(self.vertices[:, 0]),
                        np.max(self.vertices[:, 0])]
        self.limOptY = [np.min(self.vertices[:, 1]),
                        np.max(self.vertices[:, 1])]

    def propagate(self, beam=None, needNewGlobal=False):
        """Assigns the "lost" value to *beam.state* array for the rays
        intercepted by the aperture. The "lost" value is
        ``-self.ordinalNum - 1000.``

        .. Returned values: beamLocal
        """
        if self.bl is not None:
            self.bl.auto_align(self, beam)
        good = beam.state > 0
# beam in local coordinates
        lo = rs.Beam(copyFrom=beam)
        raycing.global_to_virgin_local(self.bl, beam, lo, self.center, good)
        path = -lo.y[good] / lo.b[good]
        lo.x[good] += lo.a[good] * path
        lo.z[good] += lo.c[good] * path
        lo.path[good] += path

        footprint = mplPath(self.vertices)
        badIndices = np.invert(footprint.contains_points(np.array(
                list(zip(lo.x, lo.z)))))
        beam.state[badIndices] = self.lostNum

        lo.state[good] = beam.state[good]
        lo.y[good] = 0.

        if hasattr(lo, 'Es'):
            propPhase = np.exp(1e7j * (lo.E[good]/CHBAR) * path)
            lo.Es[good] *= propPhase
            lo.Ep[good] *= propPhase

        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, beam)
        if needNewGlobal:
            glo = rs.Beam(copyFrom=lo)
            raycing.virgin_local_to_global(self.bl, glo, self.center, good)
            return glo, lo
        else:
            raycing.append_to_flow(self.propagate, [lo],
                                   inspect.currentframe())
            return lo

    def local_to_global(self, glo, **kwargs):
        raycing.virgin_local_to_global(self.bl, glo, self.center, **kwargs)

    def prepare_wave(self, prevOE, nrays):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *nrays* of samples are randomly distributed over the slit area.
        """
        from . import waves as rw

        nrays = int(nrays)
        wave = rs.Beam(nrays=nrays, forceState=1, withAmplitudes=True)

        dX = self.limOptX[1] - self.limOptX[0]
        dZ = self.limOptY[1] - self.limOptY[0]

        footprint = mplPath(self.vertices, closed=True)
        randRays = 0
        goodX = []
        goodY = []
        while randRays < nrays:
            xy = np.random.rand(nrays, 2)
            rndX = xy[:, 0] * dX + self.limOptX[0]
            rndY = xy[:, 1] * dZ + self.limOptY[0]
            inDots = footprint.contains_points(zip(rndX, rndY))
            goodX = rndX[inDots] if randRays == 0 else\
                np.append(goodX, rndX[inDots])
            goodY = rndY[inDots] if randRays == 0 else\
                np.append(goodY, rndY[inDots])
            randRays = len(goodX)
            if raycing._VERBOSITY_ > 10:
                print("Generated {0} dots of {1}".format(randRays, nrays))

        wave.x[:] = goodX[:nrays]
        wave.z[:] = goodY[:nrays]
        wave.area = 0.5 * np.abs(
                np.dot(self.vertices[:, 0], np.roll(self.vertices[:, 1], 1)) -
                np.dot(self.vertices[:, 1], np.roll(self.vertices[:, 0], 1)))
        wave.dS = wave.area / nrays
        wave.toOE = self

        glo = rs.Beam(copyFrom=wave)
        self.local_to_global(glo)
        rw.prepare_wave(prevOE, wave, glo.x, glo.y, glo.z)
        return wave

# -*- coding: utf-8 -*-
"""
Positioning stages
------------------

The positioning stages define mechanical systems that move optical elements.
LinearStage and RotationStage provide stackable positioning objects.
A stage's globalPose describes its base and its mountPose describes its moving
attachment. Distances are in mm. Euler angles follow the xrt OE convention
and are stored in radians.

The legacy Tripod, OneXStage and TwoXStages classes work in both directions:
they set the primary stages given the OE orientation and, inversely, find the
OE orientation given the primary stages. They are meant to be used as co-parent
classes along with a class derived from
:class:`~xrt.backends.raycing.oes.OE`. They use some fields of
:class:`~xrt.backends.raycing.oes.OE` (essentially the position and rotations)
without subclassing from it. Therefore, another co-parent class derived from
:class:`~xrt.backends.raycing.oes.OE` must be initialized before the
positioning stages in the inheriting class. Consider
:class:`MirrorOnTripodWithTwoXStages(OE, Tripod, TwoXStages)` as an example.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "07 Jan 2016"
# import copy
from dataclasses import dataclass
import logging
import math
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .. import raycing
from ._orientation import (
    DEFAULT_ROTATION_SEQUENCE, Orientation, _angle, _rotation_axes, vector3)


logger = logging.getLogger(__name__)


AXES = {'x': (1., 0., 0.), 'y': (0., 1., 0.), 'z': (0., 0., 1.)}


class Pose:
    """A translation and an orientation relative to a reference frame.

    *center* is a mutable three-component list in mm. *orientation* is an
    :class:`Orientation` and defaults to the identity rotation.
    ``parent @ child`` takes a child pose expressed in the parent frame and
    returns it expressed in the parent's reference frame.
    """

    def __init__(self, center=(0, 0, 0), orientation=None):
        self.center = center
        self.orientation = Orientation() if orientation is None else orientation

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = vector3(value, 'center').tolist()

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        if not isinstance(value, Orientation):
            raise TypeError('orientation must be an Orientation')
        self._orientation = value

    @property
    def matrix(self):
        """4×4 homogeneous transform from local to reference coordinates."""
        matrix = translation(self.center)
        matrix[:3, :3] = self.orientation.matrix
        return matrix

    @property
    def pitch(self):
        return self.orientation.pitch

    @property
    def roll(self):
        return self.orientation.roll

    @property
    def yaw(self):
        return self.orientation.yaw

    @classmethod
    def from_matrix(cls, matrix):
        """Construct a pose from a homogeneous transform."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError('Pose matrix must be 4×4')
        rotation = Rotation.from_matrix(matrix[:3, :3])
        return cls(matrix[:3, 3], Orientation._from_rotation(rotation))

    def __matmul__(self, child):
        if not isinstance(child, Pose):
            return NotImplemented
        parent_rotation = self.orientation._rotation
        rotation = parent_rotation * child.orientation._rotation
        center = vector3(self.center, 'center') + parent_rotation.apply(
            vector3(child.center, 'center'))
        return Pose(center, Orientation._from_rotation(rotation))


def pose_matrix(center, x, z):
    return Pose(center, Orientation(x=x, z=z)).matrix


def translation(vector):
    matrix = np.eye(4)
    matrix[:3, 3] = vector3(vector)
    return matrix


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class StageState:
    """Resolved placement for one evaluation of a stage.

    *globalPose* maps stage-base coordinates to world coordinates.
    *mountPose* maps moving attachment coordinates to world coordinates.
    Pass a fresh states dictionary to share ancestor calculations within
    a single evaluation of several stages.
    """

    globalPose: Pose
    mountPose: Pose


class Stage:
    """Installation geometry and stack evaluation shared by all stage types."""

    _axes = AXES
    _units = raycing.allUnitsLen
    _unitSymbols = raycing.allUnitsLenStr
    _defaultPositionUnit = 'mm'

    def __init__(self, name='stage', center=(0, 0, 0), pitch=0., roll=0.,
                 yaw=0., positionRoll=0.,
                 rotationSequence=DEFAULT_ROTATION_SEQUENCE, pose=None,
                 mount=None, mountOffset=None, axis=None, position=0.,
                 positionMin=None, positionMax=None, positionUnit=None):
        """
        *name*: str
            Human-readable name of the stage.

        *center*: 3-sequence of floats
            World origin of the stage base in mm when unmounted. The stage
            copies it into a list whose components can be edited individually.

        *pitch, roll, yaw*: float or str
            World base angles for an unmounted stage. Numeric values are in
            radians. Strings can specify angle units.

        *positionRoll*: float or str
            User-supplied roll contribution, using the OE convention
            ``roll + positionRoll``. When mounted, this contribution is
            subtracted from the resolved Y Euler angle to report *roll*.

        *rotationSequence*: str
            OE rotation sequence used to compose the base orientation when
            unmounted or to extract its Euler angles when mounted.

        *pose*: :class:`Pose` or None
            Optional complete unmounted base placement. Takes precedence over
            *center*, *pitch*, *roll* and *yaw*.

        *mount*: :class:`Stage` or None
            Parent stage. When mounted, base placement follows its moving
            attachment, and *center*, *pitch*, *roll* and *yaw* are calculated
            world readouts. The authored base placement is used when detached.

        *mountOffset*: :class:`Pose` or None
            This stage's translation and orientation in the parent stage's
            moving attachment frame. Defaults to the identity pose.

        *axis*: str
            Motion axis in the stage base. The allowed axes depend on the
            stage type. Defaults to the first allowed axis.

        *position*: float
            Signed actuator position in *positionUnit*. Values beyond the
            travel limits are clamped to the nearest limit.

        *positionMin, positionMax*: float or None
            Inclusive lower and upper travel limits, in *positionUnit*. A
            limit of None leaves travel unbounded on that side. Tightening
            a limit clamps the current position. A limit edit that crosses
            the opposite limit is ignored.

        *positionUnit*: str or None
            Unit of the position and limits, fixed at construction. Defaults
            to 'mm' for a linear stage and 'rad' for a rotation stage. Other
            units use the raycing length or angle unit dictionary.
        """
        self.uuid = str(uuid.uuid4())
        self.name = name
        self._mount = None
        self._axis = next(iter(self._axes))
        if axis is not None:
            self.axis = axis
        unit = self._defaultPositionUnit if positionUnit is None else positionUnit
        unit = raycing.normalize_mu(unit)
        if isinstance(unit, str) and unit not in self._units:
            unit = next((key for key, symbol in self._unitSymbols.items()
                         if raycing.normalize_mu(symbol) == unit), unit)
        if not isinstance(unit, str) or unit not in self._units:
            raise ValueError('positionUnit must be one of {}'.format(
                ', '.join(self._units)))
        self._positionUnit = unit
        self._position = 0.
        self._positionMin = self._positionMax = None
        self.positionMin = positionMin
        self.positionMax = positionMax
        self.position = position
        self.rotationSequence = rotationSequence
        self.positionRoll = positionRoll
        if pose is not None:
            if not isinstance(pose, Pose):
                raise TypeError('pose must be a Pose')
            center = pose.center
            pitch, roll, yaw = pose.orientation.as_euler(
                self.rotationSequence, self.positionRoll)
        self.center = center
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.mountOffset = Pose() if mountOffset is None else mountOffset
        self.mount = mount

    def _require_unmounted(self, field):
        if self.mount is not None:
            raise AttributeError(
                '{} is calculated when mounted; edit mountOffset'.format(field))

    @property
    def center(self):
        if self.mount is None:
            return self._center
        return tuple(self.globalPose.center)

    @center.setter
    def center(self, value):
        self._require_unmounted('center')
        self._center = vector3(value, 'center').tolist()

    @property
    def pitch(self):
        return self._pitch if self.mount is None else self.orientation.pitch

    @pitch.setter
    def pitch(self, value):
        self._require_unmounted('pitch')
        self._pitch = _angle(value)

    @property
    def roll(self):
        return self._roll if self.mount is None else self.orientation.roll

    @roll.setter
    def roll(self, value):
        self._require_unmounted('roll')
        self._roll = _angle(value)

    @property
    def yaw(self):
        return self._yaw if self.mount is None else self.orientation.yaw

    @yaw.setter
    def yaw(self, value):
        self._require_unmounted('yaw')
        self._yaw = _angle(value)

    @property
    def positionRoll(self):
        return self._positionRoll

    @positionRoll.setter
    def positionRoll(self, value):
        self._positionRoll = _angle(value)

    @property
    def rotationSequence(self):
        return self._rotationSequence

    @rotationSequence.setter
    def rotationSequence(self, value):
        _rotation_axes(value)
        self._rotationSequence = value

    @property
    def orientation(self):
        """Resolved world orientation of the stage base."""
        return self.globalPose.orientation

    @orientation.setter
    def orientation(self, value):
        self._require_unmounted('orientation')
        if not isinstance(value, Orientation):
            raise TypeError('orientation must be an Orientation')
        self._pitch, self._roll, self._yaw = value.as_euler(
            self.rotationSequence, self.positionRoll)

    @property
    def mount(self):
        return self._mount

    @mount.setter
    def mount(self, stage):
        if stage is not None and not isinstance(stage, Stage):
            raise TypeError('mount must be a Stage or None')
        ancestor = stage
        while ancestor is not None:
            if ancestor is self:
                raise ValueError('A stage stack cannot contain a cycle')
            ancestor = ancestor.mount
        self._mount = stage

    @property
    def mountOffset(self):
        return self._mountOffset

    @mountOffset.setter
    def mountOffset(self, value):
        if not isinstance(value, Pose):
            raise TypeError('mountOffset must be a Pose')
        self._mountOffset = value

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        if isinstance(value, str) and value in self._axes:
            self._axis = value
        else:
            logger.warning('%s: axis must be one of %s', self.name,
                           ', '.join(self._axes))

    @property
    def positionUnit(self):
        """Unit of position and travel limits."""
        return self._positionUnit

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        value = self._travel_value(value, 'position')
        if value is not None:
            self._position = self._clamp_position(value)

    @property
    def positionMin(self):
        return self._positionMin

    @positionMin.setter
    def positionMin(self, value):
        if value is not None:
            value = self._travel_value(value, 'positionMin')
            if value is None:
                return
            if self.positionMax is not None and value > self.positionMax:
                logger.warning('%s: positionMin must not exceed positionMax',
                               self.name)
                return
        self._positionMin = value
        self._position = self._clamp_position(self._position)

    @property
    def positionMax(self):
        return self._positionMax

    @positionMax.setter
    def positionMax(self, value):
        if value is not None:
            value = self._travel_value(value, 'positionMax')
            if value is None:
                return
            if self.positionMin is not None and value < self.positionMin:
                logger.warning('%s: positionMax must not fall below positionMin',
                               self.name)
                return
        self._positionMax = value
        self._position = self._clamp_position(self._position)

    def _travel_value(self, value, field):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isfinite(value * self._units[self.positionUnit]):
                return value
        logger.warning('%s: %s must be a finite number', self.name, field)
        return None

    def _clamp_position(self, value):
        if self.positionMin is not None:
            value = max(value, self.positionMin)
        if self.positionMax is not None:
            value = min(value, self.positionMax)
        return value

    @property
    def _position_in_base_units(self):
        return self.position * self._units[self.positionUnit]

    def motion_pose(self):
        """Return the transform produced by the actuator in base coordinates."""
        raise NotImplementedError

    def _global_pose(self, states=None):
        if self.mount is None:
            orientation = Orientation(
                pitch=self._pitch, roll=self._roll, yaw=self._yaw,
                positionRoll=self.positionRoll, rotationSequence=self.rotationSequence)
            return Pose(self._center, orientation)
        pose = self.mount.evaluate(states).mountPose @ self.mountOffset
        pose.orientation = Orientation._from_rotation(
            pose.orientation._rotation, self.rotationSequence, self.positionRoll)
        return pose

    @property
    def globalPose(self):
        """World pose of the stage base."""
        return self._global_pose()

    @property
    def mountPose(self):
        """World pose of the moving attachment.

        At zero travel it equals the base pose.
        """
        return self.evaluate().mountPose

    def evaluate(self, states=None):
        """Resolve both poses, optionally sharing a cache for this snapshot."""
        if states is None:
            states = {}
        if self not in states:
            base = self._global_pose(states)
            attachment = base @ self.motion_pose()
            states[self] = StageState(base, attachment)
        return states[self]


class LinearStage(Stage):
    """One signed translation along x, y or z of the stage base."""

    def __init__(self, name='linear', axis='x', position=0., **kwargs):
        super().__init__(name=name, axis=axis, position=position, **kwargs)

    def motion_pose(self):
        return Pose(np.asarray(self._axes[self.axis]) *
                    self._position_in_base_units)


class RotationStage(Stage):
    """One signed rotation about an axis fixed in the stage base.

    *axis* is 'pitch', 'roll' or 'yaw', corresponding to the x, y and z
    directions of the base. Positive travel follows the right-hand rule.
    *pivotPoint* is a point on the rotation axis, in base coordinates (mm).
    It defaults to the stage origin. Set it away from the origin to model
    a cradle. Travel and its limits use *positionUnit*, radians by default.
    """

    _axes = {'pitch': AXES['x'], 'roll': AXES['y'], 'yaw': AXES['z']}
    _units = raycing.allUnitsAng
    _unitSymbols = raycing.allUnitsAngStr
    _defaultPositionUnit = 'rad'

    def __init__(self, name='rotation', axis='pitch', position=0.,
                 pivotPoint=(0, 0, 0), **kwargs):
        super().__init__(name=name, axis=axis, position=position, **kwargs)
        self._pivotPoint = [0., 0., 0.]
        self.pivotPoint = pivotPoint

    @property
    def pivotPoint(self):
        """A point on the axis of rotation, in stage-base coordinates (mm)."""
        return self._pivotPoint

    @pivotPoint.setter
    def pivotPoint(self, value):
        try:
            self._pivotPoint = vector3(value, 'pivotPoint').tolist()
        except (TypeError, ValueError, OverflowError):
            logger.warning('%s: pivotPoint must contain three finite numbers',
                           self.name)

    def motion_pose(self):
        pivot = vector3(self.pivotPoint, 'pivotPoint')
        axis = np.asarray(self._axes[self.axis])
        rotation = Rotation.from_rotvec(axis * self._position_in_base_units)
        # T(pivot) @ R @ T(-pivot) fixes every point on the rotation axis.
        return Pose(pivot - rotation.apply(pivot),
                    Orientation._from_rotation(rotation))


class Tripod(object):
    """Implements a tripod - a positioning system on three jacks, which can set
    Z, pitch and roll."""
    def __init__(self, jack1=None, jack2=None, jack3=None):
        """*jack1, jack2, jack3*: lists
        are 3D points in global coordinate system at the horizontal state of
        OE.


        """
        if jack1 is None or jack2 is None or jack3 is None:
            raise ValueError('jack1, jack2 and jack3 must be given')

        self.jack1 = list(jack1)  # [x, y, z] in global system
        self.jack2 = list(jack2)  # [x, y, z] in global system
        self.jack3 = list(jack3)  # [x, y, z] in global system
        centerMinusNominal = self.center[2] - self.bl.height
        self.jack1Offset = centerMinusNominal - self.jack1[2]
        self.jack2Offset = centerMinusNominal - self.jack2[2]
        self.jack3Offset = centerMinusNominal - self.jack3[2]
        self.init_jacks_local()
        self.set_jacks()

    @property
    def _jack1Init(self):
        return [self.jack1[0], self.jack1[1],
                self.center[2] - self.jackToMirrorInvariant]

    @property
    def _jack2Init(self):
        return [self.jack2[0], self.jack2[1],
                self.center[2] - self.jackToMirrorInvariant]

    @property
    def _jack3Init(self):
        return [self.jack3[0], self.jack3[1],
                self.center[2] - self.jackToMirrorInvariant]

    def pop_kwargs(self, **kwargs):
        jack1 = kwargs.pop('jack1', None)  # [x, y, z] in global system
        jack2 = kwargs.pop('jack2', None)
        jack3 = kwargs.pop('jack3', None)
        argsT = jack1, jack2, jack3
        return kwargs, argsT

    def init_jacks_local(self):
        """To be invoked in ``self.__init__``. Calculates the invariant offset
        and the jack positions in local virgin system."""
        if not (self.jack1[2] == self.jack2[2] == self.jack3[2]):
            raise ValueError('The mirror must be initially horizontal!')
        self.jackToMirrorInvariant = self.center[2] - self.jack1[2]
# jacks in local virgin system
        self.jack1local = [ji - ci for ji, ci in zip(self.jack1, self.center)]
        self.jack2local = [ji - ci for ji, ci in zip(self.jack2, self.center)]
        self.jack3local = [ji - ci for ji, ci in zip(self.jack3, self.center)]
        for jl in [self.jack1local, self.jack2local, self.jack3local]:
            jl[0], jl[1] = raycing.rotate_z(jl[0], jl[1], self.bl.cosAzimuth,
                                            self.bl.sinAzimuth)

    def set_jacks(self):
        """Finds z of each jack given the center, pitch and roll values."""
#        Ax + By + Cz = D in local system:
        A, B, C = 0.0, 0.0, 1.0
        pitch = self.pitch * math.cos(self.positionRoll)
        if self.roll != 0:
            cosRoll = math.cos(self.roll)
            sinRoll = math.sin(self.roll)
            A, C = raycing.rotate_y(A, C, cosRoll, sinRoll)
        if pitch != 0:
            cosPitch = math.cos(pitch)
            sinPitch = math.sin(pitch)
            B, C = raycing.rotate_x(B, C, cosPitch, sinPitch)
# D of optical plane = 0 because (0, 0, 0) belongs to it:
        D = 0
# D of balls plane:
#        D -= self.jackToMirrorInvariant * (A**2 + B**2 + C**2)**0.5
#        but because rotations are unitary (A**2 + B**2 + C**2) = 1 and:
        D -= self.jackToMirrorInvariant

        for jl, j in zip([self.jack1local, self.jack2local, self.jack3local],
                         [self.jack1, self.jack2, self.jack3]):
# C is never 0 because the plane through the 3 balls is never vertical
            jl[2] = (D - A*jl[0] - B*jl[1]) / C
            j[2] = jl[2] + self.center[2]
        self.jack1Calib = self.jack1[2] + self.jack1Offset
        self.jack2Calib = self.jack2[2] + self.jack2Offset
        self.jack3Calib = self.jack3[2] + self.jack3Offset

    def get_orientation(self):
        """Finds orientation (pitch, roll and central Z) given the jacks."""
        if not hasattr(self, 'jack1'):  # not yet initialized
            return
#        Ax + By + Cz = D in global system:
        A = (self.jack2[1]-self.jack1[1]) * (self.jack3[2]-self.jack1[2])\
            - (self.jack3[1]-self.jack1[1]) * (self.jack2[2]-self.jack1[2])
        B = (self.jack3[0]-self.jack1[0]) * (self.jack2[2]-self.jack1[2])\
            - (self.jack2[0]-self.jack1[0]) * (self.jack3[2]-self.jack1[2])
        C = (self.jack2[0]-self.jack1[0]) * (self.jack3[1]-self.jack1[1])\
            - (self.jack3[0]-self.jack1[0]) * (self.jack2[1]-self.jack1[1])
        ABCNorm = (A**2 + B**2 + C**2)**0.5
        if C < 0:
            ABCNorm *= -1  # its normal looks upwards!
        A /= ABCNorm
        B /= ABCNorm
        C /= ABCNorm
        D = A*self.jack1[0] + B*self.jack1[1] + C*self.jack1[2]  # balls plane
        D += self.jackToMirrorInvariant  # of optical plane
# C is never 0, i.e. the normal to the optical element is never horizontal
        self.center[2] = (D - A*self.center[0] - B*self.center[1]) / C

# A  and B in local system (C is unchanged):
        locA, locB = raycing.rotate_z(
            A, B, self.bl.cosAzimuth, self.bl.sinAzimuth)
        rollInit = getattr(self, '_rollInit', self.roll)
        pitchInit = getattr(self, '_pitchInit', self.pitch)
        tanRoll = locA / C
        self.roll = math.atan(tanRoll)
        self._rollInit = rollInit
        tanPitch = -locB / (locA*math.sin(self.roll) + C*math.cos(self.roll))
        self.pitch = math.atan(tanPitch) * math.cos(self.positionRoll)
        self._pitchInit = pitchInit


class OneXStage(object):
    """A simple X-stage."""
    def __init__(self, dx=0):
        """dx is the nominal x shift of the center in local system."""
        self.dx = dx

        if self.surface is None:
            return
        if not raycing.is_sequence(self.surface):
            raise ValueError('"surface" must be a sequence!')

        for alim in (self.limOptX, self.limOptY):
            if alim is not None:
                if not (raycing.is_sequence(alim[0]) and
                        raycing.is_sequence(alim[1])):
                    raise ValueError('"limOptX" must be a tuple of sequences!')
                if not (len(alim[0]) == len(alim[1]) == len(self.surface)):
                    raise ValueError(
                        'len(self.limOptX[0,1]) != len(surface) !!!')

        for alim in (self.limPhysX[0], self.limPhysX[1],
                     self.limPhysY[0], self.limPhysY[1]):
            if raycing.is_sequence(alim):
                if len((self.surface)) != len(alim):
                    raise ValueError(
                        'length of "surface" and "limPhys..." must be equal!')

    def pop_kwargs(self, **kwargs):
        dx = kwargs.pop('dx', 0)
        argsX = dx,
        return kwargs, argsX

    def select_surface(self, surfaceName):
        """Updates self.curSurface index and finds dx offset corresponding to
        the requested surface."""
        if self.surface is None:
            return
        self.curSurface = self.surface.index(surfaceName)
        cs = self.curSurface
        if self.limOptX is None:
            self.dx = -(self.limPhysX[0][cs] + self.limPhysX[1][cs]) * 0.5
        else:
            self.dx = -(self.limOptX[0][cs] + self.limOptX[1][cs]) * 0.5
        self.get_surface_limits()


class TwoXStages(OneXStage):
    """Two X-stages which can change X and yaw."""
    def __init__(self, tx1=None, tx2=None, dx=0):
        """tx1, tx2 [lists!] are [x, y] points in local system.
        dx is the nominal x shift of the center in local system.
        """
        if tx1 is None or tx2 is None:
            raise ValueError('tx1 and tx2 must be given')

        self._tx1Init = list(tx1)
        self._tx2Init = list(tx2)
        self.tx1 = list(tx1)  # [x, y] in local system
        self.tx2 = list(tx2)  # [x, y] in local system
        if self.tx2[1] == self.tx1[1]:
            raise ValueError('tx1 and tx2 stages must be at different y''s!')
        OneXStage.__init__(self, dx)
        self.set_x_stages()

    def pop_kwargs(self, **kwargs):
        tx1 = kwargs.pop('tx1', None)  # [x, y] in local system
        tx2 = kwargs.pop('tx2', None)  # [x, y] in local system
        dx = kwargs.pop('dx', 0)
        argsX = tx1, tx2, dx
        return kwargs, argsX

    def set_x_stages(self):
        """Finds x of each stage given the x shift and yaw."""
        tanYaw = math.tan(self.yaw)
        self.tx1[0] = (-tanYaw*self.tx1[1] + self.dx)
        self.tx2[0] = (-tanYaw*self.tx2[1] + self.dx)
        if self.positionRoll != 0:
            self.tx1[0] *= math.cos(self.positionRoll)
            self.tx2[0] *= math.cos(self.positionRoll)

    def select_surface(self, surfaceName):
        OneXStage.select_surface(self, surfaceName)
        self.set_x_stages()

    def get_orientation(self):
        """Finds orientation (x shift and yaw) given the tx1 and tx2 stages."""
        if not hasattr(self, 'tx1') or not hasattr(self, 'tx2'):
            return

        tx10 = self.tx1[0]
        tx20 = self.tx2[0]
        if self.positionRoll != 0:
            tx10 *= math.cos(self.positionRoll)
            tx20 *= math.cos(self.positionRoll)
        self.dx = tx10 - (tx20-tx10) * self.tx1[1] / (self.tx2[1]-self.tx1[1])
        self.yaw = -math.atan((tx20-tx10) / (self.tx2[1]-self.tx1[1]))

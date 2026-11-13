# -*- coding: utf-8 -*-
"""Quaternion orientation shared by raycing elements and stages."""

import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from ._flow_utils import auto_units_angle


DEFAULT_ROTATION_SEQUENCE = 'RzRyRx'


def vector3(value, name='vector'):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError('{} must contain three finite numbers'.format(name))
    return array


def _rotation_axes(sequence):
    """Return the fixed-axis application order for an xrt rotation sequence."""
    if (not isinstance(sequence, str) or len(sequence) != 6 or
            sequence[::2] != 'RRR' or set(sequence[1::2]) != set('xyz')):
        raise ValueError('rotationSequence must contain Rx, Ry and Rz once each')
    return sequence[1::2][::-1]


def _angle(value):
    angle = float(auto_units_angle(value))
    if not math.isfinite(angle):
        raise ValueError('Angle must be finite')
    return angle


class Orientation:
    """A quaternion rotation with xrt Euler and x/z basis readouts.

    Supply both *x* and *z*, or supply *pitch*, *roll* and *yaw* (omitted
    angles default to zero). The axes must be finite, nonzero and orthogonal.
    They are normalized and y is formed as z cross x.

    In Euler form, pitch, roll + positionRoll, and yaw are angles about
    x, y and z. *rotationSequence* follows the OE convention. Bare angles
    are in radians and xrt angle strings such as '90 deg' are accepted.
    With x/z input, *rotationSequence* and *positionRoll* set the convention
    for reading back Euler angles.
    """

    def __init__(self, *, x=None, z=None, pitch=None, roll=None, yaw=None,
                 positionRoll=0., rotationSequence=DEFAULT_ROTATION_SEQUENCE):
        axes = _rotation_axes(rotationSequence)
        self._rotationSequence = rotationSequence
        self._positionRoll = _angle(positionRoll)
        if x is not None or z is not None:
            if x is None or z is None:
                raise ValueError('Supply both x and z')
            if any(value is not None for value in (pitch, roll, yaw)):
                raise ValueError('Supply x/z or Euler angles')
            x, z = vector3(x, 'x'), vector3(z, 'z')
            xnorm, znorm = np.linalg.norm(x), np.linalg.norm(z)
            if min(xnorm, znorm) < 1e-12:
                raise ValueError('Orientation vectors must be nonzero')
            x, z = x / xnorm, z / znorm
            if abs(np.dot(x, z)) > 1e-8:
                raise ValueError('x and z must be orthogonal')
            self._rotation = Rotation.from_matrix(
                np.column_stack((x, np.cross(z, x), z)))
        else:
            angles = {axis: _angle(value) if value is not None else 0.
                      for axis, value in zip('xyz', (pitch, roll, yaw))}
            angles['y'] += self.positionRoll
            self._rotation = Rotation.from_euler(
                axes, [angles[axis] for axis in axes])

    @classmethod
    def _from_rotation(cls, rotation, rotationSequence=DEFAULT_ROTATION_SEQUENCE,
                       positionRoll=0.):
        orientation = cls.__new__(cls)
        orientation._rotation = rotation
        orientation._rotationSequence = rotationSequence
        orientation._positionRoll = positionRoll
        return orientation

    @property
    def rotationSequence(self):
        return self._rotationSequence

    @property
    def positionRoll(self):
        return self._positionRoll

    @property
    def quaternion(self):
        """Unit quaternion in scalar-last (x, y, z, w) order."""
        return self._rotation.as_quat()

    @property
    def matrix(self):
        """3×3 rotation taking local vectors into the reference frame."""
        return self._rotation.as_matrix()

    @property
    def x(self):
        return self.matrix[:, 0].tolist()

    @property
    def z(self):
        return self.matrix[:, 2].tolist()

    def as_euler(self, rotationSequence=None, positionRoll=None):
        """Return pitch, roll and yaw in radians.

        At gimbal lock SciPy chooses an equivalent decomposition with one
        angle fixed to zero in the specified sequence.
        """
        sequence = (self.rotationSequence if rotationSequence is None
                    else rotationSequence)
        axes = _rotation_axes(sequence)
        position_roll = (self.positionRoll if positionRoll is None
                         else _angle(positionRoll))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected')
            angles = dict(zip(axes, self._rotation.as_euler(axes)))
        return angles['x'], angles['y'] - position_roll, angles['z']

    @property
    def pitch(self):
        return self.as_euler()[0]

    @property
    def roll(self):
        return self.as_euler()[1]

    @property
    def yaw(self):
        return self.as_euler()[2]

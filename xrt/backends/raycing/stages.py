# -*- coding: utf-8 -*-
"""
Positioning stages
------------------

The positioning stages define mechanical systems that move optical elements.
The classes defined below work in both directions: they set the primary stages
given the OE orientation and, inversely, find the OE orientation given the
primary stages.

These classes are meant to be used as a co-parent classes along with a class
derived from :class:`~xrt.backends.raycing.oes.OE`. They use some fields of
:class:`~xrt.backends.raycing.oes.OE` (essentially the position and rotations)
without subclassing from it. Therefore, another co-parent class derived from
:class:`~xrt.backends.raycing.oes.OE` must be initialized before the
positioning stages in the inheriting class. Consider
:class:`MirrorOnTripodWithTwoXStages(OE, Tripod, TwoXStages)` as an example.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "07 Jan 2016"
# import copy
import math
from .. import raycing


class Tripod(object):
    """Implements a tripod - a positioning system on three jacks, which can set
    Z, pitch and roll."""
    def __init__(self, jack1, jack2, jack3):
        """*jack1, jack2, jack3*: lists
        are 3D points in global coordinate system at the horizontal state of
        OE.
        """
        self.jack1 = jack1  # [x, y, z] in global system
        self.jack2 = jack2  # [x, y, z] in global system
        self.jack3 = jack3  # [x, y, z] in global system
        centerMinusNominal = self.center[2] - self.bl.height
        self.jack1Offset = centerMinusNominal - self.jack1[2]
        self.jack2Offset = centerMinusNominal - self.jack2[2]
        self.jack3Offset = centerMinusNominal - self.jack3[2]
        self.init_jacks_local()
        self.set_jacks()

    def pop_kwargs(self, **kwargs):
        jack1 = kwargs.pop('jack1')  # [x, y, z] in global system
        jack2 = kwargs.pop('jack2')
        jack3 = kwargs.pop('jack3')
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
        tanRoll = locA / C
        self.roll = math.atan(tanRoll)
        tanPitch = -locB / (locA*math.sin(self.roll) + C*math.cos(self.roll))
        self.pitch = math.atan(tanPitch) * math.cos(self.positionRoll)


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
    def __init__(self, tx1, tx2, dx=0):
        """tx1, tx2 [lists!] are [x, y] points in local system.
        dx is the nominal x shift of the center in local system.
        """
        self.tx1 = tx1  # [x, y] in local system
        self.tx2 = tx2  # [x, y] in local system
        if self.tx2[1] == self.tx1[1]:
            raise ValueError('tx1 and tx2 stages must be at different y''s!')
        OneXStage.__init__(self)
        self.set_x_stages()

    def pop_kwargs(self, **kwargs):
        tx1 = kwargs.pop('tx1')  # [x, y] in local system
        tx2 = kwargs.pop('tx2')  # [x, y] in local system
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
        tx10 = self.tx1[0]
        tx20 = self.tx2[0]
        if self.positionRoll != 0:
            tx10 *= math.cos(self.positionRoll)
            tx20 *= math.cos(self.positionRoll)
        self.dx = tx10 - (tx20-tx10) * self.tx1[1] / (self.tx2[1]-self.tx1[1])
        self.yaw = -math.atan((tx20-tx10) / (self.tx2[1]-self.tx1[1]))

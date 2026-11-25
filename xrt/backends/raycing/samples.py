# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "25 Nov 2026"

import numpy as np

import matplotlib as mpl

from .. import raycing
from . import sources as rs
from .oes import Plate
from .materials import EmptyMaterial


class Sphere(Plate):
    def __init__(self, *args, **kwargs):
        self.r = kwargs.pop('r')
        super().__init__(*args, **kwargs)

    def rays_good(self, x, y, z, is2ndXtal=False):
        locState = np.ones(x.size, dtype=np.int32)
        z2 = self.r**2 - x**2 - y**2
        locState[z2 < 0] = raycing.stateOver
        return locState

    def local_z1(self, x, y):
        z2 = self.r**2 - x**2 - y**2
        z = np.zeros_like(x)
        z[z2 > 0] = z2[z2 > 0]**0.5
        return z

    def local_n1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        z2 = self.r**2 - x**2 - y**2
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        a[z2 > 0] = -x[z2 > 0] / z2[z2 > 0]**(-0.5)  # -dz/dx
        b[z2 > 0] = -y[z2 > 0] / z2[z2 > 0]**(-0.5)  # -dz/dy
        c = np.ones_like(x)
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n2(self, x, y):
        return self.local_n1(x, y)


class SpheresInPlate(Plate):
    """
    A plate with embedded spheres. The propagation method ``transmit()`` traces
    an incoming beam from one side of the plate to the other and returns the
    outgoing beam in the global coordinate system together with two local
    beams. The center of the plate is on the first surface. The plate is
    supposed to be positioned nearly normally to the beam with pitch=pi/2.

    .. note::
        If spheres overlap, rays intercepted by a sphere are excluded from
        propagation through subsequent spheres, following the order in which
        the spheres are listed in *spheres*.


    """

    def __init__(self, bl=None, name='', center=[0, 0, 0],
                 material=None, t=0, spheres=[], **kwargs):
        """
        *material*: instance of Material or None

        *t*: float
            Thickness in mm.

        *spheres*: list of dicts
            The dict keys are 'center' (local coords), 'r', 'material'. If the
            3rd coordinate (z) of 'center' is omitted, it will be put to the
            middle plane of the plate.
        """
        self.limPhysX2 = kwargs.pop(
            'limPhysX2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.limPhysY2 = kwargs.pop(
            'limPhysY2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.surfPhysX2 = self.limPhysX2
        self.surfPhysY2 = self.limPhysY2

        super().__init__(bl, name, center, t=t, **kwargs)
        if material is None:
            self.material = EmptyMaterial(kind='plate')
        else:
            self.material = material
        self.spheres = spheres
        self.sphereOEs = []
        self.isphOE = None
        for sph in spheres:
            c = list(sph['center'])
            if len(c) == 2:
                c += [-t*0.5]
            sphereOE = Sphere(center=c, r=sph['r'], material=sph['material'])
            self.sphereOEs.append(sphereOE)

    def rays_good(self, x, y, z, is2ndXtal=False):
        locState = np.ones(x.size, dtype=np.int32)
        noLimPhysX = (self.limPhysX is None or
                      self.limPhysX[1] <= self.limPhysX[0])
        noLimPhysY = (self.limPhysY is None or
                      self.limPhysY[1] <= self.limPhysY[0])
        if noLimPhysX and noLimPhysY:
            return locState
        if noLimPhysX:
            self.limPhysX = self.limPhysY
        if noLimPhysY:
            self.limPhysY = self.limPhysX

        if isinstance(self.shape, raycing.basestring):
            if self.shape.startswith('re'):
                outside = (x < self.limPhysX[0]) | (x > self.limPhysX[1]) |\
                    (y < self.limPhysY[0]) | (y > self.limPhysY[1])
                locState[outside] = self.lostNum
            elif self.shape.startswith('ro'):
                centerX = (self.limPhysX[0]+self.limPhysX[1]) * 0.5
                if np.isnan(centerX):
                    centerX = 0
                radiusX = (self.limPhysX[1]-self.limPhysX[0]) * 0.5
                centerY = (self.limPhysY[0]+self.limPhysY[1]) * 0.5
                if np.isnan(centerY):
                    centerY = 0
                radiusY = (self.limPhysY[1]-self.limPhysY[0]) * 0.5
                locState[((x-centerX)/radiusX)**2 +
                         ((y-centerY)/radiusY)**2 > 1] = self.lostNum
        elif isinstance(self.shape, list):
            footprint = mpl.path.Path(self.shape)
            locState[:] = footprint.contains_points(np.column_stack((x, y)))
            locState[(locState == 0)] = self.lostNum
        else:
            raise ValueError('Unknown shape of OE {0}!'.format(self.name))

        if self.isphOE is not None:
            spState = self.sphereOEs[self.isphOE].rays_good(x, y, z)
            locState[(locState > 0)] = spState[(locState > 0)]

        return locState

    def transmit(self, beam):
        gb = rs.Beam(copyFrom=beam)  # output beam in global coordinates
        lo1 = rs.Beam(copyFrom=beam)  # output beam in local coordinates
        good1 = beam.state > 0
        if good1.sum() == 0:
            return gb, lo1, lo1
        raycing.global_to_virgin_local(self.bl, beam, lo1, self.center, good1)
        self._reflect_local(
            good1, lo1, gb,
            self.pitch, self.roll + self.positionRoll, self.yaw,
            local_z=self.local_z, local_n=self.local_n,
            fromVacuum=True, material=self.material)
        goodAfter1 = gb.state > 0  # (gb.state == 1) | (gb.state == 2)

        # not intersected rays remain unchanged except their state:
        notGood = ~goodAfter1
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)

        for isphOE, sphOE in enumerate(self.sphereOEs):
            self.isphOE = isphOE
            gbs1 = rs.Beam(copyFrom=gb)
            los1 = rs.Beam(copyFrom=gbs1)
            self._reflect_local(
                goodAfter1, los1, gbs1,
                self.pitch, self.roll + self.positionRoll, self.yaw,
                dx=sphOE.center[0], dy=sphOE.center[1], dz=sphOE.center[2],
                local_z=sphOE.local_z1, local_n=sphOE.local_n1,
                fromVacuum=True,
                material=sphOE.material, vacuumMaterial=self.material)
            goodAfterSph1 = gbs1.state == 1
            if goodAfterSph1.sum() == 0:
                continue
            gbs2 = rs.Beam(copyFrom=gbs1)
            los2 = rs.Beam(copyFrom=gbs2)
            self._reflect_local(
                goodAfterSph1, los2, gbs2,
                -self.pitch, self.roll + self.positionRoll, -self.yaw,
                dx=-sphOE.center[0], dy=sphOE.center[1], dz=-sphOE.center[2],
                local_z=sphOE.local_z2, local_n=sphOE.local_n2,
                fromVacuum=False, material=sphOE.material, is2ndXtal=True,
                vacuumMaterial=self.material)
            goodAfterSph2 = gbs2.state == 1
            if goodAfterSph2.sum() > 0:
                rs.copy_beam(gb, gbs2, goodAfterSph2, includeState=True)

        self.isphOE = None
        gb2 = rs.Beam(copyFrom=gb)
        lo2 = rs.Beam(copyFrom=gb2)  # output beam in local coordinates
        good2 = goodAfter1
        if (~good2).sum() > 0:
            # Rays that never reached surface 2 have no meaningful local2 data.
            lo2.state[~good2] = 0
            lo2.x[~good2] = 0.
            lo2.y[~good2] = 0.
            lo2.z[~good2] = 0.
        gb2.state[~good2] = self.lostNum
        if good2.sum() == 0:
            return gb2, lo1, lo2
        self._reflect_local(
            good2, lo2, gb2,
            -self.pitch, self.roll + self.positionRoll, -self.yaw,
            dz=self.t, local_z=self.local_z2, local_n=self.local_n2,
            fromVacuum=False, material=self.material, is2ndXtal=True)
        goodAfter2 = gb2.state > 0  # (gb2.state == 1) | (gb2.state == 2)

        # in global coordinate system:
        raycing.virgin_local_to_global(self.bl, gb2, self.center, goodAfter2)

        # not intersected rays remain unchanged except their state:
        notGood = ~goodAfter2
        gb2.state[notGood] = self.lostNum
        if notGood.sum() > 0:
            rs.copy_beam(gb2, beam, notGood)

        # raycing.append_to_flow(self.double_reflect, [gb2, lo1, lo2],
        #                        inspect.currentframe())

        return gb2, lo1, lo2  # in global and local(lo1 and lo2) coordinates

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import inspect
import copy

from ... import raycing
from .. import sources as rs
from .base import OE


class DCM(OE):
    """Implements a Double Crystal Monochromator with flat crystals."""

    hiddenMethods = ['reflect', 'multiple_reflect', 'propagate_wave']

    def __init__(self, *args, **kwargs):
        u"""
        *bragg*: float, str, list
            Bragg angle in rad. Can be calculated automatically if alignment
            energy is given as a single element list [energy]. If 'auto',
            the alignment energy will be taken from beamLine.alignE.

        *braggOffset*: float
            Bragg angle offset in rad.

        *cryst1roll*, *cryst2roll*, *cryst2pitch*, *cryst2finePitch*: float
            Misalignment angles in rad.

        *cryst2perpTransl*, *cryst2longTransl*: float
            perpendicular and longitudinal translations of the 2nd crystal in
            respect to the 1st one.

        *limPhysX2*, *limPhysY2*, *limOptX2*, *limOptY2*, *material2*:
            refer to the 2nd crystal and are similar to the same parameters
            of the parent class :class:`OE` without the trailing "2".

        *fixedOffset*: float
            Offset between the incoming and outcoming beams in mm. If not None
            or zero the value of *cryst2perpTransl* is replaced by
            *fixedOffset*/2/cos(*bragg*)


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)
        self.energyMin = rs.defaultEnergy - 5.
        self.energyMax = rs.defaultEnergy + 5.

    def __pop_kwargs(self, **kwargs):
        self.braggOffset = kwargs.pop('braggOffset', 0)
        self.bragg = kwargs.pop('bragg', 0)
        self.cryst1roll = kwargs.pop('cryst1roll', 0)
        self.cryst2roll = kwargs.pop('cryst2roll', 0)
        self.cryst2pitch = kwargs.pop('cryst2pitch', 0)
        self.cryst2finePitch = kwargs.pop('cryst2finePitch', 0)
        self.cryst2perpTransl = kwargs.pop('cryst2perpTransl', 0)
        self.cryst2longTransl = kwargs.pop('cryst2longTransl', 0)
        self.limPhysX2 = kwargs.pop(
            'limPhysX2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.limPhysY2 = kwargs.pop(
            'limPhysY2', [-raycing.maxHalfSizeOfOE, raycing.maxHalfSizeOfOE])
        self.limOptX2 = kwargs.pop('limOptX2', None)
        self.limOptY2 = kwargs.pop('limOptY2', None)
#        self.material = kwargs.get('material', None)
        self.material2 = kwargs.pop('material2', None)
        self.fixedOffset = kwargs.pop('fixedOffset', None)
        return kwargs

    @property
    def bragg(self):
        return self._bragg if self._braggVal is None else self._braggVal

    @bragg.setter
    def bragg(self, bragg):
        if isinstance(bragg, (raycing.basestring, list, tuple)):
            self._braggInit = copy.deepcopy(bragg)  # For glow auto-recognition
        bragg = raycing.auto_units_angle(bragg)
        if isinstance(bragg, (raycing.basestring, list, tuple)):
            self._bragg = copy.deepcopy(bragg)
            self._braggVal = None
#            if hasattr(self, 'material') and self.material is not None:
#                print("Setting auto-bragg. Material:", self.material.name)
#                if hasattr(self.material, 'get_Bragg_angle'):
#                    print("Calculating new Bragg value for", bragg, type(bragg))
#                    self._braggVal = self.material.get_Bragg_angle(bragg[0]) - self.braggOffset
#                    print(self._braggVal)
        else:
            self._braggVal = raycing.auto_units_angle(bragg)

    @property
    def braggOffset(self):
        return self._braggOffset

    @braggOffset.setter
    def braggOffset(self, braggOffset):
        self._braggOffset = raycing.auto_units_angle(braggOffset)

    @property
    def cryst1roll(self):
        return self._cryst1roll

    @cryst1roll.setter
    def cryst1roll(self, cryst1roll):
        self._cryst1roll = raycing.auto_units_angle(cryst1roll)
#        self.update_orientation_quaternion()

    @property
    def cryst2roll(self):
        return self._cryst2roll

    @cryst2roll.setter
    def cryst2roll(self, cryst2roll):
        self._cryst2roll = raycing.auto_units_angle(cryst2roll)
#        self.update_orientation_quaternion()

    @property
    def cryst2pitch(self):
        return self._cryst2pitch

    @cryst2pitch.setter
    def cryst2pitch(self, cryst2pitch):
        self._cryst2pitch = raycing.auto_units_angle(cryst2pitch)
#        self.update_orientation_quaternion()

    @property
    def cryst2finePitch(self):
        return self._cryst2finePitch

    @cryst2finePitch.setter
    def cryst2finePitch(self, cryst2finePitch):
        self._cryst2finePitch = raycing.auto_units_angle(cryst2finePitch)
#        self.update_orientation_quaternion()

    @property
    def material2(self):
        def resolve(mat):
            if not raycing.is_valid_uuid(mat):
                return mat

            if self.bl is None:
                print(f"Material with UUID {mat} doesn't exist!")
                return None

            return self.bl.materialsDict.get(mat)

        m = self._material2

        if raycing.is_sequence(m):
            return [resolve(x) for x in m]
        else:
            return resolve(m)

    @material2.setter
    def material2(self, material2):
        self._material2 = material2

    @property
    def limPhysX2(self):
        return self._limPhysX2

    @limPhysX2.setter
    def limPhysX2(self, limPhysX2):
        if limPhysX2 is None:
            self._limPhysX2 = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                              raycing.maxHalfSizeOfOE])
        else:
            self._limPhysX2 = raycing.Limits(limPhysX2)

    @property
    def limPhysY2(self):
        return self._limPhysY2

    @limPhysY2.setter
    def limPhysY2(self, limPhysY2):
        if limPhysY2 is None:
            self._limPhysY2 = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                              raycing.maxHalfSizeOfOE])
        else:
            self._limPhysY2 = raycing.Limits(limPhysY2)

    def get_surface_limits(self):
        """Returns surface_limits."""
        # TODO: multiple surfaces
        OE.get_surface_limits(self)
        if not all([hasattr(self, arg) for arg in
                    ['curSurface', 'limPhysX2',
                     'limPhysY2', 'limOptX2', 'limOptY2']]):
            return
        cs = self.curSurface
        self.surfPhysX2 = self.limPhysX2
        if self.limPhysX2 is not None:
            if raycing.is_sequence(self.limPhysX2[0]):
                self.surfPhysX2 = (self.limPhysX2[0][cs],
                                   self.limPhysX2[1][cs])
        self.surfPhysY2 = self.limPhysY2
        if self.limPhysY2 is not None:
            if raycing.is_sequence(self.limPhysY2[0]):
                self.surfPhysY = (self.limPhysY2[0][cs], self.limPhysY2[1][cs])
        self.surfOptX2 = self.limOptX2
        if self.limOptX2 is not None:
            if raycing.is_sequence(self.limOptX2[0]):
                self.surfOptX = (self.limOptX2[0][cs], self.limOptX2[1][cs])
        self.surfOptY2 = self.limOptY2
        if self.limOptY2 is not None:
            if raycing.is_sequence(self.limOptY2[0]):
                self.surfOptY = (self.limOptY2[0][cs], self.limOptY2[1][cs])

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # just flat:
        return self.local_z(x, y)

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # just flat:
        return self.local_n(x, y)

    def local_n2(self, x, y):
        res = self.local_n1(x, y)
        if self.alpha:
            res[1] *= -1
        return res

    def get_orientation(self):
        if self.fixedOffset not in [0, None]:
            self.cryst2perpTransl = self.fixedOffset/2./np.cos(self.bragg)

    @raycing.append_to_flow_decorator
    def double_reflect(self, beam=None, needLocal=True,
                       fromVacuum1=True, fromVacuum2=True,
                       returnLocalAbsorbed=None):
        """
        Returns the reflected beam in global and two local (if *needLocal*
        is true) systems.

        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam. If
            equals zero, total absorbed intensity is return in the last local
            beam, otherwise the N-th local beam returns the
            absorbed intensity on N-th surface of the optical element.


        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
#        kwArgsIn = {'needLocal': needLocal,
#                    'fromVacuum1': fromVacuum1,
#                    'fromVacuum2': fromVacuum2,
#                    'returnLocalAbsorbed': returnLocalAbsorbed}
#        self.footprint = []
#        if self.bl is not None:
#            if self.bl.flowSource != 'double_refract':
#                if raycing.is_valid_uuid(beam):
#                    kwArgsIn['beam'] = beam
#                    beam = self.bl.beamsDictU[beam]['beamGlobal']
#                else:
#                    kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
#        self.get_orientation()
        gb = rs.Beam(copyFrom=beam)  # output beam in global coordinates
        if needLocal:
            lo1 = rs.Beam(copyFrom=beam)  # output beam in local coordinates
        else:
            lo1 = gb

        good1 = beam.state > 0
        if good1.sum() == 0:
            return gb, lo1, lo1
        raycing.global_to_virgin_local(self.bl, beam, lo1, self.center, good1)
        self._reflect_local(
            good1, lo1, gb, self.pitch + self.bragg,
            self.roll + self.positionRoll + self.cryst1roll, self.yaw, self.dx,
            local_z=self.local_z1, local_n=self.local_n1,
            fromVacuum=fromVacuum1, material=self.material)
        goodAfter1 = (gb.state == 1) | (gb.state == 2)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter1
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)

        gb2 = rs.Beam(copyFrom=gb)
        if needLocal:
            lo2 = rs.Beam(copyFrom=gb2)  # output beam in local coordinates
        else:
            lo2 = gb2
        good2 = goodAfter1
        if hasattr(self, 't'):  # is instance of Plate
            gb2.state[~good2] = self.lostNum
        if good2.sum() == 0:
            return gb2, lo1, lo2
        self._reflect_local(
            good2, lo2, gb2,
            -self.pitch - self.bragg + self.cryst2pitch + self.cryst2finePitch,
            self.roll + self.cryst2roll + self.positionRoll, -self.yaw,
            -self.dx, self.cryst2longTransl, -self.cryst2perpTransl,
            local_z=self.local_z2, local_n=self.local_n2,
            fromVacuum=fromVacuum2, material=self.material2, is2ndXtal=True)
        goodAfter2 = (gb2.state == 1) | (gb2.state == 2)
# in global coordinate system:
        raycing.virgin_local_to_global(self.bl, gb2, self.center, goodAfter2)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter2
        if hasattr(self, 't'):  # is instance of Plate
            gb2.state[notGood] = self.lostNum
        if notGood.sum() > 0:
            rs.copy_beam(gb2, beam, notGood)

        if returnLocalAbsorbed is not None:
            if returnLocalAbsorbed == 0:
                absorbedLb = rs.Beam(copyFrom=lo2)
                absorbedLb.absorb_intensity(beam)
                lo2 = absorbedLb
            elif returnLocalAbsorbed == 1:
                absorbedLb = rs.Beam(copyFrom=lo1)
                absorbedLb.absorb_intensity(beam)
                lo1 = absorbedLb
            elif returnLocalAbsorbed == 1:
                absorbedLb = rs.Beam(copyFrom=lo2)
                absorbedLb.absorb_intensity(lo1)
                lo2 = absorbedLb
#        gb2.parentId = self.uuid
#        lo1.parentId = self.uuid
#        lo2.parentId = self.uuid
        raycing.append_to_flow(self.double_reflect, [gb2, lo1, lo2],
                               inspect.currentframe())
#        if self.bl and self.bl.flowSource != 'double_refract':
#            self.bl.flowU[self.uuid] = {'method': self.double_reflect,
#                                        'kwArgsIn': kwArgsIn}
#            self.bl.beamsDictU[self.uuid] = {'beamGlobal': gb2,
#                                             'beamLocal1': lo1,
#                                             'beamLocal2': lo2}

        return gb2, lo1, lo2  # in global and local(lo1 and lo2) coordinates

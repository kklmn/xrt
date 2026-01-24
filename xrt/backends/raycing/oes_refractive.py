# -*- coding: utf-8 -*-
import copy
import numpy as np
import inspect

from .. import raycing
from . import sources as rs

from .oes_dcm import DCM


class Plate(DCM):
    """Implements a body with two surfaces. It is derived from :class:`DCM`
    because it also has two interfaces but the parameters referring to the 2nd
    crystal should be ignored."""

    hiddenMethods = DCM.hiddenMethods + ['double_reflect']
    hiddenParams = ['order', 'bragg', 'cryst1roll', 'cryst2roll',
                    'cryst2pitch', 'cryst2finePitch', 'cryst2perpTransl',
                    'cryst2longTransl', 'limPhysX2', 'limPhysY2', 'limOptX2',
                    'limOptY2', 'surface', 'material2', 'fixedOffset']

    def __init__(self, *args, **kwargs):
        r"""
        *t*: float
            Thickness in mm.

        *wedgeAngle*: float
            Relative angular misorientation of the back plane.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        DCM.__init__(self, *args, **kwargs)
        self.cryst2perpTransl = -self.t
        self.cryst2pitch = self.wedgeAngle
        if isinstance(self.limPhysX, (list, tuple)):
            self.limPhysX2 = [-x for x in reversed(self.limPhysX)]
        else:
            self.limPhysX2 = self.limPhysX
        self.limPhysY2 = self.limPhysY
        if isinstance(self.limOptX, (list, tuple)):
            self.limOptX2 = [-x for x in reversed(self.limOptX)]
        else:
            self.limOptX2 = self.limOptX
        self.limOptY2 = self.limOptY
#        self.material2 = self.material

    def _resolve_material(self, mat):
        if mat is None:
            return None

        if raycing.is_valid_uuid(mat) and self.bl is not None:
            return self.bl.materialsDict.get(mat)

        if isinstance(mat, str) and self.bl is not None and \
                mat in self.bl.matnamesToUUIDs:
            uuid = self.bl.matnamesToUUIDs.get(mat)
            return self.bl.materialsDict.get(uuid)

        return mat

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t
        self.cryst2perpTransl = -t

    @property
    def limPhysX(self):
        return self._limPhysX

    @limPhysX.setter
    def limPhysX(self, limPhysX):
        if limPhysX is None:
            self._limPhysX = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                             raycing.maxHalfSizeOfOE])
            self._limPhysX2 = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                              raycing.maxHalfSizeOfOE])
#        elif isinstance(limPhysX, (list, tuple)): # np.arrays?
#            self._limPhysX = raycing.Limits(limPhysX)
#            self._limPhysX2 = raycing.Limits(
#                    [-x for x in reversed(limPhysX)])
        else:
            self._limPhysX = raycing.Limits(limPhysX)
            self._limPhysX2 = raycing.Limits(
                    [-x for x in reversed(limPhysX)])
        self.get_surface_limits()

    @property
    def limPhysY(self):
        return self._limPhysY

    @limPhysY.setter
    def limPhysY(self, limPhysY):
        if limPhysY is None:
            self._limPhysY = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                             raycing.maxHalfSizeOfOE])
            self._limPhysY2 = raycing.Limits([-raycing.maxHalfSizeOfOE,
                                              raycing.maxHalfSizeOfOE])
        else:
            self._limPhysY = raycing.Limits(limPhysY)
            self._limPhysY2 = raycing.Limits(
                    [-x for x in reversed(limPhysY)])
        self.get_surface_limits()

    @property
    def material(self):
        m = self._material
        if raycing.is_sequence(m):
            return [self._resolve_material(x) for x in m]
        else:
            return self._resolve_material(m)

    @property
    def material2(self):
        m = self._material2
        if raycing.is_sequence(m):
            return [self._resolve_material(x) for x in m]
        else:
            return self._resolve_material(m)

    @material2.setter
    def material2(self, material2):
        self._material2 = material2

    @material.setter
    def material(self, material):
        self._material = material
        self._material2 = material

        mats = material if raycing.is_sequence(material) else (material,)

        for m in mats:
            mat = self._resolve_material(m)

            if mat is None:
                continue

            if mat.kind not in "plate lens FZP":
                try:
                    name = self.name
                except AttributeError:
                    name = self.__class__.__name__

                raycing.colorPrint(
                    'Warning: material of {0} is not of kind {1}!'
                    .format(name, "plate or lens or FZP"),
                    "YELLOW"
                )

        if hasattr(self, '_nCRLlist') and self._nCRLlist is not None:
            self.nCRL = self._nCRLlist

    @property
    def wedgeAngle(self):
        return self._wedgeAngle

    @wedgeAngle.setter
    def wedgeAngle(self, wedgeAngle):
        self._wedgeAngle = raycing.auto_units_angle(wedgeAngle)
        self.cryst2pitch = self._wedgeAngle

    def __pop_kwargs(self, **kwargs):
        self.t = kwargs.pop('t', 0)  # difference of z zeros in mm
        self.wedgeAngle = raycing.auto_units_angle(
                kwargs.pop('wedgeAngle', 0))
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'plate'

    @raycing.append_to_flow_decorator
    def double_refract(self, beam=None, needLocal=True,
                       returnLocalAbsorbed=None):
        """
        Returns the refracted beam in global and two local (if *needLocal*
        is true) systems.

        *returnLocalAbsorbed*: None, int
            If not None, returned local beam represents the absorbed intensity
            instead of transmitted. The parameter defines the ordinal number of
            the surface in multi-surface element to return the absorbed
            intensity, i.e. 1 for the entrance surface of the plate, 2 for the
            exit, 0 for total intensity absorbed in the element.


        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
#        kwArgsIn = {'needLocal': needLocal,
#                    'returnLocalAbsorbed': returnLocalAbsorbed}
#        if self.bl is not None:
#            if self.bl.flowSource != 'multiple_refract':
#                if raycing.is_valid_uuid(beam):
#                    kwArgsIn['beam'] = beam
#                    beam = self.bl.beamsDictU[beam]['beamGlobal']
#                else:
#                    kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
#        self.material2 = self.material
#        self.cryst2perpTransl = -self.t
        if self.bl is not None:
            tmpFlowSource = self.bl.flowSource
            if self.bl.flowSource != 'multiple_refract':
                self.bl.flowSource = 'double_refract'

        gb, lb1, lb2 = self.double_reflect(beam=beam, needLocal=needLocal,
                                           fromVacuum1=True,
                                           fromVacuum2=False)

        if self.bl is not None:
            if self.bl.flowSource == 'double_refract':
                self.bl.flowSource = tmpFlowSource

        if returnLocalAbsorbed is not None:
            if returnLocalAbsorbed == 0:
                absorbedLb = rs.Beam(copyFrom=lb2)
                absorbedLb.absorb_intensity(beam)
                lb2 = absorbedLb
            elif returnLocalAbsorbed == 1:
                absorbedLb = rs.Beam(copyFrom=lb1)
                absorbedLb.absorb_intensity(beam)
                lb1 = absorbedLb
            elif returnLocalAbsorbed == 2:
                absorbedLb = rs.Beam(copyFrom=lb2)
                absorbedLb.absorb_intensity(lb1)
                lb2 = absorbedLb
#        gb.parentId = self.uuid
#        lb1.parentId = self.uuid
#        lb2.parentId = self.uuid
        raycing.append_to_flow(self.double_refract, [gb, lb1, lb2],
                               inspect.currentframe())

#        if self.bl.flowSource != 'multiple_refract':
#            self.bl.flowU[self.uuid] = {'method': self.double_refract,
#                                        'kwArgsIn': kwArgsIn}
#            self.bl.beamsDictU[self.uuid] = {'beamGlobal': gb,
#                                             'beamLocal1': lb1,
#                                             'beamLocal2': lb2}

        return gb, lb1, lb2


class ParaboloidFlatLens(Plate):
    """Implements a refractive lens or a stack of lenses (CRL) with one side
    as paraboloid and the other one flat."""

    hiddenMethods = Plate.hiddenMethods + ['double_refract']
    cl_plist = ("zmax", "focus")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if  (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
      return 0;
    }

    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (i == 1)
            res = local_z1(cl_plist, x, y);
        if (i == 2)
            res = local_z2(cl_plist, x, y);
        return res;
    }"""
    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }

    float3 local_n2(float8 cl_plist, float x, float y)
    {
      return (float3)(0.,0.,1.);
    }

    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res=0;
        if (i == 1)
            res = local_n1(cl_plist, x, y);
        if (i == 2)
            res = local_n2(cl_plist, x, y);
        return res;
    }"""

    def __init__(self, *args, **kwargs):
        r"""
        *focus*: float
            The focal distance of the of paraboloid in mm. The paraboloid is
            then defined by the equation:

            .. math::
                z = (x^2 + y^2) / (4 * \mathit{focus})

            .. note::

                This is not the focal distance of the lens but of the
                parabola! The former also depends on the refractive index.
                *focus* is only a shape parameter!

        *pitch*: float
            the default value is set to π/2, i.e. to normal incidence.

        *zmax*: float
            If given, limits the *z* coordinate; the object becomes then a
            plate of the thickness *zmax* + *t* with a paraboloid hole at the
            origin.

        *nCRL*: int or tuple (*focalDistance*, *E*)
            If used as CRL (a stack of several lenslets), the number of the
            lenslets nCRL is either given by the user directly or calculated
            for *focalDistance* at energy *E* and then rounded. The lenses are
            stacked along the local [0, 0, -1] direction with the step equal to
            *zmax* + *t* for curved-flat lenses or 2\ *\ *zmax* + *t* for
            double curved lenses. For propagation with *nCRL* > 1 please use
            :meth:`multiple_refract`.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        Plate.__init__(self, *args, **kwargs)

    @property
    def nCRL(self):
        return self._nCRL

    @nCRL.setter
    def nCRL(self, nCRL):
        if isinstance(nCRL, (int, float)):
            self._nCRL = max(int(round(nCRL)), 1)
            self._nCRLlist = None
        elif isinstance(nCRL, (list, tuple)):
            self._nCRL = max(int(round(self.get_nCRL(*nCRL))), 1)
            self._nCRLlist = copy.copy(nCRL)
#            print('nCRL={0}'.format(nCRL))
        else:
            self._nCRL = 1
#            raise ValueError("wrong nCRL value!")

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, focus):
        self._focus = focus
        if hasattr(self, '_nCRLlist'):
            if self._nCRLlist is not None:
                self.nCRL = self._nCRLlist

    def __pop_kwargs(self, **kwargs):
        self.focus = kwargs.pop('focus', 1.)
        self.zmax = kwargs.pop('zmax', None)
        self.nCRL = kwargs.pop('nCRL', 1)
        kwargs['pitch'] = kwargs.get('pitch', np.pi/2)
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        z = (x**2 + y**2) / (4 * self.focus)
        if self.zmax is not None:
            z[z > self.zmax] = self.zmax
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        return self.local_z(x, y)

    def local_n1(self, x, y):
        """Determines the normal vector of OE at (x, y) position. If OE is an
        asymmetric crystal, *local_n* must return 2 normals: the 1st one of the
        atomic planes and the 2nd one of the surface."""
        a = -x / (2*self.focus)  # -dz/dx
        b = -y / (2*self.focus)  # -dz/dy
        if self.zmax is not None:
            z = (x**2 + y**2) / (4*self.focus)
            if isinstance(a, np.ndarray):
                a[z > self.zmax] = 0
            if isinstance(b, np.ndarray):
                b[z > self.zmax] = 0
        c = np.ones_like(x)
        norm = (a**2 + b**2 + 1)**0.5
        return [a/norm, b/norm, c/norm]

    def local_n2(self, x, y):
        return self.local_n(x, y)

    def local_z(self, x, y):
        return self.local_z1(x, y)

    def local_n(self, x, y):
        return self.local_n1(x, y)

    def get_nCRL(self, f, E):
        nCRL, nFactor = 1, 1.
        if all([hasattr(self, val) for val in ['focus', 'material']]):
            if self.focus is not None and self.material is not None:
                if isinstance(self, (DoubleParaboloidLens,
                                     DoubleParabolicCylinderLens)):
                    nFactor *= 0.5
                nCRL = 2 * self.focus / float(f) /\
                    (1. - self.material.get_refractive_index(E).real) * nFactor
        return nCRL

    @raycing.append_to_flow_decorator
    def multiple_refract(self, beam=None, needLocal=True,
                         returnLocalAbsorbed=None):
        """
        Sequentially applies the :meth:`double_refract` method to the stack of
        lenses, center of each of *nCRL* lens is shifted by *zmax* mm
        relative to the previous one along the beam propagation direction.
        Returned global beam emerges from the exit surface of the last lens,
        returned local beams correspond to the entrance and exit surfaces of
        the first lens.

        *returnLocalAbsorbed*: None, 0 or 1
            If not None, returns the absorbed intensity in local beam. If
            equals zero, the last local beam returns total absorbed intensity,
            otherwise the absorbed intensity on single element of the stack.


        .. Returned values: beamGlobal, beamLocal1, beamLocal2
        """
        tmpFlowSource = self.bl.flowSource
        self.bl.flowSource = 'multiple_refract'

        # kwArgsIn = {'needLocal': needLocal,
        #             'returnLocalAbsorbed': returnLocalAbsorbed}
        # if self.bl is not None:
        #     if raycing.is_valid_uuid(beam):
        #         kwArgsIn['beam'] = beam
        #         beam = self.bl.beamsDictU[beam]['beamGlobal']
        #     else:
        #         kwArgsIn['beam'] = beam.parentId
        #     self.bl.auto_align(self, beam)
        if self.nCRL == 1:
            self.centerShift = np.zeros(3)

            lglobal, llocal1, llocal2 = self.double_refract(
                beam=beam, needLocal=needLocal,
                returnLocalAbsorbed=returnLocalAbsorbed)
            # self.bl.flowSource = tmpFlowSource
        else:
            # tmpFlowSource = self.bl.flowSource
            # self.bl.flowSource = 'multiple_refract'
            tempCenter = [c for c in self.center]
            beamIn = beam
            zmax = 5 if self.zmax is None else self.zmax
            if isinstance(self, (DoubleParaboloidLens,
                                 DoubleParabolicCylinderLens)):
                step = 2.*zmax + self.t
            else:
                step = zmax + self.t
            toward = [0, -step, 0]
            for ilens in range(self.nCRL):
                lglobal, tlocal1, tlocal2 = self.double_refract(
                    beam=beamIn, needLocal=needLocal)
                if self.zmax is not None:
                    toward = raycing.rotate_point(
                        [0, 0, 1], self.rotationSequence, self.pitch,
                        self.roll+self.positionRoll, self.yaw)
                    self.center[0] -= step * toward[0]
                    self.center[1] -= step * toward[1]
                    self.center[2] -= step * toward[2]
                beamIn = lglobal
                if ilens == 0:
                    llocal1, llocal2 = tlocal1, tlocal2
            self.centerShift = step * np.array(toward)
            self.center = tempCenter

            if returnLocalAbsorbed is not None:
                if returnLocalAbsorbed == 0:
                    absorbedLb = rs.Beam(copyFrom=llocal2)
                    absorbedLb.absorb_intensity(beam)
                    llocal2 = absorbedLb
                elif returnLocalAbsorbed == 1:
                    absorbedLb = rs.Beam(copyFrom=llocal1)
                    absorbedLb.absorb_intensity(beam)
                    llocal1 = absorbedLb
                elif returnLocalAbsorbed == 2:
                    absorbedLb = rs.Beam(copyFrom=llocal2)
                    absorbedLb.absorb_intensity(llocal1)
                    llocal2 = absorbedLb

            raycing.append_to_flow(self.multiple_refract,
                                   [lglobal, llocal1, llocal2],
                                   inspect.currentframe())

        self.bl.flowSource = tmpFlowSource

#        self.bl.flowU[self.uuid] = {'method': self.multiple_refract,
#                                    'kwArgsIn': kwArgsIn}
#        self.bl.beamsDictU[self.uuid] = {'beamGlobal': lglobal,
#                                         'beamLocal1': llocal1,
#                                         'beamLocal2': llocal2}

#        lglobal.parentId = self.uuid
#        llocal1.parentId = self.uuid
#        llocal2.parentId = self.uuid
        return lglobal, llocal1, llocal2


class ParabolicCylinderFlatLens(ParaboloidFlatLens):
    u"""Implements a refractive lens or a stack of lenses (CRL) with one side
    as parabolic cylinder and the other one flat. The lenslets focalize in one
    direction and they are flat in their local *x* direction and curved in the
    local *y* direction."""

    cl_plist = ("zmax", "focus")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if  (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
      return 0;
    }

    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (i == 1)
            res = local_z1(cl_plist, 0, y);
        if (i == 2)
            res = local_z2(cl_plist, 0, y);
        return res;
    }"""
    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }

    float3 local_n2(float8 cl_plist, float x, float y)
    {
      return (float3)(0.,0.,1.);
    }

    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        if (i == 1)
            res = local_n1(cl_plist, 0, y);
        if (i == 2)
            res = local_n2(cl_plist, 0, y);
        return res;
    }"""

    def local_z1(self, x, y):
        return super().local_z1(0, y)

    def local_n1(self, x, y):
        return super().local_n1(0, y)


class DoubleParaboloidLens(ParaboloidFlatLens):
    """Implements a refractive lens or a stack of lenses (CRL) with two equal
    paraboloids from both sides."""

    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }"""

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n2(self, x, y):
        return self.local_n1(x, y)


class DoubleParabolicCylinderLens(ParabolicCylinderFlatLens):
    """Implements a refractive lens or a stack of lenses (CRL) with two equal
    parabolic cylinders from both sides."""

    def local_z2(self, x, y):
        return self.local_z1(x, y)

    def local_n2(self, x, y):
        return self.local_n1(x, y)


# class ConvConcParaboloidLens(ParaboloidFlatLens):
#    def local_z2(self, x, y):
#        return -self.local_z1(x, y)
#
#    def local_n2(self, x, y):
#        n1 = self.local_n1(x, y)
#        return -n1[0], -n1[1], n1[2]

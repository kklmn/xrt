# -*- coding: utf-8 -*-
import os
import numpy as np

from ... import raycing
from .base import OE

__fdir__ = os.path.dirname(__file__)


class LauePlate(OE):
    """A flat Laue plate. The thickness is defined in its *material* part."""

    def local_n(self, x, y):
        a, b, c = 0, 0, 1
        if self.alpha:
            bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            bB, cB = c, -b
        return [a, bB, cB, a, b, c]

    def local_n_depth(self, x, y, z):
        return self.local_n(x, y)


class BentLaueCylinder(OE):
    """Simply bent reflective optical element in Laue geometry (duMond).
    This element supports volumetric diffraction model, if corresponding
    parameter is enabled in the assigned *material*."""

    cl_plist = ("crossSectionInt", "R")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
      if (cl_plist.s0 == 0)
        {
          return cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1 - y*y);
        }
      else
        {
          return 0.5 * y * y / cl_plist.s1;
        }
    }"""

    def __init__(self, *args, **kwargs):
        """
        *R*: float or 2-tuple.
            Meridional radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.

        *crossSection*: str
            Determines the bending shape: either 'circular' or 'parabolic'.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    @property
    def crossSection(self):
        return self._crossSection

    @crossSection.setter
    def crossSection(self, crossSection):
        if not (crossSection.startswith('circ') or
                crossSection.startswith('parab')):
            raise ValueError('unknown crossSection!')
        self._crossSection = crossSection
        self.crossSectionInt = 0 if self.crossSection.startswith('circ') else 1

    @property
    def R(self):
        return self._R if self._RVal is None else self._RVal

    @R.setter
    def R(self, R):
        if R in [None, 0]:
            self._RVal = np.inf
            self._R = None
        elif isinstance(R, (tuple, list)):
            if hasattr(self, '_braggVal') and self._braggVal != 0:
                self._RVal = self.get_Rmer_from_Coddington(
                        R[0], R[1], self._braggVal)
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RVal = self.get_Rmer_from_Coddington(
                        R[0], R[1], self._pitchVal)
            else:
                self._RVal = np.inf
            self._R = R
        else:
            self._RVal = R
            self._R = None

        self._reset_material()

    @property
    def material(self):
        if raycing.is_sequence(self._material):
            matSur = self._material[self.curSurface]
        else:
            matSur = self._material

        if raycing.is_valid_uuid(matSur) and self.bl is not None:
            mat = self.bl.materialsDict.get(matSur)
        else:
            mat = matSur

        return mat

    @material.setter
    def material(self, material):
        self._material = material
        self._reset_material()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """Sets the asymmetry angle *alpha* for a crystal OE. It calculates
        cos(alpha) and sin(alpha) which are then used for rotating the normal
        to the crystal planes."""
        self._alpha = raycing.auto_units_angle(alpha)
        if self.alpha is not None:
            self.cosalpha = np.cos(self.alpha)
            self.sinalpha = np.sin(self.alpha)
            self.tanalpha = self.sinalpha / self.cosalpha
            self._reset_material()

    def _reset_material(self):
        if not all([hasattr(self, v) for v in
                    ['_R', '_material', '_alpha']]):
            return
#        if raycing.is_sequence(self.material):
#            matSur = self.material[self.curSurface]
#        else:
        matSur = self.material
        if hasattr(matSur, 'set_OE_properties'):
            matSur.set_OE_properties(self.alpha, self.R, None)

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 1.0e4)
        self.crossSection = kwargs.pop('crossSection', 'parabolic')
        return kwargs

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if self.crossSection.startswith('circ'):  # 'circular'
            return self.R - np.sqrt(self.R**2 - y**2)
        else:  # 'parabolic'
            return y**2 / 2.0 / self.R

    def local_n_cylinder(self, x, y, R, alpha):
        """The main part of :meth:`local_n`. It introduces two new arguments
        to simplify the implementation of :meth:`local_n` in the derived class
        :class:`GroundBentLaueCylinder`."""
        a = np.zeros_like(x)  # -dz/dx
        b = -y / R  # -dz/dy
        if self.crossSection.startswith('circ'):  # 'circular'
            c = (R**2 - y**2)**0.5 / R
        elif self.crossSection.startswith('parab'):  # 'parabolic'
            norm = (b**2 + 1)**0.5
            b /= norm
            c = 1. / norm
        else:
            raise ValueError('unknown crossSection!')
        if alpha:
            bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            # bB, cB = raycing.rotate_x(b, c, 0, -1)
            bB, cB = c, -b
        return [a, bB, cB, a, b, c]

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        return self.local_n_cylinder(x, y, self.R, self.alpha)

    def local_n_depth(self, x, y, z):

        a = -x*0.  # -dz/dx
        b = -y / self.R  # -dz/dy
        c = 1.

        norm = np.sqrt(a**2 + b**2 + 1.)
        a /= norm
        b /= norm
        c /= norm

        plane_h = np.array([np.zeros_like(x),
                            np.cos(self.alpha)*np.ones_like(x),
                            -np.sin(self.alpha)*np.ones_like(x)])

#        displacement_pytte = [-xg*zg*invR2 + 0.5*coef3*zg**2,
#                              -yg*zg*invR1 + 0.5*coef2*zg**2,
#                              0.5*invR2*xg**2+0.5*invR1*yg**2+0.5*coef1*zg**2]

#        displacement_pp = [-nu*x*z*invR1,
#                           -y*z*invR1,
#                           (y**2-nu*x**2+nu*z**2)*invR1*0.5]

        if hasattr(self.material, 'djparams'):
            coef1, coef2, invR1, coef3, invR2 = self.material.djparams
            duh_dx = np.einsum('ij,ij->j', plane_h, np.array([-z*invR2,
                                                              np.zeros_like(x),
                                                              x*invR2])*1e3)
            duh_dy = np.einsum('ij,ij->j', plane_h, np.array([np.zeros_like(x),
                                                              -z*invR1,
                                                              y*invR1])*1e3)
            duh_dz = np.einsum('ij,ij->j', plane_h, np.array([-x*invR2+z*coef3,
                                                              -y*invR1+z*coef2,
                                                              z*coef1])*1e3)
        else:
            if hasattr(self.material, 'nu'):
                nu = self.material.nu
            else:
                nu = 0.22   # debug only, using Si
            duh_dx = np.dot(plane_h, np.array([np.zeros_like(x),
                                               np.zeros_like(x),
                                               np.zeros_like(x)]))
            duh_dy = np.dot(plane_h, np.array([np.zeros_like(x),
                                               -z/self.R, y/self.R]))
            duh_dz = np.dot(plane_h, np.array([np.zeros_like(x),
                                               -y/self.R,
                                               nu*z/self.R]))
        hprime = plane_h - np.array([duh_dx, duh_dy, duh_dz])
        hnorm = np.linalg.norm(hprime, axis=0)
        hprime /= hnorm

        return [hprime[0, :], hprime[1, :], hprime[2, :], a, b, c]


class BentLaue2D(OE):
    """Parabolically bent reflective optical element in Laue geometry.
    Meridional and sagittal radii (Rm, Rs) can be defined independently and
    have same or opposite sign, representing concave (+, +), convex (-, -) or
    saddle (+, -) shaped profile.
    This element supports volumetric diffraction model, if corresponding
    parameter is enabled in the assigned *material*.
    """

#    cl_plist = ("crossSectionInt", "R")
#    cl_local_z = """
#    float local_z(float8 cl_plist, int i, float x, float y)
#    {
#        if (cl_plist.s0 == 0)
#        {
#            return cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1 - x*x -y*y);
#        }
#        else
#        {
#          return 0.5 * (y * y + x * x) / cl_plist.s1;
#        }
#    }"""

    def __init__(self, *args, **kwargs):
        """
        *Rm*: float or 2-tuple.
            Meridional bending radius.

        *Rs*: float or 2-tuple.
            Sagittal radius.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    @property
    def Rm(self):
        return self._Rm if self._RmVal is None else self._RmVal

    @Rm.setter
    def Rm(self, Rm):
        if Rm in [None, 0]:
            self._RmVal = np.inf
            self._Rm = None
        elif isinstance(Rm, (tuple, list)):
            if hasattr(self, '_braggVal') and self._braggVal != 0:
                self._RmVal = self.get_Rmer_from_Coddington(
                        Rm[0], Rm[1], self._braggVal)
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RmVal = self.get_Rmer_from_Coddington(
                        Rm[0], Rm[1], self._pitchVal)
            else:
                self._RmVal = np.inf
            self._Rm = Rm
        else:
            self._RmVal = Rm
            self._Rm = None

        self._reset_material()

    @property
    def Rs(self):
        return self._Rs if self._RsVal is None else self._RsVal

    @Rs.setter
    def Rs(self, Rs):
        if Rs in [None, 0]:
            self._RsVal = np.inf
            self._Rs = None
        elif isinstance(Rs, (tuple, list)):
            if hasattr(self, '_braggVal') and self._braggVal != 0:
                self._RsVal = self.get_rsag_from_Coddington(
                        Rs[0], Rs[1], self._braggVal)
            elif hasattr(self, '_pitchVal') and self._pitchVal != 0:
                self._RsVal = self.get_rsag_from_Coddington(
                        Rs[0], Rs[1], self._pitchVal)
            else:
                self._RsVal = np.inf
            self._Rs = Rs
        else:
            self._RsVal = Rs
            self._Rs = None

        self._reset_material()

    @property
    def material(self):
        if raycing.is_sequence(self._material):
            matSur = self._material[self.curSurface]
        else:
            matSur = self._material

        if raycing.is_valid_uuid(matSur) and self.bl is not None:
            mat = self.bl.materialsDict.get(matSur)
        else:
            mat = matSur

        return mat

    @material.setter
    def material(self, material):
        self._material = material
        self._reset_material()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """Sets the asymmetry angle *alpha* for a crystal OE. It calculates
        cos(alpha) and sin(alpha) which are then used for rotating the normal
        to the crystal planes."""
        self._alpha = raycing.auto_units_angle(alpha)
        if self.alpha is not None:
            self.cosalpha = np.cos(self.alpha)
            self.sinalpha = np.sin(self.alpha)
            self.tanalpha = self.sinalpha / self.cosalpha
            self._reset_material()

    def _reset_material(self):
        if not all([hasattr(self, v) for v in
                    ['_Rm', '_Rs', '_material', '_alpha']]):
            return
#        if raycing.is_sequence(self.material):
#            matSur = self.material[self.curSurface]
#        else:
        matSur = self.material
        if hasattr(matSur, 'set_OE_properties'):
            matSur.set_OE_properties(self.alpha, self.Rm, self.Rs)

    def __pop_kwargs(self, **kwargs):
        self.Rm = kwargs.pop('Rm', 1.0e4)
        self.Rs = kwargs.pop('Rs', -5.0e4)
        # Coddington equations don't work for Laue crystals
        return kwargs

    def local_z(self, x, y):
        return 0.5*x**2 / self.Rs + 0.5*y**2 / self.Rm

    def local_n_depth(self, x, y, z):

        a = -x / self.Rs  # -dz/dx
        b = -y / self.Rm  # -dz/dy
        c = 1.

        norm = np.sqrt(a**2 + b**2 + 1.)
        a /= norm
        b /= norm
        c /= norm

        plane_h = np.array([np.zeros_like(x),
                            np.cos(self.alpha)*np.ones_like(x),
                            -np.sin(self.alpha)*np.ones_like(x)])

#        displacement_pytte = [-xg*zg*invR2 + 0.5*coef3*zg**2,
#                              -yg*zg*invR1 + 0.5*coef2*zg**2,
#                              0.5*invR2*xg**2+0.5*invR1*yg**2+0.5*coef1*zg**2]

#        displacement_pp = [-nu*x*z*invR1,
#                           -y*z*invR1,
#                           (y**2-nu*x**2+nu*z**2)*invR1*0.5]

        if hasattr(self.material, 'djparams'):
            coef1, coef2, invR1, coef3, invR2 = self.material.djparams
            duh_dx = np.einsum('ij,ij->j', plane_h, np.array([-z*invR2,
                                                              np.zeros_like(x),
                                                              x*invR2])*1e3)
            duh_dy = np.einsum('ij,ij->j', plane_h, np.array([np.zeros_like(x),
                                                              -z*invR1,
                                                              y*invR1])*1e3)
            duh_dz = np.einsum('ij,ij->j', plane_h, np.array([-x*invR2+z*coef3,
                                                              -y*invR1+z*coef2,
                                                              z*coef1])*1e3)
        else:
            if hasattr(self.material, 'nu'):
                nu = self.material.nu
            else:   # debug only, using Si and anticlastic bending
                nu = 0.22
            duh_dx = np.dot(plane_h, np.array([-z*nu/self.Rm, x*0,
                                               -x*nu/self.Rm]))
            duh_dy = np.dot(plane_h, np.array([x*0, -z/self.Rm, y/self.Rm]))
            duh_dz = np.dot(plane_h, np.array([-x*nu/self.Rm,
                                               -y/self.Rm,
                                               nu*z/self.Rm]))
        hprime = plane_h - np.array([duh_dx, duh_dy, duh_dz])
        hnorm = np.linalg.norm(hprime, axis=0)
        hprime /= hnorm

        return [hprime[0, :], hprime[1, :], hprime[2, :], a, b, c]

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        a = -x / self.Rs  # -dz/dx
        b = -y / self.Rm  # -dz/dy
        c = 1.

        norm = np.sqrt(a**2 + b**2 + 1)
        a /= norm
        b /= norm
        c /= norm

        sinpitch = -b
        cospitch = np.sqrt(1 - b**2)

        sinroll = -a
        cosroll = np.sqrt(1 - a**2)

        aB = np.zeros_like(a)
#        bB = c
#        cB = -b
        bB = np.ones_like(a)
        cB = np.zeros_like(a)

        if self.alpha:
            bB, cB = raycing.rotate_x(bB, cB, self.cosalpha, -self.sinalpha)

#        if self.alpha:  from BentLaueCylinder
#            b, c = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
#        else:
#            b, c = c, -b
        aB, cB = raycing.rotate_y(aB, cB, cosroll, -sinroll)
        bB, cB = raycing.rotate_x(bB, cB, cospitch, sinpitch)

        normB = (bB**2 + cB**2 + aB**2)**0.5

        return [aB/normB, bB/normB, cB/normB, a/norm, b/norm, c/norm]


class GroundBentLaueCylinder(BentLaueCylinder):
    """Ground-bent reflective optical element in Laue geometry."""

    def local_n(self, x, y):
        """Determines the normal vectors of OE at (*x*, *y*) position: of the
        atomic planes and of the surface."""
        nSurf = self.local_n_cylinder(x, y, self.R, None)
# approximate expressions:
#        nBragg = self.local_n_cylinder(x, y, self.R*2, self.alpha)
#        return [nBragg[0], nBragg[1], nBragg[2],
#            nSurf[-3], nSurf[-2], nSurf[-1]]
# exact expressions:
        a = np.zeros_like(x)
        b = -y
        c = (self.R**2 - y**2)**0.5 + self.R
        if self.alpha:
            b, c = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            b, c = c, -b
        norm = np.sqrt(b**2 + c**2)
        return [a/norm, b/norm, c/norm, nSurf[-3], nSurf[-2], nSurf[-1]]


class BentLaueSphere(BentLaueCylinder):
    """Spherically bent reflective optical element in Laue geometry."""

    cl_plist = ("crossSectionInt", "R")
    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        if (cl_plist.s0 == 0)
        {
            return cl_plist.s1 - sqrt(cl_plist.s1*cl_plist.s1 - x*x -y*y);
        }
        else
        {
          return 0.5 * (y * y + x * x) / cl_plist.s1;
        }
    }"""

    def local_z(self, x, y):
        if self.crossSection.startswith('circ'):  # 'circular'
            return self.R - np.sqrt(self.R**2 - x**2 - y**2)
        else:  # 'parabolic'
            return (x**2+y**2) / 2.0 / self.R

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        if self.crossSection.startswith('circ'):  # 'circular'
            a = -x * (self.R**2 - x**2 - y**2)**(-0.5)  # -dx/dy
            b = -y * (self.R**2 - x**2 - y**2)**(-0.5)  # -dz/dy
        else:  # 'parabolic'
            a = -x / self.R  # -dz/dx
            b = -y / self.R  # -dz/dy
        c = 1.
        norm = (a**2 + b**2 + 1)**0.5
        aB = 0.
        bB = c
        cB = -b
        normB = (b**2 + c**2)**0.5
        return [aB/normB, bB/normB, cB/normB, a/norm, b/norm, c/norm]

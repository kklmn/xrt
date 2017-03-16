# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "12 Apr 2016"
import numpy as np
from .. import raycing

defaultEnergy = 9.0e3


class Beam(object):
    """Container for the beam arrays. *x, y, z* give the starting points.
    *a, b, c* give normalized vectors of ray directions (the source must take
    care about the normalization). *E* is energy. *Jss*, *Jpp* and *Jsp* are
    the components  of the coherency matrix. The latter one is complex. *Es*
    and *Ep* are *s* and *p* field amplitudes (not always used). *path* is the
    total path length from the source to the last impact point. *theta* is the
    incidence angle. *order* is the order of grating diffraction. If multiple
    reflections are considered: *nRefl* is the number of reflections,
    *elevationD* is the maximum elevation distance between the rays and the
    surface as the ray travels from one impact point to the next one,
    *elevationX*, *elevationY*, *elevationZ* are the coordinates of the
    highest elevation points. If an OE uses a parametric representation,
    *s*, *phi*, *r* arrays store the impact points in the parametric
    coordinates.
    """
    def __init__(self, nrays=raycing.nrays, copyFrom=None, forceState=False,
                 withNumberOfReflections=False, withAmplitudes=False,
                 xyzOnly=False):
        if copyFrom is None:
            # coordinates of starting points
            self.x = np.zeros(nrays)
            self.y = np.zeros(nrays)
            self.z = np.zeros(nrays)
            if not xyzOnly:
                self.sourceSIGMAx = 0.
                self.sourceSIGMAz = 0.
                self.filamentDtheta = 0.
                self.filamentDpsi = 0.
                self.filamentDX = 0.
                self.filamentDZ = 0.
                self.state = np.zeros(nrays, dtype=np.int)
                # components of direction
                self.a = np.zeros(nrays)
                self.b = np.ones(nrays)
                self.c = np.zeros(nrays)
                # total ray path
                self.path = np.zeros(nrays)
                # energy
                self.E = np.ones(nrays) * defaultEnergy
                # components of coherency matrix
                self.Jss = np.ones(nrays)
                self.Jpp = np.zeros(nrays)
                self.Jsp = np.zeros(nrays, dtype=complex)
                if withAmplitudes:
                    self.Es = np.zeros(nrays, dtype=complex)
                    self.Ep = np.zeros(nrays, dtype=complex)
        else:
            self.x = np.copy(copyFrom.x)
            self.y = np.copy(copyFrom.y)
            self.z = np.copy(copyFrom.z)
            self.sourceSIGMAx = copyFrom.sourceSIGMAx
            self.sourceSIGMAz = copyFrom.sourceSIGMAz
            self.filamentDX = copyFrom.filamentDX
            self.filamentDZ = copyFrom.filamentDZ
            self.filamentDtheta = copyFrom.filamentDtheta
            self.filamentDpsi = copyFrom.filamentDpsi
            self.state = np.copy(copyFrom.state)
            self.a = np.copy(copyFrom.a)
            self.b = np.copy(copyFrom.b)
            self.c = np.copy(copyFrom.c)
            self.path = np.copy(copyFrom.path)
            self.E = np.copy(copyFrom.E)
            self.Jss = np.copy(copyFrom.Jss)
            self.Jpp = np.copy(copyFrom.Jpp)
            self.Jsp = np.copy(copyFrom.Jsp)
            if withNumberOfReflections and hasattr(copyFrom, 'nRefl'):
                self.nRefl = np.copy(copyFrom.nRefl)
            if hasattr(copyFrom, 'elevationD'):
                self.elevationD = np.copy(copyFrom.elevationD)
                self.elevationX = np.copy(copyFrom.elevationX)
                self.elevationY = np.copy(copyFrom.elevationY)
                self.elevationZ = np.copy(copyFrom.elevationZ)
            if hasattr(copyFrom, 's'):
                self.s = np.copy(copyFrom.s)
            if hasattr(copyFrom, 'phi'):
                self.phi = np.copy(copyFrom.phi)
            if hasattr(copyFrom, 'r'):
                self.r = np.copy(copyFrom.r)
            if hasattr(copyFrom, 'theta'):
                self.theta = np.copy(copyFrom.theta)
            if hasattr(copyFrom, 'order'):
                self.order = np.copy(copyFrom.order)
            if hasattr(copyFrom, 'accepted'):  # for calculating flux
                self.accepted = copyFrom.accepted
                self.acceptedE = copyFrom.acceptedE
                self.seeded = copyFrom.seeded
                self.seededI = copyFrom.seededI
            if hasattr(copyFrom, 'Es'):
                self.Es = np.copy(copyFrom.Es)
                self.Ep = np.copy(copyFrom.Ep)
            if hasattr(copyFrom, 'area'):
                self.area = copyFrom.area

        if type(forceState) == int:
            self.state[:] = forceState

    def concatenate(self, beam):
        """Adds *beam* to *self*. Useful when more than one source is
        presented."""
        self.state = np.concatenate((self.state, beam.state))
        self.x = np.concatenate((self.x, beam.x))
        self.y = np.concatenate((self.y, beam.y))
        self.z = np.concatenate((self.z, beam.z))
        self.a = np.concatenate((self.a, beam.a))
        self.b = np.concatenate((self.b, beam.b))
        self.c = np.concatenate((self.c, beam.c))
        self.path = np.concatenate((self.path, beam.path))
        self.E = np.concatenate((self.E, beam.E))
        self.Jss = np.concatenate((self.Jss, beam.Jss))
        self.Jpp = np.concatenate((self.Jpp, beam.Jpp))
        self.Jsp = np.concatenate((self.Jsp, beam.Jsp))
        if hasattr(self, 'nRefl') and hasattr(beam, 'nRefl'):
            self.nRefl = np.concatenate((self.nRefl, beam.nRefl))
        if hasattr(self, 'elevationD') and hasattr(beam, 'elevationD'):
            self.elevationD = np.concatenate(
                (self.elevationD, beam.elevationD))
            self.elevationX = np.concatenate(
                (self.elevationX, beam.elevationX))
            self.elevationY = np.concatenate(
                (self.elevationY, beam.elevationY))
            self.elevationZ = np.concatenate(
                (self.elevationZ, beam.elevationZ))
        if hasattr(self, 's') and hasattr(beam, 's'):
            self.s = np.concatenate((self.s, beam.s))
        if hasattr(self, 'phi') and hasattr(beam, 'phi'):
            self.phi = np.concatenate((self.phi, beam.phi))
        if hasattr(self, 'r') and hasattr(beam, 'r'):
            self.r = np.concatenate((self.r, beam.r))
        if hasattr(self, 'theta') and hasattr(beam, 'theta'):
            self.theta = np.concatenate((self.theta, beam.theta))
        if hasattr(self, 'order') and hasattr(beam, 'order'):
            self.order = np.concatenate((self.order, beam.order))
        if hasattr(self, 'accepted') and hasattr(beam, 'accepted'):
            seeded = self.seeded + beam.seeded
            self.accepted = (self.accepted / self.seeded +
                             beam.accepted / beam.seeded) * seeded
            self.acceptedE = (self.acceptedE / self.seeded +
                              beam.acceptedE / beam.seeded) * seeded
            self.seeded = seeded
            self.seededI = self.seededI + beam.seededI
        if hasattr(self, 'Es') and hasattr(beam, 'Es'):
            self.Es = np.concatenate((self.Es, beam.Es))
            self.Ep = np.concatenate((self.Ep, beam.Ep))

    def filter_by_index(self, indarr):
        self.state = self.state[indarr]
        self.x = self.x[indarr]
        self.y = self.y[indarr]
        self.z = self.z[indarr]
        self.a = self.a[indarr]
        self.b = self.b[indarr]
        self.c = self.c[indarr]
        self.path = self.path[indarr]
        self.E = self.E[indarr]
        self.Jss = self.Jss[indarr]
        self.Jpp = self.Jpp[indarr]
        self.Jsp = self.Jsp[indarr]
        if hasattr(self, 'nRefl'):
            self.nRefl = self.nRefl[indarr]
        if hasattr(self, 'elevationD'):
            self.elevationD = self.elevationD[indarr]
            self.elevationX = self.elevationX[indarr]
            self.elevationY = self.elevationY[indarr]
            self.elevationZ = self.elevationZ[indarr]
        if hasattr(self, 's'):
            self.s = self.s[indarr]
        if hasattr(self, 'phi'):
            self.phi = self.phi[indarr]
        if hasattr(self, 'r'):
            self.r = self.r[indarr]
        if hasattr(self, 'theta'):
            self.theta = self.theta[indarr]
        if hasattr(self, 'order'):
            self.order = self.order[indarr]
        if hasattr(self, 'Es'):
            self.Es = self.Es[indarr]
            self.Ep = self.Ep[indarr]
        return self

    def replace_by_index(self, indarr, beam):
        self.state[indarr] = beam.state[indarr]
        self.x[indarr] = beam.x[indarr]
        self.y[indarr] = beam.y[indarr]
        self.z[indarr] = beam.z[indarr]
        self.a[indarr] = beam.a[indarr]
        self.b[indarr] = beam.b[indarr]
        self.c[indarr] = beam.c[indarr]
        self.path[indarr] = beam.path[indarr]
        self.E[indarr] = beam.E[indarr]
        self.Jss[indarr] = beam.Jss[indarr]
        self.Jpp[indarr] = beam.Jpp[indarr]
        self.Jsp[indarr] = beam.Jsp[indarr]
        if hasattr(self, 'nRefl') and hasattr(beam, 'nRefl'):
            self.nRefl[indarr] = beam.nRefl[indarr]
        if hasattr(self, 'elevationD') and hasattr(beam, 'elevationD'):
            self.elevationD[indarr] = beam.elevationD[indarr]
            self.elevationX[indarr] = beam.elevationX[indarr]
            self.elevationY[indarr] = beam.elevationY[indarr]
            self.elevationZ[indarr] = beam.elevationZ[indarr]
        if hasattr(self, 's') and hasattr(beam, 's'):
            self.s[indarr] = beam.s[indarr]
        if hasattr(self, 'phi') and hasattr(beam, 'phi'):
            self.phi[indarr] = beam.phi[indarr]
        if hasattr(self, 'r') and hasattr(beam, 'r'):
            self.r[indarr] = beam.r[indarr]
        if hasattr(self, 'theta') and hasattr(beam, 'theta'):
            self.theta[indarr] = beam.theta[indarr]
        if hasattr(self, 'order') and hasattr(beam, 'order'):
            self.order[indarr] = beam.order[indarr]
        if hasattr(self, 'Es') and hasattr(beam, 'Es'):
            self.Es[indarr] = beam.Es[indarr]
        if hasattr(self, 'Ep') and hasattr(beam, 'Ep'):
            self.Ep[indarr] = beam.Ep[indarr]
        return self

    def filter_good(self):
        return self.filter_by_index(self.state == 1)

    def absorb_intensity(self, inBeam, sign=1):
        self.Jss = (inBeam.Jss - self.Jss) * sign
        self.Jpp = (inBeam.Jpp - self.Jpp) * sign
        self.Jsp = (inBeam.Jsp - self.Jsp) * sign
        self.displayAsAbsorbedPower = True

    def add_wave(self, wave, sign=1):
        self.Es += sign*wave.Es
        self.Ep += sign*wave.Ep
        self.Jss = (self.Es * self.Es.conjugate()).real
        self.Jss = (self.Ep * self.Ep.conjugate()).real
        self.Jsp = self.Es * self.Ep.conjugate()

    def project_energy_to_band(self, EnewMin, EnewMax):
        """Uniformly projects the energy array self.E to a new band determined
        by *EnewMin* and *EnewMax*. This function is useful for simultaneous
        ray tracing of white beam and monochromatic beam parts of a beamline.
        """
        EoldMin = np.min(self.E)
        EoldMax = np.max(self.E)
        if EoldMin >= EoldMax:
            return
        self.E[:] = EnewMin +\
            (self.E-EoldMin) / (EoldMax-EoldMin) * (EnewMax-EnewMin)

    def make_uniform_energy_band(self, EnewMin, EnewMax):
        """Makes a uniform energy distribution. This function is useful for
        simultaneous ray tracing of white beam and monochromatic beam parts of
        a beamline.
        """
        self.E[:] = np.random.uniform(EnewMin, EnewMax, len(self.E))

    def diffract(self, wave):
        from . import waves as rw
        return rw.diffract(self, wave)


def copy_beam(
        beamTo, beamFrom, indarr, includeState=False, includeJspEsp=True):
    """Copies arrays of *beamFrom* to arrays of *beamTo*. The slicing of the
    arrays is given by *indarr*."""
    beamTo.x[indarr] = beamFrom.x[indarr]
    beamTo.y[indarr] = beamFrom.y[indarr]
    beamTo.z[indarr] = beamFrom.z[indarr]
    beamTo.a[indarr] = beamFrom.a[indarr]
    beamTo.b[indarr] = beamFrom.b[indarr]
    beamTo.c[indarr] = beamFrom.c[indarr]
    beamTo.path[indarr] = beamFrom.path[indarr]
    beamTo.E[indarr] = beamFrom.E[indarr]
    if includeState:
        beamTo.state[indarr] = beamFrom.state[indarr]
    if hasattr(beamFrom, 'nRefl') and hasattr(beamTo, 'nRefl'):
        beamTo.nRefl[indarr] = beamFrom.nRefl[indarr]
    if hasattr(beamFrom, 'order'):
        beamTo.order = beamFrom.order
    if hasattr(beamFrom, 'elevationD') and hasattr(beamTo, 'elevationD'):
        beamTo.elevationD[indarr] = beamFrom.elevationD[indarr]
        beamTo.elevationX[indarr] = beamFrom.elevationX[indarr]
        beamTo.elevationY[indarr] = beamFrom.elevationY[indarr]
        beamTo.elevationZ[indarr] = beamFrom.elevationZ[indarr]
    if hasattr(beamFrom, 'accepted'):
        beamTo.accepted = beamFrom.accepted
        beamTo.acceptedE = beamFrom.acceptedE
        beamTo.seeded = beamFrom.seeded
        beamTo.seededI = beamFrom.seededI
    if hasattr(beamTo, 'area'):
        beamTo.area = beamFrom.area
    if includeJspEsp:
        beamTo.Jss[indarr] = beamFrom.Jss[indarr]
        beamTo.Jpp[indarr] = beamFrom.Jpp[indarr]
        beamTo.Jsp[indarr] = beamFrom.Jsp[indarr]
        if hasattr(beamFrom, 'Es') and hasattr(beamTo, 'Es'):
            beamTo.Es[indarr] = beamFrom.Es[indarr]
            beamTo.Ep[indarr] = beamFrom.Ep[indarr]


def rotate_coherency_matrix(beam, indarr, roll):
    r"""Rotates the coherency matrix :math:`J`:

    .. math::

        J = \left( \begin{array}{ccc}
        J_{ss} & J_{sp} \\
        J^*_{sp} & J_{pp}\end{array} \right)

    by angle :math:`\phi` around the beam direction as :math:`J' = R_{\phi}
    J R^{-1}_{\phi}` with the rotation matrix :math:`R_{\phi}` defined as:

    .. math::

        R_{\phi} = \left( \begin{array}{ccc}
        \cos{\phi} & \sin{\phi} \\
        -\sin{\phi} & \cos{\phi}\end{array} \right)
    """
#    if (roll == 0).all():
#        return beam.Jss[indarr], beam.Jpp[indarr], beam.Jsp[indarr]
    c = np.cos(roll)
    s = np.sin(roll)
    c2 = c**2
    s2 = s**2
    cs = c * s
    JssN = beam.Jss[indarr]*c2 + beam.Jpp[indarr]*s2 +\
        2*beam.Jsp[indarr].real*cs
    JppN = beam.Jss[indarr]*s2 + beam.Jpp[indarr]*c2 -\
        2*beam.Jsp[indarr].real*cs
    JspN = (beam.Jpp[indarr]-beam.Jss[indarr])*cs +\
        beam.Jsp[indarr].real*(c2-s2) + beam.Jsp[indarr].imag*1j
    return JssN, JppN, JspN

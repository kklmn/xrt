# -*- coding: utf-8 -*-
import time
import numpy as np
import inspect

from .. import raycing
from . import sources as rs
from .physconsts import CH, CHBAR

__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "1 Feb 2026"


class OEMainMethods(object):
    """The base class hosting the main reflection methods."""

    @raycing.append_to_flow_decorator
    def reflect(self, beam=None, needLocal=True, noIntersectionSearch=False,
                returnLocalAbsorbed=None):
        r"""
        Returns the reflected or transmitted beam as :math:`\vec{out}` in
        global and local (if *needLocal* is true) systems.

        .. rubric:: Mirror [wikiSnell]_:

        .. math::

            \vec{out}_{\rm reflect} &= \vec{in} + 2\cos{\theta_1}\vec{n}\\
            \vec{out}_{\rm refract} &= \frac{n_1}{n_2}\vec{in} +
            \left(\frac{n_1}{n_2}\cos{\theta_1} - \cos{\theta_2}\right)\vec{n},

        where

        .. math::

            \cos{\theta_1} &= -\vec{n}\cdot\vec{in}\\
            \cos{\theta_2} &= sign(\cos{\theta_1})\sqrt{1 -
            \left(\frac{n_1}{n_2}\right)^2\left(1-\cos^2{\theta_1}\right)}.

        .. rubric:: Grating or FZP [SpencerMurty]_:

        For the reciprocal grating vector :math:`\vec{g}` and the :math:`m`\ th
        diffraction order:

        .. math::

            \vec{out} = \vec{in} - dn\vec{n} + \vec{g}m\lambda

        where

        .. math::

            dn = -\cos{\theta_1} \pm \sqrt{\cos^2{\theta_1} -
            2(\vec{g}\cdot\vec{in})m\lambda - \vec{g}^2 m^2\lambda^2}

        .. rubric:: Crystal [SanchezDelRioCerrina]_:

        Crystal (generally asymmetrically cut) is considered a grating with the
        reciprocal grating vector equal to

        .. math::

            \vec{g} = \left(\vec{n_\vec{H}} -
            (\vec{n_\vec{H}}\cdot\vec{n})\vec{n})\right) / d_\vec{H}.

        Note that :math:`\vec{g}` is along the line of the intersection of the
        crystal surface with the plane formed by the two normals
        :math:`\vec{n_\vec{H}}` and :math:`\vec{n}` and its length is
        :math:`|\vec{g}|=\sin{\alpha}/d_\vec{H}`, with :math:`\alpha`
        being the asymmetry angle.

        .. [wikiSnell] http://en.wikipedia.org/wiki/Snell%27s_law .
        .. [SpencerMurty] G. H. Spencer and M. V. R. K. Murty,
           J. Opt. Soc. Am. **52** (1962) 672.
        .. [SanchezDelRioCerrina] M. Sánchez del Río and F. Cerrina,
           Rev. Sci. Instrum. **63** (1992) 936.


        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam.

        *noIntersectionSearch*: bool
            Used in wave propagation, normally should be False. Certainly
            should be False if the OE is distorted or OE is parametric.


        .. .. Returned values: beamGlobal, beamLocal
        """
        if noIntersectionSearch and self.isParametric:
            raycing.colorPrint(
                'You should remove "noIntersectionSearch" for this parametric'
                'surface ({0})'.format(self.name), "RED")
        self.footprint = []
#        if self.bl is not None:
#            if raycing.is_valid_uuid(beam):
#                kwArgsIn['beam'] = beam
#                beam = self.bl.beamsDictU[beam]['beamGlobal']
#            else:
#                kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
#        self.get_orientation()
        # output beam in global coordinates

        gb = rs.Beam(copyFrom=beam)
        if needLocal:
            # output beam in local coordinates
            lb = rs.Beam(copyFrom=beam)
        else:
            lb = gb
        good = beam.state > 0
#        good = beam.state == 1
        if good.sum() == 0:
            return gb, lb
# coordinates in local virgin system:
        pitch = self.pitch
        if hasattr(self, 'bragg'):
            pitch += self.bragg
        raycing.global_to_virgin_local(self.bl, beam, lb, self.center, good)

        self._reflect_local(good, lb, gb,
                            pitch, self.roll+self.positionRoll, self.yaw,
                            self.dx, noIntersectionSearch=noIntersectionSearch,
                            material=self.material)
        if hasattr(beam, 'createdByDiffract'):
            goodAfter = gb.state == 1
        else:
            goodAfter = (gb.state == 1) | (gb.state == 2)
# in global coordinate system:
        if goodAfter.sum() > 0:
            raycing.virgin_local_to_global(self.bl, gb, self.center, goodAfter)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)
        if returnLocalAbsorbed is not None:
            absorbedLb = rs.Beam(copyFrom=lb)
            absorbedLb.absorb_intensity(beam)
            lb = absorbedLb
        raycing.append_to_flow(self.reflect, [gb, lb], inspect.currentframe())

#        gb.parentId = self.uuid
#        lb.parentId = self.uuid

        oq = self.get_orientation_quaternion()
        globalNorm = np.array(raycing.quat_vec_rotate(np.array([0, 0, 1]), oq))
        globalNorm /= np.linalg.norm(globalNorm)
#        print(self.name, "global norm", globalNorm)
        if hasattr(beam, 'basis'):
            newBasis = np.identity(3)
            for line in range(3):
                cmpt = beam.basis[line, :]
                newcmpt = cmpt - 2*np.dot(cmpt, globalNorm)*globalNorm
                newBasis[line, :] = newcmpt / np.linalg.norm(newcmpt)

            det = np.linalg.det(newBasis)
            if det < 0:
                newBasis[-1, :] *= -1
            gb.basis = newBasis
        # TODO: what if one element processing multiple beams?
#        self.bl.flowU[self.uuid] = {'method': self.reflect.__name__,
#                                    'kwArgsIn': kwArgsIn}
#        self.bl.beamsDictU[self.uuid] = {'beamGlobal': gb, 'beamLocal': lb}
        return gb, lb  # in global(gb) and local(lb) coordinates

    def multiple_reflect(
            self, beam=None, maxReflections=1000, needElevationMap=False,
            returnLocalAbsorbed=None):
        """
        Does the same as :meth:`reflect` but with up to *maxReflections*
        reflection on the same surface.

        The returned beam has additional fields: *nRefl* for the number of
        reflections, *elevationD* for the maximum elevation distance between
        the rays and the surface as the ray travels between the impact points,
        *elevationX*, *elevationY*, *elevationZ* for the coordinates of the
        maximum elevation points.

        *returnLocalAbsorbed*: None or int
            If not None, returns the absorbed intensity in local beam.


        .. .. Returned values: beamGlobal, beamLocal
        """
#        kwArgsIn = {'maxReflections': maxReflections,
#                    'needElevationMap': needElevationMap,
#                    'returnLocalAbsorbed': returnLocalAbsorbed}
        self.footprint = []
#        if self.bl is not None:
#            if raycing.is_valid_uuid(beam):
#                kwArgsIn['beam'] = beam
#                beam = self.bl.beamsDictU[beam]['beamGlobal']
#            else:
#                kwArgsIn['beam'] = beam.parentId
#            self.bl.auto_align(self, beam)
#        self.get_orientation()
# output beam in global coordinates
        gb = rs.Beam(copyFrom=beam)
        lb = gb
        good = beam.state > 0
        if good.sum() == 0:
            return gb, lb
# coordinates in local virgin system:
        raycing.global_to_virgin_local(self.bl, beam, lb, self.center, good)
        iRefl = 0
        isMulti = False
        while iRefl < maxReflections:
            tmpX, tmpY, tmpZ =\
                np.copy(lb.x[good]), np.copy(lb.y[good]), np.copy(lb.z[good])
            if raycing._VERBOSITY_ > 10:
                print('reflection No {0}'.format(iRefl + 1))
            if iRefl == 0:
                if needElevationMap:
                    lb.elevationD = -np.ones_like(lb.x)
                    lb.elevationX = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
                    lb.elevationY = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
                    lb.elevationZ = -np.ones_like(lb.x)*raycing.maxHalfSizeOfOE
            self._reflect_local(good, lb, gb, self.pitch,
                                self.roll+self.positionRoll, self.yaw,
                                self.dx, material=self.material,
                                needElevationMap=needElevationMap,
                                isMulti=isMulti)
            if iRefl == 0:
                isMulti = True
                lb.nRefl = np.zeros_like(lb.state)
            ov = lb.state[good] == 3
            lb.x[np.where(good)[0][ov]] = tmpX[ov]
            lb.y[np.where(good)[0][ov]] = tmpY[ov]
            lb.z[np.where(good)[0][ov]] = tmpZ[ov]
            good = (lb.state == 1) | (lb.state == 2)
            lb.nRefl[good] += 1
            if iRefl == 0:
                # all local footprints:
                lbN = rs.Beam(copyFrom=lb, withNumberOfReflections=True)
            else:
                lbN.concatenate(lb)
            iRefl += 1
            if raycing._VERBOSITY_ > 10:
                print('iRefl=', iRefl, 'remains=', good.sum())
#                if good.sum() > 0:
#                    print('y min max ', lb.y[good].min(), lb.y[good].max())
            if good.sum() == 0:
                break
#            gc.collect()
# in global coordinate system:
        goodAfter = gb.nRefl > 0
        gb.state[goodAfter] = 1
        if goodAfter.sum() > 0:
            raycing.virgin_local_to_global(self.bl, gb, self.center, goodAfter)
# not intersected rays remain unchanged except their state:
        notGood = ~goodAfter
        if notGood.sum() > 0:
            rs.copy_beam(gb, beam, notGood)
# in global(gb) and local(lbN) coordinates. lbN holds all the reflection spots.
        if returnLocalAbsorbed is not None:
            absorbedLb = rs.Beam(copyFrom=lb)
            absorbedLb.absorb_intensity(beam)
            lbN = absorbedLb
#        lbN.parentId = self.uuid
        raycing.append_to_flow(self.multiple_reflect, [gb, lbN],
                               inspect.currentframe())
#        self.bl.flowU[self.uuid] = {'method': self.multiple_reflect,
#                                    'kwArgsIn': kwArgsIn}
#        self.bl.beamsDictU[self.uuid] = {'beamGlobal': gb, 'beamLocal': lbN}
        return gb, lbN

    def prepare_wave(self, prevOE, nrays, shape='auto', area='auto', rw=None):
        """Creates the beam arrays used in wave diffraction calculations.
        *prevOE* is the diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`.
        *nrays*: if int, specifies the number of randomly distributed samples
        the surface within ``self.limPhysX`` limits; if 2-tuple of ints,
        specifies (nx, ny) sizes of a uniform mesh of samples.
        """

        if rw is None:
            from . import waves as rw

        if isinstance(nrays, (int, float)):
            nsamples = int(nrays)
        elif isinstance(nrays, (list, tuple)):
            nsamples = nrays[0] * nrays[1]
        else:
            raise ValueError('wrong type of `nrays`!')

        lb = rs.Beam(nrays=nsamples, forceState=1, withAmplitudes=True)
        lb.parentId = prevOE.uuid
        if isinstance(nrays, (int, float)):
            xy = np.random.rand(nsamples, 2)
            if shape == 'auto':
                shape = self.shape
            if shape.startswith('ro'):  # round
                dR = (self.limPhysX[1] - self.limPhysX[0]) / 2
                r = xy[:, 0]**0.5 * dR
                phi = xy[:, 1] * 2*np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                if area == 'auto':
                    area = np.pi * dR**2
            elif shape.startswith('re'):  # rect
                dX = self.limPhysX[1] - self.limPhysX[0]
                dY = self.limPhysY[1] - self.limPhysY[0]
                x = xy[:, 0] * dX + self.limPhysX[0]
                y = xy[:, 1] * dY + self.limPhysY[0]
                if area == 'auto':
                    area = dX * dY
            else:
                raise ValueError("unknown shape!")
        elif isinstance(nrays, (list, tuple)):
            if shape.startswith('ro'):  # round
                raise ValueError("must be rectangular")
            xx = np.linspace(*self.limPhysX, nrays[0])
            yy = np.linspace(*self.limPhysY, nrays[1])
            X, Y = np.meshgrid(xx, yy)
            x = X.ravel()
            y = Y.ravel()
            if area == 'auto':
                dX = self.limPhysX[1] - self.limPhysX[0]
                dY = self.limPhysY[1] - self.limPhysY[0]
                area = dX * dY
        else:
            raise ValueError('wrong type of `nrays`!')

        # These are approximate samples (exact for undistorted and
        # non-parametric cases). This works even for a parametric case because
        # `reflect()` (that follows `diffract()`) will make it exact. Make sure
        # that `noIntersectionSearch=False` in `reflect()`.
        z = self.local_z(x, y)
        lb.x[:] = x
        lb.y[:] = y
        lb.z[:] = z
        self.local_to_global(lb)
        lb.a[:] = lb.x - prevOE.center[0]
        lb.b[:] = lb.y - prevOE.center[1]
        lb.c[:] = lb.z - prevOE.center[2]
        norm = (lb.a**2 + lb.b**2 + lb.c**2)**0.5
        lb.a /= norm
        lb.b /= norm
        lb.c /= norm
        lb.x[:] = prevOE.center[0]
        lb.y[:] = prevOE.center[1]
        lb.z[:] = prevOE.center[2]

        lbn = rs.Beam(nrays=1)
        lbn.b[:] = 0.
        lbn.c[:] = 1.
        self.local_to_global(lbn)
        a = lbn.x - prevOE.center[0]
        b = lbn.y - prevOE.center[1]
        c = lbn.z - prevOE.center[2]
        norm = (a**2 + b**2 + c**2)**0.5
        areaNormalFact = \
            abs(float((a*lbn.a[0] + b*lbn.b[0] + c*lbn.c[0]) / norm))

        waveGlobal, waveLocal = self.reflect(lb)
        good = (waveLocal.state == 1) | (waveLocal.state == 2)
        waveGlobal.filter_by_index(good)
        waveLocal.filter_by_index(good)
        area *= good.sum() / float(len(good))
        waveLocal.area = area
        waveLocal.areaNormal = area * areaNormalFact
        waveLocal.dS = area / float(len(good))
        waveLocal.toOE = self
#        waveLocal.xGlobal = waveGlobal.x
#        waveLocal.yGlobal = waveGlobal.y
#        waveLocal.zGlobal = waveGlobal.z
        rw.prepare_wave(
            prevOE, waveLocal, waveGlobal.x, waveGlobal.y, waveGlobal.z)
        return waveLocal

    def propagate_wave(self, wave=None, beam=None, nrays='auto'):
        """
        Propagates the incoming *wave* through an optical element using the
        Kirchhoff diffraction theorem. Returnes two Beam objects, one in global
        and one in local coordinate systems, which can be
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
#        if wave is None and beam is not None:
#            wave = beam
        prevOE = self.bl.oesDict[wave.parentId][0]
        print("Diffract on", self.name, " Prev OE:", prevOE.name)
        if self.bl is not None:
            if beam is not None:
                self.bl.auto_align(self, beam)
            elif 'source' in str(type(prevOE)):
                self.bl.auto_align(self, wave)
            else:
                self.bl.auto_align(self, prevOE.local_to_global(
                    wave, returnBeam=True))
        waveOnSelf = self.prepare_wave(prevOE, waveSize, rw=rw)
        if 'source' in str(type(prevOE)):
            beamToSelf = prevOE.shine(wave=waveOnSelf)
            nIS = False
        else:
            beamToSelf = rw.diffract(wave, waveOnSelf)
            nIS = True
        retGlo, retLoc = self.reflect(beamToSelf, noIntersectionSearch=nIS)
        retLoc.parentId = self.uuid
        return retGlo, retLoc

    def _grating_deflection(
            self, goodN, lb, g, oeNormal, beamInDotNormal, order=1, sig=None):
        beamInDotG = lb.a[goodN]*g[0] + lb.b[goodN]*g[1] + lb.c[goodN]*g[2]
        G2 = g[0]**2 + g[1]**2 + g[2]**2
        locOrder = order if isinstance(order, int) else \
            np.array(order)[np.random.randint(len(order), size=goodN.sum())]
        lb.order = np.zeros(len(lb.a))
        lb.order[goodN] = locOrder
        orderLambda = locOrder * CH / lb.E[goodN] * 1e-7

        u = beamInDotNormal**2 - 2*beamInDotG*orderLambda - G2*orderLambda**2
#        u[u < 0] = 0
        gs = np.sign(beamInDotNormal) if sig is None else sig
        dn = beamInDotNormal + gs*np.sqrt(abs(u))
        a_out = lb.a[goodN] - oeNormal[-3]*dn + g[0]*orderLambda
        b_out = lb.b[goodN] - oeNormal[-2]*dn + g[1]*orderLambda
        c_out = lb.c[goodN] - oeNormal[-1]*dn + g[2]*orderLambda
        norm = (a_out**2 + b_out**2 + c_out**2)**0.5
        return a_out/norm, b_out/norm, c_out/norm

    def _reflect_crystal_cl(self, goodN, lb, matcr, oeNormal):
        DW = self.cl_precisionF(matcr.factDW)
        thickness = self.cl_precisionF(0 if matcr.t is None else matcr.t)
        geometry = np.int32(matcr.geometry)
        if matcr.tK is not None:
            temperature = self.cl_precisionF(matcr.tK)
        else:
            temperature = self.cl_precisionF(0)
        if not np.all(np.array(matcr.atoms) == 14):
            temperature = self.cl_precisionF(0)

        lenGood = np.int32(len(lb.E[goodN]))
        bOnes = np.ones(lenGood)

        iHKL = np.zeros(4, dtype=np.int32)
        iHKL[0:3] = np.array(matcr.hkl, dtype=np.int32)

        Nel = len(matcr.elements)
        elements_in = np.zeros((Nel, 8), dtype=self.cl_precisionF)
        E_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f1_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f2_in = np.zeros((Nel, 300), dtype=self.cl_precisionF)
        f0_in = np.zeros((Nel, 11), dtype=self.cl_precisionF)
        elements_in[:, 0:3] = matcr.atomsXYZ
        elements_in[:, 4] = matcr.atomsFraction

        for iNel in range(Nel):
            f_len = len(matcr.elements[iNel].E)
            elements_in[iNel, 5] = matcr.elements[iNel].Z
            elements_in[iNel, 6] = matcr.elements[iNel].mass
            elements_in[iNel, 7] = self.cl_precisionF(f_len-1)
            E_in[iNel, 0:f_len] = matcr.elements[iNel].E
            f1_in[iNel, 0:f_len] = matcr.elements[iNel].f1
            f2_in[iNel, 0:f_len] = matcr.elements[iNel].f2
            f0_in[iNel, :] = matcr.elements[iNel].f0coeffs

        lattice_in = np.array([matcr.a, matcr.b, matcr.c, 0,
                               matcr.alphaRad, matcr.betaRad, matcr.gammaRad,
                               0], dtype=self.cl_precisionF)
        calctype = 0
        if matcr.kind == "powder":
            calctype = 5
        elif matcr.kind == "monocrystal":
            calctype = 10 + matcr.Nmax
        elif matcr.kind == "crystal harmonics":
            calctype = 100 + matcr.Nmax

        scalarArgs = [np.int32(calctype), iHKL, DW, thickness, temperature,
                      geometry, np.int32(Nel), lattice_in]

        slicedROArgs = [self.cl_precisionF(lb.a[goodN]),  # a_in
                        self.cl_precisionF(lb.b[goodN]),  # b_in
                        self.cl_precisionF(lb.c[goodN]),  # c_in
                        self.cl_precisionF(lb.E[goodN]),  # Energy
                        self.cl_precisionF(oeNormal[0]*bOnes),  # planeNormalX
                        self.cl_precisionF(oeNormal[1]*bOnes),  # planeNormalY
                        self.cl_precisionF(oeNormal[2]*bOnes),  # planeNormalZ
                        self.cl_precisionF(oeNormal[-3]*bOnes),  # surfNormalX
                        self.cl_precisionF(oeNormal[-2]*bOnes),  # surfNormalY
                        self.cl_precisionF(oeNormal[-1]*bOnes)]  # surfNormalZ

        slicedROArgs.extend([self.cl_precisionF(np.random.rand(lenGood))])

        nonSlicedROArgs = [elements_in.flatten(),  # elements
                           f0_in.flatten(),  # f0
                           E_in.flatten(),   # E_in
                           f1_in.flatten(),  # f1
                           f2_in.flatten()]  # f2

        slicedRWArgs = [np.zeros(lenGood, dtype=self.cl_precisionC),  # reflS
                        np.zeros(lenGood, dtype=self.cl_precisionC),  # reflP
                        np.zeros(lenGood, dtype=self.cl_precisionF),  # a_out
                        np.zeros(lenGood, dtype=self.cl_precisionF),  # b_out
                        np.zeros(lenGood, dtype=self.cl_precisionF)]  # c_out

        curveS, curveP, a_out, b_out, c_out = self.ucl.run_parallel(
            'reflect_crystal', scalarArgs, slicedROArgs, nonSlicedROArgs,
            slicedRWArgs, None, lenGood)
        return a_out, b_out, c_out, curveS, curveP

    def _reflect_local(
        self, good, lb, vlb, pitch, roll, yaw, dx=None, dy=None, dz=None,
        local_z=None, local_n=None, local_g=None, fromVacuum=True,
        material=None, is2ndXtal=False, needElevationMap=False,
            noIntersectionSearch=False, isMulti=False):
        """Finds the intersection points of rays in the beam *lb* indexed by
        *good* array. *vlb* is the same beam in virgin local system.
        *pitch, roll, yaw* determine the transformation between true local and
        virgin local coordinates.
        *local_n* gives the normal (two normals (h, surface) if for crystal).
        *local_g* for a grating gives the local reciprocal groove vector in
        1/mm. *fromVacuum* tells the beam direction for the vacuum-OE
        interface. *material* is an instance of :class:`Material` or
        :class:`Crystal` or its derivatives. Depending on the geometry used, it
        must have either the method :meth:`get_refractive_index` or the
        :meth:`get_amplitude`."""

        def _get_asymmetric_reflection_grating(
                _gNormal, _oeNormal, _beamInDotSurfaceNormal,
                _beamInDotNormal, xd=None, yd=None):
            normalDotSurfNormal = _oeNormal[0]*_oeNormal[-3] +\
                _oeNormal[1]*_oeNormal[-2] + _oeNormal[2]*_oeNormal[-1]
            bdn = _beamInDotNormal.sum() / len(_beamInDotNormal)
            sgbdn = 1 if bdn < 0 else -1
            # note:
            # _oeNormal[0:3] is n_B
            # _oeNormal[-3:] is n_s
            # normalDotSurfNormal is dot(n_B, n_s)
            # |vec(n_B) - dot(n_B, n_s)*vec(n_s)| = sin(alpha)
#            wH = matSur.get_refractive_correction(
#                lb.E[goodN], abs(_beamInDotSurfaceNormal))
            wH = 0
            if hasattr(matSur, 'get_d') and xd is not None and yd is not None:
                crystd = matSur.get_d(xd, yd)
            else:
                crystd = matSur.d

            wHd = (1 + wH) / (crystd * 1e-7)
            gNormalCryst = np.asarray((
                (_oeNormal[0]-normalDotSurfNormal*_oeNormal[-3]) * wHd,
                (_oeNormal[1]-normalDotSurfNormal*_oeNormal[-2]) * wHd,
                (_oeNormal[2]-normalDotSurfNormal*_oeNormal[-1]) * wHd),
                order='F') * sgbdn
            if matSur.geom.endswith('Fresnel'):
                if isinstance(self.order, int):
                    locOrder = self.order
                else:
                    locOrder = np.array(self.order)[np.random.randint(
                        len(self.order), size=goodN.sum())]
                if _gNormal is None:
                    _gNormal = local_g(lb.x[goodN], lb.y[goodN])
                _gNormal = np.asarray(_gNormal, order='F') * locOrder
                _gNormal[0] += gNormalCryst[0]
                _gNormal[1] += gNormalCryst[1]
                _gNormal[2] += gNormalCryst[2]
            else:
                _gNormal = gNormalCryst

            sg = 1 if matSur.geom.startswith('Laue') else -1
            res = self._grating_deflection(
                goodN, lb, _gNormal, _oeNormal, _beamInDotSurfaceNormal, 1, sg)
            return res

# rotate the world around the mirror.
# lb is truly local coordinates whereas vlb is in virgin local coordinates:
        if local_n is None:
            local_n = self.local_n
        extraAnglesSign = 1.  # only for pitch and yaw
        if is2ndXtal:
            raycing.rotate_beam(lb, good, roll=-np.pi)
            extraAnglesSign = -1.  # only for pitch and yaw
        raycing.rotate_beam(
            lb, good, rotationSequence=self.rotationSequence,
            pitch=-pitch, roll=-roll, yaw=-yaw)
        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                lb, good, rotationSequence=self.extraRotationSequence,
                pitch=-extraAnglesSign*self.extraPitch, roll=-self.extraRoll,
                yaw=-extraAnglesSign*self.extraYaw)
        if dx:
            lb.x[good] -= dx
        if dy:
            lb.y[good] -= dy
        if dz:
            lb.z[good] -= dz

# x, y, z:
        if hasattr(self, 'invertNormal'):
            invertNormal = self.invertNormal
        else:
            invertNormal = 1 if fromVacuum else -1

#        mainPartForBracketing = lb.state[good] > 0
        mainPartForBracketing = lb.state[good] == 1
        tMin = np.zeros_like(lb.x)
        tMax = np.zeros_like(lb.x)
        tMin[good], tMax[good], elev = self._bracketing(
            local_n, lb.x[good], lb.y[good], lb.z[good], lb.a[good],
            lb.b[good], lb.c[good], invertNormal, is2ndXtal, isMulti=isMulti,
            needElevationMap=needElevationMap, mainPart=mainPartForBracketing)
        if needElevationMap and elev:
            lb.elevationD[good] = elev[0]
            if self.isParametric:
                tX, tY, tZ = self.param_to_xyz(elev[1], elev[2], elev[3])
            else:
                tX, tY, tZ = elev[1], elev[2], elev[3]
            lb.elevationX[good] = tX
            lb.elevationY[good] = tY
            lb.elevationZ[good] = tZ

        _lost = None
        if noIntersectionSearch:
            # lb.x[good], lb.y[good], lb.z[good] unchanged
            tMax[good] = 0.
            if self.isParametric:
                lb.x[good], lb.y[good], lb.z[good] = self.xyz_to_param(
                    lb.x[good], lb.y[good], lb.z[good])
        else:
            if True:  # self.cl_ctx is None:
                res_find = \
                    self.find_intersection(
                        local_z, tMin[good], tMax[good],
                        lb.x[good], lb.y[good], lb.z[good],
                        lb.a[good], lb.b[good], lb.c[good], invertNormal)
                tMax[good], lb.x[good], lb.y[good], lb.z[good] = res_find[:4]
                if len(res_find) > 4:
                    _lost = res_find[4]
            else:  # To be refactored in future versions
                tMax[good], lb.x[good], lb.y[good], lb.z[good] = \
                    self.find_intersection_CL(
                        local_z, tMin[good], tMax[good],
                        lb.x[good], lb.y[good], lb.z[good],
                        lb.a[good], lb.b[good], lb.c[good], invertNormal)
# state:
# the distortion part has moved from here to find_dz
        if self.isParametric:
            tX, tY, tZ = self.param_to_xyz(lb.x[good], lb.y[good], lb.z[good])
        else:
            tX, tY, tZ = lb.x[good], lb.y[good], lb.z[good]

        if self.use_rays_good_gn:
            lb.state[good], gNormal = self.rays_good_gn(tX, tY, tZ)
        else:
            gNormal = None
            lb.state[good] = self.rays_good(tX, tY, tZ, is2ndXtal)
        if _lost is not None:
            lb.state[np.where(good)[0][_lost]] = self.lostNum

#        goodN = (lb.state == 1) | (lb.state == 2)
        goodN = (lb.state == 1)
# normal at x, y, z:
        goodNsum = goodN.sum()
        needMosaicity = False
        crystalVD = False
        if goodNsum > 0:
            lb.path[goodN] += tMax[goodN]

            toWhere = 0  # 0: reflect, 1: refract, 2: pass straight
            if material is not None:
                if raycing.is_sequence(material):
                    matSur = material[self.curSurface]
                else:
                    matSur = material

                if matSur.kind == 'auto':
                    self.assign_auto_material_kind(matSur)
                if matSur.kind in ('plate', 'lens'):
                    toWhere = 1
                elif matSur.kind in ('crystal', 'multilayer'):
                    if matSur.kind == 'crystal':
                        if matSur.mosaicity:
                            needMosaicity = True
                        if hasattr(matSur, 'volumetricDiffraction'):
                            if matSur.volumetricDiffraction:
                                crystalVD = True
                    if matSur.geom.endswith('transmitted'):
                        toWhere = 2
                elif matSur.kind == 'grating':
                    toWhere = 3
                elif matSur.kind == 'FZP':
                    toWhere = 4
                elif matSur.kind == 'powder':
                    toWhere = 5
                elif matSur.kind == 'monocrystal':
                    toWhere = 6
                elif matSur.kind == 'crystal harmonics':
                    toWhere = 7

            if toWhere == 5:
                oeNormal = list(
                    self.local_n_random(len(lb.E[goodN]), matSur.chi))
#                n = matSur.get_refractive_index(lb.E[goodN])
#                mu = abs(n.imag) * lb.E[goodN] / CHBAR * 2e8  # 1/cm
#                att = np.exp(-mu * tMax[goodN] * 0.1)
                depth = np.random.rand(len(lb.a[goodN])) * matSur.t
                lb.x[goodN] += lb.a[goodN] * depth
                lb.y[goodN] += lb.b[goodN] * depth
                lb.z[goodN] += lb.c[goodN] * depth
            else:
                oeNormal = list(local_n(lb.x[goodN], lb.y[goodN]))

            n_distorted = self.local_n_distorted(lb.x[goodN], lb.y[goodN])
            if n_distorted is not None:
                if len(n_distorted) == 2:
                    cosX, sinX = np.cos(n_distorted[0]), np.sin(n_distorted[0])
                    oeNormal[-2], oeNormal[-1] = raycing.rotate_x(
                        oeNormal[-2], oeNormal[-1], cosX, sinX)
                    cosY, sinY = np.cos(n_distorted[1]), np.sin(n_distorted[1])
                    oeNormal[-3], oeNormal[-1] = raycing.rotate_y(
                        oeNormal[-3], oeNormal[-1], cosY, sinY)
                elif len(n_distorted) == 3:
                    oeNormal[-3] += n_distorted[0]
                    oeNormal[-2] += n_distorted[1]
                    oeNormal[-1] += n_distorted[2]
                    norm = (oeNormal[-3]**2 + oeNormal[-2]**2 +
                            oeNormal[-1]**2)**0.5
                    oeNormal[-3] /= norm
                    oeNormal[-2] /= norm
                    oeNormal[-1] /= norm
                else:
                    raise ValueError(
                        "wrong length returned by 'local_n_distorted'")
            if toWhere < 5:
                isAsymmetric = len(oeNormal) == 6
                oeNormal = np.asarray(oeNormal, order='F')
                beamInDotNormal = lb.a[goodN]*oeNormal[0] +\
                    lb.b[goodN]*oeNormal[1] + lb.c[goodN]*oeNormal[2]
                lb.theta = np.zeros_like(lb.x)
                beamInDotNormal[beamInDotNormal < -1] = -1
                beamInDotNormal[beamInDotNormal > 1] = 1
                lb.theta[goodN] = np.arccos(beamInDotNormal) - np.pi/2

                if isAsymmetric:
                    beamInDotSurfaceNormal = lb.a[goodN]*oeNormal[-3] +\
                        lb.b[goodN]*oeNormal[-2] + lb.c[goodN]*oeNormal[-1]
                    # The following code will consider finite thickness of
                    # the Laue crystal
                    if crystalVD:
                        # Max depth for flat crystal.
                        thMax = -matSur.t / beamInDotSurfaceNormal
                        # Point of diffraction along the path is approximated
                        # with uniform distribution.
                        dpth = np.random.rand(len(lb.a[goodN])) * thMax
                        # dpth = np.linspace(0, 1, len(lb.a[goodN])) * thMax
                        lb.x[goodN] += lb.a[goodN] * dpth
                        lb.y[goodN] += lb.b[goodN] * dpth
                        lb.z[goodN] += lb.c[goodN] * dpth
                        # Local orientation of the plane inside the crystal
                        deepNormal = list(self.local_n_depth(
                                lb.x[goodN], lb.y[goodN], lb.z[goodN]))
                        oeNormal[0:3] = deepNormal[0:3]
                        beamInDotNormal =\
                            lb.a[goodN]*oeNormal[0] +\
                            lb.b[goodN]*oeNormal[1] +\
                            lb.c[goodN]*oeNormal[2]
                        lb.theta = np.zeros_like(lb.x)
                        beamInDotNormal[beamInDotNormal < -1] = -1
                        beamInDotNormal[beamInDotNormal > 1] = 1
                        lb.theta[goodN] = np.arccos(beamInDotNormal)\
                            - np.pi/2
                else:
                    beamInDotSurfaceNormal = beamInDotNormal

                if needMosaicity:
                    oeNormalN, beamInDotNormalN = self._mosaic_normal(
                        matSur, oeNormal, beamInDotNormal, lb, goodN)
                    if isAsymmetric:
                        o1 = np.ones_like(lb.a[goodN])
                        oeNormalN.extend([
                            oeNormal[-3]*o1, oeNormal[-2]*o1, oeNormal[-1]*o1])
                    oeNormal = np.asarray(oeNormalN, order='F')
                    beamInDotNormalOld = beamInDotNormal
                    beamInDotNormal = beamInDotNormalN
# direction:
            if toWhere in [3, 4]:  # grating, FZP
                if gNormal is None:
                    if self.isParametric:
                        tXN, tYN = self.param_to_xyz(
                            lb.x[goodN], lb.y[goodN], lb.z[goodN])[0:2]
                    else:
                        tXN, tYN = lb.x[goodN], lb.y[goodN]
                    if local_g is None:
                        local_g = self.local_g
                    gNormal = np.asarray(local_g(tXN, tYN), order='F')
                    if toWhere == 4:  # FZP
                        gN = (oeNormal[-3]*gNormal[0] + oeNormal[-2]*gNormal[1]
                              + oeNormal[-1]*gNormal[2])
                        if not np.allclose(gN, np.zeros_like(gN)):
                            print("Warning: "
                                  "local_g is not orthogonal to local_n!")

                giveSign = 1 if toWhere == 4 else -1
                lb.a[goodN], lb.b[goodN], lb.c[goodN] =\
                    self._grating_deflection(goodN, lb, gNormal, oeNormal,
                                             beamInDotSurfaceNormal,
                                             self.order, giveSign)
            elif toWhere in [0, 2]:  # reflect, straight
                useAsymmetricNormal = False
                if material is not None:
                    if matSur.kind in ('crystal', 'multilayer') and\
                            toWhere == 0 and (not needMosaicity) and\
                            (not crystalVD):
                        useAsymmetricNormal = True

                if useAsymmetricNormal:
                    a_out, b_out, c_out = _get_asymmetric_reflection_grating(
                        gNormal, oeNormal, beamInDotSurfaceNormal,
                        beamInDotNormal, lb.x[goodN], lb.y[goodN])
                else:
                    a_out = lb.a[goodN] - oeNormal[0]*2*beamInDotNormal
                    b_out = lb.b[goodN] - oeNormal[1]*2*beamInDotNormal
                    c_out = lb.c[goodN] - oeNormal[2]*2*beamInDotNormal

                if toWhere == 0:  # reflect
                    if needMosaicity:
                        lb.olda = np.array(lb.a[goodN])
                        lb.oldb = np.array(lb.b[goodN])
                        lb.oldc = np.array(lb.c[goodN])
                        lb.oldJss = np.array(lb.Jss[goodN])
                        lb.oldJpp = np.array(lb.Jpp[goodN])
                        lb.oldJsp = np.array(lb.Jsp[goodN])
                        if hasattr(lb, 'Es'):
                            lb.oldEs = np.array(lb.Es[goodN])
                            lb.oldEp = np.array(lb.Ep[goodN])

                    lb.a[goodN] = a_out
                    lb.b[goodN] = b_out
                    lb.c[goodN] = c_out
            elif toWhere == 1:  # refract
                refractive_index = \
                    matSur.get_refractive_index(lb.E[goodN]).real
                if fromVacuum:
                    n1overn2 = 1. / refractive_index
                else:
                    n1overn2 = refractive_index
                signN = np.sign(-beamInDotNormal)
                n1overn2cosTheta1 = -n1overn2 * beamInDotNormal
                cosTheta2 = signN * \
                    np.sqrt(1 - n1overn2**2 + n1overn2cosTheta1**2)
                dn = (n1overn2cosTheta1 - cosTheta2)
                lb.a[goodN] = lb.a[goodN] * n1overn2 + oeNormal[0]*dn
                lb.b[goodN] = lb.b[goodN] * n1overn2 + oeNormal[1]*dn
                lb.c[goodN] = lb.c[goodN] * n1overn2 + oeNormal[2]*dn
            elif toWhere in [5, 6, 7]:  # powder, 'monocrystal', 'harmonics'
                trc0 = time.time()
                aP, bP, cP, rasP, rapP =\
                    self._reflect_crystal_cl(goodN, lb, matSur, oeNormal)
                print('Reflect_crystal completed in {0} s'.format(
                    time.time() - trc0))
#                lb.concatenate(lb)
                lb.a[goodN] = aP
                lb.b[goodN] = bP
                lb.c[goodN] = cP
                goodN = (lb.state == 1) | (lb.state == 2)
#                good = np.append(good, good)
            else:  # pass straight, do nothing
                pass
# flux:
            findReflectivity = False
            if material is not None:
                if hasattr(matSur, 'get_amplitude'):
                    findReflectivity = True
                if toWhere in [5, 6, 7]:  # powder,
                    findReflectivity = True

            # rotate coherency matrix:
            # {np.arctan2: 0./0.: =0, 1./0.: =pi/2}
            rollAngle = roll + np.arctan2(oeNormal[-3], oeNormal[-1])
            localJ = rs.rotate_coherency_matrix(lb, goodN, -rollAngle)
            if hasattr(lb, 'Es'):
                cosY, sinY = np.cos(rollAngle), np.sin(rollAngle)
                lb.Es[goodN], lb.Ep[goodN] = raycing.rotate_y(
                    lb.Es[goodN], lb.Ep[goodN], cosY, -sinY)

            if findReflectivity:
                if toWhere in [5, 6, 7]:  # powder,
                    refl = rasP, rapP
                elif matSur.kind == 'crystal':
                    beamOutDotSurfaceNormal = a_out*oeNormal[-3] + \
                        b_out*oeNormal[-2] + c_out*oeNormal[-1]
                    if needMosaicity:
                        refl = matSur.get_amplitude_mosaic(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormalOld)
                    elif matSur.useTT:
                        Ry = self.R if hasattr(self, 'R') else self.Rm \
                            if hasattr(self, 'Rm') else None
                        lcname = self.__class__.__name__.lower()
                        if 'johansson' in lcname or 'ground' in lcname:
                            Ry *= 2
                        Rx = self.Rs if hasattr(self, 'Rs') else None
                        refl = matSur.get_amplitude_pytte(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal,
                            alphaAsym=self.alpha,
                            Ry=Ry, Rx=Rx, ucl=self.ucl)

                        # if '_R' in self.__dict__.keys():
                        #     Ry = self.R
                        # elif '_Rm' in self.__dict__.keys():
                        #     Ry = self.Rm
                        # else:
                        #     Ry = None
                        # refl = matSur.get_amplitude_pytte(
                        #     lb.E[goodN], beamInDotSurfaceNormal,
                        #     beamOutDotSurfaceNormal, beamInDotNormal,
                        #     alphaAsym=self.alpha,
                        #     Ry=Ry, Rx=self.Rs if 'Rs' in self.__dict__.keys()
                        #     else None,
                        #     ucl=self.ucl)
                    else:
                        refl = matSur.get_amplitude(
                            lb.E[goodN], beamInDotSurfaceNormal,
                            beamOutDotSurfaceNormal, beamInDotNormal,
                            lb.x[goodN], lb.y[goodN])
                elif matSur.kind == 'multilayer':
                    refl = matSur.get_amplitude(
                        lb.E[goodN], beamInDotSurfaceNormal,
                        lb.x[goodN], lb.y[goodN],
                        ucl=self.ucl)
                else:  # 'mirror', 'thin mirror', 'plate', 'lens', 'grating'
                    hasEfficiency = False
                    if hasattr(matSur, 'efficiency'):
                        if (matSur.kind in ('grating', 'FZP')) and\
                                (matSur.efficiency is not None):
                            hasEfficiency = True
                    if hasEfficiency:
                        refl = matSur.get_grating_efficiency(lb, goodN)
                    else:
                        refl = matSur.get_amplitude(
                            lb.E[goodN], beamInDotNormal, fromVacuum)
            else:
                refl = 1., 1.
            ras, rap = refl[0], refl[1]
            nanSum = np.isnan(ras).sum()
            if nanSum > 0:
                ras[np.isnan(ras)] = 0.
#                    self._reportNaN(ras, 'ras')
            nanSum = np.isnan(rap).sum()
            if nanSum > 0:
                rap[np.isnan(rap)] = 0.
#                    self._reportNaN(rap, 'rap')

            lb.Jss[goodN] = (localJ[0] * ras * np.conjugate(ras)).real
            lb.Jpp[goodN] = (localJ[1] * rap * np.conjugate(rap)).real
            lb.Jsp[goodN] = localJ[2] * ras * np.conjugate(rap)
#                self._reportNaN(lb.Jss[goodN], 'lb.Jss[goodN]')
#                self._reportNaN(lb.Jpp[goodN], 'lb.Jpp[goodN]')
#                self._reportNaN(lb.Jsp[goodN], 'lb.Jsp[goodN]')
            if hasattr(lb, 'Es'):
                lb.Es[goodN] *= ras
                lb.Ep[goodN] *= rap

            if (not fromVacuum) and material is not None and\
                    matSur.kind not in ('crystal', 'multilayer'):
                # tMax in mm, refl[2]=mu0 in 1/cm
                att = np.exp(-refl[2] * tMax[goodN] * 0.1)
                lb.Jss[goodN] *= att
                lb.Jpp[goodN] *= att
                lb.Jsp[goodN] *= att
                if hasattr(lb, 'Es'):
                    # refl[3] = n.real * k in 1/cm
                    mPh = att**0.5 * np.exp(0.1j * refl[3] * tMax[goodN])
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh
            else:
                if hasattr(lb, 'Es'):
                    mPh = np.exp(1e7j * lb.E[goodN]/CHBAR * tMax[goodN])
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh

        if self.isParametric:
            lb.s = np.copy(lb.x)
            lb.phi = np.copy(lb.y)
            lb.r = np.copy(lb.z)
            lb.x[good], lb.y[good], lb.z[good] = self.param_to_xyz(
                lb.x[good], lb.y[good], lb.z[good])

        if goodNsum > 0:
            if needMosaicity:  # secondary extinction and attenuation
                length, through = self._mosaic_length(
                    matSur, beamInDotSurfaceNormal, lb, goodN)
                n = matSur.get_refractive_index(lb.E[goodN])
                if through is not None:  # if mat.t
                    # using double slicing, see
                    # stackoverflow.com/questions/1687566/why-does-an
                    # -assignment-for-double-sliced-numpy-arrays-not-work
                    lb.a[np.where(goodN)[0][through]] = lb.olda[through]
                    lb.b[np.where(goodN)[0][through]] = lb.oldb[through]
                    lb.c[np.where(goodN)[0][through]] = lb.oldc[through]
                    att = np.exp(-abs(n.imag) * lb.E[goodN] / CHBAR * 2e8 *
                                 length * 0.1)
                    lb.Jss[np.where(goodN)[0][through]] =\
                        lb.oldJss[through] * att[through]
                    lb.Jpp[np.where(goodN)[0][through]] = \
                        lb.oldJpp[through] * att[through]
                    lb.Jsp[np.where(goodN)[0][through]] = \
                        lb.oldJsp[through] * att[through]
                    if hasattr(lb, 'Es'):
                        lb.Es[np.where(goodN)[0][through]] = lb.oldEs[through]
                        lb.Ep[np.where(goodN)[0][through]] = lb.oldEp[through]
                if hasattr(lb, 'Es'):
                    nk = n.real * lb.E[goodN] / CHBAR * 1e8  # [1/cm]
                    mPh = np.exp(1j * nk * 0.2*length)  # *2: in and out
                    if through is not None:
                        mPh[through] =\
                            (att**0.5 * np.exp(1j*nk*0.1*length))[through]
                    lb.Es[goodN] *= mPh
                    lb.Ep[goodN] *= mPh

# rotate coherency matrix back:
            vlb.Jss[goodN], vlb.Jpp[goodN], vlb.Jsp[goodN] =\
                rs.rotate_coherency_matrix(lb, goodN, rollAngle)
            if hasattr(lb, 'Es'):
                vlb.Es[goodN], vlb.Ep[goodN] = raycing.rotate_y(
                    lb.Es[goodN], lb.Ep[goodN], cosY, sinY)

        if vlb is not lb:
            # includeJspEsp=False because Jss, Jpp, Jsp, Es and Ep are in vlb
            # already:
            rs.copy_beam(vlb, lb, good, includeState=True, includeJspEsp=False)
# rotate the world back for the virgin local beam:
        if dx:
            vlb.x[good] += dx
        if dy:
            vlb.y[good] += dy
        if dz:
            vlb.z[good] += dz
        if self.extraPitch or self.extraRoll or self.extraYaw:
            raycing.rotate_beam(
                vlb, good, rotationSequence='-'+self.extraRotationSequence,
                pitch=extraAnglesSign*self.extraPitch, roll=self.extraRoll,
                yaw=extraAnglesSign*self.extraYaw)
        raycing.rotate_beam(vlb, good,
                            rotationSequence='-'+self.rotationSequence,
                            pitch=pitch, roll=roll, yaw=yaw)
        if is2ndXtal:
            raycing.rotate_beam(vlb, good, roll=np.pi)
        self.footprint.extend([np.hstack((np.min(np.vstack((
            vlb.x[good], vlb.y[good], vlb.z[good])), axis=1),
            np.max(np.vstack((vlb.x[good], vlb.y[good], vlb.z[good])),
                   axis=1))).reshape(2, 3)])
#        print(self.name, self.footprint)
        if self.alarmLevel is not None:
            raycing.check_alarm(self, good, vlb)

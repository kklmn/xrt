# -*- coding: utf-8 -*-
import time
import numpy as np
# from scipy.special import jn as besselJn
from ..physconsts import PI, PI2, CH, CHBAR, R0, SQRT2PI
from .material import Material

ch = CH  # left here for copatibility
chbar = CHBAR  # left here for copatibility

try:
    from ..pyTTE_x import TTcrystal, Quantity
    isPyTTE = True
    # print("Importing pyTTE")
except ImportError:
    isPyTTE = False
    # print("pyTTE not found")


class Crystal(Material):
    u"""The parent class for crystals. The descendants must define
    :meth:`get_structure_factor`. :class:`Crystal` gives reflectivity and
    transmittivity of a crystal in Bragg and Laue cases."""

    hiddenParams = {'kind'}

    def __init__(self, hkl=[1, 1, 1], d=0, V=None, elements='Si',
                 quantities=None, rho=0, t=None, factDW=1.,
                 geom='Bragg reflected', table='Chantler total', name='',
                 volumetricDiffraction=False, useTT=False, nu=None,
                 mosaicity=0, **kwargs):
        u"""
        *hkl*: sequence
            hkl indices.

        *d*: float
            Interatomic spacing in Å.

        *V*: float
            Unit cell volume in Å³. If not given, is calculated from *d*
            assuming a cubic symmetry.

        *factDW*: float
            Debye-Waller factor applied to the structure factor.

        *geom*: str
            The 1st word is either 'Bragg' or 'Laue', the 2nd word is either
            'transmitted' or 'reflected' or 'Fresnel' (the optical element must
            then provide `local_g` method that gives the grating vector).

        *table*: str
            This parameter is explained in the description of the parent class
            :class:`Material`.

        .. _volumetricDiffraction:

        *volumetricDiffraction*: bool
            By default the diffracted ray originates in the point of incidence
            on the surface of the crystal in both Bragg and Laue case. When
            volumetricDiffraction is enabled, the point of diffraction is
            generated randomly along the transmitted beam path, effectively
            broadening the meridional beam profile in plain Laue crystal. If
            the crystal is bent, local deformation of the diffracting plane is
            taken into account, creating the polychromatic focusing effect.

        .. _useTT:

        *useTT*: bool
            Specifies whether the reflectivity is calculated by analytical
            formula (*useTT* is False) or by solution of the Takagi-Taupin
            equations (when *useTT* is True). The latter case is based on PyTTE
            code [PyTTE1]_ [PyTTE2]_ that was adapted to running the
            calculations on GPUs.

            .. [PyTTE1] https://github.com/aripekka/pyTTE

            .. [PyTTE2] A.-P. Honkanen, S. Huotari, IUCrJ 8 (2021) 102-115.
               doi:10.1107/S2052252520014165

            .. warning::
                You need a good graphics card to run these calculations!
                The corresponding optical element, that utilizes the present
                crystal material class, must specify *targetOpenCL* (typically,
                'auto') and *precisionOpenCL* (in Bragg cases 'float32' is
                typically sufficient and 'float64' is typically needed in Laue
                cases).

        *nu*: float
            Poisson's ratio. Can be used for calculation of reflectivity
            in bent isotropic crystals with [PyTTE1]_. Not required for plain
            crystals or for crystals with predefined compliance matrix, see
            :mod:`~xrt.backends.raycing.materials.crystals`. If
            provided, overrides existing compliance matrix.

        *mosaicity*: float, radians
            The sigma of the normal distribution of the crystallite normals.

            xrt follows the concept of mosaic crystals from
            [SanchezDelRioMosaic]_. This concept has three main parts: (i) a
            random distribution of the crystallite normals results in a
            distribution in the reflected directions, (ii) the secondary
            extinction results in a mean free path distribution of the new ray
            origins and (iii) the reflectivity is calculated following the work
            [BaconLowde]_.

            In the above stage (ii), the impact points are sampled according to
            the secondary extinction distribution. For a thin crystal (when *t*
            is specified) those rays that go over the crystal thickness retain
            the original incoming direction and their x, y, z coordinates (the
            ray heads) are put on the back crystal surface. These rays are also
            attenuated by the mosaic crystal. Note again that the impact points
            do not lie on the front crystal surface, so to plot them in a 2D XY
            plot becomes useless. You can still plot them as YZ or XZ to study
            the secondary extinction depth. The remaining rays (those sampled
            within the crystal depth) get reflected and also attenuated
            depending on the penetration depth.

            .. note::
                The amplitude calculation in the mosaic case is implemented
                only in the reflection geometry. The transmitted beam can still
                be studied by ray-tracing as we split the beam in our modeling
                of secondary extinction, see the previous paragraph.

            .. note::
                The mosaicity is assumed large compared with the Darwin width.
                Therefore, there is no continuous transition mosaic-to-perfect
                crystal at a continuously reduced mosaicity parameter.

            See the tests :ref:`here <tests_mosaic>`.

            .. [SanchezDelRioMosaic] M. Sánchez del Río et al.,
               Rev. Sci. Instrum. 63 (1992) 932.

            .. [BaconLowde] G. E. Bacon and R. D. Lowde,
               Acta Crystallogr. 1, (1948) 303.

        """
        super().__init__(elements, quantities, rho=rho, table=table, name=name,
                         **kwargs)
        self.hkl = hkl
        self.d = d
        self.V = V
        self.geom = geom
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.volumetricDiffraction = volumetricDiffraction
        self.useTT = useTT
        self.nu = nu
        self.mosaicity = mosaicity

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, geom):
        if len(geom) < 6:
            geom = geom.strip()+" reflected"
        self._geom = geom
        self.geometry = 2*int(geom.startswith('Bragg')) +\
            int(geom.endswith('transmitted'))  # Key for OpenCL calculations

    @property
    def hkl(self):
        return self._hkl

    @hkl.setter
    def hkl(self, hkl):
        self._hkl = hkl
        self.sqrthkl2 = (sum(i**2 for i in hkl))**0.5

        if hasattr(self, 'get_a'):
            d = self.get_a() / self.sqrthkl2
            if hasattr(self, '_VInit') and self._VInit is None:
                self.V = (d * self.sqrthkl2)**3
            self.d = d
        elif hasattr(self, 'set_cell_volume'):
            self.set_cell_volume()
        elif hasattr(self, 'a'):
            self.d = self.a / self.sqrthkl2

            if hasattr(self, '_VInit') and self._VInit is None:
                self.V = (self.d * self.sqrthkl2)**3

            if hasattr(self, '_V'):
                self.chiToF = -R0 / PI / self.V  # minus!
                self.chiToFd2 = abs(self.chiToF) * self.d**2

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        self._d = d
        if hasattr(self, '_V') and self.V is not None:
            self.chiToF = -R0 / PI / self.V  # minus!
            self.chiToFd2 = abs(self.chiToF) * d**2

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, V):
        self._VInit = V
        if V is None and hasattr(self, '_d') and hasattr(self, 'sqrthkl2'):
            V = (self.d * self.sqrthkl2)**3
        self._V = V
        if hasattr(self, '_d') and V is not None:
            self.chiToF = -R0 / PI / V  # minus!
            self.chiToFd2 = abs(self.chiToF) * self.d**2

#    def get_amplitude_Authie(self, E, gamma0, gammah, beamInDotHNormal):
#        """A. Authier, Dynamical Theory of X-ray Diffraction -1. Perfect
#        Crystals, in X-ray and Neutron Dynamical Diffraction: Theory and
#        Applications, ed. A. Authier, S. Lagomarsino & B. K. Tanner, NATO ASI
#        Ser., Ser. B: Physics 357 (1996) 1–32, Plenum Press: New York and
#        London."""
#        def _dynamical_theory_Bragg():
#            rx = np.sqrt(eta**2 - 1)
#            if self.t is not None:
#                arg = self.t * 1e7 * rx * math.pi/ lambdaExt
#                if self.geom.endswith('transmitted'):
#                    mu0 = -twoPi / waveLength * chi0.imag
#                    att = np.exp(-mu0 / 4 * (-1. / gamma0 - 1. / gammah) *
#                                 self.t)
#                    ta = att / (np.cos(arg) + 1j * eta * np.sin(arg) / rx)
#                    return ta
#                eps = 1.0j / np.tan (arg)
#            else:
#                eps = 1.
#            ra = 1. / (eta - rx * eps)
#            rb = 1. / (eta + rx * eps)
#            indB = np.where(abs(rb) < abs(ra))
#            ra[indB] = rb[indB]
#            return ra
#        def _dynamical_theory_Laue():
#            rx = np.sqrt(eta**2 + 1)
#            mu0 = -twoPi / waveLength * chi0.imag
#            t = self.t * 1e7
#            att = np.exp(-mu0 / 4 * (-1. / gamma0 - 1. / gammah) * t)
#            arg = t * rx * math.pi / lambdaExt
#            if self.geom.endswith('transmitted'):
#                ta = att * (np.cos(arg) + 1j * eta * np.sin(arg) / rx)
#                return ta
#            ra = abs(chih / chih_) * att * np.sin(arg) / rx
#            return ra
#        if self.geom.startswith('Bragg'):
#            _dynamical_theory = _dynamical_theory_Bragg
#        else:
#            _dynamical_theory = _dynamical_theory_Laue
#        waveLength = ch / E#the word "lambda" is reserved
#        sinThetaOverLambda = abs(beamInDotHNormal / waveLength)
#        F0, Fhkl, Fhkl_ = self.get_structure_factor(E, sinThetaOverLambda)
#        lambdaSquare = waveLength ** 2
#        chiToFlambdaSquare = self.chiToF * lambdaSquare
#        chi0 = F0 * chiToFlambdaSquare
#        chih = Fhkl * chiToFlambdaSquare
#        chih_ = Fhkl_ * chiToFlambdaSquare
#        gamma = gammah / gamma0# asymmetry parameter = 1/b
#        theta = np.arcsin(abs(beamInDotHNormal))
#        sin2theta = np.sin(2. * theta)
#        cos2theta = np.cos(2. * theta)
#        theta0 = np.arcsin(ch / (2 * self.d * E))
#        dtheta0 = - chi0 * (1 - gamma) / 2 / sin2theta
#        delta = np.sqrt(abs(gamma) * chih * chih_)/ sin2theta
#        if self.t is not None:
#            lambdaExt = waveLength * abs(gammah) / (delta * sin2theta)
#        else:
#            lambdaExt = None
#        eta = (theta - theta0 - dtheta0) / delta
# # s polarization:
#        resS = _dynamical_theory()
# # p polarization:
#        eta /= cos2theta
#        if self.t is not None:
#            lambdaExt /= cos2theta
#        resP = _dynamical_theory()
#        return resS, resP

    def get_F_chi(self, E, sinThetaOverLambda):
        F0, Fhkl, Fhkl_ = self.get_structure_factor(E, sinThetaOverLambda)
        waveLength = CH / E
        lambdaSquare = waveLength**2
        chiToFlambdaSquare = self.chiToF * lambdaSquare
# notice conjugate() needed for the formulas of Belyakov & Dmitrienko!!!
        chi0 = F0.conjugate() * chiToFlambdaSquare
        chih = Fhkl.conjugate() * chiToFlambdaSquare
        chih_ = Fhkl_.conjugate() * chiToFlambdaSquare
        return F0, Fhkl, Fhkl_, chi0, chih, chih_

    def get_Darwin_width(self, E, b=1., polarization='s'):
        r"""Calculates the Darwin width as

        .. math::

            2\delta = |C|\sqrt{\chi_h\chi_{\overline{h}} / b}/\sin{2\theta}
        """
        theta0 = self.get_Bragg_angle(E)
        sin2theta = np.sin(2. * theta0)
        waveLength = CH / E  # the word "lambda" is reserved
        sinThetaOverL = np.sin(theta0) / waveLength
        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, sinThetaOverL)
        if polarization == 's':
            polFactor = 1.
        else:
            polFactor = np.cos(2. * theta0)
        return 2 * (np.sqrt((polFactor**2 * chih*chih_ / b)) / sin2theta).real

    def get_epsilon_h(self, E, b=1., polarization='s'):
        r"""Calculates the relative spectral width :math:`epsilon_h` as
        (Shvyd'ko, Eq.2119)

        .. math::

            \epsilon_h = epsilon_h^{(s)}/\sqrt{|b|},
            \epsilon_h^{(s)} = \frac{4r_e d_h^2}{\pi V}|CF_h|.

        """
        F0, Fhkl, Fhkl_, _, _, _ = self.get_F_chi(E, 0.5/self.d)
        if polarization == 's':
            polFactor = 1.
        else:
            theta0 = self.get_Bragg_angle(E)
            polFactor = np.abs(np.cos(2. * theta0))
        return 4 * self.chiToFd2 * polFactor * np.abs(Fhkl) / abs(b)**0.5

    """
    def get_Borrmann_out(self, goodN, oeNormal, lb, a_out, b_out, c_out,
                         alphaAsym=None, Rcurvmm=None, ucl=None, useTT=False):

        asymmAngle = alphaAsym if alphaAsym is not None else 0

        if Rcurvmm is not None:
            Rcurv = Rcurvmm * 1e7
            if ucl is None:
                useTT = False
                print('OpenCL is required for bent crystals calculations. ')
                print('Emulating perfect crystal.')
        else:
            Rcurv = np.inf

        E = lb.E[goodN]
        bLength = len(E)

        if self.calcBorrmann.lower() in ['tt', 'bessel']:
            thetaB = self.get_Bragg_angle(E)
            beamOutDotNormal = a_out * oeNormal[-3] + \
                b_out * oeNormal[-2] + c_out * oeNormal[-1]

            beamInDotNormal = lb.a[goodN]*oeNormal[-3] +\
                lb.b[goodN]*oeNormal[-2] + lb.c[goodN]*oeNormal[-1]

            beamInDotHNormal = lb.a[goodN]*oeNormal[0] +\
                lb.b[goodN]*oeNormal[1] + lb.c[goodN]*oeNormal[2]

            waveLength = ch / E  # the word "lambda" is reserved
            thickness = self.t * 1e7 if self.t is not None else 0
            k = PI2 / waveLength
            HH = PI2 / self.d
            F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, 0.5/self.d)
            gamma_0h = beamInDotNormal * beamOutDotNormal

            if thickness == 0:
                N_layers = 10000
            else:
                N_layers = int(thickness / 200.)
                if N_layers < 2000:
                    N_layers = 2000
            IhMap = np.zeros((bLength, (N_layers+1)))

            for ipolFactor in [1., np.cos(2. * thetaB)]:
                if useTT and self.calcBorrmann.lower() == 'tt':
                    k0H = abs(beamInDotHNormal) * HH * k
                    dtsin2tb = (HH**2/2. - k0H) / (k**2)
                    betah = dtsin2tb - 0.5 * chi0.conjugate()
                    pmod = thickness / np.abs(beamInDotNormal) / (N_layers-1)
                    qmod = thickness / np.abs(beamOutDotNormal) / (N_layers-1)
                    AA = -0.25j * k * ipolFactor * chih_.conjugate() * pmod
                    BB = -0.25j * k * ipolFactor * chih.conjugate() * qmod
                    WW = 0.5j * k * betah * qmod
                    VV = -0.25j * k * chi0.conjugate() * pmod

                    if Rcurvmm is not None:
                        if self.geom.startswith('Bragg'):
                            Wgrad = np.zeros_like(AA)
                        else:
                            Bm = np.sin(asymmAngle) *\
                                (1. + gamma_0h * (1. + self.nuPoisson)) /\
                                gamma_0h
                            Wgrad = -0.25j * HH * Bm * pmod * qmod / Rcurv
                    else:
                        Wgrad = np.zeros_like(AA)

                    D0_local = np.zeros(bLength*(N_layers+1),
                                        dtype=np.complex128)
                    Dh_local = np.zeros(bLength*(N_layers+1),
                                        dtype=np.complex128)
                    D0t = np.zeros(bLength*(N_layers+3), dtype=np.complex128)
                    Dht = np.zeros(bLength*(N_layers+3), dtype=np.complex128)
                    scalarArgs = [np.int32(N_layers)]

                    slicedROArgs = [np.complex128(Wgrad),
                                    np.complex128(AA),
                                    np.complex128(BB),
                                    np.complex128(WW),
                                    np.complex128(VV)]

                    nonSlicedROArgs = [D0t, Dht]

                    slicedRWArgs = [D0_local, Dh_local]

                    kernel = 'tt_laue_spherical'

                    if Rcurvmm is not None:
                        kernel += '_bent'

                    D0_local, Dh_local = ucl.run_parallel(
                        kernel, scalarArgs, slicedROArgs,
                        nonSlicedROArgs, slicedRWArgs, None, bLength)
                    if self.geom.endswith('transmitted'):
                        bFan = np.abs(D0_local.reshape((
                            bLength, (N_layers+1))))**2
                    else:
                        bFan = np.abs(Dh_local.reshape((
                            bLength, (N_layers+1))))**2
                else:
                    sqrtchchm = np.sqrt(chih.conjugate()*chih_.conjugate())
                    exctDepth = waveLength * np.sqrt(np.abs(gamma_0h)) /\
                        sqrtchchm/ipolFactor
                    yrange = np.linspace(-1, 1, N_layers+1)
                    besselArgument = PI * thickness / exctDepth
                    bFan = np.abs(besselJn(
                        0, besselArgument[:, np.newaxis] *
                        np.sqrt(1.-np.square(yrange))))**2

                IhMap += bFan
            IhMax = np.max(IhMap, axis=1)

            #  Now sampling the position along the base of the Borrmann fan
            index = np.array(range(bLength))
            iLeft = index
            raysLeft = bLength
            totalX = np.zeros(bLength)
            counter = 0
            while raysLeft > 0:
                counter += 1
                disc = np.random.random(raysLeft)*IhMax[index]
                rawRand = np.random.random(raysLeft)
                xrand = rawRand * 2. - 1.
                if useTT:
                    deltaRand, ipLeft = np.modf(rawRand * N_layers)
                    rndmIntensity = IhMap[index, np.int32(ipLeft)] *\
                        (1. - deltaRand) +\
                        IhMap[index, np.int32(np.ceil(rawRand * N_layers))] *\
                        deltaRand
                else:
                    rndmIntensity = np.abs(besselJn(
                        0, besselArgument[index] *
                        np.sqrt(1-np.square(xrand))))**2

                passed = np.where(rndmIntensity > disc)[0]
                totalX[index[passed]] = xrand[passed]
                iLeft = np.where(rndmIntensity <= disc)[0]
                index = index[iLeft]
                raysLeft = len(index)
            totalX = 0.5*(totalX + 1.)
        elif self.calcBorrmann == 'uniform':
            totalX = np.random.random(bLength)
        else:  # You should never get here
            totalX = 0.5*np.ones(bLength)

        return totalX
    """

    def get_amplitude(self, E, beamInDotNormal, beamOutDotNormal=None,
                      beamInDotHNormal=None, xd=None, yd=None):
        r"""
        Calculates complex amplitude reflectivity and transmittivity for s- and
        p-polarizations (:math:`\gamma = s, p`) in Bragg and Laue cases for the
        crystal of thickness *L*, based upon Belyakov & Dmitrienko [BD]_:

        .. math::

            R_{\gamma}^{\rm Bragg} &= \chi_{\vec{H}}C_{\gamma}\left(\alpha +
            i\Delta_{\gamma}\cot{l_{\gamma}}\right)^{-1}|b|^{-\frac{1}{2}}\\
            T_{\gamma}^{\rm Bragg} &= \left(\cos{l{_\gamma}} - i\alpha\Delta
            {_\gamma}^{-1}\sin{l_{\gamma}}\right)^{-1}
            \exp{\left(i\vec{\kappa}_0^2 L
            (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1}\right)}\\
            R_{\gamma}^{\rm Laue} &= \chi_{\vec{H}}C_{\gamma}
            \Delta_{\gamma}^{-1}\sin{l_{\gamma}}\exp{\left(i\vec{\kappa}_0^2 L
            (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1}\right)}
            |b|^{-\frac{1}{2}}\\
            T_{\gamma}^{\rm Laue} &= \left(\cos{l_{\gamma}} + i\alpha
            \Delta_{\gamma}^{-1}\sin{l_{\gamma}}\right)
            \exp{\left(i\vec{\kappa}_0^2
            L (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1}\right)}

        where

        .. math::

            \alpha &= \frac{\vec{H}^2 + 2\vec{\kappa}_0\vec{H}}
            {2\vec{\kappa}_0^2}+\frac{\chi_0(1-b)}{2b}\\
            \Delta_{\gamma} &= \left(\alpha^2 +\frac{C_{\gamma}^2\chi_{\vec{H}}
            \chi_{\overline{\vec{H}}}}{b}\right)^{\frac{1}{2}}\\
            l_{\gamma} &= \frac{\Delta_{\gamma}\vec{\kappa}_0^2L}
            {2\vec{\kappa}_{\vec{H}}\vec{s}}\\
            b &= \frac{\vec{\kappa}_0\vec{s}}{\vec{\kappa}_{\vec{H}}\vec{s}}\\
            C_s &= 1, \quad C_p = \cos{2\theta_B}

        In the case of thick crystal in Bragg geometry:

        .. math::

            R_{\gamma}^{\rm Bragg} = \frac{\chi_{\vec{H}} C_{\gamma}}
            {\alpha\pm\Delta_{\gamma}}|b|^{-\frac{1}{2}}

        with the sign in the denominator that gives the smaller modulus of
        :math:`R_\gamma`.

        :math:`\chi_{\vec{H}}` is the Fourier harmonic of the x-ray
        susceptibility, and :math:`\vec{H}` is the reciprocal lattice vector of
        the crystal. :math:`\vec{\kappa}_0` and :math:`\vec{\kappa}_{\vec{H}}`
        are the wave vectors of the direct and diffracted waves.
        :math:`\chi_{\vec{H}}` is calculated as:

        .. math::

            \chi_{\vec{H}} = - \frac{r_0\lambda^2}{\pi V}F_{\vec{H}},

        where :math:`r_e = e^2 / mc^2` is the classical radius of the electron,
        :math:`\lambda` is the wavelength, *V* is the volume of the unit cell.

        Notice :math:`|b|^{-\frac{1}{2}}` added to the formulas of Belyakov &
        Dmitrienko in the cases of Bragg and Laue reflections. This is needed
        because ray tracing deals not with wave fields but with rays and
        therefore not with intensities (i.e. per cross-section) but with flux.

        .. [BD] V. A. Belyakov and V. E. Dmitrienko, *Polarization phenomena in
           x-ray optics*, Uspekhi Fiz. Nauk. **158** (1989) 679–721, Sov. Phys.
           Usp. **32** (1989) 697–719.

        *xd* and *yd* are local coordinates of the corresponding optical
        element. If they are not None and crystal's `get_d` method exists, the
        d spacing is given by the `get_d` method, otherwise it equals to
        `self.d`. In a parametric representation, *xd* and *yd* are the same
        parametric coordinates used in `local_r` and local_n` methods of the
        corresponding optical element.
        """
        def for_one_polarization(polFactor):
            delta = np.sqrt((alpha**2 + polFactor**2 * chih * chih_ / b))
            if self.t is None:  # thick Bragg
                # if (alpha==np.nan).sum()>0: print('(alpha==np.nan).sum()>0!')
                with np.errstate(divide='ignore'):
                    ra = chih * polFactor / (alpha+delta)
                ad = alpha - delta
                ad[ad == 0] = 1e-100
                rb = chih * polFactor / ad
                indB = np.where(np.isnan(ra))
                ra[indB] = rb[indB]
                indB = np.where(abs(rb) < abs(ra))
                ra[indB] = rb[indB]
#                if np.isnan(ra).sum() > 0:
#                    if (alpha == -delta).sum() > 0:
#                        print('alpha = -delta!', (alpha == -delta).sum())
#                        print('alpha ',alpha[alpha == -delta])
#                        print('delta ', delta[alpha == -delta])
#                        print('chih ', chih[alpha == -delta])
#                        print('b ', b[alpha == -delta]_
#                    if (alpha == delta).sum() > 0:
#                        print('alpha = delta!', (alpha == delta).sum())
#                    if np.isnan(alpha).sum() > 0:
#                        print('alpha contains nan!')
#                    if np.isnan(delta).sum() > 0:
#                        print('delta contains nan!')
#                    if np.isnan(chih).sum() > 0:
#                        print('chih contains nan!')
#                    raise ValueError('reflectivity contains nan!')
                return ra / np.sqrt(abs(b))
            t = self.t * 1e7
            l = t * delta * k02 / 2. / kHs  # analysis:ignore
            with np.errstate(over='ignore'):
                if self.geom.startswith('Bragg'):
                    if self.geom.endswith('transmitted'):
                        ra = 1 / (np.cos(l) - 1j * alpha * np.sin(l) / delta) \
                            * np.exp(1j * k02 * t * (chi0 - alpha*b) / 2 / k0s)
                    else:
                        ra = chih * polFactor / (alpha + 1j*delta / np.tan(l))
                else:  # Laue
                    if self.geom.endswith('transmitted'):
                        ra = (np.cos(l) + 1j * alpha * np.sin(l) / delta) *\
                            np.exp(1j * k02 * t * (chi0 - alpha*b) / 2 / k0s)
                    else:
                        ra = chih * polFactor * np.sin(l) / delta *\
                            np.exp(1j * k02 * t * (chi0 - alpha*b) / 2 / k0s)
            if not self.geom.endswith('transmitted'):
                ra /= np.sqrt(abs(b))
            return ra

        waveLength = CH / E  # the word "lambda" is reserved
        k = PI2 / waveLength
        k0s = -beamInDotNormal * k
        if beamOutDotNormal is None:
            beamOutDotNormal = -beamInDotNormal
        kHs = -beamOutDotNormal * k
        if beamInDotHNormal is None:
            beamInDotHNormal = beamInDotNormal
        if hasattr(self, 'get_d') and xd is not None and yd is not None:
            crystd = self.get_d(xd, yd)
        else:
            crystd = self.d
        HH = PI2 / crystd
        k0H = abs(beamInDotHNormal) * HH * k
        k02 = k**2
        H2 = HH**2
        kHs0 = kHs == 0
        kHs[kHs0] = 1
        b = k0s / kHs
        b[kHs0] = -1
        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, 0.5/crystd)
        thetaB = self.get_Bragg_angle(E)  # variation of d ignored in polFactor
        alpha = (H2/2 - k0H) / k02 + chi0/2 * (1/b - 1)

        curveS = for_one_polarization(1.)  # s polarization
        polFactor = np.cos(2. * thetaB)
        curveP = for_one_polarization(polFactor)  # p polarization
        return curveS, curveP  # , phi.real

    def set_OE_properties(self, alpha=0, Rm=None, Rs=None,
                          inPlaneRotation=None):
        """
        This function is used with get_amplitudes_pytte(), it passes the
        curvature and asymmetry of the parent optical element to the
        underlying pyTTE.TTcrystal. Returned elastic constants are
        then used for reflectivity/transmittivity calculations.

        Parameters
        ----------
        alpha : float
            Angle of asymmetry in radians.
        Rm : float
            Meridional curvature in mm.
        Rs : float
            Sagittal curvature in mm.
        inPlaneRotation : float, optional
            Angle of in-plane rotation in radians.

        Returns
        -------
        None.

        """

        Rmum = Rm*1e3 if Rm not in [np.inf, None] else np.inf  # [um] Mer
        Rsum = Rs*1e3 if Rs not in [np.inf, None] else np.inf  # [um] Sag
        geotag = 0 if self.geom.startswith('B') else np.pi*0.5
        alpha = 0 if alpha is None else alpha+geotag

        classname = type(self).__name__
        thickness = 1. if self.t is None else self.t

        # Instantiating pytte TTCrystal to calculate displacement vectors
        ttcrystal_kwargs = {'crystal': classname,
                            'hkl': self.hkl,
                            'thickness': Quantity(thickness*1000, 'um'),
                            'debye_waller': 1,
                            'xrt_crystal': self,
                            'Rx': Quantity(Rmum, 'um'),
                            'Ry': Quantity(Rsum, 'um'),
                            'asymmetry': Quantity(alpha, 'rad')}

        if inPlaneRotation is not None:
            ttcrystal_kwargs['in_plane_rotation'] =\
                Quantity(inPlaneRotation, 'rad')

        if hasattr(self, 'nu'):
            if self.nu is not None:  # Using isotropic model
                ttcrystal_kwargs['nu'] = self.nu
                ttcrystal_kwargs['E'] = Quantity(1, 'Pa')

        ttx = TTcrystal(**ttcrystal_kwargs)
        self.djparams = ttx.djparams

    def get_amplitude_pytte(
            self, E, beamInDotNormal, beamOutDotNormal=None,
            beamInDotHNormal=None, xd=None, yd=None,
            alphaAsym=None, inPlaneRotation=None,
            Ry=None, Rx=None, ucl=None, tolerance=1e-6, maxSteps=1e7,
            autoLimits=True, signal=None):
        r"""
        Calculates complex amplitude reflectivity for s- and
        p-polarizations (:math:`\gamma = s, p`) in Bragg and Laue cases, based
        on modified `PyTTE code <https://github.com/aripekka/pyTTE>`_

        *alphaAsymm*: float
            Angle of asymmetry in radians.

        *inPlaneRotation*: float
            Counterclockwise-positive rotation of the crystal directions around
            the normal vector of (hkl) in radians. (see pyTTE.TTcrystal).
            In-plane rotation definition as vector is not supported in xrt
            currently.

        *Ry*: float
            Meridional radius of curvature in mm. Positive for concave bend.

        *Rx*: float
            Sagittal radius of curvature in mm. Positive for concave bend.

        *ucl*:
            instance of XRT_CL class, defines the OpenCL device and precision
            of calculation. Calculations should run fine in single precision,
            float32. See XRT_CL.

        *tolerance*: float
            Precision tolerance for RK adaptive step algorithm.

        *maxSteps*: int
            Emergency exit to avoid kernel freezing if the step gets too small.

        *autoLimits*: bool
            If True, the algorithm will try to automatically determine the
            angular range where reflectivity will be calculated by numeric
            integration. Useful for ray-tracing applications where angle of
            incidence can be too far from Bragg condition, and integration
            might take unnesessarily long time.


        """

        if beamOutDotNormal is None:
            beamOutDotNormal = -beamInDotNormal
        if beamInDotHNormal is None:  # Bragg
            beamInDotHNormal = beamInDotNormal

        thickness = 1. if self.t is None else self.t
        dh0tag = 0 if self.geom.endswith('reflected') else 1

        # if not hasattr(self, 'djparams'):  # Same material can be used in
        # different OEs with different surface curvatures
        self.set_OE_properties(alphaAsym, Ry, Rx, inPlaneRotation)

        geotag = 0 if self.geom.startswith('B') else np.pi*0.5
        if dh0tag and not geotag:
            return self.get_amplitude(
                E, beamInDotNormal, beamOutDotNormal,
                beamInDotHNormal, xd, yd)
        alphaAsym = 0 if alphaAsym is None else alphaAsym+geotag
        Ryum = Ry*1e3 if Ry not in [np.inf, None] else np.inf  # [um] Mer
        Rxum = Rx*1e3 if Rx not in [np.inf, None] else np.inf  # [um] Sag

        if all(np.isinf([Ryum, Rxum])):
            return self.get_amplitude(
                E, beamInDotNormal, beamOutDotNormal,
                beamInDotHNormal, xd, yd)

        # Step 1. Evaluating the boundaries where the amplitudes are calculated

        if isinstance(E, np.ndarray):
            NRAYS = len(E)
            beamInDotNormal *= np.ones_like(E)
        else:
            NRAYS = len(beamInDotNormal)
            E *= np.ones_like(beamInDotNormal)

        tmaxCL = np.ones(NRAYS, dtype=ucl.cl_precisionF)*np.pi
        tminCL = -tmaxCL
        if hasattr(self, 'get_d') and xd is not None and yd is not None:
            crystd = self.get_d(xd, yd)
        else:
            crystd = self.d

        h = 2*np.pi / crystd  # in inverse um
        thetaB = self.get_Bragg_angle(E)  # variation of d ignored in polFactor

        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, 0.5/crystd)
        checkLimits = autoLimits
        if hasattr(self, 'auto_PyTTE_Limits'):
            checkLimits = self.auto_PyTTE_Limits
        if checkLimits:
            chcbmod = np.sqrt(np.abs(chih*chih_))
            alpha0 = thetaB + alphaAsym
            alphah = thetaB - alphaAsym
            gamma_term = np.sin(alphah)/np.sin(alpha0)
            k_bragg = 0.5*h / abs(beamInDotHNormal)
            # gamma0/gammah in pytte notation:
            b_const_term = -0.5*k_bragg*(1 + gamma_term)*np.real(chi0)
            scalarArgs = [ucl.cl_precisionF(self.djparams[0]),
                          ucl.cl_precisionF(self.djparams[1]),
                          ucl.cl_precisionF(self.djparams[2]),  # InvR1 in 1/um
                          ucl.cl_precisionF(alphaAsym),
                          ucl.cl_precisionF(thickness*1e3),  # From mm to um
                          ucl.cl_precisionF(h*1e4)]  # h = 2*np.pi/d From 1/A to 1/um
            slicedROArgs = [ucl.cl_precisionF(b_const_term*1e4),  # From 1/A to 1/um
                            ucl.cl_precisionF(thetaB),
                            ucl.cl_precisionF(chcbmod)]
            slicedRWArgs = [tminCL, tmaxCL]

            tminCL, tmaxCL = ucl.run_parallel(
                'estimate_bent_width', scalarArgs, slicedROArgs,
                None, slicedRWArgs, None, dimension=NRAYS)

            limExtend = 3 if (abs(Ryum) > 1e9 and abs(Rxum) > 1e9) else 1.5
            tmid = 0.5*(tmaxCL+tminCL)
            thw = 0.5*(tmaxCL-tminCL)
            tminCL = tmid - limExtend*thw  # Initial estimate is too narrow
            tmaxCL = tmid + limExtend*thw  # Increasing the range

        # Step 2. Working with real angles

        waveLength = CH / E  # wavelength in Angstrom [1e-7mm]
        k = PI2 / waveLength  # in inverse Angstrom [1e7 1/mm]
        beta = abs(beamInDotHNormal) - 0.5*h/k

        # For solving ksi = Dh/D0
#        c0 = 0.5e4*k*chi0*(1/abs(beamInDotNormal)+1/beamOutDotNormal)
#        ch = 0.5e4*k*chih/beamOutDotNormal
#        cb = 0.5e4*k*chih_/abs(beamInDotNormal)

        c0 = 0.5e4*k*chi0*(-1/beamInDotNormal+1/beamOutDotNormal)
        ch = 0.5e4*k*chih/beamOutDotNormal
        cb = -0.5e4*k*chih_/beamInDotNormal

        # For solving Y = D0
        g0 = -0.5e4*k*chi0/beamInDotNormal  # to both kernels but used only by Laue
        # gb = 0.5*k*chih_/beamInDotNormal  # Same as cb

        theta = np.arcsin(abs(beamInDotHNormal))
        alpha0 = theta+alphaAsym
        alphah = theta-alphaAsym

        dtheta = theta - thetaB
        nzrays = np.where((dtheta > tminCL) & (dtheta < tmaxCL))[0]  #

        startSteps = 2000000

        bLength = len(nzrays)
        t001 = time.time()
        amp_s = np.zeros(bLength, dtype=ucl.cl_precisionC)
        amp_p = np.zeros(bLength, dtype=ucl.cl_precisionC)

        npoints = np.zeros(bLength, dtype=ucl.cl_precisionF)
        hgammah = h*1e4/beamOutDotNormal[nzrays]
        scalarArgs = [ucl.cl_precisionF(self.djparams[0]),  # C1
                      ucl.cl_precisionF(self.djparams[1]),  # C2
                      ucl.cl_precisionF(self.djparams[2]),  # InvR1
                      ucl.cl_precisionF(alphaAsym),
                      ucl.cl_precisionF(thickness*1e3),
                      ucl.cl_precisionF(tolerance),  # RK Adaptive step control
                      np.int32(maxSteps),
                      np.int32(startSteps),
                      np.int32(geotag),
                      np.int32(dh0tag)
                      ]
        slicedROArgs = [ucl.cl_precisionF(hgammah),
                        ucl.cl_precisionF(hgammah*beta[nzrays]),
                        ucl.cl_precisionF(thetaB[nzrays]),
                        ucl.cl_precisionF(alpha0[nzrays]),
                        ucl.cl_precisionF(alphah[nzrays]),
                        ucl.cl_precisionC(c0[nzrays]),
                        ucl.cl_precisionC(ch[nzrays]),
                        ucl.cl_precisionC(cb[nzrays]),
                        ucl.cl_precisionC(g0[nzrays])]

        slicedRWArgs = [amp_s, amp_p, npoints]
        print('Calculating bent crystal reflectivity...')
        amp_s, amp_p, npoints = ucl.run_parallel(
            'get_amplitudes_pytte', scalarArgs, slicedROArgs,
            None, slicedRWArgs, None, dimension=bLength, complexity=startSteps,
            signal=signal)

#        from matplotlib import pyplot as plt
#        plt.figure("npoints")
#        plt.plot(dtheta[nzrays], npoints)
#        plt.savefig("npoints.png")

        if True:  # background.startswith('zero'):
            curveS = np.zeros(NRAYS, dtype=np.complex128)
            curveP = np.zeros_like(curveS)
        else:
            curveS, curveP = self.get_amplitude(
                    E, beamInDotNormal, beamOutDotNormal, beamInDotHNormal)
        norm = np.ones_like(beamInDotNormal) if dh0tag else\
            np.sqrt(np.abs(beamOutDotNormal)/np.abs(beamInDotNormal))
        curveS[nzrays] = amp_s*norm[nzrays]
        curveP[nzrays] = amp_p*norm[nzrays]
        if signal is not None:
            signal.emit(("Calculation completed in {:.3f}s".format(
                    time.time()-t001), 100))
        print("Amplitude calculation for {0} points takes {1:.3f} s".format(
            bLength, time.time()-t001))
        return curveS, curveP

    """
    def get_amplitude_TT(self, E, beamInDotNormal, beamOutDotNormal=None,
                         beamInDotHNormal=None, alphaAsym=None,
                         Rcurvmm=None, ucl=None):

        def for_one_polarization_TT(polFactor):

            if thickness == 0:
                pmod = 1.0e2/np.abs(beamInDotNormal)
                qmod = 1.0e2/np.abs(beamOutDotNormal)
            else:
                pmod = thickness/np.abs(beamInDotNormal)/N_layers
                qmod = thickness/np.abs(beamOutDotNormal)/N_layers

            AA = -0.25j * k * polFactor * chih_.conjugate() * pmod
            BB = -0.25j * k * polFactor * chih.conjugate() * qmod
            WW = 0.5j * k * beta_h * qmod
            VV = -0.25j * k * chi0.conjugate() * pmod

            gamma_0h = beamInDotNormal * beamOutDotNormal

            if Rcurvmm is not None:
                if self.geom.startswith('Bragg'):
                    Wgrad = np.zeros_like(AA)
                    print("Bending in Bragg geometry is not implemented")
                    print("Emulating perfect crystal.")
#  Bending in reflection geometry is not implemented
#                    if thickness == 0:
#                        Wgrad = -0.5 * 1j * qmod**2 *\
#                            (1. - beamOutDotNormal**2) * HH *\
#                            np.cos(alphaAsym) / Rcurv
                else:
                    Bm = np.tan(asymmAngle) *\
                        (1. + gamma_0h * (1. + self.nuPoisson)) / gamma_0h
#  Calculating reflectivities in Laue geometry is still experimental
#  Use at your own risk
                    Wgrad = -0.25j * HH * Bm * pmod * qmod / Rcurv

            else:
                Wgrad = np.zeros_like(AA)

            D0_local = np.zeros_like(AA)
            Dh_local = np.zeros_like(AA)

            scalarArgs = [np.int32(N_layers)]
            slicedROArgs = [np.complex128(Wgrad),
                            np.complex128(AA),
                            np.complex128(BB),
                            np.complex128(WW),
                            np.complex128(VV)]

            slicedRWArgs = [D0_local, Dh_local]

            if self.geom.startswith('Bragg'):
                D0t = np.zeros(bLength*(N_layers+1), dtype=np.complex128)
                Dht = np.zeros(bLength*(N_layers+1), dtype=np.complex128)
                nonSlicedROArgs = [D0t, Dht]
                kernel = "tt_bragg"
            else:
                nonSlicedROArgs = None
                kernel = "tt_laue"

            if Rcurvmm is None:
                kernel += '_plain'
            else:
                kernel += '_plain_bent'

            D0_local, Dh_local = ucl.run_parallel(
                kernel, scalarArgs, slicedROArgs,
                nonSlicedROArgs, slicedRWArgs, None, bLength)

            if self.geom.endswith('transmitted'):
                ra = D0_local
            else:
                ra = Dh_local

            if not self.geom.endswith('transmitted'):
                ra /= np.sqrt(abs(beamInDotNormal/beamOutDotNormal))
            return ra

        asymmAngle = alphaAsym if alphaAsym is not None else 0
        waveLength = CH / E  # the word "lambda" is reserved
        k = PI2 / waveLength
        k0s = -beamInDotNormal * k
        if beamOutDotNormal is None:
            beamOutDotNormal = -beamInDotNormal
        kHs = -beamOutDotNormal * k
        if beamInDotHNormal is None:
            beamInDotHNormal = beamInDotNormal
        HH = PI2 / self.d
        k0H = abs(beamInDotHNormal) * HH * k
        k02 = k**2
        H2 = HH**2
        kHs0 = kHs == 0
        kHs[kHs0] = 1
        b = k0s / kHs
        b[kHs0] = -1
        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, 0.5/self.d)
        thetaB = self.get_Bragg_angle(E)
        alpha = (H2/2 - k0H) / k02 + chi0/2 * (1/b - 1)

        thickness = 0 if self.t is None else self.t * 1e7
        if thickness == 0:
            N_layers = 10000
        else:
            N_layers = thickness / 100.
            if N_layers < 2000:
                N_layers = 2000
        bLength = len(E)
        dtsin2tb = (H2/2. - k0H) / (k**2)
        if Rcurvmm in [0, None]:
            Rcurv = np.inf
        else:
            Rcurv = Rcurvmm * 1.0e7
        beta_h = dtsin2tb - 0.5 * chi0.conjugate()

        curveS = for_one_polarization_TT(1.)  # s polarization
        polFactor = np.cos(2. * thetaB)
        curveP = for_one_polarization_TT(polFactor)  # p polarization
        return curveS, curveP  # , phi.real
    """

    def get_amplitude_mosaic(self, E, beamInDotNormal, beamOutDotNormal=None,
                             beamInDotHNormal=None):
        """Based on Bacon and Lowde"""
        def for_one_polarization(Q):
            a = Q*w / mu
            b = (1 + 2*a)**0.5
            if self.t is None:  # thick Bragg
                return a / (1 + a + b)
            A = mu*t / g0
            if self.geom.startswith('Bragg'):
                return a / (1 + a + b/np.tanh(A*b))  # Eq. (17)
            else:  # Laue
                # return np.sinh(A*a) * np.exp(-A*(1+a))  # Eq. (18)
                sigma = Q*w / g0
                overGamma = 0.5 * (1/g0 + 1/gH)
                overG = 0.5 * (1/g0 - 1/gH)
                sm = (sigma**2 + mu**2*overG**2)**0.5
                sGamma = sigma + mu*overGamma
                # Eq. (24):
                return sigma/sm * np.sinh(sm*t) * np.exp(-sGamma*t)
        Qs, Qp, thetaB = self.get_kappa_Q(E)[2:5]  # in cm^-1
        if beamInDotHNormal is None:
            beamInDotHNormal = beamInDotNormal
        delta = np.arcsin(np.abs(beamInDotHNormal)) - thetaB
        g0 = np.abs(beamInDotNormal)
        gH = g0 if beamOutDotNormal is None else np.abs(beamOutDotNormal)
        w = np.exp(-0.5*delta**2/self.mosaicity**2) / (SQRT2PI*self.mosaicity)
        mu = self.get_absorption_coefficient(E)  # in cm^-1
        if self.geom.startswith('Bragg'):
            mu *= 0.5 * (1 + g0/gH)  # Eq. (23)
        if self.t is not None:
            t = self.t*0.1  # t is in cm
        curveS = for_one_polarization(Qs)
        curveP = for_one_polarization(Qp)
        return curveS**0.5, curveP**0.5

    def get_kappa_Q(self, E):
        """kappa: inversed extinction length;
        Q: integrated reflecting power per unit propagation path.
        Returned as a tuple (kappas, kappap, Qs, Qp), all in cm^-1."""
        thetaB = self.get_Bragg_angle(E) - self.get_dtheta(E)
        waveLength = CH / E
        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, 0.5/self.d)
        polFactor = np.cos(2*thetaB)
#        kappas = abs(chih) / waveLength * PI  # or the same:
        kappas = abs(Fhkl) * waveLength * R0 / self.V
        Qs = kappas**2 * waveLength / np.sin(2*thetaB)
        kappap = kappas * abs(polFactor)
        Qp = Qs * polFactor**2  # as by Kato, note power 1 in Shadow paper
#        return kappas, kappap, Qs, Qp  # in Å^-1
        return kappas*1e8, kappap*1e8, Qs*1e8, Qp*1e8, thetaB  # in cm^-1

    def get_extinction_lengths(self, E):
        """Returns a tuple of primary extinction lengths for s and p and, if
        mosaicity is given, secondary extinction lengths: l1s, l1p, {l2s, l2p},
        all in mm."""
        kappas, kappap, Qs, Qp = self.get_kappa_Q(E)[0:4]
        if self.mosaicity:
            w = 1. / (SQRT2PI*self.mosaicity)
            return 10./kappas, 10./kappap, 10./(w*Qs), 10./(w*Qp)  # in mm
        else:
            return 10./kappas, 10./kappap  # in mm

    def get_extinction_depth(self, E, polarization='s'):
        """Same as get_extinction_length but measured normal to the surface."""
        theta = self.get_Bragg_angle(E)
        res = self.get_extinction_length(E, polarization)
        return [r * np.sin(theta) for r in res]

    def get_sin_Bragg_angle(self, E, order=1):
        """ensures that -1 <= sin(theta) <= 1"""
        a = order * CH / (2*self.d*E)
        try:
            a[a > 1] = 1 - 1e-16
            a[a < -1] = -1 + 1e-16
        except TypeError:
            if a > 1:
                a = 1 - 1e-16
            elif a < -1:
                a = -1 + 1e-16
        return a

    def get_Bragg_angle(self, E, order=1):
        a = self.get_sin_Bragg_angle(E, order)
        return np.arcsin(a)

    def get_backscattering_energy(self):
        return CH / (2*self.d)

    def get_dtheta_symmetric_Bragg(self, E):
        r"""
        The angle correction for the symmetric Bragg case:

        .. math::

            \delta\theta = \chi_0 / \sin{2\theta_B}
        """
        F0, Fhkl, Fhkl_ = self.get_structure_factor(E, 0.5 / self.d)
        waveLength = CH / E  # the word "lambda" is reserved
        lambdaSquare = waveLength ** 2
        chiToFlambdaSquare = self.chiToF * lambdaSquare
        chi0 = F0 * chiToFlambdaSquare
        thetaB = self.get_Bragg_angle(E)
        return (chi0 / np.sin(2*thetaB)).real

    def get_dtheta(self, E, alpha=None):
        r"""
        .. _get_dtheta:

        The angle correction for the general asymmetric case:

        .. math::

            \delta\theta = \frac{\mp \gamma_0 \pm \sqrt{\gamma_0^2 \mp
            (\gamma_0 - \gamma_h) \sqrt{1 - \gamma_0^2} \chi_0 /
            \sin{2\theta_B}}}{\sqrt{1 - \gamma_0^2}}\\

        where :math:`\gamma_0 = \sin(\theta_B + \alpha)`,
        :math:`\gamma_h = \mp \sin(\theta_B - \alpha)` and the upper sign is
        for Bragg and the lower sign is for Laue geometry.

        Taken from [Authier]_ Eq. (8.3). See the comparison between the two
        expressions (`get_dtheta()` and `get_dtheta_regular()`) in Fig. 8.3.

        .. [Authier] A. Authier, Dynamical theory of X-ray diffraction,
           Oxford University Press, 2001.
        """
        if alpha is None:
            alpha = 0
        thetaB = self.get_Bragg_angle(E)
        pm = -1 if self.geom.startswith('Bragg') else 1
        gamma0 = np.sin(thetaB + alpha)
        gammah = pm * np.sin(thetaB - alpha)
        symm_dt = self.get_dtheta_symmetric_Bragg(E)
        osqg0 = np.sqrt(1. - gamma0**2)
        dtheta0 = (pm*gamma0 - pm*np.sqrt(gamma0**2 +
                   pm*(gamma0 - gammah) * osqg0 * symm_dt)) / osqg0
        return -dtheta0

    def get_dtheta_regular(self, E, alpha=None):
        r"""
        The angle correction for the general asymmetric case in its simpler
        version:

        .. math::
            \delta\theta = (1 - b)/2 \cdot \chi_0 / \sin{2\theta_B}\\
            |b| = \sin(\theta_B + \alpha) / \sin(\theta_B - \alpha)

        For the symmetric Bragg *b* = -1 and for the symmetric Laue *b* = +1.
        """
        if alpha is not None:
            thetaB = self.get_Bragg_angle(E)
            b = np.sin(thetaB + alpha) / np.sin(thetaB - alpha)
            if self.geom.startswith('Bragg'):
                b *= -1
            return (1 - b)/2 * self.get_dtheta_symmetric_Bragg(E)
        else:
            if self.geom.startswith('Bragg'):
                return self.get_dtheta_symmetric_Bragg(E)
            else:
                return 0.

    def get_refractive_correction(self, E, beamInDotNormal=None, alpha=None):
        r"""
        The difference in the glancing angle of incidence for incident and exit
        waves, Eqs. (2.152) and (2.112) in [Shvydko_XRO]_:

        .. math::
            \theta_c - \theta'_c = \frac{w_H^{(s)}}{2} \left(b - \frac{1}{b}
            \right) \tan{\theta_c}

        .. note::
            Not valid close to backscattering.

        .. [Shvydko_XRO] Yu. Shvyd'ko, X-Ray Optics High-Energy-Resolution
           Applications, Springer-Verlag Berlin Heidelberg, 2004.

        """
        thetaB = self.get_Bragg_angle(E)
        bothNone = (beamInDotNormal is None) and (alpha is None)
        bothNotNone = (beamInDotNormal is not None) and (alpha is not None)
        if bothNone or bothNotNone:
            raise ValueError(
                "one of 'beamInDotNormal' or 'alpha' must be given")
        if beamInDotNormal is not None:
            # beamInDotNormal[beamInDotNormal > 1] = 1 - 1e-16
            alpha = np.arcsin(beamInDotNormal) - thetaB
        if alpha is not None:
            beamInDotNormal = np.sin(thetaB + alpha)
        pm = -1 if self.geom.startswith('Bragg') else 1
        beamOutDotNormal = pm * np.sin(thetaB - alpha)
        b = beamInDotNormal / beamOutDotNormal
        F0, _, _ = self.get_structure_factor(E, needFhkl=False)
        return -self.chiToFd2 * F0.real * (b - 1/b) * np.tan(thetaB)

# -*- coding: utf-8 -*-
import time
import numpy as np

from ... import raycing
from ..physconsts import CH, CHBAR

ch = CH  # left here for copatibility
chbar = CHBAR  # left here for copatibility


class Multilayer(object):
    u"""
    :class:`Multilayer` serves for getting reflectivity of a multilayer. The
    multilayer may have variable thicknesses of the two alternating layers as
    functions of local *x* and *y* and/or as a function of the layer number.
    """

    hiddenParams = {'power', 'substRoughness', 'tThicknessLow',
                    'bThicknessLow'}

    def __init__(self, tLayer=None, tThickness=0., bLayer=None, bThickness=0.,
                 nPairs=0, substrate=None, tThicknessLow=0., bThicknessLow=0.,
                 idThickness=0., power=2., substRoughness=0.,
                 substThickness=np.inf, name='', geom='reflected', **kwargs):
        u"""
        *tLayer*, *bLayer*, *substrate*: instance of :class:`Material`
            The top layer material, the bottom layer material and the substrate
            material.

        *tThickness* and *bThickness*: float in Å
            The thicknesses of the layers. If the multilayer is depth
            graded, *tThickness* and *bThickness* are at the top and
            *tThicknessLow* and *bThicknessLow* are at the substrate. If you
            need laterally graded thicknesses, modify `get_t_thickness` and/or
            `get_b_thickness` in a subclass.

        *power*: float
            Defines the exponent of the layer thickness power law, if the
            multilayer is depth graded:

            .. math::
                d_n = A / (B + n)^{power}.

        *tThicknessLow* and *bThicknessLow*: float
            Are ignored (left as zeros) if not depth graded.

        *nPairs*: int
            The number of layer pairs.

        *idThickness*: float in Å
            RMS thickness :math:`\\sigma_{j,j-1}` of the
            interdiffusion/roughness interface.

        *substThickness*: float in Å
            Is only relevant in transmission if *substrate* is present.

        *geom*: str
            Either 'transmitted' or 'reflected'.

        """
        self.tLayer = tLayer
        self.bLayer = bLayer
        self.substrate = substrate

        self.nPairs = nPairs
        self.power = power

        self.tThicknessLow = tThicknessLow  # in Å
        self.bThicknessLow = bThicknessLow  # in Å

        self.tThickness = tThickness  # in Å
        self.bThickness = bThickness  # in Å

        self.kind = 'multilayer'
        self.geom = geom
        if not self.geom:
            self.geom = 'reflected'

        self.idThickness = idThickness
        self.substRoughness = substRoughness
        self.substThickness = substThickness

        if name:
            self.name = name
        else:
            self.name = ''
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.bl = kwargs.get('bl')
#        if bl is not None:
#            bl.materialsDict[self.uuid] = self

    @property
    def d(self):
        return float(self.tThickness + self.bThickness)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, p):
        self._power = p
        self.set_dti()
        self.set_dbi()

    @property
    def nPairs(self):
        return self._nPairs

    @nPairs.setter
    def nPairs(self, n):
        self._nPairs = int(n)
        self.set_dti()
        self.set_dbi()

    @property
    def tThickness(self):
        return self.tThicknessHigh

    @tThickness.setter
    def tThickness(self, t):
        self.tThicknessHigh = float(t)
        self.set_dti()

    @property
    def tThicknessLow(self):
        return self._tThicknessLow

    @tThicknessLow.setter
    def tThicknessLow(self, t):
        self._tThicknessLow = float(t)
        self.set_dti()

    @property
    def bThickness(self):
        return self.bThicknessHigh

    @bThickness.setter
    def bThickness(self, t):
        self.bThicknessHigh = float(t)
        self.set_dbi()

    @property
    def bThicknessLow(self):
        return self._bThicknessLow

    @bThicknessLow.setter
    def bThicknessLow(self, t):
        self._bThicknessLow = float(t)
        self.set_dbi()

    @property
    def substRoughness(self):
        return self.subRough

    @substRoughness.setter
    def substRoughness(self, t):
        self.subRough = float(t)

    @property
    def tLayer(self):
        if raycing.is_valid_uuid(self._tLayer) and self.bl is not None:
            mat = self.bl.materialsDict.get(self._tLayer)
        else:
            mat = self._tLayer
        return mat

    @tLayer.setter
    def tLayer(self, tLayer):
        self._tLayer = tLayer

    @property
    def bLayer(self):
        if raycing.is_valid_uuid(self._bLayer) and self.bl is not None:
            mat = self.bl.materialsDict.get(self._bLayer)
        else:
            mat = self._bLayer
        return mat

    @bLayer.setter
    def bLayer(self, bLayer):
        self._bLayer = bLayer

    @property
    def substrate(self):
        if raycing.is_valid_uuid(self._substrate) and self.bl is not None:
            mat = self.bl.materialsDict.get(self._substrate)
        else:
            mat = self._substrate
        return mat

    @substrate.setter
    def substrate(self, substrate):
        self._substrate = substrate

    def set_dti(self):
        if not all([hasattr(self, v) for v in
                    ['_nPairs', 'tThicknessHigh', '_tThicknessLow',
                     '_power']]):
            return

        if self.tThicknessLow:
            layers = np.arange(1, self.nPairs+1)
            tqRoot = (self.tThicknessHigh/self.tThicknessLow)**(1./self.power)
            tqB = (self.nPairs-tqRoot) / (tqRoot-1.)
            tqA = self.tThicknessHigh * (tqB+1)**self.power
            self.dti = tqA * (tqB+layers)**(-self.power)
        else:
            self.dti = np.ones(self.nPairs) * float(self.tThickness)

    def set_dbi(self):
        if not all([hasattr(self, v) for v in
                    ['_nPairs', 'bThicknessHigh', '_bThicknessLow',
                     '_power']]):
            return

        if self.bThicknessLow:
            layers = np.arange(1, self.nPairs+1)
            bqRoot = (self.bThicknessHigh/self.bThicknessLow)**(1./self.power)
            bqB = (self.nPairs-bqRoot) / (bqRoot-1.)
            bqA = self.bThicknessHigh * (bqB+1)**self.power
            self.dbi = bqA * (bqB+layers)**(-self.power)
        else:
            self.dbi = np.ones(self.nPairs) * float(self.bThickness)

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

    def get_dtheta(self, E, order=1):
        return self.get_dtheta_symmetric_Bragg(E, order=order)

    def get_dtheta_symmetric_Bragg(self, E, order=1):
        r"""
        The angle correction for the symmetric Bragg case:

        .. math::

            \delta\theta = \theta_B - \arcsin(\sqrt{m^2\lambda^2 + 8 d^2
            \overline\delta} / 2d),

        where :math:`\overline\delta` is the period-averaged real part of the
        refractive index.
        """
        nt = self.tLayer.get_refractive_index(E).real if self.tLayer else 1.
        nb = self.bLayer.get_refractive_index(E).real if self.bLayer else 1.
        tThickness = self.tThicknessHigh
        bThickness = self.bThicknessHigh
        d_ = abs((nt-1) * tThickness + (nb-1) * bThickness) / self.d
        return self.get_Bragg_angle(E, order) - np.arcsin(
            ((order * CH / E)**2 + self.d**2 * 8*d_)**0.5 / (2*self.d))

    def get_t_thickness(self, x, y, iPair):
        u"""
        The top (the upper in the period pair) layer thickness in Å as a
        function of local coordinates *x* and *y* and the index (zero at
        vacuum) of the period pair.

        For parametric surfaces, the *x* and *y* local coordinates are assumed
        to be *s* and *phi* of the parametric representation."""
        f = 1.
#       f = np.random.normal(size=len(x))*self.tError + 1 if self.tError else 1
        return self.dti[iPair] * f

    def get_b_thickness(self, x, y, iPair):
        u"""
        The bottom (the lower in the period pair) layer thickness in Å as a
        function of local coordinates *x* and *y* and the index (zero at
        vacuum) of the period pair.

        For parametric surfaces, the *x* and *y* local coordinates are assumed
        to be *s* and *phi* of the parametric representation."""
        f = 1.
#       f = np.random.normal(size=len(x))*self.tError + 1 if self.tError else 1
        return self.dbi[iPair] * f

    def get_amplitude(self, E, beamInDotNormal, x=None, y=None, ucl=None):
        r"""
        Calculates amplitude of reflectivity [Als-Nielsen]_. *E* is energy,
        *beamInDotNormal* is cosine of the angle between the incoming beam and
        the normal (:math:`\theta_0` below), both can be scalars or arrays. The
        top interface of the multilayer is assumed to be with vacuum. Returns a
        tuple of the amplitudes of s and p polarizations.

        The calculation starts from the bottommost layer (with index
        :math:`N`). The reflectivity from its top into the adjacent layer
        (:math:`N-1`) is:

        .. math::

            R_N = \frac{r_{N-1, N} + r_{N, N+1} p_N^2}
            {1 + r_{N-1, N} r_{N, N+1} p_N^2},

        where the capital :math:`R` denotes the net reflectivity of the layer
        and the small letters :math:`r` denote the interface reflectivity
        (Fresnel equations):

        .. math::

            r_{j, j+1} = \frac{Q_j - Q_{j+1}}{Q_j + Q_{j+1}},

        here :math:`N+1` refers to the substrate material and

        .. math::

            Q_j = \sqrt{Q^2 - 8k^2\delta_j + i8k^2\beta_j}, \quad
            Q = 2k\sin{\theta_0}

        and :math:`\delta_j` and :math:`\beta_j` are parts of the refractive
        index :math:`n_j = 1 - \delta_j + i\beta_j`. The phase factor
        :math:`p_j^2` is :math:`\exp(i\Delta_j Q_j)`, :math:`\Delta_j` being
        the layer thickness. The calculation proceeds recursively upwards by
        layers as

        .. math::

            R_j = \frac{r_{j-1, j} + R_{j+1} p_j^2}
            {1 + r_{j-1, j} R_{j+1} p_j^2},

        until :math:`R_1` is reached, where the 0th layer is vacuum and
        :math:`Q_0 = Q`.

        If the interdiffusion thickness is not zero,
        the reflectivity at each interface is attenuated by a factor
        of :math:`exp(-2k_{j,z}k_{j-1,z}\sigma^{2}_{j,j-1})`,
        where :math:`k_{j,z}` is longitudinal component of the wave vector
        in j-th layer [Nevot-Croce]_.

        The above formulas refer to *s* polarization. The *p* part differs at
        the interface:

        .. math::

            r^p_{j, j+1} = \frac{Q_j\frac{n_{j+1}}{n_j} -
            Q_{j+1}\frac{n_{j}}{n_{j+1}}}{Q_j\frac{n_{j+1}}{n_j} +
            Q_{j+1}\frac{n_{j}}{n_{j+1}}}

        and thus the *p* polarization part requires a separate recursive
        chain.

        .. _descr_ml_tran:

        In transmission, the recursion is the following:

        .. math::

            T_{N+1} = \frac{t_{N, N+1}t_{N+1, N+2}p_{N+1}}
            {1 + r_{N, N+1} r_{N+1, N+2} p_N^2}, \quad
            T_j = \frac{T_{j+1}t_{j-1, j}p_j}{1 + r_{j-1, j} R_{j+1} p_j^2},

        where the layer :math:`N+2` is vacuum and the interface
        transmittivities for the two polarizations are equal to:

        .. math::

            t^s_{j, j+1} = \frac{2Q_j}{Q_j + Q_{j+1}}, \quad
            t^p_{j, j+1} = \frac{2Q_j\frac{n_{j+1}}{n_j}}
            {Q_j\frac{n_{j+1}}{n_j} + Q_{j+1}\frac{n_{j}}{n_{j+1}}}

        .. [Nevot-Croce] L. Nevot and P. Croce, Rev. Phys. Appl. **15**,
            (1980) 761
        """

        k = E / CHBAR
        nt = self.tLayer.get_refractive_index(E).conjugate() if self.tLayer else 1.  # analysis:ignore
        nb = self.bLayer.get_refractive_index(E).conjugate() if self.bLayer else 1.  # analysis:ignore
        ns = self.substrate.get_refractive_index(E).conjugate() if self.substrate else 1.  # analysis:ignore

        Q = 2 * k * abs(beamInDotNormal)
        Q2 = Q**2
        k28 = 8 * k**2
        Qt = (Q2 + (nt-1)*k28)**0.5
        Qb = (Q2 + (nb-1)*k28)**0.5
        Qs = (Q2 + (ns-1)*k28)**0.5
        id2 = self.idThickness**2

        roughvt = np.exp(-0.5 * Q * Qt * id2)
        rvt_s = np.complex128((Q-Qt) / (Q+Qt) * roughvt)
        rvt_p = np.complex128((Q*nt - Qt/nt) / (Q*nt + Qt/nt) * roughvt)
        if 'tran' in self.geom:
            tvt_s = np.complex128(2*Q / (Q+Qt) * roughvt)
            tvt_p = np.complex128(2*Q*nt / (Q*nt + Qt/nt) * roughvt)

        roughtb = np.exp(-0.5 * Qt * Qb * id2)
        rtb_s = np.complex128((Qt-Qb) / (Qt+Qb) * roughtb)
        rtb_p = np.complex128((Qt/nt*nb - Qb/nb*nt) / (Qt/nt*nb + Qb/nb*nt) *
                              roughtb)
        rbt_s = -rtb_s
        rbt_p = -rtb_p
        if 'tran' in self.geom:
            ttb_s = np.complex128(2*Qt / (Qt+Qb) * roughtb)
            ttb_p = np.complex128(2*Qt/nt*nb / (Qt/nt*nb + Qb/nb*nt) * roughtb)
            tbt_s = np.complex128(2*Qb / (Qt+Qb) * roughtb)
            tbt_p = np.complex128(2*Qb/nb*nt / (Qt/nt*nb + Qb/nb*nt) * roughtb)

        rmsbs = id2 if self.tLayer else self.substRoughness**2
        roughbs = np.exp(-0.5 * Qb * Qs * rmsbs)
        rbs_s = np.complex128((Qb-Qs) / (Qb+Qs) * roughbs)
        rbs_p = np.complex128((Qb/nb*ns - Qs/ns*nb) / (Qb/nb*ns + Qs/ns*nb) *
                              roughbs)
        if 'tran' in self.geom:
            tbs_s = np.complex128(2*Qb / (Qb+Qs) * roughbs)
            tbs_p = np.complex128(2*Qb/nb*ns / (Qb/nb*ns + Qs/ns*nb) * roughbs)

        rsv_s = np.complex128((Qs-Q) / (Qs+Q) * roughbs)
        rsv_p = np.complex128((Qs/ns - Q*ns) / (Qs/ns + Q*ns) * roughbs)
        if 'tran' in self.geom:
            tsv_s = np.complex128(2*Qs / (Qs+Q) * roughbs)
            tsv_p = np.complex128(2*Qs/ns / (Qs/ns + Q*ns) * roughbs)

        if 'refl' in self.geom:
            rj_s, rj_p = rbs_s, rbs_p  # bottom layer to substrate
            extraLayer = 0
        elif 'tran' in self.geom:
            rj_s, rj_p = rsv_s, rsv_p  # substrate to vacuum
            tj_s, tj_p = tsv_s, tsv_p  # substrate to vacuum
            extraLayer = 1

        ri_s = np.zeros_like(rj_s)
        ri_p = np.zeros_like(rj_p)
        if 'tran' in self.geom:
            ti_s = np.zeros_like(rj_s)
            ti_p = np.zeros_like(rj_p)
        t0 = time.time()
        if ucl is None:
            for i in reversed(range(2*self.nPairs+extraLayer)):  # + substrate
                if i % 2 == 0:
                    if i == 0:  # topmost layer
                        rij_s, rij_p = rvt_s, rvt_p
                        if 'tran' in self.geom:
                            tij_s, tij_p = tvt_s, tvt_p
                        iQT = Qt * self.get_t_thickness(x, y, i//2)
                    elif i == 2*self.nPairs:  # substrate, only if 'tran'
                        rij_s, rij_p = rbs_s, rbs_p
                        tij_s, tij_p = tbs_s, tbs_p
                        iQT = Qs * self.substThickness
                    else:
                        rij_s, rij_p = rbt_s, rbt_p
                        if 'tran' in self.geom:
                            tij_s, tij_p = tbt_s, tbt_p
                        iQT = Qt * self.get_t_thickness(x, y, i//2)
                else:
                    rij_s, rij_p = rtb_s, rtb_p
                    if 'tran' in self.geom:
                        tij_s, tij_p = ttb_s, ttb_p
                    iQT = Qb * self.get_b_thickness(x, y, i//2)
                p1i = np.complex128(np.exp(0.5j*iQT))
                p2i = p1i**2

                rj2i_s = rj_s * p2i
                rj2i_p = rj_p * p2i
                ri_s = (rij_s + rj2i_s) / (1 + rij_s*rj2i_s)
                ri_p = (rij_p + rj2i_p) / (1 + rij_p*rj2i_p)
                if 'tran' in self.geom:
                    ti_s = tij_s * tj_s * p1i / (1 + rij_s*rj2i_s)
                    ti_p = tij_p * tj_p * p1i / (1 + rij_p*rj2i_p)
                    tj_s, tj_p = ti_s, ti_p
                rj_s, rj_p = ri_s, ri_p
            t2 = time.time()
            if raycing._VERBOSITY_ > 10:
                print('ML reflection calculated with CPU in {0:.3f} s'.format(
                      t2-t0))
        else:
            nonSlicedROArgs = [np.float64(self.dti), np.float64(self.dbi)]

            try:
                iterator = iter(E)  # analysis:ignore
            except TypeError:  # not iterable
                E *= np.ones_like(beamInDotNormal)

            if 'refl' in self.geom:
                scalarArgs = [np.int32(self.nPairs)]
                slicedROArgs = [rbs_s, rbs_p,
                                rtb_s, rtb_p,
                                rvt_s, rvt_p,
                                Qt, Qb]
                slicedRWArgs = [ri_s, ri_p]
                ri_s, ri_p = ucl.run_parallel(
                    'get_amplitude_graded_multilayer',
                    scalarArgs, slicedROArgs,
                    nonSlicedROArgs, slicedRWArgs, None, len(E))
            elif 'tran' in self.geom:
                scalarArgs = [np.int32(self.nPairs),
                              np.float64(self.substThickness)]
                slicedROArgs = [
                        rvt_s, rvt_p, tvt_s, tvt_p,
                        rbs_s, rbs_p, tbs_s, tbs_p,
                        rsv_s, rsv_p, tsv_s, tsv_p,
                        rbt_s, rbt_p, tbt_s, tbt_p,
                        rtb_s, rtb_p, ttb_s, ttb_p,
                        Qt, Qb, Qs]
                slicedRWArgs = [ti_s, ti_p]
                ti_s, ti_p = ucl.run_parallel(
                    'get_amplitude_graded_multilayer_tran',
                    scalarArgs, slicedROArgs,
                    nonSlicedROArgs, slicedRWArgs, None, len(E))
            t2 = time.time()
            if raycing._VERBOSITY_ > 10:
                print('ML reflection calculated with OCL in {0:.3f} s'.format(
                      t2-t0))

        if 'refl' in self.geom:
            # n.real (i.e. delta) may be > 0, which is a problem of tabulation
            nn = nt[0] if isinstance(nt, np.ndarray) else nt
            if (nn - 1) > 0:  # e.g. for n[Sc][Henke] at 398eV
                return ri_s.conjugate(), ri_p.conjugate()
            return ri_s, ri_p
        elif 'tran' in self.geom:
            return ti_s, ti_p


class GradedMultilayer(Multilayer):
    """
    Derivative class from :class:`Mutilayer` with graded layer thicknesses.
    """

    hiddenParams = {'substRoughness'}


class Coated(Multilayer):
    """
    Derivative class from :class:`Mutilayer` with a single reflective layer on
    a substrate.
    """

    hiddenParams = {'tLayer', 'tThickness', 'bLayer', 'bThickness', 'power',
                    'tThicknessLow', 'bThicknessLow', 'idThickness',
                    'thicknessError', 'nPairs'}

    def __init__(self, *args, **kwargs):
        u"""
        *coating*, *substrate*: instance of :class:`Material`
            Material of the mirror coating layer, and the substrate material.

        *cThickness*: float
            The thicknesses of mirror coating in Å.

        *surfaceRoughness*: float
            RMS rougness of the mirror surface in Å.

        *substRoughness*: float
            RMS rougness of the mirror substrate in Å.


        """
        coating = kwargs.pop('coating', None)
        cThickness = kwargs.pop('cThickness', 0)
        surfaceRoughness = kwargs.pop('surfaceRoughness', 0)
        super().__init__(
            bLayer=coating, bThickness=cThickness,
            idThickness=surfaceRoughness, nPairs=1, *args, **kwargs)
        self.kind = 'mirror'

    @property
    def coating(self):
        return self.bLayer

    @coating.setter
    def coating(self, cmat):
        self.bLayer = cmat

    @property
    def cThickness(self):
        return self.bThickness

    @cThickness.setter
    def cThickness(self, ct):
        self.bThickness = ct

    @property
    def surfaceRoughness(self):
        return self.idThickness

    @surfaceRoughness.setter
    def surfaceRoughness(self, ct):
        self.idThickness = ct

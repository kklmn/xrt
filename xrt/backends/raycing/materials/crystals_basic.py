# -*- coding: utf-8 -*-
import numpy as np
from ... import raycing
from ..physconsts import PI, CH, AVOGADRO
from .element import Element
from .crystal import Crystal


class CrystalFcc(Crystal):
    r"""
    A derivative class from :class:`Crystal` that defines the structure factor
    for an fcc crystal as:

    .. math::

        F_{hkl}^{fcc} = f \times \left\{ \begin{array}{rl}
        4 &\mbox{if $h,k,l$ are all even or all odd} \\ 0 &\mbox{ otherwise}
        \end{array} \right.

    """

    def get_structure_factor(self, E, sinThetaOverLambda=0, needFhkl=True):
        anomalousPart = self.elements[0].get_f1f2(E)
        F0 = 4 * (self.elements[0].Z+anomalousPart) * self.factDW
        residue = sum(i % 2 for i in self.hkl)
        if residue == 0 or residue == 3:
            f0 = self.elements[0].get_f0(sinThetaOverLambda) if needFhkl else 0
            Fhkl = 4 * (f0+anomalousPart) * self.factDW
        else:
            Fhkl = 0.
        return F0, Fhkl, Fhkl


class CrystalDiamond(CrystalFcc):
    r"""
    A derivative class from :class:`Crystal` that defines the structure factor
    for a diamond-like crystal as:

    .. math::

        F_{hkl}^{\rm diamond} = F_{hkl}^{fcc}\left(1 + e^{i\frac{\pi}{2}
        (h + k + l)}\right).
    """

    def __init__(self, *args, **kwargs):
        """"Add extra attributes needed for TT bent crystal calculation"""
        if 'name' not in kwargs:
            kwargs['name'] = 'Diamond'

        a = None
        if 'a' in kwargs:
            a = kwargs.pop('a')
            if 'hkl' in kwargs:
                hkl = kwargs['hkl']
            elif len(args) > 0:
                hkl = args[0]
            else:
                hkl = [1, 1, 1]
                kwargs['hkl'] = hkl
            sqrthkl2 = (sum(i**2 for i in hkl))**0.5
            d = a / sqrthkl2
            if len(args) > 1:
                args[1] = d
            else:
                kwargs['d'] = d

        super().__init__(*args, **kwargs)
        if a is None:
            sqrthkl2 = (sum(i**2 for i in self.hkl))**0.5
            a = self.d * sqrthkl2
        self.a = self.b = self.c = a
        self.alphaRad = self.betaRad = self.gammaRad = np.pi*0.5

    def get_structure_factor(self, E, sinThetaOverLambda=0, needFhkl=True):
        diamondToFcc = 1 + np.exp(0.5j * PI * sum(self.hkl))
        F0, Fhkl, Fhkl_ = super().get_structure_factor(
            E, sinThetaOverLambda, needFhkl)
        return F0 * 2, Fhkl * diamondToFcc, Fhkl_ * diamondToFcc.conjugate()


class CrystalSi(CrystalDiamond):
    """
    A derivative class from :class:`CrystalDiamond` that defines the crystal
    d-spacing as a function of temperature.
    """

    def __init__(self, *args, **kwargs):
        """
        *tK*: float
            Temperature in Kelvin.

        *hkl*: sequence
            hkl indices.


        """
        self.a0 = 5.430710
        self.dl_l0 = self.dl_l(273.15 + 19.9)
        self.tK = kwargs.pop('tK', 297.15)
        self.hkl = kwargs.get('hkl', (1, 1, 1))
# O'Mara, William C. Handbook of Semiconductor Silicon Technology.
# William Andrew Inc. (1990) pp. 349–352.
#        self.sqrthkl2 = (sum(i**2 for i in self.hkl))**0.5
        if 'a' in kwargs and kwargs['a'] is None:
            kwargs.pop('a')

        kwargs['d'] = self.get_a() / self.sqrthkl2
        kwargs['elements'] = 'Si'
        kwargs['hkl'] = self.hkl
        if 'name' not in kwargs:
            kwargs['name'] = 'Si'
        super().__init__(*args, **kwargs)

    def dl_l(self, t=None):
        """Calculates the crystal elongation at temperature *t*. Uses the
        parameterization from [Swenson]_. Less than 1% error; the reference
        temperature is 19.9C; data is in units of unitless; *t* must be in
        degrees Kelvin.

        .. [Swenson] C.A. Swenson, J. Phys. Chem. Ref. Data **12** (1983) 179
        """
        if t is None:
            t = self.tK
        if t >= 0.0 and t < 30.0:
            return -2.154537e-004
        elif t >= 30.0 and t < 130.0:
            return -2.303956e-014 * t**4 + 7.834799e-011 * t**3 - \
                1.724143e-008 * t**2 + 8.396104e-007 * t - 2.276144e-004
        elif t >= 130.0 and t < 293.0:
            return -1.223001e-011 * t**3 + 1.532991e-008 * t**2 - \
                3.263667e-006 * t - 5.217231e-005
        elif t >= 293.0 and t <= 1000.0:
            return -1.161022e-012 * t**3 + 3.311476e-009 * t**2 + \
                1.124129e-006 * t - 5.844535e-004
        else:
            return 1.0e+100

    def get_a(self):
        """Gives the lattice parameter."""
        return self.a0 * (self.dl_l()-self.dl_l0+1)

    def get_Bragg_offset(self, E, Eref):
        """Calculates the Bragg angle offset due to a mechanical (mounting)
        misalignment.

        *E* is the calculated energy of a spectrum feature, typically the edge
        position.

        *Eref* is the tabulated position of the same feature."""
        self.d = self.get_a() / self.sqrthkl2
        chOverTwod = CH / 2 / self.d
        return np.arcsin(chOverTwod/E) - np.arcsin(chOverTwod/Eref)


class CrystalFromCell(Crystal):
    """:class:`CrystalFromCell` builds a crystal from cell parameters and
    atomic positions which can be found e.g. in Crystals.dat of XOP [XOP]_ or
    xraylib. See also predefined crystals in module
    :mod:`~xrt.backends.raycing.materials.crystals`.

    Examples:
        >>> xtalQu = rm.CrystalFromCell(
        >>>     'alphaQuartz', (1, 0, 2), a=4.91304, c=5.40463, gamma=120,
        >>>     atoms=[14]*3 + [8]*6,
        >>>     atomsXYZ=[[0.4697, 0., 0.],
        >>>               [-0.4697, -0.4697, 1./3],
        >>>               [0., 0.4697, 2./3],
        >>>               [0.4125, 0.2662, 0.1188],
        >>>               [-0.1463, -0.4125, 0.4521],
        >>>               [-0.2662, 0.1463, -0.2145],
        >>>               [0.1463, -0.2662, -0.1188],
        >>>               [-0.4125, -0.1463, 0.2145],
        >>>               [0.2662, 0.4125, 0.5479]])
        >>>
        >>> xtalGr = rm.CrystalFromCell(
        >>>     'graphite', (0, 0, 2), a=2.456, c=6.696, gamma=120,
        >>>     atoms=[6]*4, atomsXYZ=[[0., 0., 0.], [0., 0., 0.5],
        >>>                            [1./3, 2./3, 0.], [2./3, 1./3, 0.5]])
        >>>
        >>> xtalBe = rm.CrystalFromCell(
        >>>     'Be', (0, 0, 2), a=2.287, c=3.583, gamma=120,
        >>>     atoms=[4]*2, atomsXYZ=[[1./3, 2./3, 0.25], [2./3, 1./3, 0.75]])

    """

    hiddenParams = {'d', 'V', 'kind'}

    def __init__(self, name='', hkl=[1, 1, 1],
                 a=5.430710, b=None, c=None, alpha=90, beta=90, gamma=90,
                 atoms=[14]*8,
                 atomsXYZ=[[0., 0., 0.],
                           [0., 0.5, 0.5],
                           [0.5, 0.5, 0.],
                           [0.5, 0., 0.5],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 atomsFraction=None, tK=0,
                 t=None, factDW=1.,
                 geom='Bragg reflected', table='Chantler total',
                 volumetricDiffraction=False, useTT=False, nu=0, mosaicity=0,
                 **kwargs):
        u"""
        *name*: str
            Crystal name.

        *hkl*: sequence
            hkl indices.

        *a*, *b*, *c*: float
            Cell parameters in Å. *a* must be given. *b*, *c*, if not given,
            are equlized to *a*.

        *alpha*, *beta*, *gamma*: float
            Cell angles in degrees. If not given, are equal to 90.

        *atoms*: list of str or list of int
            List of atoms in the cell given by element Z's or element names.

        *atomsXYZ*: list of 3-sequences
            List of atomic coordinates in cell units.

            .. note::

                *atoms* and *atomsXYZ* must contain *all* the atoms, not only
                the unique ones for a given symmetry group (we do not consider
                symmetry here). For example, the unit cell of magnetite (Fe3O4)
                has 3 unique atomic positions and 56 in total; here, all 56 are
                needed.

        *atomsFraction*: a list of float or None
            Atomic fractions. If None, all values are 1.


        """
        self.table = table
        self.name = name
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())
        # TODO: Do we want to call Crystal init()?
        self.hkl = hkl
        h, k, l = hkl  # analysis:ignore
        self.tK = 0
        self.a = a
        self.b = b or a  # if b is None else b
        self.c = c or a  # if c is None else c

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

#        self.alphaRad = np.radians(alpha)
#        self.betaRad = np.radians(beta)
#        self.gammaRad = np.radians(gamma)

        self.atomsXYZ = atomsXYZ
        self.atoms = atoms
        self.atomsFraction = atomsFraction
#        self.elements = []
#        uniqueElements = {}
#        for atom in atoms:
#            if atom in uniqueElements:
#                element = uniqueElements[atom]
#            else:
#                element = Element(atom, table)
#                uniqueElements[atom] = element
#            self.elements.append(element)

#        self.atomsFraction =\
#            [1 for atom in atoms] if atomsFraction is None else atomsFraction
#        self.quantities = self.atomsFraction
#        ca, cb, cg = np.cos((self.alphaRad, self.betaRad, self.gammaRad))
#        sa, sb, sg = np.sin((self.alphaRad, self.betaRad, self.gammaRad))
#        self.V = self.a * self.b * self.c *\
#            (1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)**0.5
#
#        self.mass = 0.
#        for atom, xi in zip(atoms, self.atomsFraction):
#            self.mass += xi * element.mass
#        self.rho = self.mass / AVOGADRO / self.V * 1e24
#        self.d = self.V / (self.a * self.b * self.c) *\
#            ((h*sa/self.a)**2 + (k*sb/self.b)**2 + (l*sg/self.c)**2 +
#             2*h*k * (ca*cb - cg) / (self.a*self.b) +
#             2*h*l * (ca*cg - cb) / (self.a*self.c) +
#             2*k*l * (cb*cg - ca) / (self.b*self.c))**(-0.5)
#        self.chiToF = -R0 / PI / self.V  # minus!
#        self.chiToFd2 = abs(self.chiToF) * self.d**2
        self.geom = geom
#        self.geometry = 2*int(geom.startswith('Bragg')) +\
#            int(geom.endswith('transmitted'))
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.volumetricDiffraction = volumetricDiffraction
        self.nu = nu
        self.useTT = useTT
        self.mosaicity = mosaicity
        self.refractiveIndex = None

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, table):
        self._table = table
        self.set_elements()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a
        self.set_cell_volume()

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b
        self.set_cell_volume()

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        self._c = c
        self.set_cell_volume()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self.alphaRad = np.radians(alpha)
        self._alpha = alpha
        self.set_cell_volume()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self.betaRad = np.radians(beta)
        self._beta = beta
        self.set_cell_volume()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self.gammaRad = np.radians(gamma)
        self._gamma = gamma
        self.set_cell_volume()

    @property
    def atomsFraction(self):
        return self._atomsFraction

    @atomsFraction.setter
    def atomsFraction(self, atomsFraction):
        self._atomsFraction = atomsFraction or [1 for atom in self.atoms]
        self.quantities = self._atomsFraction
        self.set_cell_volume()

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms
        self.set_elements()
        self.set_cell_volume()

    def set_elements(self):
        if hasattr(self, '_table') and hasattr(self, '_atoms'):
            self.elements = []
            uniqueElements = {}
            for atom in self._atoms:
                if atom in uniqueElements:
                    element = uniqueElements[atom]
                else:
                    element = Element(atom, self._table)
                    uniqueElements[atom] = element
                self.elements.append(element)

    def set_cell_volume(self):
        if not all([hasattr(self, v) for v in
                    ['_a', '_b', '_c',
                     '_alpha', '_beta', '_gamma',
                     '_atoms', '_atomsFraction', '_hkl']]):
            return

        ca, cb, cg = np.cos((self.alphaRad, self.betaRad, self.gammaRad))
        sa, sb, sg = np.sin((self.alphaRad, self.betaRad, self.gammaRad))
        self.V = self.a * self.b * self.c *\
            (1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)**0.5
        h, k, l = self.hkl
        self.mass = 0.
        for element, xi in zip(self.elements, self.atomsFraction):
            self.mass += xi * element.mass
        self.rho = self.mass / AVOGADRO / self.V * 1e24
        self.d = self.V / (self.a * self.b * self.c) *\
            ((h*sa/self.a)**2 + (k*sb/self.b)**2 + (l*sg/self.c)**2 +
             2*h*k * (ca*cb - cg) / (self.a*self.b) +
             2*h*l * (ca*cg - cb) / (self.a*self.c) +
             2*k*l * (cb*cg - ca) / (self.b*self.c))**(-0.5)

    def get_structure_factor(self, E, sinThetaOverLambda=0, needFhkl=True):
        F0, Fhkl, Fhkl_ = 0, 0, 0
        uniqueElements = {}
        for el, xyz, af in zip(
                self.elements, self.atomsXYZ, self.atomsFraction):
            if el.Z in uniqueElements:
                f0, anomalousPart = uniqueElements[el.Z]
            else:
                f0 = el.get_f0(sinThetaOverLambda) if needFhkl else 0
                anomalousPart = el.get_f1f2(E)
                uniqueElements[el.Z] = f0, anomalousPart
            F0 += af * (el.Z+anomalousPart) * self.factDW
            fact = af * (f0+anomalousPart) * self.factDW
            expiHr = np.exp(2j * np.pi * np.dot(xyz, self.hkl))
            Fhkl += fact * expiHr
            Fhkl_ += fact / expiHr
        return F0, Fhkl, Fhkl_


class Powder(CrystalFromCell):
    u"""
    A derivative class from :class:`CrystalFromCell` with randomly distributed
    atomic plane orientations similar to the real polycrystalline powders. The
    distribution is uniform in the spherical coordinates, so that the angles of
    longitudinal and transverse deflection (θ and χ) are both functions of
    uniformly sampled over [0, 1) variables μ and ν: θ = arccos(μ), χ = 2πν.

    The class parameter *hkl* defines the highest reflex, so that
    reflectivities are calculated for all possible combinations of indices
    [mnp], where 0 ≤ m ≤ h, 0 ≤ n ≤ k, 0 ≤ p ≤ l. Only one reflection with the
    highest amplitude is picked for each incident ray.

    .. warning::
        Heavy computational load. Requires OpenCL.

    """

    def __init__(self, *args, **kwargs):
        u"""
        *chi*: 2-list of floats [min, max]
            Limits of the χ angle distribution. Zero and π/2 angles correspond
            to the positive directions of *x* and *z* axes.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        CrystalFromCell.__init__(self, *args, **kwargs)
        self.kind = 'powder'

    def __pop_kwargs(self, **kwargs):

        self.chi = kwargs.pop('chi', [0, 0.5*np.pi])
        return kwargs


class CrystalHarmonics(CrystalFromCell):
    u"""
    A derivative class from :class:`CrystalFromCell`, used to calculate
    multiple orders of the given reflex in one run: n*[hkl], where 1 ≤ n ≤ Nmax
    i.e. [111], [222], [333] or [220], [440], [660]. Only one harmonic with
    highest reflectivity is picked for each incident ray. Use this class to
    estimate the efficiency of higher harmonic rejection schemes.

    .. warning::
        Heavy computational load. Requires OpenCL.

    """

    def __init__(self, *args, **kwargs):
        u"""
        *Nmax*: int
            Specifies the highest order of reflection to be calculated.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        CrystalFromCell.__init__(self, *args, **kwargs)
        self.kind = 'crystal harmonics'

    def __pop_kwargs(self, **kwargs):
        self.Nmax = kwargs.pop('Nmax', 3)
        return kwargs


class MonoCrystal(CrystalFromCell):
    u"""
    A derivative class from :class:`CrystalFromCell`, used for calculation of
    the single crystal diffraction patterns (so far cubic symettries only).
    Similar to the parent class, parameter *hkl* defines the cut orientation,
    whereas *Nmax* stands for the highest index to consider, i.e. for every ray
    the code would calculate the range of reflexes from [-Nmax, -Nmax, -Nmax]
    to [Nmax, Nmax, Nmax] (required amount of reflectivity calculations is
    therefore 2*(2*Nmax+1)^3 per every ray), but only return one of them
    regarding their intensities. Brighter reflexes would be selected with
    higher probability.

    .. warning::
        Heavy computational load. Requires OpenCL. Decent GPU highly
        recommended.

    """

    def __init__(self, *args, **kwargs):
        u"""
        *Nmax*: int
            Specifies the highest order of reflection to be calculated.


        """
        kwargs = self.__pop_kwargs(**kwargs)
        CrystalFromCell.__init__(self, *args, **kwargs)
        self.kind = 'monocrystal'

    def __pop_kwargs(self, **kwargs):
        self.Nmax = kwargs.pop('Nmax', 3)
        return kwargs

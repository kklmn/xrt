# -*- coding: utf-8 -*-
"""
Materials
---------

Module :mod:`~xrt.backends.raycing.materials` defines atomic and material
properties related to x-ray scattering, diffraction and propagation:
reflectivity, transmittivity, refractive index, absorption coefficient etc.

.. autofunction:: read_atomic_data

.. autoclass:: Element()
   :members: __init__, read_f0_Kissel, get_f0, read_f1f2_vs_E, get_f1f2
.. autoclass:: Material()
   :members: __init__, get_refractive_index, get_absorption_coefficient,
             get_amplitude

.. autoclass:: Multilayer()
   :members: __init__, get_amplitude, get_dtheta_symmetric_Bragg
.. autoclass:: Crystal(Material)
   :members: __init__, get_Darwin_width, get_amplitude,
             get_dtheta_symmetric_Bragg, get_dtheta, get_dtheta_regular

.. autoclass:: CrystalFcc(Crystal)
   :members: get_structure_factor
.. autoclass:: CrystalDiamond(CrystalFcc)
   :members: get_structure_factor
.. autoclass:: CrystalSi(CrystalDiamond)
   :members: __init__, dl_l, get_a, get_Bragg_offset
.. autoclass:: CrystalFromCell(Crystal)
   :members: __init__
.. autoclass:: Powder(CrystalFromCell)
   :members: __init__
.. autoclass:: CrystalHarmonics(CrystalFromCell)
   :members: __init__
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "16 Mar 2017"
__all__ = ('Material', 'EmptyMaterial', 'Multilayer', 'GradedMultilayer',
           'CoatedMirror', 'Crystal', 'CrystalFcc',
           'CrystalDiamond', 'CrystalSi', 'CrystalFromCell',
           'Powder', 'CrystalHarmonics')
import collections
__allSectioned__ = collections.OrderedDict([
    ('Material', None),
    ('Crystals', ('CrystalSi', 'CrystalDiamond', 'CrystalFcc',
                  'CrystalFromCell')),  # don't include 'Crystal'
    ('Layered', ('CoatedMirror', 'Multilayer', 'GradedMultilayer')),
    ('Advanced', ('Powder', 'CrystalHarmonics', 'EmptyMaterial'))
    ])
import sys
import os
import time
# import struct
import pickle
import numpy as np
from scipy.special import jn as besselJn

from .physconsts import PI, PI2, CH, CHBAR, R0, AVOGADRO

try:
    import pyopencl as cl  # analysis:ignore
    isOpenCL = True
except ImportError:
    isOpenCL = False

ch = CH  # left here for copatibility
chbar = CHBAR  # left here for copatibility

try:  # for Python 3 compatibility:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    unicode = unicode
    basestring = basestring

elementsList = (
    'none', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U')


def read_atomic_data(elem):
    u"""
    Reads atomic data from ``AtomicData.dat`` file adopted from XOP [XOP]_.
    It has the following data:
    0  AtomicRadius[Å]  CovalentRadius[Å]  AtomicMass  BoilingPoint[K]
    MeltingPoint[K]  Density[g/ccm]  AtomicVolume
    CoherentScatteringLength[1E-12cm]  IncoherentX-section[barn]
    Absorption@1.8Å[barn]  DebyeTemperature[K]  ThermalConductivity[W/cmK]

    In :meth:`read_atomic_data` only the mass is inquired. The user may
    extend the method to get the other values by simply adding the
    corresponding array elements to the returned value."""
    if isinstance(elem, basestring):
        Z = elementsList.index(elem)
    elif isinstance(elem, int):
        Z = elem
    else:
        raise NameError('Wrong element')
    dataDir = os.path.dirname(__file__)
    with open(os.path.join(dataDir, 'data', 'AtomicData.dat')) as f:
        for li in f:
            fields = li.split()
            if int(fields[0]) == Z:
                atomicData = [float(x) for x in fields]
                break
    return atomicData[3]


class Element(object):
    """This class serves for accessing the scattering factors f0, f1 and f2 of
    a chemical element. It can also report other atomic data listed in
    ``AtomicData.dat`` file adopted from XOP [XOP]_.
    """
    def __init__(self, elem=None, table='Chantler'):
        u"""
        *elem*: str or int
            The element can be specified by its name (case sensitive) or its
            ordinal number.

        *table*: str
            This parameter is explained in the description of
            :class:`Material`.


           """
        if isinstance(elem, basestring):
            self.name = elem
            self.Z = elementsList.index(elem)
        elif isinstance(elem, int):
            self.name = elementsList[elem]
            self.Z = elem
        else:
            raise NameError('Wrong element')
        self.f0coeffs = self.read_f0_Kissel()
        self.E, self.f1, self.f2 = self.read_f1f2_vs_E(table=table)
        self.mass = read_atomic_data(self.Z)

    def read_f0_Kissel(self):
        r"""
        Reads f0 scattering factors from the tabulation of XOP [XOP]_. These
        were calculated by [Kissel]_ and then parameterized as [Waasmaier]_:

        .. math::

            f_0\left(\frac{q}{4\pi}\right) = c + \sum_{i=1}^5{a_i\exp\left(-b_i
            \left(q/(4\pi)\right)^2\right)}

        where :math:`q/(4\pi) = \sin{\theta} / \lambda` and :math:`a_i`,
        :math:`b_i` and :math:`c` are the coefficients tabulated in the file
        ``f0_xop.dat``.

        .. [Kissel] L. Kissel, Radiation physics and chemistry **59** (2000)
           185-200, http://www-phys.llnl.gov/Research/scattering/RTAB.html

        .. [Waasmaier] D. Waasmaier & A. Kirfel, Acta Cryst. **A51** (1995)
           416-413
        """
        dataDir = os.path.dirname(__file__)
        with open(os.path.join(dataDir, 'data', 'f0_xop.dat')) as f:
            for li in f:
                if li.startswith("#S"):
                    fields = li.split()
                    if int(fields[1]) == self.Z:
                        break
            else:
                raise ValueError('cannot find the element {0}'.format(self.Z))
            for li in f:
                if li.startswith("#UP"):
                    if sys.version_info < (3, 1):
                        li = f.next()
                    else:
                        li = next(f)
                    break
            else:
                raise ValueError('wrong file format!')
        return [float(x) for x in li.split()]
#              = [a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]

    def get_f0(self, qOver4pi=0):  # qOver4pi = sin(theta) / lambda
        """Calculates f0 for the given *qOver4pi*."""
        return self.f0coeffs[5] + sum(
            a * np.exp(-b * qOver4pi**2)
            for a, b in zip(self.f0coeffs[:5], self.f0coeffs[6:]))

    def read_f1f2_vs_E(self, table):
        """Reads f1 and f2 scattering factors from the given *table* at the
        instantiation time."""
        dataDir = os.path.dirname(__file__)

#        pname = os.path.join(dataDir, 'data', table+'.pickle')
#        with open(pname, 'rb') as f:
#            res = pickle.load(f, encoding='bytes') if isPython3 else\
#                pickle.load(f)
#        return res[self.Z]

        table_fn = table.split()[0]
        pname = os.path.join(dataDir, 'data', table_fn+'.npz')
        f2key = '_f2tot' if 'total' in table else '_f2'
        with open(pname, 'rb') as f:
            res = np.load(f)
            ef1f2 = (np.array(res[self.name+'_E']),
                     np.array(res[self.name+'_f1']),
                     np.array(res[self.name+f2key]))
        return ef1f2

#        pname = os.path.join(dataDir, 'data', table+'.Ef')
#        E, f1, f2 = [], [], []
#        startFound = False
#        with open(pname, "rb") as f:
#            while True:
#                structEf1f2 = f.read(12)
#                if not structEf1f2:
#                    break
#                ELoc, f1Loc, f2Loc = struct.unpack_from("<3f", structEf1f2)
#                if startFound and ELoc == -1:
#                    break
#                if ELoc == -1 and f2Loc == self.Z:
#                    startFound = True
#                    continue
#                if startFound:
#                    E.append(ELoc)
#                    f1.append(f1Loc - self.Z)
#                    f2.append(f2Loc)
#        return np.array(E), np.array(f1), np.array(f2)

    def get_f1f2(self, E):
        """Calculates (interpolates) f1 and f2 for the given array *E*."""
        if np.any(E < self.E[0]) or np.any(E > self.E[-1]):
            raise ValueError(
                ('E={0} is out of the data table range ' +
                 '[{1}, {2}]!!! Use another table.').format(
                    E[np.where((E < self.E[0]) | (E > self.E[-1]))], self.E[0],
                    self.E[-1]))
        f1 = np.interp(E, self.E, self.f1)
        f2 = np.interp(E, self.E, self.f2)
        return f1 + 1j*f2


class Material(object):
    """
    :class:`Material` serves for getting reflectivity, transmittivity,
    refractive index and absorption coefficient of a material specified by its
    chemical formula and density."""
    def __init__(self, elements=None, quantities=None, kind='auto', rho=0,
                 t=None, table='Chantler total', efficiency=None,
                 efficiencyFile=None, name=''):
        r"""
        *elements*: str or sequence of str
            Contains all the constituent elements (symbols)

        *quantities*: None or sequence of floats of length of *elements*
            Coefficients in the chemical formula. If None, the coefficients
            are all equal to 1.

        *kind*: str
            One of 'mirror', 'thin mirror', 'plate', 'lens', 'grating', 'FZP'.
            If 'auto', the optical element will decide which material kind to
            use via its method :meth:`assign_auto_material_kind`.

        *rho*: float
            Density in g/cm\ :sup:`3`.

        *t*: float
            Thickness in mm, required only for 'thin mirror'.

        *table*: str
            At the time of instantiation the tabulated scattering factors of
            each element are read and then interpolated at the requested **q**
            value and energy. *table* can be 'Henke' (10 eV < *E* < 30 keV)
            [Henke]_, 'Chantler' (11 eV < *E* < 405 keV) [Chantler]_ or 'BrCo'
            (30 eV < *E* < 509 keV) [BrCo]_.

            The tables of f2 factors consider only photoelectric
            cross-sections. The tabulation by Chantler can optionally have
            *total* absorption cross-sections. This option is enabled by
            *table* = 'Chantler total'.

        .. [Henke] http://henke.lbl.gov/optical_constants/asf.html
           B.L. Henke, E.M. Gullikson, and J.C. Davis, *X-ray interactions:
           photoabsorption, scattering, transmission, and reflection at
           E=50-30000 eV, Z=1-92*, Atomic Data and Nuclear Data Tables
           **54** (no.2) (1993) 181-342.

        .. [Chantler] http://physics.nist.gov/PhysRefData/FFast/Text/cover.html
           http://physics.nist.gov/PhysRefData/FFast/html/form.html
           C. T. Chantler, *Theoretical Form Factor, Attenuation, and
           Scattering Tabulation for Z = 1 - 92 from E = 1 - 10 eV to E = 0.4 -
           1.0 MeV*, J. Phys. Chem. Ref. Data **24** (1995) 71-643.

        .. [BrCo] http://www.bmsc.washington.edu/scatter/periodic-table.html
           ftp://ftpa.aps.anl.gov/pub/cross-section_codes/
           S. Brennan and P.L. Cowan, *A suite of programs for calculating
           x-ray absorption, reflection and diffraction performance for a
           variety of materials at arbitrary wavelengths*, Rev. Sci. Instrum.
           **63** (1992) 850-853.

        *efficiency*: sequence of pairs [*order*, *value*]
            Can be given for *kind* = 'grating' and *kind* = 'FZP'. It must
            correspond to the field *order* of the OE. It can be given as a
            constant per diffraction order or as an energy dependence, also per
            diffraction order. It is a sequence of pairs [*order*, *value*],
            where *value* is either the efficiency itself or an index in the
            data file given by *efficiencyFile*. The data file can either be
            (1) a pickle file with *energy* and *efficiency* arrays as two
            first dump elements and *efficiency* shape as (len(*energy*),
            *orders*) or (2) a column file with energy in the leftmost column
            and the order efficiencies in the next columns. The *value* is a
            corresponding array index (zero-based) or a column number (also
            zero-based, the 0th column is energy). An example of the efficiency
            calculation can be found in
            ``\examples\withRaycing\11_Wave\waveGrating.py``.

        *efficiencyFile*: str
            See the definition of *efficiency*.

        *name*: str
            Material name. Not used by xrt. Can be used by the user for
            annotations of graphs or other output purposes. If empty, the name
            is constructed from the *elements* and the *quantities*.


        """
        if isinstance(elements, basestring):
            elements = elements,
        if quantities is None:
            self.quantities = [1. for elem in elements]
        else:
            self.quantities = quantities
        self.elements = []
        self.mass = 0.
        if name:
            self.name = name
            autoName = False
        else:
            self.name = r''
            autoName = True
        for elem, xi in zip(elements, self.quantities):
            newElement = Element(elem, table)
            self.elements.append(newElement)
            self.mass += xi * newElement.mass
            if autoName:
                self.name += elem
                if xi != 1:
                    self.name += '$_{' + '{0}'.format(xi) + '}$'
        self.kind = kind  # 'mirror', 'thin mirror', 'plate', 'lens'
        if self.kind == 'thin mirror':
            if t is None:
                raise ValueError('Give the thin mirror a thickness!')
            self.t = t  # t in mm
        self.rho = rho  # density g/cm^3
        self.geom = ''
        self.efficiency = efficiency
        self.efficiencyFile = efficiencyFile
        if efficiencyFile is not None:
            self.read_efficiency_file()

    def read_efficiency_file(self):
        cols = [c[1] for c in self.efficiency]
        if self.efficiencyFile.endswith('.pickle'):
            with open(self.efficiencyFile, 'rb') as f:
                res = pickle.load(f)
                es, eff = res[0], res[1].T[cols, :]
        else:
            es = np.loadtxt(self.efficiencyFile, usecols=(0,), unpack=True)
            eff = (np.loadtxt(self.efficiencyFile, usecols=cols,
                              unpack=True)).reshape(len(cols), -1)
        self.efficiency_E = es
        self.efficiency_I = eff

    def get_refractive_index(self, E):
        r"""
        Calculates refractive index at given *E*. *E* can be an array.

        .. math::

            n = 1 - \frac{r_0\lambda^2 N_A \rho}{2\pi M}\sum_i{x_i f_i(0)}

        where :math:`r_0` is the classical electron radius, :math:`\lambda` is
        the wavelength, :math:`N_A` is Avogadro’s number, :math:`\rho` is the
        material density, *M* is molar mass, :math:`x_i` are atomic
        concentrations (coefficients in the chemical formula) and
        :math:`f_i(0)` are the complex atomic scattering factor for the forward
        scattering.
        """
        xf = np.zeros_like(E) * 0j
        for elem, xi in zip(self.elements, self.quantities):
            xf += (elem.Z + elem.get_f1f2(E)) * xi
        return 1 - 1e-24 * AVOGADRO * R0 / PI2 * (CH/E)**2 * self.rho * \
            xf / self.mass  # 1e-24 = A^3/cm^3

    def get_absorption_coefficient(self, E):  # mu0
        r"""
        Calculates the linear absorption coefficient from the imaginary part of
        refractive index. *E* can be an array. The result is in cm\ :sup:`-1`.

        .. math::

            \mu = \Im(n)/\lambda.
        """
        return abs((self.get_refractive_index(E)).imag) * E / CHBAR * 2e8

    def get_grating_efficiency(self, beam, good):
        """Gets grating efficiency from the parameters *efficiency* and
        *efficiencyFile* supplied at the instantiation."""
        resI = np.zeros(good.sum())
        order = beam.order[good]
        if self.efficiencyFile is None:
            for eff in self.efficiency:
                resI[order == eff[0]] = eff[1]
        else:
            E = beam.E[good]
            Emin = self.efficiency_E[0]
            Emax = self.efficiency_E[-1]
            if (np.any(E < Emin) or np.any(E > Emax)):
                raise ValueError(
                    ('E={0} is out of the efficiency table range ' +
                     '[{1}, {2}]!!! Use another table.').format(
                        E[np.where((E < Emin) | (E > Emax))], Emin, Emax))
            for ieff, eff in enumerate(self.efficiency):
                resI[order == eff[0]] = np.interp(
                    E[order == eff[0]], self.efficiency_E,
                    self.efficiency_I[ieff])
        resA = resI**0.5
        return resA, resA, 0

    def get_amplitude(self, E, beamInDotNormal, fromVacuum=True):
        r"""
        Calculates amplitude of reflectivity (for 'mirror' and 'thin mirror')
        or transmittivity (for 'plate' and 'lens') [wikiFresnelEq]_,
        [Als-Nielsen]_. *E* is energy, *beamInDotNormal* is cosine of the angle
        between the incoming beam and the normal (:math:`\theta_1` below), both
        can be scalars or arrays. The interface of the material is assumed to
        be with vacuum; the direction is given by boolean *fromVacuum*. Returns
        a tuple of the amplitudes of s and p polarizations and the absorption
        coefficient in cm\ :sup:`-1`.

        .. math::

            r_s^{\rm mirror} &= \frac{n_1\cos{\theta_1} - n_2\cos{\theta_2}}
            {n_1\cos{\theta_1} + n_2\cos{\theta_2}}\\
            r_p^{\rm mirror} &= \frac{n_2\cos{\theta_1} - n_1\cos{\theta_2}}
            {n_2\cos{\theta_1} + n_1\cos{\theta_2}}\\
            r_{s,p}^{\rm thin\ mirror} &= r_{s,p}^{\rm mirror}\frac{1 - p^2}
            {1 - (r_{s,p}^{\rm mirror})^2p^2},

        where the phase factor
        :math:`p^2 = \exp(2iEtn_2\cos{\theta_2}/c\hbar)`.

        .. math::

            t_s^{\rm plate,\ lens} &= 2\frac{n_1\cos{\theta_1}}
            {n_1\cos{\theta_1} + n_2\cos{\theta_2}}t_f\\
            t_p^{\rm plate,\ lens} &= 2\frac{n_1\cos{\theta_1}}
            {n_2\cos{\theta_1} + n_1\cos{\theta_2}}t_f\\

        where :math:`t_f = \sqrt{\frac{\Re(n_2n_1)\cos{\theta_2}}
        {cos{\theta_1}}}/|n_1|`.

        .. [wikiFresnelEq] http://en.wikipedia.org/wiki/Fresnel_equations .
        .. [Als-Nielsen] Jens Als-Nielsen, Des McMorrow, *Elements of Modern
           X-ray Physics*, John Wiley and Sons, 2001.
        """
#        if self.kind in ('grating', 'FZP'):
        if self.kind in ('FZP'):
            return 1, 1, 0
        n = self.get_refractive_index(E)
        if fromVacuum:
            n1 = 1.
            n2 = n
        else:
            n1 = n
            n2 = 1.
        cosAlpha = abs(beamInDotNormal)
        sinAlpha2 = 1 - beamInDotNormal**2
        if isinstance(sinAlpha2, np.ndarray):
            sinAlpha2[sinAlpha2 < 0] = 0
        n1cosAlpha = n1 * cosAlpha
#        cosBeta = np.sqrt(1 - (n1.real/n2.real*sinAlpha)**2)
        cosBeta = np.sqrt(1 - (n1/n2)**2*sinAlpha2)
        n2cosBeta = n2 * cosBeta
        if self.kind in ('mirror', 'thin mirror', 'grating'):  # reflectivity
            rs = (n1cosAlpha - n2cosBeta) / (n1cosAlpha + n2cosBeta)
            rp = (n2*cosAlpha - n1*cosBeta) / (n2*cosAlpha + n1*cosBeta)
            if self.kind == 'thin mirror':
                p2 = np.exp(2j * E / CHBAR * n2cosBeta * self.t * 1e7)
                rs *= (1 - p2) / (1 - rs**2*p2)
                rp *= (1 - p2) / (1 - rp**2*p2)
        elif self.kind in ('plate', 'lens', 'FZP'):  # transmittivity
            tf = np.sqrt(
                (n2cosBeta * n1.conjugate()).real / cosAlpha) / abs(n1)
            rs = 2 * n1cosAlpha / (n1cosAlpha + n2cosBeta) * tf
            rp = 2 * n1cosAlpha / (n2*cosAlpha + n1*cosBeta) * tf
        else:
            raise ValueError('Unknown kind of material for {0}'.format(
                    self.name))
#        return rs, rp, abs(n.imag) * E / CHBAR * 2e8  # 1/cm
        return (rs, rp,
                abs(n.imag) * E / CHBAR * 2e8,  # 1/cm
                n.real * E / CHBAR * 1e8)


class EmptyMaterial(object):
    """
    This class provides an empty (i.e. without reflectivity) 'grating'
    material. For other kinds of empty materials just use None.
    """
    def __init__(self, kind='grating'):
        self.kind = kind
        self.geom = ''


class Multilayer(object):
    u"""
    :class:`Multilayer` serves for getting reflectivity of a multilayer. The
    multilayer may have variable thicknesses of the two alternating layers as
    functions of local *x* and *y* and/or as a function of the layer number.
    """

    hiddenParams = ['power', 'substRoughness', 'tThicknessLow',
                    'bThicknessLow']

    def __init__(self, tLayer=None, tThickness=0., bLayer=None, bThickness=0.,
                 nPairs=0., substrate=None, tThicknessLow=0., bThicknessLow=0.,
                 idThickness=0., power=2., substRoughness=0, name=''):
        u"""
        *tLayer*, *bLayer*, *substrate*: instance of :class:`Material`
            The top layer material, the bottom layer material and the substrate
            material.

        *tThickness* and *bThickness*: float
            The thicknesses of the layers in Å. If the multilayer is depth
            graded, *tThickness* and *bThickness* are at the top and
            *tThicknessLow* and *bThicknessLow* are at the substrate.

        *power*: float
            Defines the exponent of the layer thickness power law, if the
            multilayer is depth graded:

            .. math::
                d_n = A / (B + n)^{power}.

        *tThicknessLow* and *bThicknessLow*: float
            Are ignored (left as zeros) if not depth graded.

        *nPairs*: int
            The number of layer pairs.

        *idThickness*: float
            RMS thickness :math:`\\sigma_{j,j-1}` of the
            interdiffusion/roughness interface in Å.


        """
        self.tLayer = tLayer
        self.tThicknessHigh = float(tThickness)  # in Å
        self.tThicknessLow = float(tThicknessLow)  # in Å
        self.bLayer = bLayer
        self.bThicknessHigh = float(bThickness)
        self.bThicknessLow = float(bThicknessLow)  # in Å
        self.nPairs = nPairs
        self.substrate = substrate
        self.d = float(tThickness + bThickness)
        # self.tb = tThicknessTop/self.d
        # self.dLow = float(tThicknessLow + bThicknessLow)
        self.kind = 'multilayer'
        self.geom = 'Bragg reflected'
        self.idThickness = idThickness
        self.subRough = substRoughness
        if name:
            self.name = name
        else:
            self.name = ''

        layers = np.arange(1, nPairs+1)
        if tThicknessLow:
            tqRoot = (self.tThicknessHigh/self.tThicknessLow)**(1./power)
            tqB = (nPairs-tqRoot) / (tqRoot-1.)
            tqA = self.tThicknessHigh * (tqB+1)**power
            self.dti = tqA * (tqB+layers)**(-power)
        else:
            self.dti = np.ones(self.nPairs) * float(tThickness)
#            self.dti = np.array([float(tThickness)] * self.nPairs)

        if bThicknessLow:
            bqRoot = (self.bThicknessHigh/self.bThicknessLow)**(1./power)
            bqB = (nPairs-bqRoot) / (bqRoot-1.)
            bqA = self.bThicknessHigh * (bqB+1)**power
            self.dbi = bqA * (bqB+layers)**(-power)
        else:
            self.dbi = np.ones(self.nPairs) * float(bThickness)
#            self.dbi = np.array([float(bThickness)] * self.nPairs)

        print(self.idThickness, self.subRough, self.nPairs)

    def get_Bragg_angle(self, E, order=1):
        a = order * CH / (2*self.d*E)
        try:
            a[a > 1] = 1 - 1e-16
            a[a < -1] = -1 + 1e-16
        except TypeError:
            if a > 1:
                a = 1 - 1e-16
            elif a < -1:
                a = -1 + 1e-16
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
        return self.get_Bragg_angle(E) - np.arcsin(
            ((order * CH / E)**2 + self.d**2 * 8*d_)**0.5 / (2*self.d))

    def get_t_thickness(self, x, y, iPair):
        f = 1.
#       f = np.random.normal(size=len(x))*self.tError + 1 if self.tError else 1
        return self.dti[iPair] * f

    def get_b_thickness(self, x, y, iPair):
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

            Q_j = \sqrt{Q^2 - 8k^2\delta_j + i8k^2\beta_j},

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
        in j-th layer [Nevot-Croce].


        The above formulas refer to *s* polarization. The *p* part differs at
        the interface:

        .. math::

            r^p_{j, j+1} = \frac{Q_j\frac{n_{j+1}}{n_j} -
            Q_{j+1}\frac{n_{j}}{n_{j+1}}}{Q_j\frac{n_{j+1}}{n_j} +
            Q_{j+1}\frac{n_{j}}{n_{j+1}}}

        and thus the *p* polarization part requires a separate recursive
        chain.

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

        roughtb = np.exp(-0.5 * Qt * Qb * id2)
        rtb_s = np.complex128((Qt-Qb) / (Qt+Qb) * roughtb)
        rtb_p = np.complex128((Qt/nt*nb - Qb/nb*nt) / (Qt/nt*nb + Qb/nb*nt) *
                              roughtb)
        rbt_s = -rtb_s
        rbt_p = -rtb_p

        roughvt = np.exp(-0.5 * Q * Qt * id2)
        rvt_s = np.complex128((Q-Qt) / (Q+Qt) * roughvt)
        rvt_p = np.complex128((Q*nt - Qt/nt) / (Q*nt + Qt/nt) * roughvt)

        rmsbs = id2 if self.tLayer else self.subRough**2
        roughbs = np.exp(-0.5 * Qb * Qs * rmsbs)
        rbs_s = np.complex128((Qb-Qs) / (Qb+Qs) * roughbs)
        rbs_p = np.complex128((Qb/nb*ns - Qs/ns*nb) / (Qb/nb*ns + Qs/ns*nb) *
                              roughbs)
        rj_s, rj_p = rbs_s, rbs_p  # bottom layer to substrate
        ri_s = np.zeros_like(rj_s)
        ri_p = np.zeros_like(rj_p)
        t0 = time.time()
        if ucl is None:
            for i in reversed(range(2*self.nPairs)):
                if i % 2 == 0:
                    if i == 0:  # topmost layer
                        rij_s, rij_p = rvt_s, rvt_p
                    else:
                        rij_s, rij_p = rbt_s, rbt_p
                    p2i = np.complex128(
                        np.exp(1j*Qt*self.get_t_thickness(x, y, i//2)))
                else:
                    rij_s, rij_p = rtb_s, rtb_p
                    p2i = np.complex128(
                        np.exp(1j*Qb*self.get_b_thickness(x, y, i//2)))
                ri_s = (rij_s + rj_s*p2i) / (1 + rij_s*rj_s*p2i)
                ri_p = (rij_p + rj_p*p2i) / (1 + rij_p*rj_p*p2i)
                rj_s, rj_p = ri_s, ri_p
            t2 = time.time()
            print('ML reflection calculated with CPU in {} s'.format(t2-t0))
        else:
            scalarArgs = [np.int32(self.nPairs)]

            slicedROArgs = [rbs_s, rbs_p,
                            rtb_s, rtb_p,
                            rvt_s, rvt_p,
                            Qt, Qb]

            nonSlicedROArgs = [np.float64(self.dti), np.float64(self.dbi)]

            slicedRWArgs = [ri_s,
                            ri_p]

            try:
                iterator = iter(E)
            except TypeError:  # not iterable
                E *= np.ones_like(beamInDotNormal)
            try:
                iterator = iter(beamInDotNormal)  # analysis:ignore
            except TypeError:  # not iterable
                beamInDotNormal *= np.ones_like(E)
            ri_s, ri_p = ucl.run_parallel(
                'get_amplitude_graded_multilayer', scalarArgs, slicedROArgs,
                nonSlicedROArgs, slicedRWArgs, None, len(E))
            t2 = time.time()
            print('ML reflection calculated with OCL in {} s'.format(t2-t0))
        return ri_s, ri_p


class GradedMultilayer(Multilayer):
    """
    Derivative class from :class:`Mutilayer` with single reflective layer on
    substrate.
    """

    hiddenParams = ['substRoughness']


class CoatedMirror(Multilayer):
    """
    Derivative class from :class:`Mutilayer` with single reflective layer on
    substrate.
    """

    hiddenParams = ['tLayer', 'tThickness', 'bLayer', 'bThickness', 'power',
                    'tThicknessLow', 'bThicknessLow', 'idThickness',
                    'thicknessError', 'nPairs']

    def __init__(self, *args, **kwargs):
        u"""
        *coating*, *substrate*: instance of :class:`Material`
            Material of the mirror coating layer, and the substrate
            material.

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
        super(CoatedMirror, self).__init__(
            bLayer=coating, bThickness=cThickness,
            idThickness=surfaceRoughness, nPairs=1, *args, **kwargs)
        self.kind = 'mirror'


class Crystal(Material):
    u"""The parent class for crystals. The descendants must define
    :meth:`get_structure_factor`. :class:`Crystal` gives reflectivity and
    transmittivity of a crystal in Bragg and Laue cases."""

    hiddenParams = ['nuPoisson', 'calcBorrmann', 'useTT']

    def __init__(self, hkl=[1, 1, 1], d=0, V=None, elements='Si',
                 quantities=None, rho=0, t=None, factDW=1.,
                 geom='Bragg reflected', table='Chantler', name='',
                 nuPoisson=0., calcBorrmann=None, useTT=False):
        u"""
        *hkl*: sequence
            hkl indices.

        *d*: float
            Interatomic spacing in Å.

        *V*: float
            Unit cell volume in Å\ :sup:`3`. If not given, is calculated from
            *d* assuming a cubic symmetry.

        *factDW*: float
            Debye-Waller factor applied to the structure factor.

        *geom*: str
            The 1st word is either 'Bragg' or 'Laue', the 2nd word is either
            'transmitted' or 'reflected' or 'Fresnel' (the optical element must
            then provide `local_g` method that gives the grating vector).

        *table*: str
            This parameter is explained in the description of the parent class
            :class:`Material`.

        *nuPoisson*: float
            Poisson's ratio. Used to calculate the properties of bent crystals.

        *calcBorrmann*: str
            Controls the origin of the ray leaving the crystal. Can be 'None',
            'uniform', 'Bessel' or 'TT'. If 'None', the point of reflection
            is located on the surface of incidence. In all other cases the
            coordinate of the exit point is sampled according to the
            corresponding distribution: 'uniform' is a fast approximation for
            thick crystals, 'Bessel' is exact solution for the flat crystals,
            'TT' is exact solution of Takagi-Taupin equations for bent and flat
            crystals ('TT' requires *targetOpenCL* in the Optical Element to be
            not 'None' and *useTT* in the :class:`Crystal` to be 'True'. Not
            recommended for crystals thicker than 100 µm due to heavy
            computational load).

        *useTT*: bool
            Specifies whether the reflectivity will by calculated by analytical
            formula or by solution of the Takagi-Taupin equations (so far only
            for the Laue geometry). Must be set to 'True' in order to calculate
            the reflectivity of bent crystals.


        """
        super(Crystal, self).__init__(
            elements, quantities, rho=rho, table=table, name=name)
        self.hkl = hkl
        self.sqrthkl2 = (sum(i**2 for i in hkl))**0.5
        self.d = d
        if V is None:
            V = (d * self.sqrthkl2)**3
        self.V = V
        self.chiToF = -R0 / PI / self.V  # minus!
        self.geom = geom
        self.geometry = 2*int(geom.startswith('Bragg')) +\
            int(geom.endswith('transmitted'))
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.nuPoisson = nuPoisson
        self.calcBorrmann = calcBorrmann
        self.useTT = useTT

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

    def get_extinction_depth(self, E, polarization='s'):  # in microns
        theta0 = self.get_Bragg_angle(E)
        dw = self.get_Darwin_width(E, 1., polarization)
        return self.d / 2. / dw * np.tan(theta0) * 1e-4   # in microns

    def get_Borrmann_out(self, goodN, oeNormal, lb, a_out, b_out, c_out,
                         alphaAsym=None, Rcurvmm=None, ucl=None, useTT=False):

        asymmAngle = alphaAsym if alphaAsym is not None else 0

        if Rcurvmm is not None:
            Rcurv = Rcurvmm * 1e7
            if ucl is None:
                useTT = False
                print('OpenCL is required for bent crystals calculations.')
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
            F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, HH/4./PI)
            gamma_0h = beamInDotNormal * beamOutDotNormal

            if thickness == 0:
                N_layers = 10000
            else:
                N_layers = thickness / 200.
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

    def get_amplitude(self, E, beamInDotNormal, beamOutDotNormal=None,
                      beamInDotHNormal=None, alphaAsym=None,
                      Rcurvmm=None, ucl=None, useTT=False):
        r"""
        Calculates complex amplitude reflectivity and transmittivity for s- and
        p-polarizations (:math:`\gamma = s, p`) in Bragg and Laue cases for the
        crystal of thickness *L*, based upon Belyakov & Dmitrienko [BD]_:

        .. math::

            R_{\gamma}^{\rm Bragg} &= \chi_{\vec{H}}C_{\gamma}(\alpha +
            i\Delta_{\gamma}\cot{l_{\gamma}})^{-1}|b|^{-\frac{1}{2}}\\
            T_{\gamma}^{\rm Bragg} &= (\cos{l{_\gamma}} - i\alpha\Delta
            {_\gamma}^{-1}\sin{l_{\gamma}})^{-1}\exp{(i\vec{\kappa}_0^2 L
            (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1})}\\
            R_{\gamma}^{\rm Laue} &= \chi_{\vec{H}}C_{\gamma}
            \Delta_{\gamma}^{-1}\sin{l_{\gamma}}\exp{(i\vec{\kappa}_0^2 L
            (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1})}
            |b|^{-\frac{1}{2}}\\
            T_{\gamma}^{\rm Laue} &= (\cos{l_{\gamma}} + i\alpha
            \Delta_{\gamma}^{-1}\sin{l_{\gamma}})\exp{(i\vec{\kappa}_0^2 L
            (\chi_0 - \alpha b) (2\vec{\kappa}_0\vec{s})^{-1})}

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
            l = t * delta * k02 / 2 / kHs
            if self.geom.startswith('Bragg'):
                if self.geom.endswith('transmitted'):
                    ra = 1 / (np.cos(l) - 1j * alpha * np.sin(l) / delta) *\
                        np.exp(1j * k02 * t * (chi0 - alpha*b) / 2 / k0s)
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
                    Bm = np.sin(asymmAngle) *\
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
        H2 = (PI2 / self.d)**2
        b = k0s / kHs
        F0, Fhkl, Fhkl_, chi0, chih, chih_ = self.get_F_chi(E, HH/4./PI)
        thetaB = self.get_Bragg_angle(E)
        alpha = (H2/2 - k0H) / k02 + chi0/2 * (1/b - 1)

        if useTT:
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
            calc_model = for_one_polarization_TT
        else:
            calc_model = for_one_polarization
        curveS = calc_model(1.)  # s polarization
        polFactor = np.cos(2. * thetaB)
        curveP = calc_model(polFactor)  # p polarization
        return curveS, curveP  # , phi.real

    def get_Bragg_angle(self, E):
        a = CH / (2*self.d*E)
        try:
            a[a > 1] = 1 - 1e-16
        except TypeError:
            if a > 1:
                a = 1 - 1e-16
        return np.arcsin(a)

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


class CrystalFcc(Crystal):
    r"""
    A derivative class from :class:`Crystal` that defines the structure factor
    for an fcc crystal as:

    .. math::

        F_{hkl}^{fcc} = f \times \left\{ \begin{array}{rl}
        4 &\mbox{if $h,k,l$ are all even or all odd} \\ 0 &\mbox{ otherwise}
        \end{array} \right.

    """
    def get_structure_factor(self, E, sinThetaOverLambda):
        anomalousPart = self.elements[0].get_f1f2(E)
        F0 = 4 * (self.elements[0].Z+anomalousPart) * self.factDW
        residue = sum(i % 2 for i in self.hkl)
        if residue == 0 or residue == 3:
            f0 = self.elements[0].get_f0(sinThetaOverLambda)
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
    def get_structure_factor(self, E, sinThetaOverLambda):
        diamondToFcc = 1 + np.exp(0.5j * PI * sum(self.hkl))
        F0, Fhkl, Fhkl_ = super(CrystalDiamond, self).get_structure_factor(
            E, sinThetaOverLambda)
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
        self.tK = kwargs.pop('tK', 297.15)
        self.hkl = kwargs.get('hkl', (1, 1, 1))
# O'Mara, William C. Handbook of Semiconductor Silicon Technology.
# William Andrew Inc. (1990) pp. 349–352.
        self.a0 = 5.430710
        self.dl_l0 = self.dl_l(273.15 + 19.9)
        self.sqrthkl2 = (sum(i**2 for i in self.hkl))**0.5
        kwargs['d'] = self.get_a() / self.sqrthkl2
        kwargs['elements'] = 'Si'
        kwargs['hkl'] = self.hkl
# Mechanics of Materials, 23 (1996), p.314
        kwargs['nuPoisson'] = 0.22
        super(CrystalSi, self).__init__(*args, **kwargs)

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
    xraylib.

    Example:
        >>> xtal = rm.CrystalFromCell(
        >>>     'alphaQuartz', (1, 0, 2), a=4.91304, c=5.40463, gamma=120,
        >>>     atoms=[14, 14, 14, 8, 8, 8, 8, 8, 8],
        >>>     atomsXYZ=[[0.4697, 0., 0.],
        >>>               [-0.4697, -0.4697, 1./3],
        >>>               [0., 0.4697, 2./3],
        >>>               [0.4125, 0.2662, 0.1188],
        >>>               [-0.1463, -0.4125, 0.4521],
        >>>               [-0.2662, 0.1463, -0.2145],
        >>>               [0.1463, -0.2662, -0.1188],
        >>>               [-0.4125, -0.1463, 0.2145],
        >>>               [0.2662, 0.4125, 0.5479]])

    """
    def __init__(self, name='', hkl=[1, 1, 1],
                 a=5.419490, b=None, c=None, alpha=90, beta=90, gamma=90,
                 atoms=[14]*8,
                 atomsXYZ=[[0., 0., 0.],
                           [0., 0.5, 0.5],
                           [0.5, 0.5, 0.],
                           [0.5, 0., 0.5],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 atomsFraction=None, tK=297.15,
                 t=None, factDW=1.,
                 geom='Bragg reflected', table='Chantler',
                 nuPoisson=0., calcBorrmann=None, useTT=False):
        u"""
        *name*: str
            Crystal name. Not used by xrt.

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

        *atomsFraction*: a list of float or None
            Atomic fractions. If None, all values are 1.

        *nuPoisson*: float
            Poisson's ratio. Used to calculate the properties of bent crystals.

        *calcBorrmann*: str
            Controls the origin of the ray leaving the crystal. Can be 'None',
            'uniform', 'Bessel' or 'TT'. If 'None', the point of reflection
            is located on the surface of incidence. In all other cases the
            coordinate of the exit point is sampled according to the
            corresponding distribution: 'uniform' is a fast approximation for
            thick crystals, 'Bessel' is exact solution for the flat crystals,
            'TT' is exact solution of Takagi-Taupin equations for bent and flat
            crystals ('TT' requires *targetOpenCL* in the Optical Element to be
            not 'None' and *useTT* in the :class:`Crystal` to be 'True'. Not
            recommended for crystals thicker than 100 µm due to heavy
            computational load).

        *useTT*: bool
            Specifies whether the reflectivity will by calculated by analytical
            formula or by solution of the Takagi-Taupin equations (so far only
            for the Laue geometry). Must be set to 'True' in order to calculate
            the reflectivity of bent crystals.


        """
        self.name = name
        self.hkl = hkl
        h, k, l = hkl
        self.tK = tK
        self.a = a
        self.b = a if b is None else b
        self.c = a if c is None else c
        self.alpha = np.radians(alpha)
        self.beta = np.radians(beta)
        self.gamma = np.radians(gamma)
        self.atoms = atoms
        self.elements = []
        self.atomsXYZ = atomsXYZ
        uniqueElements = {}
        for atom in atoms:
            if atom in uniqueElements:
                element = uniqueElements[atom]
            else:
                element = Element(atom, table)
                uniqueElements[atom] = element
            self.elements.append(element)

        self.atomsFraction =\
            [1 for atom in atoms] if atomsFraction is None else atomsFraction
        self.quantities = self.atomsFraction
        ca, cb, cg = np.cos((self.alpha, self.beta, self.gamma))
        sa, sb, sg = np.sin((self.alpha, self.beta, self.gamma))
        self.V = self.a * self.b * self.c *\
            (1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)**0.5
        self.mass = 0.
        for atom, xi in zip(atoms, self.atomsFraction):
            self.mass += xi * element.mass
        self.rho = self.mass / AVOGADRO / self.V * 1e24
        self.d = self.V / (self.a * self.b * self.c) *\
            ((h*sa/self.a)**2 + (k*sb/self.b)**2 + (l*sg/self.c)**2 +
             2*h*k * (ca*cb - cg) / (self.a*self.b) +
             2*h*l * (ca*cg - cb) / (self.a*self.c) +
             2*k*l * (cb*cg - ca) / (self.b*self.c))**(-0.5)
        self.chiToF = -R0 / PI / self.V  # minus!
        self.geom = geom
        self.geometry = 2*int(geom.startswith('Bragg')) +\
            int(geom.endswith('transmitted'))
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.nuPoisson = nuPoisson
        self.calcBorrmann = calcBorrmann
        self.useTT = useTT

    def get_structure_factor(self, E, sinThetaOverLambda):
        F0, Fhkl, Fhkl_ = 0, 0, 0
        uniqueElements = {}
        for el, xyz, af in zip(
                self.elements, self.atomsXYZ, self.atomsFraction):
            if el.Z in uniqueElements:
                f0, anomalousPart = uniqueElements[el.Z]
            else:
                f0 = el.get_f0(sinThetaOverLambda)
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

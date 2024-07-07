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
   :members: __init__, get_t_thickness, get_b_thickness, get_amplitude,
             get_dtheta_symmetric_Bragg
.. autoclass:: Coated()
   :members: __init__

.. autoclass:: Crystal(Material)
   :members: __init__, get_Darwin_width, get_amplitude, get_amplitude_pytte,
             get_dtheta_symmetric_Bragg, get_dtheta, get_dtheta_regular,
             get_refractive_correction

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
.. autoclass:: MonoCrystal(CrystalFromCell)
   :members: __init__

.. _predefmats:

Predefined Materials
--------------------

.. tabs::

   .. tab:: Crystals

      .. automodule:: xrt.backends.raycing.materials_crystals

   .. tab:: Compounds

      .. automodule:: xrt.backends.raycing.materials_compounds

   .. tab:: Elemental

      .. automodule:: xrt.backends.raycing.materials_elemental

"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "6 Jul 2023"
__all__ = ('Material', 'EmptyMaterial', 'Multilayer', 'GradedMultilayer',
           'Coated', 'Crystal', 'CrystalFcc',
           'CrystalDiamond', 'CrystalSi', 'CrystalFromCell',
           'Powder', 'CrystalHarmonics', 'MonoCrystal')
import collections
__allSectioned__ = collections.OrderedDict([
    ('Material', None),
    ('Crystals', ('CrystalSi', 'CrystalDiamond', 'CrystalFcc',
                  'CrystalFromCell')),  # don't include 'Crystal'
    ('Layered', ('Coated', 'Multilayer', 'GradedMultilayer')),
    ('Advanced', ('Powder', 'CrystalHarmonics', 'MonoCrystal',
                  'EmptyMaterial'))
    ])
import sys
import os
import time
# import struct
import pickle
import numpy as np
# from scipy.special import jn as besselJn
from scipy.interpolate import interp1d
from .. import raycing
from . import myopencl as mcl
from .physconsts import PI, PI2, CH, CHBAR, R0, AVOGADRO, SQRT2PI

# try:
#     import pyopencl as cl  # analysis:ignore
#     isOpenCL = True
# except ImportError:
#     isOpenCL = False

if mcl.isOpenCL or mcl.isZMQ:
    isOpenCL = True
else:
    isOpenCL = False

ch = CH  # left here for copatibility
chbar = CHBAR  # left here for copatibility
spl_kw = {'kind': 'cubic', 'bounds_error': False, 'fill_value': 'extrapolate'}

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

try:
    from .pyTTE_x import TTcrystal, Quantity
    isPyTTE = True
#    print("Importing pyTTE")
except ImportError:
    isPyTTE = False
#    print("pyTTE not found")

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
            raise NameError('Wrong chemical element')
        self.f0coeffs = self.read_f0_Kissel()
        self.E, self.f1, self.f2 = self.read_f1f2_vs_E(table=table)
        self.table = table
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
                    E if isinstance(E, (int, float)) else
                    E[np.where((E < self.E[0]) | (E > self.E[-1]))], self.E[0],
                    self.E[-1]))
        f1 = np.interp(E, self.E, self.f1)
        f2 = np.interp(E, self.E, self.f2)
        return f1 + 1j*f2


class Material(object):
    """
    :class:`Material` serves for getting reflectivity, transmittivity,
    refractive index and absorption coefficient of a material specified by its
    chemical formula and density. See also predefined materials in modules
    :mod:`~xrt.backends.raycing.materials_compounds` and
    :mod:`~xrt.backends.raycing.materials_elemental`.


    """

    def __init__(self, elements=None, quantities=None, kind='auto', rho=0,
                 t=None, table='Chantler total', efficiency=None,
                 efficiencyFile=None, name='', refractiveIndex=None):
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
            Density in g/cm³.

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

        *refractiveIndex*: float or complex or numpy array or str
            Material refractive index is calculated from the tabulated
            scattering factors by :meth:`get_refractive_index`. If the target
            energy range is not covered by the tables of scattering factors
            (e.g. at IR or visible energies), refractive index can be
            externally defined by *refractiveIndex* as:
            a) float or complex value, for a constant, energy-independent
            refractive index.
            b) a 3-column numpy array containing Energy in eV, real and
            imaginary parts of the complex refractive index
            c) filename for an \*.xls or CSV table containig same columns as
            a numpy array in b).


        """
        if isinstance(elements, basestring):
            elements = elements,
        if quantities is None:
            self.quantities = [1. for elem in elements]
        else:
            self.quantities = quantities
        self.elements = []
        self.mass = 0.
        self.refractiveIndex = refractiveIndex
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
        else:
            self.t = t
        self.rho = rho  # density g/cm^3
        self.geom = ''
        self.efficiency = efficiency
        self.efficiencyFile = efficiencyFile
        if efficiencyFile is not None:
            self.read_efficiency_file()

    @property
    def refractiveIndex(self):
        return self._refractiveIndexVal

    @refractiveIndex.setter
    def refractiveIndex(self, refractiveIndex):
        self._refractiveIndex = refractiveIndex
        fname = None
        if refractiveIndex is not None:
            if isinstance(refractiveIndex, (float, complex)):
                self._refractiveIndexVal = complex(refractiveIndex)
            elif isinstance(refractiveIndex, np.ndarray):
                if len(refractiveIndex.shape) > 1:
                    if refractiveIndex.shape[1] > 2:
                        En = refractiveIndex[:, 0]
                        n = refractiveIndex[:, 1]
                        k = refractiveIndex[:, 2]
                        rIndex = np.complex128(n + 1j*k)
                        self._refractiveIndexVal = [
                            En, interp1d(En, rIndex, **spl_kw)]
            else:
                fname = refractiveIndex
                try:
                    self._refractiveIndexVal = self.read_ri_file(fname)
                except Exception as e:
                    print(e)
                    self._refractiveIndexVal = None
        else:
            self._refractiveIndexVal = None

    def read_ri_file(self, fname):
        # dataDir = os.path.dirname(__file__)
        # dataFile = os.path.join(dataDir, 'data', fname)
        dataPath = os.path.abspath(fname)
        if os.path.exists(dataPath):
            En = None
            if fname.endswith('.xls') or fname.endswith('.xlsx'):
                from pandas import read_excel
                data = read_excel(dataPath).values
                if len(data.shape) > 1:
                    if data.shape[1] > 2:
                        En = data[:, 0]
                        Ek = En
                        n = data[:, 1]
                        k = data[:, 2]
            else:
                # data = np.loadtxt(dataFile, skiprows=1, delimiter=',')
                En = []
                Ek = []
                n = []
                k = []
                # Reads specific format with sparse k column
                with open(dataPath) as f:
                    for li in f:
                        fields = li.split(',')
                        # if fields[0].strip('\"').startswith('Photon'):
                        #   continue
                        try:
                            tmpv = float(fields[0])
                        except ValueError:
                            continue
                        if len(fields) < 3:
                            En.append(float(fields[0]))
                            n.append(float(fields[-1]))
                        else:
                            Ek.append(float(fields[0]))
                            k.append(float(fields[-1]))
                            if len(fields[1].strip()) > 0:
                                En.append(float(fields[0]))
                                n.append(float(fields[1]))

            if En is not None:
                Ek = np.array(Ek)
                En = np.array(En)
                k = np.array(k)
                k = interp1d(Ek, k, **spl_kw)(En)
                rIndex = np.array(n) + 1j*k

                return [En, interp1d(En, rIndex, **spl_kw)]
        else:
            print(dataPath, "not found! Using refractive index of 1")
            return complex(1.)

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
        if self.refractiveIndex is not None:
            if isinstance(self.refractiveIndex, (tuple, list)):
                if np.min(E) > self.refractiveIndex[0][0] and\
                        np.max(E) < self.refractiveIndex[0][-1]:
                    return self.refractiveIndex[1](E)
                else:
                    print("Cannot calculate refractive index. "
                          "Energy outside of the range. "
                          "Using atomic scattering factors")
            elif isinstance(self.refractiveIndex, complex):
                return self.refractiveIndex
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

            \mu = 2 \Im(n) k.
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
# in case `assign_auto_material_kind hasn't happened before, which can be
# e.g. in calculator where materials are used without oes:
        if self.kind == 'auto':
            self.kind = 'mirror'  # used to be the default kind before xrtQook

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
                 idThickness=0., power=2., substRoughness=0,
                 substThickness=np.inf, name='', geom='reflected'):
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
        self.geom = geom
        if not self.geom:
            self.geom = 'reflected'
        self.idThickness = idThickness
        self.subRough = substRoughness
        self.substThickness = substThickness
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

        rmsbs = id2 if self.tLayer else self.subRough**2
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

    hiddenParams = ['substRoughness']


class Coated(Multilayer):
    """
    Derivative class from :class:`Mutilayer` with a single reflective layer on
    a substrate.
    """

    hiddenParams = ['tLayer', 'tThickness', 'bLayer', 'bThickness', 'power',
                    'tThicknessLow', 'bThicknessLow', 'idThickness',
                    'thicknessError', 'nPairs']

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


class Crystal(Material):
    u"""The parent class for crystals. The descendants must define
    :meth:`get_structure_factor`. :class:`Crystal` gives reflectivity and
    transmittivity of a crystal in Bragg and Laue cases."""

#    hiddenParams = ['nuPoisson', 'calcBorrmann']

    def __init__(self, hkl=[1, 1, 1], d=0, V=None, elements='Si',
                 quantities=None, rho=0, t=None, factDW=1.,
                 geom='Bragg reflected', table='Chantler total', name='',
                 volumetricDiffraction=False, useTT=False, nu=None,
                 mosaicity=0):
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
            :mod:`~xrt.backends.raycing.materials_crystals`. If
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
        super().__init__(elements, quantities, rho=rho, table=table, name=name)
        self.hkl = hkl
        self.sqrthkl2 = (sum(i**2 for i in hkl))**0.5
        self.d = d
        if V is None:
            V = (d * self.sqrthkl2)**3
        self.V = V
        self.chiToF = -R0 / PI / self.V  # minus!
        self.chiToFd2 = abs(self.chiToF) * self.d**2
        if len(geom) < 6:
            geom = geom.strip()+" reflected"
            # print(geom)
        self.geom = geom
        self.geometry = 2*int(geom.startswith('Bragg')) +\
            int(geom.endswith('transmitted'))
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.volumetricDiffraction = volumetricDiffraction
        self.useTT = useTT
        self.nu = nu
        self.mosaicity = mosaicity

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

    def set_OE_properties(self, alpha=0, Rm=None, Rs=None):
        Rmum = Rm*1e3 if Rm not in [np.inf, None] else np.inf  # [um] Meridional
        Rsum = Rs*1e3 if Rs not in [np.inf, None] else np.inf  # [um] Sagittal
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

        if hasattr(self, 'nu'):
            if self.nu is not None:  # Using isotropic model
                ttcrystal_kwargs['nu'] = self.nu
                ttcrystal_kwargs['E'] = Quantity(1, 'Pa')

        ttx = TTcrystal(**ttcrystal_kwargs)
        self.djparams = ttx.djparams

    def get_amplitude_pytte(
            self, E, beamInDotNormal, beamOutDotNormal=None,
            beamInDotHNormal=None, xd=None, yd=None, alphaAsym=None,
            Ry=None, Rx=None, ucl=None, tolerance=1e-6, maxSteps=1e7,
            autoLimits=True, signal=None):
        r"""
        Calculates complex amplitude reflectivity for s- and
        p-polarizations (:math:`\gamma = s, p`) in Bragg and Laue cases, based
        on modified `PyTTE code <https://github.com/aripekka/pyTTE>`_

        *alphaAsymm*: float
            Angle of asymmetry in radians.

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
        self.set_OE_properties(alphaAsym, Ry, Rx)

        geotag = 0 if self.geom.startswith('B') else np.pi*0.5
        alphaAsym = 0 if alphaAsym is None else alphaAsym+geotag
        Ryum = Ry*1e3 if Ry not in [np.inf, None] else np.inf  # [um] Meridional
        Rxum = Rx*1e3 if Rx not in [np.inf, None] else np.inf  # [um] Sagittal

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
            b_const_term = -0.5*k_bragg*(1 + gamma_term)*np.real(chi0)  # gamma0/gammah in pytte notation
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
        g0 = -0.5e4*k*chi0/beamInDotNormal  # passed into both kernels but used only by Laue
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
        self.alpha = self.beta = self.gamma = np.pi*0.5

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
        if 'name' not in kwargs:
            kwargs['name'] = 'Si'
# Mechanics of Materials, 23 (1996), p.314
#        kwargs['nuPoisson'] = 0.22
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
    :mod:`~xrt.backends.raycing.materials_crystals`.

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
                 nuPoisson=0., calcBorrmann=None, useTT=False, mosaicity=0):
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

            .. note::

                *atoms* and *atomsXYZ* must contain *all* the atoms, not only
                the unique ones for a given symmetry group (we do not consider
                symmetry here). For example, the unit cell of magnetite (Fe3O4)
                has 3 unique atomic positions and 56 in total; here, all 56 are
                needed.

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
        h, k, l = hkl  # analysis:ignore
        self.tK = 0
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
        self.chiToFd2 = abs(self.chiToF) * self.d**2
        self.geom = geom
        self.geometry = 2*int(geom.startswith('Bragg')) +\
            int(geom.endswith('transmitted'))
        self.factDW = factDW
        self.kind = 'crystal'
        self.t = t  # in mm
        self.nuPoisson = nuPoisson
        self.calcBorrmann = calcBorrmann
        self.useTT = useTT
        self.mosaicity = mosaicity

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

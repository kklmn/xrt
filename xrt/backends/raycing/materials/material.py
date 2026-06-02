# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d

from ... import raycing
from ..physconsts import PI2, CH, CHBAR, R0, AVOGADRO
from .element import Element

spl_kw = {'kind': 'cubic', 'bounds_error': False, 'fill_value': 'extrapolate'}


class Material(object):
    """
    :class:`Material` serves for getting reflectivity, transmittivity,
    refractive index and absorption coefficient of a material specified by its
    chemical formula and density. See also predefined materials in modules
    :mod:`~xrt.backends.raycing.materials.compounds` and
    :mod:`~xrt.backends.raycing.materials.elemental`.


    """

    def __init__(self, elements=None, quantities=None, kind='auto', rho=0,
                 t=None, table='Chantler total', efficiency=None,
                 efficiencyFile=None, name='', refractiveIndex=None, **kwargs):
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

        self.name = name
        self.mass = 0.
        self.table = table
        self.elements = elements
        self.quantities = quantities

#        if isinstance(elements, basestring):
#            elements = elements,
#
#        if quantities is None:
#            self.quantities = [1. for elem in elements]
#        else:
#            self.quantities = quantities
#        self.elements = []
#        self.mass = 0.
        self.refractiveIndex = refractiveIndex

#        for elem, xi in zip(elements, self.quantities):
#            newElement = Element(elem, table)
#            self.elements.append(newElement)
#            self.mass += xi * newElement.mass
#            if autoName:
#                self.name += elem
#                if xi != 1:
#                    self.name += '$_{' + '{0}'.format(xi) + '}$'
        self.t = t
        self.kind = kind  # 'mirror', 'thin mirror', 'plate', 'lens'

        self.rho = rho  # density g/cm^3
        self.geom = ''
        self.efficiency = efficiency
        self.efficiencyFile = efficiencyFile
        if efficiencyFile is not None:
            self.read_efficiency_file()
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

        self.bl = kwargs.get('bl')

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        if kind == 'thin mirror' and hasattr(self, '_t'):
            if self.t is None:
                print('Cannot change type for None thickness')
#                raise ValueError('Give the thin mirror a thickness!')
            else:
                self._kind = kind
        else:
            self._kind = kind

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        if hasattr(self, '_kind') and self.kind == 'thin mirror' and t is None:
            print('Cannot assign None thickness to thin mirror')
#            raise ValueError('Give the thin mirror a thickness!')
        else:
            self._t = t

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name:
            self._name = name
            self.autoName = False
        else:
            self._name = r''
            self.autoName = True

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements):
        self._elements = []
        if isinstance(elements, raycing.basestring):
            elements = elements,
        if hasattr(self, 'table'):
            for elem in elements:
                newElement = Element(elem, self.table)
                self._elements.append(newElement)
        self.set_mass()

    @property
    def quantities(self):
        return self._quantities

    @quantities.setter
    def quantities(self, quantities):
        if quantities is None:
            if hasattr(self, '_elements'):
                self._quantities = [1. for elem in self.elements]
            else:
                self._quantities = [1]
        elif not isinstance(quantities, (list, tuple)):
            self._quantities = [quantities]
        else:
            self._quantities = quantities
        self.set_mass()

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, table):
        self._table = table
        self.set_mass()

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

    def set_mass(self):
        self.mass = 0.
        if self.autoName:
            self.name = ''
        if not all([hasattr(self, v) for v in
                    ['_elements', '_quantities']]):
            return

        for elem, xi in zip(self.elements, self.quantities):
            self.mass += xi * elem.mass
            if self.autoName:
                self.name += elem.name
                if xi != 1:
                    self.name += '$_{' + '{0}'.format(xi) + '}$'

    def read_ri_file(self, fname):
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
                            _ = float(fields[0])
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

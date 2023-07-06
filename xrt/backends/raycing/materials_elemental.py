r"""
Predefined Materials: Elemental
-------------------------------

The module :mod:`~xrt.backends.raycing.materials_elemental` contains predefined
classes for elemental materials. Most atomic densities have been adopted from
the `NIST table of X-Ray Mass Attenuation Coefficients
<https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html>`_.
Densities for Fr and At given according to [Lavrukhina_Pozdnyakov]_.

.. [Lavrukhina_Pozdnyakov] Lavrukhina, A. K. and Pozdnyakov, A. A. (1970).
   Analytical Chemistry of Technetium, Promethium, Astatine, and Francium.
   Translated by R. Kondor. Ann Arbor–Humphrey Science Publishers.
   p. 269. ISBN 978-0-250-39923-9

.. note::
    Densities are given for the phase state at ambient conditions, e.g.
    Nitrogen as N\ :sub:`2` gas.

To use an elemental material in a script simply import the module and
instantiate its class:

.. code-block:: python

    import xrt.backends.raycing.materials_elemental as xmat
    nitrogenGas = xmat.N()

The elemental materials inherit from :class:`.Material` and can use its methods
to calculate reflection or transmission amplitudes, absorption coefficient,
refractive index etc.

.. note::
    The elemental materials do not provide crystal diffraction amplitudes even
    if they occur naturally as crystals. To calculate diffraction on crystals
    please use :mod:`~xrt.backends.raycing.materials_crystals`.

The following elemental classes are defined in this module:
 | |elemall|

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "6 Jul 2017"

from . import materials as rmat
import collections

__all__ = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U')

__allSectioned__ = collections.OrderedDict([
    ('H-Ne',
        ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne')),
    ('Na-Ar',
        ('Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar')),
    ('K-Kr',
        ('K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
         'Cu', 'Zn', 'Ga', 'Ge''As', 'Se', 'Br', 'Kr')),
    ('Rb-Xe',
        ('Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
         'Ag', 'Cd''In', 'Sn', 'Sb', 'Te', 'I', 'Xe')),
    ('Cs-Lu',
        ('Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
         'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu')),
    ('Hf-Bi',
        ('Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
         'Pb', 'Bi')),
    ('Po-U',
        ('Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'))])


class H(rmat.Material):
    def __init__(self, name='Hydrogen', elements='H', rho=8.375e-05,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class He(rmat.Material):
    def __init__(self, name='Helium', elements='He', rho=1.663e-04,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Li(rmat.Material):
    def __init__(self, name='Lithium', elements='Li', rho=5.340e-01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Be(rmat.Material):
    def __init__(self, name='Beryllium', elements='Be', rho=1.848e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class B(rmat.Material):
    def __init__(self, name='Boron', elements='B', rho=2.370e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class C(rmat.Material):
    def __init__(self, name='Carbon', elements='C', rho=1.700e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class N(rmat.Material):
    def __init__(self, name='Nitrogen', elements='N', rho=1.165e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class O(rmat.Material):
    def __init__(self, name='Oxygen', elements='O', rho=1.332e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class F(rmat.Material):
    def __init__(self, name='Fluorine', elements='F', rho=1.580e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ne(rmat.Material):
    def __init__(self, name='Neon', elements='Ne', rho=8.385e-04,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Na(rmat.Material):
    def __init__(self, name='Sodium', elements='Na', rho=9.710e-01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Mg(rmat.Material):
    def __init__(self, name='Magnesium', elements='Mg', rho=1.740e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Al(rmat.Material):
    def __init__(self, name='Aluminum', elements='Al', rho=2.699e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Si(rmat.Material):
    def __init__(self, name='Silicon', elements='Si', rho=2.330e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class P(rmat.Material):
    def __init__(self, name='Phosphorus', elements='P', rho=2.200e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class S(rmat.Material):
    def __init__(self, name='Sulfur', elements='S', rho=2.000e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Cl(rmat.Material):
    def __init__(self, name='Chlorine', elements='Cl', rho=2.995e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ar(rmat.Material):
    def __init__(self, name='Argon', elements='Ar', rho=1.662e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class K(rmat.Material):
    def __init__(self, name='Potassium', elements='K', rho=8.620e-01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ca(rmat.Material):
    def __init__(self, name='Calcium', elements='Ca', rho=1.550e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Sc(rmat.Material):
    def __init__(self, name='Scandium', elements='Sc', rho=2.989e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ti(rmat.Material):
    def __init__(self, name='Titanium', elements='Ti', rho=4.540e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class V(rmat.Material):
    def __init__(self, name='Vanadium', elements='V', rho=6.110e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Cr(rmat.Material):
    def __init__(self, name='Chromium', elements='Cr', rho=7.180e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Mn(rmat.Material):
    def __init__(self, name='Manganese', elements='Mn', rho=7.440e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Fe(rmat.Material):
    def __init__(self, name='Iron', elements='Fe', rho=7.874e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Co(rmat.Material):
    def __init__(self, name='Cobalt', elements='Co', rho=8.900e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ni(rmat.Material):
    def __init__(self, name='Nickel', elements='Ni', rho=8.902e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Cu(rmat.Material):
    def __init__(self, name='Copper', elements='Cu', rho=8.960e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Zn(rmat.Material):
    def __init__(self, name='Zinc', elements='Zn', rho=7.133e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ga(rmat.Material):
    def __init__(self, name='Gallium', elements='Ga', rho=5.904e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ge(rmat.Material):
    def __init__(self, name='Germanium', elements='Ge', rho=5.323e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class As(rmat.Material):
    def __init__(self, name='Arsenic', elements='As', rho=5.730e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Se(rmat.Material):
    def __init__(self, name='Selenium', elements='Se', rho=4.500e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Br(rmat.Material):
    def __init__(self, name='Bromine', elements='Br', rho=7.072e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Kr(rmat.Material):
    def __init__(self, name='Krypton', elements='Kr', rho=3.478e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Rb(rmat.Material):
    def __init__(self, name='Rubidium', elements='Rb', rho=1.532e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Sr(rmat.Material):
    def __init__(self, name='Strontium', elements='Sr', rho=2.540e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Y(rmat.Material):
    def __init__(self, name='Yttrium', elements='Y', rho=4.469e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Zr(rmat.Material):
    def __init__(self, name='Zirconium', elements='Zr', rho=6.506e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Nb(rmat.Material):
    def __init__(self, name='Niobium', elements='Nb', rho=8.570e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Mo(rmat.Material):
    def __init__(self, name='Molybdenum', elements='Mo', rho=1.022e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Tc(rmat.Material):
    def __init__(self, name='Technetium', elements='Tc', rho=1.150e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ru(rmat.Material):
    def __init__(self, name='Ruthenium', elements='Ru', rho=1.241e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Rh(rmat.Material):
    def __init__(self, name='Rhodium', elements='Rh', rho=1.241e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pd(rmat.Material):
    def __init__(self, name='Palladium', elements='Pd', rho=1.202e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ag(rmat.Material):
    def __init__(self, name='Silver', elements='Ag', rho=1.050e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Cd(rmat.Material):
    def __init__(self, name='Cadmium', elements='Cd', rho=8.650e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class In(rmat.Material):
    def __init__(self, name='Indium', elements='In', rho=7.310e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Sn(rmat.Material):
    def __init__(self, name='Tin', elements='Sn', rho=7.310e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Sb(rmat.Material):
    def __init__(self, name='Antimony', elements='Sb', rho=6.691e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Te(rmat.Material):
    def __init__(self, name='Tellurium', elements='Te', rho=6.240e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class I(rmat.Material):
    def __init__(self, name='Iodine', elements='I', rho=4.930e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Xe(rmat.Material):
    def __init__(self, name='Xenon', elements='Xe', rho=5.485e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Cs(rmat.Material):
    def __init__(self, name='Cesium', elements='Cs', rho=1.873e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ba(rmat.Material):
    def __init__(self, name='Barium', elements='Ba', rho=3.500e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class La(rmat.Material):
    def __init__(self, name='Lanthanum', elements='La', rho=6.154e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ce(rmat.Material):
    def __init__(self, name='Cerium', elements='Ce', rho=6.657e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pr(rmat.Material):
    def __init__(self, name='Praseodymium', elements='Pr', rho=6.710e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Nd(rmat.Material):
    def __init__(self, name='Neodymium', elements='Nd', rho=6.900e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pm(rmat.Material):
    def __init__(self, name='Promethium', elements='Pm', rho=7.220e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Sm(rmat.Material):
    def __init__(self, name='Samarium', elements='Sm', rho=7.460e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Eu(rmat.Material):
    def __init__(self, name='Europium', elements='Eu', rho=5.243e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Gd(rmat.Material):
    def __init__(self, name='Gadolinium', elements='Gd', rho=7.900e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Tb(rmat.Material):
    def __init__(self, name='Terbium', elements='Tb', rho=8.229e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Dy(rmat.Material):
    def __init__(self, name='Dysprosium', elements='Dy', rho=8.550e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ho(rmat.Material):
    def __init__(self, name='Holmium', elements='Ho', rho=8.795e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Er(rmat.Material):
    def __init__(self, name='Erbium', elements='Er', rho=9.066e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Tm(rmat.Material):
    def __init__(self, name='Thulium', elements='Tm', rho=9.321e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Yb(rmat.Material):
    def __init__(self, name='Ytterbium', elements='Yb', rho=6.730e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Lu(rmat.Material):
    def __init__(self, name='Lutetium', elements='Lu', rho=9.840e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Hf(rmat.Material):
    def __init__(self, name='Hafnium', elements='Hf', rho=1.331e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ta(rmat.Material):
    def __init__(self, name='Tantalum', elements='Ta', rho=1.665e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class W(rmat.Material):
    def __init__(self, name='Tungsten', elements='W', rho=1.930e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Re(rmat.Material):
    def __init__(self, name='Rhenium', elements='Re', rho=2.102e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Os(rmat.Material):
    def __init__(self, name='Osmium', elements='Os', rho=2.257e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ir(rmat.Material):
    def __init__(self, name='Iridium', elements='Ir', rho=2.242e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pt(rmat.Material):
    def __init__(self, name='Platinum', elements='Pt', rho=2.145e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Au(rmat.Material):
    def __init__(self, name='Gold', elements='Au', rho=1.932e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Hg(rmat.Material):
    def __init__(self, name='Mercury', elements='Hg', rho=1.355e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Tl(rmat.Material):
    def __init__(self, name='Thallium', elements='Tl', rho=1.172e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pb(rmat.Material):
    def __init__(self, name='Lead', elements='Pb', rho=1.135e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Bi(rmat.Material):
    def __init__(self, name='Bismuth', elements='Bi', rho=9.747e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Po(rmat.Material):
    def __init__(self, name='Polonium', elements='Po', rho=9.320e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class At(rmat.Material):
    def __init__(self, name='Astatine', elements='At', rho=8.91e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Rn(rmat.Material):
    def __init__(self, name='Radon', elements='Rn', rho=9.066e-03,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Fr(rmat.Material):
    def __init__(self, name='Francium', elements='Fr', rho=2.48e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ra(rmat.Material):
    def __init__(self, name='Radium', elements='Ra', rho=5.000e+00,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Ac(rmat.Material):
    def __init__(self, name='Actinium', elements='Ac', rho=1.007e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Th(rmat.Material):
    def __init__(self, name='Thorium', elements='Th', rho=1.172e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class Pa(rmat.Material):
    def __init__(self, name='Protactinium', elements='Pa', rho=1.537e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)


class U(rmat.Material):
    def __init__(self, name='Uranium', elements='U', rho=1.895e+01,
                 *args, **kwargs):
        super().__init__(name=name, rho=rho, elements=elements,
                         *args, **kwargs)



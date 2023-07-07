r"""
The module :mod:`~xrt.backends.raycing.materials_crystals` contains predefined
classes for most commonly used crystals. Lattice parameters, atomic positions
and references have been semi-automatically parsed from XOP/DABAX [XOP]_
``Crystals.dat``. To use a crystal in a script simply import the module and
instantiate its class:

.. code-block:: python

    import xrt.backends.raycing.materials_crystals as xcryst
    myInSbXtal = xcryst.InSb(hkl=(3, 1, 1))  # default hkl=(1, 1, 1)

The crystals inherit from :class:`.Crystal` and can use its methods to
calculate diffraction amplitudes, the Darwin width, extinction depth etc.

The following crystal classes are defined in this module (sorted by cell
volume in Å³), marked in bold are those with available elastic constants:

   | |xtalall|

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "6 Jul 2017"

from . import materials as rmat
import collections

__all__ = ('Si', 'Ge', 'Diamond', 'GaAs', 'GaSb', 'GaP',
           'InAs', 'InP', 'InSb', 'SiC', 'NaCl', 'CsF', 'LiF', 'KCl', 'CsCl',
           'Be', 'Graphite', 'PET', 'Beryl', 'KAP', 'RbAP', 'TlAP',
           'Muscovite', 'AlphaQuartz', 'Copper', 'LiNbO3', 'Platinum',
           'Gold', 'Sapphire', 'LaB6', 'LaB6NIST', 'KTP', 'AlphaAlumina',
           'Aluminum', 'Iron', 'Titanium')

__allSectioned__ = collections.OrderedDict([
    ('Cubic',
        ('Si', 'Ge', 'Diamond', 'GaAs', 'GaSb', 'GaP', 'InAs', 'InP', 'InSb',
         'SiC', 'NaCl', 'CsF', 'LiF', 'KCl', 'CsCl', 'Copper', 'Platinum',
         'Gold', 'LaB6', 'Aluminum', 'Iron')),
    ('Hexagonal',
        ('Be', 'Graphite', 'Beryl', 'AlphaQuartz', 'Sapphire', 'Titanium')),
    ('Tetragonal',
        ('PET',)),
    ('Orthorhombic',
        ('KAP', 'RbAP', 'TlAP', 'KTP')),
    ('Monoclinic',
        ('Muscovite',)),
    ('Trigonal',
        ('LiNbO3', 'AlphaAlumina'))])


class Si(rmat.CrystalDiamond):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.26
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...
       Warning: This definition is different from xrt materials.CrystalSi


       '''

    def __init__(self, name='Si', elements='Si', a=5.4307,
                 *args, **kwargs):
        super().__init__(a=a, name=name, elements=elements, *args, **kwargs)


class SiNIST(rmat.CrystalDiamond):
    '''System: Cubic
       Structure: ZincBlende
       NIST Standard Reference Material 640c (Silicon Powder)
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='Si_NIST', elements='Si', a=5.4311946,
                 *args, **kwargs):
        super().__init__(a=a, name=name, elements=elements, *args, **kwargs)


class Si2(rmat.CrystalDiamond):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.26
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='Si2', elements='Si', a=5.4307,
                 *args, **kwargs):
        super().__init__(a=a, name=name, elements=elements, *args, **kwargs)


class Ge(rmat.CrystalDiamond):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.26
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='Ge', elements='Ge', a=5.65735,
                 *args, **kwargs):
        super().__init__(a=a, name=name, elements=elements, *args, **kwargs)


class Diamond(rmat.CrystalDiamond):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.26
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='Diamond', elements='C', a=3.56679,
                 *args, **kwargs):
        super().__init__(a=a, name=name, elements=elements, *args, **kwargs)


class GaAs(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.110
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='GaAs',
                 atoms=[31, 31, 31, 31, 33, 33, 33, 33],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=5.6537, b=5.6537, c=5.6537,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class GaSb(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       http://www.ioffe.rssi.ru/SVA/NSM/Semicond/GaSb/basic.html
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='GaSb',
                 atoms=[31, 31, 31, 31, 51, 51, 51, 51],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=6.09593, b=6.09593, c=6.09593,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class GaP(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.110
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='GaP',
                 atoms=[31, 31, 31, 31, 15, 15, 15, 15],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=5.4505, b=5.4505, c=5.4505,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class InAs(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.110
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='InAs',
                 atoms=[49, 49, 49, 49, 33, 33, 33, 33],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=6.036, b=6.036, c=6.036,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class InP(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.110
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='InP',
                 atoms=[49, 49, 49, 49, 15, 15, 15, 15],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=5.8687, b=5.8687, c=5.8687,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class InSb(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.110
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='InSb',
                 atoms=[49, 49, 49, 49, 51, 51, 51, 51],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=6.4782, b=6.4782, c=6.4782,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class SiC(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: ZincBlende
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1
       The ZincBlende structure is defined by atom A located at (0 0 0)
       and atom B at (1/4 1/4 1/4) of the fcc lattice. Ex: Si(a=5.4309)
       Ge(a=5.657)  Diamond(a=3.56)  GaAs  GaP  InAs InP  InSb  SiC...


       '''

    def __init__(self, name='SiC',
                 atoms=[14, 14, 14, 14, 6, 6, 6, 6],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.25, 0.25, 0.25],
                           [0.25, 0.75, 0.75],
                           [0.75, 0.25, 0.75],
                           [0.75, 0.75, 0.25]],
                 a=4.348, b=4.348, c=4.348,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class NaCl(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: NaCl
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.88
       The RockSalt structure is defined by atom A located at
       (0 0 0) and atom B at (1/2 1/2 1/2) of the fcc lattice.
       Examples: NaCl CsF LiF KCl...


       '''

    def __init__(self, name='NaCl',
                 atoms=[11, 11, 11, 11, 17, 17, 17, 17],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.5, 0.5],
                           [0.5, 0.0, 0.0],
                           [0.0, 0.5, 0.0],
                           [0.0, 0.0, 0.5]],
                 a=5.63978, b=5.63978, c=5.63978,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class CsF(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: NaCl
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.84
       The RockSalt structure is defined by atom A located at
       (0 0 0) and atom B at (1/2 1/2 1/2) of the fcc lattice.
       Examples: NaCl CsF LiF KCl...


       '''

    def __init__(self, name='CsF',
                 atoms=[55, 55, 55, 55, 9, 9, 9, 9],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.5, 0.5],
                           [0.5, 0.0, 0.0],
                           [0.0, 0.5, 0.0],
                           [0.0, 0.0, 0.5]],
                 a=6.008, b=6.008, c=6.008,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class LiF(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: NaCl
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.86
       The RockSalt structure is defined by atom A located at
       (0 0 0) and atom B at (1/2 1/2 1/2) of the fcc lattice.
       Examples: NaCl CsF LiF KCl...


       '''

    def __init__(self, name='LiF',
                 atoms=[3, 3, 3, 3, 9, 9, 9, 9],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.5, 0.5],
                           [0.5, 0.0, 0.0],
                           [0.0, 0.5, 0.0],
                           [0.0, 0.0, 0.5]],
                 a=4.0263, b=4.0263, c=4.0263,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class KCl(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: NaCl
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.86
       The RockSalt structure is defined by atom A located at
       (0 0 0) and atom B at (1/2 1/2 1/2) of the fcc lattice.
       Examples: NaCl CsF LiF KCl...


       '''

    def __init__(self, name='KCl',
                 atoms=[19, 19, 19, 19, 17, 17, 17, 17],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.5, 0.5],
                           [0.5, 0.0, 0.0],
                           [0.0, 0.5, 0.0],
                           [0.0, 0.0, 0.5]],
                 a=6.29294, b=6.29294, c=6.29294,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class CsCl(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: CsCl
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.86
       The CsCl structure is defined by an atom A located at
       (0 0 0) and an atom B at (1/2 1/2 1/2) of the cubic lattice
       If A = B  then it is a bcc lattice


       '''

    def __init__(self, name='CsCl',
                 atoms=[55, 17],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5]],
                 a=7.02, b=7.02, c=7.02,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Be(rmat.CrystalFromCell):
    '''System: Hexagonal
       Structure: Berilium
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1
       The Hexagonal Closed-packed structure is defined by an atom A
       located at (1/3 2/3 1/4) and an atom B at (2/3  1/3  3/4) in the
       prism cell. Example: Berilium (with a=2.287  c=3.583)


       '''

    def __init__(self, name='Be',
                 atoms=[4, 4],
                 atomsXYZ=[[0.333333, 0.666667, 0.25],
                           [0.666667, 0.333333, 0.75]],
                 a=2.2866, b=2.2866, c=3.5833,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Graphite(rmat.CrystalFromCell):
    '''System: Hexagonal
       Structure: Graphite
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.28
       The graphite structure contains four atom in unit prism in
       (0 0 0) (0 0 1/2) (1/3 2/3 0) and (2/3  1/3  1/2)
       Example: Graphite (with C atoms and a=2.456  c=6.696)


       '''

    def __init__(self, name='Graphite',
                 atoms=[6, 6, 6, 6],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.5],
                           [0.333333, 0.666667, 0.0],
                           [0.666667, 0.333333, 0.5]],
                 a=2.456, b=2.456, c=6.696,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class PET(rmat.CrystalFromCell):
    '''System: Tetragonal
       PET (Pentaerythritol) C(CH2OH)4
       Unit cell a: 6.10 A  c: 8.73 A
       See also:
       http://www.photonic.saint-gobain.com/Media/Documents/S0000000000000001020/%20XRay.pdf


       '''

    def __init__(self, name='PET',
                 atoms=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.99, 0.75, 0.85],
                           [-0.99, -0.75, 0.85],
                           [-0.75, 0.99, -0.85],
                           [0.75, -0.99, -0.85],
                           [0.5, 0.5, 0.5],
                           [1.49, 1.25, 1.35],
                           [-0.49, -0.25, 1.35],
                           [-0.25, 1.49, -0.35],
                           [1.25, -0.49, -0.35],
                           [1.93, 1.5, 0.017],
                           [-1.93, -1.5, 0.017],
                           [-1.5, 1.93, -0.017],
                           [1.5, -1.93, -0.017],
                           [2.43, 2.0, 0.517],
                           [-1.43, -1.0, 0.517],
                           [-1.0, 2.43, 0.483],
                           [2.0, -1.43, 0.483]],
                 a=6.1, b=6.1, c=8.73,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Beryl(rmat.CrystalFromCell):
    '''System: Hexagonal
       Al2 Be3 (SiO3)6
       Unit cell a: 9.088 A  c: 9.1896 A


       '''

    def __init__(self, name='Beryl',
                 atoms=[13, 13, 13, 13, 4, 4, 4, 4, 4, 4, 14, 14, 14, 14, 14,
                        14, 14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.6667, 0.3333, 0.25],
                           [0.3333, 0.6667, 0.25],
                           [0.6667, 0.3333, 0.75],
                           [0.3333, 0.6667, 0.75],
                           [0.5, 0.0, 0.25],
                           [0.0, 0.5, 0.25],
                           [0.5, 0.5, 0.25],
                           [0.5, 0.0, 0.75],
                           [0.0, 0.5, 0.75],
                           [0.5, 0.5, 0.75],
                           [0.3875, 0.1159, 0.0],
                           [-0.3875, -0.1159, 0.0],
                           [0.2716, 0.3875, 0.0],
                           [-0.2716, -0.3875, 0.0],
                           [0.1159, -0.2716, 0.0],
                           [-0.1159, 0.2716, 0.0],
                           [0.3875, 0.1159, 0.5],
                           [-0.3875, -0.1159, 0.5],
                           [0.2716, 0.3875, 0.5],
                           [-0.2716, -0.3875, 0.5],
                           [0.1159, -0.2716, 0.5],
                           [-0.1159, 0.2716, 0.5],
                           [0.31, 0.2366, 0.0],
                           [-0.31, -0.2366, 0.0],
                           [0.0734, 0.31, 0.0],
                           [-0.0734, -0.31, 0.0],
                           [0.2366, -0.0734, 0.0],
                           [-0.2366, 0.0734, 0.0],
                           [0.31, 0.2366, 0.5],
                           [-0.31, -0.2366, 0.5],
                           [0.0734, 0.31, 0.5],
                           [-0.0734, -0.31, 0.5],
                           [0.2366, -0.0734, 0.5],
                           [-0.2366, 0.0734, 0.5],
                           [0.4988, 0.1455, 0.1453],
                           [-0.4988, -0.1455, 0.1453],
                           [0.3533, 0.4988, 0.1453],
                           [-0.3533, -0.4988, 0.1453],
                           [-0.1455, 0.3533, 0.1453],
                           [0.1455, -0.3533, 0.1453],
                           [0.4988, 0.1455, 0.6453],
                           [-0.4988, -0.1455, 0.6453],
                           [0.3533, 0.4988, 0.6453],
                           [-0.3533, -0.4988, 0.6453],
                           [-0.1455, 0.3533, 0.6453],
                           [0.1455, -0.3533, 0.6453],
                           [0.4988, 0.3533, 0.3547],
                           [0.1455, 0.4988, 0.3547],
                           [-0.3533, 0.1455, 0.3547],
                           [0.3533, -0.1455, 0.3547],
                           [-0.4988, -0.3533, 0.3547],
                           [-0.1455, -0.4988, 0.3547],
                           [0.4988, 0.3533, 0.8547],
                           [0.1455, 0.4988, 0.8547],
                           [-0.3533, 0.1455, 0.8547],
                           [0.3533, -0.1455, 0.8547],
                           [-0.4988, -0.3533, 0.8547],
                           [-0.1455, -0.4988, 0.8547]],
                 a=9.088, b=9.088, c=9.1896,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class KAP(rmat.CrystalFromCell):
    '''System: Orthorhombic
       KAP Acide Phatalate K (HOOC C6H4 COO)
       (a=6.460  b=9.600  c=13.850)


       '''

    def __init__(self, name='KAP',
                 atoms=[19, 19, 19, 19, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.25, 0.09898, 0.03878],
                           [0.75, -0.09898, -0.03878],
                           [0.25, 0.59898, -0.03878],
                           [0.75, 0.40102, 0.03878],
                           [0.00561, -0.1778, 0.21761],
                           [-0.23842, 0.068887, 0.15685],
                           [-0.02519, -0.05921, 0.28937],
                           [0.06263, -0.06367, 0.38466],
                           [0.03824, 0.04731, 0.4505],
                           [-0.0735, 0.1628, 0.42058],
                           [-0.15805, 0.16928, 0.32513],
                           [-0.13588, 0.05799, 0.25841],
                           [0.50561, 0.1778, -0.21761],
                           [0.26158, -0.068887, -0.15685],
                           [0.47481, 0.05921, -0.28937],
                           [0.56263, 0.06367, -0.38466],
                           [0.53824, -0.04731, -0.4505],
                           [0.4265, -0.1628, -0.42058],
                           [0.34195, -0.16928, -0.32513],
                           [0.36412, -0.05799, -0.25841],
                           [0.00561, 0.3222, -0.21761],
                           [-0.23842, 0.568887, -0.15685],
                           [-0.02519, 0.44079, -0.28937],
                           [0.06263, 0.43633, -0.38466],
                           [0.03824, 0.54731, -0.4505],
                           [-0.0735, 0.6628, -0.42058],
                           [-0.15805, 0.66928, -0.32513],
                           [-0.13588, 0.55799, -0.25841],
                           [0.50561, 0.6778, 0.21761],
                           [0.26158, 0.431113, 0.15685],
                           [0.47481, 0.55921, 0.28937],
                           [0.56263, 0.56367, 0.38466],
                           [0.53824, 0.45269, 0.4505],
                           [0.4265, 0.3372, 0.42058],
                           [0.34195, 0.33072, 0.32513],
                           [0.36412, 0.44201, 0.25841],
                           [0.07871, -0.29831, 0.26258],
                           [0.01748, -0.16109, 0.12732],
                           [-0.15861, 0.14522, 0.09314],
                           [-0.40385, 6e-05, 0.14401],
                           [0.57871, 0.29831, -0.26258],
                           [0.51748, 0.16109, -0.12732],
                           [0.34139, -0.14522, -0.09314],
                           [0.09615, -6e-05, -0.14401],
                           [0.07871, 0.20169, -0.26258],
                           [0.01748, 0.33891, -0.12732],
                           [-0.15861, 0.64522, -0.09314],
                           [-0.40385, 0.50006, -0.14401],
                           [0.57871, 0.79831, 0.26258],
                           [0.51748, 0.66109, 0.12732],
                           [0.34139, 0.35478, 0.09314],
                           [0.09615, 0.49994, 0.14401]],
                 a=6.46, b=9.6, c=13.85,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class RbAP(rmat.CrystalFromCell):
    '''System: Orthorhombic
       RbAP Acide Phatalate Rb (HOOC C6H4 COO)
       (a=6.561  b=10.064  c=13.068)


       '''

    def __init__(self, name='RbAP',
                 atoms=[37, 37, 37, 37, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[-0.25, 0.1014, 0.0459],
                           [0.25, -0.1014, -0.0459],
                           [-0.25, 0.6014, -0.0459],
                           [0.25, 0.3986, 0.0459],
                           [0.127, 0.051, 0.267],
                           [0.009, -0.054, 0.296],
                           [-0.088, -0.055, 0.392],
                           [-0.07, 0.053, 0.453],
                           [0.048, 0.164, 0.425],
                           [0.136, 0.161, 0.332],
                           [0.239, 0.06, 0.168],
                           [-0.011, -0.176, 0.226],
                           [0.627, -0.051, -0.267],
                           [0.509, 0.054, -0.296],
                           [0.412, 0.055, -0.392],
                           [0.43, -0.053, -0.453],
                           [0.548, -0.164, -0.425],
                           [0.636, -0.161, -0.332],
                           [0.739, -0.06, -0.168],
                           [0.489, 0.176, -0.226],
                           [0.127, 0.551, -0.267],
                           [0.009, 0.446, -0.296],
                           [-0.088, 0.445, -0.392],
                           [-0.07, 0.553, -0.453],
                           [0.048, 0.664, -0.425],
                           [0.136, 0.661, -0.332],
                           [0.239, 0.56, -0.168],
                           [-0.011, 0.324, -0.226],
                           [0.627, 0.449, 0.267],
                           [0.509, 0.554, 0.296],
                           [0.412, 0.555, 0.392],
                           [0.43, 0.447, 0.453],
                           [0.548, 0.336, 0.425],
                           [0.636, 0.339, 0.332],
                           [0.739, 0.44, 0.168],
                           [0.489, 0.676, 0.226],
                           [0.184, 0.133, 0.1],
                           [0.397, -0.014, 0.157],
                           [-0.013, -0.163, 0.134],
                           [-0.039, -0.287, 0.275],
                           [0.684, -0.133, -0.1],
                           [0.897, 0.014, -0.157],
                           [0.487, 0.163, -0.134],
                           [0.461, 0.287, -0.275],
                           [0.184, 0.633, -0.1],
                           [0.397, 0.486, -0.157],
                           [-0.013, 0.337, -0.134],
                           [-0.039, 0.213, -0.275],
                           [0.684, 0.367, 0.1],
                           [0.897, 0.514, 0.157],
                           [0.487, 0.663, 0.134],
                           [0.461, 0.787, 0.275]],
                 a=6.561, b=10.064, c=13.068,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class TlAP(rmat.CrystalFromCell):
    '''System: Orthorhombic
       TlAP Acide Phatalate Tl (HOOC C6H4 COO)
       (a=6.615  b=10.047  c=12.878)


       '''

    def __init__(self, name='TlAP',
                 atoms=[81, 81, 81, 81, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                        6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.25, -0.10277, -0.0476],
                           [0.75, 0.10277, 0.0476],
                           [0.25, 0.39723, 0.0476],
                           [0.75, 0.60277, -0.0476],
                           [0.254, 0.059, 0.163],
                           [0.124, 0.052, 0.266],
                           [0.144, 0.163, 0.325],
                           [0.042, 0.167, 0.427],
                           [-0.072, 0.056, 0.456],
                           [-0.086, -0.058, 0.392],
                           [0.014, -0.057, 0.296],
                           [-0.02, -0.176, 0.229],
                           [0.754, -0.059, -0.163],
                           [0.624, -0.052, -0.266],
                           [0.644, -0.163, -0.325],
                           [0.542, -0.167, -0.427],
                           [0.428, -0.056, -0.456],
                           [0.414, 0.058, -0.392],
                           [0.514, 0.057, -0.296],
                           [0.48, 0.176, -0.229],
                           [0.254, 0.559, -0.163],
                           [0.124, 0.552, -0.266],
                           [0.144, 0.663, -0.325],
                           [0.042, 0.667, -0.427],
                           [-0.072, 0.556, -0.456],
                           [-0.086, 0.442, -0.392],
                           [0.014, 0.443, -0.296],
                           [-0.02, 0.324, -0.229],
                           [0.754, 0.441, 0.163],
                           [0.624, 0.448, 0.266],
                           [0.644, 0.337, 0.325],
                           [0.542, 0.333, 0.427],
                           [0.428, 0.444, 0.456],
                           [0.414, 0.558, 0.392],
                           [0.514, 0.557, 0.296],
                           [0.48, 0.676, 0.229],
                           [0.19, 0.137, 0.097],
                           [0.394, -0.02, 0.159],
                           [-0.008, -0.164, 0.132],
                           [-0.043, -0.288, 0.275],
                           [0.69, -0.137, -0.097],
                           [0.894, 0.02, -0.159],
                           [0.492, 0.164, -0.132],
                           [0.457, 0.288, -0.275],
                           [0.19, 0.637, -0.097],
                           [0.394, 0.48, -0.159],
                           [-0.008, 0.336, -0.132],
                           [-0.043, 0.212, -0.275],
                           [0.69, 0.363, 0.097],
                           [0.894, 0.52, 0.159],
                           [0.492, 0.664, 0.132],
                           [0.457, 0.788, 0.275]],
                 a=6.615, b=10.047, c=12.878,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Muscovite(rmat.CrystalFromCell):
    '''System: Monoclinic
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 4, p.  346
       KAl2(AlSi3)O10(OH)2
       The muscovite or mica is composed by K Al Si O and H atoms in 10
       different sites  making 13 independent element-sites. It belongs
       to the monoclinic system with abc=5.189 8.995 20.097 beta=95.18


       '''

    def __init__(self, name='Muscovite',
                 atoms=[19, 19, 19, 19, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
                        14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13,
                        14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13,
                        13, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1,
                        1, 1, 1, 1],
                 atomsXYZ=[[0.0, 0.1016, 0.25],
                           [0.5, 0.6016, 0.25],
                           [-0.0, -0.1016, -0.25],
                           [-0.5, -0.6016, -0.25],
                           [0.2484, 0.0871, 0.0016],
                           [0.2484, -0.0871, 0.5016],
                           [0.7484, 0.5871, 0.0016],
                           [0.7484, 0.4129, 0.5016],
                           [-0.2484, -0.0871, -0.0016],
                           [-0.2484, 0.0871, -0.5016],
                           [-0.7484, -0.5871, -0.0016],
                           [-0.7484, -0.4129, -0.5016],
                           [0.4625, 0.9242, 0.1372],
                           [0.4625, -0.9242, 0.6372],
                           [0.9625, 1.4242, 0.1372],
                           [0.9625, -0.4242, 0.6372],
                           [-0.4625, -0.9242, -0.1372],
                           [-0.4625, 0.9242, -0.6372],
                           [-0.9625, -1.4242, -0.1372],
                           [-0.9625, 0.4242, -0.6372],
                           [0.4625, 0.9242, 0.1372],
                           [0.4625, -0.9242, 0.6372],
                           [0.9625, 1.4242, 0.1372],
                           [0.9625, -0.4242, 0.6372],
                           [-0.4625, -0.9242, -0.1372],
                           [-0.4625, 0.9242, -0.6372],
                           [-0.9625, -1.4242, -0.1372],
                           [-0.9625, 0.4242, -0.6372],
                           [0.4593, 0.255, 0.1365],
                           [0.4593, -0.255, 0.6365],
                           [0.9593, 0.755, 0.1365],
                           [0.9593, 0.245, 0.6365],
                           [-0.4593, -0.255, -0.1365],
                           [-0.4593, 0.255, -0.6365],
                           [-0.9593, -0.755, -0.1365],
                           [-0.9593, -0.245, -0.6365],
                           [0.4593, 0.255, 0.1365],
                           [0.4593, -0.255, 0.6365],
                           [0.9593, 0.755, 0.1365],
                           [0.9593, 0.245, 0.6365],
                           [-0.4593, -0.255, -0.1365],
                           [-0.4593, 0.255, -0.6365],
                           [-0.9593, -0.755, -0.1365],
                           [-0.9593, -0.245, -0.6365],
                           [0.2629, 0.3713, 0.1674],
                           [0.2629, -0.3713, 0.6674],
                           [0.7629, 0.8713, 0.1674],
                           [0.7629, 0.1287, 0.6674],
                           [-0.2629, -0.3713, -0.1674],
                           [-0.2629, 0.3713, -0.6674],
                           [-0.7629, -0.8713, -0.1674],
                           [-0.7629, -0.1287, -0.6674],
                           [0.245, 0.802, 0.162],
                           [0.245, -0.802, 0.662],
                           [0.745, 1.302, 0.162],
                           [0.745, -0.302, 0.662],
                           [-0.245, -0.802, -0.162],
                           [-0.245, 0.802, -0.662],
                           [-0.745, -1.302, -0.162],
                           [-0.745, 0.302, -0.662],
                           [0.408, 0.096, 0.168],
                           [0.408, -0.096, 0.668],
                           [0.908, 0.596, 0.168],
                           [0.908, 0.404, 0.668],
                           [-0.408, -0.096, -0.168],
                           [-0.408, 0.096, -0.668],
                           [-0.908, -0.596, -0.168],
                           [-0.908, -0.404, -0.668],
                           [0.465, 0.945, 0.0527],
                           [0.465, -0.945, 0.5527],
                           [0.965, 1.445, 0.0527],
                           [0.965, -0.445, 0.5527],
                           [-0.465, -0.945, -0.0527],
                           [-0.465, 0.945, -0.5527],
                           [-0.965, -1.445, -0.0527],
                           [-0.965, 0.445, -0.5527],
                           [0.425, 0.26, 0.0542],
                           [0.425, -0.26, 0.5542],
                           [0.925, 0.76, 0.0542],
                           [0.925, 0.24, 0.5542],
                           [-0.425, -0.26, -0.0542],
                           [-0.425, 0.26, -0.5542],
                           [-0.925, -0.76, -0.0542],
                           [-0.925, -0.24, -0.5542],
                           [0.453, 0.558, 0.052],
                           [0.453, -0.558, 0.552],
                           [0.953, 1.058, 0.052],
                           [0.953, -0.058, 0.552],
                           [-0.453, -0.558, -0.052],
                           [-0.453, 0.558, -0.552],
                           [-0.953, -1.058, -0.052],
                           [-0.953, 0.058, -0.552],
                           [0.453, 0.558, 0.052],
                           [0.453, -0.558, 0.552],
                           [0.953, 1.058, 0.052],
                           [0.953, -0.058, 0.552],
                           [-0.453, -0.558, -0.052],
                           [-0.453, 0.558, -0.552],
                           [-0.953, -1.058, -0.052],
                           [-0.953, 0.058, -0.552]],
                 a=5.189, b=8.995, c=20.097,
                 alpha=90.0, beta=95.18, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class AlphaQuartz(rmat.CrystalFromCell):
    '''System: Hexagonal
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1
       Structure: AlphaQuartz
       The AlphaQuartz (SiO2) structure has three molecules in a
       hexagonal unit of dimensions a=4.91304 A  c=5.40463 A


       '''

    def __init__(self, name='AlphaQuartz',
                 atoms=[14, 14, 14, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.4697, 0.0, 0.0],
                           [-0.4697, -0.4697, 0.3333333333],
                           [0.0, 0.4697, 0.6666666666],
                           [0.4125, 0.2662, 0.1188],
                           [-0.1463, -0.4125, 0.4521],
                           [-0.2662, 0.1463, -0.2145],
                           [0.1463, -0.2662, -0.1188],
                           [-0.4125, -0.1463, 0.2145],
                           [0.2662, 0.4125, 0.5479]],
                 a=4.91304, b=4.91304, c=5.40463,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Copper(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: Copper
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1965, 1, p.10
       The FCC structure is defined by atom A located at (0 0 0)
       and (.5,0,.5), (.5,.5,0) and (0,.5,.5)
       Example: Ag, Al, Fe, Cu, Co, etc.


       '''

    def __init__(self, name='Copper',
                 atoms=[29, 29, 29, 29],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]],
                 a=3.61496, b=3.61496, c=3.61496,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class LiNbO3(rmat.CrystalFromCell):
    '''System: Trigonal
       R.S. Weis and T.K. Gaylord, Appl. Phys. 191-203 (1985)
       LiNbO3 Ref:R.S. Weis and T.K. Gaylord, Appl. Phys. 191-203 (1985)
       (coordinates calculated by Olivier Mathon (mathon@esrf.fr)


       '''

    def __init__(self, name='LiNbO3',
                 atoms=[3, 3, 3, 3, 3, 3, 41, 41, 41, 41, 41, 41, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.0, 0.0, 0.2829],
                           [0.3333, 0.6667, 0.9496],
                           [0.6667, 0.3333, 0.6162],
                           [0.0, 0.0, 0.7829],
                           [0.3333, 0.6667, 0.4496],
                           [0.6667, 0.3333, 0.1162],
                           [0.0, 0.0, 0.0],
                           [0.3333, 0.6667, 0.6667],
                           [0.6667, 0.3333, 0.3333],
                           [0.0, 0.0, 0.5],
                           [0.3333, 0.6667, 0.1667],
                           [0.6667, 0.3333, 0.8333],
                           [0.0492, 0.3446, 0.0647],
                           [0.3825, 0.0113, 0.7314],
                           [0.7159, 0.6779, 0.398],
                           [0.6554, 0.7046, 0.0647],
                           [0.9887, 0.3713, 0.7314],
                           [0.3221, 0.0379, 0.398],
                           [0.2954, 0.9508, 0.0647],
                           [0.6287, 0.6175, 0.7314],
                           [0.9621, 0.2841, 0.398],
                           [0.6554, 0.9508, 0.5647],
                           [0.9887, 0.6175, 0.2314],
                           [0.3221, 0.2841, 0.898],
                           [0.0492, 0.7046, 0.5647],
                           [0.3825, 0.3713, 0.2314],
                           [0.7159, 0.0379, 0.898],
                           [0.2954, 0.3446, 0.5647],
                           [0.6287, 0.0113, 0.2314],
                           [0.9621, 0.6779, 0.898]],
                 a=5.148, b=5.148, c=13.863,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Platinum(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: Copper
       The FCC structure is defined by atom A located at (0 0 0)
       and (.5,0,.5), (.5,.5,0) and (0,.5,.5)
       Example: Ag, Al, Fe, Cu, Co, etc.


       '''

    def __init__(self, name='Platinum',
                 atoms=[78, 78, 78, 78],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]],
                 a=3.9242, b=3.9242, c=3.9242,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Gold(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: Copper
       The FCC structure is defined by atom A located at (0 0 0)
       and (.5,0,.5), (.5,.5,0) and (0,.5,.5)
       Example: Ag, Al, Fe, Cu, Co, etc.


       '''

    def __init__(self, name='Gold',
                 atoms=[79, 79, 79, 79],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]],
                 a=4.0782, b=4.0782, c=4.0782,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Sapphire(rmat.CrystalFromCell):
    '''System: Hexagonal
       Yuri Shvyd'ko X-ray Optics High Energy Resolution Applications, Springer
       p.337 Sapphire, Alpha-Al2O3, rhombohedral lattice, space group R-3c
       (D3d,6)


       '''

    def __init__(self, name='Sapphire',
                 atoms=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.0, 0.0, 0.3522],
                           [0.0, 0.0, 0.1478],
                           [0.0, 0.0, -0.3522],
                           [0.0, 0.0, 0.8522],
                           [0.66667, 0.33333, 0.68553],
                           [0.66667, 0.33333, 0.48113],
                           [0.66667, 0.33333, -0.01887],
                           [0.66667, 0.33333, 1.18553],
                           [0.33333, 0.66667, 1.01887],
                           [0.33333, 0.66667, 0.81447],
                           [0.33333, 0.66667, 0.31447],
                           [0.33333, 0.66667, 1.51887],
                           [0.30627, 0.0, 0.25],
                           [0.0, 0.30627, 0.25],
                           [-0.30627, -0.30627, 0.25],
                           [-0.30627, 0.0, 0.75],
                           [0.0, -0.30627, 0.75],
                           [0.30627, 0.30627, 0.75],
                           [0.97294, 0.33333, 0.58333],
                           [0.66667, 0.63961, 0.58333],
                           [0.36039, 0.02706, 0.58333],
                           [0.36039, 0.33333, 1.08333],
                           [0.66667, 0.02706, 1.08333],
                           [0.97294, 0.63961, 1.08333],
                           [0.63961, 0.66667, 0.91667],
                           [0.33333, 0.97294, 0.91667],
                           [0.02706, 0.36039, 0.91667],
                           [0.02706, 0.66667, 1.41667],
                           [0.33333, 0.36039, 1.41667],
                           [0.63961, 0.97294, 1.41667]],
                 a=4.7581322, b=4.7581322, c=12.9883093,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class LaB6(rmat.CrystalFromCell):
    '''System: Cubic
       C.H. Booth et al. Phys. Rev. B 63, 224302 (2001) ICSD Code 94251
       C.H. Booth et al. Phys. Rev. B 63, 224302 (2001) ICSD Code 94251


       '''

    def __init__(self, name='LaB6',
                 atoms=[57, 5, 5, 5, 5, 5, 5],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.1993, 0.5, 0.5],
                           [0.8007, 0.5, 0.5],
                           [0.5, 0.1993, 0.5],
                           [0.5, 0.5, 0.1993],
                           [0.5, 0.8007, 0.5],
                           [0.5, 0.5, 0.8007]],
                 a=4.15271, b=4.15271, c=4.15271,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class LaB6NIST(rmat.CrystalFromCell):
    '''System: Cubic
       NIST Standard Reference Material 660a (Lanthanum Hexaboride Powder)
       C.H. Booth et al. Phys. Rev. B 63, 224302 (2001) ICSD Code 94251


       '''

    def __init__(self, name='LaB6_NIST',
                 atoms=[57, 5, 5, 5, 5, 5, 5],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.1993, 0.5, 0.5],
                           [0.8007, 0.5, 0.5],
                           [0.5, 0.1993, 0.5],
                           [0.5, 0.5, 0.1993],
                           [0.5, 0.8007, 0.5],
                           [0.5, 0.5, 0.8007]],
                 a=4.1569162, b=4.1569162, c=4.1569162,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class KTP(rmat.CrystalFromCell):
    '''System: Orthorhombic
       Tordjman et al. Zeitschrift fuer Kristallographie, (1974) 103-115
       Tordjman et al. (1974)
       Data calculated with data from http://icsd.ill.fr/icsd/index.html


       '''

    def __init__(self, name='KTP',
                 atoms=[22, 22, 22, 22, 22, 22, 22, 22, 15, 15, 15, 15, 15, 15,
                        15, 15, 19, 19, 19, 19, 19, 19, 19, 19, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.37291, 0.0, 0.50011],
                           [0.12709, 0.5, 0.00011122],
                           [0.62709, 0.5, 0.49989],
                           [0.87291, 0.0, 0.99989],
                           [0.24671, 0.25145, 0.26915],
                           [0.25329, 0.75145, 0.76915],
                           [0.75329, 0.75145, 0.73085],
                           [0.74671, 0.25145, 0.23085],
                           [0.49801, 0.26078, 0.33639],
                           [0.00199, 0.76078, 0.83639],
                           [0.50199, 0.76078, 0.66361],
                           [0.99801, 0.26078, 0.16361],
                           [0.18081, 0.5131, 0.50197],
                           [0.31919, 0.013101, 0.001972],
                           [0.81919, 0.013101, 0.49803],
                           [0.68081, 0.5131, 0.99803],
                           [0.37801, 0.31201, 0.78042],
                           [0.12199, 0.81201, 0.28042],
                           [0.62199, 0.81201, 0.21958],
                           [0.87801, 0.31201, 0.71958],
                           [0.10528, 0.066902, 0.69892],
                           [0.39472, 0.5669, 0.19892],
                           [0.89472, 0.5669, 0.30108],
                           [0.60528, 0.066902, 0.80108],
                           [0.48623, 0.15043, 0.48656],
                           [0.01377, 0.65043, 0.98656],
                           [0.51377, 0.65043, 0.51344],
                           [0.98623, 0.15043, 0.01344],
                           [0.51003, 0.38363, 0.46507],
                           [0.98997, 0.88363, 0.96507],
                           [0.48997, 0.88363, 0.53493],
                           [0.01003, 0.38363, 0.03493],
                           [0.39993, 0.28004, 0.19896],
                           [0.10007, 0.78004, 0.69896],
                           [0.60007, 0.78004, 0.80104],
                           [0.89993, 0.28004, 0.30104],
                           [0.59373, 0.24224, 0.19256],
                           [0.90627, 0.74224, 0.69256],
                           [0.40627, 0.74224, 0.80744],
                           [0.09373, 0.24224, 0.30744],
                           [0.22473, 0.64473, 0.96664],
                           [0.27527, 0.14473, 0.46664],
                           [0.77527, 0.14473, 0.03336],
                           [0.72473, 0.64473, 0.53336],
                           [0.22383, 0.39074, 0.04056],
                           [0.27617, 0.89074, 0.54056],
                           [0.77617, 0.89074, 0.95944],
                           [0.72383, 0.39074, 0.45944],
                           [0.11203, 0.54214, 0.31055],
                           [0.38797, 0.04214, 0.81055],
                           [0.88797, 0.04214, 0.68945],
                           [0.61203, 0.54214, 0.18945],
                           [0.11113, 0.48784, 0.69185],
                           [0.38887, 0.98784, 0.19185],
                           [0.88887, 0.98784, 0.30815],
                           [0.61113, 0.48784, 0.80815],
                           [0.25263, 0.62854, 0.53967],
                           [0.24737, 0.12854, 0.03967],
                           [0.74737, 0.12854, 0.46033],
                           [0.75263, 0.62854, 0.96033],
                           [0.25303, 0.39953, 0.46056],
                           [0.24697, 0.89953, 0.96056],
                           [0.74697, 0.89953, 0.53944],
                           [0.75303, 0.39953, 0.03944]],
                 a=12.8146, b=10.6165, c=6.4042,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class AlphaAlumina(rmat.CrystalFromCell):
    '''System: Trigonal
       P. Ballirano, R. Caminiti J Appl Crystall (2001) 34, 757-762
       Data calculated with xop/xpowder
       Al2 O3 - [Corundum] Dialuminium trioxide - alpha


       '''

    def __init__(self, name='AlphaAlumina',
                 atoms=[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 atomsXYZ=[[0.0, 0.0, 0.35242],
                           [0.0, 0.0, 0.14758],
                           [-0.0, -0.0, 0.64758],
                           [-0.0, -0.0, 0.85242],
                           [0.66667, 0.33333, 0.68575],
                           [0.66667, 0.33333, 0.48091],
                           [0.66667, 0.33333, 0.98091],
                           [0.66667, 0.33333, 0.18575],
                           [0.33333, 0.66667, 0.019085],
                           [0.33333, 0.66667, 0.81425],
                           [0.33333, 0.66667, 0.31425],
                           [0.33333, 0.66667, 0.51909],
                           [0.30634, 0.0, 0.25],
                           [-0.0, 0.30634, 0.25],
                           [0.69366, 0.69366, 0.25],
                           [0.69366, -0.0, 0.75],
                           [0.0, 0.69366, 0.75],
                           [0.30634, 0.30634, 0.75],
                           [0.97301, 0.33333, 0.58333],
                           [0.66667, 0.63967, 0.58333],
                           [0.36033, 0.026993, 0.58333],
                           [0.36033, 0.33333, 0.083333],
                           [0.66667, 0.026993, 0.083333],
                           [0.97301, 0.63967, 0.083333],
                           [0.63967, 0.66667, 0.91667],
                           [0.33333, 0.97301, 0.91667],
                           [0.026993, 0.36033, 0.91667],
                           [0.026993, 0.66667, 0.41667],
                           [0.33333, 0.36033, 0.41667],
                           [0.63967, 0.97301, 0.41667]],
                 a=4.758846, b=4.758846, c=12.99306,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Aluminum(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: Copper
       The FCC structure is defined by atom A located at (0 0 0)
       and (.5,0,.5), (.5,.5,0) and (0,.5,.5)
       Example: Ag, Al, Fe, Cu, Co, etc.
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1963, 1, p.10


       '''

    def __init__(self, name='Aluminum',
                 atoms=[13, 13, 13, 13],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]],
                 a=4.04958, b=4.04958, c=4.04958,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Iron(rmat.CrystalFromCell):
    '''System: Cubic
       Structure: Copper
       The FCC structure is defined by atom A located at (0 0 0)
       and (.5,0,.5), (.5,.5,0) and (0,.5,.5)
       Example: Ag, Al, Fe, Cu, Co, etc.
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1963, 1, p.10


       '''

    def __init__(self, name='Iron',
                 atoms=[26, 26, 26, 26],
                 atomsXYZ=[[0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]],
                 a=3.591, b=3.591, c=3.591,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)


class Titanium(rmat.CrystalFromCell):
    '''System: Hexagonal
       Structure: Berilium
       R.W.G. Wyckoff, Crystal Structures, Interscience Publ. 1963, vol 1, 11
       The Hexagonal Closed-packed structure is defined by an atom A
       located at (1/3 2/3 1/4) and an atom B at (2/3  1/3  3/4) in the
       prism cell. Example: Berilium (with a=2.287  c=3.583)


       '''

    def __init__(self, name='Titanium',
                 atoms=[22, 22],
                 atomsXYZ=[[0.333333, 0.666667, 0.25],
                           [0.666667, 0.333333, 0.75]],
                 a=2.95, b=2.95, c=4.686,
                 alpha=90.0, beta=90.0, gamma=120.0,
                 *args, **kwargs):
        super().__init__(name=name,
                         a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                         *args, **kwargs)

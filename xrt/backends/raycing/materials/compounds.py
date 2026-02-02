r"""
The module :mod:`~xrt.backends.raycing.materials.compounds` contains predefined
classes for compound materials found at the
`CXRO Table of Densities of Common Materials
<https://henke.lbl.gov/cgi-bin/density.pl>`_.
Air composition and density are given as in [Cox]_.

.. [Cox] Cox, Arthur N., ed. (2000), Allen's Astrophysical Quantities
   (Fourth ed.), AIP Press, pp. 258–259, ISBN 0-387-98746-0

To use a compound material in a script, simply import the module and
instantiate its class:

.. code-block:: python

    import xrt.backends.raycing.materials.compounds as xcomp
    kapton = xcomp.Polyimide()

The compound materials inherit from :class:`.Material` and can use its methods
to calculate reflection or transmission amplitudes, absorption coefficient,
refractive index etc.

.. note::
    The compound materials do not provide crystal diffraction amplitudes even
    if they occur naturally as crystals. To calculate diffraction on crystals
    please use :mod:`~xrt.backends.raycing.materials.crystals`.

The following compound classes are defined in this module (sorted by chemical
formula):

   | |compall|

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "6 Jul 2017"

from .material import Material
import collections

__all__ = ('SilverBromide', 'AluminumArsenide', 'Sapphire',
           'AluminumPhosphide', 'BoronOxide', 'BoronCarbide', 'BerylliumOxide',
           'BoronNitride', 'Polyimide', 'Polypropylene', 'PMMA',
           'Polycarbonate', 'Kimfol', 'Mylar', 'Teflon', 'ParyleneC',
           'ParyleneN', 'Fluorite', 'CadmiumTungstate', 'CadmiumSulfide',
           'CadmiumTelluride', 'CobaltSilicide', 'Cromium3Oxide',
           'CesiumIodide', 'CopperIodide', 'IndiumNitride', 'Indium3Oxide',
           'IndiumAntimonide', 'IridiumOxide', 'GalliumArsenide',
           'GalliumNitride', 'GalliumPhosphide', 'HafniumOxide',
           'LithiumFluoride', 'LithiumHydride', 'LithiumHydroxide',
           'MagnesiumFluoride', 'MagnesiumOxide', 'MagnesiumSilicide',
           'Mica', 'Manganese2Oxide', 'Manganese4Oxide', 'Molybdenum4Oxide',
           'Molybdenum6Oxide', 'MolybdenumSilicide', 'RockSalt',
           'NiobiumSilicide', 'NiobiumNitride', 'NickelOxide',
           'NickelSilicide', 'RutheniumSilicide', 'Ruthenium4Oxide',
           'SiliconCarbide', 'SiliconNitride', 'Silica', 'Quartz',
           'TantalumNitride', 'TantalumOxide', 'TitaniumNitride',
           'TitaniumSilicide', 'TantalumSilicide', 'Rutile', 'ULEGlass',
           'Uranium4Oxide', 'VanadiumNitride', 'Vacuum', 'Water',
           'TungstenCarbide', 'Zerodur', 'ZincOxide', 'ZincSulfide',
           'ZirconiumNitride', 'Zirconia', 'ZirconiumSilicide', 'Air',
           'CVDDiamond',
           )

__allSectioned__ = collections.OrderedDict([
    ('Li-C',
        ('LithiumFluoride', 'LithiumHydride', 'LithiumHydroxide',
         'BoronOxide', 'BoronCarbide', 'BoronNitride', 'Polyimide',
         'Polypropylene', 'PMMA', 'Polycarbonate', 'Kimfol', 'Mylar',
         'Teflon', 'ParyleneC', 'ParyleneN', 'CVDDiamond')),
    ('Na-Ca',
        ('RockSalt', 'MagnesiumFluoride', 'MagnesiumOxide',
         'MagnesiumSilicide', 'Sapphire', 'AluminumPhosphide',
         'AluminumArsenide', 'SiliconCarbide', 'SiliconNitride',
         'Silica', 'Quartz', 'ULEGlass', 'Zerodur', 'Mica', 'Fluorite')),
    ('Ti-Ga',
        ('TitaniumNitride', 'TitaniumSilicide', 'Rutile', 'VanadiumNitride',
         'Cromium3Oxide', 'Manganese2Oxide', 'Manganese4Oxide',
         'CobaltSilicide', 'NickelOxide', 'NickelSilicide', 'CopperIodide',
         'ZincOxide', 'ZincSulfide', 'GalliumNitride', 'GalliumPhosphide',
         'GalliumArsenide')),
    ('Zr-Cd',
        ('ZirconiumNitride', 'Zirconia', 'ZirconiumSilicide', 'NiobiumNitride',
         'NiobiumSilicide', 'Molybdenum4Oxide', 'Molybdenum6Oxide',
         'MolybdenumSilicide', 'Ruthenium4Oxide', 'RutheniumSilicide',
         'SilverBromide', 'CadmiumSulfide', 'CadmiumTelluride',
         'CadmiumTungstate')),
    ('In-U',
        ('Indium3Oxide', 'IndiumNitride', 'IndiumAntimonide', 'CesiumIodide',
         'HafniumOxide', 'TantalumOxide', 'TitaniumNitride',
         'TitaniumSilicide', 'TungstenCarbide',
         'IridiumOxide', 'Uranium4Oxide')),
    ('Other',
        ('Vacuum', 'Air', 'Water'))])


class SilverBromide(Material):
    def __init__(self, name='AgBr',
                 elements=['Ag', 'Br'],
                 quantities=[1, 1],
                 rho=6.473, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class AluminumArsenide(Material):
    def __init__(self, name='AlAs',
                 elements=['Al', 'As'],
                 quantities=[1, 1],
                 rho=3.81, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Sapphire(Material):
    def __init__(self, name='Al2O3',
                 elements=['Al', 'O'],
                 quantities=[2.0, 3.0],
                 rho=3.97, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class AluminumPhosphide(Material):
    def __init__(self, name='AlP',
                 elements=['Al', 'P'],
                 quantities=[1, 1],
                 rho=2.42, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class BoronOxide(Material):
    def __init__(self, name='B2O3',
                 elements=['B', 'O'],
                 quantities=[2.0, 3.0],
                 rho=3.11, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class BoronCarbide(Material):
    def __init__(self, name='B4C',
                 elements=['B', 'C'],
                 quantities=[4.0, 1],
                 rho=2.52, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class BerylliumOxide(Material):
    def __init__(self, name='BeO',
                 elements=['Be', 'O'],
                 quantities=[1, 1],
                 rho=3.01, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class BoronNitride(Material):
    def __init__(self, name='BN',
                 elements=['B', 'N'],
                 quantities=[1, 1],
                 rho=2.25, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Polyimide(Material):
    def __init__(self, name='C22H10N2O5',
                 elements=['C', 'H', 'N', 'O'],
                 quantities=[22.0, 10.0, 2.0, 5.0],
                 rho=1.43, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Polypropylene(Material):
    def __init__(self, name='C3H6',
                 elements=['C', 'H'],
                 quantities=[3.0, 6.0],
                 rho=0.9, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class PMMA(Material):
    def __init__(self, name='C5H8O2',
                 elements=['C', 'H', 'O'],
                 quantities=[5.0, 8.0, 2.0],
                 rho=1.19, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Polycarbonate(Material):
    def __init__(self, name='C16H14O3',
                 elements=['C', 'H', 'O'],
                 quantities=[16.0, 14.0, 3.0],
                 rho=1.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Kimfol(Material):
    def __init__(self, name='C16H14O3',
                 elements=['C', 'H', 'O'],
                 quantities=[16.0, 14.0, 3.0],
                 rho=1.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Mylar(Material):
    def __init__(self, name='C10H8O4',
                 elements=['C', 'H', 'O'],
                 quantities=[10.0, 8.0, 4.0],
                 rho=1.4, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Teflon(Material):
    def __init__(self, name='C2F4',
                 elements=['C', 'F'],
                 quantities=[2.0, 4.0],
                 rho=2.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ParyleneC(Material):
    def __init__(self, name='C8H7Cl',
                 elements=['C', 'H', 'Cl'],
                 quantities=[8.0, 7.0, 1],
                 rho=1.29, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ParyleneN(Material):
    def __init__(self, name='C8H8',
                 elements=['C', 'H'],
                 quantities=[8.0, 8.0],
                 rho=1.11, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Fluorite(Material):
    def __init__(self, name='CaF2',
                 elements=['Ca', 'F'],
                 quantities=[1, 2.0],
                 rho=3.18, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CadmiumTungstate(Material):
    def __init__(self, name='CdWO4',
                 elements=['Cd', 'W', 'O'],
                 quantities=[1, 1, 4.0],
                 rho=7.9, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CadmiumSulfide(Material):
    def __init__(self, name='CdS',
                 elements=['Cd', 'S'],
                 quantities=[1, 1],
                 rho=4.826, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CadmiumTelluride(Material):
    def __init__(self, name='CdTe',
                 elements=['Cd', 'Te'],
                 quantities=[1, 1],
                 rho=5.85, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CobaltSilicide(Material):
    def __init__(self, name='CoSi2',
                 elements=['Co', 'Si'],
                 quantities=[1, 2.0],
                 rho=5.3, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Cromium3Oxide(Material):
    def __init__(self, name='Cr2O3',
                 elements=['Cr', 'O'],
                 quantities=[2.0, 3.0],
                 rho=5.21, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CesiumIodide(Material):
    def __init__(self, name='CsI',
                 elements=['Cs', 'I'],
                 quantities=[1, 1],
                 rho=4.51, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CopperIodide(Material):
    def __init__(self, name='CuI',
                 elements=['Cu', 'I'],
                 quantities=[1, 1],
                 rho=5.63, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class IndiumNitride(Material):
    def __init__(self, name='InN',
                 elements=['In', 'N'],
                 quantities=[1, 1],
                 rho=6.88, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Indium3Oxide(Material):
    def __init__(self, name='In2O3',
                 elements=['In', 'O'],
                 quantities=[2.0, 3.0],
                 rho=7.179, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class IndiumAntimonide(Material):
    def __init__(self, name='InSb',
                 elements=['In', 'Sb'],
                 quantities=[1, 1],
                 rho=5.775, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class IridiumOxide(Material):
    def __init__(self, name='IrO2',
                 elements=['Ir', 'O'],
                 quantities=[1, 2.0],
                 rho=11.66, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class GalliumArsenide(Material):
    def __init__(self, name='GaAs',
                 elements=['Ga', 'As'],
                 quantities=[1, 1],
                 rho=5.316, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class GalliumNitride(Material):
    def __init__(self, name='GaN',
                 elements=['Ga', 'N'],
                 quantities=[1, 1],
                 rho=6.1, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class GalliumPhosphide(Material):
    def __init__(self, name='GaP',
                 elements=['Ga', 'P'],
                 quantities=[1, 1],
                 rho=4.13, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class HafniumOxide(Material):
    def __init__(self, name='HfO2',
                 elements=['Hf', 'O'],
                 quantities=[1, 2.0],
                 rho=9.68, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class LithiumFluoride(Material):
    def __init__(self, name='LiF',
                 elements=['Li', 'F'],
                 quantities=[1, 1],
                 rho=2.635, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class LithiumHydride(Material):
    def __init__(self, name='LiH',
                 elements=['Li', 'H'],
                 quantities=[1, 1],
                 rho=0.783, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class LithiumHydroxide(Material):
    def __init__(self, name='LiOH',
                 elements=['Li', 'O', 'H'],
                 quantities=[1, 1, 1],
                 rho=1.43, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class MagnesiumFluoride(Material):
    def __init__(self, name='MgF2',
                 elements=['Mg', 'F'],
                 quantities=[1, 2.0],
                 rho=3.18, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class MagnesiumOxide(Material):
    def __init__(self, name='MgO',
                 elements=['Mg', 'O'],
                 quantities=[1, 1],
                 rho=3.58, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class MagnesiumSilicide(Material):
    def __init__(self, name='Mg2Si',
                 elements=['Mg', 'Si'],
                 quantities=[2.0, 1],
                 rho=1.94, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Mica(Material):
    def __init__(self, name='KAl3Si3O12H2',
                 elements=['K', 'Al', 'Si', 'O', 'H'],
                 quantities=[1, 3.0, 3.0, 12.0, 2.0],
                 rho=2.83, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Manganese2Oxide(Material):
    def __init__(self, name='MnO',
                 elements=['Mn', 'O'],
                 quantities=[1, 1],
                 rho=5.44, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Manganese4Oxide(Material):
    def __init__(self, name='MnO2',
                 elements=['Mn', 'O'],
                 quantities=[1, 2.0],
                 rho=5.03, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Molybdenum4Oxide(Material):
    def __init__(self, name='MoO2',
                 elements=['Mo', 'O'],
                 quantities=[1, 2.0],
                 rho=6.47, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Molybdenum6Oxide(Material):
    def __init__(self, name='MoO3',
                 elements=['Mo', 'O'],
                 quantities=[1, 3.0],
                 rho=4.69, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class MolybdenumSilicide(Material):
    def __init__(self, name='MoSi2',
                 elements=['Mo', 'Si'],
                 quantities=[1, 2.0],
                 rho=6.31, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class RockSalt(Material):
    def __init__(self, name='NaCl',
                 elements=['Na', 'Cl'],
                 quantities=[1, 1],
                 rho=2.165, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class NiobiumSilicide(Material):
    def __init__(self, name='NbSi2',
                 elements=['Nb', 'Si'],
                 quantities=[1, 2.0],
                 rho=5.37, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class NiobiumNitride(Material):
    def __init__(self, name='NbN',
                 elements=['Nb', 'N'],
                 quantities=[1, 1],
                 rho=8.47, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class NickelOxide(Material):
    def __init__(self, name='NiO',
                 elements=['Ni', 'O'],
                 quantities=[1, 1],
                 rho=6.67, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class NickelSilicide(Material):
    def __init__(self, name='Ni2Si',
                 elements=['Ni', 'Si'],
                 quantities=[2.0, 1],
                 rho=7.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class RutheniumSilicide(Material):
    def __init__(self, name='Ru2Si3',
                 elements=['Ru', 'Si'],
                 quantities=[2.0, 3.0],
                 rho=6.96, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Ruthenium4Oxide(Material):
    def __init__(self, name='RuO2',
                 elements=['Ru', 'O'],
                 quantities=[1, 2.0],
                 rho=6.97, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class SiliconCarbide(Material):
    def __init__(self, name='SiC',
                 elements=['Si', 'C'],
                 quantities=[1, 1],
                 rho=3.217, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class SiliconNitride(Material):
    def __init__(self, name='Si3N4',
                 elements=['Si', 'N'],
                 quantities=[3.0, 4.0],
                 rho=3.44, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Silica(Material):
    def __init__(self, name='SiO2',
                 elements=['Si', 'O'],
                 quantities=[1, 2.0],
                 rho=2.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Quartz(Material):
    def __init__(self, name='SiO2',
                 elements=['Si', 'O'],
                 quantities=[1, 2.0],
                 rho=2.65, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TantalumNitride(Material):
    def __init__(self, name='TaN',
                 elements=['Ta', 'N'],
                 quantities=[1, 1],
                 rho=16.3, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TantalumOxide(Material):
    def __init__(self, name='Ta2O5',
                 elements=['Ta', 'O'],
                 quantities=[2.0, 5.0],
                 rho=8.2, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TitaniumNitride(Material):
    def __init__(self, name='TiN',
                 elements=['Ti', 'N'],
                 quantities=[1, 1],
                 rho=5.22, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TitaniumSilicide(Material):
    def __init__(self, name='TiSi2',
                 elements=['Ti', 'Si'],
                 quantities=[1, 2.0],
                 rho=4.02, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TantalumSilicide(Material):
    def __init__(self, name='Ta2Si',
                 elements=['Ta', 'Si'],
                 quantities=[2.0, 1],
                 rho=14, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Rutile(Material):
    def __init__(self, name='TiO2',
                 elements=['Ti', 'O'],
                 quantities=[1, 2.0],
                 rho=4.26, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ULEGlass(Material):
    def __init__(self, name='Si.925Ti.075O2',
                 elements=['Si', 'Ti', 'O'],
                 quantities=[0.925, 0.075, 2.0],
                 rho=2.205, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Uranium4Oxide(Material):
    def __init__(self, name='UO2',
                 elements=['U', 'O'],
                 quantities=[1, 2.0],
                 rho=10.96, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class VanadiumNitride(Material):
    def __init__(self, name='VN',
                 elements=['V', 'N'],
                 quantities=[1, 1],
                 rho=6.13, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Vacuum(Material):
    def __init__(self, name=' ',
                 elements=['H'],
                 quantities=[1],
                 rho=0, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Water(Material):
    def __init__(self, name='H2O',
                 elements=['H', 'O'],
                 quantities=[2.0, 1],
                 rho=1, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class TungstenCarbide(Material):
    def __init__(self, name='WC',
                 elements=['W', 'C'],
                 quantities=[1, 1],
                 rho=15.63, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Zerodur(Material):
    def __init__(self, name='Si.56Al.5P.16Li.04Ti.02Zr.02Zn.03O2.46',
                 elements=['Si', 'Al', 'P', 'Li', 'Ti', 'Zr', 'Zn', 'O'],
                 quantities=[0.56, 0.5, 0.16, 0.04, 0.02, 0.02, 0.03, 2.46],
                 rho=2.53, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ZincOxide(Material):
    def __init__(self, name='ZnO',
                 elements=['Zn', 'O'],
                 quantities=[1, 1],
                 rho=5.675, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ZincSulfide(Material):
    def __init__(self, name='ZnS',
                 elements=['Zn', 'S'],
                 quantities=[1, 1],
                 rho=4.079, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ZirconiumNitride(Material):
    def __init__(self, name='ZrN',
                 elements=['Zr', 'N'],
                 quantities=[1, 1],
                 rho=7.09, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Zirconia(Material):
    def __init__(self, name='ZrO2',
                 elements=['Zr', 'O'],
                 quantities=[1, 2.0],
                 rho=5.68, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class ZirconiumSilicide(Material):
    def __init__(self, name='ZrSi2',
                 elements=['Zr', 'Si'],
                 quantities=[1, 2.0],
                 rho=4.88, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class Air(Material):
    def __init__(self, name='N0.781O0.209Ar0.009',
                 elements=['N', 'O', 'Ar'],
                 quantities=[0.781, 0.209, 0.009],
                 rho=1.20E-06, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)


class CVDDiamond(Material):
    def __init__(self, name='C',
                 elements=['C'],
                 quantities=[1],
                 rho=3.52, *args, **kwargs):
        super().__init__(
           name=name, rho=rho, elements=elements, quantities=quantities,
           *args, **kwargs)

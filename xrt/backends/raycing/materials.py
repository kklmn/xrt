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


from .. import raycing
from .physconsts import CH, CHBAR
from .materials_element import elementsList, Element
from .materials_material import Material
from .materials_multilayer import Multilayer, GradedMultilayer, Coated
from .materials_crystal import Crystal
from .materials_crystals_basic import (
    CrystalFcc, CrystalDiamond, CrystalSi, CrystalFromCell, Powder,
    CrystalHarmonics, MonoCrystal)

ch = CH  # left here for copatibility
chbar = CHBAR  # left here for copatibility


class EmptyMaterial(object):
    """
    This class provides an empty (i.e. without reflectivity) 'grating'
    material. For other kinds of empty materials just use None.
    """

    def __init__(self, kind='grating', **kwargs):
        self.kind = kind
        self.geom = ''
        self.name = ''
        if not hasattr(self, 'uuid'):  # uuid must not change on re-init
            self.uuid = kwargs['uuid'] if 'uuid' in kwargs else\
                str(raycing.uuid.uuid4())

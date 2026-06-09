# -*- coding: utf-8 -*-

import unittest
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt.backends.raycing import materials as rmats

from tests.raycing.auto._property_test_helpers import (
    assert_converted_case, assert_metadata_contract, assert_plain_roundtrip,
    assert_reference_container_roundtrip, assert_reference_roundtrip,
    make_reference_beamline)


METADATA_CLASSES = (
    rmats.Material,
    rmats.Multilayer,
    rmats.Coated,
    rmats.TXMMaterial,
)

MATERIAL_VALUES = {
    'name': 'test material',
    'kind': 'mirror',
    'rho': 2.7,
    't': 0.2,
    'quantities': [1, 2],
}

MULTILAYER_VALUES = {
    'name': 'test multilayer',
    'nPairs': 3,
    'tThickness': 20.,
    'bThickness': 30.,
    'idThickness': 1.5,
    'substThickness': 100.,
    'geom': 'transmitted',
}


class MaterialPropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class MaterialPropertyRoundTripTest(unittest.TestCase):
    def test_material_plain_roundtrip(self):
        material = rmats.Material(
            elements='Si', rho=2.33, kind='plate', name='Si')

        assert_plain_roundtrip(self, material, MATERIAL_VALUES)

    def test_material_refractive_index_converted_roundtrip(self):
        material = rmats.Material(
            elements='Si', rho=2.33, kind='plate', name='Si')

        assert_converted_case(
            self, material, 'refractiveIndex', 1.25,
            {'get': complex(1.25), 'val': complex(1.25), 'raw': 1.25})

    def test_multilayer_plain_roundtrip(self):
        multilayer = rmats.Multilayer(name='ml')

        assert_plain_roundtrip(self, multilayer, MULTILAYER_VALUES)


class MaterialReferencePropertyTest(unittest.TestCase):
    def test_multilayer_layer_reference_forms(self):
        bl, refs = make_reference_beamline()
        multilayer = rmats.Multilayer(name='ml', bl=bl)

        assert_reference_roundtrip(
            self, multilayer, 'tLayer', bl, 'material', refs['water'],
            direct_object_forms=('object', 'uuid', 'name'))
        assert_reference_roundtrip(
            self, multilayer, 'bLayer', bl, 'material', refs['silicon'],
            direct_object_forms=('object', 'uuid', 'name'))
        assert_reference_roundtrip(
            self, multilayer, 'substrate', bl, 'material', refs['water'],
            direct_object_forms=('object', 'uuid', 'name'))

    def test_coated_coating_reference_forms(self):
        bl, refs = make_reference_beamline()
        coated = rmats.Coated(name='coated', bl=bl)

        assert_reference_roundtrip(
            self, coated, 'coating', bl, 'material', refs['silicon'],
            direct_object_forms=('object', 'uuid', 'name'))

    def test_txm_materials_index_reference_forms(self):
        bl, refs = make_reference_beamline()
        txm = rmats.TXMMaterial(name='txm', bl=bl)

        txm.materialsIndex = {
            0: refs['water'],
            1: refs['silicon'].uuid,
            2: 'Water',
        }

        assert_reference_container_roundtrip(
            self, txm, 'materialsIndex', bl, 'material',
            {
                0: refs['water'],
                1: refs['silicon'],
                2: refs['water'],
            })

    def test_txm_materials_index_sequence_forms(self):
        bl, refs = make_reference_beamline()
        txm = rmats.TXMMaterial(name='txm', bl=bl)

        txm.materialsIndex = [refs['water'], refs['silicon'].uuid, 'Water']

        assert_reference_container_roundtrip(
            self, txm, 'materialsIndex', bl, 'material',
            {
                0: refs['water'],
                1: refs['silicon'],
                2: refs['water'],
            })


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-

import math
import unittest
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt.backends import raycing
from xrt.backends.raycing import oes as roes

from tests.raycing.auto._property_test_helpers import (
    assert_converted_case, assert_equivalent, assert_metadata_contract,
    assert_plain_roundtrip, assert_reference_roundtrip,
    make_reference_beamline)


METADATA_CLASSES = (
    roes.OE,
    roes.DCM,
    roes.Plate,
    roes.ToroidMirror,
    roes.BentFlatMirror,
)

OE_PLAIN_VALUES = {
    'center': [1., 2., 3.],
    'roll': 0.01,
    'yaw': 0.02,
    'extraPitch': 0.001,
    'extraRoll': 0.002,
    'extraYaw': 0.003,
    'positionRoll': 0.004,
    'rotationSequence': 'RzRyRx',
    'surface': ('s1', 's2'),
    'alpha': 0.005,
    'limPhysX': [-10., 10.],
    'limPhysY': [-5., 5.],
}

OE_CONVERTED_CASES = (
    ('pitch', '1 mrad',
     {'get': 1e-3, 'val': 1e-3, 'init': '1 mrad', 'raw': None}),
    ('pitch', 'auto',
     {'get': 'auto', 'val': None, 'init': 'auto', 'raw': 'auto'}),
)

DCM_CONVERTED_CASES = (
    ('bragg', '2 mrad',
     {'get': 2e-3, 'val': 2e-3, 'init': '2 mrad', 'raw': None}),
    ('bragg', '8 keV',
     {'get': '8 keV', 'val': None, 'init': '8 keV', 'raw': '8 keV'}),
)


def meridional_radius(p, q, pitch):
    return 2 * p * q / (p + q) / math.sin(abs(pitch))


def sagittal_radius(p, q, pitch):
    return 2 * p * q / (p + q) * math.sin(abs(pitch))


class OEPropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class OEPropertyRoundTripTest(unittest.TestCase):
    def test_oe_plain_roundtrip(self):
        bl = raycing.BeamLine()
        oe = roes.OE(bl=bl, name='oe')

        assert_plain_roundtrip(self, oe, OE_PLAIN_VALUES)

    def test_oe_auto_converted_roundtrip(self):
        oe = roes.OE(name='oe')

        for attr, value, expected in OE_CONVERTED_CASES:
            with self.subTest(attr=attr, value=value):
                assert_converted_case(self, oe, attr, value, expected)

    def test_dcm_bragg_auto_converted_roundtrip(self):
        dcm = roes.DCM(name='dcm')

        for attr, value, expected in DCM_CONVERTED_CASES:
            with self.subTest(attr=attr, value=value):
                assert_converted_case(self, dcm, attr, value, expected)


class OEReferencePropertyTest(unittest.TestCase):
    def test_oe_material_reference_forms(self):
        bl, refs = make_reference_beamline()
        oe = roes.OE(bl=bl, name='oe')

        assert_reference_roundtrip(
            self, oe, 'material', bl, 'material', refs['water'])

    def test_oe_figure_error_reference_forms(self):
        bl, refs = make_reference_beamline()
        oe = roes.OE(bl=bl, name='oe')

        assert_reference_roundtrip(
            self, oe, 'figureError', bl, 'figureError', refs['bump'])

    def test_dcm_material2_reference_forms(self):
        bl, refs = make_reference_beamline()
        dcm = roes.DCM(bl=bl, name='dcm')

        assert_reference_roundtrip(
            self, dcm, 'material2', bl, 'material', refs['silicon'])

    def test_plate_material_reference_forms(self):
        bl, refs = make_reference_beamline()
        plate = roes.Plate(bl=bl, name='plate', t=1.)

        assert_reference_roundtrip(
            self, plate, 'material', bl, 'material', refs['water'],
            direct_object_forms=('object', 'uuid', 'name'))
        assert_reference_roundtrip(
            self, plate, 'material2', bl, 'material', refs['silicon'],
            direct_object_forms=('object', 'uuid', 'name'))


class OECalculatedPropertyTest(unittest.TestCase):
    def test_toroid_tuple_radius_readbacks(self):
        toroid = roes.ToroidMirror(name='toroid', pitch='3 mrad')

        toroid.R = (1000., 2000.)
        toroid.r = (1000., 2000.)

        assert_equivalent(self, toroid._R, [1000., 2000.])
        assert_equivalent(self, toroid._r, [1000., 2000.])
        assert_equivalent(self, toroid.R, meridional_radius(1000., 2000., 3e-3))
        assert_equivalent(self, toroid.r, sagittal_radius(1000., 2000., 3e-3))

    def test_toroid_tuple_radius_updates_when_pitch_changes(self):
        toroid = roes.ToroidMirror(name='toroid', pitch='3 mrad')
        toroid.R = (1000., 2000.)
        toroid.r = (1000., 2000.)

        toroid.pitch = '4 mrad'

        assert_equivalent(self, toroid.R, meridional_radius(1000., 2000., 4e-3))
        assert_equivalent(self, toroid.r, sagittal_radius(1000., 2000., 4e-3))

    def test_toroid_tuple_with_pitch_uses_tuple_pitch(self):
        toroid = roes.ToroidMirror(name='toroid', pitch='3 mrad')

        toroid.R = (1000., 2000., '5 mrad')
        toroid.r = (1000., 2000., '5 mrad')

        assert_equivalent(self, toroid.R, meridional_radius(1000., 2000., 5e-3))
        assert_equivalent(self, toroid.r, sagittal_radius(1000., 2000., 5e-3))


if __name__ == '__main__':
    unittest.main()

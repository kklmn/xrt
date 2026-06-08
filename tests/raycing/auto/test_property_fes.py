# -*- coding: utf-8 -*-

import unittest
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt.backends.raycing import figure_error as rfe

from tests.raycing.auto._property_test_helpers import (
    assert_converted_case, assert_metadata_contract, assert_plain_roundtrip,
    assert_reference_roundtrip, make_reference_beamline)


METADATA_CLASSES = (
    rfe.GaussianBump,
    rfe.Waviness,
    rfe.RandomRoughness,
    rfe.PlanarRidge,
)

FE_BASE_KWARGS = {
    'limPhysX': [-1., 1.],
    'limPhysY': [-1., 1.],
    'gridStep': 2.,
    'skip_build_spline': True,
}

GAUSSIAN_VALUES = {
    'name': 'gaussian test',
    'gridStep': 1.,
    'limPhysX': [-2., 2.],
    'limPhysY': [-3., 3.],
    'bumpHeight': 5.,
    'sigmaX': 0.5,
    'sigmaY': 0.75,
    'cX': 0.1,
    'cY': -0.2,
}

WAVINESS_VALUES = {
    'amplitude': 2.,
    'xWaveLength': 4.,
    'yWaveLength': 5.,
}

ROUGHNESS_VALUES = {
    'rms': 0.5,
    'rmsKind': 'height',
    'corrLength': 1.5,
    'seed': 12345,
}

RIDGE_CONVERTED_CASES = (
    ('slopeAngle', '1 mrad',
     {'get': 1e-3, 'val': 1e-3, 'init': '1 mrad', 'raw': 1e-3}),
    ('orientationAngle', '2 mrad',
     {'get': 2e-3, 'val': 2e-3, 'init': '2 mrad', 'raw': 2e-3}),
)


class FigureErrorPropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class FigureErrorPropertyRoundTripTest(unittest.TestCase):
    def test_gaussian_plain_roundtrip(self):
        fe = rfe.GaussianBump(name='bump', **FE_BASE_KWARGS)

        assert_plain_roundtrip(self, fe, GAUSSIAN_VALUES)

    def test_waviness_plain_roundtrip(self):
        fe = rfe.Waviness(name='wave', **FE_BASE_KWARGS)

        assert_plain_roundtrip(self, fe, WAVINESS_VALUES)

    def test_random_roughness_plain_roundtrip(self):
        fe = rfe.RandomRoughness(name='rough', **FE_BASE_KWARGS)

        assert_plain_roundtrip(self, fe, ROUGHNESS_VALUES)

    def test_planar_ridge_angle_converted_roundtrip(self):
        fe = rfe.PlanarRidge(name='ridge', **FE_BASE_KWARGS)

        for attr, value, expected in RIDGE_CONVERTED_CASES:
            with self.subTest(attr=attr):
                assert_converted_case(self, fe, attr, value, expected)


class FigureErrorReferencePropertyTest(unittest.TestCase):
    def test_base_fe_reference_forms(self):
        bl, refs = make_reference_beamline()
        fe = rfe.Waviness(name='wave', bl=bl, **FE_BASE_KWARGS)

        assert_reference_roundtrip(
            self, fe, 'baseFE', bl, 'figureError', refs['base_fe'])


if __name__ == '__main__':
    unittest.main()

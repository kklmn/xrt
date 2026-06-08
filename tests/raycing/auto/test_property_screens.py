# -*- coding: utf-8 -*-

import unittest
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt.backends import raycing
from xrt.backends.raycing import screens as rscreens

from tests.raycing.auto._property_test_helpers import (
    assert_equivalent, assert_metadata_contract, assert_plain_roundtrip)


METADATA_CLASSES = (
    rscreens.Screen,
    rscreens.HemisphericScreen,
)

SCREEN_VALUES = {
    'center': [1., 2., 3.],
    'compressX': 0.5,
    'compressZ': 0.75,
    'limPhysX': [-10., 10.],
    'limPhysY': [-5., 5.],
    'cLimits': [0., 1.],
    'histShape': [128, 64],
}

SCREEN_ORIENTATION_VALUES = {
    'x': [1., 0., 0.],
    'z': [0., 0., 1.],
}


class ScreenPropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class ScreenPropertyRoundTripTest(unittest.TestCase):
    def test_screen_plain_roundtrip(self):
        screen = rscreens.Screen(bl=raycing.BeamLine(), name='screen')

        assert_plain_roundtrip(self, screen, SCREEN_VALUES)

    def test_screen_orientation_vector_roundtrip(self):
        screen = rscreens.Screen(bl=raycing.BeamLine(), name='screen')

        for attr, value in SCREEN_ORIENTATION_VALUES.items():
            with self.subTest(attr=attr):
                setattr(screen, attr, value)
                assert_equivalent(self, getattr(screen, attr), value)


if __name__ == '__main__':
    unittest.main()

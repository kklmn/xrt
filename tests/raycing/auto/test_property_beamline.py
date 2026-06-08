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

from tests.raycing.auto._property_test_helpers import (
    assert_equivalent, assert_metadata_contract, assert_plain_roundtrip)


METADATA_CLASSES = (
    raycing.BeamLine,
)

BEAMLINE_VALUES = {
    'height': 1.25,
    'alignE': 9000.,
    'name': 'strict beamline',
    'description': 'property roundtrip test',
}

BEAMLINE_INIT_VALUES = dict(BEAMLINE_VALUES, azimuth=0.125)


def assert_azimuth_state(testcase, beamline, expected):
    assert_equivalent(testcase, beamline.azimuth, expected)
    assert_equivalent(testcase, beamline._azimuth, expected)
    assert_equivalent(testcase, beamline.sinAzimuth, math.sin(expected))
    assert_equivalent(testcase, beamline.cosAzimuth, math.cos(expected))


def assert_beamline_values(testcase, beamline, expected):
    for attr, value in expected.items():
        with testcase.subTest(attr=attr):
            assert_equivalent(testcase, getattr(beamline, attr), value)


class BeamLinePropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class BeamLinePropertyRoundTripTest(unittest.TestCase):
    def test_beamline_init_roundtrip(self):
        beamline = raycing.BeamLine(**BEAMLINE_INIT_VALUES)

        assert_beamline_values(self, beamline, BEAMLINE_VALUES)
        assert_azimuth_state(self, beamline, BEAMLINE_INIT_VALUES['azimuth'])

    def test_beamline_post_init_roundtrip(self):
        beamline = raycing.BeamLine()

        assert_plain_roundtrip(self, beamline, BEAMLINE_VALUES)
        beamline.azimuth = 0.375

        assert_azimuth_state(self, beamline, 0.375)

    def test_beamline_starts_with_empty_runtime_collections(self):
        beamline = raycing.BeamLine()

        for attr in (
                'sources', 'oes', 'slits', 'screens', 'alarms'):
            with self.subTest(attr=attr):
                assert_equivalent(self, getattr(beamline, attr), [])

        for attr in (
                'oesDict', 'materialsDict', 'beamsDict', 'fesDict',
                'beamsDictU', 'flowU', 'beamsRevDict', 'beamsRevDictUsed',
                'oenamesToUUIDs', 'matnamesToUUIDs', 'fenamesToUUIDs',
                'beamNamesDict'):
            with self.subTest(attr=attr):
                assert_equivalent(self, dict(getattr(beamline, attr)), {})


if __name__ == '__main__':
    unittest.main()

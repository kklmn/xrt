# -*- coding: utf-8 -*-

import unittest
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

from xrt.backends import raycing
from xrt.backends.raycing import apertures as rapts

from tests.raycing.auto._property_test_helpers import (
    assert_equivalent, assert_metadata_contract, assert_plain_roundtrip)


METADATA_CLASSES = (
    rapts.RectangularAperture,
    rapts.RoundAperture,
    rapts.PolygonalAperture,
)

RECTANGULAR_VALUES = {
    'center': [1., 2., 3.],
    'alarmLevel': 0.25,
    'renderStyle': 'blades',
}

RECTANGULAR_BLADES = {
    'right': 5.,
    'left': -5.,
    'top': 3.,
    'bottom': -3.,
}

POLYGON_VERTICES = [
    (-2., -1.),
    (-2., 1.),
    (2., 1.),
    (2., -1.),
]


class AperturePropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class AperturePropertyRoundTripTest(unittest.TestCase):
    def test_rectangular_plain_roundtrip(self):
        aperture = rapts.RectangularAperture(
            bl=raycing.BeamLine(), name='slit')

        assert_plain_roundtrip(self, aperture, RECTANGULAR_VALUES)

    def test_rectangular_blades_roundtrip(self):
        aperture = rapts.RectangularAperture(
            bl=raycing.BeamLine(), name='slit')

        aperture.blades = RECTANGULAR_BLADES

        assert_equivalent(self, aperture.blades, {
            'left': -5.,
            'right': 5.,
            'bottom': -3.,
            'top': 3.,
        })
        assert_equivalent(
            self, aperture.kind, ['left', 'right', 'bottom', 'top'])
        assert_equivalent(self, aperture.opening, [-5., 5., -3., 3.])

    def test_rectangular_legacy_kind_opening_updates_blades(self):
        aperture = rapts.RectangularAperture(
            bl=raycing.BeamLine(), name='slit', blades=None)

        aperture.kind = ['left', 'right']
        aperture.opening = [-1., 1.]

        assert_equivalent(self, aperture.blades, {'left': -1., 'right': 1.})

    def test_rectangular_opening_dict_updates_blades(self):
        aperture = rapts.RectangularAperture(
            bl=raycing.BeamLine(), name='slit',
            blades={'left': -1., 'right': 1.})

        aperture.opening = {'bottom': -2.}

        assert_equivalent(
            self, aperture.blades,
            {'left': -1., 'right': 1., 'bottom': -2.})

    def test_polygon_vertices_roundtrip(self):
        aperture = rapts.PolygonalAperture(
            bl=raycing.BeamLine(), name='poly')

        aperture.vertices = POLYGON_VERTICES

        assert_equivalent(self, aperture.vertices, POLYGON_VERTICES)
        assert_equivalent(self, aperture.opening, POLYGON_VERTICES)
        assert_equivalent(self, aperture.limOptX, [-2., 2.])
        assert_equivalent(self, aperture.limOptY, [-1., 1.])


if __name__ == '__main__':
    unittest.main()

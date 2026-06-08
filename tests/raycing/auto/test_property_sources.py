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
from xrt.backends.raycing import sources
from xrt.backends.raycing.sources.synchr import K2B

from tests.raycing.auto._property_test_helpers import (
    assert_converted_case, assert_dependent_case, assert_metadata_contract,
    assert_plain_roundtrip)


METADATA_CLASSES = (
    sources.GeometricSource,
    sources.BendingMagnet,
    sources.Wiggler,
    sources.Undulator,
)

GEOMETRIC_SOURCE_VALUES = {
    'center': [1., 2., 3.],
    'nrays': 7,
    'distx': 'flat',
    'dx': 0.5,
    'distz': 'normal',
    'dz': 0.25,
    'distE': 'lines',
    'energies': [8000., 9000.],
    'polarization': 'vertical',
    'filamentBeam': True,
}

SOURCE_CONVERTED_CASES = (
    ('pitch', '2 mrad',
     {'get': 2e-3, 'val': 2e-3, 'init': '2 mrad', 'raw': 2e-3}),
    ('yaw', '3 mrad',
     {'get': 3e-3, 'val': 3e-3, 'init': '3 mrad', 'raw': 3e-3}),
    ('xPrimeMax', '2 mrad', {
        'get': 2.,
        'val': 2e-3,
        'init': '2 mrad',
        'raw': {'_xPrimeMin': -2e-3, '_xPrimeMax': 2e-3}}),
    ('zPrimeMax', [-1., 2.], {
        'get': [-1., 2.],
        'val': 2e-3,
        'init': [-1., 2.],
        'raw': {'_zPrimeMin': -1e-3, '_zPrimeMax': 2e-3}}),
)

DEPENDENT_GROUPS = {
    'emittanceX': {
        'inputs': ('eEpsilonX', 'betaX', 'eSigmaX'),
        'readbacks': ('eSigmaX', 'eSigmaXprime', 'betaX'),
    },
    'emittanceZ': {
        'inputs': ('eEpsilonZ', 'betaZ', 'eSigmaZ'),
        'readbacks': ('eSigmaZ', 'eSigmaZprime', 'betaZ'),
    },
    'bendingField': {
        'inputs': ('B0', 'rho'),
        'readbacks': ('B0', 'rho'),
    },
    'wigglerField': {
        'inputs': ('K', 'B0', 'period'),
        'readbacks': ('K', 'B0', 'rho'),
    },
}

DEPENDENT_GROUP_OBJECTS = {
    'emittanceX': lambda: sources.BendingMagnet(name='bm', nrays=3),
    'emittanceZ': lambda: sources.BendingMagnet(name='bm', nrays=3),
    'bendingField': lambda: sources.BendingMagnet(name='bm', nrays=3),
    'wigglerField': lambda: sources.Wiggler(name='wiggler', nrays=3),
}


class SourcePropertyMetadataTest(unittest.TestCase):
    def test_get_params_contract(self):
        for cls in METADATA_CLASSES:
            with self.subTest(cls=cls.__name__):
                assert_metadata_contract(self, cls)


class SourcePropertyRoundTripTest(unittest.TestCase):
    def test_geometric_source_plain_roundtrip(self):
        bl = raycing.BeamLine()
        source = sources.GeometricSource(bl=bl, name='src', nrays=3)

        assert_plain_roundtrip(self, source, GEOMETRIC_SOURCE_VALUES)

    def test_source_converted_roundtrip(self):
        bl = raycing.BeamLine()
        source = sources.BendingMagnet(bl=bl, name='bm', nrays=3)

        for attr, value, expected in SOURCE_CONVERTED_CASES:
            with self.subTest(attr=attr):
                assert_converted_case(self, source, attr, value, expected)


class SourceDependentPropertyTest(unittest.TestCase):
    def test_dependent_group_catalog_uses_live_properties(self):
        for group_name, group in DEPENDENT_GROUPS.items():
            source = DEPENDENT_GROUP_OBJECTS[group_name]()
            for attr in group['inputs'] + group['readbacks']:
                with self.subTest(group=group_name, attr=attr):
                    self.assertTrue(hasattr(source, attr))

    def test_emittance_x_readbacks(self):
        source = sources.BendingMagnet(name='bm', nrays=3)
        expected_sigma = math.sqrt(1e-6 * 2e3) * 1e3
        expected_prime = 1e-6 / (expected_sigma * 1e-3)

        assert_dependent_case(
            self, source,
            [('eEpsilonX', 1.0), ('betaX', 2.0)],
            {
                'eEpsilonX': 1.0,
                'betaX': 2.0,
                'eSigmaX': expected_sigma,
                'eSigmaXprime': expected_prime,
            })

    def test_emittance_z_readbacks(self):
        source = sources.BendingMagnet(name='bm', nrays=3)
        expected_sigma = math.sqrt(2e-6 * 3e3) * 1e3
        expected_prime = 2e-6 / (expected_sigma * 1e-3)

        assert_dependent_case(
            self, source,
            [('eEpsilonZ', 2.0), ('betaZ', 3.0)],
            {
                'eEpsilonZ': 2.0,
                'betaZ': 3.0,
                'eSigmaZ': expected_sigma,
                'eSigmaZprime': expected_prime,
            })

    def test_bending_magnet_field_readbacks(self):
        source = sources.BendingMagnet(name='bm', nrays=3)
        expected_rho = (
            source.ro if source.B0 == 0 else
            source.rho * source.B0 / 1.5)

        assert_dependent_case(
            self, source,
            [('B0', 1.5)],
            {'B0': 1.5, 'rho': expected_rho})

    def test_wiggler_field_readbacks(self):
        source = sources.Wiggler(name='wiggler', nrays=3, period=40., K=2.)
        expected_k = 1.0 * source.period / K2B

        assert_dependent_case(
            self, source,
            [('B0', 1.0)],
            {'B0': 1.0, 'K': expected_k})


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-

import importlib
import math
import os
import sys

_XRT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
if _XRT_ROOT not in sys.path:
    sys.path.insert(0, _XRT_ROOT)  # analysis:ignore

import numpy as np

from xrt.backends import raycing
from xrt.backends.raycing import figure_error as rfe
from xrt.backends.raycing import materials as rmats


def class_path(cls):
    return '{0}.{1}'.format(cls.__module__, cls.__name__)


def _find_all_arguments(cls):
    module = importlib.import_module(cls.__module__)
    all_arguments = getattr(module, 'allArguments', None)
    if all_arguments is not None:
        return all_arguments

    if '.' in cls.__module__:
        parent_module = importlib.import_module(cls.__module__.rsplit('.', 1)[0])
        return getattr(parent_module, 'allArguments', None)

    return None


def assert_metadata_contract(testcase, cls, require_all_arguments=True):
    path = class_path(cls)
    params = list(raycing.get_params(path))
    names = [name for name, _ in params]

    testcase.assertGreater(
        len(names), 0,
        '{0} exposes no get_params entries'.format(path))

    testcase.assertEqual(
        len(names), len(set(names)),
        '{0} has repeated get_params entries'.format(path))

    hidden = set(getattr(cls, 'hiddenParams', set()))
    for name in hidden:
        testcase.assertNotIn(
            name, names,
            '{0}.{1} is hidden but exposed by get_params'.format(path, name))

    all_arguments = _find_all_arguments(cls)
    if require_all_arguments:
        testcase.assertIsNotNone(
            all_arguments,
            '{0} has no allArguments metadata'.format(path))

    if all_arguments is None:
        return

    ordered_names = [name for name in all_arguments if name in names]
    if names != ordered_names:
        mismatch = next(
            (i for i, pair in enumerate(zip(names, ordered_names))
             if pair[0] != pair[1]),
            min(len(names), len(ordered_names)))
        actual = names[mismatch] if mismatch < len(names) else '<missing>'
        expected = (
            ordered_names[mismatch]
            if mismatch < len(ordered_names) else '<missing>')
        testcase.fail(
            '{0} get_params order does not follow allArguments at '
            'position {1}: actual {2!r}, expected {3!r}\n'
            'actual: {4!r}\nexpected: {5!r}'.format(
                path, mismatch, actual, expected, names, ordered_names))


def make_reference_beamline():
    bl = raycing.BeamLine()

    water = rmats.Material(
        elements='O', quantities=1, rho=1.0, kind='plate', name='Water')
    silicon = rmats.Material(
        elements='Si', rho=2.33, kind='plate', name='Silicon')

    base_fe = rfe.GaussianBump(
        name='Base bump', limPhysX=[-1, 1], limPhysY=[-1, 1],
        gridStep=2)
    bump = rfe.Waviness(
        name='Surface wave', baseFE=base_fe, limPhysX=[-1, 1],
        limPhysY=[-1, 1], gridStep=2, skip_build_spline=True)

    bl.materialsDict[water.uuid] = water
    bl.materialsDict[silicon.uuid] = silicon
    bl.matnamesToUUIDs[water.name] = water.uuid
    bl.matnamesToUUIDs[silicon.name] = silicon.uuid
    for material in (water, silicon):
        material.bl = bl

    bl.fesDict[base_fe.uuid] = base_fe
    bl.fesDict[bump.uuid] = bump
    bl.fenamesToUUIDs[base_fe.name] = base_fe.uuid
    bl.fenamesToUUIDs[bump.name] = bump.uuid
    for fe in (base_fe, bump):
        fe.bl = bl

    return bl, {
        'water': water,
        'silicon': silicon,
        'base_fe': base_fe,
        'bump': bump,
    }


def canonical(value):
    if hasattr(value, '_fields'):
        return [canonical(getattr(value, field)) for field in value._fields]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [canonical(item) for item in value]
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def assert_equivalent(testcase, actual, expected, places=12):
    actual = canonical(actual)
    expected = canonical(expected)

    if isinstance(expected, dict):
        testcase.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in expected:
            assert_equivalent(testcase, actual[key], expected[key], places)
        return

    if isinstance(expected, list):
        testcase.assertIsInstance(actual, list)
        testcase.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            assert_equivalent(testcase, actual_item, expected_item, places)
        return

    if isinstance(expected, complex):
        testcase.assertAlmostEqual(actual.real, expected.real, places=places)
        testcase.assertAlmostEqual(actual.imag, expected.imag, places=places)
        return

    if isinstance(expected, float):
        testcase.assertTrue(
            math.isclose(float(actual), expected, rel_tol=10**(-places),
                         abs_tol=10**(-places)),
            '{0!r} != {1!r}'.format(actual, expected))
        return

    testcase.assertEqual(actual, expected)


def assert_plain_roundtrip(testcase, obj, values):
    for attr, expected in values.items():
        with testcase.subTest(attr=attr):
            setattr(obj, attr, expected)
            assert_equivalent(testcase, getattr(obj, attr), expected)


def assert_converted_case(testcase, obj, attr, value, expected,
                          init_attr=None, val_attr=None, raw_attr=None,
                          expected_raw=None):
    init_attr = init_attr or '_{0}Init'.format(attr)
    val_attr = val_attr or '_{0}Val'.format(attr)
    raw_attr = raw_attr or '_{0}'.format(attr)

    testcase.assertIn(
        'raw', expected,
        '{0} converted case must explicitly check raw backing state'.format(
            attr))

    setattr(obj, attr, value)
    assert_equivalent(testcase, getattr(obj, attr), expected['get'])

    if 'val' in expected:
        assert_equivalent(testcase, getattr(obj, val_attr), expected['val'])
    if 'init' in expected:
        assert_equivalent(testcase, getattr(obj, init_attr), expected['init'])
    expected_raw = expected['raw'] if expected_raw is None else expected_raw
    if isinstance(expected_raw, dict):
        for raw_name, raw_value in expected_raw.items():
            assert_equivalent(testcase, getattr(obj, raw_name), raw_value)
    else:
        assert_equivalent(testcase, getattr(obj, raw_attr), expected_raw)


def assert_reference_roundtrip(testcase, obj, attr, bl, ref_kind, target,
                               direct_object_forms=('object', 'uuid')):
    forms = {
        'object': target,
        'uuid': target.uuid,
        'name': target.name,
    }
    for form_name, value in forms.items():
        with testcase.subTest(attr=attr, form=form_name):
            setattr(obj, attr, value)
            readback = getattr(obj, attr)
            normalized = raycing.normalize_ref(
                readback, bl=bl, refKind=ref_kind, target='object')
            testcase.assertIs(normalized, target)
            if form_name in direct_object_forms:
                testcase.assertIs(readback, target)


def assert_reference_container_roundtrip(testcase, obj, attr, bl, ref_kind,
                                         expected):
    readback = getattr(obj, attr)
    normalized = raycing.normalize_ref(
        readback, bl=bl, refKind=ref_kind, target='object')
    testcase.assertEqual(set(normalized), set(expected))
    for key, target in expected.items():
        testcase.assertIs(normalized[key], target)


def assert_dependent_case(testcase, obj, assignments, expected):
    for attr, value in assignments:
        setattr(obj, attr, value)
    for attr, value in expected.items():
        with testcase.subTest(attr=attr):
            assert_equivalent(testcase, getattr(obj, attr), value)

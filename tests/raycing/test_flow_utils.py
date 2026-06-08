# -*- coding: utf-8 -*-

import unittest
from collections import OrderedDict

from xrt.backends.raycing._flow_utils import (
    create_paramdict_fe, create_paramdict_mat, create_paramdict_oe,
    get_init_kwargs, normalize_ref, ref_kind_for_arg)


WATER_UUID = '11111111-1111-4111-8111-111111111111'
SILICON_UUID = '22222222-2222-4222-8222-222222222222'
FE_UUID = '33333333-3333-4333-8333-333333333333'
BASE_FE_UUID = '44444444-4444-4444-8444-444444444444'
OE_UUID = '55555555-5555-4555-8555-555555555555'


class RefObject:
    def __init__(self, name, uuid):
        self.name = name
        self.uuid = uuid


class FakeBeamLine:
    def __init__(self):
        self.water = RefObject('Water', WATER_UUID)
        self.silicon = RefObject('Silicon', SILICON_UUID)
        self.figureError = RefObject('Figure map', FE_UUID)
        self.baseFE = RefObject('Base figure map', BASE_FE_UUID)
        self.oe = RefObject('Mirror', OE_UUID)

        self.materialsDict = OrderedDict([
            (WATER_UUID, self.water),
            (SILICON_UUID, self.silicon),
        ])
        self.matnamesToUUIDs = {
            self.water.name: WATER_UUID,
            self.silicon.name: SILICON_UUID,
        }
        self.fesDict = OrderedDict([
            (FE_UUID, self.figureError),
            (BASE_FE_UUID, self.baseFE),
        ])
        self.fenamesToUUIDs = {
            self.figureError.name: FE_UUID,
            self.baseFE.name: BASE_FE_UUID,
        }
        self.oesDict = OrderedDict([(OE_UUID, [self.oe, 1])])
        self.oenamesToUUIDs = {self.oe.name: OE_UUID}


class FakeOEForInitKwargs:
    def __init__(self, name='', material=None, material2=None,
                 figureError=None, center=None, beam=None, bl=None):
        self.name = name
        self.material = material
        self.material2 = material2
        self.figureError = figureError
        self.center = center
        self.beam = beam
        self.bl = bl


class FakeMaterialForInitKwargs:
    def __init__(self, name='', tLayer=None, bLayer=None, coating=None,
                 substrate=None, materialsIndex=None):
        self.name = name
        self.tLayer = tLayer
        self.bLayer = bLayer
        self.coating = coating
        self.substrate = substrate
        self.materialsIndex = materialsIndex


class FakeFEForInitKwargs:
    def __init__(self, name='', baseFE=None):
        self.name = name
        self.baseFE = baseFE


class ReferenceNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.bl = FakeBeamLine()

    def test_ref_kind_for_arg_recognizes_reference_fields(self):
        material_fields = [
            'material', 'material2', 'tLayer', 'bLayer', 'coating',
            'substrate', 'materialsIndex']
        for field in material_fields:
            self.assertEqual(ref_kind_for_arg(field), 'material')

        for field in ['figureError', 'baseFE']:
            self.assertEqual(ref_kind_for_arg(field), 'figureError')

        for field in ['beam', 'beamLocal', 'center', 'limPhysX']:
            self.assertIsNone(ref_kind_for_arg(field))

    def test_normalize_ref_converts_material_refs_recursively(self):
        value = OrderedDict([
            ('same-key', self.bl.water),
            ('name', 'Silicon'),
            ('uuid', WATER_UUID),
            ('none', None),
            ('unknown', 'Unknown material'),
            ('nested', [self.bl.silicon, 'Water']),
        ])

        normalized = normalize_ref(
            value, self.bl, 'material', target='uuid')

        self.assertEqual(list(normalized.keys()), list(value.keys()))
        self.assertEqual(normalized['same-key'], WATER_UUID)
        self.assertEqual(normalized['name'], SILICON_UUID)
        self.assertEqual(normalized['uuid'], WATER_UUID)
        self.assertIsNone(normalized['none'])
        self.assertEqual(normalized['unknown'], 'Unknown material')
        self.assertEqual(normalized['nested'], [SILICON_UUID, WATER_UUID])

    def test_normalize_ref_can_make_display_names_and_objects(self):
        self.assertEqual(
            normalize_ref(WATER_UUID, self.bl, 'material', 'display'),
            'Water')
        self.assertEqual(
            normalize_ref(self.bl.figureError, self.bl, 'figureError',
                          'display'),
            'Figure map')
        self.assertIs(
            normalize_ref('Mirror', self.bl, 'oe', 'object'),
            self.bl.oe)
        self.assertIs(
            normalize_ref(OE_UUID, self.bl, 'oe', 'object'),
            self.bl.oe)

    def test_normalize_ref_rejects_unknown_kind_or_target(self):
        with self.assertRaises(KeyError):
            normalize_ref('Water', self.bl, 'beam', 'uuid')
        with self.assertRaises(ValueError):
            normalize_ref('Water', self.bl, 'material', 'path')


class CreateParamdictTest(unittest.TestCase):
    def setUp(self):
        self.bl = FakeBeamLine()

    def test_create_paramdict_oe_normalizes_references_but_not_beams(self):
        def_args = {
            'material': None,
            'material2': None,
            'figureError': None,
            'center': None,
            'beam': None,
            'bl': None,
        }
        param_dict = {
            'material': 'Water',
            'material2': '[Water, Silicon]',
            'figureError': 'Figure map',
            'center': '[1, 2, auto]',
            'beam': 'screen_global',
            'bl': 'bl',
        }

        kwargs = create_paramdict_oe(param_dict, def_args, self.bl)

        self.assertEqual(kwargs['material'], WATER_UUID)
        self.assertEqual(kwargs['material2'], [WATER_UUID, SILICON_UUID])
        self.assertEqual(kwargs['figureError'], FE_UUID)
        self.assertEqual(kwargs['center'], [1, 2, 'auto'])
        self.assertEqual(kwargs['beam'], 'screen_global')
        self.assertIs(kwargs['bl'], self.bl)

    def test_create_paramdict_mat_normalizes_layers_and_materials_index(self):
        def_args = {
            'tLayer': None,
            'bLayer': None,
            'substrate': None,
            'materialsIndex': None,
            'rho': 0,
        }
        param_dict = {
            'tLayer': 'Water',
            'bLayer': SILICON_UUID,
            'substrate': 'None',
            'materialsIndex':
                "{0: 'Water', 1: '%s', 2: None}" % SILICON_UUID,
            'rho': '2.33',
        }

        kwargs = create_paramdict_mat(param_dict, def_args, self.bl)

        self.assertEqual(kwargs['tLayer'], WATER_UUID)
        self.assertEqual(kwargs['bLayer'], SILICON_UUID)
        self.assertNotIn('substrate', kwargs)
        self.assertEqual(kwargs['materialsIndex'], {
            0: WATER_UUID,
            1: SILICON_UUID,
            2: None,
        })
        self.assertEqual(kwargs['rho'], 2.33)

    def test_create_paramdict_fe_normalizes_base_fe(self):
        kwargs = create_paramdict_fe(
            {'baseFE': 'Base figure map', 'scale': '3.5'},
            {'baseFE': None, 'scale': 1.},
            self.bl)

        self.assertEqual(kwargs['baseFE'], BASE_FE_UUID)
        self.assertEqual(kwargs['scale'], 3.5)


class GetInitKwargsTest(unittest.TestCase):
    def setUp(self):
        self.bl = FakeBeamLine()

    def test_get_init_kwargs_serializes_oe_references_to_uuids(self):
        obj = FakeOEForInitKwargs(
            name='oe',
            material=self.bl.water,
            material2=[self.bl.water, self.bl.silicon],
            figureError=self.bl.figureError,
            center=[0, 0, 0],
            beam='screen_global',
            bl=self.bl)

        kwargs = get_init_kwargs(obj, compact=False, blname='bl')

        self.assertEqual(kwargs['material'], WATER_UUID)
        self.assertEqual(kwargs['material2'], str([WATER_UUID, SILICON_UUID]))
        self.assertEqual(kwargs['figureError'], FE_UUID)
        self.assertEqual(kwargs['beam'], 'screen_global')
        self.assertEqual(kwargs['bl'], 'bl')

    def test_get_init_kwargs_serializes_material_references_to_uuids(self):
        obj = FakeMaterialForInitKwargs(
            name='ml',
            tLayer=self.bl.water,
            bLayer='Silicon',
            substrate=None,
            materialsIndex={
                0: self.bl.water,
                1: 'Silicon',
                2: None,
            })
        obj.bl = self.bl

        kwargs = get_init_kwargs(obj, compact=False)

        self.assertEqual(kwargs['tLayer'], WATER_UUID)
        self.assertEqual(kwargs['bLayer'], SILICON_UUID)
        self.assertEqual(kwargs['substrate'], 'None')
        self.assertEqual(kwargs['materialsIndex'], str({
            0: WATER_UUID,
            1: SILICON_UUID,
            2: None,
        }))

    def test_get_init_kwargs_serializes_figure_error_references_to_uuids(self):
        obj = FakeFEForInitKwargs(name='fe', baseFE=self.bl.baseFE)

        kwargs = get_init_kwargs(obj, compact=False)

        self.assertEqual(kwargs['baseFE'], BASE_FE_UUID)


if __name__ == '__main__':
    unittest.main()

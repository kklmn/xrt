# -*- coding: utf-8 -*-

import unittest
from collections import OrderedDict

from xrt.backends.raycing._flow import MessageHandler
from xrt.backends.raycing.beamline import BeamLine


WATER_UUID = '11111111-1111-4111-8111-111111111111'
SILICON_UUID = '22222222-2222-4222-8222-222222222222'
TXM_UUID = '33333333-3333-4333-8333-333333333333'
FE_UUID = '44444444-4444-4444-8444-444444444444'
BASE_FE_UUID = '55555555-5555-4555-8555-555555555555'
OE_UUID = '66666666-6666-4666-8666-666666666666'
HOLDER_UUID = '77777777-7777-4777-8777-777777777777'


class RefObject:
    def __init__(self, name, uuid, **kwargs):
        self.name = name
        self.uuid = uuid
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeOE:
    def __init__(self, name='oe', uuid=OE_UUID, material=None,
                 material2=None, figureError=None):
        self.name = name
        self.uuid = uuid
        self.material = material
        self.material2 = material2
        self.figureError = figureError
        self._material = material
        self._material2 = material2
        self._figureError = figureError


def make_beamline():
    bl = BeamLine()
    bl.water = RefObject('Water', WATER_UUID)
    bl.silicon = RefObject('Silicon', SILICON_UUID)
    bl.txm = RefObject('TXM', TXM_UUID)
    bl.figureError = RefObject('Figure map', FE_UUID)
    bl.baseFE = RefObject('Base figure map', BASE_FE_UUID)

    bl.materialsDict = OrderedDict([
        (WATER_UUID, bl.water),
        (SILICON_UUID, bl.silicon),
        (TXM_UUID, bl.txm),
    ])
    bl.matnamesToUUIDs = {
        bl.water.name: WATER_UUID,
        bl.silicon.name: SILICON_UUID,
        bl.txm.name: TXM_UUID,
    }
    bl.fesDict = OrderedDict([
        (FE_UUID, bl.figureError),
        (BASE_FE_UUID, bl.baseFE),
    ])
    bl.fenamesToUUIDs = {
        bl.figureError.name: FE_UUID,
        bl.baseFE.name: BASE_FE_UUID,
    }
    bl.oesDict = OrderedDict()
    return bl


class BeamlineReferenceTest(unittest.TestCase):
    def test_sort_materials_resolves_names_and_materials_index_values(self):
        bl = make_beamline()
        bl.materialsDict = OrderedDict([
            (TXM_UUID, bl.txm),
            (WATER_UUID, bl.water),
            (SILICON_UUID, bl.silicon),
        ])
        bl.txm.materialsIndex = {
            0: 'Water',
            1: SILICON_UUID,
            2: bl.silicon,
        }

        self.assertEqual(
            bl.sort_materials(),
            [WATER_UUID, SILICON_UUID, TXM_UUID])

    def test_sort_figerrors_resolves_base_fe_name(self):
        bl = make_beamline()
        bl.fesDict = OrderedDict([
            (FE_UUID, bl.figureError),
            (BASE_FE_UUID, bl.baseFE),
        ])
        bl.figureError.baseFE = 'Base figure map'

        self.assertEqual(
            bl.sort_figerrors(),
            [BASE_FE_UUID, FE_UUID])

    def test_delete_material_clears_uuid_name_and_container_refs(self):
        bl = make_beamline()
        oe = FakeOE(material=WATER_UUID,
                    material2=[SILICON_UUID, 'Water'])
        holder = RefObject(
            'Holder', HOLDER_UUID,
            tLayer='Water',
            bLayer=SILICON_UUID,
            materialsIndex={0: WATER_UUID, 1: 'Silicon'})
        bl.oesDict[OE_UUID] = [oe, 1]
        bl.materialsDict[holder.uuid] = holder

        bl.delete_mat_by_id(WATER_UUID)

        self.assertNotIn(WATER_UUID, bl.materialsDict)
        self.assertNotIn('Water', bl.matnamesToUUIDs)
        self.assertIsNone(oe.material)
        self.assertEqual(oe.material2, [SILICON_UUID, None])
        self.assertIsNone(holder.tLayer)
        self.assertEqual(holder.bLayer, SILICON_UUID)
        self.assertEqual(holder.materialsIndex, {0: None, 1: 'Silicon'})

    def test_delete_figure_error_clears_uuid_and_name_refs(self):
        bl = make_beamline()
        oe = FakeOE(figureError=FE_UUID)
        bl.oesDict[OE_UUID] = [oe, 1]
        bl.figureError.baseFE = 'Base figure map'

        bl.delete_fe_by_id(BASE_FE_UUID)
        self.assertNotIn(BASE_FE_UUID, bl.fesDict)
        self.assertNotIn('Base figure map', bl.fenamesToUUIDs)
        self.assertIsNone(bl.figureError.baseFE)

        bl.delete_fe_by_id(FE_UUID)
        self.assertIsNone(oe.figureError)

    def test_index_materials_registers_nested_refs(self):
        bl = make_beamline()
        holder = RefObject(
            'Holder', HOLDER_UUID,
            tLayer='Water',
            materialsIndex={0: 'Silicon'})
        bl.figureError.baseFE = 'Base figure map'
        oe = FakeOE(material=holder, figureError=bl.figureError)
        bl.oesDict[OE_UUID] = [oe, 1]

        bl.index_materials()

        self.assertEqual(oe.material, HOLDER_UUID)
        self.assertEqual(holder.tLayer, WATER_UUID)
        self.assertIn(HOLDER_UUID, bl.materialsDict)
        self.assertIn(WATER_UUID, bl.materialsDict)
        self.assertIn(SILICON_UUID, bl.materialsDict)
        self.assertEqual(oe.figureError, FE_UUID)
        self.assertEqual(bl.figureError.baseFE, BASE_FE_UUID)


class FlowReferenceTest(unittest.TestCase):
    def test_modify_material_restarts_oe_that_refs_material_object(self):
        bl = make_beamline()
        oe = FakeOE(material=bl.water)
        bl.oesDict[OE_UUID] = [oe, 1]
        handler = MessageHandler(bl)

        handler.handle_modify({
            'object_type': 'mat',
            'uuid': WATER_UUID,
            'kwargs': {'name': 'Water updated'},
        })

        self.assertEqual(handler.startEl, OE_UUID)
        self.assertTrue(handler.needUpdate)
        self.assertEqual(bl.water.name, 'Water updated')

    def test_delete_fe_restarts_oe_that_refs_figure_error_object(self):
        bl = make_beamline()
        oe = FakeOE(figureError=bl.figureError)
        bl.oesDict[OE_UUID] = [oe, 1]
        handler = MessageHandler(bl)

        handler.handle_delete({
            'object_type': 'fe',
            'uuid': FE_UUID,
        })

        self.assertEqual(handler.startEl, OE_UUID)
        self.assertTrue(handler.needUpdate)


if __name__ == '__main__':
    unittest.main()

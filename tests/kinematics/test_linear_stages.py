# -*- coding: utf-8 -*-

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from xrt.backends.raycing._rotate import rotate_point
from xrt.backends.raycing.stages import LinearStage, Pose
from xrt.gui.xrtKinematics.model import MountedBox, make_aperture_demo


class LinearStageTest(unittest.TestCase):
    def test_tilted_axis_and_return_to_zero_preserve_installation(self):
        stage = LinearStage(center=(4, 5, 6), x=(0, 2, 0), z=(0, 0, 3),
                            mountingPoint0=(1, 2, 3), mountingOffset=(2, 0, 0))
        original = stage.output_matrix
        installation = tuple(vector.copy() for vector in (
            stage.center, stage.x, stage.z, stage.mountingPoint0))
        base = stage.evaluate().base.copy()
        for travel in (1, -4, 10, -2, 0):
            stage.position = travel
            expected = original.copy()
            expected[:3, 3] += (0, travel, 0)
            assert_allclose(stage.output_matrix, expected, atol=1e-12)
            assert_allclose(stage.evaluate().base, base)
        self.assertEqual(installation, (
            stage.center, stage.x, stage.z, stage.mountingPoint0))

    def test_component_edits_update_the_stack_after_whole_vector_assignment(self):
        center = np.array([1., 2., 3.])
        root = LinearStage(center=center)
        child = LinearStage(baseStage=root, axis='z', position=1)
        root.center[2] = 5
        root.mountingPoint0[1] = 2
        assert_allclose(center, (1, 2, 3))
        assert_allclose(child.output_pose().center, (1, 4, 6))

        root.center = (4, 5, 6)
        self.assertIs(type(root.center), list)
        root.center[0] = 7
        root.mountingOffset[2] = 2
        assert_allclose(child.output_pose().center, (7, 7, 9))

    def test_mounting_orientation_carries_offset_and_next_stage_axes(self):
        root = LinearStage(position=2, mountingPoint0=(0, 0, 1),
                           mountingX0=(0, 1, 0), mountingZ0=(0, 0, 1),
                           mountingOffset=(1, 0, 0))
        child = LinearStage(baseStage=root, center=(0, 0, 2), position=3)
        assert_allclose(root.mounting_pose().center, (2, 0, 1))
        assert_allclose(root.output_pose().center, (2, 1, 1))
        assert_allclose(child.output_pose().center, (2, 4, 3))
        self.assertAlmostEqual(child.output_pose().yaw, math.pi / 2)
        root.position = 7
        assert_allclose(child.output_pose().center, (7, 4, 3))
        self.assertEqual(child.center, [0, 0, 2])
        self.assertEqual(child.x, [1, 0, 0])

    def test_cycles_are_rejected_without_changing_the_original_stack(self):
        root = LinearStage()
        child = LinearStage(baseStage=root)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            root.baseStage = child
        self.assertIsNone(root.baseStage)
        self.assertIs(child.baseStage, root)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            child.baseStage = child

    def test_invalid_geometries_and_nonfinite_travel_are_rejected(self):
        for kwargs in ({'x': (0, 0, 0)}, {'x': (0, 0, 1)},
                       {'center': (1, float('nan'), 2)}, {'axis': 'q'},
                       {'position': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LinearStage(**kwargs)

    def test_euler_readout_reconstructs_xrt_default_oe_rotation(self):
        for angles in ((.2, -.3, .7), (-.9, .6, -2.),
                       (.4, math.pi / 2, .8), (.4, -math.pi / 2, .8)):
            matrix = np.eye(4)
            matrix[:3, :3] = np.column_stack([
                rotate_point(axis, rotationSequence='-RzRyRx',
                             pitch=angles[0], roll=angles[1], yaw=angles[2])
                for axis in np.eye(3)])
            pose = Pose.from_matrix(matrix)
            reconstructed = np.column_stack([
                rotate_point(axis, rotationSequence='-RzRyRx',
                             pitch=pose.pitch, roll=pose.roll, yaw=pose.yaw)
                for axis in np.eye(3)])
            assert_allclose(reconstructed, matrix[:3, :3], atol=1e-12)

    def test_mounted_box_origin_is_its_center_and_pose_is_read_only(self):
        box = MountedBox('oe', LinearStage(position=3), size=(2, 4, 6))
        assert_allclose(box.vertices().mean(axis=0), box.center)
        for attribute in ('center', 'pitch', 'roll', 'yaw'):
            with self.assertRaises(AttributeError):
                setattr(box, attribute, 0)


class ApertureStageTest(unittest.TestCase):
    def setUp(self):
        self.demo = make_aperture_demo()
        self.stages = {stage.name: stage for stage in self.demo.stages}

    def test_zero_geometry_and_shared_parent(self):
        self.assertEqual(self.demo.opening(),
                         dict(left=-2., right=2., bottom=-2., top=2.))
        assert_allclose(self.demo.pose().center, (0, 0, 0))
        for blade in self.demo.blades.values():
            self.assertIs(blade.mount.baseStage, self.stages['apertureZ'])

    def test_each_blade_moves_independently(self):
        for side in ('left', 'right', 'bottom', 'top'):
            with self.subTest(side=side):
                before = {key: box.vertices() for key, box in self.demo.blades.items()}
                expected = self.demo.opening()
                self.stages[side].position = .5
                expected[side] += .5
                self.assertEqual(self.demo.opening(), expected)
                assert_allclose(self.demo.pose().center, (0, 0, 0))
                for other, vertices in before.items():
                    if other != side:
                        assert_allclose(self.demo.blades[other].vertices(), vertices)
                self.stages[side].position = 0

    def test_one_blade_changes_opening_center_without_moving_aperture_origin(self):
        self.stages['left'].position = .5
        edges = self.demo.opening()
        self.assertEqual(edges['right'] - edges['left'], 3.5)
        assert_allclose(self.demo.opening_center(), (.25, 0, 0))
        assert_allclose(self.demo.pose().center, (0, 0, 0))

    def test_whole_aperture_motion_preserves_local_opening_and_blade_settings(self):
        self.stages['left'].position = .5
        before = {key: box.vertices() for key, box in self.demo.blades.items()}
        edges = self.demo.opening()
        self.stages['apertureX'].position = 1.25
        self.stages['apertureZ'].position = -.75
        delta = (1.25, 0, -.75)
        self.assertEqual(self.demo.opening(), edges)
        for side, vertices in before.items():
            assert_allclose(self.demo.blades[side].vertices(), vertices + delta)
        assert_allclose(self.demo.pose().center, delta)
        assert_allclose(self.demo.opening_center(), (1.5, 0, -.75))
        self.assertEqual(self.stages['left'].position, .5)

    def test_rotated_whole_stage_carries_all_blades_and_local_travel_axes(self):
        self.stages['apertureX'].x = (0, 1, 0)
        origin = np.array(self.demo.pose().center)
        self.stages['apertureX'].position = 2
        self.stages['apertureZ'].position = 3
        assert_allclose(self.demo.pose().center, origin + (0, 2, 3))
        self.assertEqual(self.demo.opening(),
                         dict(left=-2., right=2., bottom=-2., top=2.))
        for box in self.demo.blades.values():
            self.assertAlmostEqual(box.yaw, math.pi / 2)

    def test_an_extra_stage_can_be_inserted_in_a_single_blade_stack(self):
        left = self.stages['left']
        extra = LinearStage(baseStage=left.baseStage, position=.5)
        left.baseStage = extra
        self.demo.stages.append(extra)
        self.assertEqual(self.demo.opening(),
                         dict(left=-1.5, right=2., bottom=-2., top=2.))

    def test_closed_aperture_does_not_clamp_or_move_motors(self):
        self.stages['left'].position = 5
        edges = self.demo.opening()
        self.assertEqual(edges['right'] - edges['left'], -1)
        self.assertEqual(self.stages['left'].position, 5)

    def test_nonrectangular_blade_orientation_has_no_misleading_gap_readout(self):
        self.stages['left'].mountingX0 = (0, 1, 0)
        with self.assertRaisesRegex(ValueError, 'align'):
            self.demo.opening()
        self.assertEqual(self.demo.blades['left'].vertices().shape, (8, 3))


if __name__ == '__main__':
    unittest.main()

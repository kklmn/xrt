# -*- coding: utf-8 -*-
"""Build Glow stage meshes from the mechanical mount hierarchy."""

from collections import defaultdict
from itertools import product

import numpy as np

from ....backends import raycing
from ....backends.raycing.stages import (
    LinearStage, RotationStage, transform_points)
from .._utils import is_oe


_CUBE = np.array(list(product((-0.5, 0.5), repeat=3)))
_DISK = np.array([
    (0.5 * np.cos(angle), 0.5 * np.sin(angle), z)
    for angle in np.linspace(0, 2 * np.pi, 60, endpoint=False)
    for z in (-0.5, 0.5)
])
_AXIAL_PLANES = ((1, 2, 0), (2, 0, 1), (0, 1, 2))


def _local_points(pose, points):
    return (points - pose[:3, 3]) @ pose[:3, :3]


def _primitive_points(index, model):
    if isinstance(index, tuple):
        # Sample the same arc that closes against the chord in the mesh.
        half_arc = np.arccos(2 * index[1])
        angles = np.pi / 2 + np.linspace(-half_arc, half_arc, 61)
        points = np.column_stack((
            0.5 * np.repeat(np.cos(angles), 2),
            0.5 * np.repeat(np.sin(angles), 2),
            np.tile((-0.5, 0.5), len(angles))))
    else:
        points = _DISK if index == 1 else _CUBE
    return transform_points(model, points)


def _extent(value, auto):
    """Resolve a centered span or local limits against an automatic extent."""
    if value is None:
        center = np.mean(auto)
        span = max(auto[1] - auto[0], 0.1)
        return np.array((center - span / 2, center + span / 2))
    if np.isscalar(value):
        return np.array((-value / 2, value / 2))
    return np.asarray(value, dtype=float)


def _plate(pose, limits, face, height, front):
    model = pose.copy()
    xy = limits.mean(axis=0)
    model[:3, 3] = transform_points(pose, (*xy, face - front * height / 2))
    model[:3, :3] *= (*(limits[1] - limits[0]), height)
    return model


def _front(pose, normals):
    # A mixed assembly uses the backing direction of the stage itself.
    return -1 if normals and all(
        normal @ pose[:3, 2] < -1e-6 for normal in normals) else 1


def _linear_plates(stage, base, mount, bounds, optic):
    sizes = stage.renderSize
    axis = 'xyz'.index(stage.axis)
    along = axis if axis < 2 else 1
    across = 1 - along
    grooves = sizes['grooves']
    if grooves is None:
        grooves = int(axis < 2 or stage.renderStyle == 'grooved')
    if stage.renderStyle == 'plates':
        grooves = 0

    # The plates face local +Z. A child stage sits at the attachment plane.
    face = bounds[0, 2] if optic is not None else 0.

    limits = bounds[:, :2].copy()
    limits[:, along] = _extent(sizes['topLength'], limits[:, along])
    limits[:, across] = _extent(sizes['width'], limits[:, across])
    top_height = sizes['topHeight']
    yield 0, _plate(mount, limits, face, top_height, 1)

    # The linear mount differs from the base by the actuator travel.
    base_limits = limits.copy()
    scale = raycing.allUnitsLen[stage.positionUnit]
    travel = scale * np.array((stage.positionMin or 0., stage.positionMax or 0.))
    gap = sizes['gap']
    base_face = face - top_height - gap
    if axis == 2:
        base_face += travel[0]
    else:
        base_limits[:, along] += np.sort(travel)
    base_limits[:, along] = _extent(
        sizes['bottomLength'], base_limits[:, along])

    depth = sizes['grooveDepth']
    width = sizes['grooveWidth']
    height = sizes['bottomHeight']
    spacing = (limits[1, across] - limits[0, across]) / (grooves + 1)
    if (spacing <= width * (1 if grooves > 1 else 0.5) or
            width <= 2 * gap or (height is not None and height <= depth)):
        grooves = 0
    height = height if height is not None else (2. + depth if grooves else 2.)
    groove_depth = depth if grooves else 0.
    base_top = base_face - groove_depth
    yield 0, _plate(base, base_limits, base_top, height - groove_depth, 1)

    if axis == 2:
        extension = stage.position_to_base_units(stage.position) - travel[0]
        cylinder_height = gap + extension + groove_depth
        # Size the cylinder to the footprint shared by the two plates.
        lower = np.maximum(limits[0], base_limits[0])
        upper = np.minimum(limits[1], base_limits[1])
        radius = .75 * min(upper - lower) / 2
        if cylinder_height > 0. and radius > 0.:
            center = (lower + upper) / 2
            cylinder_limits = np.array((center - radius, center + radius))
            yield 1, _plate(base, cylinder_limits, base_top, cylinder_height, -1)

    if grooves:
        centers = limits[0, across] + spacing * np.arange(1, grooves + 1)
        tongue = limits.copy()
        tip_face = face - top_height
        for center in centers:
            half_tongue = width / 2 - gap
            tongue[:, across] = center - half_tongue, center + half_tongue
            yield 0, _plate(mount, tongue, tip_face, depth, 1)

        edges = np.r_[base_limits[0, across],
                      np.column_stack((centers - width / 2,
                                       centers + width / 2)).ravel(),
                      base_limits[1, across]]
        shoulder = base_limits.copy()
        for lower, upper in edges.reshape(-1, 2):
            shoulder[:, across] = lower, upper
            yield 0, _plate(base, shoulder, base_face, depth, 1)


def _disk_plates(stage, base, mount, points, normals, axis, optic):
    # The disk's thickness axis is the rotation axis, whatever the actuator.
    permutation = _AXIAL_PLANES[axis]
    disk_base, disk_mount = base.copy(), mount.copy()
    disk_base[:3, :3] = base[:3, permutation]
    disk_mount[:3, :3] = mount[:3, permutation]
    front = _front(disk_mount, normals)
    local = _local_points(disk_mount, points)
    back = 0.
    if optic is not None:
        back = local[:, 2].min() if front > 0 else local[:, 2].max()
    sizes = stage.renderSize
    bearing_radius = sizes['radius']
    radius = (max(min(np.ptp(local[:, :2], axis=0)), 0.1)
              if bearing_radius is None else bearing_radius)
    pivot = np.asarray(stage.pivotPoint)[list(permutation[:2])]
    limits = np.column_stack((
        _extent(sizes['width'], pivot[0] + np.array((-radius, radius))),
        _extent(sizes['length'], pivot[1] + np.array((-radius, radius)))))
    top_index = 0 if sizes['shape'] == 'square' else 1
    height = sizes['topHeight']
    yield top_index, _plate(disk_mount, limits, back, height, front)
    if bearing_radius is None:
        bearing_radius = min(limits[1] - limits[0]) / 2
    bearing = np.array((pivot - bearing_radius, pivot + bearing_radius))
    yield 1, _plate(disk_base, bearing, back - front * height,
                    sizes['bottomHeight'], front)


def _cradle_plates(stage, base, mount, points, normals, axis, optic):
    # Use the optic's backing plane if it is parallel to the pivot axis.
    axial = mount[:3, axis]
    if optic is not None and abs(optic[1][:3, 2] @ axial) < 1e-6:
        bounds, pose = optic
        front = 1
        back = bounds[0, 2]
    else:
        pose = mount.copy()
        if axis == 2:
            pose[:3, :3] = mount[:3, _AXIAL_PLANES[0]]
        local = _local_points(pose, points)
        bounds = np.array((local.min(axis=0), local.max(axis=0)))
        front = _front(pose, normals)
        back = bounds[0 if front > 0 else 1, 2] if optic is not None else 0.
    sizes = stage.renderSize
    limits = np.column_stack((
        _extent(sizes['width'], bounds[:, 0]),
        _extent(sizes['length'], bounds[:, 1])))
    height = sizes['topHeight']
    index = int(sizes['shape'] == 'round')
    top = _plate(pose, limits, back, height, front)
    yield index, top

    pivot = transform_points(base, stage.pivotPoint)
    backing = -front * pose[:3, 2]
    tangent = np.cross(backing, axial)
    tangent /= np.linalg.norm(tangent)
    backing = np.cross(axial, tangent)
    distance = (top[:3, 3] - pivot) @ backing + height / 2
    plate_points = _primitive_points(index, top) - pivot
    along = plate_points @ axial
    across = plate_points @ tangent
    half_chord = min(-across.min(), across.max())
    if half_chord <= 0:
        half_chord = np.ptp(across) / 2
    radius = sizes['radius']
    if radius is None:
        radius = np.hypot(distance, half_chord)
    if radius <= abs(distance):
        return

    support = np.eye(4)
    support[:3, :3] = np.column_stack((tangent, backing, axial))
    support[:3, :3] *= (2 * radius, 2 * radius, np.ptp(along))
    support[:3, 3] = pivot + np.mean((along.min(), along.max())) * axial
    offset = round(float(distance / (2 * radius)), 12)
    if abs(offset) < 0.5:
        yield ('cradle', offset), support


def _tube_triangles(centers, rings, closed=False):
    lower, upper = rings[:-1], rings[1:]
    next_lower = np.roll(lower, -1, axis=1)
    next_upper = np.roll(upper, -1, axis=1)
    sides = np.stack((lower, next_lower, upper, next_lower, next_upper, upper),
                     axis=2).reshape(-1, 3)
    if closed:
        return sides

    ends = np.stack((rings[0, ::-1], rings[-1]))
    hubs = np.broadcast_to(centers[[0, -1], None, :], ends.shape)
    caps = np.stack((hubs, ends, np.roll(ends, -1, axis=1)),
                    axis=2).reshape(-1, 3)
    return np.vstack((sides, caps))


def stage_motion(stage, pose, scene, arrow_size):
    """Return the axis index, travel mesh and current-setting mesh."""
    if not isinstance(stage, (LinearStage, RotationStage)):
        return None

    positions = []
    for value in (stage.positionMin, stage.positionMax, stage.position):
        if value is not None:
            try:
                with np.errstate(invalid='ignore', divide='ignore'):
                    value = stage.position_to_base_units(value)
            except (TypeError, ValueError, OverflowError):
                value = None
            else:
                if not np.isfinite(value):
                    value = None
        positions.append(value)
    start, stop, current = positions
    has_range = start is not None and stop is not None and start != stop
    if has_range:
        start, stop = sorted((start, stop))

    transform = scene @ pose.matrix
    linear = isinstance(stage, LinearStage)
    axis = ('xyz'.index(stage.axis) if linear else
            ('pitch', 'roll', 'yaw').index(stage.axis))
    axis_vector = transform[:3, axis]
    axis_norm = np.linalg.norm(axis_vector)
    if axis_norm == 0.:
        return None
    direction = axis_vector / axis_norm
    radial = transform[:3, (axis + 1) % 3].copy()
    radial -= (radial @ direction) * direction
    radial_norm = np.linalg.norm(radial)
    if radial_norm == 0.:
        return None
    radial /= radial_norm
    tangent = np.cross(direction, radial)

    length, tip_length, radius, segments = arrow_size
    phi = np.linspace(0., 2 * np.pi, segments, endpoint=False)
    across = radius * np.cos(phi)[:, None]
    up = radius * np.sin(phi)[:, None]
    travel, marker = None, None
    if linear:
        origin = transform[:3, 3]
        section = across * radial + up * tangent
        if has_range:
            centers = origin + np.array((start, stop))[:, None] * axis_vector
            travel = _tube_triangles(centers, centers[:, None, :] + section)
        if current is not None:
            position = origin + current * axis_vector
            offsets = radius * np.array((-.5, .5, .5, -.5, -.5))
            centers = position + offsets[:, None] * direction
            # Follow the outside wall, far lip, inside wall and near lip.
            widths = np.array((1.5, 1.5, 1., 1., 1.5))
            rings = centers[:, None, :] + widths[:, None, None] * section
            marker = _tube_triangles(centers, rings, closed=True)
    else:
        pivot = transform_points(transform, stage.pivotPoint)
        if has_range:
            span = min(stop - start, 2 * np.pi)
            full_circle = span == 2 * np.pi
            count = int(np.ceil(60 * span / (2 * np.pi)))
            angles = start % (2 * np.pi) + np.linspace(0., span, count + 1)
            spokes = (np.cos(angles)[:, None] * radial +
                      np.sin(angles)[:, None] * tangent)
            centers = pivot + length * spokes
            rings = (centers[:, None, :] +
                     across * spokes[:, None, :] - up * direction)
            travel = _tube_triangles(centers, rings, closed=full_circle)
        if current is not None:
            spoke = np.cos(current) * radial + np.sin(current) * tangent
            section = across * np.cross(direction, spoke) + up * direction
            head = length - tip_length
            offsets = np.array((0., head, head, length))
            centers = pivot + offsets[:, None] * spoke
            widths = np.array((.2, .2, 1., 0.))
            rings = centers[:, None, :] + widths[:, None, None] * section
            marker = _tube_triangles(centers, rings)

    return axis, travel, marker


def stage_plates(beamline, meshes):
    """Yield each stage, primitive key and world transform in child order."""
    # Fit plates to the optical footprints and mechanical mounting points.
    # Rendered primitives are output only and never enter these bounds.
    footprints = defaultdict(list)
    optical_normals = defaultdict(list)
    direct_optics = {}
    for mesh in meshes.values():
        mount = getattr(mesh.oe, 'mount', None)
        if mount is None or not mesh.isEnabled:
            continue
        for index, bounds in mesh.surfaceBounds.items():
            pose = np.array(mesh.surface_orientation(index).data()).reshape(
                4, 4, order='F')
            points = bounds.mean(axis=0) + _CUBE * (bounds[1] - bounds[0])
            footprints[mount].append(transform_points(pose, points))
            if is_oe(mesh.oe):
                direct_optics[mount] = bounds, pose
                optical_normals[mount].append(pose[:3, 2])

    states = {}
    for stage_id in reversed(beamline.sort_stages()):
        stage = beamline.stagesDict.get(stage_id)
        if stage is None:
            continue
        samples = footprints.pop(stage, None)
        if not samples:
            continue
        normals = optical_normals.pop(stage, [])
        state = stage.evaluate(states)

        base, mount = state.globalPose.matrix, state.mountPose.matrix
        points = np.concatenate(samples)
        optic = direct_optics.get(stage) if len(samples) == 1 else None
        if isinstance(stage, RotationStage):
            axis = ('pitch', 'roll', 'yaw').index(stage.axis)
            style = stage.renderStyle
            if style == 'auto':
                style = 'disk' if axis == 2 else 'cradle'
            if style == 'disk':
                pieces = _disk_plates(
                    stage, base, mount, points, normals, axis, optic)
            else:
                pieces = _cradle_plates(
                    stage, base, mount, points, normals, axis, optic)
        else:
            local = _local_points(mount, points)
            bounds = np.array((local.min(axis=0), local.max(axis=0)))
            if isinstance(stage, LinearStage):
                pieces = _linear_plates(stage, base, mount, bounds, optic)
            else:
                front = _front(mount, normals)
                face = bounds[0 if front > 0 else 1, 2] if optic else 0.
                pieces = [(0, _plate(mount, bounds[:, :2], face, 2., front))]

        for index, model in pieces:
            yield stage, index, model

        if stage.mount is not None:
            footprints[stage.mount].extend(samples)
            footprints[stage.mount].append(np.array(
                (state.globalPose.center, state.mountPose.center)))
            optical_normals[stage.mount].extend(normals)

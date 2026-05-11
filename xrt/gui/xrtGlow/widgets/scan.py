# -*- coding: utf-8 -*-
"""
Timeline scan helpers for xrtGlow.

The module intentionally separates the scan description compiler from the Qt
widgets. The compiler turns compact timeline recipes into explicit frame
patches; the widgets provide a first UI surface for inspecting those recipes.
"""

import copy
import json
import os
import re
import string
from collections import OrderedDict

from ...commons import qt

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "7 May 2026"


SCENE_TARGETS = {'Scene', 'scene', 'xrtGlow', 'xrtglow'}
FRAME_SECTIONS = {'id', 'objects', 'scene', 'actions', 'output', 'vars'}
SCENE_PROPERTY_NAMES = {
    'scaleVec', 'rotations', 'coordOffset', 'offsetCoord', 'tVec'}
DEFAULT_OUTPUT = {'glowFrameName': 'frame{index:04d}.jpg'}


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return string.Formatter.get_value(self, key, args, kwargs)


def _format_template(value, variables):
    if not isinstance(value, str):
        return value
    try:
        return _SafeFormatter().format(value, **variables)
    except Exception:
        return value


def _linspace(start, stop, steps):
    steps = int(steps)
    if steps <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / (steps - 1)
    return [float(start) + step * index for index in range(steps)]


def _split_numeric_unit(value):
    if isinstance(value, (int, float)):
        return float(value), ''
    match = re.match(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)'
                     r'(?:[eE][+-]?\d+)?)\s*(.*?)\s*$',
                     str(value))
    if match is None:
        return None, None
    return float(match.group(1)), match.group(2)


def _format_scan_value(value, unit):
    if not unit:
        return value
    return f'{value:g} {unit}'


def default_scan_description():
    return {
        'version': 1,
        'kind': 'timeline_recipe',
        'frames': 0,
        'output': copy.deepcopy(DEFAULT_OUTPUT),
        'items': [],
        }


def find_catalog_property(catalog, target, property_name):
    target = str(target)
    property_name = str(property_name)
    for target_info in catalog or []:
        target_names = [
            target_info.get('target'),
            target_info.get('name'),
            ]
        if target not in [str(name) for name in target_names
                          if name is not None]:
            continue
        for prop in target_info.get('properties', []):
            if str(prop.get('name')) == property_name:
                return prop
    return None


def _frame_sort_key(frame_id):
    match = re.match(r'^frame_(\d+)$', str(frame_id))
    if match is None:
        return (1, str(frame_id))
    return (0, int(match.group(1)))


def _looks_like_frame_key(key):
    return re.match(r'^frame_\d+$', str(key)) is not None


def _scan_default_bounds(value):
    numeric, unit = _split_numeric_unit(value)
    if numeric is None:
        fallback = str(value or '0')
        return fallback, fallback
    lower = numeric * 0.9
    upper = numeric * 1.1
    if lower > upper:
        lower, upper = upper, lower
    return _format_scan_value(lower, unit), _format_scan_value(upper, unit)


def _value_sequence(spec, fallback_steps=None):
    if isinstance(spec, dict):
        spec_type = spec.get('type', 'linspace')
        if spec_type == 'linspace':
            steps = int(spec.get('steps', fallback_steps or 1))
            start = spec.get('start', 0.0)
            stop = spec.get('stop', 0.0)
            start_value, start_unit = _split_numeric_unit(start)
            stop_value, stop_unit = _split_numeric_unit(stop)
            if start_value is None or stop_value is None:
                raise ValueError(
                    f'Cannot create linspace from {start!r} to {stop!r}')
            if start_unit != stop_unit:
                raise ValueError(
                    f'Cannot interpolate different units: '
                    f'{start_unit!r} and {stop_unit!r}')
            return [_format_scan_value(value, start_unit)
                    for value in _linspace(start_value, stop_value, steps)]
        if spec_type == 'list':
            return list(spec.get('values', []))
        if spec_type == 'constant':
            steps = int(spec.get('steps', fallback_steps or 1))
            return [spec.get('value')] * steps
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if fallback_steps is None:
        return [spec]
    return [spec] * int(fallback_steps)


def _set_patch_value(frame, target, property_name, value):
    if target in SCENE_TARGETS:
        section = frame.setdefault('scene', OrderedDict())
        section[property_name] = value
    else:
        objects = frame.setdefault('objects', OrderedDict())
        obj_patch = objects.setdefault(target, OrderedDict())
        obj_patch[property_name] = value


def _merge_dict(dst, src, path, warnings, item_id):
    for key, value in src.items():
        next_path = path + (key,)
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value, next_path, warnings, item_id)
            continue
        if key in dst and dst[key] != value:
            warnings.append({
                'frame': path[0] if path else None,
                'path': '.'.join(str(p) for p in next_path[1:]),
                'item': item_id,
                'old': dst[key],
                'new': value,
                })
        dst[key] = value


class BaseScan:
    """A compact timeline recipe that expands into explicit frame patches."""

    def __init__(self, description=None):
        self.description = copy.deepcopy(
            description or default_scan_description())
        self.version = self.description.get('version', 1)
        self.kind = self.description.get('kind', 'timeline_recipe')
        self.expanded_frames = self._expanded_frames_from_description(
            self.description)
        frame_count = self.description.get(
            'frameCount', self.description.get('frames', 0))
        if isinstance(frame_count, dict):
            frame_count = len(frame_count)
        self.frame_count = int(frame_count or 0)
        self.items = list(self.description.get(
            'items', self.description.get('tracks', [])))
        self.actions = copy.deepcopy(self.description.get('actions', {}))
        self.output = copy.deepcopy(
            self.description.get('output', DEFAULT_OUTPUT))
        self.warnings = []

    @classmethod
    def from_json(cls, data):
        if isinstance(data, str):
            data = json.loads(data)
        return cls(data)

    @classmethod
    def property_scan(cls, target, property_name, start_frame, min_value,
                      max_value, points, frames=None, target_name=None):
        points = int(points)
        start_frame = int(start_frame)
        target = target_name or target
        return cls({
            'version': 1,
            'kind': 'timeline_recipe',
            'frames': frames or start_frame + points,
            'output': copy.deepcopy(DEFAULT_OUTPUT),
            'items': [{
                'type': 'track',
                'id': f'{target_name or target}.{property_name}',
                'start': start_frame,
                'duration': points,
                'target': target,
                'property': property_name,
                'values': {
                    'type': 'linspace',
                    'start': str(min_value),
                    'stop': str(max_value),
                    'steps': points,
                    },
                }],
            })

    def to_json(self, **kwargs):
        return json.dumps(self.description, **kwargs)

    def _expanded_frames_from_description(self, description):
        candidates = description.get('expandedFrames',
                                     description.get('frameDict'))
        if candidates is None and isinstance(description.get('frames'), dict):
            candidates = description.get('frames')
        if candidates is None:
            frame_items = [(key, value) for key, value in description.items()
                           if _looks_like_frame_key(key)]
            if frame_items:
                candidates = OrderedDict(sorted(frame_items,
                                                key=lambda item:
                                                _frame_sort_key(item[0])))
        if candidates is None:
            return None
        return OrderedDict(sorted(candidates.items(),
                                  key=lambda item: _frame_sort_key(item[0])))

    def _normalize_expanded_frame(self, frame_id, frame, index):
        if not isinstance(frame, dict):
            frame = {'objects': copy.deepcopy(frame)}
        frame = copy.deepcopy(frame)
        if any(key in FRAME_SECTIONS for key in frame):
            normalized = OrderedDict()
            normalized['id'] = frame.get('id', frame_id)
            for section in ['objects', 'scene', 'actions', 'output', 'vars']:
                if section in frame:
                    normalized[section] = frame[section]
            for key, value in frame.items():
                if key in FRAME_SECTIONS:
                    continue
                if key in SCENE_PROPERTY_NAMES:
                    normalized.setdefault('scene', OrderedDict())[key] = value
                else:
                    normalized.setdefault(
                        'objects', OrderedDict())[key] = value
        else:
            normalized = OrderedDict([('id', frame_id)])
            for key, value in frame.items():
                if key in SCENE_PROPERTY_NAMES:
                    normalized.setdefault('scene', OrderedDict())[key] = value
                else:
                    normalized.setdefault('objects',
                                          OrderedDict())[key] = value
        if self.actions and 'actions' not in normalized:
            normalized['actions'] = copy.deepcopy(self.actions)
        if self.output and 'output' not in normalized:
            variables = {'index': index, 'frame': frame_id}
            normalized['output'] = self._format_mapping(
                self.output, variables)
        return normalized

    def _compile_expanded_frames(self):
        frames = OrderedDict()
        for index, (frame_id, frame) in enumerate(
                self.expanded_frames.items()):
            frames[frame_id] = self._normalize_expanded_frame(
                frame_id, frame, index)
        self.frame_count = len(frames)
        return frames

    def _ensure_frame_count(self):
        if self.expanded_frames is not None:
            self.frame_count = len(self.expanded_frames)
            return
        if self.frame_count:
            return
        frame_count = 0
        for item in self.items:
            item_type = item.get('type', 'track')
            if item_type == 'event':
                frame_count = max(
                    frame_count, int(item.get('frame', 0)) +
                    int(item.get('duration', item.get('steps', 1))))
            elif item_type == 'loopBlock':
                frame_count = max(frame_count, int(item.get('start', 0)) +
                                  self._loop_block_length(item))
            else:
                frame_count = max(frame_count, int(item.get('start', 0)) +
                                  int(item.get('duration',
                                               item.get('steps', 1))))
        self.frame_count = frame_count

    def _make_empty_frames(self):
        self._ensure_frame_count()
        frames = OrderedDict()
        for index in range(self.frame_count):
            frame_id = f'frame_{index:04d}'
            frame = OrderedDict([('id', frame_id)])
            if self.actions:
                frame['actions'] = copy.deepcopy(self.actions)
            if self.output:
                variables = {'index': index, 'frame': frame_id}
                frame['output'] = self._format_mapping(self.output, variables)
            frames[frame_id] = frame
        return frames

    def _format_mapping(self, mapping, variables):
        formatted = OrderedDict()
        for key, value in mapping.items():
            if isinstance(value, dict):
                formatted[key] = self._format_mapping(value, variables)
            else:
                formatted[key] = _format_template(value, variables)
        return formatted

    def _merge_frame(self, frames, index, patch, item_id):
        if index < 0:
            return
        frame_id = f'frame_{index:04d}'
        if frame_id not in frames:
            for missing in range(len(frames), index + 1):
                frames[f'frame_{missing:04d}'] = OrderedDict([
                    ('id', f'frame_{missing:04d}')])
        _merge_dict(frames[frame_id], patch, (frame_id,), self.warnings,
                    item_id)

    def _compile_track(self, frames, item):
        start = int(item.get('start', 0))
        duration = int(item.get('duration', item.get('steps', 1)))
        values = _value_sequence(item.get('values'), duration)
        item_id = item.get('id',
                           f"{item.get('target')}.{item.get('property')}")
        for offset in range(duration):
            if not values:
                break
            value = values[min(offset, len(values) - 1)]
            patch = OrderedDict()
            _set_patch_value(patch, item.get('target'),
                             item.get('property'), value)
            self._merge_frame(frames, start + offset, patch, item_id)

    def _compile_event(self, frames, item):
        frame_index = int(item.get('frame', item.get('start', 0)))
        patch = OrderedDict()
        for section in ['objects', 'scene', 'actions', 'output']:
            if section in item:
                patch[section] = copy.deepcopy(item[section])
        duration = max(1, int(item.get('duration', item.get('steps', 1))))
        for offset in range(duration):
            self._merge_frame(frames, frame_index + offset,
                              copy.deepcopy(patch), item.get('id', 'event'))

    def _loop_values(self, loops):
        if not loops:
            yield {}
            return
        first = loops[0]
        rest = loops[1:]
        var = first.get('var', first.get('name'))
        for value in _value_sequence(first.get('values', first.get('range'))):
            for subvars in self._loop_values(rest):
                variables = copy.deepcopy(subvars)
                variables[var] = value
                yield variables

    def _loop_block_length(self, item):
        length = 1
        for loop in item.get('loops', []):
            length *= len(_value_sequence(loop.get(
                'values', loop.get('range'))))
        return length

    def _compile_loop_block(self, frames, item):
        start = int(item.get('start', 0))
        item_id = item.get('id', 'loopBlock')
        for offset, variables in enumerate(self._loop_values(
                item.get('loops', []))):
            variables = dict(variables)
            variables.update({'index': start + offset,
                              'frame': f'frame_{start + offset:04d}'})
            patch = OrderedDict()
            for section in ['objects', 'scene', 'actions', 'output']:
                if section in item:
                    patch[section] = self._format_mapping(
                        item[section], variables)
            if variables:
                patch['vars'] = variables
            self._merge_frame(frames, start + offset, patch, item_id)

    def compile_frames(self):
        self.warnings = []
        if self.expanded_frames is not None:
            frames = self._compile_expanded_frames()
        else:
            frames = self._make_empty_frames()
        for item in self.items:
            item_type = item.get('type', 'track')
            if item_type == 'event':
                self._compile_event(frames, item)
            elif item_type == 'loopBlock':
                self._compile_loop_block(frames, item)
            else:
                self._compile_track(frames, item)
        self.frame_count = len(frames)
        return frames


class ScanRangeDialog(qt.QDialog):
    """Small dialog for creating a one-property linear scan."""

    scanCreated = qt.Signal(dict)

    def __init__(self, target, property_name, current_value=None,
                 target_name=None, parent=None):
        super().__init__(parent)
        self.target = target_name or target
        self.property_name = property_name
        self.setWindowTitle(
            f'Create scan: {self.target}.{property_name}')

        self.startFrameEdit = qt.QLineEdit('0')
        min_value, max_value = _scan_default_bounds(current_value)
        self.minValueEdit = qt.QLineEdit(str(min_value))
        self.maxValueEdit = qt.QLineEdit(str(max_value))
        self.pointsEdit = qt.QLineEdit('10')

        layout = qt.QVBoxLayout(self)
        form = qt.QFormLayout()
        form.addRow('Target', qt.QLabel(str(self.target)))
        form.addRow('Property', qt.QLabel(str(property_name)))
        form.addRow('Start frame', self.startFrameEdit)
        form.addRow('Min value', self.minValueEdit)
        form.addRow('Max value', self.maxValueEdit)
        form.addRow('Number of points', self.pointsEdit)
        layout.addLayout(form)

        self.buttonBox = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

    def scan_item(self):
        points = int(self.pointsEdit.text())
        return {
            'type': 'track',
            'id': f'{self.target}.{self.property_name}',
            'start': int(self.startFrameEdit.text()),
            'duration': points,
            'target': self.target,
            'property': self.property_name,
            'values': {
                'type': 'linspace',
                'start': self.minValueEdit.text(),
                'stop': self.maxValueEdit.text(),
                'steps': points,
                },
            }

    def accept(self):
        try:
            item = self.scan_item()
            BaseScan({'items': [item]}).compile_frames()
        except Exception as exc:
            qt.QMessageBox.warning(self, 'Invalid scan',
                                   f'Cannot create scan: {exc}')
            return
        self.scanCreated.emit(item)
        super().accept()


class ScanInstructionDialog(qt.QDialog):
    """Dialog for adding a property event or track from the scan panel."""

    scanCreated = qt.Signal(dict)

    def __init__(self, catalog, start_frame=0, edit_item=None, parent=None):
        super().__init__(parent)
        self.catalog = list(catalog or [])
        self.propertyMap = {}
        self.propertyItems = {}
        self.selectedProperty = None
        self.editItem = copy.deepcopy(edit_item)
        self._editValuesEditable = True
        self.setWindowTitle('Add scan instruction')
        self.resize(520, 520)

        self.propertyTree = qt.QTreeWidget()
        self.propertyTree.setHeaderLabels(['Property', 'Current value'])
        self.propertyTree.setSelectionMode(
            qt.QAbstractItemView.SingleSelection)
        self.propertyTree.itemSelectionChanged.connect(
            self._selection_changed)

        self.startFrameEdit = qt.QLineEdit(str(int(start_frame)))
        self.minValueEdit = qt.QLineEdit('0')
        self.maxValueEdit = qt.QLineEdit('0')
        self.pointsEdit = qt.QLineEdit('1')

        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.QLabel('Select scene or element property'))
        layout.addWidget(self.propertyTree)

        form = qt.QFormLayout()
        form.addRow('First frame', self.startFrameEdit)
        form.addRow('Start value', self.minValueEdit)
        form.addRow('End value', self.maxValueEdit)
        form.addRow('Number of frames', self.pointsEdit)
        layout.addLayout(form)

        hint = qt.QLabel(
            '1 frame creates a single-frame injection. Equal start/end '
            'values over multiple frames create a hold/pause.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.buttonBox = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self._populate_tree()
        if self.editItem is not None:
            self._apply_edit_item()

    def _populate_tree(self):
        self.propertyTree.clear()
        self.propertyMap.clear()
        self.propertyItems.clear()
        for target in self.catalog:
            target_name = str(target.get('name', target.get('target', '')))
            target_item = qt.QTreeWidgetItem([
                target_name, ''])
            target_item.setFirstColumnSpanned(True)
            self.propertyTree.addTopLevelItem(target_item)
            for prop in target.get('properties', []):
                key = f"{target_name}::{prop.get('name')}"
                value = prop.get('value', '')
                child = qt.QTreeWidgetItem([str(prop.get('name')), str(value)])
                child.setData(0, qt.Qt.UserRole, key)
                target_item.addChild(child)
                self.propertyMap[key] = {
                    'target': target_name,
                    'property': prop.get('name'),
                    'value': value,
                    }
                self.propertyItems[key] = child
            target_item.setExpanded(False)
        self.propertyTree.resizeColumnToContents(0)

    def _apply_edit_item(self):
        target, property_name = self._item_target_property(self.editItem)
        start, points, start_value, stop_value, editable = \
            self._item_dialog_values(self.editItem)
        if target is None or property_name is None:
            return
        self.setWindowTitle(f'Edit scan: {target}.{property_name}')
        key = f'{target}::{property_name}'
        self.selectedProperty = self.propertyMap.get(key, {
            'target': target,
            'property': property_name,
            'value': start_value,
            })
        item = self.propertyItems.get(key)
        if item is not None:
            self.propertyTree.setCurrentItem(item)
        self.propertyTree.setEnabled(False)
        self.startFrameEdit.setText(str(start))
        self.minValueEdit.setText(str(start_value))
        self.maxValueEdit.setText(str(stop_value))
        self.pointsEdit.setText(str(points))
        self.minValueEdit.setReadOnly(not editable)
        self.maxValueEdit.setReadOnly(not editable)
        self._editValuesEditable = editable

    def _item_target_property(self, item):
        if item.get('target') and item.get('property'):
            return item.get('target'), item.get('property')
        objects = item.get('objects', {})
        if len(objects) == 1:
            target, patch = next(iter(objects.items()))
            if isinstance(patch, dict) and len(patch) == 1:
                return target, next(iter(patch.keys()))
        scene = item.get('scene', {})
        if isinstance(scene, dict) and len(scene) == 1:
            return 'Scene', next(iter(scene.keys()))
        return None, None

    def _item_dialog_values(self, item):
        start = int(item.get('frame', item.get('start', 0)))
        points = int(item.get('duration', item.get('steps', 1)))
        if item.get('type') == 'loopBlock':
            return start, points, '', '', False
        if item.get('type') == 'event':
            value = self._single_event_value(item)
            value = '' if value is None else value
            return start, points, value, value, value != ''
        values = item.get('values')
        if isinstance(values, dict):
            value_type = values.get('type')
            points = int(item.get('duration',
                                  values.get('steps', points)))
            if value_type == 'linspace':
                return start, points, values.get('start', ''), \
                    values.get('stop', ''), True
            if value_type == 'constant':
                value = values.get('value', '')
                return start, points, value, value, True
            if value_type == 'list':
                value_list = list(values.get('values', []))
                if not value_list:
                    return start, points, '', '', False
                return start, len(value_list), value_list[0], \
                    value_list[-1], False
        if isinstance(values, (list, tuple)):
            if not values:
                return start, points, '', '', False
            return start, len(values), values[0], values[-1], False
        return start, points, values, values, True

    def _single_event_value(self, item):
        values = []
        for patch in item.get('objects', {}).values():
            if isinstance(patch, dict):
                values.extend(patch.values())
        scene = item.get('scene', {})
        if isinstance(scene, dict):
            values.extend(scene.values())
        if len(values) != 1:
            return None
        return values[0]

    def _selection_changed(self):
        self.selectedProperty = self._current_property()
        if self.selectedProperty is None:
            return
        value = self.selectedProperty.get('value', '')
        self.minValueEdit.setText(str(value))
        self.maxValueEdit.setText(str(value))

    def _current_property(self):
        item = self.propertyTree.currentItem()
        if item is None:
            items = self.propertyTree.selectedItems()
            item = items[0] if items else None
        if item is None:
            return None
        key = item.data(0, qt.Qt.UserRole)
        prop = self.propertyMap.get(key)
        if prop is None and self.selectedProperty is not None:
            return self.selectedProperty
        return prop

    def _patch_for_value(self, value):
        prop = self._current_property()
        patch = OrderedDict()
        _set_patch_value(patch, prop['target'], prop['property'], value)
        return patch

    def scan_item(self):
        self.selectedProperty = self._current_property()
        if self.selectedProperty is None:
            raise ValueError('Select a property first')
        start = int(self.startFrameEdit.text())
        points = int(self.pointsEdit.text())
        if points < 1:
            raise ValueError('Number of frames must be at least 1')
        start_value = self.minValueEdit.text()
        stop_value = self.maxValueEdit.text()
        prop = self.selectedProperty
        if self.editItem is not None and not self._editValuesEditable:
            item = copy.deepcopy(self.editItem)
            if item.get('type') == 'event':
                item.pop('start', None)
                item['frame'] = start
                if points > 1:
                    item['duration'] = points
                else:
                    item.pop('duration', None)
                    item.pop('steps', None)
            else:
                item['start'] = start
                item['duration'] = points
            return item
        item_id = f"{prop['target']}.{prop['property']}"

        if points == 1:
            item = OrderedDict([
                ('type', 'event'),
                ('id', item_id),
                ('frame', start),
                ])
            item.update(self._patch_for_value(start_value))
            return item

        values = OrderedDict([('type', 'constant'),
                              ('value', start_value),
                              ('steps', points)])
        if start_value != stop_value:
            values = OrderedDict([('type', 'linspace'),
                                  ('start', start_value),
                                  ('stop', stop_value),
                                  ('steps', points)])
        return OrderedDict([
            ('type', 'track'),
            ('id', item_id),
            ('start', start),
            ('duration', points),
            ('target', prop['target']),
            ('property', prop['property']),
            ('values', values),
            ])

    def accept(self):
        try:
            item = self.scan_item()
            BaseScan({'items': [item]}).compile_frames()
        except Exception as exc:
            qt.QMessageBox.warning(self, 'Invalid instruction',
                                   f'Cannot create instruction: {exc}')
            return
        self.scanCreated.emit(item)
        super().accept()


class TimelineFrameListWidget(qt.QWidget):
    """Preview widget for scan timeline items and expanded frames."""

    scanStarted = qt.Signal()
    scanPaused = qt.Signal()
    scanStopped = qt.Signal()
    currentFrameChanged = qt.Signal(int, dict)
    outputTemplateChanged = qt.Signal(str)
    instructionRequested = qt.Signal(int)
    trackDeleteRequested = qt.Signal(int)
    scanLoadRequested = qt.Signal()
    scanSaveRequested = qt.Signal()
    trackTimingChanged = qt.Signal(int, dict)
    trackEditRequested = qt.Signal(int)

    TRACK_COL_ID = 0
    TRACK_COL_VALUE_START = 1
    TRACK_COL_VALUE_END = 2
    TRACK_COL_FRAMES = 3
    TRACK_COL_START_FRAME = 4
    TRACK_VALUE_COLUMNS = {TRACK_COL_VALUE_START, TRACK_COL_VALUE_END}
    TRACK_TIMING_COLUMNS = {TRACK_COL_FRAMES, TRACK_COL_START_FRAME}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan = BaseScan()
        self.frames = OrderedDict()
        self.frameIds = []
        self.currentFrame = 0
        self._playing = False
        self._updatingSelection = False
        self._updatingTracks = False

        layout = qt.QVBoxLayout(self)
        controls = qt.QHBoxLayout()
        self.addInstructionButton = self._make_tool_button(
            's_add.png', 'Add instruction')
        self.deleteTrackButton = self._make_tool_button(
            's_remove.png', 'Delete selected scan')
        self.loadScanButton = self._make_tool_button(
            's_open.png', 'Load scan JSON')
        self.saveScanButton = self._make_tool_button(
            's_save.png', 'Save scan JSON')
        self.startButton = self._make_tool_button(
            's_play.png', 'Start scan')
        self.pauseButton = self._make_tool_button(
            's_pause.png', 'Pause scan')
        self.stopButton = self._make_tool_button(
            's_stop.png', 'Stop scan')
        self.currentFrameLabel = qt.QLabel('Current frame 0 / 0')
        self.outputTemplateEdit = qt.QLineEdit(
            DEFAULT_OUTPUT['glowFrameName'])
        info_controls = qt.QHBoxLayout()
        controls.addWidget(self.addInstructionButton)
        controls.addWidget(self.deleteTrackButton)
        controls.addWidget(self.loadScanButton)
        controls.addWidget(self.saveScanButton)
        controls.addSpacing(12)
        controls.addWidget(self.startButton)
        controls.addWidget(self.pauseButton)
        controls.addWidget(self.stopButton)
        controls.addStretch()
        layout.addLayout(controls)
        info_controls.addWidget(self.currentFrameLabel)
        info_controls.addSpacing(12)
        info_controls.addWidget(qt.QLabel('Filename template'))
        info_controls.addWidget(self.outputTemplateEdit)
        info_controls.addStretch()
        layout.addLayout(info_controls)

        splitter = qt.QSplitter(qt.Qt.Vertical)
        layout.addWidget(splitter)

        self.trackTable = qt.QTableWidget(0, 5)
        self.trackTable.setHorizontalHeaderLabels(
            ['Id', 'Start', 'End', 'Frames', 'startFrame'])
        self.trackTable.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.trackTable.setEditTriggers(
            qt.QAbstractItemView.DoubleClicked |
            qt.QAbstractItemView.EditKeyPressed |
            qt.QAbstractItemView.SelectedClicked)
        self.trackTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.trackTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.frameTable = qt.QTableWidget(0, 4)
        self.frameTable.setHorizontalHeaderLabels(
            ['Frame', 'Objects', 'Scene', 'Output'])
        self.frameTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.warningList = qt.QListWidget()
        self.frameTable.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.frameTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.frameTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.frameTable.customContextMenuRequested.connect(
            self._frame_context_menu)
        self.trackTable.customContextMenuRequested.connect(
            self._track_context_menu)
        self.frameTable.itemSelectionChanged.connect(
            self._on_frame_selection_changed)
        self.trackTable.itemChanged.connect(self._on_track_item_changed)
        self.trackTable.itemDoubleClicked.connect(
            self._on_track_item_double_clicked)
        self.addInstructionButton.clicked.connect(
            self._request_instruction_at_current_frame)
        self.deleteTrackButton.clicked.connect(self._delete_selected_track)
        self.loadScanButton.clicked.connect(self.scanLoadRequested.emit)
        self.saveScanButton.clicked.connect(self.scanSaveRequested.emit)
        self.startButton.clicked.connect(self.start_scan)
        self.pauseButton.clicked.connect(self.pause_scan)
        self.stopButton.clicked.connect(self.stop_scan)
        self.outputTemplateEdit.editingFinished.connect(
            self._output_template_edited)

        splitter.addWidget(self.trackTable)

        bottom = qt.QTabWidget()
        bottom.addTab(self.frameTable, 'Frames')
        bottom.addTab(self.warningList, 'Warnings')
        splitter.addWidget(bottom)
        self.deleteTrackShortcut = qt.QShortcut(self.trackTable)
        self.deleteTrackShortcut.setKey(qt.QKeySequence.Delete)
        self.deleteTrackShortcut.activated.connect(self._delete_selected_track)
        self._update_play_buttons()

    def _make_tool_button(self, icon_name, tooltip):
        button = qt.QToolButton(self)
        icon_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), '_icons', icon_name)
        button.setIcon(qt.QIcon(icon_path))
        button.setIconSize(qt.QSize(48, 48))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
        button.setAutoRaise(True)
        return button

    def set_scan(self, scan):
        self.scan = scan if isinstance(scan, BaseScan) else BaseScan(scan)
        template = self.scan.output.get(
            'glowFrameName', DEFAULT_OUTPUT['glowFrameName'])
        self.outputTemplateEdit.blockSignals(True)
        self.outputTemplateEdit.setText(str(template))
        self.outputTemplateEdit.blockSignals(False)
        self.rebuild()

    def output_template(self):
        template = str(self.outputTemplateEdit.text()).strip()
        return template or DEFAULT_OUTPUT['glowFrameName']

    def _output_template_edited(self):
        template = self.output_template()
        self.outputTemplateEdit.setText(template)
        self.scan.description.setdefault('output', {})[
            'glowFrameName'] = template
        self.scan.output['glowFrameName'] = template
        self.rebuild()
        self.outputTemplateChanged.emit(template)

    def _request_instruction_at_current_frame(self):
        self.instructionRequested.emit(self.currentFrame)

    def _frame_context_menu(self, position):
        row = self.frameTable.rowAt(position.y())
        if row >= 0:
            self.set_current_frame(row)
        else:
            row = self.currentFrame
        menu = qt.QMenu(self)
        add_action = menu.addAction(f'Add instruction at frame {row}')
        add_action.triggered.connect(
            lambda checked=False, row=row: self.instructionRequested.emit(row))
        menu.exec_(qt.QCursor.pos())

    def _track_context_menu(self, position):
        row = self.trackTable.rowAt(position.y())
        if row < 0:
            return
        self.trackTable.selectRow(row)
        menu = qt.QMenu(self)
        delete_action = menu.addAction('Delete track')
        delete_action.triggered.connect(
            lambda checked=False, row=row: self.trackDeleteRequested.emit(row))
        menu.exec_(qt.QCursor.pos())

    def _delete_selected_track(self):
        rows = self.trackTable.selectionModel().selectedRows()
        if not rows:
            return
        self.trackDeleteRequested.emit(rows[0].row())

    def rebuild(self):
        frames = self.scan.compile_frames()
        self._populate_tracks()
        self._populate_frames(frames)
        self._populate_warnings()
        self.set_current_frame(min(self.currentFrame,
                                   max(0, len(self.frameIds) - 1)),
                               emit_signal=False)
        self._update_play_buttons()

    def start_scan(self):
        if not self.frameIds:
            return
        if self.currentFrame >= len(self.frameIds) - 1:
            self.set_current_frame(0)
        self._playing = True
        self.scanStarted.emit()
        self._update_play_buttons()

    def pause_scan(self):
        if not self._playing:
            return
        self._playing = False
        self.scanPaused.emit()
        self._update_play_buttons()

    def stop_scan(self):
        self._playing = False
        self.set_current_frame(0)
        self.scanStopped.emit()
        self._update_play_buttons()

    def mark_scan_finished(self):
        self._playing = False
        self.set_current_frame(0, emit_signal=False)
        self._update_play_buttons()

    def _update_play_buttons(self):
        has_frames = bool(self.frameIds)
        self.startButton.setEnabled(has_frames and not self._playing)
        self.pauseButton.setEnabled(has_frames and self._playing)
        self.stopButton.setEnabled(has_frames)
        self.deleteTrackButton.setEnabled(bool(self.scan.items))

    def _on_frame_selection_changed(self):
        if self._updatingSelection:
            return
        selection = self.frameTable.selectionModel()
        if selection is None:
            return
        rows = selection.selectedRows()
        if not rows:
            return
        self.set_current_frame(rows[0].row())

    def set_current_frame(self, frame_index, emit_signal=True):
        if not self.frameIds:
            self.currentFrame = 0
            self.currentFrameLabel.setText('Current frame 0 / 0')
            return
        frame_index = max(0, min(int(frame_index), len(self.frameIds) - 1))
        self.currentFrame = frame_index
        self.currentFrameLabel.setText(
            f'Current frame {frame_index + 1} / {len(self.frameIds)}')
        self._updatingSelection = True
        self.frameTable.setCurrentCell(frame_index, 0)
        self.frameTable.selectRow(frame_index)
        self._updatingSelection = False
        if emit_signal:
            frame_id = self.frameIds[frame_index]
            self.currentFrameChanged.emit(frame_index, self.frames[frame_id])

    def _populate_tracks(self):
        self._updatingTracks = True
        try:
            self.trackTable.setRowCount(len(self.scan.items))
            for row, item in enumerate(self.scan.items):
                start_frame, frames = self._track_timing(item)
                start_value, end_value, _ = self._track_values(item)
                values = [item.get('id', ''), start_value, end_value,
                          frames, start_frame]
                for col, value in enumerate(values):
                    table_item = qt.QTableWidgetItem(str(value))
                    flags = table_item.flags()
                    if self._track_column_is_editable(item, col):
                        flags |= qt.Qt.ItemIsEditable
                    else:
                        flags &= ~qt.Qt.ItemIsEditable
                    table_item.setFlags(flags)
                    self.trackTable.setItem(row, col, table_item)
            self.trackTable.resizeColumnsToContents()
        finally:
            self._updatingTracks = False

    def _track_timing(self, item):
        start_frame = int(item.get('frame', item.get('start', 0)))
        if item.get('type') == 'loopBlock':
            frames = self.scan._loop_block_length(item)
        else:
            frames = int(item.get('duration', item.get('steps', 1)))
        frames = max(1, frames)
        return start_frame, frames

    def _track_values(self, item):
        item_type = item.get('type', 'track')
        if item_type == 'loopBlock':
            return '', '', False
        if item_type == 'event':
            value = self._single_event_value(item)
            editable = value is not None
            value = '' if value is None else value
            return value, value, editable
        values = item.get('values')
        if isinstance(values, dict):
            value_type = values.get('type')
            if value_type == 'linspace':
                return values.get('start', ''), values.get('stop', ''), True
            if value_type == 'constant':
                value = values.get('value', '')
                return value, value, True
            if value_type == 'list':
                value_list = list(values.get('values', []))
                if not value_list:
                    return '', '', False
                return value_list[0], value_list[-1], False
        if isinstance(values, (list, tuple)):
            if not values:
                return '', '', False
            return values[0], values[-1], False
        return values, values, True

    def _single_event_value(self, item):
        patches = []
        for patch in item.get('objects', {}).values():
            if isinstance(patch, dict):
                patches.extend(patch.values())
        scene = item.get('scene', {})
        if isinstance(scene, dict):
            patches.extend(scene.values())
        if len(patches) != 1:
            return None
        return patches[0]

    def _track_column_is_editable(self, item, column):
        if column in self.TRACK_VALUE_COLUMNS:
            return self._track_values(item)[2]
        if column not in self.TRACK_TIMING_COLUMNS:
            return False
        if item.get('type') == 'loopBlock' and \
                column != self.TRACK_COL_START_FRAME:
            return False
        return True

    def _on_track_item_changed(self, table_item):
        if self._updatingTracks:
            return
        row = table_item.row()
        column = table_item.column()
        if row < 0 or row >= len(self.scan.items):
            return
        item = self.scan.items[row]
        if not self._track_column_is_editable(item, column):
            return
        if column in self.TRACK_VALUE_COLUMNS:
            key = 'startValue' if column == self.TRACK_COL_VALUE_START else \
                'endValue'
            self.trackTimingChanged.emit(row, {key: table_item.text()})
            return
        try:
            value = int(str(table_item.text()).strip())
        except ValueError:
            self.rebuild()
            return
        start_frame, frames = self._track_timing(item)
        if column == self.TRACK_COL_START_FRAME:
            start_frame = max(0, value)
        elif column == self.TRACK_COL_FRAMES:
            frames = max(1, value)
        self.trackTimingChanged.emit(row, {
            'startFrame': start_frame,
            'frames': frames,
            })

    def _on_track_item_double_clicked(self, table_item):
        if table_item.column() != self.TRACK_COL_ID:
            return
        self.trackEditRequested.emit(table_item.row())

    def _populate_frames(self, frames):
        self.frames = frames
        self.frameIds = list(frames.keys())
        self.frameTable.setRowCount(len(frames))
        for row, (frame_id, frame) in enumerate(frames.items()):
            values = [frame_id,
                      json.dumps(frame.get('objects', {})),
                      json.dumps(frame.get('scene', {})),
                      json.dumps(frame.get('output', {}))]
            for col, value in enumerate(values):
                table_item = qt.QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.frameTable.setItem(row, col, table_item)
        self.frameTable.resizeColumnsToContents()

    def _populate_warnings(self):
        self.warningList.clear()
        for warning in self.scan.warnings:
            path = str(warning.get('path'))
            self.warningList.addItem(
                f"{warning.get('frame')}: {path} "
                f"overwritten by {warning.get('item')}")

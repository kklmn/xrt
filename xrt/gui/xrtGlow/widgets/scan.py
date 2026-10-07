# -*- coding: utf-8 -*-
"""
Timeline scan compiler, widgets, and Glow/Qook integration.

The compiler turns compact timeline recipes into explicit frame patches.
The mixins attach scan editing, playback, and code generation to the GUIs.
"""

import ast
import copy
import csv
from datetime import datetime
import json
import os
import re
import string
from collections import OrderedDict

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from ...commons import qt, config
from ....backends import raycing
from ....backends.raycing._flow_utils import normalize_string_input
from .._constants import DEFAULT_SCENE_SETTINGS, DISPLAY_NUMBER_FORMAT
from .._utils import is_aperture, is_screen

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "7 May 2026"


SCENE_TARGETS = {'Scene', 'scene', 'xrtGlow', 'xrtglow'}
FRAME_SECTIONS = {'id', 'objects', 'scene', 'actions', 'output', 'vars'}
SCENE_PROPERTY_NAMES = {
    'scaleVec', 'rotations', 'coordOffset', 'offsetCoord', 'tVec'}
DEFAULT_OUTPUT = {'glowFrameName': 'frame{index:04d}.jpg'}
FRAMES_CLEAN_KEY = 'framesClean'
SCAN_SCALAR_SOURCES = ('intensity', 'flux', 'power')
SCAN_TARGET_SOURCES = SCAN_SCALAR_SOURCES + tuple(raycing.allBeamFields)

SCAN_SCENE_COMPONENTS = OrderedDict([
    ('scaleVec', ['x', 'y', 'z']),
    ('rotations', ['azimuth', 'elevation']),
    ('coordOffset', ['x', 'y', 'z']),
    ('tVec', ['x', 'y', 'z']),
])
SCAN_ANGLE_PROPERTIES = {
    'pitch', 'roll', 'yaw', 'bragg', 'braggOffset', 'positionRoll',
    'cryst1roll', 'cryst2roll', 'cryst2pitch', 'alpha', 'theta',
    'wedgeAngle',
}
SCAN_LIMIT_PROPERTIES = (
    'limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2',
    'limOptX', 'limOptY', 'limOptX2', 'limOptY2')
SCAN_AXIS_PROPERTIES = ('x', 'z')
SCAN_CODE_INDENT = 4 * ' '


_SCAN_INT_MAX = 2147483647
_SCAN_ICON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '_icons', 'p_scan128.png')


class _ScanLineEdit(qt.QLineEdit):
    """Line edit which treats Return as editing confirmation, not dialog OK."""

    def keyPressEvent(self, event):
        if event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            validator = self.validator()
            if validator is None or self.hasAcceptableInput():
                self.editingFinished.emit()
                self.focusNextPrevChild(True)
            event.accept()
            return
        super().keyPressEvent(event)


def _set_scan_value_validators(dialog, property_name):
    for editor in (dialog.minValueEdit, dialog.maxValueEdit):
        validator = (qt.make_argument_validator(property_name, editor)
                     if property_name is not None else None)
        editor.setValidator(validator)


def _configure_scan_editors(dialog, property_name=None):
    """Install timing and property-value validators on a scan dialog."""
    dialog.startFrameEdit.setValidator(
        qt.QIntValidator(0, _SCAN_INT_MAX, dialog.startFrameEdit))
    dialog.pointsEdit.setValidator(
        qt.QIntValidator(1, _SCAN_INT_MAX, dialog.pointsEdit))
    _set_scan_value_validators(dialog, property_name)


def _scan_item_target_property(item):
    if item.get('target') and item.get('property'):
        return item['target'], item['property']
    objects = item.get('objects', {})
    if len(objects) == 1:
        target, patch = next(iter(objects.items()))
        if isinstance(patch, dict) and len(patch) == 1:
            return target, next(iter(patch))
    scene = item.get('scene', {})
    if isinstance(scene, dict) and len(scene) == 1:
        return 'Scene', next(iter(scene))
    return None, None


class _ScanTrackDelegate(qt.QStyledItemDelegate):
    """Validated editors for the editable cells of the tracks table."""

    def __init__(self, track_widget, parent=None):
        super().__init__(parent or track_widget.trackTable)
        self.track_widget = track_widget

    def createEditor(self, parent, option, index):
        editor = _ScanLineEdit(parent)
        if index.column() == TimelineFrameListWidget.TRACK_COL_START_FRAME:
            editor.setValidator(qt.QIntValidator(
                0, _SCAN_INT_MAX, editor))
        elif index.column() == TimelineFrameListWidget.TRACK_COL_FRAMES:
            editor.setValidator(qt.QIntValidator(
                1, _SCAN_INT_MAX, editor))
        elif index.column() in TimelineFrameListWidget.TRACK_VALUE_COLUMNS:
            row = index.row()
            if 0 <= row < len(self.track_widget.scan.items):
                item = self.track_widget.scan.items[row]
                _, property_name = _scan_item_target_property(item)
                validator = qt.make_argument_validator(
                    property_name, editor) if property_name else None
                editor.setValidator(validator)
        editor.installEventFilter(self)
        return editor

    def setEditorData(self, editor, index):
        if index.column() not in TimelineFrameListWidget.TRACK_VALUE_COLUMNS:
            return super().setEditorData(editor, index)
        raw_value = index.data(qt.RAW_VALUE_ROLE)
        if raw_value is None:
            return super().setEditorData(editor, index)
        editor.setText(str(raw_value))

    def eventFilter(self, editor, event):
        if event.type() == qt.QEvent.KeyPress and event.key() in (
                qt.Qt.Key_Return, qt.Qt.Key_Enter):
            if editor.validator() is None or editor.hasAcceptableInput():
                self.commitData.emit(editor)
                self.closeEditor.emit(
                    editor, qt.QAbstractItemDelegate.EditNextItem)
            event.accept()
            return True
        return super().eventFilter(editor, event)


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
    return float(match.group(1)), raycing.normalize_mu(match.group(2))


def _format_scan_display(value):
    """Format numeric scan values without changing their stored value."""
    if value is None:
        return 'None'
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, (float, list, tuple)):
            return _format_scan_display(parsed)
        return value
    if isinstance(value, float):
        return DISPLAY_NUMBER_FORMAT.format(value)
    if isinstance(value, (list, tuple)):
        left, right = ('[', ']') if isinstance(value, list) else ('(', ')')
        return left + ', '.join(
            _format_scan_display(item) for item in value) + right
    return str(value)


def _format_scan_value(value, unit):
    if not unit:
        return value
    return f'{value:g} {unit}'


def normalize_scan_targets(rows, beam_names):
    """Validate the scan-wide beam measurements stored in JSON."""
    if not isinstance(rows, list):
        raise ValueError('scanTargets must be a list')
    beam_names = set(beam_names)
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'Invalid scan target: {row!r}')
        beam = str(row.get('beam', ''))
        source = str(row.get('source', ''))
        flags = row.get('rayFlag', [])
        if type(flags) is int:
            flags = [flags]
        if (beam not in beam_names or source not in SCAN_TARGET_SOURCES or
                not isinstance(flags, (list, tuple)) or not flags or
                any(type(flag) is not int for flag in flags)):
            raise ValueError(f'Invalid scan target: {row!r}')
        flags = list(flags)
        key = (beam, source, tuple(flags))
        if key in seen:
            raise ValueError(f'Duplicate scan target: {key!r}')
        seen.add(key)
        result.append({'beam': beam, 'source': source, 'rayFlag': flags})
    return result


def _scan_target_column_names(targets):
    names = []
    for target in targets:
        flags = ','.join(map(str, target['rayFlag']))
        name = f"{target['beam']}.{target['source']}[{flags}]"
        if target['source'] in SCAN_SCALAR_SOURCES:
            names.append(name)
        else:
            names.extend((name + '.center', name + '.fwhm'))
    return names


def default_scan_description():
    return {
        'version': 1,
        'kind': 'timeline_recipe',
        'frames': 0,
        'output': copy.deepcopy(DEFAULT_OUTPUT),
        'items': [],
        'scanTargets': [],
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
                print(
                    f'Cannot create linspace from {start!r} to {stop!r}')
            if start_unit != stop_unit:
                print(
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


def _scan_track_plot_x(track, frames):
    count = int(track.get('duration', track.get('steps', 1)))
    planned = _value_sequence(track.get('values'), count)
    if not planned:
        return np.arange(count, dtype=float), 'Point'
    start = int(track.get('start', 0))
    target, prop = track.get('target'), track.get('property')
    values = []
    for point in range(count):
        frame = frames.get(f'frame_{start + point:04d}', {})
        patch = (frame.get('scene', {}) if target in SCENE_TARGETS else
                 frame.get('objects', {}).get(target, {}))
        values.append(patch.get(prop, planned[min(point, len(planned) - 1)]))
    parsed = [_split_numeric_unit(value) for value in values]
    if (all(number is not None and np.isfinite(number)
            for number, unit in parsed) and
            len({unit for number, unit in parsed}) == 1):
        unit = parsed[0][1]
        label = f"{track.get('target', '')}.{track.get('property', '')}"
        if unit:
            label += f' ({unit})'
        return np.array([number for number, unit in parsed]), label
    return np.arange(count, dtype=float), 'Point'


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

    def __init__(self, description=None, disabled_items=()):
        self.description = normalize_string_input(copy.deepcopy(
            description or default_scan_description()))
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
        self.disabled_items = set(disabled_items)
        if (self.expanded_frames is None and
                self.disabled_items):
            self.frame_count = 0
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
        for index, item in enumerate(self.items):
            if index in self.disabled_items:
                continue
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
                missing_id = f'frame_{missing:04d}'
                frames[missing_id] = self._normalize_expanded_frame(
                    missing_id, {}, missing)
        if 'output' in patch and frame_id in frames:
            frames[frame_id].pop('output', None)
        _merge_dict(frames[frame_id], patch, (frame_id,), self.warnings,
                    item_id)

    def _compile_track(self, frames, item):
        start = int(item.get('start', 0))
        duration = int(item.get('duration', item.get('steps', 1)))
        values = _value_sequence(item.get('values'), duration)
        var_sequences = OrderedDict()
        for var_name, var_spec in item.get('vars', {}).items():
            var_sequences[str(var_name)] = _value_sequence(
                var_spec, duration)
        item_id = item.get('id',
                           f"{item.get('target')}.{item.get('property')}")
        for offset in range(duration):
            if not values:
                break
            value = values[min(offset, len(values) - 1)]
            frame_index = start + offset
            frame_id = f'frame_{frame_index:04d}'
            variables = {'index': frame_index, 'frame': frame_id,
                         'value': value}
            item_var = item.get('var', item.get('variable'))
            if item_var:
                variables[str(item_var)] = value
            for var_name, var_values in var_sequences.items():
                variables[var_name] = (
                    var_values[min(offset, len(var_values) - 1)]
                    if var_values else None)
            patch = OrderedDict()
            _set_patch_value(patch, item.get('target'),
                             item.get('property'), value)
            if 'output' in item:
                patch['output'] = self._format_mapping(
                    item['output'], variables)
            if item_var or var_sequences or 'output' in item:
                patch['vars'] = variables
            self._merge_frame(frames, frame_index, patch, item_id)

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
        if self.description.get(FRAMES_CLEAN_KEY):
            self.frame_count = 0
            return OrderedDict()
        if self.expanded_frames is not None:
            frames = self._compile_expanded_frames()
        else:
            frames = self._make_empty_frames()
        for index, item in enumerate(self.items):
            if index in self.disabled_items:
                continue
            item_type = item.get('type', 'track')
            if item_type == 'event':
                self._compile_event(frames, item)
            elif item_type == 'loopBlock':
                self._compile_loop_block(frames, item)
            else:
                self._compile_track(frames, item)
        self.frame_count = len(frames)
        return frames


class ScanTargetSelector(qt.QWidget):
    """Scan-wide beam target rows shared by both scan creation dialogs."""

    def __init__(self, beam_names=(), scan_targets=(), plots_by_beam=None,
                 parent=None):
        super().__init__(parent)
        self.beamNames = list(beam_names)
        self.originalTargets = copy.deepcopy(list(scan_targets))
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(qt.QLabel('Scan targets (shared by all tracks)'))
        self.targetTree = qt.QTreeWidget()
        self.targetTree.setHeaderLabels(['Beam', 'Source', 'Ray states'])
        self.targetTree.setColumnWidth(0, 150)
        self.targetTree.setColumnWidth(1, 170)
        self.targetTree.setEditTriggers(qt.QAbstractItemView.DoubleClicked)
        self.targetDelegate = qt.ScanTargetDelegate(
            self.beamNames, SCAN_TARGET_SOURCES, plots_by_beam or {},
            self.targetTree)
        self.targetTree.setItemDelegate(self.targetDelegate)
        layout.addWidget(self.targetTree)
        buttons = qt.QHBoxLayout()
        add_target = qt.QPushButton('Add target')
        remove_target = qt.QPushButton('Remove target')
        add_target.clicked.connect(self.add_target_row)
        remove_target.clicked.connect(self.remove_target_row)
        buttons.addWidget(add_target)
        buttons.addWidget(remove_target)
        buttons.addStretch()
        layout.addLayout(buttons)
        for target in self.originalTargets:
            self.add_target_row(target)

    def add_target_row(self, target=None):
        # QPushButton.clicked passes a bool when the row is created by hand.
        if not isinstance(target, dict):
            target = {}
        item = qt.QTreeWidgetItem(self.targetTree)
        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)
        item.setText(0, str(target.get(
            'beam', self.beamNames[0] if self.beamNames else '')))
        item.setText(1, str(target.get('source', 'intensity')))
        flags = target.get('rayFlag', [1])
        if isinstance(flags, int):
            flags = [flags]
        item.setText(2, repr(tuple(flags)))
        self.targetTree.openPersistentEditor(item, 2)
        self.targetTree.setCurrentItem(item)

    def remove_target_row(self):
        item = self.targetTree.currentItem()
        if item is not None:
            self.targetTree.takeTopLevelItem(
                self.targetTree.indexOfTopLevelItem(item))

    def targets(self):
        rows = []
        for index in range(self.targetTree.topLevelItemCount()):
            item = self.targetTree.topLevelItem(index)
            flags = ast.literal_eval(str(item.text(2)))
            rows.append({
                'beam': str(item.text(0)),
                'source': str(item.text(1)),
                'rayFlag': flags,
                })
        return normalize_scan_targets(rows, self.beamNames)


class ScanInstructionDialog(qt.QDialog):
    """Dialog for adding a property event or track to the scan."""

    scanCreated = qt.Signal(dict)
    targetsChanged = qt.Signal(list)

    def __init__(self, catalog, start_frame=0, edit_item=None, parent=None,
                 beam_names=(), scan_targets=(), plots_by_beam=None,
                 initial_property=None):
        super().__init__(parent)
        self.setWindowIcon(qt.QIcon(_SCAN_ICON_PATH))
        self.catalog = list(catalog or [])
        self.propertyMap = {}
        self.propertyItems = {}
        self.selectedProperty = None
        self.editItem = copy.deepcopy(edit_item)
        self._editValuesEditable = True
        self.setWindowTitle('Add scan instruction')
        self.resize(680, 520)

        self.propertyTree = qt.QTreeWidget()
        self.propertyTree.setHeaderLabels(['Property', 'Current value'])
        self.propertyTree.setSelectionMode(
            qt.QAbstractItemView.SingleSelection)
        self.propertyTree.itemSelectionChanged.connect(
            self._selection_changed)

        self.startFrameEdit = _ScanLineEdit(str(int(start_frame)))
        self.minValueEdit = _ScanLineEdit('0')
        self.maxValueEdit = _ScanLineEdit('0')
        self.pointsEdit = _ScanLineEdit('1')
        _configure_scan_editors(self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.QLabel('Select scene or element property'))
        layout.addWidget(self.propertyTree)

        self.targetSelector = ScanTargetSelector(
            beam_names, scan_targets, plots_by_beam, self)
        layout.addWidget(self.targetSelector)

        form = qt.QFormLayout()
        form.addRow('First frame', self.startFrameEdit)
        form.addRow('Start value', self.minValueEdit)
        form.addRow('End value', self.maxValueEdit)
        form.addRow('Number of frames', self.pointsEdit)
        layout.addLayout(form)

        hint = qt.QLabel(
            'One frame uses start value only. Equal start/end '
            'values over multiple frames create a hold/pause.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.buttonBox = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)
        ok_button = self.buttonBox.button(qt.QDialogButtonBox.Ok)
        ok_button.setAutoDefault(False)
        ok_button.setDefault(False)

        self._populate_tree()
        if self.editItem is not None:
            self._apply_edit_item()
        elif initial_property is not None:
            target, property_name = initial_property
            item = self.propertyItems.get(f'{target}::{property_name}')
            if item is not None:
                self.propertyTree.setCurrentItem(item)
                lower, upper = _scan_default_bounds(
                    self.selectedProperty.get('value'))
                self.minValueEdit.setText(str(lower))
                self.maxValueEdit.setText(str(upper))
                self.pointsEdit.setText('11')
                self.setWindowTitle(
                    f'Create scan: {target}.{property_name}')
        qt.QTimer.singleShot(0, self.startFrameEdit.setFocus)

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
                child = qt.QTreeWidgetItem([
                    str(prop.get('name')), _format_scan_display(value)])
                child.setData(0, qt.Qt.UserRole, key)
                child.setToolTip(1, str(value))
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
        target, property_name = _scan_item_target_property(self.editItem)
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
        _set_scan_value_validators(self, property_name)
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
        _set_scan_value_validators(
            self, self.selectedProperty.get('property'))
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
            print('Select a property first')
        start = int(self.startFrameEdit.text())
        points = int(self.pointsEdit.text())
        if points < 1:
            print('Number of frames must be at least 1')
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
            targets = self.targetSelector.targets()
            if self._current_property() is None and self.editItem is None:
                if targets == self.targetSelector.originalTargets:
                    raise ValueError('Select a property or change scan targets')
                item = None
            else:
                item = self.scan_item()
                BaseScan({'items': [item]}).compile_frames()
        except Exception as exc:
            qt.QMessageBox.warning(self, 'Invalid instruction',
                                   f'Cannot create instruction: {exc}')
            return
        if targets != self.targetSelector.originalTargets:
            self.targetsChanged.emit(targets)
        if item is not None:
            self.scanCreated.emit(item)
        super().accept()


class _ScanLivePlotWindow(qt.QWidget):
    """Live target values against one scan track's planned values."""

    closed = qt.Signal(int)

    def __init__(self, plot_id, track, x, x_label, columns, value_start,
                 target_label, parent=None):
        super().__init__(parent, qt.Qt.Window)
        self.plotId = plot_id
        self.valueStart = value_start
        self.valueCount = len(columns)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.setWindowTitle(
            f"Scan: {track.get('id', 'track')} / {target_label}")
        self.setWindowIcon(qt.QIcon(_SCAN_ICON_PATH))
        start = int(track.get('start', 0))
        self.frameToPoint = {
            f'frame_{start + point:04d}': point
            for point in range(len(x))}
        self.y = [np.full(len(x), np.nan) for name in columns]

        figure = Figure(figsize=(7, max(3, 2.2 * len(columns))))
        self.axes = np.atleast_1d(
            figure.subplots(len(columns), 1, sharex=True))
        self.lines = []
        for axis, name, values in zip(self.axes, columns, self.y):
            line, = axis.plot(x, values, '.-', markersize=4)
            axis.set_ylabel(name, fontsize=8)
            axis.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            axis.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            self.lines.append(line)
        self.axes[-1].set_xlabel(x_label)
        lower, upper = float(np.min(x)), float(np.max(x))
        pad = max(1., abs(lower) * .01) if lower == upper else 0.
        self.axes[-1].set_xlim(lower - pad, upper + pad)
        figure.set_tight_layout(True)

        self.canvas = qt.FigCanvas(figure)
        self.canvas.setMinimumHeight(max(300, 220 * len(columns)))
        scroll = qt.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.NavigationToolbar(self.canvas, self))
        layout.addWidget(scroll)
        self.resize(720, min(850, max(400, 230 * len(columns))))

    def add_frame(self, frame_id, values):
        point = self.frameToPoint.get(frame_id)
        if point is None:
            return
        target_values = values[
            self.valueStart:self.valueStart + self.valueCount]
        for axis, line, data, value in zip(
                self.axes, self.lines, self.y, target_values):
            data[point] = (float(value) if value is not None and
                           np.isfinite(value) else np.nan)
            line.set_ydata(data)
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self.closed.emit(self.plotId)
        super().closeEvent(event)


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
    trackEnabledChanged = qt.Signal(int, bool)
    trackEditRequested = qt.Signal(int)
    framePopulateRequested = qt.Signal()
    frameClearRequested = qt.Signal()

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
        self.populateFramesButton = self._make_tool_button(
            'db_update-2.png',
            'Bake the current preview into an explicit frame sequence')
        self.clearFramesButton = self._make_tool_button(
            'db_remove-2.png',
            'Remove the explicit frame sequence and keep scan tracks')
        info_controls = qt.QHBoxLayout()
        controls.addWidget(self.addInstructionButton)
        controls.addWidget(self.deleteTrackButton)
        controls.addWidget(self.loadScanButton)
        controls.addWidget(self.saveScanButton)
        controls.addWidget(self.populateFramesButton)
        controls.addWidget(self.clearFramesButton)
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
        self.trackTable.setItemDelegate(_ScanTrackDelegate(self))
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
        self.populateFramesButton.clicked.connect(
            self.framePopulateRequested.emit)
        self.clearFramesButton.clicked.connect(
            self.frameClearRequested.emit)
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

    def _has_frame_sequence(self):
        description = self.scan.description
        for key in ['expandedFrames', 'frameDict']:
            if isinstance(description.get(key), dict) and description[key]:
                return True
        if isinstance(description.get('frames'), dict) and \
                description['frames']:
            return True
        return any(_looks_like_frame_key(key) for key in description)

    def _has_frames_to_clean(self):
        return self._has_frame_sequence() or bool(self.frameIds)

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
        return str(self.outputTemplateEdit.text()).strip()

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
        self.populateFramesButton.setEnabled(
            has_frames or bool(self.scan.items) or self._has_frame_sequence())
        self.clearFramesButton.setEnabled(self._has_frames_to_clean())

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
                    table_item = qt.QTableWidgetItem(
                        _format_scan_display(value))
                    table_item.setData(qt.RAW_VALUE_ROLE, value)
                    table_item.setToolTip(str(value))
                    flags = table_item.flags()
                    if self._track_column_is_editable(item, col):
                        flags |= qt.Qt.ItemIsEditable
                    else:
                        flags &= ~qt.Qt.ItemIsEditable
                    if col == self.TRACK_COL_ID:
                        flags |= qt.Qt.ItemIsUserCheckable
                    table_item.setFlags(flags)
                    if col == self.TRACK_COL_ID:
                        table_item.setCheckState(
                            qt.Qt.Checked if row not in self.scan.disabled_items
                            else qt.Qt.Unchecked)
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
        if column == self.TRACK_COL_ID:
            enabled = table_item.checkState() == qt.Qt.Checked
            if enabled == (row in self.scan.disabled_items):
                self.trackEnabledChanged.emit(row, enabled)
            return
        if not self._track_column_is_editable(item, column):
            return
        if column in self.TRACK_VALUE_COLUMNS:
            value = table_item.text()
            self._updatingTracks = True
            table_item.setData(qt.RAW_VALUE_ROLE, value)
            self._updatingTracks = False
            key = 'startValue' if column == self.TRACK_COL_VALUE_START else \
                'endValue'
            self.trackTimingChanged.emit(row, {key: value})
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


class GlowScanMixin:
    """Scan editing, playback, and persistence for the Glow widget."""

    def _scan_description_from_input(self, scanDescription):
        if scanDescription is None:
            description = default_scan_description()
        elif isinstance(scanDescription, BaseScan):
            description = scanDescription.description
        elif isinstance(scanDescription, dict):
            description = scanDescription
        elif isinstance(scanDescription, (list, tuple)):
            description = default_scan_description()
            description['items'] = list(scanDescription)
        elif isinstance(scanDescription, (str, os.PathLike)):
            source = os.fspath(scanDescription).strip()
            if not source:
                description = default_scan_description()
            elif os.path.exists(source):
                with open(source, 'r', encoding='utf-8') as jsonFile:
                    description = json.load(jsonFile)
            else:
                description = json.loads(source)
        else:
            print('scanDescription must be a dict, list, JSON string, '
                  'JSON file path, BaseScan or None')

        description = copy.deepcopy(description)
        if 'items' not in description and 'tracks' in description:
            description['items'] = copy.deepcopy(description['tracks'])
        description.setdefault('version', 1)
        if BaseScan(description).expanded_frames is not None:
            description.setdefault('kind', 'expanded_frames')
        else:
            description.setdefault('kind', 'timeline_recipe')
            description.setdefault('frames', 0)
            description.setdefault('items', [])
        beam_names = self.customGlWidget.beamline.beamNamesDict
        description['scanTargets'] = normalize_scan_targets(
            description.get('scanTargets', []), beam_names)
        output = description.setdefault('output', {})
        output.setdefault('glowFrameName', DEFAULT_OUTPUT['glowFrameName'])
        return self._scan_portable_description(description)

    def setScanDescription(self, scanDescription):
        self.scanDescription = self._scan_description_from_input(
            scanDescription)
        self._scanDisabledItems = set()
        self.refreshScanPanel()

    def setScanTargets(self, targets):
        if self.scanRunning:
            return
        self.scanDescription['scanTargets'] = normalize_scan_targets(
            targets, self.customGlWidget.beamline.beamNamesDict)

    def _scan_plot_names_by_beam(self):
        beamline = self.customGlWidget.beamline
        beam_tags = beamline.beamNamesDict
        result = {name: [] for name in beam_tags}
        qook = getattr(self, 'parentRef', None)
        plot_root = getattr(qook, 'rootPlotItem', None)
        if plot_root is not None:
            plots = qook.treeToDict(plot_root)
        else:
            layout = getattr(beamline, 'layoutStr', None) or {}
            plots = layout.get('Project', {}).get('plots', {})
        if not isinstance(plots, dict):
            return result
        for index, (plot_id, props) in enumerate(plots.items()):
            if not isinstance(props, dict):
                continue
            plot_name = props.get('name')
            if not plot_name and plot_root is not None:
                plot_item = plot_root.child(index, 0)
                plot_name = plot_item.text() if plot_item is not None else None
            plot_name = str(plot_name or plot_id)
            plot_beam = props.get('beam')
            if isinstance(plot_beam, list):
                plot_beam = tuple(plot_beam)
            for beam_name, beam_tag in beam_tags.items():
                if plot_beam in (beam_name, beam_tag):
                    if plot_name not in result[beam_name]:
                        result[beam_name].append(plot_name)
        return result

    def _scan_sync_output_template(self):
        if not hasattr(self, 'scanWidget'):
            return
        self.scanDescription.setdefault('output', {})[
            'glowFrameName'] = self.scanWidget.output_template()

    def _scan_object_name_map(self):
        mapping = {}
        bl = getattr(self.customGlWidget, 'beamline', None)
        if bl is None:
            return mapping
        for object_id, oeLine in getattr(bl, 'oesDict', {}).items():
            try:
                name = getattr(oeLine[0], 'name', None)
            except Exception:
                name = None
            if name:
                mapping[str(object_id)] = name
        for dict_name in ['materialsDict', 'fesDict']:
            for object_id, obj in getattr(bl, dict_name, {}).items():
                name = getattr(obj, 'name', None)
                if name:
                    mapping[str(object_id)] = name
        return mapping

    def _scan_portable_target(self, target, fallback=None):
        if target in SCENE_TARGETS:
            return 'Scene'
        return self._scan_object_name_map().get(str(target),
                                                fallback or target)

    def _scan_portable_objects(self, objects):
        portable = OrderedDict()
        for target, patch in (objects or {}).items():
            portable[self._scan_portable_target(target)] = copy.deepcopy(patch)
        return portable

    def _scan_portable_frame(self, frame):
        if not isinstance(frame, dict):
            return copy.deepcopy(frame)
        frame = copy.deepcopy(frame)
        if 'objects' in frame:
            frame['objects'] = self._scan_portable_objects(frame['objects'])
        for target in list(frame.keys()):
            if target in FRAME_SECTIONS or target in SCENE_PROPERTY_NAMES:
                continue
            value = frame.pop(target)
            portable_target = self._scan_portable_target(target)
            if 'objects' in frame:
                existing = frame['objects'].setdefault(
                    portable_target, OrderedDict())
                if isinstance(existing, dict) and isinstance(value, dict):
                    existing.update(value)
                else:
                    frame['objects'][portable_target] = value
            else:
                frame[portable_target] = value
        return frame

    def _scan_portable_item(self, item):
        item = copy.deepcopy(item)
        fallback = item.pop('targetName', None)
        if 'target' in item:
            item['target'] = self._scan_portable_target(
                item['target'], fallback=fallback)
        if 'objects' in item:
            item['objects'] = self._scan_portable_objects(item['objects'])
        return item

    def _scan_portable_description(self, description):
        description = copy.deepcopy(description)
        description.pop('tracks', None)
        if 'items' in description:
            description['items'] = [
                self._scan_portable_item(item)
                for item in description.get('items', [])]
        for frame_key in ['expandedFrames', 'frameDict']:
            if isinstance(description.get(frame_key), dict):
                description[frame_key] = OrderedDict(
                    (key, self._scan_portable_frame(frame))
                    for key, frame in description[frame_key].items())
        if isinstance(description.get('frames'), dict):
            description['frames'] = OrderedDict(
                (key, self._scan_portable_frame(frame))
                for key, frame in description['frames'].items())
        for key, frame in list(description.items()):
            if re.match(r'^frame_\d+$', str(key)):
                description[key] = self._scan_portable_frame(frame)
        return description

    def saveScanToJson(self):
        self._scan_sync_output_template()
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("JSON files (*.json)")
        self._scan_set_dialog_directory(saveDialog)
        if not saveDialog.exec_():
            return
        filename = saveDialog.selectedFiles()[0]
        if not filename.lower().endswith('.json'):
            filename = "{0}.json".format(filename)
        try:
            description = self._scan_portable_description(
                self.scanDescription)
            with open(filename, 'w', encoding='utf-8',
                      newline='\r\n') as jsonFile:
                json.dump(description, jsonFile, indent=2)
                jsonFile.write('\n')
            config.put(config.configPaths, 'Glow', 'scan', filename)
            config.write_configs()
        except Exception as exc:
            qt.QMessageBox.warning(
                self, 'Save scan', f'Cannot save scan JSON: {exc}')

    def loadScanFromJson(self):
        if self.scanRunning:
            qt.QMessageBox.warning(
                self, 'Load scan',
                'Stop the running scan before loading another one.')
            return
        loadDialog = qt.QFileDialog()
        loadDialog.setFileMode(qt.QFileDialog.ExistingFile)
        loadDialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
        loadDialog.setNameFilter("JSON files (*.json)")
        self._scan_set_dialog_directory(loadDialog)
        if not loadDialog.exec_():
            return
        filename = loadDialog.selectedFiles()[0]
        try:
            self.setScanDescription(filename)
            config.put(config.configPaths, 'Glow', 'scan', filename)
            config.write_configs()
        except Exception as exc:
            qt.QMessageBox.warning(
                self, 'Load scan', f'Cannot load scan JSON: {exc}')
            return
        self.openScanPanel()

    def addScanItem(self, item):
        item = self._scan_portable_item(item)
        self.scanDescription.setdefault('items', []).append(item)
        start, duration = self._scan_item_span(item)
        frames_value = self.scanDescription.get('frames', 0)
        if isinstance(frames_value, dict):
            frames_key = 'frameCount'
            frames_value = self.scanDescription.get(
                frames_key, len(frames_value))
        else:
            frames_key = 'frames'
        self.scanDescription[frames_key] = max(
            int(frames_value or 0), start + duration)
        self.refreshScanPanel()
        self.openScanPanel()

    def setScanOutputDirectory(self, directory):
        if directory is None:
            self.scanOutputDirectory = None
            return
        directory = os.fspath(directory).strip()
        self.scanOutputDirectory = (
            os.path.abspath(directory) if directory else None)

    def _scan_resolve_output_filename(self, filename):
        filename = os.fspath(filename)
        if os.path.isabs(filename):
            return filename
        directory = getattr(self, 'scanOutputDirectory', None)
        if directory:
            return os.path.join(directory, filename)
        return filename

    def _scan_set_dialog_directory(self, dialog):
        directory = getattr(self, 'scanOutputDirectory', None)
        if directory:
            dialog.setDirectory(directory)

    def setScanOutputTemplate(self, template):
        template = str(template).strip()
        self.scanDescription.setdefault('output', {})[
            'glowFrameName'] = template
        self.refreshScanPanel()

    def _scan_remove_frame_sequence(self):
        self.scanDescription.pop('expandedFrames', None)
        self.scanDescription.pop('frameDict', None)
        if isinstance(self.scanDescription.get('frames'), dict):
            self.scanDescription.pop('frames', None)
        for key in list(self.scanDescription.keys()):
            if re.match(r'^frame_\d+$', str(key)):
                self.scanDescription.pop(key, None)

    def populateScanFrames(self):
        if self.scanRunning:
            qt.QMessageBox.warning(
                self, 'Populate frames',
                'Stop the running scan before changing its frame sequence.')
            return
        self._scan_sync_output_template()
        description = copy.deepcopy(self.scanDescription)
        description.pop(FRAMES_CLEAN_KEY, None)
        scan = BaseScan(
            description, disabled_items=getattr(self, '_scanDisabledItems', ()))
        frames = scan.compile_frames()
        if not frames:
            return
        self._scan_remove_frame_sequence()
        self.scanDescription.pop(FRAMES_CLEAN_KEY, None)
        self.scanDescription['expandedFrames'] = OrderedDict(
            (key, self._scan_portable_frame(frame))
            for key, frame in frames.items())
        self.scanDescription['frames'] = len(frames)
        self.refreshScanPanel()

    def clearScanFrames(self):
        if self.scanRunning:
            qt.QMessageBox.warning(
                self, 'Clean frames',
                'Stop the running scan before changing its frame sequence.')
            return
        self._scan_remove_frame_sequence()
        self.scanDescription[FRAMES_CLEAN_KEY] = True
        self.scanDescription['frames'] = 0
        self.refreshScanPanel()

    def deleteScanItem(self, item_index):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        del items[item_index]
        self._scanDisabledItems = {
            index - (index > item_index)
            for index in getattr(self, '_scanDisabledItems', ())
            if index != item_index}
        self.scanDescription['frames'] = self._scan_recipe_frame_count(items)
        self.refreshScanPanel()

    def replaceScanItem(self, item_index, item):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        items[item_index] = self._scan_portable_item(item)
        self.scanDescription['frames'] = self._scan_recipe_frame_count(items)
        self.refreshScanPanel()

    def setScanItemEnabled(self, item_index, enabled):
        if self.scanRunning:
            self.refreshScanPanel()
            return
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        disabled = getattr(self, '_scanDisabledItems', set())
        if enabled:
            disabled.discard(item_index)
        else:
            disabled.add(item_index)
        self._scanDisabledItems = disabled
        self.refreshScanPanel()

    def editScanItem(self, item_index):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        dialog = ScanInstructionDialog(
            self.scanInstructionCatalog(), edit_item=items[item_index],
            parent=self,
            beam_names=self.customGlWidget.beamline.beamNamesDict,
            scan_targets=self.scanDescription.get('scanTargets', []),
            plots_by_beam=self._scan_plot_names_by_beam())
        dialog.targetsChanged.connect(self.setScanTargets)
        dialog.scanCreated.connect(
            lambda item, row=item_index: self.replaceScanItem(row, item))
        dialog.exec_()

    def updateScanItemTiming(self, item_index, timing):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        item = items[item_index]
        current_start, current_frames = self._scan_item_span(item)
        start = max(0, int(timing.get(
            'startFrame', timing.get('start', current_start))))
        frames = max(1, int(timing.get('frames', current_frames)))
        item_type = item.get('type', 'track')
        if item_type == 'loopBlock':
            item['start'] = start
        elif item_type == 'event':
            item.pop('start', None)
            item['frame'] = start
            self._scan_update_event_value(item, timing)
            if frames > 1:
                item['duration'] = frames
            else:
                item.pop('duration', None)
                item.pop('steps', None)
        else:
            item['start'] = start
            item['duration'] = frames
            self._scan_update_track_values(item, timing, frames)
        self.scanDescription['frames'] = self._scan_recipe_frame_count(items)
        self.refreshScanPanel()

    def _scan_update_track_values(self, item, timing, frames):
        has_start = 'startValue' in timing
        has_end = 'endValue' in timing
        if not has_start and not has_end:
            values = item.get('values')
            if isinstance(values, dict) and values.get('type') in [
                    'linspace', 'constant']:
                values['steps'] = frames
            return

        values = item.get('values')
        if isinstance(values, dict):
            value_type = values.get('type')
            if value_type == 'linspace':
                if has_start:
                    values['start'] = timing['startValue']
                if has_end:
                    values['stop'] = timing['endValue']
                values['steps'] = frames
                return
            if value_type == 'constant':
                start_value = timing.get('startValue', values.get('value'))
                end_value = timing.get('endValue', values.get('value'))
                if start_value != end_value and frames > 1:
                    item['values'] = {
                        'type': 'linspace',
                        'start': start_value,
                        'stop': end_value,
                        'steps': frames,
                        }
                else:
                    values['value'] = start_value
                    values['steps'] = frames
                return

        value = timing.get('startValue', timing.get('endValue', values))
        if has_start and has_end and timing['startValue'] != \
                timing['endValue'] and frames > 1:
            item['values'] = {
                'type': 'linspace',
                'start': timing['startValue'],
                'stop': timing['endValue'],
                'steps': frames,
                }
        else:
            item['values'] = {
                'type': 'constant',
                'value': value,
                'steps': frames,
                }

    def _scan_update_event_value(self, item, timing):
        if 'startValue' not in timing and 'endValue' not in timing:
            return
        value = timing.get('startValue', timing.get('endValue'))
        patches = []
        for patch in item.get('objects', {}).values():
            if isinstance(patch, dict):
                for key in patch.keys():
                    patches.append((patch, key))
        scene = item.get('scene', {})
        if isinstance(scene, dict):
            for key in scene.keys():
                patches.append((scene, key))
        if len(patches) != 1:
            return
        patch, key = patches[0]
        patch[key] = value

    def _scan_recipe_frame_count(self, items):
        frame_count = 0
        for item in items:
            start, duration = self._scan_item_span(item)
            frame_count = max(frame_count, start + duration)
        return frame_count

    def _scan_item_span(self, item):
        item_type = item.get('type', 'track')
        if item_type == 'event':
            start = int(item.get('frame', item.get('start', 0)))
            duration = int(item.get('duration', item.get('steps', 1)))
        elif item_type == 'loopBlock':
            start = int(item.get('start', 0))
            duration = BaseScan({'items': [item]})._loop_block_length(item)
        else:
            start = int(item.get('start', 0))
            duration = int(item.get('duration', item.get('steps', 1)))
        return start, max(1, duration)

    def _scan_format_value(self, value):
        if hasattr(value, 'uuid'):
            return value.uuid
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def _scan_scene_properties(self):
        props = []
        for name, fields in SCAN_SCENE_COMPONENTS.items():
            value = getattr(self.customGlWidget, name,
                            DEFAULT_SCENE_SETTINGS.get(name))
            if value is None:
                continue
            for index, field in enumerate(fields):
                try:
                    field_value = value[index]
                except Exception:
                    field_value = ''
                props.append({
                    'name': f'{name}.{field}',
                    'value': self._scan_format_value(field_value),
                    })
        return props

    def _scan_angle_value(self, value):
        if isinstance(value, str):
            if re.search(r'[A-Za-z]', value):
                return value
            try:
                return f'{float(value) * 1e3:g} mrad'
            except ValueError:
                return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f'{float(value) * 1e3:g} mrad'
        return value

    def _scan_split_compound_property(self, name, value, surfaceIndex=0):
        if isinstance(value, str):
            parsed = raycing.parametrize(value)
        else:
            parsed = value
        fields = raycing.compoundArgs.get(name)
        if name == 'blades':
            if not isinstance(parsed, dict):
                return None
            return [{
                'name': f'{name}.{field}',
                'value': self._scan_format_value(parsed[field]),
                } for field in parsed.keys()]
        if not fields or not isinstance(parsed, (list, tuple, np.ndarray)):
            return None
        if name in SCAN_LIMIT_PROPERTIES and raycing.is_sequence(parsed[0]):
            parsed = [parsed[0][surfaceIndex], parsed[1][surfaceIndex]]
        items = []
        for index, field in enumerate(fields):
            if index >= len(parsed):
                break
            items.append({
                'name': f'{name}.{field}',
                'value': self._scan_format_value(parsed[index]),
                })
        return items

    def _scan_extra_property_names(self, oeObj):
        names = []
        if hasattr(oeObj, 'blades'):
            names.append('blades')
        for name in SCAN_LIMIT_PROPERTIES:
            if hasattr(oeObj, name):
                names.append(name)
        if is_screen(oeObj) or is_aperture(oeObj):
            for name in SCAN_AXIS_PROPERTIES:
                if hasattr(oeObj, name):
                    names.append(name)
        return names

    def _scan_init_defaults(self, oeObj):
        try:
            return OrderedDict(raycing.get_params(raycing.get_obj_str(oeObj)))
        except Exception:
            return OrderedDict()

    def _scan_is_default_scannable(self, defaults, name):
        root = str(name).split('.', 1)[0]
        if root not in defaults:
            return False
        default = defaults[root]
        return default is not None and not isinstance(default, (str, bool))

    def _scan_element_property_value(self, oeObj, name, value):
        initValue = getattr(oeObj, f'_{name}Init', None)
        if initValue is not None and str(initValue).lower() != 'none':
            value = initValue
        if name in SCAN_ANGLE_PROPERTIES:
            value = self._scan_angle_value(value)
        return self._scan_format_value(value)

    def _scan_element_properties(self):
        bl = self.customGlWidget.beamline
        blName = getattr(bl, 'name', None)
        catalog = []
        for oeid, oeLine in bl.oesDict.items():
            oeObj = oeLine[0]
            defaults = self._scan_init_defaults(oeObj)
            try:
                props = raycing.get_init_kwargs(oeObj, compact=False,
                                                blname=blName)
            except Exception:
                props = {}
            for name in self._scan_extra_property_names(oeObj):
                if name not in props:
                    props[name] = getattr(oeObj, name)
            prop_items = []
            for name, value in props.items():
                if name in ['uuid', 'name'] or str(name).endswith('rbk'):
                    continue
                if not self._scan_is_default_scannable(defaults, name):
                    continue
                value = self._scan_element_property_value(
                    oeObj, name, value)
                compound_items = self._scan_split_compound_property(
                    name, value, getattr(oeObj, 'curSurface', 0))
                if compound_items is not None:
                    prop_items.extend(compound_items)
                else:
                    prop_items.append({
                        'name': name,
                        'value': value,
                        })
            if prop_items:
                catalog.append({
                    'target': getattr(oeObj, 'name', oeid),
                    'name': getattr(oeObj, 'name', oeid),
                    'properties': prop_items,
                    })
        return catalog

    def scanInstructionCatalog(self):
        catalog = [{
            'target': 'Scene',
            'name': 'Scene',
            'properties': self._scan_scene_properties(),
            }]
        catalog.extend(self._scan_element_properties())
        return catalog

    def openScanInstructionDialog(self, frame_index=0):
        dialog = ScanInstructionDialog(
            self.scanInstructionCatalog(), start_frame=frame_index,
            parent=self,
            beam_names=self.customGlWidget.beamline.beamNamesDict,
            scan_targets=self.scanDescription.get('scanTargets', []),
            plots_by_beam=self._scan_plot_names_by_beam())
        dialog.targetsChanged.connect(self.setScanTargets)
        dialog.scanCreated.connect(self.addScanItem)
        dialog.exec_()

    def _scan_status(self, progress, message):
        signal = getattr(self.customGlWidget, 'QookSignal', None)
        if signal is not None:
            signal.emit((progress, message))

    def _scan_auto_update_state(self):
        if self.parentRef is not None and hasattr(
                self.parentRef, 'isGlowAutoUpdate'):
            return bool(self.parentRef.isGlowAutoUpdate)
        return bool(getattr(self.customGlWidget, 'autoUpdate', True))

    def _set_scan_auto_update(self, state):
        state = bool(state)
        if self.parentRef is not None and hasattr(
                self.parentRef, 'isGlowAutoUpdate'):
            self.parentRef.isGlowAutoUpdate = state
        self.customGlWidget.set_auto_update(state)

    def _scan_target_id_map(self):
        mapping = {}
        bl = getattr(self.customGlWidget, 'beamline', None)
        if bl is None:
            return mapping
        for object_id, oeLine in getattr(bl, 'oesDict', {}).items():
            mapping[str(object_id)] = object_id
            try:
                name = getattr(oeLine[0], 'name', None)
            except Exception:
                name = None
            if name:
                mapping[str(name)] = object_id
        for dict_name in ['materialsDict', 'fesDict']:
            for object_id, obj in getattr(bl, dict_name, {}).items():
                mapping[str(object_id)] = object_id
                name = getattr(obj, 'name', None)
                if name:
                    mapping[str(name)] = object_id
        return mapping

    def _scan_resolve_target_id(self, object_id):
        if object_id in SCENE_TARGETS:
            return object_id
        return self._scan_target_id_map().get(str(object_id), object_id)

    def _scan_object_for_id(self, object_id):
        object_id = self._scan_resolve_target_id(object_id)
        bl = self.customGlWidget.beamline
        if object_id in bl.oesDict:
            return bl.oesDict[object_id][0]
        if object_id in bl.materialsDict:
            return bl.materialsDict[object_id]
        if object_id in bl.fesDict:
            return bl.fesDict[object_id]
        return None

    def _scan_snapshot_value(self, obj, prop):
        prop = prop.split('.')[0]
        no_value = object()
        raw_value = getattr(obj, f'_{prop}', no_value)
        resolved_value = getattr(obj, prop, raw_value)
        raw_value_attr = getattr(obj, f'_{prop}Val', no_value)
        init_value = getattr(obj, f'_{prop}Init', no_value)
        if raw_value is not no_value and raw_value is not None and \
                raw_value_attr is None and \
                raycing.is_auto_align_value(raw_value):
            if init_value is not no_value and init_value is not None:
                value = init_value
            else:
                value = raw_value
        elif raw_value is not no_value and raw_value is not None and \
                raw_value_attr is None:
            value = raw_value
        else:
            value = resolved_value
        if hasattr(value, 'uuid'):
            value = value.uuid
        elif hasattr(value, 'name') and prop.lower().startswith(
                ('mater', 'tlay', 'blay', 'coat', 'substrate')):
            value = value.name
        return copy.deepcopy(value)

    def _scan_collect_initial_state(self, frames):
        objects = OrderedDict()
        scene = OrderedDict()
        for frame in frames.values():
            for object_id, patch in frame.get('objects', {}).items():
                object_id = self._scan_resolve_target_id(object_id)
                obj = self._scan_object_for_id(object_id)
                if obj is None:
                    continue
                object_state = objects.setdefault(object_id, OrderedDict())
                for prop in patch.keys():
                    root_prop = prop.split('.')[0]
                    if root_prop not in object_state:
                        object_state[root_prop] = self._scan_snapshot_value(
                            obj, root_prop)
            for prop in frame.get('scene', {}).keys():
                root_prop = prop.split('.')[0]
                if root_prop not in scene:
                    scene[root_prop] = copy.deepcopy(
                        getattr(self.customGlWidget, root_prop, None))
        return {'objects': objects, 'scene': scene}

    def _scan_apply_frame(self, frame):
        for object_id, patch in frame.get('objects', {}).items():
            object_id = self._scan_resolve_target_id(object_id)
            self.customGlWidget.update_beamline(
                object_id, dict(patch), sender='scan')
        if frame.get('scene'):
            scene = self._scan_expand_scene_patch(frame['scene'])
            self.applySceneProperties(scene)

    def _scan_expand_scene_patch(self, patch):
        scene = OrderedDict()
        for name, value in patch.items():
            if name == 'offsetCoord' or name.startswith('offsetCoord.'):
                name = name.replace('offsetCoord', 'coordOffset', 1)
            if '.' not in name:
                scene[name] = self._scan_parse_scene_value(name, value)
                continue
            root, field = name.split('.', 1)
            fields = SCAN_SCENE_COMPONENTS.get(root)
            if fields is None or field not in fields:
                scene[name] = self._scan_parse_scene_value(name, value)
                continue
            if root not in scene:
                current = copy.deepcopy(
                    getattr(self.customGlWidget, root,
                            DEFAULT_SCENE_SETTINGS.get(root)))
                if hasattr(current, 'tolist'):
                    current = current.tolist()
                else:
                    current = list(current)
                scene[root] = current
            scene[root][fields.index(field)] = self._scan_parse_scene_value(
                root, value)
        return scene

    def _scan_parse_scene_value(self, name, value):
        if not isinstance(value, str):
            return value
        root = name.split('.')[0]
        reference = getattr(self.customGlWidget, name,
                            getattr(self.customGlWidget, root,
                                    DEFAULT_SCENE_SETTINGS.get(root)))
        text = value.strip()
        if isinstance(reference, bool):
            return text.lower() in ['1', 'true', 'yes', 'on']
        try:
            parsed = raycing.parametrize(text)
        except Exception:
            parsed = text
        if root in SCAN_SCENE_COMPONENTS:
            return float(parsed)
        if isinstance(reference, set):
            if isinstance(parsed, (list, tuple, set)):
                return set(parsed)
            if parsed in ['', None]:
                return set()
            return {parsed}
        if isinstance(reference, np.ndarray):
            return np.array(parsed)
        if isinstance(reference, (list, tuple)):
            return parsed
        if isinstance(reference, (int, np.integer)) and not isinstance(
                reference, bool):
            return int(parsed)
        if isinstance(reference, (float, np.floating)):
            return float(parsed)
        return parsed

    def _scan_restore_initial_state(self):
        if not self.scanInitialState:
            return False
        hasObjects = bool(self.scanInitialState.get('objects'))
        for object_id, patch in self.scanInitialState.get(
                'objects', {}).items():
            self.customGlWidget.update_beamline(
                object_id, dict(patch), sender='scan')
        scene = self.scanInitialState.get('scene', {})
        if scene:
            self.applySceneProperties(dict(scene))
        else:
            self.customGlWidget.glDraw()
        return hasObjects

    def _scan_close_csv(self):
        csv_file = getattr(self, '_scanCsvFile', None)
        self._scanCsvFile = None
        self._scanCsvWriter = None
        if csv_file is not None:
            csv_file.close()

    def _scan_open_csv(self, scan):
        self._scan_close_csv()
        self._scanActiveTargets = copy.deepcopy(
            self.scanDescription.get('scanTargets', []))
        if not self._scanActiveTargets:
            return

        self._scanCsvTracks = [
            item for index, item in enumerate(scan.items)
            if (item.get('type', 'track') == 'track' and
                index not in scan.disabled_items)]
        first_id = (self._scanCsvTracks[0].get('id', 'scan')
                    if self._scanCsvTracks else 'scan')
        stem = re.sub(r'[^A-Za-z0-9._-]+', '_', str(first_id)).strip('._')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = self._scan_resolve_output_filename(
            f'{stem or "scan"}_{timestamp}.csv')
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        self._scanTrackState = {}
        catalog = self.scanInstructionCatalog()
        for track in self._scanCsvTracks:
            key = (str(track['target']), str(track['property']))
            prop = find_catalog_property(catalog, *key)
            self._scanTrackState[key] = prop.get('value') if prop else None

        header = ['frame']
        header.extend(
            f"track.{i}.{track.get('id', '')}"
            for i, track in enumerate(self._scanCsvTracks, 1))
        header.extend(_scan_target_column_names(self._scanActiveTargets))
        try:
            self._scanCsvFile = open(
                filename, 'w', newline='', encoding='utf-8')
            self._scanCsvWriter = csv.writer(self._scanCsvFile)
            self._scanCsvWriter.writerow(header)
            self._scanCsvFile.flush()
        except Exception:
            self._scan_close_csv()
            raise

    def _scan_effective_track_values(self, frame):
        for target, patch in frame.get('objects', {}).items():
            target_name = str(self._scan_portable_target(target))
            for prop, value in patch.items():
                self._scanTrackState[(target_name, str(prop))] = value
        for prop, value in frame.get('scene', {}).items():
            self._scanTrackState[('Scene', str(prop))] = value
        return [
            self._scanTrackState.get(
                (str(track['target']), str(track['property'])))
            for track in self._scanCsvTracks]

    def _scan_write_csv_row(self, frame_id, frame):
        if self._scanCsvWriter is None:
            return
        track_values = self._scan_effective_track_values(frame)
        beam_values = self.customGlWidget.scan_target_values(
            self._scanActiveTargets)
        self._scanCsvWriter.writerow(
            [frame_id, *track_values, *beam_values])
        self._scanCsvFile.flush()
        return beam_values

    def _scan_close_live_plots(self):
        for window in tuple(getattr(self, '_scanLivePlots', {}).values()):
            window.close()
        self._scanLivePlots = {}

    def _scan_plot_closed(self, plot_id):
        self._scanLivePlots.pop(plot_id, None)

    def _scan_open_live_plots(self):
        self._scan_close_live_plots()
        if not self._scanActiveTargets:
            return
        for track in self._scanCsvTracks:
            x, x_label = _scan_track_plot_x(track, self.scanFrames)
            if not len(x):
                continue
            value_start = 0
            for target in self._scanActiveTargets:
                columns = _scan_target_column_names([target])
                target_label = (columns[0].rsplit('.', 1)[0]
                                if len(columns) == 2 else columns[0])
                plot_id = getattr(self, '_scanPlotSerial', 0) + 1
                self._scanPlotSerial = plot_id
                window = _ScanLivePlotWindow(
                    plot_id, track, x, x_label, columns, value_start,
                    target_label, self)
                window.closed.connect(self._scan_plot_closed)
                self._scanLivePlots[plot_id] = window
                window.show()
                value_start += len(columns)

    def _scan_update_live_plots(self, frame_id, values):
        if values is None:
            return
        for window in tuple(getattr(self, '_scanLivePlots', {}).values()):
            window.add_frame(frame_id, values)

    def startScan(self):
        if self.scanRunning:
            if self.scanPaused:
                self.scanPaused = False
                self._scan_status(0., 'Resuming scan')
                if not self.scanWaitingPropagation:
                    qt.QTimer.singleShot(0, self.runScanFrame)
            return

        self._scan_sync_output_template()
        scan = BaseScan(
            self.scanDescription,
            disabled_items=getattr(self, '_scanDisabledItems', ()))
        self.scanFrames = scan.compile_frames()
        self.scanFrameIds = list(self.scanFrames.keys())
        if not self.scanFrameIds:
            return

        self.scanInitialState = self._scan_collect_initial_state(
            self.scanFrames)
        try:
            self._scan_open_csv(scan)
            self._scan_open_live_plots()
        except Exception as exc:
            self._scan_close_live_plots()
            self._scan_close_csv()
            qt.QMessageBox.warning(
                self, 'Start scan', f'Cannot prepare scan output: {exc}')
            return
        self.scanAutoUpdateState = self._scan_auto_update_state()
        self._set_scan_auto_update(False)
        self.scanRunning = True
        self.scanPaused = False
        self.scanStopRequested = False
        self.scanWaitingPropagation = False
        self.scanRestoringInitialState = False
        self.scanFinishWasStopped = False
        self.scanFrameIndex = 0
        self.scanWidget.set_current_frame(0, emit_signal=False)
        self._scan_status(0., 'Starting scan')
        qt.QTimer.singleShot(0, self.runScanFrame)

    def pauseScan(self):
        if not self.scanRunning:
            return
        self.scanPaused = True
        self._scan_status(0., 'Scan paused')

    def stopScan(self):
        if not self.scanRunning:
            return
        self.scanStopRequested = True
        self.scanPaused = False
        self._scan_status(0., 'Stopping scan')
        if not self.scanWaitingPropagation:
            self.finishScan()

    def runScanFrame(self):
        if not self.scanRunning:
            return
        if self.scanPaused or self.scanWaitingPropagation:
            return
        if self.scanStopRequested or self.scanFrameIndex >= len(
                self.scanFrameIds):
            self.finishScan()
            return

        frame_id = self.scanFrameIds[self.scanFrameIndex]
        frame = self.scanFrames[frame_id]
        self.scanWidget.set_current_frame(self.scanFrameIndex)
        self._scan_apply_frame(frame)
        progress = self.scanFrameIndex / max(1, len(self.scanFrameIds))
        self._scan_status(progress, f'Running scan {frame_id}')

        if not frame.get('objects'):
            self.customGlWidget.glDraw()
            qt.QTimer.singleShot(0, self.saveScanFrameAndContinue)
            return

        calc_process = getattr(self.customGlWidget, 'calc_process', None)
        if calc_process is None or not calc_process.is_alive():
            self.customGlWidget.glDraw()
            qt.QTimer.singleShot(0, self.saveScanFrameAndContinue)
            return

        self.scanWaitingPropagation = True
        self.customGlWidget.update_beamline(
            None, {'Acquire': '1'}, sender='scan')

    def onScanPropagationComplete(self, msg):
        if self.scanRestoringInitialState:
            self.scanWaitingPropagation = False
            self.customGlWidget.glDraw()
            qt.QTimer.singleShot(0, self.completeScan)
            return
        if not self.scanRunning or not self.scanWaitingPropagation:
            return
        self.scanWaitingPropagation = False
        self.customGlWidget.glDraw()
        qt.QTimer.singleShot(0, self.saveScanFrameAndContinue)

    def saveScanFrameAndContinue(self):
        if not self.scanRunning:
            return
        if self.scanFrameIndex >= len(self.scanFrameIds):
            self.finishScan()
            return
        if self.scanStopRequested:
            self.finishScan()
            return

        frame_id = self.scanFrameIds[self.scanFrameIndex]
        frame = self.scanFrames[frame_id]
        try:
            target_values = self._scan_write_csv_row(frame_id, frame)
        except Exception as exc:
            qt.QMessageBox.warning(
                self, 'Save scan', f'Cannot write scan CSV: {exc}')
            self.scanStopRequested = True
            self.finishScan()
            return
        try:
            self._scan_update_live_plots(frame_id, target_values)
        except Exception as exc:
            self._scan_close_live_plots()
            qt.QMessageBox.warning(
                self, 'Live scan plots',
                f'Live plotting stopped: {exc}. The scan will continue.')
        filename = frame.get('output', {}).get('glowFrameName')
        if filename:
            filename = self._scan_resolve_output_filename(filename)
            folder = os.path.dirname(filename)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            self.customGlWidget.repaint()
            image = self.customGlWidget.grabFramebuffer()
            image.save(filename)

        self.scanFrameIndex += 1
        progress = self.scanFrameIndex / max(1, len(self.scanFrameIds))
        self._scan_status(progress, f'Saved {frame_id}')
        if self.scanPaused:
            return
        else:
            qt.QTimer.singleShot(0, self.runScanFrame)

    def finishScan(self):
        self.scanFinishWasStopped = self.scanStopRequested
        self.scanWaitingPropagation = False
        try:
            needsPropagation = self._scan_restore_initial_state()
        except Exception:
            self.completeScan()
            return

        calc_process = getattr(self.customGlWidget, 'calc_process', None)
        if needsPropagation and calc_process is not None and \
                calc_process.is_alive():
            self.scanRestoringInitialState = True
            self.scanWaitingPropagation = True
            self._scan_status(1., 'Restoring initial beamline state')
            self.customGlWidget.update_beamline(
                None, {'Acquire': '1'}, sender='scan')
            return
        self.completeScan()

    def completeScan(self):
        self._scan_close_csv()
        was_stopped = self.scanFinishWasStopped
        self.scanRestoringInitialState = False
        self.scanWaitingPropagation = False
        self._set_scan_auto_update(self.scanAutoUpdateState)
        self.scanRunning = False
        self.scanPaused = False
        self.scanStopRequested = False
        self.scanInitialState = None
        self.scanWidget.mark_scan_finished()
        msg = 'Scan stopped' if was_stopped else 'Scan complete'
        self._scan_status(1., msg)

    def _hasScanToRun(self):
        scanDescription = getattr(self, 'scanDescription', None)
        if not isinstance(scanDescription, dict):
            return False
        if scanDescription.get(FRAMES_CLEAN_KEY):
            return False
        if scanDescription.get('items') or scanDescription.get('tracks'):
            return True
        for frameKey in ['frames', 'expandedFrames', 'frameDict']:
            frames = scanDescription.get(frameKey)
            if isinstance(frames, dict) and frames:
                return True
            try:
                if int(frames or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
        try:
            if int(scanDescription.get('frameCount', 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        return any(str(key).startswith('frame_') for key in scanDescription)


class QookScanMixin:
    """Generate a runnable scan from the Qook Glow viewer."""

    def glowScanDescription(self):
        glowWidget = getattr(self, 'blViewer', None)
        description = getattr(glowWidget, 'scanDescription', None)
        return description if isinstance(description, dict) else None

    def availableGlowScanGenerators(self):
        description = self.glowScanDescription()
        if not self._has_glow_scan(description):
            return []
        try:
            tracks, _ = self._scan_tracks_from_description(description)
        except Exception:
            return []
        return ['glow_scan'] if tracks else []

    def _has_glow_scan(self, description):
        if not isinstance(description, dict):
            return False
        if description.get('items') or description.get('tracks'):
            return True
        frames = description.get('frames')
        if isinstance(frames, dict) and frames:
            return True
        return any(str(key).startswith('frame_') for key in description)

    def _scan_literal(self, value):
        if hasattr(value, 'tolist'):
            return repr(value.tolist())
        if isinstance(value, dict):
            items = [
                f'{self._scan_literal(key)}: {self._scan_literal(val)}'
                for key, val in value.items()]
            return '{' + ', '.join(items) + '}'
        if isinstance(value, (list, tuple)):
            return '[' + ', '.join(self._scan_literal(v) for v in value) + ']'
        return repr(value)

    def _scan_identifier(self, *parts):
        text = '_'.join(str(part) for part in parts if str(part))
        text = re.sub(r'\W+', '_', text).strip('_').lower()
        if not text or text[0].isdigit():
            text = 'scan_' + text
        return text

    def _scan_target_expr(self, target):
        target = str(target)
        return f'beamLine.{target}' if target.isidentifier() else \
            f'getattr(beamLine, {target!r})'

    def _scan_frame_index(self, frame_id):
        match = re.match(r'^frame_(\d+)$', str(frame_id))
        return int(match.group(1)) if match is not None else None

    def _scan_values_expr(self, values, duration):
        duration = max(1, int(duration))
        if isinstance(values, dict):
            value_type = values.get('type', 'linspace')
            if value_type == 'linspace':
                return 'xrtrun.get_scan_values({0}, {1}, {2})'.format(
                    self._scan_literal(values.get('start', 0.0)),
                    self._scan_literal(values.get('stop', 0.0)),
                    duration)
            if value_type == 'list':
                return 'xrtrun.get_scan_values({0}, frames={1})'.format(
                    self._scan_literal(list(values.get('values', []))),
                    duration)
            if value_type == 'constant':
                return 'xrtrun.get_scan_values({0}, frames={1})'.format(
                    self._scan_literal(values.get('value')), duration)
        if isinstance(values, (list, tuple)):
            return 'xrtrun.get_scan_values({0}, frames={1})'.format(
                self._scan_literal(list(values)), duration)
        return 'xrtrun.get_scan_values({0}, frames={1})'.format(
            self._scan_literal(values), duration)

    def _scan_track_summary(self, target, prop, start, duration, values):
        end = start + max(1, int(duration)) - 1
        span = f'frame {start}' if start == end else f'frames {start}..{end}'
        if isinstance(values, dict):
            value_type = values.get('type', 'linspace')
            if value_type == 'linspace':
                return f'{target}.{prop}: {span}, ' \
                    f'{values.get("start")} -> {values.get("stop")}'
            if value_type == 'constant':
                return f'{target}.{prop}: {span}, {values.get("value")}'
            if value_type == 'list':
                return f'{target}.{prop}: {span}, list values'
        return f'{target}.{prop}: {span}'

    def _scan_add_track(self, tracks, target, prop, start, duration,
                        values_expr, summary):
        base_name = self._scan_identifier(target, prop)
        used = {track['name'] for track in tracks}
        name = base_name
        index = 2
        while name in used:
            name = f'{base_name}_{index}'
            index += 1
        if start is None:
            start_value = None
            duration_value = 0
        else:
            start_value = int(start)
            duration_value = max(1, int(duration))
        tracks.append({
            'name': name,
            'target': str(target),
            'property': str(prop),
            'start': start_value,
            'duration': duration_value,
            'values_expr': values_expr,
            'summary': summary,
            })

    def _scan_event_tracks(self, item):
        tracks = []
        frame_index = int(item.get('frame', item.get('start', 0)))
        duration = max(1, int(item.get('duration', item.get('steps', 1))))
        for target, patch in item.get('objects', {}).items():
            if str(target) in SCENE_TARGETS or not isinstance(patch, dict):
                continue
            for prop, value in patch.items():
                expr = 'xrtrun.get_scan_values({0}, frames={1})'.format(
                    self._scan_literal(value), duration)
                summary = f'{target}.{prop}: frame {frame_index}, {value}'
                tracks.append((target, prop, frame_index, duration, expr,
                               summary))
        return tracks

    def _scan_expanded_tracks(self, scan):
        schedules = OrderedDict()
        try:
            frames = BaseScan(scan).compile_frames()
        except Exception:
            return []
        for frame_id, frame in frames.items():
            frame_index = self._scan_frame_index(frame_id)
            if frame_index is None:
                continue
            for target, patch in frame.get('objects', {}).items():
                if str(target) in SCENE_TARGETS or not isinstance(patch, dict):
                    continue
                for prop, value in patch.items():
                    key = (str(target), str(prop))
                    schedules.setdefault(key, OrderedDict())[frame_index] = \
                        value
        tracks = []
        for (target, prop), schedule in schedules.items():
            expr = self._scan_literal(schedule)
            summary = f'{target}.{prop}: explicit frame values'
            tracks.append((target, prop, None, None, expr, summary))
        return tracks

    def _scan_tracks_from_description(self, description):
        tracks = []
        skipped = []
        items = description.get('items', description.get('tracks', []))
        for item in items:
            item_type = item.get('type', 'track')
            if item_type == 'track':
                target = item.get('target')
                prop = item.get('property')
                if target is None or prop is None:
                    skipped.append(item.get('id', 'unnamed scan track'))
                    continue
                if str(target) in SCENE_TARGETS:
                    skipped.append(item.get('id', f'{target}.{prop}'))
                    continue
                start = int(item.get('start', 0))
                duration = int(item.get('duration', item.get('steps', 1)))
                values = item.get('values')
                self._scan_add_track(
                    tracks, target, prop, start, duration,
                    self._scan_values_expr(values, duration),
                    self._scan_track_summary(
                        target, prop, start, duration, values))
            elif item_type == 'event':
                for track in self._scan_event_tracks(item):
                    self._scan_add_track(tracks, *track)
            else:
                fallback = {'items': [item]}
                for track in self._scan_expanded_tracks(fallback):
                    self._scan_add_track(tracks, *track)

        if not items:
            for track in self._scan_expanded_tracks(description):
                self._scan_add_track(tracks, *track)
        return tracks, skipped

    def _scan_frame_count(self, description, tracks):
        try:
            frame_count = len(BaseScan(description).compile_frames())
            if frame_count:
                return frame_count
        except Exception:
            pass
        return max([track['start'] + track['duration']
                    for track in tracks
                    if track['start'] is not None] or [1])

    def _scan_assignment_lines(self, target, prop, value_expr, indent):
        target_expr = self._scan_target_expr(target)
        if '.' not in prop:
            return [f'{indent}{target_expr}.{prop} = {value_expr}']

        root, field = prop.split('.', 1)
        component_indices = {
            'center': {'x': 0, 'y': 1, 'z': 2},
            'x': {'x': 0, 'y': 1, 'z': 2},
            'z': {'x': 0, 'y': 1, 'z': 2},
            'limPhysX': {'lmin': 0, 'lmax': 1},
            'limPhysY': {'lmin': 0, 'lmax': 1},
            'limPhysX2': {'lmin': 0, 'lmax': 1},
            'limPhysY2': {'lmin': 0, 'lmax': 1},
            'opening': {
                'left': 0, 'right': 1, 'bottom': 2, 'top': 3},
            }
        if root == 'blades':
            lines = [
                f'{indent}{root} = dict({target_expr}.{root})',
                f'{indent}{root}[{field!r}] = {value_expr}',
                f'{indent}{target_expr}.{root} = {root}',
                ]
            return lines
        index = component_indices.get(root, {}).get(field)
        if index is None:
            return [
                f'{indent}# Unsupported compound scan field: '
                f'{target}.{prop}']
        lines = [
            f'{indent}{root} = list({target_expr}.{root})',
            f'{indent}{root}[{index}] = {value_expr}',
            f'{indent}{target_expr}.{root} = {root}',
            ]
        return lines

    def makeGlowScanCode(self):
        description = self.glowScanDescription()
        if not self._has_glow_scan(description):
            return ''

        tracks, skipped = self._scan_tracks_from_description(description)
        frame_count = self._scan_frame_count(description, tracks)
        track_word = 'track' if len(tracks) == 1 else 'tracks'

        lines = [
            '\ndef glow_scan(plots, beamLine):',
            f'{SCAN_CODE_INDENT}"""Generated from the current xrtGlow scan."""',
            f'{SCAN_CODE_INDENT}# xrtGlow scan: {len(tracks)} {track_word}, '
            f'{frame_count} frames',
            ]
        for track in tracks:
            lines.append(f'{SCAN_CODE_INDENT}# {track["summary"]}')
        for item_id in skipped:
            lines.append(f'{SCAN_CODE_INDENT}# Skipped scene-only scan track: {item_id}')
        lines.append('')

        for track in tracks:
            if track['start'] is None:
                lines.append(f'{SCAN_CODE_INDENT}{track["name"]} = '
                             f'{track["values_expr"]}')
            else:
                lines.append(f'{SCAN_CODE_INDENT}{track["name"]} = dict(zip(')
                lines.append(f'{SCAN_CODE_INDENT*2}range({track["start"]}, '
                             f'{track["start"] + track["duration"]}),')
                lines.append(f'{SCAN_CODE_INDENT*2}{track["values_expr"]}))')
        if not tracks:
            lines.append(f'{SCAN_CODE_INDENT}# No beamline-property tracks to apply.')
        lines.extend([
            '',
            f'{SCAN_CODE_INDENT}for iFrame in range({frame_count}):',
            ])
        for track in tracks:
            lines.append(f'{SCAN_CODE_INDENT*2}if iFrame in {track["name"]}:')
            value_expr = f'{track["name"]}[iFrame]'
            lines.extend(self._scan_assignment_lines(
                track['target'], track['property'], value_expr, SCAN_CODE_INDENT*3))
            lines.append('')
        lines.extend([
            f'{SCAN_CODE_INDENT*2}frame_file_name = "frame{{0:04d}}.jpg".format('
            f'iFrame)',
            f'{SCAN_CODE_INDENT*2}for plot in plots:',
            f'{SCAN_CODE_INDENT*3}plot.textPanel.set_text({value_expr})',
            f'{SCAN_CODE_INDENT*3}plot.saveName = plot.title + "_" + frame_file_name',
            f'{SCAN_CODE_INDENT*2}yield',
            '\n',
            ])
        return '\n'.join(lines)

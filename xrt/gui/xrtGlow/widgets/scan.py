# -*- coding: utf-8 -*-
"""
Timeline scan helpers for xrtGlow.

The module intentionally separates the scan description compiler from the Qt
widgets. The compiler turns compact timeline recipes into explicit frame
patches; the widgets provide a first UI surface for inspecting those recipes.
"""

import copy
import json
import re
import string
from collections import OrderedDict

from ...commons import qt

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "7 May 2026"


SCENE_TARGETS = {'Scene', 'scene', 'xrtGlow', 'xrtglow'}
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
        self.description = copy.deepcopy(description or {})
        self.version = self.description.get('version', 1)
        self.kind = self.description.get('kind', 'timeline_recipe')
        self.frame_count = int(self.description.get(
            'frames', self.description.get('frameCount', 0)))
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
                'targetName': target_name or target,
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

    def _ensure_frame_count(self):
        if self.frame_count:
            return
        frame_count = 0
        for item in self.items:
            item_type = item.get('type', 'track')
            if item_type == 'event':
                frame_count = max(frame_count, int(item.get('frame', 0)) + 1)
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
        item_id = item.get('id', f"{item.get('target')}.{item.get('property')}")
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
        self._merge_frame(frames, frame_index, patch,
                          item.get('id', 'event'))

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
        frames = self._make_empty_frames()
        for item in self.items:
            item_type = item.get('type', 'track')
            if item_type == 'event':
                self._compile_event(frames, item)
            elif item_type == 'loopBlock':
                self._compile_loop_block(frames, item)
            else:
                self._compile_track(frames, item)
        return frames


class ScanRangeDialog(qt.QDialog):
    """Small dialog for creating a one-property linear scan."""

    scanCreated = qt.Signal(dict)

    def __init__(self, target, property_name, current_value=None,
                 target_name=None, parent=None):
        super().__init__(parent)
        self.target = target
        self.target_name = target_name or target
        self.property_name = property_name
        self.setWindowTitle(
            f'Create scan: {self.target_name}.{property_name}')

        self.startFrameEdit = qt.QLineEdit('0')
        min_value, max_value = _scan_default_bounds(current_value)
        self.minValueEdit = qt.QLineEdit(str(min_value))
        self.maxValueEdit = qt.QLineEdit(str(max_value))
        self.pointsEdit = qt.QLineEdit('10')

        layout = qt.QVBoxLayout(self)
        form = qt.QFormLayout()
        form.addRow('Target', qt.QLabel(str(self.target_name)))
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
            'id': f'{self.target_name}.{self.property_name}',
            'start': int(self.startFrameEdit.text()),
            'duration': points,
            'target': self.target,
            'targetName': self.target_name,
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

    def __init__(self, catalog, start_frame=0, parent=None):
        super().__init__(parent)
        self.catalog = list(catalog or [])
        self.propertyMap = {}
        self.selectedProperty = None
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

    def _populate_tree(self):
        self.propertyTree.clear()
        self.propertyMap.clear()
        for target in self.catalog:
            target_item = qt.QTreeWidgetItem([
                str(target.get('name', target.get('target', ''))), ''])
            target_item.setFirstColumnSpanned(True)
            self.propertyTree.addTopLevelItem(target_item)
            for prop in target.get('properties', []):
                key = f"{target.get('target')}::{prop.get('name')}"
                value = prop.get('value', '')
                child = qt.QTreeWidgetItem([str(prop.get('name')), str(value)])
                child.setData(0, qt.Qt.UserRole, key)
                target_item.addChild(child)
                self.propertyMap[key] = {
                    'target': target.get('target'),
                    'targetName': target.get('name', target.get('target')),
                    'property': prop.get('name'),
                    'value': value,
                    }
            target_item.setExpanded(False)
        self.propertyTree.resizeColumnToContents(0)

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
        return self.propertyMap.get(key)

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
        item_id = f"{prop['targetName']}.{prop['property']}"

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
            ('targetName', prop['targetName']),
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


class TimelineGraphicsView(qt.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sceneObj = qt.QGraphicsScene(self)
        self.setScene(self.sceneObj)
        self.setMinimumHeight(160)
        self.frame_count = 0
        self.items = []
        self.current_frame = 0

    def set_timeline(self, items, frame_count):
        self.items = list(items)
        self.frame_count = int(frame_count)
        self.rebuild()

    def set_current_frame(self, frame_index):
        self.current_frame = max(0, int(frame_index))
        self.rebuild()

    def rebuild(self):
        self.sceneObj.clear()
        left = 80
        row_h = 24
        width = max(1, self.frame_count) * 4
        self.sceneObj.addLine(left, 8, left + width, 8)
        for index, item in enumerate(self.items):
            y = 28 + index * row_h
            label = item.get('id', item.get('type', 'track'))
            self.sceneObj.addText(str(label)).setPos(0, y - 8)
            item_type = item.get('type', 'track')
            if item_type == 'event':
                start = int(item.get('frame', item.get('start', 0)))
                self.sceneObj.addRect(left + start * 4, y, 5, 14)
            else:
                start = int(item.get('start', 0))
                if item_type == 'loopBlock':
                    duration = BaseScan({'items': [item]})._loop_block_length(
                        item)
                else:
                    duration = int(item.get('duration',
                                            item.get('steps', 1)))
                rect = self.sceneObj.addRect(left + start * 4, y,
                                             max(4, duration * 4), 14)
                rect.setBrush(qt.QBrush(qt.QColor('#6fa8dc')))
        playhead_x = left + self.current_frame * 4
        pen = qt.QPen(qt.QColor('#d64545'))
        pen.setWidth(2)
        playhead = self.sceneObj.addLine(playhead_x, 4, playhead_x,
                                         34 + max(1, len(self.items)) * row_h,
                                         pen)
        playhead.setZValue(10)
        self.setSceneRect(self.sceneObj.itemsBoundingRect())


class TimelineFrameListWidget(qt.QWidget):
    """Preview widget for scan timeline items and expanded frames."""

    scanStarted = qt.Signal()
    scanPaused = qt.Signal()
    scanStopped = qt.Signal()
    currentFrameChanged = qt.Signal(int, dict)
    outputTemplateChanged = qt.Signal(str)
    instructionRequested = qt.Signal(int)
    trackDeleteRequested = qt.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scan = BaseScan()
        self.frames = OrderedDict()
        self.frameIds = []
        self.currentFrame = 0
        self._playing = False
        self._updatingSelection = False

        layout = qt.QVBoxLayout(self)
        controls = qt.QHBoxLayout()
        self.addInstructionButton = qt.QPushButton('Add instruction')
        self.deleteTrackButton = qt.QPushButton('Delete track')
        self.startButton = qt.QPushButton('Start')
        self.pauseButton = qt.QPushButton('Pause')
        self.stopButton = qt.QPushButton('Stop')
        self.currentFrameSpin = qt.QSpinBox()
        self.currentFrameSpin.setMinimum(0)
        self.currentFrameSpin.setMaximum(0)
        self.currentFrameLabel = qt.QLabel('Frame 0 / 0')
        self.outputTemplateEdit = qt.QLineEdit(
            DEFAULT_OUTPUT['glowFrameName'])
        controls.addWidget(self.addInstructionButton)
        controls.addWidget(self.deleteTrackButton)
        controls.addSpacing(12)
        controls.addWidget(self.startButton)
        controls.addWidget(self.pauseButton)
        controls.addWidget(self.stopButton)
        controls.addSpacing(12)
        controls.addWidget(qt.QLabel('Current frame'))
        controls.addWidget(self.currentFrameSpin)
        controls.addWidget(self.currentFrameLabel)
        controls.addSpacing(12)
        controls.addWidget(qt.QLabel('Filename template'))
        controls.addWidget(self.outputTemplateEdit)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = qt.QSplitter(qt.Qt.Vertical)
        layout.addWidget(splitter)

        self.trackTable = qt.QTableWidget(0, 6)
        self.trackTable.setHorizontalHeaderLabels(
            ['Id', 'Type', 'Start', 'End', 'Target', 'Property'])
        self.trackTable.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.trackTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.trackTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.trackTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.timelineView = TimelineGraphicsView()
        self.frameTable = qt.QTableWidget(0, 4)
        self.frameTable.setHorizontalHeaderLabels(
            ['Frame', 'Objects', 'Scene', 'Output'])
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
        self.currentFrameSpin.valueChanged.connect(self.set_current_frame)
        self.addInstructionButton.clicked.connect(
            self._request_instruction_at_current_frame)
        self.deleteTrackButton.clicked.connect(self._delete_selected_track)
        self.startButton.clicked.connect(self.start_scan)
        self.pauseButton.clicked.connect(self.pause_scan)
        self.stopButton.clicked.connect(self.stop_scan)
        self.outputTemplateEdit.editingFinished.connect(
            self._output_template_edited)

        top = qt.QSplitter(qt.Qt.Horizontal)
        top.addWidget(self.trackTable)
        top.addWidget(self.timelineView)
        splitter.addWidget(top)

        bottom = qt.QTabWidget()
        bottom.addTab(self.frameTable, 'Frames')
        bottom.addTab(self.warningList, 'Warnings')
        splitter.addWidget(bottom)
        self.deleteTrackShortcut = qt.QShortcut(self.trackTable)
        self.deleteTrackShortcut.setKey(qt.QKeySequence.Delete)
        self.deleteTrackShortcut.activated.connect(self._delete_selected_track)
        self._update_play_buttons()

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

    def _target_names(self):
        names = {}
        for item in self.scan.items:
            target = item.get('target')
            target_name = item.get('targetName')
            if target and target_name:
                names[target] = target_name
        return names

    def rebuild(self):
        frames = self.scan.compile_frames()
        self._populate_tracks()
        self.timelineView.set_timeline(self.scan.items, self.scan.frame_count)
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
            self.timelineView.set_current_frame(0)
            self.currentFrameSpin.blockSignals(True)
            self.currentFrameSpin.setMaximum(0)
            self.currentFrameSpin.setValue(0)
            self.currentFrameSpin.blockSignals(False)
            self.currentFrameLabel.setText('Frame 0 / 0')
            return
        frame_index = max(0, min(int(frame_index), len(self.frameIds) - 1))
        self.currentFrame = frame_index
        self.timelineView.set_current_frame(frame_index)
        self.currentFrameSpin.blockSignals(True)
        self.currentFrameSpin.setMaximum(len(self.frameIds) - 1)
        self.currentFrameSpin.setValue(frame_index)
        self.currentFrameSpin.blockSignals(False)
        self.currentFrameLabel.setText(
            f'Frame {frame_index + 1} / {len(self.frameIds)}')
        self._updatingSelection = True
        self.frameTable.setCurrentCell(frame_index, 0)
        self.frameTable.selectRow(frame_index)
        self._updatingSelection = False
        if emit_signal:
            frame_id = self.frameIds[frame_index]
            self.currentFrameChanged.emit(frame_index, self.frames[frame_id])

    def _populate_tracks(self):
        self.trackTable.setRowCount(len(self.scan.items))
        for row, item in enumerate(self.scan.items):
            start = int(item.get('frame', item.get('start', 0)))
            if item.get('type') == 'event':
                end = start
            elif item.get('type') == 'loopBlock':
                end = start + self.scan._loop_block_length(item) - 1
            else:
                end = start + int(item.get('duration',
                                           item.get('steps', 1))) - 1
            values = [item.get('id', ''), item.get('type', 'track'),
                      start, end,
                      item.get('targetName', item.get('target', '')),
                      item.get('property', '')]
            for col, value in enumerate(values):
                self.trackTable.setItem(row, col,
                                        qt.QTableWidgetItem(str(value)))

    def _populate_frames(self, frames):
        self.frames = frames
        self.frameIds = list(frames.keys())
        target_names = self._target_names()
        self.frameTable.setRowCount(len(frames))
        for row, (frame_id, frame) in enumerate(frames.items()):
            objects = OrderedDict()
            for target, patch in frame.get('objects', {}).items():
                objects[target_names.get(target, target)] = patch
            values = [frame_id,
                      json.dumps(objects),
                      json.dumps(frame.get('scene', {})),
                      json.dumps(frame.get('output', {}))]
            for col, value in enumerate(values):
                self.frameTable.setItem(row, col,
                                        qt.QTableWidgetItem(value))

    def _populate_warnings(self):
        target_names = self._target_names()
        self.warningList.clear()
        for warning in self.scan.warnings:
            path = str(warning.get('path'))
            for target, target_name in target_names.items():
                path = path.replace(f'objects.{target}.',
                                    f'objects.{target_name}.')
            self.warningList.addItem(
                f"{warning.get('frame')}: {path} "
                f"overwritten by {warning.get('item')}")

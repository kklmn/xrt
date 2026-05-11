# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:03:13 2026

"""

import os
import re
import copy
import json
import numpy as np
from functools import partial
import matplotlib as mpl
from collections import OrderedDict, ChainMap
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from .._constants import (
        _DEBUG_, DEFAULT_SCENE_SETTINGS, COLOR_CONTROL_LABELS,
        SCENE_CONTROL_LABELS, SCENE_TEXTEDITS, itemTypes)
from .._utils import is_source, is_oe, is_aperture, is_screen
from .inspector import InstanceInspector
from .scan import (
    BaseScan, DEFAULT_OUTPUT, FRAME_SECTIONS, SCENE_PROPERTY_NAMES,
    SCENE_TARGETS, ScanInstructionDialog, TimelineFrameListWidget,
    default_scan_description)
from .opengl import xrtGlWidget

from ...commons import qt
from .nodeeditor import _FlowGraphPanel, FLOW_NODE_STYLES, FLOW_SCENE_STYLE

from ....backends import raycing
from ....backends.raycing import sources as rsources
from ....plotter import colorFactor, colorSaturation

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

# Controls placement mode for xrtGlow:
# "fixed" keeps the original right-side tab widget,
# while the collapsible modes expose the same panels from a toolbar.
CONTROL_TAB = "collapsible right"
CONTROL_TAB_MODES = (
    "fixed",
    "collapsible top",
    "collapsible left",
    "collapsible right",
)
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
SCAN_LIMIT_PROPERTIES = ('limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2')
SCAN_AXIS_PROPERTIES = ('x', 'z')


class _ToolbarPopupPanel(qt.QFrame):
    popupHidden = qt.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, qt.Qt.Popup | qt.Qt.FramelessWindowHint)
        self.setFrameShape(qt.QFrame.StyledPanel)
        self.setObjectName('xrtGlowToolbarPopup')
        self.setStyleSheet("""
            QFrame#xrtGlowToolbarPopup {
                background: palette(window);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
        """)
        self.titleLabel = qt.QLabel()
        self.titleLabel.setStyleSheet("font-weight: 600;")
        self.stack = qt.QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(qt.QSizePolicy.Ignored,
                                 qt.QSizePolicy.Ignored)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.stack)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.popupHidden.emit()


class xrtGlow(qt.QWidget):
    def __init__(self, arrayOfRays=None, parent=None, progressSignal=None,
                 layout=None, epicsPrefix=None, epicsMap={},
                 sceneSettings={}, scanDescription=None):
        super(xrtGlow, self).__init__()
        self.parentRef = parent
        self.cAxisLabelSize = 10
        mplFont = {'size': self.cAxisLabelSize}
        mpl.rc('font', **mplFont)
        self.setWindowTitle('xrtGlow')
        iconsDir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '_icons')
        self.setWindowIcon(qt.QIcon(os.path.join(iconsDir, 'icon-GLow.ico')))
        iconsQookDir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), '../xrtQook', '_icons')
        self.iconLib = {}
        for oeType in ['source', 'oe', 'aperture', 'screen']:
            self.iconLib[oeType] = qt.QIcon(os.path.join(
                    iconsQookDir, f'add-{oeType}.png'))

#        if arrayOfRays is not None:
#            self.populateOEsList(arrayOfRays)
#        print("arrayOfRays", arrayOfRays)
        self.segmentsModel = self.initSegmentsModel()
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()

        self.fluxDataModel = qt.QStandardItemModel()

        for colorField in raycing.allBeamFields:
            self.fluxDataModel.appendRow(qt.QStandardItem(colorField))

        glwInitKwargs = {'parent': self, 'modelRoot': self.segmentsModelRoot,
                         'epicsPrefix': epicsPrefix, 'epicsMap': epicsMap,
                         'signal': progressSignal}

        if arrayOfRays is not None:
            glwInitKwargs.update(
                    {'arrayOfRays': arrayOfRays,
                     })
        elif layout is not None:
            glwInitKwargs.update({'beamLayout': layout})

        self.customGlWidget = xrtGlWidget(**glwInitKwargs)

        self.populateSegmentsModel(arrayOfRays)

        self.customGlWidget.rotationUpdated.connect(self.updateRotationFromGL)
        self.customGlWidget.scaleUpdated.connect(self.updateScaleFromGL)
        self.customGlWidget.histogramUpdated.connect(self.updateColorMap)
        self.customGlWidget.propagationComplete.connect(
            self.onScanPropagationComplete)
        self.customGlWidget.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customGlWidget.customContextMenuRequested.connect(self.glMenu)
        self.customGlWidget.openElViewer.connect(self.runElementViewer)
        self.scanDescription = self._scan_description_from_input(
            scanDescription)
        self.scanOutputDirectory = None
        self.scanRunning = False
        self.scanPaused = False
        self.scanStopRequested = False
        self.scanWaitingPropagation = False
        self.scanFrames = OrderedDict()
        self.scanFrameIds = []
        self.scanFrameIndex = 0
        self.scanInitialState = None
        self.scanAutoUpdateState = True
        self.scanRestoringInitialState = False
        self.scanFinishWasStopped = False

        self.makeNavigationPanel()
        self.makeTransformationPanel()
        self.makeColorsPanel()
        self.makeGridAndProjectionsPanel()
        self.makeScenePanel()
        self.makeScanPanel()
        self.initControlPanels()

        self.controlTabMode = self._resolveControlTabMode(CONTROL_TAB)

        if self.controlTabMode == 'fixed':
            mainLayout = qt.QHBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            self.canvasSplitter = qt.QSplitter()
            self.canvasSplitter.setChildrenCollapsible(False)
            self.canvasSplitter.setOrientation(qt.Qt.Horizontal)
            mainLayout.addWidget(self.canvasSplitter)
            self.canvasSplitter.addWidget(self.customGlWidget)
            self.controlsTabs = self.makeControlsTabsWidget()
            self.canvasSplitter.addWidget(self.controlsTabs)
        else:
            self.makeControlsToolbar(self.controlTabMode)
            if self.controlTabMode == 'collapsible top':
                mainLayout = qt.QVBoxLayout()
            else:
                mainLayout = qt.QHBoxLayout()
            mainLayout.setContentsMargins(0, 0, 0, 0)
            if self.controlTabMode in ['collapsible top', 'collapsible left']:
                mainLayout.addWidget(self.controlsToolBar)
            self.canvasSplitter = qt.QSplitter()
            self.canvasSplitter.setChildrenCollapsible(False)
            self.canvasSplitter.setOrientation(qt.Qt.Horizontal)
            mainLayout.addWidget(self.canvasSplitter)
            self.canvasSplitter.addWidget(self.customGlWidget)
            self.sidePlaceholder = qt.QWidget()
            self.sidePlaceholder.setMinimumWidth(0)
            self.sidePlaceholder.setMaximumWidth(0)
            self.canvasSplitter.addWidget(self.sidePlaceholder)
            self.canvasSplitter.setSizes([1, 0])
            if self.controlTabMode == 'collapsible right':
                mainLayout.addWidget(self.controlsToolBar)

        self.setLayout(mainLayout)
#        tabs.tabBar().setStyleSheet("""
#            QTabBar::tab {
#                padding: 6px 12px;
#                background: qlineargradient(
#                    x1:0, y1:0, x2:0, y2:1,
#                    stop:0 rgba(255,255,255,12%),
#                    stop:0.45 transparent,
#                    stop:1 rgba(0,0,0,12%)
#                );
#                border-left: 1px solid rgba(255,255,255,20%);
#                border-right: 1px solid rgba(0,0,0,15%);
#            }
#            QTabBar::tab:hover {
#                background: qlineargradient(
#                    x1:0, y1:0, x2:0, y2:1,
#                    stop:0 rgba(255,255,255,20%),
#                    stop:0.45 transparent,
#                    stop:1 rgba(0,0,0,20%)
#                );
#                }
#
#            QTabBar::tab:selected {
#                border-bottom: 2px solid rgba(0,0,0,40%);
#            }
#        """)

        toggleHelp = qt.QShortcut(self)
        toggleHelp.setKey("F1")
        toggleHelp.activated.connect(self.openHelpDialog)
        fastSave = qt.QShortcut(self)
        fastSave.setKey("F5")
        fastSave.activated.connect(partial(self.saveScene, '_xrtScnTmp_.npy'))
        fastLoad = qt.QShortcut(self)
        fastLoad.setKey("F6")
        fastLoad.activated.connect(partial(self.loadScene, '_xrtScnTmp_.npy'))
        startMovie = qt.QShortcut(self)
        startMovie.setKey("F7")
        startMovie.activated.connect(self.startRecordingMovie)
        toggleScreen = qt.QShortcut(self)
        toggleScreen.setKey("F3")
        toggleScreen.activated.connect(self.customGlWidget.toggleVScreen)

        toggleLoop = qt.QShortcut(self)
        toggleLoop.setKey("F8")
        toggleLoop.activated.connect(self.customGlWidget.toggleLoop)

        tiltScreen = qt.QShortcut(self)
        tiltScreen.setKey("Ctrl+T")
        tiltScreen.activated.connect(self.customGlWidget.switchVScreenTilt)

        sceneProps = {}
        for key, val in DEFAULT_SCENE_SETTINGS.items():
            sceneProps[key] = sceneSettings.get(key, val)
            if key == 'rayFlag':
                sceneProps[key] = set(sceneProps[key])

        self.applySceneProperties(sceneProps)

    def _resolveControlTabMode(self, control_mode):
        control_mode = str(control_mode).strip().lower()
        if control_mode not in CONTROL_TAB_MODES:
            return "fixed"
        return control_mode

    def _makeCheckBox(self, cbLabel, cbControl, cbRegistry=None):
        cb = qt.QCheckBox(cbLabel)
        cb.setChecked(DEFAULT_SCENE_SETTINGS.get(cbControl))
        cb.toggled.connect(partial(self._setGlFlag, cbControl))
        if cbRegistry is not None:
            cbRegistry[cbControl] = cb
        return cb

    def _makeLabeledLineEdit(self, label, tooptip, teControl, ctrlRegistry):
        tLabel = qt.QLabel(label)
        tLabel.setToolTip(tooptip)
        tEdit = qt.QLineEdit()
        tEdit.setToolTip(tooptip)

        defv = DEFAULT_SCENE_SETTINGS.get(teControl)
        if defv is not None:
            tEdit.setText(str(defv))

        tEdit.editingFinished.connect(
            partial(self._setGlFlag, teControl, tEdit))

        layout = qt.QHBoxLayout()
        tLabel.setMinimumWidth(145)
        layout.addWidget(tLabel)
        tEdit.setMaximumWidth(96)
        layout.addWidget(tEdit)
        layout.addStretch()

        if ctrlRegistry is not None:
            ctrlRegistry[teControl] = tEdit

        return layout

    def _setGlFlag(self, pName, state):
        if not isinstance(state, bool):  # editor
            state = float(re.sub(',', '.', str(state.text())))
        setattr(self.customGlWidget, pName, state)
        self.customGlWidget.glDraw()

    def _setRayFlag(self, rayFlag, state):
        flags = set(getattr(self.customGlWidget, 'rayFlag', set()))
        if state:
            flags.add(rayFlag)
        else:
            flags.discard(rayFlag)
        self.customGlWidget.rayFlag = flags
        if rayFlag == 4:
            self.customGlWidget.showLostRays = state
        self.customGlWidget.glDraw()

    def _makeRayVisibilityPanel(self):
        rayPanel = qt.QGroupBox('Ray Visibility', self)
        rayLayout = qt.QHBoxLayout()
        rayLayout.setContentsMargins(6, 8, 6, 6)
        rayLayout.setSpacing(8)
        self.rayFlagControls = OrderedDict()
        rayFlagProps = OrderedDict([
            (1, ('Good', 'Rays hit the surface within optical limits')),
            (2, ('Out', 'Rays hit the surface within physical limits, '
                 'but outside optical limits')),
            (3, ('Over', 'Rays went over the surface without interaction')),
            (4, ('Lost', 'Rays absorbed at the optical element')),
            ])
        rayFlags = set(getattr(self.customGlWidget, 'rayFlag', set()))
        for rayFlag, (label, tooltip) in rayFlagProps.items():
            checkBox = qt.QCheckBox(label)
            checkBox.setToolTip(tooltip)
            checkBox.setChecked(rayFlag in rayFlags)
            checkBox.toggled.connect(partial(self._setRayFlag, rayFlag))
            self.rayFlagControls[rayFlag] = checkBox
            rayLayout.addWidget(checkBox)
        rayLayout.addStretch()
        rayPanel.setLayout(rayLayout)
        return rayPanel

    def closeEvent(self, event):
        self.customGlWidget.cleanup_gl_resources()
        self.customGlWidget.close_calc_process()
        event.accept()

    def runElementViewer(self, oeuuid=None):
        if oeuuid is not None and hasattr(self.customGlWidget, 'beamline'):
            oe = self.customGlWidget.beamline.oesDict.get(oeuuid, None)
            if oe is None:
                return
            oeObj = oe[0]
            oeType = oe[-1]
            blName = self.customGlWidget.beamline.name
            oeProps = raycing.get_init_kwargs(oeObj, compact=False,
                                              blname=blName)
            oeProps.update({'uuid': oeuuid})
            oeInitProps = {}
            for argName, argValue in oeProps.items():
                if hasattr(oeObj, f'_{argName}Init'):
                    oeInitProps[argName] = getattr(oeObj, f'_{argName}Init')
                if any(argName.lower().startswith(v) for v in
                        ['mater', 'tlay', 'blay', 'coat', 'substrate']) and\
                        raycing.is_valid_uuid(argValue):
                    argMat = self.customGlWidget.beamline.materialsDict.get(
                            argValue)
                    if argMat is not None:
                        oeProps[argName] = argMat.name
                if any(argName.lower().startswith(v) for v in
                        ['figureer', 'basefe']) and\
                        raycing.is_valid_uuid(argValue):
                    argFE = self.customGlWidget.beamline.fesDict.get(
                            argValue)
                    if argFE is not None:
                        oeProps[argName] = argFE.name

            catDict = {'Position': raycing.orientationArgSet}
            if oeType == 0:  # source
                if hasattr(oeObj, 'eE'):
                    catDict.update({
                        'Electron Beam': rsources.electronBeamArgSet,
                        'Magnetic Structure': rsources.magneticStructureArgSet
                        })

                catDict.update({
                        'Distributions': rsources.distributionsArgSet,
                        'Source Limits': rsources.sourceLimitsArgSet})
            else:
                catDict.update({'Shape': raycing.shapeArgSet})

            if any([hasattr(oeObj, arg) for arg in raycing.diagnosticArgs]):
                catDict.update({
                    'Diagnostic': raycing.diagnosticArgs})
                diagProps = {argName: getattr(oeObj, argName) for
                             argName in raycing.diagnosticArgs if
                             hasattr(oeObj, argName)}
                oeProps.update(diagProps)

            viewOnly = self.customGlWidget.renderingMode == 'static'

            elViewer = InstanceInspector(
                    self, oeProps,
                    initDict=oeInitProps,
                    epicsDict=getattr(self.customGlWidget,
                                      'epicsInterface', None),
                    viewOnly=viewOnly,
                    beamLine=self.customGlWidget.beamline,
                    categoriesDict=catDict)

            self.customGlWidget.beamUpdated.connect(elViewer.update_beam)
            self.customGlWidget.oePropsUpdated.connect(elViewer.update_param)
            # TODO: update tree
            elViewer.propertiesChanged.connect(
                    partial(self.customGlWidget.update_beamline, oeuuid,
                            sender='OEE'))
            elViewer.scanCreated.connect(self.addScanItem)
            if (elViewer.show()):
                pass

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
            raise TypeError(
                'scanDescription must be a dict, list, JSON string, '
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
        output = description.setdefault('output', {})
        output.setdefault('glowFrameName', DEFAULT_OUTPUT['glowFrameName'])
        return self._scan_portable_description(description)

    def setScanDescription(self, scanDescription):
        self.scanDescription = self._scan_description_from_input(
            scanDescription)
        self.refreshScanPanel()

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

    def makeScanPanel(self):
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.scanWidget = TimelineFrameListWidget(self)
        self.scanWidget.scanStarted.connect(self.startScan)
        self.scanWidget.scanPaused.connect(self.pauseScan)
        self.scanWidget.scanStopped.connect(self.stopScan)
        self.scanWidget.outputTemplateChanged.connect(
            self.setScanOutputTemplate)
        self.scanWidget.instructionRequested.connect(
            self.openScanInstructionDialog)
        self.scanWidget.trackDeleteRequested.connect(self.deleteScanItem)
        self.scanWidget.scanLoadRequested.connect(self.loadScanFromJson)
        self.scanWidget.scanSaveRequested.connect(self.saveScanToJson)
        self.scanWidget.trackTimingChanged.connect(self.updateScanItemTiming)
        self.scanWidget.trackEditRequested.connect(self.editScanItem)
        layout.addWidget(self.scanWidget)
        self.scanPanel = qt.QWidget(self)
        self.scanPanel.setLayout(layout)
        self.refreshScanPanel()

    def refreshScanPanel(self):
        if hasattr(self, 'scanWidget'):
            self.scanWidget.set_scan(BaseScan(self.scanDescription))

    def openScanPanel(self):
        if hasattr(self, 'sideTabs') and hasattr(self, 'scanPanel'):
            self.sideTabs.setCurrentWidget(self.scanPanel)
            return
        if hasattr(self, 'controlButtons'):
            button = self.controlButtons.get("Scans")
            if button is not None:
                button.setChecked(True)
                self.toggleControlPanel("Scans", button, True)

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
        template = str(template).strip() or 'frame{index:04d}.jpg'
        self.scanDescription.setdefault('output', {})[
            'glowFrameName'] = template
        self.refreshScanPanel()

    def deleteScanItem(self, item_index):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        del items[item_index]
        self.scanDescription['frames'] = self._scan_recipe_frame_count(items)
        self.refreshScanPanel()

    def replaceScanItem(self, item_index, item):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        items[item_index] = self._scan_portable_item(item)
        self.scanDescription['frames'] = self._scan_recipe_frame_count(items)
        self.refreshScanPanel()

    def editScanItem(self, item_index):
        items = self.scanDescription.get('items', [])
        if item_index < 0 or item_index >= len(items):
            return
        dialog = ScanInstructionDialog(
            self.scanInstructionCatalog(), edit_item=items[item_index],
            parent=self)
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

    def _scan_split_compound_property(self, name, value):
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
                    name, value)
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
            parent=self)
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

    def startScan(self):
        if self.scanRunning:
            if self.scanPaused:
                self.scanPaused = False
                self._scan_status(0., 'Resuming scan')
                if not self.scanWaitingPropagation:
                    qt.QTimer.singleShot(0, self.runScanFrame)
            return

        scan = BaseScan(self.scanDescription)
        self.scanFrames = scan.compile_frames()
        self.scanFrameIds = list(self.scanFrames.keys())
        if not self.scanFrameIds:
            return

        self.scanInitialState = self._scan_collect_initial_state(
            self.scanFrames)
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
            raise
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

    def makeNavigationPanel(self):
        self.navigationLayout = qt.QVBoxLayout()
#        centerCBLabel = qt.QLabel('Center view at:')
#        self.centerCB = qt.QComboBox()
#        self.centerCB.setMaxVisibleItems(48)
#        self.centerCB.setSizeAdjustPolicy(qt.QComboBox.AdjustToContents)
#        self.centerProxyModel = qt.ComboBoxFilterProxyModel(self.centerCB)
#        self.centerProxyModel.setSourceModel(self.segmentsModel)
#        self.centerCB.setModel(self.centerProxyModel)
#        self.centerCB.setModelColumn(0)
#        self.centerCB.currentIndexChanged['int'].connect(
#                lambda elementid: self.centerEl(
#                        self.centerCB.itemData(elementid,
#                                               role=qt.Qt.UserRole)))
#        self.centerCB.setCurrentIndex(0)

        layout = qt.QHBoxLayout()
#        layout.addWidget(centerCBLabel)
#        layout.addWidget(self.centerCB)
        layout.addStretch()
        self.navigationLayout.addLayout(layout)
        self.oeTree = qt.QTreeView()
        self.oeTree.setModel(self.segmentsModel)
        self.oeTree.setIconSize(qt.QSize(32, 32))
        self.oeTree.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.oeTree.customContextMenuRequested.connect(self.oeTreeMenu)
        for clmn in range(4):
            self.oeTree.resizeColumnToContents(clmn)
        self.navigationLayout.addWidget(self.oeTree)
        self.navigationPanel = qt.QWidget(self)
        self.navigationPanel.setLayout(self.navigationLayout)

    def makeTransformationPanel(self):
        self.zoomPanel = qt.QGroupBox(self)
        self.zoomPanel.setFlat(False)
        self.zoomPanel.setTitle("Log scale")
        zoomLayout = qt.QVBoxLayout()
        fitLayout = qt.QHBoxLayout()
        scaleValidator = qt.QDoubleValidator()
        scaleValidator.setRange(0, 7, 7)
        self.zoomSliders = []
        self.zoomEditors = []
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = qt.QLabel(axis)
            axEdit = qt.QLineEdit()
            axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)
            axSlider.setRange(0, 7, 0.01)
            value = 1 if iaxis == 1 else 3
            axSlider.setValue(value)
            axEdit.setText("{0:.2f}".format(value))
            axEdit.setValidator(scaleValidator)
            axEdit.editingFinished.connect(
                partial(self.updateScaleFromQLE, axEdit, axSlider))
            axSlider.valueChanged.connect(
                partial(self.updateScale, axSlider, iaxis, axEdit))
            self.zoomSliders.append(axSlider)
            self.zoomEditors.append(axEdit)

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(12)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addWidget(axSlider)
            zoomLayout.addLayout(layout)

        for iaxis, axis in enumerate(['x', 'y', 'z', 'all']):
            fitX = qt.QPushButton("fit {}".format(axis))
            dim = [iaxis] if iaxis < 3 else [0, 1, 2]
            fitX.clicked.connect(partial(self.fitScales, dim))
            fitLayout.addWidget(fitX)
        zoomLayout.addLayout(fitLayout)
        self.zoomPanel.setLayout(zoomLayout)

        self.rotationPanel = qt.QGroupBox(self)
        self.rotationPanel.setFlat(False)
        self.rotationPanel.setTitle("Rotation (deg)")
        rotationLayout = qt.QVBoxLayout()
        fixedViewsLayout = qt.QHBoxLayout()

        self.rotationSliders = []
        self.rotationEditors = []
        for iaxis, axis in enumerate(['Azimuth', 'Elevation']):
            axLabel = qt.QLabel(axis)
            axEdit = qt.QLineEdit("0.")
            rLim = 89.99 if iaxis else 180.
            rotValidator = qt.QDoubleValidator()
            rotValidator.setRange(-rLim, rLim, 9)
            axEdit.setValidator(rotValidator)
            axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)
            axSlider.setRange(-rLim, rLim, 0.01)
            axSlider.setValue(0)
            axEdit.editingFinished.connect(
                partial(self.updateRotationFromQLE, axEdit, axSlider))
            axSlider.valueChanged.connect(
                partial(self.updateRotation, axSlider, iaxis, axEdit))
            self.rotationSliders.append(axSlider)
            self.rotationEditors.append(axEdit)

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(64)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addWidget(axSlider)
            rotationLayout.addLayout(layout)

        for axis, angles in zip(['Side', 'Front', 'Top', 'Isometric'],
                                [(0., 0.), (89.99, 0.), (0., 89.99),
                                 (-45., 35.)]):
            setView = qt.QPushButton(axis)
            setView.clicked.connect(partial(self.updateRotationFromGL, angles))
            fixedViewsLayout.addWidget(setView)
        rotationLayout.addLayout(fixedViewsLayout)

        self.rotationPanel.setLayout(rotationLayout)

        self.transformationPanel = qt.QWidget(self)
        transformationLayout = qt.QVBoxLayout()
        transformationLayout.addWidget(self.zoomPanel)
        transformationLayout.addWidget(self.rotationPanel)
        transformationLayout.addStretch()
        self.transformationPanel.setLayout(transformationLayout)

    def fitScales(self, dims):
        minmax = self.customGlWidget.minmax

        for dim in dims:
            dimMin = minmax[0, dim]
            dimMax = minmax[1, dim]
            newScale = 1.5 * self.customGlWidget.aPos[dim] /\
                (dimMax - dimMin) * self.customGlWidget.maxLen
            self.customGlWidget.coordOffset = np.zeros(3)
            self.customGlWidget.tVec[dim] = -0.5 * (dimMin + dimMax)
            self.customGlWidget.scaleVec[dim] = newScale
        self.updateScaleFromGL(self.customGlWidget.scaleVec)

    def makeColorsPanel(self):
        self.opacityPanel = qt.QGroupBox(self)
        self.opacityPanel.setFlat(False)
        self.opacityPanel.setTitle("Opacity")

        opacityLayout = qt.QVBoxLayout()
        self.opacitySliders = []
        self.opacityEditors = []
        for iaxis, (axis, rstart, rend, rstep, val) in enumerate(zip(
                ('Line opacity', 'Line width', 'Point opacity', 'Point size'),
                (0, 0, 0, 0), (1., 20., 1., 20.), (0.001, 0.01, 0.001, 0.01),
                (0.2, 2., 0.2, 3.))):
            axLabel = qt.QLabel(axis)
            opacityValidator = qt.QDoubleValidator()
            axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)

            axSlider.setRange(rstart, rend, rstep)
            axSlider.setValue(val)
            axEdit = qt.QLineEdit()
            opacityValidator.setRange(rstart, rend, 5)
            self.updateOpacity(None, iaxis, axEdit, val)

            axEdit.setValidator(opacityValidator)
            axEdit.editingFinished.connect(
                partial(self.updateOpacityFromQLE, axEdit, axSlider))
            axSlider.valueChanged.connect(
                partial(self.updateOpacity, axSlider, iaxis, axEdit))
            self.opacitySliders.append(axSlider)
            self.opacityEditors.append(axEdit)

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(80)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addWidget(axSlider)
            opacityLayout.addLayout(layout)
        self.opacityPanel.setLayout(opacityLayout)

        self.colorPanel = qt.QGroupBox(self)
        self.colorPanel.setFlat(False)
        self.colorPanel.setTitle("Color")
        colorLayout = qt.QVBoxLayout()
        self.mplFig = Figure(dpi=self.logicalDpiX()*0.8)
        self.mplFig.patch.set_alpha(0.)
        self.mplFig.subplots_adjust(left=0.15, bottom=0.15, top=0.92)
        self.mplAx = self.mplFig.add_subplot(111)
        self.mplFig.suptitle("")

        self.drawColorMap('energy')
        self.paletteWidget = qt.FigCanvas(self.mplFig)
        self.paletteWidget.setSizePolicy(qt.QSizePolicy.Expanding,
                                         qt.QSizePolicy.MinimumExpanding)
        self.paletteWidget.setMinimumHeight(260)
        self.paletteWidget.span = RectangleSelector(
            self.mplAx, self.updateColorSelFromMPL,
            # drawtype='box',
            useblit=True,
            # rectprops=dict(alpha=0.4, facecolor='white'),
            button=1, interactive=True)

        layout = qt.QHBoxLayout()
        self.colorControls = []
        colorCBLabel = qt.QLabel('Color Axis:')
        colorCB = qt.QComboBox()
        colorCB.setMaxVisibleItems(48)
        colorCB.setModel(self.fluxDataModel)
        colorCB.setCurrentIndex(colorCB.findText('energy'))
        colorCB.currentIndexChanged['QString'].connect(self.changeColorAxis)
        self.colorControls.append(colorCB)
        layout.addWidget(colorCBLabel)
        layout.addWidget(colorCB)
        layout.addStretch()
        colorLayout.addLayout(layout)
        colorLayout.addWidget(self.paletteWidget)
        colorLayout.addSpacing(8)

        layout = qt.QHBoxLayout()
        for icSel, cSelText in enumerate(['Color Axis min', 'Color Axis max']):
            if icSel > 0:
                layout.addStretch()
            selLabel = qt.QLabel(cSelText)
            selValidator = qt.QDoubleValidator()
            selValidator.setRange(-1.0e20 if icSel == 0 else
                                  self.customGlWidget.colorMin,
                                  self.customGlWidget.colorMax if icSel == 0
                                  else 1.0e20, 5)
            selQLE = qt.QLineEdit()
            selQLE.setValidator(selValidator)
            selQLE.setText('{0:.6g}'.format(
                self.customGlWidget.colorMin if icSel == 0 else
                self.customGlWidget.colorMax))
            selQLE.editingFinished.connect(
                partial(self.updateColorAxis, icSel))
            selQLE.setMaximumWidth(80)
            self.colorControls.append(selQLE)

            layout.addWidget(selLabel)
            layout.addWidget(selQLE)
        colorLayout.addLayout(layout)

        layout = qt.QHBoxLayout()
        for icSel, cSelText in enumerate(['Selection min', 'Selection max']):
            if icSel > 0:
                layout.addStretch()
            selLabel = qt.QLabel(cSelText)
            selValidator = qt.QDoubleValidator()
            selValidator.setRange(self.customGlWidget.colorMin,
                                  self.customGlWidget.colorMax, 5)
            selQLE = qt.QLineEdit()
            selQLE.setValidator(selValidator)
            selQLE.setText('{0:.6g}'.format(
                self.customGlWidget.colorMin if icSel == 0 else
                self.customGlWidget.colorMax))
            selQLE.editingFinished.connect(
                partial(self.updateColorSelFromQLE, selQLE, icSel))
            selQLE.setMaximumWidth(80)
            self.colorControls.append(selQLE)

            layout.addWidget(selLabel)
            layout.addWidget(selQLE)
        colorLayout.addLayout(layout)

        selSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)
        rStep = (self.customGlWidget.colorMax -
                 self.customGlWidget.colorMin) / 100.
        rValue = (self.customGlWidget.colorMax +
                  self.customGlWidget.colorMin) * 0.5
        selSlider.setRange(self.customGlWidget.colorMin,
                           self.customGlWidget.colorMax, rStep)
        selSlider.setValue(rValue)
        selSlider.sliderMoved.connect(partial(self.updateColorSel, selSlider))
        self.colorControls.append(selSlider)
        colorLayout.addWidget(selSlider)

        layout = qt.QHBoxLayout()
        axLabel = qt.QLabel("Intensity cut-off")
        axEdit = qt.QLineEdit("0.01")
        cutValidator = qt.QDoubleValidator()
        cutValidator.setRange(0, 1, 3)
        axEdit.setValidator(cutValidator)
        axEdit.editingFinished.connect(
            partial(self.updateCutoffFromQLE, axEdit))
        axLabel.setMinimumWidth(144)
        layout.addWidget(axLabel)
        axEdit.setMaximumWidth(48)
        layout.addWidget(axEdit)
        layout.addStretch()
        colorLayout.addLayout(layout)

        layout = qt.QHBoxLayout()
        explLabel = qt.QLabel("Color bump height, mm")
        explEdit = qt.QLineEdit("0.0")
        explValidator = qt.QDoubleValidator()
        explValidator.setRange(-1000, 1000, 3)
        explEdit.setValidator(explValidator)
        explEdit.editingFinished.connect(
            partial(self.updateExplosionDepth, explEdit))
        explLabel.setMinimumWidth(144)
        layout.addWidget(explLabel)
        explEdit.setMaximumWidth(48)
        layout.addWidget(explEdit)
        layout.addStretch()
        colorLayout.addLayout(layout)

#        axSlider = qt.glowSlider(
#            self, qt.Qt.Horizontal, qt.glowTopScale)
#        axSlider.setRange(0, 1, 0.001)
#        axSlider.setValue(0.01)
#        axSlider.valueChanged.connect(self.updateCutoff)
#        colorLayout.addWidget(axSlider, 3+3, 0, 1, 2)
        self.colorCbControls = OrderedDict()
        for scControl, cbText in COLOR_CONTROL_LABELS.items():
            aaCheckBox = self._makeCheckBox(cbText, scControl,
                                            self.colorCbControls)
            colorLayout.addWidget(aaCheckBox)

#        glNormCB = qt.QCheckBox('Global Normalization')
#        glNormCB.setChecked(True)
#        glNormCB.stateChanged.connect(self.checkGlobalNorm)
#        colorLayout.addWidget(glNormCB)
#        self.glNormCB = glNormCB
#
#        iHSVCB = qt.QCheckBox('Intensity as HSV Value')
#        iHSVCB.setChecked(False)
#        iHSVCB.stateChanged.connect(self.checkIHSV)
#        colorLayout.addWidget(iHSVCB)
#        self.iHSVCB = iHSVCB

        self.colorPanel.setLayout(colorLayout)

        self.colorOpacityPanel = qt.QWidget(self)
        self.colorOpacityPanel.setProperty('popupMinHeight', 620)
        colorOpacityLayout = qt.QVBoxLayout()
        colorOpacityLayout.addWidget(self.colorPanel)
        colorOpacityLayout.addWidget(self.opacityPanel)
        colorOpacityLayout.addStretch()
        self.colorOpacityPanel.setLayout(colorOpacityLayout)

    def makeGridAndProjectionsPanel(self):
        self.gridPanel = qt.QGroupBox(self)
        self.gridPanel.setFlat(False)
        self.gridPanel.setTitle("Show coordinate grid")
        self.gridPanel.setCheckable(True)
        self.gridPanel.toggled.connect(partial(
                self._setGlFlag, 'drawGrid'))

        scaleValidator = qt.QDoubleValidator()
        scaleValidator.setRange(0, 7, 7)
        xyzGridLayout = qt.QVBoxLayout()
        self.gridSliders = []
        self.gridEditors = []
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = qt.QLabel(axis)
            axEdit = qt.QLineEdit("0.9")
            axEdit.setValidator(scaleValidator)
            axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)
            axSlider.setRange(0, 10, 0.01)
            axSlider.setValue(0.9)
            axEdit.editingFinished.connect(
                partial(self.updateGridFromQLE, axEdit, axSlider))
            axSlider.valueChanged.connect(
                partial(self.updateGrid, axSlider, iaxis, axEdit))
            self.gridSliders.append(axSlider)
            self.gridEditors.append(axEdit)

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(20)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addWidget(axSlider)
            xyzGridLayout.addLayout(layout)

        checkBox = qt.QCheckBox('Fine grid')
        checkBox.setChecked(False)
        checkBox.toggled.connect(partial(
                self._setGlFlag, 'fineGridEnabled'))
        xyzGridLayout.addWidget(checkBox)
        self.checkBoxFineGrid = checkBox
        self.gridControls = []

        projectionLayout = qt.QVBoxLayout()
        checkBox = qt.QCheckBox('Perspective')
        checkBox.setChecked(True)
        checkBox.toggled.connect(partial(
                self._setGlFlag, 'perspectiveEnabled'))
        self.checkBoxPerspective = checkBox
        projectionLayout.addWidget(self.checkBoxPerspective)

        self.gridControls.append(self.checkBoxPerspective)
        self.gridControls.append(self.gridPanel)
        self.gridControls.append(self.checkBoxFineGrid)

        self.gridPanel.setLayout(xyzGridLayout)

        self.projVisPanel = qt.QGroupBox(self)
        self.projVisPanel.setFlat(False)
        self.projVisPanel.setTitle("Projections visibility")
        projVisLayout = qt.QVBoxLayout()
        self.projLinePanel = qt.QGroupBox(self)
        self.projLinePanel.setFlat(False)
        self.projLinePanel.setTitle("Projections opacity")
        self.projectionControls = []
        for iaxis, axis in enumerate(['Side (YZ)', 'Front (XZ)', 'Top (XY)']):
            checkBox = qt.QCheckBox(axis)
            checkBox.setChecked(False)
            checkBox.stateChanged.connect(partial(self.projSelection, iaxis))
            self.projectionControls.append(checkBox)
            projVisLayout.addWidget(checkBox)
        self.projLinePanel.setEnabled(False)

        self.projVisPanel.setLayout(projVisLayout)

        projLineLayout = qt.QVBoxLayout()
        self.projectionOpacitySliders = []
        self.projectionOpacityEditors = []
        for iaxis, axis in enumerate(
                ['Line opacity', 'Line width', 'Point opacity', 'Point size']):
            axLabel = qt.QLabel(axis)
            projectionValidator = qt.QDoubleValidator()
            axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)

            if iaxis in [0, 2]:
                axSlider.setRange(0, 1., 0.001)
                axSlider.setValue(0.1)
                axEdit = qt.QLineEdit("0.1")
                projectionValidator.setRange(0, 1., 5)

            else:
                axSlider.setRange(0, 20, 0.01)
                axSlider.setValue(1.)
                axEdit = qt.QLineEdit("1")
                projectionValidator.setRange(0, 20., 5)

            axEdit.setValidator(projectionValidator)
            axEdit.editingFinished.connect(
                partial(self.updateProjectionOpacityFromQLE, axEdit, axSlider))
            axSlider.valueChanged.connect(
                partial(self.updateProjectionOpacity, axSlider, iaxis, axEdit))
            self.projectionOpacitySliders.append(axSlider)
            self.projectionOpacityEditors.append(axEdit)

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(80)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addWidget(axSlider)
            projLineLayout.addLayout(layout)
        self.projLinePanel.setLayout(projLineLayout)

        self.projectionPanel = qt.QWidget(self)
        projectionLayout.addWidget(self.gridPanel)
        projectionLayout.addWidget(self.projVisPanel)
        projectionLayout.addWidget(self.projLinePanel)
        projectionLayout.addStretch()
        self.projectionPanel.setLayout(projectionLayout)

    def makeScenePanel(self):
        sceneLayout = qt.QVBoxLayout()
        self.sceneControls = OrderedDict()
        self.sceneTextedits = OrderedDict()
        self.sceneSliders = OrderedDict()
        self.rayFlagControls = OrderedDict()

        for scControl, cbText in SCENE_CONTROL_LABELS.items():
            aaCheckBox = self._makeCheckBox(cbText, scControl,
                                            self.sceneControls)
            sceneLayout.addWidget(aaCheckBox)

        for teControl, teProps in SCENE_TEXTEDITS.items():
            teLayout = self._makeLabeledLineEdit(
                    teProps['label'], teProps['tooltip'], teControl,
                    self.sceneTextedits)
            sceneLayout.addLayout(teLayout)

        axLabel = qt.QLabel('Font Size')
        axSlider = qt.glowSlider(self, qt.Qt.Horizontal, qt.glowTopScale)
        axSlider.setRange(1, 20, 0.5)
        axSlider.setValue(4)
        axSlider.valueChanged.connect(partial(self.updateFontSize, axSlider))

        layout = qt.QHBoxLayout()
        layout.addWidget(axLabel)
        layout.addWidget(axSlider)
        sceneLayout.addLayout(layout)

        labelPrec = qt.QComboBox()
        for order in range(5):
            labelPrec.addItem("{}mm".format(10**-order))
        labelPrec.setCurrentIndex(1)
        labelPrec.currentIndexChanged['int'].connect(self.setLabelPrec)
        self.sceneSliders['labelCoordPrec'] = labelPrec
        aaLabel = qt.QLabel('Label Precision')
        layout = qt.QHBoxLayout()
        aaLabel.setMinimumWidth(100)
        layout.addWidget(aaLabel)
        labelPrec.setMaximumWidth(120)
        layout.addWidget(labelPrec)
        layout.addStretch()
        sceneLayout.addLayout(layout)

        oeTileValidator = qt.QIntValidator()
        oeTileValidator.setRange(1, 200)
        for ia, (axis, defv) in enumerate(zip(
                ['OE tessellation X', 'OE tessellation Y'],
                self.customGlWidget.tiles)):

            axLabel = qt.QLabel(axis)
            axEdit = qt.QLineEdit(str(defv))
            axEdit.setValidator(oeTileValidator)
            axEdit.editingFinished.connect(
                partial(self.updateTileFromQLE, axEdit, ia))
            self.sceneTextedits[f'tiles_{ia}'] = axEdit
            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(100)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addStretch()
            sceneLayout.addLayout(layout)

        self.scenePanel = qt.QWidget(self)
        sceneLayout.addWidget(self._makeRayVisibilityPanel())
        sceneLayout.addStretch()
        self.scenePanel.setLayout(sceneLayout)

    def initControlPanels(self):
        self.controlPanels = OrderedDict([
            ("Selection", self.navigationPanel),
            ("Transformations", self.transformationPanel),
            ("Colors", self.colorOpacityPanel),
            ("Grid/Projections", self.projectionPanel),
            ("Scene", self.scenePanel),
            ("Scans", self.scanPanel),
        ])

    def makeControlsTabsWidget(self):
        tabs = qt.QTabWidget()
        for panelName, panelWidget in self.controlPanels.items():
            tabs.addTab(panelWidget, panelName)
        self.sideTabs = tabs
        return tabs

    def makeControlsToolbar(self, control_mode):
        self.controlButtons = OrderedDict()
        self.controlPopup = _ToolbarPopupPanel(self)
        self.controlPopup.popupHidden.connect(self._resetToolbarButtons)
        iconsDir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '_icons')
        panelIcons = {
            "Selection": "p_selection128.png",
            "Transformations": "p_transform128.png",
            "Colors": "p_colors128.png",
            "Grid/Projections": "p_grid128.png",
            "Scene": "p_scene128.png",
            "Scans": "p_scan128.png",
        }

        for panel in self.controlPanels.values():
            self.controlPopup.stack.addWidget(panel)

        toolbar = qt.QToolBar("Controls", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(qt.QSize(64, 64))
        toolbar.setOrientation(
            qt.Qt.Horizontal if control_mode == 'collapsible top'
            else qt.Qt.Vertical)
        self.controlsToolBar = toolbar

        titleLabel = qt.QLabel("Controls")
        titleLabel.setStyleSheet("font-weight: 600; padding: 6px 4px;")
        titleLabel.setAlignment(
            qt.Qt.AlignLeft if control_mode == 'collapsible top'
            else qt.Qt.AlignHCenter)
        toolbar.addWidget(titleLabel)

        for panelName in self.controlPanels.keys():
            button = qt.QToolButton(self)
            iconName = panelIcons.get(panelName)
            iconPath = os.path.join(iconsDir, iconName) if iconName else None
            if iconPath and os.path.exists(iconPath):
                button.setIcon(qt.QIcon(iconPath))
                button.setToolTip(panelName)
                button.setText("")
                button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
            else:
                button.setText(panelName)
                button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
            button.setCheckable(True)
            button.clicked.connect(
                partial(self.toggleControlPanel, panelName, button))
            toolbar.addWidget(button)
            self.controlButtons[panelName] = button

        spacer = qt.QWidget(self)
        if control_mode == 'collapsible top':
            spacer.setSizePolicy(
                qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        else:
            spacer.setSizePolicy(
                qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

    def _resetToolbarButtons(self):
        for button in self.controlButtons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

    def _controlPopupSize(self, panelWidget):
        layout = self.controlPopup.layout()
        margins = layout.contentsMargins()
        spacing = layout.spacing()
        frame = 2 * self.controlPopup.frameWidth()
        titleHint = self.controlPopup.titleLabel.sizeHint()
        panelHint = panelWidget.sizeHint()
        panelMinHint = panelWidget.minimumSizeHint()
        panelMin = panelWidget.minimumSize()

        panelWidth = max(panelHint.width(), panelMinHint.width(),
                         panelMin.width(), 280)
        panelHeight = max(panelHint.height(), panelMinHint.height(),
                          panelMin.height(), 120)
        popupMinHeight = panelWidget.property('popupMinHeight')
        if popupMinHeight is not None:
            panelHeight = max(panelHeight, int(popupMinHeight))
        if panelWidget is getattr(self, 'navigationPanel', None):
            model = self.oeTree.model()
            columnCount = model.columnCount() if model is not None else 0
            treeWidth = sum(self.oeTree.columnWidth(column)
                            for column in range(columnCount))
            rowCount = 0
            parents = [qt.QModelIndex()]
            while model is not None and parents:
                parent = parents.pop()
                for row in range(model.rowCount(parent)):
                    rowCount += 1
                    index = model.index(row, 0, parent)
                    if self.oeTree.isExpanded(index):
                        parents.append(index)
            rowHeight = max(self.oeTree.sizeHintForRow(0),
                            self.oeTree.iconSize().height() + 8,
                            self.oeTree.fontMetrics().height() + 8)
            treeHeight = self.oeTree.header().height() + rowCount * rowHeight
            panelWidth = max(panelWidth, treeWidth + 48, 420)
            panelHeight = max(panelHeight, treeHeight + 32)

        width = max(titleHint.width(), panelHint.width(),
                    panelMinHint.width(), panelMin.width(), panelWidth)
        height = titleHint.height() + spacing + panelHeight

        width += margins.left() + margins.right() + frame
        height += margins.top() + margins.bottom() + frame

        viewSize = self.customGlWidget.size()
        maxWidth = max(420, min(760, int(viewSize.width() * 0.70)))
        if panelWidget in [getattr(self, 'navigationPanel', None),
                           getattr(self, 'colorOpacityPanel', None)]:
            maxHeight = max(320, self.height(), viewSize.height())
        else:
            maxHeight = max(320, min(760, int(viewSize.height() * 0.85)))
        return qt.QSize(min(width, maxWidth), min(height, maxHeight))

    def _controlPopupPosition(self, button, popupSize):
        if self.controlTabMode == 'collapsible top':
            pos = button.mapToGlobal(button.rect().bottomLeft())
            pos.setY(pos.y() + 4)
        elif self.controlTabMode == 'collapsible right':
            pos = button.mapToGlobal(button.rect().topLeft())
            pos.setX(pos.x() - popupSize.width() - 6)
        else:
            pos = button.mapToGlobal(button.rect().topRight())
            pos.setX(pos.x() + 6)

        margin = 6
        origin = self.mapToGlobal(self.rect().topLeft())
        minX = origin.x() + margin
        minY = origin.y() + margin
        maxX = origin.x() + self.width() - popupSize.width() - margin
        maxY = origin.y() + self.height() - popupSize.height() - margin
        pos.setX(min(max(pos.x(), minX), max(minX, maxX)))
        pos.setY(min(max(pos.y(), minY), max(minY, maxY)))
        return pos

    def toggleControlPanel(self, panelName, button, checked):
        if not checked:
            self.controlPopup.hide()
            return

        for otherName, otherButton in self.controlButtons.items():
            if otherName != panelName:
                otherButton.blockSignals(True)
                otherButton.setChecked(False)
                otherButton.blockSignals(False)

        self.controlPopup.titleLabel.setText(panelName)
        panelWidget = self.controlPanels[panelName]
        self.controlPopup.stack.setCurrentWidget(panelWidget)
        self.controlPopup.adjustSize()

        popupSize = self._controlPopupSize(panelWidget)
        self.controlPopup.resize(popupSize)
        popupSize = self.controlPopup.size()
        self.controlPopup.move(self._controlPopupPosition(button, popupSize))
        self.controlPopup.show()
        self.controlPopup.raise_()

    def makeNodeEditorPanel(self):
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.nodeEditorWidget = _FlowGraphPanel(self)
        if self.nodeEditorWidget.scene is not None:
            self.nodeEditorWidget.scene.node_double_clicked.connect(
                self.onNodeEditorDoubleClicked)
        layout.addWidget(self.nodeEditorWidget)
        self.nodeEditorPanel = qt.QWidget(self)
        self.nodeEditorPanel.setLayout(layout)
        self.refreshNodeEditorPanel()

    def onNodeEditorDoubleClicked(self, node):
        element_id = getattr(node.model, 'element_id', None)
        if raycing.is_valid_uuid(element_id):
            self.runElementViewer(element_id)

    def _iterFlowGraph(self):
        nodes = OrderedDict()
        edges = []
        order = []
        seen_edges = set()
        beamline = getattr(self.customGlWidget, 'beamline', None)
        if beamline is None:
            return nodes, edges, order

        oes_dict = getattr(beamline, 'oesDict', {})
        flow_u = getattr(beamline, 'flowU', OrderedDict())

        def add_node(element_id):
            if (not raycing.is_valid_uuid(element_id) or element_id in nodes):
                return
            element_line = oes_dict.get(element_id)
            if element_line is None:
                return
            element_obj = element_line[0]
            if element_obj.name == "VirtualScreen":
                return
            if is_source(element_obj):
                node_kind = 'source'
            elif is_aperture(element_obj):
                node_kind = 'aperture'
            elif is_screen(element_obj):
                node_kind = 'screen'
            else:
                node_kind = 'oe'
            nodes[element_id] = {
                'title': element_obj.name,
                'subtitle': type(element_obj).__name__,
                'node_kind': node_kind,
                'style': FLOW_NODE_STYLES.get(node_kind, FLOW_SCENE_STYLE),
            }
            order.append(element_id)

        for element_id in flow_u.keys():
            add_node(element_id)

        for element_id in oes_dict.keys():
            add_node(element_id)

        for target_id, target_operations in flow_u.items():
            if target_id not in nodes:
                continue
            for kwargset in target_operations.values():
                source_id = kwargset.get('beam', None)
                edge = (source_id, target_id)
                if (raycing.is_valid_uuid(source_id) and source_id in nodes and
                        edge not in seen_edges and source_id != target_id):
                    edges.append(edge)
                    seen_edges.add(edge)

        return nodes, edges, order

    def refreshNodeEditorPanel(self):
        if not hasattr(self, 'nodeEditorWidget'):
            return
        nodes, edges, order = self._iterFlowGraph()
        self.nodeEditorWidget.set_graph(nodes, edges, order)

    def initSegmentsModel(self, isNewModel=True):
        newModel = qt.QStandardItemModel()
        newModel.setHorizontalHeaderLabels(['Rays',
                                            'Footprint',
                                            'Surface',
                                            'Label'])
        if isNewModel:
            headerRow = []
            for i in range(4):
                child = qt.QStandardItem("")
                child.setEditable(False)
                child.setCheckable(True)
                child.setCheckState(
                    qt.Qt.Checked if i < 3 else qt.Qt.Unchecked)
                headerRow.append(child)
            newModel.invisibleRootItem().appendRow(headerRow)
        newModel.itemChanged.connect(self.updateRaysList)
        return newModel

    def getItem(self, iId, itemType='beam', targetId=None):
        item = None
        model = self.segmentsModel
        start_index = model.index(0, 0)
        flags = qt.Qt.MatchExactly
        matches = model.match(start_index, qt.Qt.UserRole, iId, hits=1,
                              flags=flags)
        if matches:
            item = model.item(matches[0].row(), itemTypes[itemType])
            if itemType == 'beam' and item.rowCount() > 0:
                parent_index = model.indexFromItem(item)
                fcIndex = item.child(0, 0).index()
                tgt_matches = model.match(fcIndex, qt.Qt.UserRole,
                                          targetId, hits=-1, flags=flags)
                for line in tgt_matches:
                    if line.parent() == parent_index:
                        item = model.itemFromIndex(line)
                        break
        return item

    def toggleCheckItem(self, oeid, field, status):
        for ie in range(self.segmentsModelRoot.rowCount()):
            item = self.segmentsModelRoot.child(ie, 0)
            itemId = item.data(qt.Qt.UserRole)
            if itemId == oeid:
                fpItem = self.segmentsModelRoot.child(ie, itemTypes[field])
                fpItem.setCheckState(int(status)*2)
                break

    def createRow(self, text, segMode, uuid=None, icon=None):
        newRow = []
        for iCol in range(4):
            newItem = qt.QStandardItem(str(text) if iCol == 0 else "")
            if iCol == 0:
                newItem.setData(uuid, qt.Qt.UserRole)
                if icon is not None:
                    newItem.setIcon(icon)
            newItem.setCheckable(True if (segMode == 3 and iCol == 0) or
                                 (segMode == 1 and iCol > 0) else False)
            if newItem.isCheckable():
                newItem.setCheckState(
                    qt.Qt.Checked if iCol < 3 else qt.Qt.Unchecked)
            newItem.setEditable(False)
            newRow.append(newItem)
        return newRow

#    def updateSegmentsModel(self, arrayOfRays):
#        def copyRow(item, row):
#            newRow = []
#            for iCol in range(4):
#                oldItem = item.child(row, iCol)
#                newItem = qt.QStandardItem(str(oldItem.text()))
#                newItem.setCheckable(oldItem.isCheckable())
#                if newItem.isCheckable():
#                    newItem.setCheckState(oldItem.checkState())
#                newItem.setEditable(oldItem.isEditable())
#                newRow.append(newItem)
#            return newRow
#
#        newSegmentsModel = self.initSegmentsModel(isNewModel=False)
#        newSegmentsModel.invisibleRootItem().appendRow(
#            copyRow(self.segmentsModelRoot, 0))
#        for element, elRecord in self.oesList.items():
#            for iel in range(self.segmentsModelRoot.rowCount()):
#                elItem = self.segmentsModelRoot.child(iel, 0)
#                elName = str(elItem.text())
#                if str(element) == elName:
#                    elRow = copyRow(self.segmentsModelRoot, iel)
#                    for segment in arrayOfRays[0]:
#                        if segment[3] is not None:
#                            endBeamText = "to {}".format(
#                                self.beamsToElements[segment[3]])
#                            if str(segment[1]) == str(elRecord[1]):
#                                if elItem.hasChildren():
#                                    for ich in range(elItem.rowCount()):
#                                        if str(elItem.child(ich, 0).text()) ==\
#                                                endBeamText:
#                                            elRow[0].appendRow(
#                                                copyRow(elItem, ich))
#                                            break
#                                    else:
#                                        elRow[0].appendRow(self.createRow(
#                                            endBeamText, 3))
#                                else:
#                                    elRow[0].appendRow(self.createRow(
#                                        endBeamText, 3))
#                    newSegmentsModel.invisibleRootItem().appendRow(elRow)
#                    break
#            else:
#                elRow = self.createRow(str(element), 1)
#                for segment in arrayOfRays[0]:
#                    if str(segment[1]) == str(elRecord[1]) and\
#                            segment[3] is not None:
#                        endBeamText = "to {}".format(
#                            self.beamsToElements[segment[3]])
#                        elRow[0].appendRow(self.createRow(endBeamText, 3))
#                newSegmentsModel.invisibleRootItem().appendRow(elRow)
#        self.segmentsModel = newSegmentsModel
#        self.centerProxyModel.setSourceModel(self.segmentsModel)
#        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()
#        self.oeTree.setModel(self.segmentsModel)
#        self.refreshNodeEditorPanel()

    def getIcon(self, oe):
        if is_aperture(oe):
            icon = self.iconLib['aperture']
        elif is_source(oe):
            icon = self.iconLib['source']
        elif is_screen(oe):
            icon = self.iconLib['screen']
        else:
            icon = self.iconLib['oe']
        return icon

    def populateSegmentsModel(self, arrayOfRays=None):
        if arrayOfRays is not None:
            for eluuid, elline in arrayOfRays[2].items():
                element = elline[0].name
                newRow = self.createRow(element, 1, uuid=eluuid,
                                        icon=self.getIcon(elline[0]))
                for flowLine in arrayOfRays[0]:
                    if flowLine[0] == eluuid:
                        targetuuid = flowLine[2]
                        if targetuuid is not None and targetuuid != eluuid:
                            try:
                                targetName = arrayOfRays[2][targetuuid][0].name
                                endBeamText = "to {}".format(targetName)
                                newRow[0].appendRow(self.createRow(
                                        endBeamText, 3, uuid=targetuuid))
                            except:
                                continue
                self.segmentsModelRoot.appendRow(newRow)
        else:
            for eluuid, elementLine in\
                    self.customGlWidget.beamline.oesDict.items():
                if elementLine[0].name == "VirtualScreen":
                    continue
                self.addElementToModel(eluuid)
        self.refreshNodeEditorPanel()

    def addElementToModel(self, uuid):
        elementLine = self.customGlWidget.beamline.oesDict.get(uuid)
        if elementLine is not None:
            element = elementLine[0]
            elname = element.name
            newRow = self.createRow(elname, 1, uuid=uuid,
                                    icon=self.getIcon(element))
        else:
            return

        for targetuuid, targetoperations in\
                self.customGlWidget.beamline.flowU.items():
            for kwargset in targetoperations.values():
                if kwargset.get('beam', 'none') == uuid:
                    try:
                        targetName = self.customGlWidget.beamline.oesDict[
                                targetuuid][0].name
                        endBeamText = "to {}".format(targetName)
                        newRow[0].appendRow(self.createRow(
                                endBeamText, 3, uuid=targetuuid))
                    except Exception:
                        continue
        self.segmentsModelRoot.appendRow(newRow)

    def updateNames(self):
        def getOeName(uuid):
            oeline = self.customGlWidget.beamline.oesDict.get(uuid)
            return oeline[0].name if oeline is not None else None

        for iel in range(self.segmentsModelRoot.rowCount()):
            oeItem = self.segmentsModelRoot.child(iel, 0)
            oeid = oeItem.data(qt.Qt.UserRole)
            newName = getOeName(oeid)
            oeItem.setText(newName)
            for iech in range(oeItem.rowCount()):
                targetItem = oeItem.child(iech, 0)
                targetId = targetItem.data(qt.Qt.UserRole)
                targetName = getOeName(targetId)
                targetItem.setText("to {}".format(targetName))
        self.refreshNodeEditorPanel()

    def updateTargets(self):

        tmpDict = dict()
#        self.centerCB.blockSignals(True)
        # Stage 1. Remove children
        for iel in range(self.segmentsModelRoot.rowCount()):
            oeItem = self.segmentsModelRoot.child(iel, 0)
            uuid = oeItem.data(qt.Qt.UserRole)
            for iech in reversed(range(oeItem.rowCount())):
                oeItem.removeRow(iech)

        # Stage 2a. Move all element rows into a temporary dictionary
        for iel in reversed(range(self.segmentsModelRoot.rowCount())):
            oeItem = self.segmentsModelRoot.child(iel, 0)
            uuid = oeItem.data(qt.Qt.UserRole)
            if raycing.is_valid_uuid(uuid):
                tmpDict[uuid] = self.segmentsModelRoot.takeRow(iel)

        # Stage 2b. Return element rows according to new flow order
        for segment in self.customGlWidget.beamline.flowU:
            modelRow = tmpDict.pop(segment, None)
            if modelRow is not None:
                self.segmentsModelRoot.appendRow(modelRow)

        # Stage 2c. Return non-flow elements
        for modelRow in tmpDict.values():
            if modelRow is not None:
                elementId = str(modelRow[0].data(qt.Qt.UserRole))
                if elementId in self.customGlWidget.beamline.oesDict:
                    self.segmentsModelRoot.appendRow(modelRow)

        # Stage 3. Add children
        for iel in range(self.segmentsModelRoot.rowCount()):
            oeItem = self.segmentsModelRoot.child(iel, 0)
            uuid = oeItem.data(qt.Qt.UserRole)
            for targetuuid, targetoperations in\
                    self.customGlWidget.beamline.flowU.items():
                for kwargset in targetoperations.values():
                    if kwargset.get('beam', 'none') == uuid:
                        try:
                            targetName = self.customGlWidget.beamline.oesDict[
                                        targetuuid][0].name
                            endBeamText = "to {}".format(targetName)
                            oeItem.appendRow(self.createRow(
                                    endBeamText, 3, uuid=targetuuid))
#                            print(oeItem.text(), ": Appending", endBeamText)
                        except:  # analysis:ignore
                            continue

        # Stage 4. Clear dead references
        tmpDict.clear()
        self.oeTree.resizeColumnToContents(0)
#        self.centerCB.blockSignals(False)
        self.refreshNodeEditorPanel()

    def drawColorMap(self, axis):
        xv, yv = np.meshgrid(np.linspace(0, colorFactor, 200),
                             np.linspace(0, 1, 200))
        xv = xv.flatten()
        yv = yv.flatten()
        self.im = self.mplAx.imshow(hsv_to_rgb(np.vstack((
            xv, np.ones_like(xv)*colorSaturation, yv)).T).reshape((
                200, 200, 3)),
            aspect='auto', origin='lower',
            extent=(self.customGlWidget.colorMin,
                    self.customGlWidget.colorMax,
                    0, 1))
        self.mplAx.set_xlabel(axis)
        self.mplAx.set_ylabel('Intensity')

    def updateColorMap(self, histArray):
        if histArray[0] is not None:
            size = len(histArray[0])
            histImage = np.zeros((size, size, 3))
            colorMin = self.customGlWidget.colorMin
            colorMax = self.customGlWidget.colorMax
            hMax = np.float64(np.max(histArray[0]))
            intensity = np.float64(np.array(histArray[0]) / hMax)
            histVals = np.int32(intensity * (size-1))
            for col in range(size):
                histImage[0:histVals[col], col, :] = hsv_to_rgb(
                    (colorFactor * (histArray[1][col] - colorMin) /
                     (colorMax - colorMin),
                     colorSaturation, intensity[col]))
            self.im.set_data(histImage)
            try:
                topEl = np.where(intensity >= 0.5)[0]
                hwhm = (np.abs(histArray[1][topEl[0]] -
                               histArray[1][topEl[-1]])) * 0.5
                cntr = (histArray[1][topEl[0]] + histArray[1][topEl[-1]]) * 0.5
                newLabel = u"{0:.3f}\u00b1{1:.3f}".format(cntr, hwhm)
                self.mplAx.set_title(newLabel, fontsize=self.cAxisLabelSize)
            except Exception:
                pass
            self.mplFig.canvas.draw()
            self.mplFig.canvas.blit()
            self.paletteWidget.span.extents = self.paletteWidget.span.extents
        else:
            xv, yv = np.meshgrid(np.linspace(0, colorFactor, 200),
                                 np.linspace(0, 1, 200))
            xv = xv.flatten()
            yv = yv.flatten()
            self.im.set_data(mpl.colors.hsv_to_rgb(np.vstack((
                xv, np.ones_like(xv)*colorSaturation, yv)).T).reshape((
                    200, 200, 3)))
            self.mplAx.set_title("")
            self.mplFig.canvas.draw()
            self.mplFig.canvas.blit()
            try:
                if self.paletteWidget.span.visible:
                    self.paletteWidget.span.extents =\
                        self.paletteWidget.span.extents
            except AttributeError:
                pass
        self.mplFig.canvas.blit()

    def setSceneParam(self, iAction, state):
        self.sceneControls[iAction].setChecked(state)

    def setProjectionParam(self, iAction, state):
        self.projectionControls[iAction].setChecked(state)

    def setGridParam(self, iAction, state):
        self.gridControls[iAction].setChecked(state)

    def setLabelPrec(self, prec):
        self.customGlWidget.labelCoordPrec = prec
        self.customGlWidget.glDraw()

    def updateColorAxis(self, icSel):
        if icSel == 0:
            txt = re.sub(',', '.', str(self.colorControls[1].text()))
            if txt == "{0:.3f}".format(self.customGlWidget.colorMin):
                return
            newColorMin = float(txt)
            self.customGlWidget.colorMin = newColorMin
            self.colorControls[2].validator().setBottom(newColorMin)
        else:
            txt = re.sub(',', '.', str(self.colorControls[2].text()))
            if txt == "{0:.3f}".format(self.customGlWidget.colorMax):
                return
            newColorMax = float(txt)
            self.customGlWidget.colorMax = newColorMax
            self.colorControls[1].validator().setTop(newColorMax)
        self.changeColorAxis(None, newLimits=True)

    def changeColorAxis(self, selAxis, newLimits=False):
        if selAxis is None:
            selAxis = self.colorControls[0].currentText()
            self.customGlWidget.newColorAxis = False if\
                self.customGlWidget.selColorMin is not None else True
        else:
            self.customGlWidget.colorAxis = str(selAxis)
            self.customGlWidget.newColorAxis = True
        oldColorMin = self.customGlWidget.colorMin
        oldColorMax = self.customGlWidget.colorMax
        self.customGlWidget.change_beam_colorax()
        self.mplAx.set_xlabel(selAxis)
        if oldColorMin == self.customGlWidget.colorMin and\
                oldColorMax == self.customGlWidget.colorMax and not newLimits:
            return
        self.customGlWidget.selColorMin = self.customGlWidget.colorMin
        self.customGlWidget.selColorMax = self.customGlWidget.colorMax
        extents = (self.customGlWidget.colorMin,
                   self.customGlWidget.colorMax, 0, 1)
        self.im.set_extent(extents)
        self.mplFig.gca().ticklabel_format(useOffset=True)
#        self.mplFig.gca().autoscale_view()
        extents = list(extents)
        self.colorControls[1].setText(
            '{0:.3f}'.format(self.customGlWidget.colorMin))
        self.colorControls[2].setText(
            '{0:.3f}'.format(self.customGlWidget.colorMax))
        self.colorControls[3].setText(
            '{0:.3f}'.format(self.customGlWidget.colorMin))
        self.colorControls[4].setText(
            '{0:.3f}'.format(self.customGlWidget.colorMax))
        self.colorControls[3].validator().setRange(
            self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
        self.colorControls[4].validator().setRange(
            self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
        slider = self.colorControls[5]
        center = 0.5 * (extents[0] + extents[1])
        newMin = self.customGlWidget.colorMin
        newMax = self.customGlWidget.colorMax
        newRange = (newMax - newMin) * 0.01
        slider.setRange(newMin, newMax, newRange)
        slider.setValue(center)
        self.mplFig.canvas.draw()
        self.paletteWidget.span.active_handle = None
        # self.paletteWidget.span.to_draw.set_visible(False)
        self.customGlWidget.glDraw()

    def updateColorSelFromMPL(self, eclick, erelease):
        try:
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.selColorMin = np.min([extents[0], extents[1]])
            self.customGlWidget.selColorMax = np.max([extents[0], extents[1]])
            self.colorControls[3].setText(
                "{0:.3f}".format(self.customGlWidget.selColorMin))
            self.colorControls[4].setText(
                "{0:.3f}".format(self.customGlWidget.selColorMax))
            self.colorControls[3].validator().setTop(
                self.customGlWidget.selColorMax)
            self.colorControls[4].validator().setBottom(
                self.customGlWidget.selColorMin)
            slider = self.colorControls[5]
            center = 0.5 * (extents[0] + extents[1])
            halfWidth = (extents[1] - extents[0]) * 0.5
            newMin = self.customGlWidget.colorMin + halfWidth
            newMax = self.customGlWidget.colorMax - halfWidth
            newRange = (newMax - newMin) * 0.01
            slider.setRange(newMin, newMax, newRange)
            slider.setValue(center)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except Exception:
            pass

    def updateColorSel(self, slider, position):
        if isinstance(position, int):
            try:
                position /= slider.scale
            except Exception:
                pass
        try:
            extents = list(self.paletteWidget.span.extents)
            width = np.abs(extents[1] - extents[0])
            self.customGlWidget.selColorMin = position - 0.5*width
            self.customGlWidget.selColorMax = position + 0.5*width
            self.colorControls[3].setText('{0:.3f}'.format(position-0.5*width))
            self.colorControls[4].setText('{0:.3f}'.format(position+0.5*width))
            self.colorControls[3].validator().setTop(position + 0.5*width)
            self.colorControls[4].validator().setBottom(position - 0.5*width)
            newExtents = (position - 0.5*width, position + 0.5*width,
                          extents[2], extents[3])
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except Exception:
            pass

    def updateColorSelFromQLE(self, editor, icSel):
        try:
            txt = str(editor.text())
            value = float(txt)
            extents = list(self.paletteWidget.span.extents)
            if icSel == 0:
                if txt == "{0:.3f}".format(self.customGlWidget.selColorMin):
                    return
                if value < self.customGlWidget.colorMin:
                    self.im.set_extent(
                        [value, self.customGlWidget.colorMax, 0, 1])
                    self.customGlWidget.colorMin = value
                self.customGlWidget.selColorMin = value
                newExtents = (value, extents[1], extents[2], extents[3])
#                self.colorControls[2].validator().setBottom(value)
            else:
                if txt == "{0:.3f}".format(self.customGlWidget.selColorMax):
                    return
                if value > self.customGlWidget.colorMax:
                    self.im.set_extent(
                        [self.customGlWidget.colorMin, value, 0, 1])
                    self.customGlWidget.colorMax = value
                self.customGlWidget.selColorMax = value
                newExtents = (extents[0], value, extents[2], extents[3])
#                self.colorControls[1].validator().setTop(value)
            center = 0.5 * (newExtents[0] + newExtents[1])
            halfWidth = (newExtents[1] - newExtents[0]) * 0.5
            newMin = self.customGlWidget.colorMin + halfWidth
            newMax = self.customGlWidget.colorMax - halfWidth
            newRange = (newMax - newMin) * 0.01
            slider = self.colorControls[5]
            slider.setRange(newMin, newMax, newRange)
            slider.setValue(center)
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
            self.mplFig.canvas.draw()
        except Exception:
            pass

    def projSelection(self, ind, state):
        self.customGlWidget.projectionsVisibility[ind] = state
        self.customGlWidget.glDraw()
        anyOf = False
        for proj in self.projectionControls:
            anyOf = anyOf or proj.isChecked()
            if anyOf:
                break
        self.projLinePanel.setEnabled(anyOf)

    def updateRotation(self, slider, iax, editor, position):
        if isinstance(position, int):
            try:
                position /= slider.scale
            except Exception:
                pass
        editor.setText("{0:.2f}".format(position))
        self.customGlWidget.rotations[iax] = np.float32(position)
        self.customGlWidget.updateQuats()
        self.customGlWidget.glDraw()

    def updateRotationFromGL(self, actPos):
        self.customGlWidget.rotations = np.float32(actPos)
        for iaxis, (slider, editor) in\
                enumerate(zip(self.rotationSliders, self.rotationEditors)):
            value = actPos[iaxis]
            oldState = slider.blockSignals(True)
            try:
                slider.setValue(value)
                editor.setText("{0:.2f}".format(value))
            finally:
                slider.blockSignals(oldState)
        self.customGlWidget.updateQuats()
        self.customGlWidget.glDraw()

    def updateRotationFromQLE(self, editor, slider):
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateScale(self, slider, iax, editor, position):
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                pass
        editor.setText("{0:.2f}".format(position))
        self.customGlWidget.scaleVec[iax] = np.float32(np.power(10, position))
        try:
            self.customGlWidget.update_coord_grid()
        except AttributeError:
            pass
        self.customGlWidget.glDraw()

    def updateScaleFromGL(self, scale):
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        self.customGlWidget.scaleVec = np.float32(scale)
        for iaxis, (slider, editor) in \
                enumerate(zip(self.zoomSliders, self.zoomEditors)):
            value = np.log10(scale[iaxis])
            oldState = slider.blockSignals(True)
            try:
                slider.setValue(value)
                editor.setText("{0:.2f}".format(value))
            finally:
                slider.blockSignals(oldState)
        try:
            self.customGlWidget.update_coord_grid()
        except AttributeError:
            pass
        self.customGlWidget.glDraw()

    def updateScaleFromQLE(self, editor, slider):
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateMaxLenFromGL(self, value):
        mlte = self.sceneTextedits.get('maxLen')
        if mlte is not None:
            mlte.setText("{0:.2f}".format(float(value)))

    def updateFontSize(self, slider, position):
        if isinstance(position, int):
            try:
                position /= slider.scale
            except Exception:
                pass
        self.customGlWidget.cBox.fontScale = position
        self.customGlWidget.glDraw()

    def updateRaysList(self, item):
        if item.parent() is None:
            if item.row() == 0:
                if item.checkState != 1:
                    model = item.model()
                    column = item.column()
                    model.blockSignals(True)
                    parent = self.segmentsModelRoot
                    try:
                        for iChild in range(parent.rowCount()):
                            if iChild > 0:
                                cItem = parent.child(iChild, column)
                                if cItem.isCheckable():
                                    cItem.setCheckState(
                                        item.checkState())
                                if cItem.hasChildren():
                                    for iGChild in range(cItem.rowCount()):
                                        gcItem = cItem.child(iGChild, 0)
                                        if gcItem.isCheckable():
                                            gcItem.setCheckState(
                                                item.checkState())
                    finally:
                        model.blockSignals(False)
                        model.layoutChanged.emit()
            else:
                parent = self.segmentsModelRoot
                model = item.model()
                for iChild in range(parent.rowCount()):
                    outState = item.checkState()
                    if iChild > 0:
                        cItem = parent.child(iChild, item.column())
                        if item.column() > 0:
                            if cItem.checkState() != item.checkState():
                                outState = 1
                                break
                model.blockSignals(True)
                parent.child(0, item.column()).setCheckState(outState)
                model.blockSignals(False)
                model.layoutChanged.emit()
        else:
            parent = self.segmentsModelRoot
            model = item.model()
            for iChild in range(parent.rowCount()):
                outState = item.checkState()
                if iChild > 0:
                    cItem = parent.child(iChild, item.column())
                    if cItem.hasChildren():
                        for iGChild in range(cItem.rowCount()):
                            gcItem = cItem.child(iGChild, 0)
                            if gcItem.isCheckable():
                                if gcItem.checkState() !=\
                                        item.checkState():
                                    outState = 1
                                    break
                if outState == 1:
                    break
            model.blockSignals(True)
            parent.child(0, item.column()).setCheckState(outState)
            model.blockSignals(False)
            model.layoutChanged.emit()

#        if item.column() == 3:
#            self.customGlWidget.labelsToPlot = []
#            for ioe in range(self.segmentsModelRoot.rowCount() - 1):
#                if self.segmentsModelRoot.child(ioe + 1, 3).checkState() == 2:
#                    self.customGlWidget.labelsToPlot.append(str(
#                        self.segmentsModelRoot.child(ioe + 1, 0).text()))
#        else:
#            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def oeTreeMenu(self, position):
        indexes = self.oeTree.selectedIndexes()
        level = 100
        if len(indexes) > 0:
            level = 0
            index = indexes[0]
            selectedItem = self.segmentsModel.itemFromIndex(index)
            while index.parent().isValid():
                index = index.parent()
                level += 1
        if level == 0:
            menu = qt.QMenu()
            menu.addAction('Center here',
                           partial(self.centerEl,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('to Local',
                           partial(self.toLocal,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('Align Beam with Y',
                           partial(self.toBeamLocal,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('restore Global',
                           partial(self.toGlobal,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('View Properties',
                           partial(self.runElementViewer,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('Export OE shape to STL',
                           partial(self.exportOeShape,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.exec_(qt.QCursor.pos())
        else:
            pass

    def updateGrid(self, slider, iax, editor, position):
        if isinstance(position, int):
            try:
                position /= slider.scale
            except Exception:
                pass
        editor.setText("{0:.2f}".format(position))
        if position != 0:
            self.customGlWidget.aPos[iax] = np.float32(position)
            self.customGlWidget.update_coord_grid()
            self.customGlWidget.glDraw()

    def updateGridFromQLE(self, editor, slider):
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateGridFromGL(self, aPos):
        self.customGlWidget.aPos = np.float32(aPos)
        for iaxis, (slider, editor) in\
                enumerate(zip(self.gridSliders, self.gridEditors)):
            value = aPos[iaxis]
            oldState = slider.blockSignals(True)
            try:
                slider.setValue(value)
                editor.setText("{0:.2f}".format(value))
            finally:
                slider.blockSignals(oldState)
        self.customGlWidget.update_coord_grid()
        self.customGlWidget.glDraw()

    def glMenu(self, position):
        glw = self.customGlWidget
        menu = qt.QMenu()
        subMenuF = menu.addMenu('File')
        for actText, actFunc in zip(['Export to image', 'Save scene geometry',
                                     'Load scene geometry'],
                                    [self.exportToImage, self.saveSceneDialog,
                                     self.loadSceneDialog]):
            mAction = qt.QAction(self)
            mAction.setText(actText)
            mAction.triggered.connect(actFunc)
            subMenuF.addAction(mAction)
        menu.addSeparator()
        mAction = qt.QAction(self)
        mAction.setText("Show Virtual Screen")
        mAction.setCheckable(True)
        mAction.setChecked(glw.showVirtualScreen)
        mAction.triggered.connect(glw.toggleVScreen)
        menu.addAction(mAction)
        for iAction, actCnt in self.sceneControls.items():
            if 'Virtual Screen' not in actCnt.text():
                continue
            mAction = qt.QAction(self)
            mAction.setText(actCnt.text())
            mAction.setCheckable(True)
            mAction.setChecked(bool(actCnt.checkState()))
            mAction.triggered.connect(partial(self.setSceneParam, iAction))
            menu.addAction(mAction)
        menu.addSeparator()
        for iAction, actCnt in enumerate(self.gridControls):
            mAction = qt.QAction(self)
            if actCnt.staticMetaObject.className() == 'QCheckBox':
                actText = actCnt.text()
                actCheck = bool(actCnt.checkState())
            else:
                actText = actCnt.title()
                actCheck = actCnt.isChecked()
            mAction.setText(actText)
            mAction.setCheckable(True)
            mAction.setChecked(actCheck)
            mAction.triggered.connect(
                partial(self.setGridParam, iAction))
            if iAction == 0:  # perspective
                menu.addAction(mAction)
            elif iAction == 1:  # show grid
                subMenuG = menu.addMenu('Coordinate grid')
                subMenuG.addAction(mAction)
            elif iAction == 2:  # fine grid
                subMenuG.addAction(mAction)
        menu.addSeparator()
        subMenuP = menu.addMenu('Projections')
        for iAction, actCnt in enumerate(self.projectionControls):
            mAction = qt.QAction(self)
            mAction.setText(actCnt.text())
            mAction.setCheckable(True)
            mAction.setChecked(bool(actCnt.checkState()))
            mAction.triggered.connect(
                partial(self.setProjectionParam, iAction))
            subMenuP.addAction(mAction)
        menu.addSeparator()
        subMenuS = menu.addMenu('Scene')
        for iAction, actCnt in self.sceneControls.items():
            if 'Virtual Screen' in actCnt.text():
                continue
            mAction = qt.QAction(self)
            mAction.setText(actCnt.text())
            mAction.setCheckable(True)
            mAction.setChecked(bool(actCnt.checkState()))
            mAction.triggered.connect(partial(self.setSceneParam, iAction))
            subMenuS.addAction(mAction)
        menu.addSeparator()
        if glw.selectedOE in glw.selectableOEs:
            oe = glw.beamline.oesDict[glw.selectableOEs[int(
                    glw.selectedOE)]][0]
#            oe = glw.uuidDict[glw.selectableOEs[int(glw.selectedOE)]]
            oeName = str(oe.name)
            oeuuid = oe.uuid
            menu.addAction('Center view at {}'.format(oeName),
                           partial(self.centerEl, oeuuid))
            menu.addAction('Transform to {} Local'.format(oeName),
                           partial(self.toLocal, oeuuid))
            menu.addAction('Align Beam with Y',
                           partial(self.toBeamLocal, oeuuid))
            menu.addAction('Restore Global at {}.center'.format(oeName),
                           partial(self.toGlobal, oeuuid))
            menu.addAction('View Properties',
                           partial(self.runElementViewer, oeuuid))
            menu.addAction('Export OE shape to STL',
                           partial(self.exportOeShape, oeuuid))
        menu.addSeparator()
        menu.exec_(self.customGlWidget.mapToGlobal(position))

    def exportToImage(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        saveDialog.setNameFilter(
            "BMP files (*.bmp);;JPG files (*.jpg);;JPEG files (*.jpeg);;"
            "PNG files (*.png);;TIFF files (*.tif)")
        saveDialog.selectNameFilter("JPG files (*.jpg)")
        if (saveDialog.exec_()):
            image = self.customGlWidget.grabFramebuffer()
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            image.save(filename)

    def exportOeShape(self, oeid):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("STL files (*.stl)")
        saveDialog.selectNameFilter("STL files (*.stl)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            mesh = self.customGlWidget.meshDict.get(oeid)
            if mesh is not None:
                mesh.export_stl(filename)

    def saveSceneDialog(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("Numpy files (*.npy)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = 'npy'
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            self.saveScene(filename)

    def loadSceneDialog(self):
        loadDialog = qt.QFileDialog()
        loadDialog.setFileMode(qt.QFileDialog.AnyFile)
        loadDialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
        loadDialog.setNameFilter("Numpy files (*.npy)")  # analysis:ignore
        if (loadDialog.exec_()):
            filename = loadDialog.selectedFiles()[0]
            extension = 'npy'
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            self.loadScene(filename)

    def saveScene(self, filename):
        params = dict()
        for param in list(DEFAULT_SCENE_SETTINGS.keys()):
            params[param] = getattr(self.customGlWidget, param)
        params['size'] = self.geometry()
        params['sizeGL'] = self.canvasSplitter.sizes()

        try:
            np.save(filename, params)
#            with open(filename+'.json', 'w') as json_file:
#                json.dump(params, json_file, indent=4)
        except Exception as e:  # analysis:ignore
            print('Error saving file', e)
            return
        print('Saved scene to {}'.format(filename))

    def loadScene(self, filename):
        try:
            params = np.load(filename, allow_pickle=True).item()
        except Exception as e:  # analysis:ignore
            print('Error loading file', e)
            return

        print('Loaded scene from {}'.format(filename))

        self.applySceneProperties(params)

    def applySceneProperties(self, params):
        if not params:
            return

        for pName, pValue in params.items():
            if pName in ['scaleVec', 'rotations',
                         'tmpOffset', 'tVec', 'coordOffset']:
                pValue = np.array(pValue)
            elif pName == 'rayFlag':
                pValue = set(pValue)
            setattr(self.customGlWidget, pName, pValue)

        if 'size' in params:
            self.setGeometry(params['size'])
        if 'sizeGL' in params:
            self.canvasSplitter.setSizes(params['sizeGL'])
        if 'scaleVec' in params:
            self.updateScaleFromGL(self.customGlWidget.scaleVec)
        if 'rotations' in params:
            self.updateRotationFromGL(self.customGlWidget.rotations)
        if any([x in params for x in ['colorAxis', 'colorMin', 'colorMax',
                                      'selColorMin', 'selColorMax']]):
            colorCB = self.colorControls[0]
            currentIndex = colorCB.findText(self.customGlWidget.colorAxis)
            if currentIndex >= 0:
                updateStatus = colorCB.blockSignals(True)
                colorCB.setCurrentIndex(currentIndex)
                colorCB.blockSignals(updateStatus)
            self.mplAx.set_xlabel(self.customGlWidget.colorAxis)
            self.im.set_extent((self.customGlWidget.colorMin,
                                self.customGlWidget.colorMax, 0, 1))
            self.colorControls[1].setText('{0:.6g}'.format(
                self.customGlWidget.colorMin))
            self.colorControls[2].setText('{0:.6g}'.format(
                self.customGlWidget.colorMax))
            self.colorControls[3].setText('{0:.6g}'.format(
                self.customGlWidget.selColorMin))
            self.colorControls[4].setText('{0:.6g}'.format(
                self.customGlWidget.selColorMax))
            self.colorControls[1].validator().setRange(
                -1.0e20, self.customGlWidget.colorMax, 5)
            self.colorControls[2].validator().setRange(
                self.customGlWidget.colorMin, 1.0e20, 5)
            self.colorControls[3].validator().setRange(
                self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
            self.colorControls[4].validator().setRange(
                self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
            slider = self.colorControls[5]
            rStep = (self.customGlWidget.colorMax -
                     self.customGlWidget.colorMin) / 100.
            rValue = (self.customGlWidget.colorMax +
                      self.customGlWidget.colorMin) * 0.5
            slider.setRange(self.customGlWidget.colorMin,
                            self.customGlWidget.colorMax, rStep)
            slider.setValue(rValue)
            try:
                self.paletteWidget.span.extents = (
                    self.customGlWidget.selColorMin,
                    self.customGlWidget.selColorMax, 0, 1)
            except Exception:
                pass
            self.paletteWidget.span.active_handle = None

        self.blockSignals(True)

        if any([x in params for x in ['lineOpacity', 'lineWidth',
                                      'pointOpacity', 'pointSize']]):
            self.updateOpacityFromGL([self.customGlWidget.lineOpacity,
                                      self.customGlWidget.lineWidth,
                                      self.customGlWidget.pointOpacity,
                                      self.customGlWidget.pointSize])

        for scCtrlName, scCtrlCB in ChainMap(self.sceneControls,
                                             self.colorCbControls).items():
            if scCtrlName in params:
                scCtrlCB.setChecked(params[scCtrlName])

        if any([x in params for x in ['lineProjectionOpacity',
                                      'lineProjectionWidth',
                                      'pointProjectionOpacity',
                                      'pointProjectionSize']]):
            self.updateProjectionOpacityFromGL(
                [self.customGlWidget.lineProjectionOpacity,
                 self.customGlWidget.lineProjectionWidth,
                 self.customGlWidget.pointProjectionOpacity,
                 self.customGlWidget.pointProjectionSize])

        if 'drawGrid' in params:
            self.gridPanel.setChecked(self.customGlWidget.drawGrid)
        if 'projectionsVisibility' in params:
            for iax, checkBox in enumerate(self.projectionControls):
                checkBox.setChecked(
                        self.customGlWidget.projectionsVisibility[iax])
#        if 'fineGrid' in params:
#            self.checkBoxFineGrid.setChecked(
#                    self.customGlWidget.fineGridEnabled)
        if 'aPos' in params:
            self.updateGridFromGL(self.customGlWidget.aPos)
        if 'perspectiveEnabled' in params:
            self.checkBoxPerspective.setChecked(
                    self.customGlWidget.perspectiveEnabled)

        if 'showLostRays' in params and params['showLostRays']:
            self.customGlWidget.rayFlag = set(self.customGlWidget.rayFlag)
            self.customGlWidget.rayFlag.add(4)
        if 'rayFlag' in params or 'showLostRays' in params:
            rayFlags = set(getattr(self.customGlWidget, 'rayFlag', set()))
            for rayFlag, rayFlagCB in self.rayFlagControls.items():
                rayFlagCB.blockSignals(True)
                rayFlagCB.setChecked(rayFlag in rayFlags)
                rayFlagCB.blockSignals(False)
            self.customGlWidget.showLostRays = 4 in rayFlags

        for scProp in ['oeThickness', 'oeThicknessForce',
                       'slitThicknessFraction', 'maxLen']:
            if scProp in params:
                scPval = params.get(scProp)
                self.sceneTextedits[scProp].setText(
                        "{0:.2f}".format(scPval) if scPval is not None else "")

        if 'labelCoordPrec' in params:
            self.sceneSliders['labelCoordPrec'].setCurrentIndex(
                   params['labelCoordPrec'])

        if 'tiles' in params:
            for itn, tnv in enumerate(params['tiles']):
                self.sceneTextedits[f'tiles_{itn}'].setText(str(tnv))

        self.blockSignals(False)
        self.mplFig.canvas.draw()
        self.customGlWidget.glDraw()

#        newExtents = list(self.paletteWidget.span.extents)
#        newExtents[0] = params['selColorMin']
#        newExtents[1] = params['selColorMax']

#        try:
#            self.paletteWidget.span.extents = newExtents
#        except:  # analysis:ignore
#            pass
#        self.updateColorSelFromMPL(0, 0)

    def startRecordingMovie(self):  # by F7
        if self.generator is None:
            return
        startFrom = self.startFrom if hasattr(self, 'startFrom') else 0
        for it in self.generator(*self.generatorArgs):
            self.bl.propagate_flow(startFrom=startFrom)
            rayPath = self.bl.export_to_glow()
            self.updateOEsList(rayPath)
            self.customGlWidget.glDraw()
            if self.isHidden():
                self.show()
            image = self.customGlWidget.grabFrameBuffer(withAlpha=True)
            try:
                image.save(self.bl.glowFrameName)
                cNameSp = os.path.splitext(self.bl.glowFrameName)
                cName = cNameSp[0] + "_color" + cNameSp[1]
                self.mplFig.savefig(cName)
            except AttributeError:
                print('no glowFrameName was given!')
        print("Finished with the movie.")

    def openHelpDialog(self):
        d = qt.QDialog(self)
        d.setMinimumSize(300, 200)
        d.resize(600, 600)
        layout = qt.QVBoxLayout()
        helpText = """
- **F1**: Open this help window
- **F3**: Add/Remove Virtual Screen
- **F4**: Dock/Undock xrtGlow if launched from xrtQook
- **F5/F6**: Quick Save/Load Scene
                    """
        if hasattr(self, 'generator'):
            helpText += "- **F7**: Start recording movie"
        helpText += """
- **LeftMouse**: Rotate the Scene
- **SHIFT+LeftMouse**: Translate in perpendicular to the shortest view axis
- **ALT+LeftMouse**: Translate in parallel to the shortest view axis
- **CTRL+LeftMouse**: Drag Virtual Screen
- **ALT+WheelMouse**: Scale Virtual Screen
- **CTRL+SHIFT+LeftMouse**: Translate the Beamline around Virtual Screen
                      (with Beamline along the longest view axis)
- **CTRL+ALT+LeftMouse**: Translate the Beamline around Virtual Screen
                    (with Beamline along the shortest view axis)
- **CTRL+T**: Toggle Virtual Screen orientation (vertical/normal to the beam)
- **WheelMouse**: Zoom the Beamline
- **CTRL+WheelMouse**: Zoom the Scene
        """
        helpWidget = qt.QTextEdit()
        try:
            helpWidget.setMarkdown(helpText)
        except AttributeError:
            helpWidget.setText(helpText)
        helpWidget.setReadOnly(True)
        layout.addWidget(helpWidget)
        closeButton = qt.QPushButton("Close", d)
        closeButton.clicked.connect(d.hide)
        layout.addWidget(closeButton)
        d.setLayout(layout)
        d.setWindowTitle("Quick Help")
        d.setModal(False)
        d.show()

    def centerEl(self, oeName):
        oeLine = self.customGlWidget.beamline.oesDict.get(oeName)
        if oeLine is None:
            return
        if any([isinstance(x, str) for x in oeLine[0].center]):  # raw 'auto'
            return
        off0 = np.array(oeLine[0].center) - np.array(
            self.customGlWidget.tmpOffset)
        cOffset = qt.QVector4D(off0[0], off0[1], off0[2], 0)
        off1 = self.customGlWidget.mModLocal * cOffset
        self.customGlWidget.coordOffset = np.array(
            [off1.x(), off1.y(), off1.z()])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        if hasattr(self.customGlWidget, 'cBox'):
            self.customGlWidget.update_coord_grid()
        self.customGlWidget.glDraw()

    def toLocal(self, oeuuid):
        oe = self.customGlWidget.beamline.oesDict[oeuuid][0]
        self.customGlWidget.mModLocal =\
            self.customGlWidget.meshDict[oeuuid].transMatrix[0].inverted()[0]
        self.customGlWidget.coordOffset = np.float32([0, 0, 0])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.tmpOffset = oe.center
        self.customGlWidget.update_coord_grid()
        self.customGlWidget.glDraw()

    def toGlobal(self, oeuuid):
        self.customGlWidget.mModLocal = qt.QMatrix4x4()
        self.customGlWidget.tmpOffset = np.float32([0, 0, 0])
        self.customGlWidget.coordOffset = list(
                self.customGlWidget.beamline.oesDict[oeuuid][0].center)
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.update_coord_grid()
        self.customGlWidget.glDraw()

    def toBeamLocal(self, oeuuid):
        bEnd0 = None

        oeStart = self.customGlWidget.beamline.oesDict[oeuuid][0]
        bStart0 = oeStart.center

        if self.customGlWidget.renderingMode == 'dynamic':
            for elid, operations in self.customGlWidget.beamline.flowU.items():
                for kwargset in operations.values():
                    if 'beam' in kwargset and kwargset['beam'] == oeuuid:
                        oeEnd = self.customGlWidget.beamline.oesDict[elid][0]
                        bEnd0 = oeEnd.center
                        break
                else:
                    continue
                break
        else:
            for flowLine in self.customGlWidget.beamline.flow:
                sourceuuid = flowLine[0]
                if sourceuuid == oeuuid:
                    elid = flowLine[2]
                    oeLine = self.customGlWidget.beamline.oesDict.get(elid)
                    if oeLine is not None:
                        oeEnd = oeLine[0]
                        bEnd0 = oeEnd.center
                        break

        if bEnd0 is None:
            return

        if any([isinstance(x, str) for x in bEnd0]):  # unresolved auto
            return

        transMatrix = self.customGlWidget.meshDict[oeuuid].transMatrix[0]
        bEndLoc = transMatrix.inverted()[0] * qt.QVector3D(*bEnd0)
        bEndLoc.normalize()

        extraQ = qt.QQuaternion.rotationTo(qt.QVector3D(0, 1, 0), bEndLoc)
        extraRot = qt.QMatrix4x4()
        extraRot.rotate(extraQ)

        orientation = transMatrix * extraRot
        self.customGlWidget.mModLocal = orientation.inverted()[0]
        self.customGlWidget.coordOffset = np.float32([0, 0, 0])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.tmpOffset = np.float32(bStart0)
        self.customGlWidget.update_coord_grid()
        self.customGlWidget.glDraw()

    def updateCutoffFromQLE(self, editor):
        try:
            value = float(re.sub(',', '.', str(editor.text())))
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.cutoffI = np.float32(value)
            self.customGlWidget.updateCutOffI(np.float32(value))
            newExtents = (extents[0], extents[1],
                          self.customGlWidget.cutoffI, extents[3])
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.glDraw()
        except:  # analysis:ignore
            pass

    def updateExplosionDepth(self, editor):
        try:
            # editor = self.sender()
            value = float(re.sub(',', '.', str(editor.text())))
            self.customGlWidget.depthScaler = np.float32(value)
            if self.customGlWidget.virtScreen is not None:
                self.customGlWidget.populateVScreen()
                self.customGlWidget.glDraw()
        except:  # analysis:ignore
            pass

    def updateOpacity(self, slider, iax, editor, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    pass
        editor.setText("{0:.2f}".format(position))
        if iax == 0:
            self.customGlWidget.lineOpacity = np.float32(position)
        elif iax == 1:
            self.customGlWidget.lineWidth = np.float32(position)
        elif iax == 2:
            self.customGlWidget.pointOpacity = np.float32(position)
        elif iax == 3:
            self.customGlWidget.pointSize = np.float32(position)
        self.customGlWidget.glDraw()

    def updateOpacityFromQLE(self, editor, slider):
        # editor = self.sender()
        value = float(str(editor.text()))
        slider.setValue(value)
        self.customGlWidget.glDraw()

    def updateOpacityFromGL(self, ops):
        for iaxis, (slider, editor, op) in\
                enumerate(zip(self.opacitySliders, self.opacityEditors, ops)):
            slider.setValue(op)
            editor.setText("{0:.2f}".format(op))

    def updateTileFromQLE(self, editor, ia):
        # editor = self.sender()
        value = float(str(editor.text()))
        if self.customGlWidget.tiles[ia] == np.int32(value):
            return
        self.customGlWidget.tiles[ia] = np.int32(value)
        for oeid, oeLine in self.customGlWidget.beamline.oesDict.items():
            if is_oe(oeLine[0]) and\
                    oeid not in self.customGlWidget.needMeshUpdate:
                self.customGlWidget.needMeshUpdate.append(oeid)
        self.customGlWidget.glDraw()

    def updateProjectionOpacity(self, slider, iax, editor, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    pass
        editor.setText("{0:.2f}".format(position))
        if iax == 0:
            self.customGlWidget.lineProjectionOpacity = np.float32(position)
        elif iax == 1:
            self.customGlWidget.lineProjectionWidth = np.float32(position)
        elif iax == 2:
            self.customGlWidget.pointProjectionOpacity = np.float32(position)
        elif iax == 3:
            self.customGlWidget.pointProjectionSize = np.float32(position)
        self.customGlWidget.glDraw()

    def updateProjectionOpacityFromQLE(self, editor, slider):
        # editor = self.sender()
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)
        self.customGlWidget.glDraw()

    def updateProjectionOpacityFromGL(self, ops):
        for iaxis, (slider, editor, op) in\
                enumerate(zip(self.projectionOpacitySliders,
                              self.projectionOpacityEditors, ops)):
            slider.setValue(op)
            editor.setText("{0:.2f}".format(op))

# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "16 Nov 2025"

import qtpy
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtOpenGL import *

#from qtpy import QtCore, QtGui, QtWidgets
#from qtpy.QtCore import (
#    QObject, QProcess, QSize, QSortFilterProxyModel, QThread, QTimer, QUrl,
#    Qt, Signal)
#from qtpy.QtGui import (
#    QBrush, QColor, QCursor, QDoubleValidator, QFont, QIcon, QImage,
#    QIntValidator, QKeySequence, QMatrix4x4, QPainter, QPixmap, QQuaternion,
#    QVector2D, QVector3D, QVector4D)
#from qtpy.QtWidgets import (
#    QAbstractItemView, QAction, QApplication, QCheckBox, QComboBox, QDialog,
#    QDialogButtonBox, QDockWidget, QFileDialog, QFrame, QGroupBox,
#    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
#    QOpenGLWidget, QProgressBar, QPushButton, QSizePolicy, QSlider,
#    QShortcut, QSplitter, QStackedWidget, QStyle, QStyledItemDelegate, QTabBar,
#    QTabWidget, QTextEdit, QToolBar, QToolButton, QToolTip, QTreeView,
#    QVBoxLayout, QWidget)
from ctypes import c_int, sizeof
from functools import partial
from math import isfinite

from qtpy.QtSql import (QSqlDatabase, QSqlQuery, QSqlTableModel,
                        QSqlQueryModel)


RAW_VALUE_ROLE = Qt.UserRole + 1
EDITOR_HINT_ROLE = Qt.UserRole + 2


def _qt_attr(name, *modules):
    for module in modules:
        if hasattr(module, name):
            return getattr(module, name)
    moduleNames = ", ".join(module.__name__ for module in modules)
    raise ImportError(f"cannot import name {name!r} from {moduleNames}")


#QOpenGLBuffer = _qt_attr("QOpenGLBuffer", QtGui)
#QOpenGLShader = _qt_attr("QOpenGLShader", QtGui)
#QOpenGLShaderProgram = _qt_attr("QOpenGLShaderProgram", QtGui)
#QOpenGLTexture = _qt_attr("QOpenGLTexture", QtGui)
#QOpenGLVertexArrayObject = _qt_attr("QOpenGLVertexArrayObject", QtGui)
#QStandardItem = _qt_attr("QStandardItem", QtGui, QtCore)
#QStandardItemModel = _qt_attr("QStandardItemModel", QtGui, QtCore)

QtName = qtpy.API_NAME

if QtName == "PyQt4":
    import PyQt4.QtWebKit as QtWeb
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg as FigCanvas,
        NavigationToolbar2QT as NavigationToolbar)
else:
    try:
        from qtpy import QtWebEngineWidgets as QtWeb
    except ImportError:
        from qtpy import QtWebKitWidgets as QtWeb

    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigCanvas,
        NavigationToolbar2QT as NavigationToolbar)

QT_VERSION_STR = qtpy.API_NAME
PYQT_VERSION_STR = qtpy.QT_VERSION


class mySlider(QSlider):
    _INT_BITS = sizeof(c_int) * 8
    _INT_MAX = (1 << (_INT_BITS - 1)) - 1
    _INT_MIN = -(1 << (_INT_BITS - 1))

    def __init__(self, parent, scaleDirection, scalePosition):
        super(mySlider, self).__init__(scaleDirection)
        self.setTickPosition(scalePosition)
        self.scale = 1.

    def setRange(self, start, end, step):
        try:
            start = float(start)
            end = float(end)
            step = float(step)
        except (TypeError, ValueError):
            return
        if not all(isfinite(v) for v in (start, end, step)) or step == 0:
            return
        maxAbs = max(abs(start), abs(end))
        if maxAbs > 0:
            stepSign = -1. if step < 0 else 1.
            step = stepSign * max(abs(step), maxAbs/self._INT_MAX)
        self.scale = 1. / step
        # QSlider.setRange(self, int(start/step), int(end/step))
        super(mySlider, self).setRange(int(start/step), int(end/step))

    def setValue(self, value):
        # QSlider.setValue(self, int(value*self.scale))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not isfinite(value):
            return
        value = int(value*self.scale)
        value = max(self._INT_MIN, min(self._INT_MAX, value))
        super(mySlider, self).setValue(value)


glowSlider = mySlider
glowTopScale = QSlider.TicksAbove


class DictEditorDialog(QDialog):
    def __init__(self, value=None, hint=None, bl=None, excludeRefs=None,
                 parent=None):
        super().__init__(parent)
        self.hint = {} if hint is None else dict(hint)
        self.valueHint = dict(self.hint.get('valueHint') or {})
        self.bl = bl
        self.excludeRefs = set(
            str(ref) for ref in (excludeRefs or []) if ref is not None)
        self.resultValue = None

        self.setWindowTitle(self.hint.get('title', 'Edit values'))
        self.resize(520, 360)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels([
            self.hint.get('keyHeader', 'Key'),
            self.hint.get('valueHeader', 'Value')])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.table)

        buttonsLayout = QHBoxLayout()
        addButton = QPushButton('Add row', self)
        addButton.clicked.connect(self.add_empty_row)
        buttonsLayout.addWidget(addButton)
        buttonsLayout.addStretch()
        layout.addLayout(buttonsLayout)

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        for key, itemValue in self._parse_value(value).items():
            self.add_row(key, itemValue)
        if self.table.rowCount() == 0:
            self.add_empty_row()

    def _parse_value(self, value):
        from ...backends import raycing
        return raycing.parse_editor_mapping(value)

    def add_empty_row(self):
        row = self.table.rowCount()
        self.add_row(row, None)

    def add_row(self, key, value):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(key)))
        self._set_value_cell(row, value)

    def _set_value_cell(self, row, value):
        if self.valueHint.get('editor') == 'reference':
            combo = QComboBox(self.table)
            combo.setEditable(False)
            combo.addItems(self._reference_items())
            text = self._display_value(value)
            idx = combo.findText(text)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setEditText(text)
            self.table.setCellWidget(row, 1, combo)
        else:
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def _reference_items(self):
        refKind = self.valueHint.get('refKind')
        if refKind != 'material' or self.bl is None:
            return ['None']
        names = []
        for name, uuid in sorted(getattr(
                self.bl, 'matnamesToUUIDs', {}).items()):
            if str(name) in self.excludeRefs or str(uuid) in self.excludeRefs:
                continue
            names.append(name)
        return ['None'] + names

    def _display_value(self, value):
        from ...backends import raycing
        return raycing.format_editor_scalar(
            value, self.valueHint, self.bl, quote_strings=False)

    def _parse_key(self, text):
        keyType = str(self.hint.get('keyType', 'str')).lower()
        if keyType == 'int':
            return int(str(text).strip())
        if keyType == 'float':
            return float(str(text).strip())
        return str(text)

    def _cell_value(self, row):
        widget = self.table.cellWidget(row, 1)
        if isinstance(widget, QComboBox):
            text = str(widget.currentText()).strip()
        else:
            item = self.table.item(row, 1)
            text = '' if item is None else str(item.text()).strip()
        if text in ['', 'None']:
            return None
        return text

    def value(self):
        out = {}
        seen = set()
        for row in range(self.table.rowCount()):
            keyItem = self.table.item(row, 0)
            if keyItem is None or not str(keyItem.text()).strip():
                continue
            key = self._parse_key(keyItem.text())
            if key in seen:
                raise ValueError('Duplicate key {0!r}'.format(key))
            seen.add(key)
            out[key] = self._cell_value(row)
        return out

    def serialized_value(self):
        if self.resultValue is None:
            self.resultValue = self.value()
        from ...backends import raycing
        return raycing.serialize_editor_value(
            self.resultValue, self.hint, self.bl)

    def show_context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        menu = QMenu(self.table)
        removeAction = QAction('Remove row', self)
        menu.addAction(removeAction)
        removeAction.triggered.connect(
            lambda: self.table.removeRow(index.row()))
        menu.exec_(self.table.viewport().mapToGlobal(position))

    def accept(self):
        try:
            self.resultValue = self.value()
        except Exception as error:
            QMessageBox.warning(self, 'Invalid values', str(error))
            return
        super().accept()


class DynamicArgumentDelegate(QStyledItemDelegate):
    def __init__(self, nameToModel=None, parent=None, mainWidget=None,
                 bl=None):
        super().__init__(parent)
        self.nameToModel = nameToModel
        self.mainWidget = mainWidget
        self.bl = bl

    def argumentEditorHint(self, index, argName):
        model = index.model()
        if hasattr(model, 'itemFromIndex'):
            valueItem = model.itemFromIndex(index)
            if valueItem is not None:
                hint = valueItem.data(EDITOR_HINT_ROLE)
                if isinstance(hint, dict):
                    return hint
            keyItem = model.itemFromIndex(model.index(
                index.row(), 0, index.parent()))
            if keyItem is not None:
                hint = keyItem.data(EDITOR_HINT_ROLE)
                if isinstance(hint, dict):
                    return hint
        if self.mainWidget is not None and\
                hasattr(self.mainWidget, 'getArgumentEditorHint'):
            hint = self.mainWidget.getArgumentEditorHint(argName, index)
            if isinstance(hint, dict):
                return hint
        return None

    def _indexRawValue(self, index):
        model = index.model()
        if hasattr(model, 'itemFromIndex'):
            item = model.itemFromIndex(index)
            if item is not None:
                rawValue = item.data(RAW_VALUE_ROLE)
                if rawValue is not None:
                    return rawValue
        return index.data()

    def _beamLine(self):
        if self.bl is not None:
            return self.bl
        if self.mainWidget is not None:
            bl = getattr(self.mainWidget, 'beamLine', None)
            if bl is not None:
                return bl
            customGlWidget = getattr(self.mainWidget, 'customGlWidget', None)
            bl = getattr(customGlWidget, 'beamline', None)
            if bl is not None:
                return bl
        return None

    def _excludedReferenceValues(self, index, hint):
        valueHint = (hint or {}).get('valueHint') or {}
        if valueHint.get('refKind') != 'material':
            return set()

        editorObject = getattr(self.mainWidget, 'editorObject', None)
        uuid = getattr(editorObject, 'uuid', None)
        if uuid is not None:
            return {uuid}

        bl = self._beamLine()
        if bl is None or not hasattr(index.model(), 'itemFromIndex'):
            return set()

        item = index.model().itemFromIndex(index)
        while item is not None and item.parent() is not None:
            item = item.parent()
        if item is None:
            return set()

        uuid = item.data(Qt.UserRole)
        if str(uuid) in getattr(bl, 'materialsDict', {}):
            return {uuid}
        return set()

    def createEditor(self, parent, option, index):
        # TODO: split into oe/mat/plot
        model = index.model()
        row = index.row()
        nameIndex = model.index(row, 0, index.parent())
        argName = str(nameIndex.data())
        argNameL = argName.lower()
        argValue = str(index.data())
        parentIndex = index.parent()
        if parentIndex is not None:
            parentIndexName = parentIndex.data()

        if parentIndexName is None:
            parentIndexName = str(model.invisibleRootItem().data(Qt.UserRole))

        parentIndexName = str(parentIndexName)

        # beamModel - only in propagation and plots
        # fluxLabels - only in plots
        # units - only in plots

        editorHint = self.argumentEditorHint(index, argName)
        if editorHint is not None and editorHint.get('editor') == 'dict':
            btn = QPushButton('Edit...', parent)
            btn.clicked.connect(partial(self.openDictDialog, index,
                                        editorHint))
            return btn

        combo = QComboBox(parent)
        combo.activated.connect(lambda: self.commitData.emit(combo))
#        if isinstance(self.mainWidget.getVal(argValue), bool):
        if str(argValue).lower() in ['false', 'true']:
            combo.addItems(['False', 'True'])
            return combo
        elif argNameL == 'generator':
            combo.addItems(['None'])
            if hasattr(self.mainWidget, 'availableGlowScanGenerators'):
                for generatorName in \
                        self.mainWidget.availableGlowScanGenerators():
                    if combo.findText(generatorName) < 0:
                        combo.addItem(generatorName)
            if argValue and combo.findText(argValue) < 0:
                combo.addItem(argValue)
            return combo
#        elif argName in ['bl', 'beamline']:
#            combo.setEditable(True)
#            combo.setModel(self.mainWidget.beamLineModel)
#            return combo
        elif argName.startswith('beam'):
            if hasattr(self.mainWidget, 'beamModel'):
                if parentIndexName == 'parameters':
                    fpModel = MultiColumnFilterProxy({1: "Global"}, combo)
                    hiddenBeams = self._collect_sibling_output_beams(
                        model, index)
                    if hiddenBeams:
                        beamProxy = ComboFilterText(hiddenBeams, combo)
                        beamProxy.setSourceModel(self.mainWidget.beamModel)
                        fpModel.setSourceModel(beamProxy)
                    else:
                        fpModel.setSourceModel(self.mainWidget.beamModel)
                else:

                    fpModel = self.mainWidget.beamModel
                combo.setModel(fpModel)
            elif hasattr(self.mainWidget, 'beamDict'):
                itemsList = list(self.mainWidget.beamDict.keys())
                if 'beamAbsorb' in itemsList:
                    itemsList.remove('beamAbsorb')
                combo.addItems(itemsList)
            else:
                return QLineEdit(parent)
            return combo
        elif argName.startswith('wave'):
            fpModel = MultiColumnFilterProxy({1: "Local"}, combo)
            fpModel.setSourceModel(self.mainWidget.beamModel)
            combo.setModel(fpModel)
            return combo
#        elif argName.startswith('plots'):
#            combo.setModel(self.mainWidget.plotModel)
#            combo.setEditable(True)
#            combo.setInsertPolicy(QComboBox.InsertAtCurrent)
#            return combo
        elif any(argNameL.startswith(v) for v in
                 ['mater', 'tlay', 'blay', 'coat', 'substrate']):  # mat and bl
            currentElement = None
            for i in range(model.rowCount(parentIndex)):
                fieldName = str(model.index(i, 0, parentIndex).data())
                fieldVal = str(model.index(i, 1, parentIndex).data())
                if fieldName == 'name':
                    currentElement = fieldVal
                    break
            if self.bl is not None:
                allElements = list(self.bl.matnamesToUUIDs.keys())
                if currentElement in allElements:
                    allElements.remove(currentElement)
                combo.addItems(['None'] + allElements)
            elif self.mainWidget is not None:
                proxy = ComboFilterText({currentElement, }, combo)
                proxy.setSourceModel(self.mainWidget.materialsModel)
                combo.setModel(proxy)
            else:
                return QLineEdit(parent)
            return combo
        elif any(argNameL.startswith(v) for v in
                 ['figureerr', 'basefe']):  # mat and bl
            currentElement = None
            for i in range(model.rowCount(parentIndex)):
                fieldName = str(model.index(i, 0, parentIndex).data())
                fieldVal = str(model.index(i, 1, parentIndex).data())
                if fieldName == 'name':
                    currentElement = fieldVal
                    break
            if self.bl is not None:
                allElements = list(self.bl.fenamesToUUIDs.keys())
                if currentElement in allElements:
                    allElements.remove(currentElement)
                combo.addItems(['None'] + allElements)
            elif self.mainWidget is not None:
                proxy = ComboFilterText({currentElement, }, combo)
                proxy.setSourceModel(self.mainWidget.fesModel)
                combo.setModel(proxy)
            else:
                return QLineEdit(parent)
            return combo
        elif argNameL == 'kind':  # material and bl
            matKindItems = ['mirror', 'thin mirror',
                            'plate', 'lens', 'grating', 'FZP', 'auto']
#            if str(model.index(0, 0).data()).lower() == 'none':  # material
            if argValue in matKindItems:
                combo.addItems(matKindItems)
            else:  # aperture
                group = QFrame(parent)
                group.setAutoFillBackground(True)
                layout = QHBoxLayout()
                layout.setContentsMargins(2, 2, 2, 2)
                layout.setSpacing(4)
                group.cb = []
                for name in ['left', 'right', 'bottom', 'top']:
                    cb = QCheckBox(name, group)
                    layout.addWidget(cb)
                    group.cb.append(cb)

                layout.addStretch()
                group.setLayout(layout)
                group.setProperty('fieldName', 'kind')
                return group
            return combo
        elif argNameL == 'rmskind':
            combo.addItems(['height', 'slope'])
            return combo
        elif argNameL == 'surfacehint':
            combo.addItems(['flat', 'quad', 'spline'])
            return combo
        elif 'density' in argName:  # uniformRayDensity would fall under bool
            combo.addItems(['histogram', 'kde'])
            return combo
        elif 'polarization' in argName:  # bl only
            combo.addItems(['horizontal', 'vertical',
                            '+45', '-45', 'left', 'right', 'None'])
            return combo
        elif 'diste' in argNameL:  # source only
            combo.addItems(['eV', 'BW'])
            return combo
        elif argNameL == 'shape':  # bl only
            combo.addItems(['rect', 'round'])
            return combo
        elif 'renderstyle' in argNameL:  # bl only
            combo.addItems(['mask', 'blades'])
            return combo
        elif 'table' in argNameL:  # material only
            combo.addItems(['Chantler', 'Chantler total', 'Henke', 'BrCo'])
            return combo
        elif 'data' in argNameL and 'axis' in parentIndexName:  # plot
            combo.addItems(self.mainWidget.fluxDataList)
            return combo
        elif 'geom' in argNameL:  # mat only
            combo.addItems(['Bragg reflected', 'Bragg transmitted',
                            'Laue reflected', 'Laue transmitted',
                            'Fresnel'])
            return combo
        elif 'fluxkind' in argNameL:  # plot only
            combo.addItems(['total', 'power', 's', 'p',
                            '+45', '-45', 'left', 'right'])
            return combo
        elif 'aspect' in argNameL:  # plot only
            combo.addItems(['equal', 'auto'])
            return combo
        elif argNameL.endswith('pos'):  # plot only
            combo.addItems(['0', '1'])
            if argName.startswith('e'):
                combo.addItems(['2'])
            return combo
        elif 'precisionopencl' in argNameL:  # bl only
            combo.addItems(['auto', 'float32', 'float64'])
            return combo
        elif 'targetopencl' in argNameL:
            if hasattr(self.mainWidget, 'openClDevList'):
                combo.addItems(self.mainWidget.openClDevList)
                combo.setEditable(True)
                return combo
            else:
                return QLineEdit(parent)
        elif argNameL.endswith('label'):  # plot only
            if parentIndexName.lower() in ['xaxis', 'yaxis']:
                combo.addItems(['x', 'y', 'z', 'x\'', 'z\'', 'energy'])
            elif hasattr(self.mainWidget, 'fluxLabelList'):  # caxis
                combo.addItems(self.mainWidget.fluxLabelList)
            else:
                return QLineEdit(parent)
            return combo
        elif 'rayflag' in argNameL:  # plot only
            group = QWidget(parent)
            group.setAutoFillBackground(True)
            layout = QHBoxLayout()
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(4)
            group.cb = []
            for ix, name in enumerate(['lost', 'good', 'out', 'over']):
                cb = QCheckBox(name, group)
                layout.addWidget(cb)
                group.cb.append((cb, ix))

            layout.addStretch()
            group.setLayout(layout)
            group.setProperty('fieldName', 'rayflag')
            return group
        elif argNameL.endswith('unit'):
            if parentIndexName.lower() in ['xaxis', 'yaxis', 'caxis']:
                for i in range(model.rowCount(parentIndex)):
                    fieldName = str(model.index(i, 0, parentIndex).data())
                    fieldVal = str(model.index(i, 1, parentIndex).data())
                    if fieldName == 'label':
                        if fieldVal in ['x', 'y', 'z']:
                            combo.addItems(self.mainWidget.lengthUnitList)
#                            combo.setModel(self.mainWidget.lengthUnitModel)
                        elif fieldVal in ['x\'', 'z\'']:
                            combo.addItems(self.mainWidget.angleUnitList)
#                            combo.setModel(self.mainWidget.angleUnitModel)
                        elif fieldVal in ['energy']:
                            combo.addItems(self.mainWidget.energyUnitList)
#                            combo.setModel(self.mainWidget.energyUnitModel)
                        else:
                            return QLineEdit(parent)
                        break
                return combo
            else:
                combo.addItems(self.mainWidget.angleUnitList)
                return combo
        elif argNameL in ['filename', 'customField']:
            fExts = ["STL"]
            if parentIndex is not None:
                prtItem = model.itemFromIndex(parentIndex)

            if prtItem is None:
                prtItem = model.invisibleRootItem()

            for i in range(prtItem.rowCount()):  # query siblings
                fieldName = str(prtItem.child(i, 0).text())
                if fieldName.lower() == 'distributions':
                    fExts = ["NPY", "NPZ"]
                    break
                elif fieldName.lower() == 'basefe':
                    fExts = ["All"]
                    break
                elif fieldName.lower() == 'materialsindex':
                    fExts = ["H5", "HDF5", "All"]
                    break
            btn = QPushButton("Open file...", parent)
            btn.clicked.connect(partial(self.openDialog, index, fExts))
            return btn
        elif "from source" in argNameL:
            elList = ['None']
            if self.bl is not None:
                for key, val in self.bl.oesDict.items():
                    oeObj = val[0]
                    if hasattr(oeObj, 'nrays'):  # Source
                        elList.append(oeObj.name)
            combo.addItems(elList)
            return combo

        elif "from oe" in argNameL:
            elList = ['None']
            if self.bl is not None:
                for key, val in self.bl.oesDict.items():
                    oeObj = val[0]
                    if hasattr(oeObj, 'material'):  # OE
                        elList.append(oeObj.name+': local beam')
                        elList.append(oeObj.name+': pitch')
            combo.addItems(elList)
            combo.setMaxVisibleItems(30)
            return combo
        else:
            return QLineEdit(parent)

    def _collect_sibling_output_beams(self, model, index):
        hiddenBeams = set()
        parentIndex = index.parent()
        if not parentIndex.isValid():
            return hiddenBeams

        methodIndex = parentIndex.parent()
        if not methodIndex.isValid():
            return hiddenBeams

        methodItem = model.itemFromIndex(methodIndex)
        if methodItem is None:
            return hiddenBeams

        for i in range(methodItem.rowCount()):
            sectionItem = methodItem.child(i, 0)
            if sectionItem is None or str(sectionItem.text()) != 'output':
                continue
            for j in range(sectionItem.rowCount()):
                beamItem = sectionItem.child(j, 1)
                if beamItem is not None:
                    hiddenBeams.add(str(beamItem.text()))
            break

        return hiddenBeams

    def setEditorData(self, editor, index):
        value = index.data()
        if isinstance(editor, QComboBox):
            idx = editor.findText(value)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            elif editor.isEditable():
                editor.setEditText(value)
        elif isinstance(editor, QLineEdit):
            editor.setText(value)
        elif isinstance(editor, QPushButton):
            editor.setText('Edit...')
#        elif isinstance(editor, QWidget):  # TODO: need better condition
        elif editor.property('fieldName') == 'kind':
            for cb in editor.cb:
                cb.setChecked(str(cb.text()) in value)
        elif editor.property('fieldName') == 'rayflag':
            for cb in editor.cb:
                cb[0].setChecked(str(cb[1]) in value)

    def _setModelValue(self, model, index, value):
        if hasattr(model, 'itemFromIndex'):
            item = model.itemFromIndex(index)
            if item is not None:
                signalsBlocked = model.signalsBlocked()
                model.blockSignals(True)
                try:
                    item.setData(value, RAW_VALUE_ROLE)
                finally:
                    model.blockSignals(signalsBlocked)
        model.setData(index, value)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            self._setModelValue(model, index, editor.currentText())
        elif isinstance(editor, QLineEdit):
            self._setModelValue(model, index, editor.text())
        elif isinstance(editor, QPushButton):
            pass
        elif editor.property('fieldName') == 'kind':
            text = "["
            for cb in editor.cb:
                if cb.isChecked():
                    text += "'{}',".format(cb.text())
            text = text.strip(",")
            text += "]"
            self._setModelValue(model, index, text)
        elif editor.property('fieldName') == 'rayflag':
            text = "("
            for cb in editor.cb:
                if cb[0].isChecked():
                    text += "{},".format(cb[1])
#            text = text.strip(",")
            text += ")"
            self._setModelValue(model, index, text)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def openDialog(self, index, fileFormats):
        openDialog = QFileDialog()
        openDialog.setFileMode(QFileDialog.ExistingFile)
        openDialog.setAcceptMode(QFileDialog.AcceptOpen)
        formats = [fmt for fmt in fileFormats if fmt.lower() != 'all']
        filters = []
        if formats:
            exts = " ".join(formats)
            mask = " ".join(f"*.{fmt.lower()}" for fmt in formats)
            filters.append(f"{exts} files ({mask})")
        if any(fmt.lower() == 'all' for fmt in fileFormats):
            filters.append("All files (*)")
        if not filters:
            filters.append("All files (*)")
        openDialog.setNameFilters(filters)
        if (openDialog.exec_()):
            openFileName = openDialog.selectedFiles()[0]
            if openFileName:
                self._setModelValue(index.model(), index, openFileName)

    def openDictDialog(self, index, hint):
        dialog = DictEditorDialog(
            self._indexRawValue(index), hint, bl=self._beamLine(),
            excludeRefs=self._excludedReferenceValues(index, hint),
            parent=self.parent())
        if dialog.exec_():
            valueText = dialog.serialized_value()
            self._setModelValue(index.model(), index, valueText)


class MultiColumnFilterProxy(QSortFilterProxyModel):
    """Fields must be a dictionary {column: "filterValue"}"""
    def __init__(self, fields=None, parent=None):
        super().__init__(parent)
        self.fields = {} if fields is None else dict(fields)

    def setColumnFilter(self, fields):
        self.fields.update(fields)
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        output = []
        for key, value in self.fields.items():
            tabIndex = model.index(row, key, parent)
            output.append(value is None or str(value) in
                          str(model.data(tabIndex)))

        return all(output)


class ComboBoxFilterProxyModel(QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        # Skip the top element (row 0)
        if source_row == 0:
            return False
        return True


class ComboFilterText(QSortFilterProxyModel):
    def __init__(self, hiddenText, parent=None):
        super().__init__(parent)
        self.hiddenText = set(hiddenText)

    def filterAcceptsRow(self, row, parent):
        index = self.sourceModel().index(row, 0, parent)
        text = self.sourceModel().data(index)
        return text not in self.hiddenText


class StateButtons(QFrame):
    statesActive = Signal(list)

    def __init__(self, parent, names, active=None):
        """
        *names*: a list of any objects that will be displayed as str(object),

        *active*: a subset of names that will be displayed as checked,

        The signal *statesActive* is emitted on pressing a button. It sends a
        list of selected names, as a subset of *names*.
        """

        super().__init__(parent)
        self.names = names
        self.buttons = []
        layout = QHBoxLayout()
        styleSheet = """
        QPushButton {
            border-style: outset;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightsalmon;}
        QPushButton:checked {
            border-style: inset;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightgreen;}
        QPushButton:hover {
            border-style: solid;
            border-width: 2px;
            border-radius: 5px;
            border-color: lightblue;}
        """
        for name in names:
            strName = str(name)
            but = QPushButton(strName)
            but.setCheckable(True)

            bbox = but.fontMetrics().boundingRect(strName)
            but.setFixedSize(bbox.width()+12, bbox.height()+4)
            # but.setToolTip("go to the key frame")
            but.clicked.connect(self.buttonClicked)
            but.setStyleSheet(styleSheet)

            self.buttons.append(but)
            layout.addWidget(but)
        layout.addStretch()
        self.setLayout(layout)

        self.setActive(active)

    def getActive(self):
        return [name for (button, name) in
                zip(self.buttons, self.names) if button.isChecked()]

    def setActive(self, active):
        if not isinstance(active, (list, tuple)):
            return
        for button, name in zip(self.buttons, self.names):
            button.setChecked(name in active)

    def buttonClicked(self, checked):
        self.statesActive.emit(self.getActive())


def print_model(model, label="Model"):
    print(f"--- {label} ---")
    rows = model.rowCount()
    cols = model.columnCount()
    for row in range(rows):
        row_data = []
        for col in range(cols):
            index = model.index(row, col)
            data = model.data(index)
            row_data.append(str(data))
        print(f"Row {row}: {row_data}")

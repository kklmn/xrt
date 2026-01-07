# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "16 Nov 2025"

import qtpy
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtOpenGL import *
from functools import partial

from qtpy.QtSql import (QSqlDatabase, QSqlQuery, QSqlTableModel,
                        QSqlQueryModel)

QtName = qtpy.API_NAME

if QtName == "PyQt4":
    import PyQt4.QtWebKit as QtWeb
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg as FigCanvas,
        NavigationToolbar2QT as NavigationToolbar)
else:
    # import PyQt5.QtCore
    # print(list(vars(PyQt5.QtCore.Qt)))
    # locals().update(vars(PyQt5.QtCore.Qt))
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
    def __init__(self, parent, scaleDirection, scalePosition):
        super(mySlider, self).__init__(scaleDirection)
        self.setTickPosition(scalePosition)
        self.scale = 1.

    def setRange(self, start, end, step):
        if step == 0:
            return
        self.scale = 1. / step
        # QSlider.setRange(self, int(start/step), int(end/step))
        super(mySlider, self).setRange(int(start/step), int(end/step))

    def setValue(self, value):
        # QSlider.setValue(self, int(value*self.scale))
        super(mySlider, self).setValue(int(value*self.scale))


glowSlider = mySlider
glowTopScale = QSlider.TicksAbove


#class QComboBox(StdQComboBox):
#    """
#    Disabling off-focus mouse wheel scroll is based on the following solution:
#    https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox/3242107#3242107
#    """
#    def __init__(self, *args, **kwargs):
#        super(QComboBox, self).__init__(*args, **kwargs)
##        self.scrollWidget=scrollWidget
#        self.setFocusPolicy(QtCore.Qt.StrongFocus)
#
#    def wheelEvent(self, *args, **kwargs):
#        if self.hasFocus():
#            return StdQComboBox.wheelEvent(self, *args, **kwargs)
#        else:
#            try:
#                return self.parent().wheelEvent(*args, **kwargs)
#            except RuntimeError:
#                return


class DynamicArgumentDelegate(QStyledItemDelegate):
    def __init__(self, nameToModel=None, parent=None, mainWidget=None,
                 bl=None):
        super().__init__(parent)
        self.nameToModel = nameToModel
        self.mainWidget = mainWidget
        self.bl = bl

    def createEditor(self, parent, option, index):
        # TODO: split into oe/mat/plot
        model = index.model()
        row = index.row()
        nameIndex = model.index(row, 0, index.parent())
        argName = str(nameIndex.data())
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

        combo = QComboBox(parent)
        combo.activated.connect(lambda: self.commitData.emit(combo))
#        if isinstance(self.mainWidget.getVal(argValue), bool):
        if str(argValue).lower() in ['false', 'true']:
            combo.addItems(['False', 'True'])
            return combo
#        elif argName in ['bl', 'beamline']:
#            combo.setEditable(True)
#            combo.setModel(self.mainWidget.beamLineModel)
#            return combo
        elif argName.startswith('beam'):
            if hasattr(self.mainWidget, 'beamModel'):
                if parentIndexName == 'parameters':
                    fpModel = MultiColumnFilterProxy({1: "Global"})
                    fpModel.setSourceModel(self.mainWidget.beamModel)
                else:

                    fpModel = self.mainWidget.beamModel
                combo.setModel(fpModel)
            elif hasattr(self.mainWidget, 'beamDict'):
                combo.addItems(list(self.mainWidget.beamDict.keys()))
            else:
                return QLineEdit(parent)
            return combo
        elif argName.startswith('wave'):
            fpModel = MultiColumnFilterProxy({1: "Local"})
            fpModel.setSourceModel(self.mainWidget.beamModel)
            combo.setModel(fpModel)
            return combo
#        elif argName.startswith('plots'):
#            combo.setModel(self.mainWidget.plotModel)
#            combo.setEditable(True)
#            combo.setInsertPolicy(QComboBox.InsertAtCurrent)
#            return combo
        elif any(argName.lower().startswith(v) for v in
                 ['mater', 'tlay', 'blay', 'coat', 'substrate']):  # mat and bl
            if self.bl is not None:
                combo.addItems(['None']+list(
                        self.bl.matnamesToUUIDs.keys()))
            elif self.mainWidget is not None:
                combo.setModel(self.mainWidget.materialsModel)
            else:
                return QLineEdit(parent)
            return combo
        elif any(argName.lower().startswith(v) for v in
                 ['figureerr', 'basefe']):  # mat and bl
            if self.bl is not None:
                combo.addItems(['None']+list(
                        self.bl.fenamesToUUIDs.keys()))
            elif self.mainWidget is not None:
                combo.setModel(self.mainWidget.fesModel)
            else:
                return QLineEdit(parent)
            return combo
        elif argName.lower() == 'kind':  # material and bl
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
        elif 'density' in argName:  # uniformRayDensity would fall under bool
            combo.addItems(['histogram', 'kde'])
            return combo
        elif 'polarization' in argName:  # bl only
            combo.addItems(['horizontal', 'vertical',
                            '+45', '-45', 'left', 'right', 'None'])
            return combo
        elif 'shape' in argName.lower():  # bl only
            combo.addItems(['rect', 'round'])
            return combo
        elif 'renderstyle' in argName.lower():  # bl only
            combo.addItems(['mask', 'blades'])
            return combo
        elif 'table' in argName.lower():  # material only
            combo.addItems(['Chantler', 'Chantler total', 'Henke', 'BrCo'])
            return combo
        elif 'data' in argName.lower() and 'axis' in parentIndexName:  # plot
            combo.addItems(self.mainWidget.fluxDataList)
            return combo
        elif 'geom' in argName.lower():  # mat only
            combo.addItems(['Bragg reflected', 'Bragg transmitted',
                            'Laue reflected', 'Laue transmitted',
                            'Fresnel'])
            return combo
        elif 'fluxkind' in argName.lower():  # plot only
            combo.addItems(['total', 'power', 's', 'p',
                            '+45', '-45', 'left', 'right'])
            return combo
        elif 'aspect' in argName.lower():  # plot only
            combo.addItems(['equal', 'auto'])
            return combo
        elif argName.lower().endswith('pos'):  # plot only
            combo.addItems(['0', '1'])
            if argName.startswith('e'):
                combo.addItems(['2'])
            return combo
        elif 'precisionopencl' in argName.lower():  # bl only
            combo.addItems(['auto', 'float32', 'float64'])
            return combo
        elif argName.lower().endswith('label'):  # plot only
            if parentIndexName.lower() in ['xaxis', 'yaxis']:
                combo.addItems(['x', 'y', 'z', 'x\'', 'z\'', 'energy'])
            elif hasattr(self.mainWidget, 'fluxLabelList'):  # caxis
                combo.addItems(self.mainWidget.fluxLabelList)
            else:
                return QLineEdit(parent)
            return combo
        elif 'rayflag' in argName.lower():  # plot only
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
        elif argName.lower().endswith('unit'):
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
        elif argName.lower() in ['filename', 'customField']:
            fExts = ["STL"]
            if parentIndex is not None:
                prtItem = model.itemFromIndex(parentIndex)
            
            if prtItem is None:
                prtItem = model.invisibleRootItem()

            for i in range(prtItem.rowCount()):
                fieldName = str(prtItem.child(i, 0).text())
                if fieldName.lower() == 'distributions':
                    fExts = ["NPY", "NPZ"]
                    break
                elif fieldName.lower() == 'basefe':
                    fExts = ["All"]
                    break
            btn = QPushButton("Open file...", parent)
            btn.clicked.connect(partial(self.openDialog, index, fExts))
            return btn
        elif "from source" in argName.lower():
            elList = ['None']
            if self.bl is not None:
                for key, val in self.bl.oesDict.items():
                    oeObj = val[0]
                    if hasattr(oeObj, 'nrays'):  # Source
                        elList.append(oeObj.name)
            combo.addItems(elList)
            return combo

        elif "from oe" in argName.lower():
            elList = ['None']
            if self.bl is not None:
                for key, val in self.bl.oesDict.items():
                    oeObj = val[0]
                    if hasattr(oeObj, 'material'):  # OE
                        elList.append(oeObj.name)
            combo.addItems(elList)
            return combo
        else:
            return QLineEdit(parent)

    def setEditorData(self, editor, index):
        value = index.data()
        if isinstance(editor, QComboBox):
            idx = editor.findText(value)
            if idx >= 0:
                editor.setCurrentIndex(idx)
        elif isinstance(editor, QLineEdit):
            editor.setText(value)
#        elif isinstance(editor, QWidget):  # TODO: need better condition
        elif editor.property('fieldName') == 'kind':
            for cb in editor.cb:
                cb.setChecked(str(cb.text()) in value)
        elif editor.property('fieldName') == 'rayflag':
            for cb in editor.cb:
                cb[0].setChecked(str(cb[1]) in value)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText())
        elif isinstance(editor, QLineEdit):
            model.setData(index, editor.text())
        elif editor.property('fieldName') == 'kind':
            text = "["
            for cb in editor.cb:
                if cb.isChecked():
                    text += "'{}',".format(cb.text())
            text = text.strip(",")
            text += "]"
            model.setData(index, text)
        elif editor.property('fieldName') == 'rayflag':
            text = "("
            for cb in editor.cb:
                if cb[0].isChecked():
                    text += "{},".format(cb[1])
#            text = text.strip(",")
            text += ")"
            model.setData(index, text)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def openDialog(self, index, fileFormats):
        openDialog = QFileDialog()
        openDialog.setFileMode(QFileDialog.ExistingFile)
        openDialog.setAcceptMode(QFileDialog.AcceptOpen)
        if 'All' in fileFormats:
            exts = 'All'
            mask = '*'
        else:
            exts = " ".join(f"{e}" for e in fileFormats)
            mask = " ".join(f"*.{e.lower()}" for e in fileFormats)
        openDialog.setNameFilter(
                f"{exts} files ({mask})")
        if (openDialog.exec_()):
            openFileName = openDialog.selectedFiles()[0]
            if openFileName:
                index.model().setData(index, openFileName)


class MultiColumnFilterProxy(QSortFilterProxyModel):
    """Fields must be a dictionary {column: "filterValue"}"""
    def __init__(self, fields={}, parent=None):
        super().__init__(parent)
        self.fields = fields

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

# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "01 Nov 2017"

#try:
#    from matplotlib.backends import qt_compat
#except ImportError:
#    from matplotlib.backends import qt4_compat
#    qt_compat = qt4_compat

QtImports = 'PyQt5', 'PyQt4', 'PySide2', 'PySide6'

for QtImport in QtImports:
    try:
        __import__(QtImport)
        QtName = QtImport
        break
    except ImportError:
        QtName = None
else:
    raise ImportError("Cannot import any PyQt package!")

starImport = False  # star import doesn't work with mock import needed for rtfd

#if 'pyqt4' in qt_compat.QT_API.lower():  # also 'PyQt4v2'
if QtName == "PyQt4":
    from PyQt4 import QtGui, QtCore
    import PyQt4.QtGui as myQtGUI

    if starImport:
        from PyQt4.QtGui import *
        from PyQt4.QtCore import *
        Signal = pyqtSignal
    else:
        from PyQt4.QtCore import (
            SIGNAL, QUrl, QObject, QTimer, QProcess,
            QThread, QT_VERSION_STR, PYQT_VERSION_STR, QSize)
        from PyQt4.QtGui import QSortFilterProxyModel
        try:
            from PyQt4.QtCore import Signal
        except ImportError:
            from PyQt4.QtCore import pyqtSignal as Signal
    import PyQt4.QtCore
    locals().update(vars(PyQt4.QtCore.Qt))

    from PyQt4.QtOpenGL import QGLWidget
    from PyQt4.QtSql import (QSqlDatabase, QSqlQuery, QSqlTableModel,
                             QSqlQueryModel)
    import PyQt4.QtWebKit as QtWeb
    try:
        import PyQt4.Qwt5 as Qwt
    except:  # analysis:ignore
        pass
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as\
        FigCanvas
#elif 'pyqt5' in qt_compat.QT_API.lower():
elif QtName == "PyQt5":
    from PyQt5 import QtGui, QtCore
    import PyQt5.QtWidgets as myQtGUI

    if starImport:
        from PyQt5.QtGui import *
        from PyQt5.QtCore import *
        from PyQt5.QtWidgets import *
        Signal = pyqtSignal
    else:
        from PyQt5.QtCore import (
            pyqtSignal, QUrl, QObject, QTimer, QProcess, QThread,
            QT_VERSION_STR, PYQT_VERSION_STR, QSortFilterProxyModel, QSize)
        try:
            from PyQt5.QtCore import Signal
        except ImportError:
            from PyQt5.QtCore import pyqtSignal as Signal
    import PyQt5.QtCore
    locals().update(vars(PyQt5.QtCore.Qt))

    from PyQt5.QtOpenGL import QGLWidget
    from PyQt5.QtSql import (QSqlDatabase, QSqlQuery, QSqlTableModel,
                             QSqlQueryModel)
    try:
        import PyQt5.QtWebEngineWidgets as QtWeb
    except ImportError:
        import PyQt5.QtWebKitWidgets as QtWeb
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as\
        FigCanvas
elif QtName == "PySide2":
    from PySide2 import QtGui, QtCore
    import PySide2.QtWidgets as myQtGUI

    if starImport:
        from PySide2.QtGui import *
        from PySide2.QtCore import *
        from PySide2.QtWidgets import *
    else:
        from PySide2.QtCore import (
            QUrl, QObject, QTimer, QProcess, QThread, QSortFilterProxyModel,
            QSize)
        try:
            from PySide2.QtCore import Signal
        except ImportError:
            from PySide2.QtCore import pyqtSignal as Signal
    import PySide2.QtCore
    QT_VERSION_STR = PySide2.QtCore.qVersion()
    PYQT_VERSION_STR = PySide2.__version__
    locals().update(vars(PySide2.QtCore.Qt))

    from PySide2.QtOpenGL import QGLWidget
    from PySide2.QtSql import (QSqlDatabase, QSqlQuery, QSqlTableModel,
                              QSqlQueryModel)
    try:
        import PySide2.QtWebEngineWidgets as QtWeb
    except ImportError:
        import PySide2.QtWebKitWidgets as QtWeb
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as\
        FigCanvas
else:
    raise ImportError("Cannot import any Python Qt package!")

if not starImport:
    (QWidget, QApplication, QAction, QTabWidget, QToolBar, QStatusBar,
     QTreeView, QShortcut, QAbstractItemView, QHBoxLayout, QVBoxLayout,
     QSplitter, StdQComboBox, QMenu, QListWidget, QTextEdit, QMessageBox,
     QFileDialog, QListWidgetItem, QGroupBox, QProgressBar, QLabel, QTableView,
     QSizePolicy, QLineEdit, QCheckBox, QSpinBox, QSlider, QToolButton,
     QPushButton, QDialog, QOpenGLWidget, QToolTip, QDialogButtonBox,
     QStyledItemDelegate, QDockWidget, QMainWindow, QStyle, QTabBar) = (
        myQtGUI.QWidget, myQtGUI.QApplication, myQtGUI.QAction,
        myQtGUI.QTabWidget, myQtGUI.QToolBar, myQtGUI.QStatusBar,
        myQtGUI.QTreeView, myQtGUI.QShortcut, myQtGUI.QAbstractItemView,
        myQtGUI.QHBoxLayout, myQtGUI.QVBoxLayout, myQtGUI.QSplitter,
        myQtGUI.QComboBox, myQtGUI.QMenu, myQtGUI.QListWidget,
        myQtGUI.QTextEdit, myQtGUI.QMessageBox, myQtGUI.QFileDialog,
        myQtGUI.QListWidgetItem, myQtGUI.QGroupBox, myQtGUI.QProgressBar,
        myQtGUI.QLabel, myQtGUI.QTableView, myQtGUI.QSizePolicy,
        myQtGUI.QLineEdit, myQtGUI.QCheckBox, myQtGUI.QSpinBox,
        myQtGUI.QSlider, myQtGUI.QToolButton, myQtGUI.QPushButton,
        myQtGUI.QDialog, myQtGUI.QOpenGLWidget, myQtGUI.QToolTip,
        myQtGUI.QDialogButtonBox, myQtGUI.QStyledItemDelegate,
        myQtGUI.QDockWidget, myQtGUI.QMainWindow, myQtGUI.QStyle,
        myQtGUI.QTabBar)
    (QIcon, QFont, QKeySequence, QStandardItemModel, QStandardItem, QPixmap,
     QDoubleValidator, QIntValidator, QDrag, QImage, QOpenGLTexture,
     QMatrix4x4, QVector4D, QOpenGLShaderProgram, QOpenGLShader, QVector3D,
     QVector2D, QMatrix3x3,
     QQuaternion, QOpenGLVertexArrayObject, QOpenGLBuffer, QBrush, QColor) = (
        QtGui.QIcon, QtGui.QFont, QtGui.QKeySequence, QtGui.QStandardItemModel,
        QtGui.QStandardItem, QtGui.QPixmap, QtGui.QDoubleValidator,
        QtGui.QIntValidator, QtGui.QDrag, QtGui.QImage, QtGui.QOpenGLTexture,
        QtGui.QMatrix4x4, QtGui.QVector4D, QtGui.QOpenGLShaderProgram,
        QtGui.QOpenGLShader, QtGui.QVector3D, QtGui.QVector2D, QtGui.QMatrix3x3,
        QtGui.QQuaternion, QtGui.QOpenGLVertexArrayObject, QtGui.QOpenGLBuffer,
        QtGui.QBrush, QtGui.QColor)


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


try:
    glowSlider = Qwt.QwtSlider
    glowTopScale = Qwt.QwtSlider.TopScale
except:  # analysis:ignore
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

QComboBox = StdQComboBox


class DynamicArgumentDelegate(QStyledItemDelegate):
    def __init__(self, nameToModel=None, parent=None, mainWidget=None):
        super().__init__(parent)
        self.nameToModel = nameToModel
        self.mainWidget = mainWidget

    def createEditor(self, parent, option, index):
        # TODO: split into oe/mat/plot
        model = index.model()
        row = index.row()
        nameIndex = model.index(row, 0, index.parent())
        argName = str(nameIndex.data())
        argValue = str(index.data())
        parentIndex = index.parent()
        parentIndexName = str(parentIndex.data())

        combo = QComboBox(parent)
        combo.activated.connect(lambda: self.commitData.emit(combo))
        if isinstance(self.mainWidget.getVal(argValue), bool):
            combo.addItems(['False', 'True'])
            return combo
        elif argName in ['bl', 'beamline']:
            combo.setEditable(True)
            combo.setModel(self.mainWidget.beamLineModel)
            return combo
        elif argName.startswith('beam'):
            if parentIndexName == 'output':  # Not sure we need it if renaming is disabled
                oeuuid = index.parent().parent().parent().data(UserRole)
                fpModel = MultiColumnFilterProxy({1: argName,
                                                  2: oeuuid})
                fpModel.setSourceModel(self.mainWidget.beamModel)
            elif parentIndexName == 'parameters':
                fpModel = MultiColumnFilterProxy({1: "Global"})
                fpModel.setSourceModel(self.mainWidget.beamModel)
            else:
                fpModel = self.mainWidget.beamModel
            combo.setModel(fpModel)
            return combo
        elif argName.startswith('wave'):
            fpModel = MultiColumnFilterProxy({1: "Local"})
            fpModel.setSourceModel(self.mainWidget.beamModel)
            combo.setModel(fpModel)
            return combo
        elif argName.startswith('plots'):
            combo.setModel(self.mainWidget.plotModel)
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertAtCurrent)
            return combo
        elif any(argName.lower().startswith(v) for v in
                 ['mater', 'tlay', 'blay', 'coat', 'substrate']):
            combo.setModel(self.mainWidget.materialsModel)
            return combo
        elif 'kind' in argName.lower() and model is self.mainWidget.materialsModel:  # mat kind
            combo.addItems(['mirror', 'thin mirror',
                            'plate', 'lens', 'grating', 'FZP', 'auto'])
            return combo
        elif 'density' in argName:  # uniformRayDensity would fall under bool
            combo.addItems(['histogram', 'kde'])
            return combo
        elif 'polarization' in argName:
            combo.addItems(['horizontal', 'vertical',
                            '+45', '-45', 'left', 'right', 'None'])
            return combo
        elif 'shape' in argName.lower():
            combo.addItems(['rect', 'round'])
            return combo
        elif 'renderstyle' in argName.lower():
            combo.addItems(['mask', 'blades'])
            return combo
        elif 'table' in argName.lower():
            combo.addItems(['Chantler', 'Chantler total', 'Henke', 'BrCo'])
            return combo
        elif 'data' in argName.lower() and 'axis' in parentIndexName:
            combo.setModel(self.mainWidget.fluxDataModel)
            return combo
        elif 'geom' in argName.lower():
            combo.addItems(['Bragg reflected', 'Bragg transmitted',
                            'Laue reflected', 'Laue transmitted',
                            'Fresnel'])
            return combo
        elif 'fluxkind' in argName.lower():
            combo.addItems(['total', 'power', 's', 'p',
                            '+45', '-45', 'left', 'right'])
            return combo
        elif 'aspect' in argName.lower():
            combo.addItems(['equal', 'auto'])
            return combo
        elif 'precisionopencl' in argName.lower():
            combo.addItems(['auto', 'float32', 'float64'])
            return combo
        elif argName.lower().endswith('label'):
            if parentIndexName.lower() in ['xaxis', 'yaxis']:
                combo.addItems(['x', 'y', 'z', 'x\'', 'z\'', 'energy'])
            else:  # caxis
                combo.setModel(self.mainWidget.fluxLabelModel)
            return combo
        elif argName.lower().endswith('unit') and parentIndexName.lower() in [
                'xaxis', 'yaxis', 'caxis']:

#    row_count = model.rowCount(parent_index)
#    siblings = [model.index(row, index.column(), parent_index) for row in range(row_count)]

            for i in range(model.rowCount(parentIndex)):
                fieldName = str(model.index(i, 0, parentIndex).data())
                fieldVal = str(model.index(i, 1, parentIndex).data())
                if fieldName == 'label':
                    if fieldVal in ['x', 'y', 'z']:
                        combo.setModel(self.mainWidget.lengthUnitModel)
                    elif fieldVal in ['x\'', 'z\'']:
                        combo.setModel(self.mainWidget.angleUnitModel)
                    elif fieldVal in ['energy']:
                        combo.setModel(self.mainWidget.energyUnitModel)
                    else:
                        return QLineEdit(parent)
                    break
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

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText())
        elif isinstance(editor, QLineEdit):
            model.setData(index, editor.text())


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

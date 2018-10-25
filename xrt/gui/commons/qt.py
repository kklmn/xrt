# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "01 Nov 2017"

#try:
#    from matplotlib.backends import qt_compat
#except ImportError:
#    from matplotlib.backends import qt4_compat
#    qt_compat = qt4_compat

QtName = None
try:
    import PyQt5
    QtName = "PyQt5"
except ImportError:
    try:
        import PyQt4
        QtName = "PyQt4"
    except ImportError:
        raise ImportError("Cannot import any PyQt package!")

starImport = False  # star import doesn't work with mock import needed for rtfd

#if 'pyqt4' in qt_compat.QT_API.lower():  # also 'PyQt4v2'
if QtName == "PyQt4":
    from PyQt4 import QtGui, QtCore
    import PyQt4.QtGui as myQtGUI

    if starImport:
        from PyQt4.QtGui import *
        from PyQt4.QtCore import *
    else:
        from PyQt4.QtCore import (
            pyqtSignal, SIGNAL, QUrl, QObject, QTimer, QProcess,
            QThread, QT_VERSION_STR, PYQT_VERSION_STR)
        from PyQt4.QtGui import QSortFilterProxyModel
        try:
            from PyQt4.QtCore import Signal
        except ImportError:
            from PyQt4.QtCore import pyqtSignal as Signal
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
            QT_VERSION_STR, PYQT_VERSION_STR, QSortFilterProxyModel)
        try:
            from PyQt5.QtCore import Signal
        except ImportError:
            from PyQt5.QtCore import pyqtSignal as Signal
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
else:
    raise ImportError("Cannot import any Python Qt package!")

if not starImport:
    (QWidget, QApplication, QAction, QTabWidget, QToolBar, QStatusBar,
     QTreeView, QShortcut, QAbstractItemView, QHBoxLayout, QVBoxLayout,
     QSplitter, StdQComboBox, QMenu, QListWidget, QTextEdit, QMessageBox,
     QFileDialog, QListWidgetItem, QGroupBox, QProgressBar, QLabel, QTableView,
     QSizePolicy, QLineEdit, QCheckBox, QSpinBox, QSlider, QToolButton) = (
        myQtGUI.QWidget, myQtGUI.QApplication, myQtGUI.QAction,
        myQtGUI.QTabWidget, myQtGUI.QToolBar, myQtGUI.QStatusBar,
        myQtGUI.QTreeView, myQtGUI.QShortcut, myQtGUI.QAbstractItemView,
        myQtGUI.QHBoxLayout, myQtGUI.QVBoxLayout, myQtGUI.QSplitter,
        myQtGUI.QComboBox, myQtGUI.QMenu, myQtGUI.QListWidget,
        myQtGUI.QTextEdit, myQtGUI.QMessageBox, myQtGUI.QFileDialog,
        myQtGUI.QListWidgetItem, myQtGUI.QGroupBox, myQtGUI.QProgressBar,
        myQtGUI.QLabel, myQtGUI.QTableView, myQtGUI.QSizePolicy,
        myQtGUI.QLineEdit, myQtGUI.QCheckBox, myQtGUI.QSpinBox,
        myQtGUI.QSlider, myQtGUI.QToolButton)
    (QIcon, QFont, QKeySequence, QStandardItemModel, QStandardItem, QPixmap,
     QDoubleValidator, QIntValidator, QDrag) = (
        QtGui.QIcon, QtGui.QFont, QtGui.QKeySequence, QtGui.QStandardItemModel,
        QtGui.QStandardItem, QtGui.QPixmap, QtGui.QDoubleValidator,
        QtGui.QIntValidator, QtGui.QDrag)


class mySlider(QSlider):
    def __init__(self, parent, scaleDirection, scalePosition):
        super(mySlider, self).__init__(scaleDirection)
        self.setTickPosition(scalePosition)
        self.scale = 1.

    def setRange(self, start, end, step):
        if step == 0:
            return
        self.scale = 1. / step
        QSlider.setRange(self, start / step, end / step)

    def setValue(self, value):
        QSlider.setValue(self, int(value*self.scale))


try:
    glowSlider = Qwt.QwtSlider
    glowTopScale = Qwt.QwtSlider.TopScale
except:  # analysis:ignore
    glowSlider = mySlider
    glowTopScale = QSlider.TicksAbove

class QComboBox(StdQComboBox):
    """
    Disabling off-focus mouse wheel scroll is based on the following solution:
    https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox/3242107#3242107
    """
    def __init__(self, *args, **kwargs):
        super(QComboBox, self).__init__(*args, **kwargs)
#        self.scrollWidget=scrollWidget
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return StdQComboBox.wheelEvent(self, *args, **kwargs)
        else:
            return self.parent().wheelEvent(*args, **kwargs)

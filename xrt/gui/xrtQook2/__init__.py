# -*- coding: utf-8 -*-
u"""
.. _qook:

xrtQook -- a GUI for creating a beamline
----------------------------------------

The main interface to xrt is through a python script. Many examples of such
scripts can be found in the supplied folder ‘examples’. The script imports the
modules of xrt, instantiates beamline parts, such as synchrotron or geometric
sources, various optical elements, apertures and screens, specifies required
materials for reflection, refraction or diffraction, defines plots and sets job
parameters.

The Qt tool :mod:`xrtQook` takes these ingredients and prepares a ready to use
script that can be run within the tool itself or in an external Python context.
:mod:`xrtQook` features a parallelly updated help panel that provides a
complete list of parameters for the used classes, also including those from the
parental classes. :mod:`xrtQook` writes/reads the recipes of beamlines
into/from xml files.

In the present version, :mod:`xrtQook` does not provide automated generation of
*scans* and does not create *wave propagation* sequences. For these two tasks,
the corresponding script parts have to be written manually based on the
supplied examples and the present documentation.

See a brief :ref:`tutorial for xrtQook <qook_tutorial>`.

.. imagezoom:: _images/xrtQook.png
   :alt: &ensp;A view of xrtQook with an empty beamline tree on the left and a
       help panel on the right.

"""

from __future__ import print_function
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "25 Jun 2017"
__version__ = "1.3"

_DEBUG_ = False  # If False, exceptions inside the module are ignored
redStr = ':red:`{0}`'

import os
import sys
import textwrap
import numpy as np  # analysis:ignore , really needed
import time
from datetime import date
import inspect
import re
import xml.etree.ElementTree as ET
from functools import partial
from collections import OrderedDict
try:
    import pyopencl as cl
    cl_platforms = cl.get_platforms()
    isOpenCL = True
    isOpenStatus = 'present'
except ImportError:
    isOpenCL = False
    isOpenStatus = redStr.format('not found')
except cl.LogicError:
    isOpenCL = False
    isOpenStatus = 'is installed '+redStr.format('but no OpenCL driver found')
import platform as pythonplatform
import webbrowser

from ..commons import ext

sys.path.append(os.path.join('..', '..', '..'))
import xrt  #analysis:ignore
from ...backends import raycing  # analysis:ignore
from ...backends.raycing import sources as rsources  # analysis:ignore
from ...backends.raycing import screens as rscreens  # analysis:ignore
from ...backends.raycing import materials as rmats  # analysis:ignore
from ...backends.raycing import oes as roes  # analysis:ignore
from ...backends.raycing import apertures as rapts  # analysis:ignore
from ...backends.raycing import oes as roes  # analysis:ignore
from ...backends.raycing import run as rrun  # analysis:ignore
from ...version import __version__ as xrtversion
from ... import plotter as xrtplot  # analysis:ignore
from ... import runner as xrtrun  # analysis:ignore
from ..commons import qt  # analysis:ignore
from ..commons import gl  # analysis:ignore
from . import tutorial
if gl.isOpenGL:
    from .. import xrtGlow as xrtglow  # analysis:ignore

#from PyQt4 import QtSql
#from PyQt4.QtGui import QDrag
#from PyQt4.QtCore import  QMimeData, QIODevice, QDataStream, QByteArray, QPoint
path_to_xrt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
myTab = 4*" "

useSlidersInTree = False
withSlidersInTree = ['pitch', 'roll', 'yaw', 'bragg']
slidersInTreeScale = {'pitch': 0.1, 'roll': 0.1, 'yaw': 0.1, 'bragg': 1e-3}

try:
    QWebView = qt.QtWeb.QWebView
except AttributeError:
    # QWebKit deprecated in Qt 5.7
    # The idea and partly the code of the compatibility fix is borrowed from
    # spyderlib.widgets.browser
    class WebPage(qt.QtWeb.QWebEnginePage):
        """
        Web page subclass to manage hyperlinks for WebEngine

        Note: This can't be used for WebKit because the
        acceptNavigationRequest method has a different
        functionality for it.
        """
        linkClicked = qt.Signal(qt.QUrl)
        linkDelegationPolicy = 2

        def setLinkDelegationPolicy(self, policy):
            self.linkDelegationPolicy = policy

        def acceptNavigationRequest(self, url, navigation_type, isMainFrame):
            """
            Overloaded method to handle links ourselves
            """
            strURL = str(url.toString())
            if strURL.endswith('png'):
                return False
            elif strURL.startswith('file'):
                if strURL.endswith('tutorial.html') or\
                        strURL.endswith('tutorial'):
                    self.linkClicked.emit(url)
                    return False
                else:
                    return True
            else:
                self.linkClicked.emit(url)
                return False

    class QWebView(qt.QtWeb.QWebEngineView):
        """Web view"""
        def __init__(self):
            qt.QtWeb.QWebEngineView.__init__(self)
            web_page = WebPage(self)
            self.setPage(web_page)


class SphinxWorker(qt.QObject):
    html_ready = qt.pyqtSignal()

    def prepare(self, doc=None, docName=None, docArgspec=None,
                docNote=None, img_path=""):
        self.doc = doc
        self.docName = docName
        self.docArgspec = docArgspec
        self.docNote = docNote
        self.img_path = img_path

    def render(self):
        cntx = ext.generate_context(
            name=self.docName,
            argspec=self.docArgspec,
            note=self.docNote)
        ext.sphinxify(self.doc, cntx, img_path=self.img_path)
        self.thread().terminate()
        self.html_ready.emit()


class MyTableView(qt.QTableView):
    sig_MoveItem = qt.pyqtSignal(int, int)

    def dropEvent(self, event):
        oldRow = None
        newRow = None
        indexes = self.selectedIndexes()
        if len(indexes) > 0:
            oldRow = indexes[0].row()
        index = self.indexAt(event.pos())
        while index.parent().isValid():
            child = index
            newRow = index.row()
            index = index.parent()
        if all([oldRow, newRow]):
            if oldRow != newRow:
                print("Moving by DND", oldRow, newRow)
                self.sig_MoveItem.emit(oldRow, newRow)

    def updateColumnSize(self):
        self.setVisible(False)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.horizontalHeader().setStretchLastSection(True)
        self.setVisible(True)

    def setDataFromCombo(self, index, table, elStr):
        query = qt.QSqlQuery()
        query.exec_("SELECT id FROM {0} WHERE name={1}".format(
                table, str(elStr)))
        if query.next():
            self.model().setData(index, query.value(0))


class ElementSqlTableModel(qt.QSqlTableModel):

    def __init__(self, db, iconsDir, parentView):
        super(ElementSqlTableModel, self).__init__(db=db)
        self.iconsDir = iconsDir
        self.parentView = parentView

    def data(self, index, role):
        if not index.isValid():
            return
        dataR = super(qt.QSqlTableModel, self).data(
            index, role=qt.QtCore.Qt.DisplayRole)
        if (role == qt.QtCore.Qt.DisplayRole):
            if index.column() == 3 and self.tableName() == 'plots':
                return ""
            elif (index.column() == 3 and self.tableName() == 'oes') or\
                    (index.column() == 2 and self.tableName() == 'materials'):
                if int(dataR) == 0:
                    statStr = "Init Failed"
                elif int(dataR) == 2:
                    statStr = "Init OK"
                else:
                    statStr = "Awaiting Init"
                return statStr
            else:
                return dataR
        elif (role == qt.QtCore.Qt.EditRole):
            return super(qt.QSqlTableModel, self).data(
                    index, role=qt.QtCore.Qt.EditRole)
        elif (role == qt.QtCore.Qt.DecorationRole):
            if index.column() == 0 and self.tableName() == 'oes':
                aicon = 'add{0:1d}'.format(self.index(index.row(), 4).data()+1)
                return qt.QIcon(os.path.join(
                                self.iconsDir, '{}.png'.format(aicon)))
        elif (role == qt.QtCore.Qt.CheckStateRole):
            if index.column() == 3 and self.tableName() == 'plots':
                return qt.QtCore.Qt.Checked if int(dataR) else\
                    qt.QtCore.Qt.Unchecked
        elif (role == qt.QtCore.Qt.FontRole):
            if (index.column() == 0):
                font = qt.QFont()
                font.setBold(True)
                return font
            if (index.column() == 3 and self.tableName() == 'oes') or\
                    (index.column() == 2 and self.tableName() == 'materials'):
                font = qt.QFont()
                font.setBold(True)
                return font
        elif (role == qt.QtCore.Qt.ForegroundRole):
            if (index.column() == 3 and self.tableName() == 'oes') or\
                    (index.column() == 2 and self.tableName() == 'materials'):
                if int(dataR) == 0:
                    color = qt.QtGui.QColor(192, 0, 0)
                elif int(dataR) == 2:
                    color = qt.QtGui.QColor(0, 128, 0)
                else:
                    color = qt.QtGui.QColor(0, 0, 0)
                return qt.QtGui.QBrush(color)

    def flags(self, index):
        itemState = qt.QtCore.Qt.ItemIsEnabled | qt.QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            return itemState | qt.QtCore.Qt.ItemIsEditable
        if index.column() == 3 and self.tableName() == 'plots':
            return itemState | qt.QtCore.Qt.ItemIsUserCheckable
        return itemState 

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if (role == qt.QtCore.Qt.CheckStateRole and
                self.tableName() == 'plots' and
                index.column() == 3):
            value = 1 if value else 0
            role = qt.QtCore.Qt.EditRole

        sdState = qt.QSqlTableModel.setData(self, index, value, role)
        self.dataChanged.emit(index, index)
        return sdState


class ParamSqlTableModel(qt.QSqlTableModel):
    dataReloaded = qt.pyqtSignal()

    def __init__(self, db, filterStr, parentView, parentTable):
        super(ParamSqlTableModel, self).__init__(db=db)
        self.filterStr = filterStr
        self.parentView = parentView
        self.parentTable = parentTable

    def updateFilterRow(self, index):
        filterStr = self.filter()
        rr = index.row()
        cc = self.parentView.model().columnCount()
        self.parentId = self.parentView.model().index(rr, cc-1).data()
        newFilterStr = self.filterStr.format(self.parentTable, index.row())
        if filterStr != newFilterStr:
            self.setFilter(newFilterStr)
            self.select()
            self.dataReloaded.emit()

    def data(self, index, role):
        if not index.isValid():
            return
        dataR = super(qt.QSqlTableModel, self).data(
            index, role=qt.QtCore.Qt.DisplayRole)
        if (role == qt.QtCore.Qt.DisplayRole):
            return dataR
        elif (role == qt.QtCore.Qt.EditRole):
            return super(qt.QSqlTableModel, self).data(
            index, role=qt.QtCore.Qt.EditRole)
#        elif (role == qt.QtCore.Qt.DecorationRole):
#            if index.column() == 0 and self.tableName() == 'oes':
#                aicon = 'add{0:1d}'.format(self.index(index.row(), 4).data()+1)
#                return qt.QIcon(os.path.join(
#                                self.iconsDir, '{}.png'.format(aicon)))
        elif (role == qt.QtCore.Qt.FontRole):
            if (self.tableName() == 'oes_methods' and index.column() == 1
                and index.row() == 0):
                font = qt.QFont()
                font.setItalic(True)
                return font
        elif (role == qt.QtCore.Qt.ForegroundRole):
            if (index.column() == 0):
                color = qt.QtGui.QColor(0, 0, 0)
                if self.tableName() == 'oes_methods' and index.row() == 0:
                    return color
                row = index.row()
                if self.index(row, 1).data() != self.index(row, 2).data():
                    color = qt.QtGui.QColor(0, 0, 225)
                return qt.QtGui.QBrush(color)

    def flags(self, index):
        itemState = qt.QtCore.Qt.ItemIsEnabled | qt.QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            return itemState 
        return itemState | qt.QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role):
#        if not index.isValid():
#            return False
#        if (role == qt.QtCore.Qt.CheckStateRole and
#                self.tableName() == 'plots' and
#                index.column() == 3):
#            value = 1 if value else 0
#            role = qt.QtCore.Qt.EditRole

        sdState = qt.QSqlTableModel.setData(self, index, value, role)
        self.dataChanged.emit(index, index)
        return sdState

class MyComboBox(qt.QComboBox):
    def reload_model(self):
        if self.model() is not None:
            tmpIndex = self.currentIndex()
            queryStr = self.model().query().lastQuery()
            self.model().query().clear()
            query = qt.QSqlQuery()
            query.exec_(queryStr)
            self.model().setQuery(query)
            self.setCurrentIndex(tmpIndex)


class xrtPlotWidget(qt.QWidget):
    windowClosed = qt.pyqtSignal(int)

    def __init__(self, parent=None, plotId=0):
        super(xrtPlotWidget, self).__init__()
        self.parentRef = parent
        self.plotId = plotId

    def closeEvent(self, event):
        self.setVisible(False)
        self.windowClosed.emit(self.plotId)
        event.ignore()

coordParams = ['center', 'pitch', 'roll', 'yaw', 'positionRoll', 'extraPitch',
               'extraRoll', 'extraYaw', 'rotationSequence',
               'extraRotationSequence']
plotBeamParams = ['beam', 'rayFlag', 'fluxKind', 'beamState', 'beamC']


class XrtQook(qt.QWidget):
    statusUpdate = qt.pyqtSignal(tuple)
    sig_resized = qt.Signal("QResizeEvent")
    sig_moved = qt.Signal("QMoveEvent")
    updateBeamCombos = qt.pyqtSignal()

    def __init__(self):
        super(XrtQook, self).__init__()
        self.setAttribute(qt.QtCore.Qt.WA_DeleteOnClose, True)
        self.xrtQookDir = os.path.dirname(os.path.abspath(__file__))
        self.setAcceptDrops(True)
        self.xrt_pypi_version = self.check_pypi_version()  # pypi_ver, cur_ver

        self.prepareViewer = False
        self.isGlowAutoUpdate = False
        self.experimentalMode = False
        self.experimentalModeFilter = ['propagate_wave',
                                       'diffract', 'expose_wave']
        self.statusUpdate.connect(self.updateProgressBar)
        self.iconsDir = os.path.join(self.xrtQookDir, '_icons')
        self.setWindowIcon(qt.QIcon(os.path.join(self.iconsDir, 'xrQt1.ico')))

        self.xrtModules = ['rsources', 'rscreens', 'rmats', 'roes', 'rapts',
                           'rrun', 'raycing', 'xrtplot', 'xrtrun']
        self.axesIndex = {'xaxis': 2, 'yaxis': 3, 'caxis': 4}
        self.db = qt.QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(":memory:")
#        self.db.setDatabaseName("db1.db")
        if not self.db.open():
            print("DATABASE NOT OPEN")
        self.query = qt.QSqlQuery()
        self.queryStrings = dict()
        self.templates = ['Total Flux', 'Absorbed Power', 'Divergence',
                          'Custom']
        self.populateDatabase()

        self.objectFlag = qt.ItemFlags(0)
        self.paramFlag = qt.ItemFlags(qt.ItemIsEnabled | qt.ItemIsSelectable)
        self.valueFlag = qt.ItemFlags(
            qt.ItemIsEnabled | qt.ItemIsEditable | qt.ItemIsSelectable)
        self.checkFlag = qt.ItemFlags(
            qt.ItemIsEnabled | qt.ItemIsUserCheckable | qt.ItemIsSelectable)

        self.objectsDict = dict()
        self.plotWidgets = dict()
        self.initAllModels()
        self.initToolBar()
        self.initTabs()

        self.blViewer = None
        canvasBox = qt.QHBoxLayout()
        canvasSplitter = qt.QSplitter()
        canvasSplitter.setChildrenCollapsible(False)

        mainWidget = qt.QWidget()
        mainWidget.setMinimumWidth(430)
        mainBox = qt.QVBoxLayout()
        mainBox.setContentsMargins(0, 0, 0, 0)
        docBox = qt.QVBoxLayout()
        docBox.setContentsMargins(0, 0, 0, 0)

        self.helptab = qt.QTabWidget()
        docWidget = qt.QWidget()
        docWidget.setMinimumWidth(500)
        docWidget.setMinimumHeight(620)
        # Add worker thread for handling rich text rendering
        self.sphinxThread = qt.QThread(self)
        self.sphinxWorker = SphinxWorker()
        self.sphinxWorker.moveToThread(self.sphinxThread)
        self.sphinxThread.started.connect(self.sphinxWorker.render)
        self.sphinxWorker.html_ready.connect(self._on_sphinx_thread_html_ready)
        self.showWelcomeScreen()

        mainBox.addWidget(self.toolBar)
        tabsLayout = qt.QHBoxLayout()
        tabsLayout.addWidget(self.vToolBar)
        tabsLayout.addWidget(self.tabs)
        mainBox.addItem(tabsLayout)
        mainBox.addWidget(self.progressBar)
        docBox.addWidget(self.webHelp)

        mainWidget.setLayout(mainBox)
        docWidget.setLayout(docBox)

        self.helptab.addTab(docWidget, "Live Doc")
        self.helptab.tabBar().setVisible(False)

        canvasBox.addWidget(canvasSplitter)

        canvasSplitter.addWidget(mainWidget)
        canvasSplitter.addWidget(self.helptab)
        self.setLayout(canvasBox)
        self.beamLineName = 'BeamLine'
        self.beamLine = raycing.BeamLine()
        self.beamLine.flowSource = 'Qook'

    def check_pypi_version(self):
        try:
            import requests
            import distutils.version as dv
            import json
            PyPI = 'https://pypi.python.org/pypi/xrt/json'
            req = requests.get(PyPI)
            if req.status_code != requests.codes.ok:
                return
            rels = json.loads(req.text)['releases']
            v = max([dv.LooseVersion(r) for r in rels if 'b' not in r])
            return v, dv.LooseVersion(xrtversion)
        except:
            pass

    def _addAction(self, module, elname, table, afunction, menu):
        objName = '{0}.{1}'.format(module.__name__, elname)
        elAction = qt.QAction(self)
        elAction.setText(elname)
        elAction.hovered.connect(
            partial(self.showObjHelp, objName))
        elAction.triggered.connect(
            partial(afunction, elname, objName, table, None,
                    True if table=='oes' else False))
        menu.addAction(elAction)

    def initToolBar(self):
        newBLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'filenew.png')),
            'New Beamline Layout',
            self)
        newBLAction.setShortcut('Ctrl+N')
        newBLAction.setIconText('New Beamline Layout')
#        newBLAction.triggered.connect(self.newBL)

        loadBLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'fileopen.png')),
            'Load Beamline Layout',
            self)
        loadBLAction.setShortcut('Ctrl+L')
        loadBLAction.triggered.connect(self.importLayout)

        saveBLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'filesave.png')),
            'Save Beamline Layout',
            self)
        saveBLAction.setShortcut('Ctrl+S')
#        saveBLAction.triggered.connect(self.exportLayout)

        saveBLAsAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'filesaveas.png')),
            'Save Beamline Layout As ...',
            self)
        saveBLAsAction.setShortcut('Ctrl+A')
#        saveBLAsAction.triggered.connect(self.exportLayoutAs)

        generateCodeAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'pythonscript.png')),
            'Generate Python Script',
            self)
        generateCodeAction.setShortcut('Ctrl+G')
        generateCodeAction.triggered.connect(self.generateCode)

        saveScriptAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'pythonscriptsave.png')),
            'Export Python Script',
            self)
        saveScriptAction.setShortcut('Alt+S')
#        saveScriptAction.triggered.connect(self.saveCode)

#        saveScriptAsAction = qt.QAction(
#            qt.QIcon(os.path.join(self.iconsDir, 'pythonscriptsaveas.png')),
#            'Save Python Script As ...',
#            self)
#        saveScriptAsAction.setShortcut('Alt+A')
#        saveScriptAsAction.triggered.connect(self.saveCodeAs)

        runScriptAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'run.png')),
            'Save Python Script And Run',
            self)
        runScriptAction.setShortcut('Ctrl+R')
        runScriptAction.triggered.connect(self.execCode)

        glowAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'eyeglasses7_128.png')),
            'Enable xrtGlow Live Update',
            self)
        if gl.isOpenGL:
            glowAction.setShortcut('CTRL+F1')
            glowAction.setCheckable(True)
            glowAction.setChecked(False)
#            glowAction.toggled.connect(self.toggleGlow)

        OCLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'GPU4.png')),
            'OpenCL Info',
            self)
        if isOpenCL:
            OCLAction.setShortcut('Alt+I')
            OCLAction.triggered.connect(self.showOCLinfo)

        tutorAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'home.png')),
            'Show Welcome Screen',
            self)
        tutorAction.setShortcut('Ctrl+H')
        tutorAction.triggered.connect(self.showWelcomeScreen)

        self.vToolBar = qt.QToolBar('Add Elements buttons')
        self.vToolBar.setOrientation(qt.QtCore.Qt.Vertical)
        self.vToolBar.setIconSize(qt.QtCore.QSize(56, 56))

        modCounter = 0
        queryStr = """INSERT INTO classes (class_name, module_name, primary_type, secondary_type
            ) VALUES ('{0}', '{1}', {2}, {3})"""
        for menuName, amodule, afunction, table, aicon in zip(
                ['Add Source', 'Add OE', 'Add Aperture', 'Add Screen',
                 'Add Material', 'Add Plot'],
                [rsources, roes, rapts, rscreens, rmats, None],
                [self.addElement]*5 + [self.preparePlot],
                ['oes']*4 + ['materials'] + ['plots'],
                ['add{0:1d}'.format(i+1) for i in range(6)]):
            amenuButton = qt.QToolButton()
            amenuButton.setIcon(qt.QIcon(os.path.join(
                self.iconsDir, '{}.png'.format(aicon))))
            amenuButton.setToolTip(menuName)

            tmenu = qt.QMenu()
            if amodule is not None:
                if hasattr(amodule, '__allSectioned__'):
                    for isec, [sec, elnames] in enumerate(list(
                            amodule.__allSectioned__.items())):
                        if isinstance(elnames, (tuple, list)):
                            smenu = tmenu.addMenu(sec)
                            for elname in elnames:
                                self.query.exec_(queryStr.format(
                                        elname, amodule.__name__, modCounter,
                                        isec))
                                self._addAction(
                                    amodule, elname, table, afunction, smenu)
                        else:  # as single entry itself
                            self.query.exec_(queryStr.format(
                                    sec, amodule.__name__, modCounter, -1))
                            self._addAction(amodule, sec, table, afunction,
                                            tmenu)
                else:  # only with __all__
                    for elname in amodule.__all__:
                        self.query.exec_(queryStr.format(
                                elname, amodule.__name__, modCounter, -1))
                        self._addAction(amodule, elname, table, afunction,
                                        tmenu)
            else:  # adding plots
                for template in self.templates:
                    subAction = qt.QAction(self)
                    subAction.setText(template)
                    subAction.hovered.connect(partial(
                            self.populatePlotsMenu, template))
                    tmenu.addAction(subAction)

            amenuButton.setMenu(tmenu)
            amenuButton.setPopupMode(qt.QToolButton.InstantPopup)
            self.vToolBar.addWidget(amenuButton)
            if menuName in ['Add Screen', 'Add Material']:
                self.vToolBar.addSeparator()
            modCounter += 1

        self.tabs = qt.QTabWidget()

        # compacting the default (wider) tabs: (doesn't work in macOS)
#        self.tabs.setStyleSheet("QTabBar::tab {padding: 5px 5px 5px 5px;}")
        self.toolBar = qt.QToolBar('Action buttons')

        self.progressBar = qt.QProgressBar()
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, 100)
        self.progressBar.setAlignment(qt.AlignCenter)
        self.toolBar.addAction(newBLAction)
        self.toolBar.addAction(loadBLAction)
        self.toolBar.addAction(saveBLAction)
        self.toolBar.addAction(saveBLAsAction)
        self.toolBar.addSeparator()
        self.toolBar.addAction(generateCodeAction)
        self.toolBar.addAction(saveScriptAction)
#        self.toolBar.addAction(saveScriptAsAction)
        self.toolBar.addAction(runScriptAction)
        self.toolBar.addSeparator()
        if gl.isOpenGL:
            self.toolBar.addAction(glowAction)
        if isOpenCL:
            self.toolBar.addAction(OCLAction)
        self.toolBar.addAction(tutorAction)
#        self.toolBar.addAction(aboutAction)
        bbl = qt.QShortcut(self)
        bbl.setKey(qt.Key_F4)
#        bbl.activated.connect(self.catchViewer)
        amt = qt.QShortcut(self)
        amt.setKey(qt.CTRL + qt.Key_E)
#        amt.activated.connect(self.toggleExperimentalMode)

    def initTabs(self):
        self.defaultFont = qt.QFont("Courier New", 9)
        if ext.isSphinx:
            self.webHelp = QWebView()
            self.webHelp.page().setLinkDelegationPolicy(2)
            self.webHelp.setContextMenuPolicy(qt.CustomContextMenu)
#            self.webHelp.customContextMenuRequested.connect(self.docMenu)

            self.lastBrowserLink = ''
#            self.webHelp.page().linkClicked.connect(
#                partial(self.linkClicked), type=qt.UniqueConnection)
        else:
            self.webHelp = qt.QTextEdit()
            self.webHelp.setFont(self.defaultFont)
            self.webHelp.setReadOnly(True)

#        for itree in [self.beamlineTable, self.matTree, self.plotTree, self.runTree]:
#            itree.setContextMenuPolicy(qt.CustomContextMenu)
#            itree.clicked.connect(self.showDoc)

#        self.plotTree.customContextMenuRequested.connect(self.plotMenu)
#        self.matTree.customContextMenuRequested.connect(self.matMenu)
#        self.beamlineTable.customContextMenuRequested.connect(self.openMenu)

#        if ext.isSpyderlib:
#            self.codeEdit = ext.codeeditor.CodeEditor(self)
#            self.codeEdit.setup_editor(linenumbers=True, markers=True,
#                                       tab_mode=False, language='py',
#                                       font=self.defaultFont,
#                                       color_scheme='Pydev')
#            qt.QShortcut(qt.QKeySequence.ZoomIn, self, lambda: self.zoom(1))
#            qt.QShortcut(qt.QKeySequence.ZoomOut, self, lambda: self.zoom(-1))
#            qt.QShortcut("Ctrl+0", self, lambda: self.zoom(0))
#            for action in self.codeEdit.menu.actions()[-3:]:
#                self.codeEdit.menu.removeAction(action)
#        else:
#            self.codeEdit = qt.QTextEdit()
#            self.codeEdit.setFont(self.defaultFont)

        self.descrEdit = qt.QTextEdit()
        self.descrEdit.setFont(self.defaultFont)
#        self.descrEdit.textChanged.connect(self.updateDescription)
#        self.typingTimer = qt.QTimer(self)
#        self.typingTimer.setSingleShot(True)
#        self.typingTimer.timeout.connect(self.updateDescriptionDelayed)

        self.setGeometry(100, 100, 1200, 600)

#        if ext.isSpyderConsole:
#            self.codeConsole = ext.pythonshell.ExternalPythonShell(
#                wdir=os.path.dirname(__file__))
#
#        else:
#            self.qprocess = qt.QProcess()
#            self.qprocess.setProcessChannelMode(qt.QProcess.MergedChannels)
#            self.qprocess.readyReadStandardOutput.connect(self.readStdOutput)
#            qt.QShortcut("Ctrl+X", self, self.qprocess.kill)
#            self.codeConsole = qt.QTextEdit()
#            self.codeConsole.setFont(self.defaultFont)
#            self.codeConsole.setReadOnly(True)

        self.beamlineTable = MyTableView()
        self.oeCoordTable = MyTableView()
        self.oeParamTable = MyTableView()
        self.oeMethodTable = MyTableView()

        self.materialsTable = MyTableView()
        self.matParamTable = MyTableView()

        self.plotsTable = MyTableView()
        self.plotBeamTable = MyTableView()
        self.plotFormatTable = MyTableView()
        self.xParamTable = MyTableView()
        self.yParamTable = MyTableView()
        self.cParamTable = MyTableView()

        self.runTable = MyTableView()
        self.blParamTable = MyTableView()

        oeParamTabs = qt.QTabWidget()
        oeParamTabs.addTab(self.oeCoordTable, "Orientation")
        oeParamTabs.addTab(self.oeParamTable, "Parameters")
        oeParamTabs.addTab(self.oeMethodTable, "Propagation")

        tabSplitter = qt.QSplitter()
        tabSplitter.setOrientation(qt.QtCore.Qt.Vertical)
        tabSplitter.addWidget(self.beamlineTable)
        tabSplitter.addWidget(oeParamTabs)
        tabSplitter.setChildrenCollapsible(False)
        self.tabs.addTab(tabSplitter, "Beamline")

        tabSplitter = qt.QSplitter()
        tabSplitter.setOrientation(qt.QtCore.Qt.Vertical)
        tabSplitter.addWidget(self.materialsTable)
        tabSplitter.addWidget(self.matParamTable)
        tabSplitter.setChildrenCollapsible(False)
        self.tabs.addTab(tabSplitter, "Materials")

        plotParamTabs = qt.QTabWidget()
        plotParamTabs.addTab(self.plotBeamTable, "Beam")
        plotParamTabs.addTab(self.plotFormatTable, "Format")
        plotParamTabs.addTab(self.xParamTable, "x-axis")
        plotParamTabs.addTab(self.yParamTable, "y-axis")
        plotParamTabs.addTab(self.cParamTable, "c-axis")

        tabSplitter = qt.QSplitter()
        tabSplitter.setOrientation(qt.QtCore.Qt.Vertical)
        tabSplitter.addWidget(self.plotsTable)
        tabSplitter.addWidget(plotParamTabs)
        tabSplitter.setChildrenCollapsible(False)
        self.tabs.addTab(tabSplitter, "Plots")

        tabSplitter = qt.QSplitter()
        tabSplitter.setOrientation(qt.QtCore.Qt.Vertical)
        tabSplitter.addWidget(self.runTable)
        tabSplitter.addWidget(self.blParamTable)
        self.tabs.addTab(tabSplitter, "Job Settings")

        self.tabs.addTab(self.descrEdit, "Description")
#        self.tabs.addTab(self.codeEdit, "Code")
#        self.tabs.addTab(self.codeConsole, "Console")

        self.initTables()

    def initTables(self):
        for table, view, labels, hidC in zip(
                ['oes', 'materials', 'plots'],
                [self.beamlineTable, self.materialsTable,
                 self.plotsTable],
                [['Element Name', 'Class', 'Beam Source', 'Init State'],
                 ['Material Name', 'Class', 'Init State'],
                 ['Plot Title', 'Element Name', 'Plot Type', 'Show Plot']],
                [[4, 5, 6], [3, 4, 5], [4, 5]]):
            self.objectsDict[table] = dict()  # Preparing the dicts for objects
            fModel = ElementSqlTableModel(db=self.db, iconsDir=self.iconsDir,
                                          parentView=view)
            fModel.setTable(table)
            fModel.setEditStrategy(qt.QSqlTableModel.OnFieldChange)
            fModel.dataChanged.connect(partial(self.addElementCombo, view))
            fModel.rowsInserted.connect(partial(self.addElementCombo, view))
            for ilabel, label in enumerate(labels):
                fModel.setHeaderData(ilabel, qt.QtCore.Qt.Horizontal, label, 0)
            view.setModel(fModel)
            view.setIconSize(qt.QtCore.QSize(32, 32))
            view.resizeRowsToContents()
            view.resizeColumnsToContents()
            view.horizontalHeader().setStretchLastSection(True)
            view.setAlternatingRowColors(True)
#            view.setVisible(False)
#            view.setVisible(True)
            for tc in hidC:
                view.setColumnHidden(tc, True)

        filterBase = "parent_id = (SELECT id FROM {0} WHERE position={1}) AND ptype"
        for table, parent_table, view, parent_view, ptypeStr, hidC in zip(
                ['oes_params']*2 + ['oes_methods'] + ['materials_params'] +
                ['plots_params']*2 + ['axes_params']*3,
                ['oes']*3 + ['materials'] + ['plots']*5,
                [self.oeCoordTable, self.oeParamTable, self.oeMethodTable,
                 self.matParamTable, self.plotBeamTable, self.plotFormatTable,
                 self.xParamTable, self.yParamTable, self.cParamTable],
                [self.beamlineTable]*3 + [self.materialsTable] +
                [self.plotsTable]*5,
                ["=1", "=2", ">=0", ">=0", "=0", "=1", "=2", "=3", "=4"],
                [list(range(2, 6))]*9):
            filterStr = filterBase + ptypeStr
            fModel = ParamSqlTableModel(
                    db=self.db, filterStr=filterStr, parentView=parent_view,
                    parentTable=parent_table)
            fModel.setTable(table)
            fModel.setEditStrategy(qt.QSqlTableModel.OnFieldChange)
            for ilabel, label in enumerate(['Parameter', 'Value']):
                fModel.setHeaderData(ilabel, qt.QtCore.Qt.Horizontal, label, 0)
            fModel.dataReloaded.connect(partial(self.addParamCombo, view))
            fModel.dataReloaded.connect(view.updateColumnSize)
            fModel.dataChanged.connect(partial(self.addParamCombo, view))
            fModel.dataChanged.connect(view.updateColumnSize)
            fModel.dataChanged.connect(self.reInitElement)
            parent_view.clicked.connect(fModel.updateFilterRow)
            view.setModel(fModel)
            view.setAlternatingRowColors(True)
            view.resizeColumnsToContents()
            view.resizeRowsToContents()
            view.horizontalHeader().setStretchLastSection(True)
            for tc in hidC:
                view.setColumnHidden(tc, True)

    def reInitElement(self, indexTL, indexBR):
        sender = self.sender()
        self.initPythonObject(sender.parentTable, sender.parentId)
        sender.parentView.model().select()

    def populatePlotsMenu(self, template):
        sender = self.sender()
        subMenu = qt.QMenu(self)
        scrPwr = " WHERE type == 1" if template == 'Absorbed Power' else ""
        queryStr = "SELECT name FROM oes{}".format(scrPwr)
        self.query.exec_(queryStr)
        while self.query.next():
            oe_name = self.query.value(0)
            pAction = qt.QAction(self)
            pAction.setText(oe_name)
            pAction.triggered.connect(
                partial(self.preparePlot, oe_name, template))
            subMenu.addAction(pAction)
        sender.setMenu(subMenu)

    def populateDatabase(self):
        self.query.exec_("""CREATE TABLE classes (
                    class_name TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    primary_type INTEGER,
                    secondary_type INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE oes (
                    name TEXT NOT NULL,
                    class TEXT NOT NULL,
                    oe_input TEXT,
                    state INTEGER,
                    type INTEGER,
                    position INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE oes_params (
                    pname TEXT NOT NULL,
                    pvalue TEXT,
                    dvalue TEXT,
                    ptype INTEGER,
                    parent_id INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE oes_methods (
                    pname TEXT NOT NULL,
                    pvalue TEXT,
                    dvalue TEXT,
                    ptype INTEGER,
                    parent_id INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE materials (
                    name TEXT NOT NULL,
                    class TEXT NOT NULL,
                    state INTEGER,
                    type INTEGER,
                    position INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE materials_params (
                    pname TEXT NOT NULL,
                    pvalue TEXT,
                    dvalue TEXT,
                    ptype INTEGER,
                    parent_id INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE plots (
                    name TEXT NOT NULL,
                    oe_id INTEGER,
                    type INTEGER,
                    visibility INTEGER,
                    position INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE plots_params (
                    pname TEXT NOT NULL,
                    pvalue TEXT,
                    dvalue TEXT,
                    ptype INTEGER,
                    parent_id INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE default_params (
                    class_name TEXT NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value TEXT,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE axes_params (
                    pname TEXT NOT NULL,
                    pvalue TEXT,
                    dvalue TEXT,
                    ptype INTEGER,
                    parent_id INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE plot_types (
                    type_name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")
        self.query.exec_("""CREATE TABLE beams (
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    parent_id INTEGER NOT NULL,
                    id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL)""")

        self.queryStrings['oes_insert'] =\
            """INSERT INTO oes (name, class, type, oe_input, state, position)
            SELECT "{0}", classes.class_name, classes.primary_type,
            IFNULL((SELECT oes.id FROM oes, beams WHERE
            beams.parent_id = oes.id AND
            beams.type='beamGlobal'
            ORDER BY oes.position DESC LIMIT 1), -1), 1,
            (SELECT COUNT(*) FROM oes) FROM classes WHERE class_name="{1}" """
        self.queryStrings['materials_insert'] =\
            """INSERT INTO materials (name, class, state, type, position)
            SELECT "{0}", classes.class_name, 1, classes.primary_type,
            (SELECT COUNT(*) FROM materials)
            FROM classes WHERE classes.class_name="{1}" """
        self.queryStrings['params_insert'] =\
            """INSERT INTO {3}_params (pname, pvalue,
            dvalue, ptype, parent_id) SELECT "{0}", "{1}",  "{1}", {2},
            (SELECT MAX(id) FROM {3})"""
        self.queryStrings['oes_input_update'] =\
            """UPDATE oes SET oe_input = -1 WHERE id=
            (SELECT MAX(id) FROM oes) AND type=0"""
        self.queryStrings['beams_insert'] =\
            """INSERT INTO beams (name, type, parent_id) VALUES
            ('{0}', '{1}', {2})"""
        self.queryStrings['method_param_insert'] =\
            """INSERT INTO oes_methods (pname, pvalue,
             dvalue, ptype, parent_id) SELECT "{0}", "{1}", "{1}", {2},
            (SELECT oes.id FROM oes WHERE oes.name="{3}")"""
        self.queryStrings['plots_insert'] =\
            """INSERT INTO plots (name, oe_id, type,
            visibility, position) SELECT "{0}", (SELECT id FROM oes WHERE
            name="{1}"), "{2}", {3}, (SELECT COUNT(*) FROM plots)"""
        self.queryStrings['plots_param_insert'] =\
            """INSERT INTO {4}_params (pname, pvalue,
            dvalue, ptype, parent_id) SELECT "{0}", "{1}", "{2}", {3},
            (SELECT MAX(id) FROM plots)"""
        self.queryStrings['get_last_rowid'] =\
            "SELECT LAST_INSERT_ROWID()"
        self.query.exec_("""INSERT INTO beams (name, type,
                         parent_id) VALUES ('None', 'beamGlobal', -1)""")

        self.populateDBPlotParams()

    def processSqlError(self, query):
        sqlError = query.lastError().text()
        if len(sqlError.strip()) > 0:
            print("SQL Error!!!")
            print(query.lastQuery())
            print(sqlError)

#    def polulatePlotTemplates(self):
#
#        for ptype, pname, pvalue in zip([0, ])
#
#
#
#
#        self.query.exec_("""INSERT INTO plot_template_params (ttype, ptype, pname,
#            pvalue) VALUES (0, 0, "fluxKind", "total")""")



    def initAllModels(self):
        self.boolModel = qt.QStandardItemModel()
        self.boolModel.appendRow(qt.QStandardItem('False'))
        self.boolModel.appendRow(qt.QStandardItem('True'))

        self.densityModel = qt.QStandardItemModel()
        self.densityModel.appendRow(qt.QStandardItem('histogram'))
        self.densityModel.appendRow(qt.QStandardItem('kde'))

        self.oclPrecnModel = qt.QStandardItemModel()
        for opm in ['auto', 'float32', 'float64']:
            self.oclPrecnModel.appendRow(qt.QStandardItem(opm))

        self.plotAxisModel = qt.QStandardItemModel()
        for ax in ['x', 'y', 'z', 'x\'', 'z\'', 'energy']:
            self.addValue(self.plotAxisModel, ax)

        self.OCLModel = qt.QStandardItemModel()
        oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                     "None",
                                                     "None")
        oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                     "auto",
                                                     "auto")
        if isOpenCL:
            iDeviceCPU = []
            iDeviceGPU = []
            CPUdevices = []
            GPUdevices = []
            for platform in cl_platforms:
                try:  # at old pyopencl versions:
                    CPUdevices =\
                        platform.get_devices(
                            device_type=cl.device_type.CPU)
                    GPUdevices =\
                        platform.get_devices(
                            device_type=cl.device_type.GPU)
                except cl.RuntimeError:
                    pass
                if len(CPUdevices) > 0:
                    if len(iDeviceCPU) > 0:
                        if CPUdevices[0].vendor == \
                                CPUdevices[0].platform.vendor:
                            iDeviceCPU = CPUdevices
                    else:
                        iDeviceCPU.extend(CPUdevices)
                iDeviceGPU.extend(GPUdevices)

            if len(iDeviceCPU) > 0:
                oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                             "CPU",
                                                             "CPU")
            if len(iDeviceGPU) > 0:
                oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                             "GPU",
                                                             "GPU")
            iDeviceCPU.extend(iDeviceGPU)

            for iplatform, platform in enumerate(cl_platforms):
                for idevice, device in enumerate(platform.get_devices()):
                    if device in iDeviceCPU:
                        oclDev = '({0}, {1})'.format(iplatform, idevice)
                        try:
                            oclDevStr = cl.device_type.to_string(device.type)
                        except ValueError:
                            oclDevStr = str(cl.device_type)
                        oclToolTip = 'Platform: {0}\nDevice: {1}\nType: \
{2}\nCompute Units: {3}\nFP64 Support: {4}'.format(
                            platform.name, device.name,
                            oclDevStr,
                            device.max_compute_units,
                            bool(device.double_fp_config))
                        oclItem, oclItemStr = self.addParam(
                            self.OCLModel, device.name, oclDev)
                        oclItem.setToolTip(oclToolTip)

        self.fluxDataModel = qt.QStandardItemModel()
        self.fluxDataModel.appendRow(qt.QStandardItem("auto"))
        for rfName, rfObj in inspect.getmembers(raycing):
            if rfName.startswith('get_') and\
                    rfName != "get_output":
                flItem = qt.QStandardItem(rfName.replace("get_", ''))
                self.fluxDataModel.appendRow(flItem)

        self.fluxKindModel = qt.QStandardItemModel()
        for flKind in ['total', 'power', 's', 'p',
                       '+45', '-45', 'left', 'right']:
            flItem = qt.QStandardItem(flKind)
            self.fluxKindModel.appendRow(flItem)

        self.polarizationsModel = qt.QStandardItemModel()
        for pol in ['horizontal', 'vertical',
                    '+45', '-45', 'left', 'right', 'None']:
            polItem = qt.QStandardItem(pol)
            self.polarizationsModel.appendRow(polItem)

        self.matKindModel = qt.QStandardItemModel()
        for mtKind in ['mirror', 'thin mirror',
                       'plate', 'lens', 'grating', 'FZP', 'auto']:
            mtItem = qt.QStandardItem(mtKind)
            self.matKindModel.appendRow(mtItem)

        self.matTableModel = qt.QStandardItemModel()
        for mtTable in ['Chantler', 'Chantler total', 'Henke', 'BrCo']:
            mtTItem = qt.QStandardItem(mtTable)
            self.matTableModel.appendRow(mtTItem)

        self.shapeModel = qt.QStandardItemModel()
        for shpEl in ['rect', 'round']:
            shpItem = qt.QStandardItem(shpEl)
            self.shapeModel.appendRow(shpItem)

        self.matGeomModel = qt.QStandardItemModel()
        for mtGeom in ['Bragg reflected', 'Bragg transmitted',
                       'Laue reflected', 'Laue transmitted',
                       'Fresnel']:
            mtGItem = qt.QStandardItem(mtGeom)
            self.matGeomModel.appendRow(mtGItem)

        self.aspectModel = qt.QStandardItemModel()
        for aspect in ['equal', 'auto']:
            aspItem = qt.QStandardItem(aspect)
            self.aspectModel.appendRow(aspItem)

        self.distEModelG = qt.QStandardItemModel()
        for distEMod in ['None', 'normal', 'flat', 'lines']:
            dEItem = qt.QStandardItem(distEMod)
            self.distEModelG.appendRow(dEItem)

        self.distEModelS = qt.QStandardItemModel()
        for distEMod in ['eV', 'BW']:
            dEItem = qt.QStandardItem(distEMod)
            self.distEModelS.appendRow(dEItem)

        self.rayModel = qt.QStandardItemModel()
        for iray, ray in enumerate(['Good', 'Out', 'Over', 'Alive']):
            rayItem, rItemStr = self.addParam(self.rayModel, ray, iray+1)
            rayItem.setCheckable(True)
            rayItem.setCheckState(qt.Checked)

#        self.runModel = qt.QStandardItemModel()
#        self.addProp(self.runModel.invisibleRootItem(), "run_ray_tracing()")
#        self.runModel.itemChanged.connect(self.colorizeChangedParam)
#        self.rootRunItem = self.runModel.item(0, 0)
#        self.blUpdateLatchOpen = True

    def updateProgressBar(self, dataTuple):
        self.progressBar.setValue(self.prbStart +
                                  int(dataTuple[0] * self.prbRange))
        self.progressBar.setFormat(dataTuple[1])

    def classNameToStr(self, name):
        className = str(name)
        if len(className) > 3:
            return '{0}{1}'.format(className[:3].lower(), className[3:])
        else:
            return className.lower()

    def processListWidget(self, view, index):
        iWidget = self.sender()
        chItemText = "("
        for rState in range(iWidget.count()):
            if int(iWidget.item(rState).checkState()) == 2:
                chItemText += str(rState+1) + ","
        else:
            chItemText += ")"
        view.model().setData(index, chItemText)

    def addParam(self, parent, paramName, value, source=None):
        """Add a pair of Parameter-Value Items"""
        toolTip = None
        child0 = qt.QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem(str(value))
        child1.setFlags(self.valueFlag)
        if str(paramName) == "center":
            toolTip = '\"x\" and \"z\" can be set to "auto"\
 for automatic alignment if \"y\" is known'
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        child1.setDropEnabled(False)
        child1.setDragEnabled(False)
        if toolTip is not None:
            child1.setToolTip(toolTip)
            # self.setIItalic(child0)
        row = [child0, child1]
        if useSlidersInTree:
            child2 = qt.QStandardItem()
            row.append(child2)
            child2.setDropEnabled(False)
            child2.setDragEnabled(False)
        if source is None:
            parent.appendRow(row)
        else:
            parent.insertRow(source.row() + 1, row)

        if useSlidersInTree:
            if paramName in withSlidersInTree:
                ind = child0.index().sibling(child0.index().row(), 2)
                slider = qt.QSlider(qt.Horizontal)
                slider.setRange(-10, 10)
                slider.setValue(0)
                slider.valueChanged.connect(
                    partial(self.updateSlider, child1, paramName))
                self.tree.setIndexWidget(ind, slider)

        return child0, child1

    def addProp(self, parent, propName):
        """Add non-editable Item"""
        child0 = qt.QStandardItem(str(propName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem()
        child1.setFlags(self.paramFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        child1.setDropEnabled(False)
        child1.setDragEnabled(False)
        parent.appendRow([child0, child1])
        return child0

    def addValue(self, parent, value, source=None):
        """Add editable Item"""
        child0 = qt.QStandardItem(str(value))
        child0.setFlags(self.valueFlag)
        child1 = qt.QStandardItem()
        child1.setFlags(self.paramFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        child1.setDropEnabled(False)
        child1.setDragEnabled(False)
        if source is None:
            parent.appendRow([child0, child1])
        else:
            parent.insertRow(source.row() + 1, [child0, child1])
        return child0

    def addElementCombo(self, view, index=None):
        if view.model().tableName() in ['oes', 'plots']:
            comboColumn = 2 if view.model().tableName() == 'oes' else 1
            if index is not None:
                if comboColumn == 1 and index.column() == 3:
                    plotPos = index.row()
                    plotVis = view.model().data(
                            index, role=qt.QtCore.Qt.CheckStateRole)
                    plotId = int(view.model().index(plotPos, 5).data())
                    self.plotWidgets[plotId].setVisible(bool(plotVis))
            for ii in range(view.model().rowCount()):
                valueIndex = view.model().index(ii, comboColumn)
                value = valueIndex.data()
                query = qt.QSqlQuery()
                query.exec_("SELECT name FROM oes WHERE id={}".format(value))
                currentName = query.value(0) if query.next() else ""
                queryStr = """SELECT oes.name, oes.id FROM oes, beams WHERE
                    beams.parent_id = oes.id AND
                    beams.type='beamGlobal' AND
                    oes.position < {} ORDER BY oes.position""".format(
                        ii if int(view.model().index(ii, 4).data()) > 0
                        else -10) if comboColumn == 2 else\
                    "SELECT name FROM oes"
                query.exec_(queryStr)
                self.processSqlError(query)
                fModel = qt.QSqlQueryModel()
                fModel.setQuery(query)
                combo = MyComboBox()
                combo.setModel(fModel)
                combo.setCurrentIndex(combo.findText(currentName))
                combo.currentIndexChanged['QString'].connect(partial(
                    view.setDataFromCombo, valueIndex, 'oes'))
                view.setIndexWidget(valueIndex, combo)
                self.beamlineTable.model().dataChanged.connect(
                        combo.reload_model)

        view.setVisible(False)
        view.setVisible(True)

    def addParamCombo(self, view):
        for ii in range(view.model().rowCount()):
            valueIndex = view.model().index(ii, 1)
            value = valueIndex.data()
            paramName = view.model().index(ii, 0).data()
            iWidget = view.indexWidget(valueIndex)
            if iWidget is None:
                combo = None
                if isinstance(self.getVal(value), bool):
                    combo = self.addStandardCombo(
                        self.boolModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif any(paraStr in paramName.lower() for paraStr in
                         ['material', 'tlayer', 'blayer', 'coating',
                          'substrate']):
                    combo = self.addStandardCombo(
                        self.materialsTable.model(), value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall(
                        "density", paramName)) > 0 and\
                        paramName != 'uniformRayDensity':
                    combo = self.addStandardCombo(
                        self.densityModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("polarization",
                                    paramName.lower())) > 0:
                    combo = self.addEditableCombo(
                        self.polarizationsModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("shape",
                                    paramName.lower())) > 0:
                    combo = self.addEditableCombo(
                        self.shapeModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("table",
                                    paramName.lower())) > 0:
                    combo = self.addStandardCombo(
                        self.matTableModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("data",
                         paramName.lower())) > 0 and\
                    view in [self.xParamTable, self.yParamTable,
                             self.cParamTable]:
                    combo = self.addStandardCombo(
                        self.fluxDataModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("label",
                         paramName.lower())) > 0 and\
                    view in [self.xParamTable, self.yParamTable,
                             self.cParamTable]:
                    if value == '':
                        if view == self.xParamTable:
                            value = "x"
                        elif view == self.yParamTable:
                            value = "z"
                        else:
                            value = "energy"
                    combo = self.addEditableCombo(
                        self.plotAxisModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("fluxkind",
                                    paramName.lower())) > 0:
                    combo = self.addStandardCombo(
                        self.fluxKindModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif view == self.matParamTable and len(
                        re.findall("kind", paramName.lower())) > 0:
                    combo = self.addStandardCombo(
                        self.matKindModel, value)
                    view.setIndexWidget(valueIndex, combo)
#                elif len(re.findall("distE",
#                                    paramName)) > 0:
#                    combo = qt.QComboBox()
#                    for icd in range(item.parent().rowCount()):
#                        if item.parent().child(icd,
#                                               0).text() == '_object':
#                            if len(re.findall('Source',
#                                   item.parent().child(
#                                       icd, 1).text())) > 0:
#                                combo.setModel(self.distEModelG)
#                            else:
#                                combo.setModel(self.distEModelS)
#                    combo.setCurrentIndex(combo.findText(value))
#                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("geom", paramName.lower())) > 0:
                    combo = self.addStandardCombo(
                        self.matGeomModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("aspect",
                                    paramName.lower())) > 0:
                    combo = self.addEditableCombo(
                        self.aspectModel, value)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("rayflag",
                                    paramName.lower())) > 0:
                    combo = qt.QListWidget()
                    for iray, ray in enumerate(['Good',
                                                'Out',
                                                'Over',
                                                'Alive']):
                        rayItem = qt.QListWidgetItem(str(ray))
                        if len(re.findall(str(iray + 1),
                                          str(value))) > 0:
                            rayItem.setCheckState(qt.Checked)
                        else:
                            rayItem.setCheckState(qt.Unchecked)
                        combo.addItem(rayItem)
                    combo.setMaximumHeight((
                        combo.sizeHintForRow(1) + 1) *
                            self.rayModel.rowCount())
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("targetopencl",
                                    paramName.lower())) > 0:
                    combo = MyComboBox()
                    combo.setModel(self.OCLModel)
                    oclInd = self.OCLModel.findItems(
                        value, flags=qt.MatchExactly, column=1)
                    if len(oclInd) > 0:
                        oclInd = oclInd[0].row()
                    else:
                        oclInd = 1
                    combo.setCurrentIndex(oclInd)
                    view.setIndexWidget(valueIndex, combo)
                elif len(re.findall("precisionopencl",
                                    paramName.lower())) > 0:
                    combo = self.addStandardCombo(
                        self.oclPrecnModel, value)
                    view.setIndexWidget(valueIndex, combo)
                if combo is not None:
                    if combo.staticMetaObject.className() ==\
                            'QListWidget':
                        combo.clicked.connect(
                            partial(self.processListWidget, view, valueIndex))
                    if combo.staticMetaObject.className() in\
                            ['QComboBox', 'MyComboBox']:
                        combo.currentIndexChanged['QString'].connect(
                            partial(view.model().setData, valueIndex))
        view.setVisible(False)
        view.setVisible(True)

    def addStandardCombo(self, model, value):
        combo = MyComboBox()
        combo.setModel(model)
        combo.setCurrentIndex(combo.findText(value))
        return combo

    def addEditableCombo(self, model, value):
        combo = qt.QComboBox()
        combo.setModel(model)
        if combo.findText(value) < 0:
            newItem = qt.QStandardItem(value)
            model.appendRow(newItem)
        combo.setCurrentIndex(combo.findText(value))
        combo.setEditable(True)
        combo.setInsertPolicy(1)
        return combo

    def addElement(self, name, obj, table, paramDict=None, mtdAuto=False):
        view = self.beamlineTable if table == 'oes' else self.materialsTable
        model = view.model()
        clsName = str(obj).split(".")[-1]
        tableLength = model.rowCount()
        elementId = None

        elementName = name
        baseName = name
        dupl = False
        for i in range(99):
            if dupl:
                elementName = baseName + '{:02d}'.format(i)
            dupl = False
            for ibm in range(tableLength):
                if str(model.index(ibm, 0).data()) ==\
                        str(elementName):
                    dupl = True
            if not dupl:
                break

        queryStr = self.queryStrings['{}_insert'.format(table)].format(
                elementName, clsName)
        self.query.exec_(queryStr)
        self.processSqlError(self.query)
        self.query.exec_(self.queryStrings['get_last_rowid'])
        if self.query.next():
            elementId = self.query.value(0)

        if table == 'oes':
            self.query.exec_(self.queryStrings['oes_input_update'])
            self.processSqlError(self.query)

        for arg, argVal in self.getParams(obj):
            paramtype = 1 if arg in coordParams else 2
            if paramDict is not None:
                try:
                    argVal = paramDict[arg]
                except (KeyError, TypeError):
                    pass
            argVal = elementName if arg == 'name' else argVal
            self.query.exec_(self.queryStrings['params_insert'].format(
                arg, self.halfQuotes(argVal), paramtype, table))
            self.processSqlError(self.query)

        t1 = time.time()
        self.initPythonObject(table, elementId)
        t2 = time.time()
        print("Python Object Init time:", t2-t1, "s")
        model.select()
        self.showDoc(obj)
        if table == 'oes' and mtdAuto:
            self.autoAssignMethod(obj, elementName)

    def addPlot(self, plotList=None, paramsDict=None):
        self.query.exec_(self.queryStrings['plots_insert'].format(
            *plotList))  # (plotName, oeName, template, initial_visibility)
        self.processSqlError(self.query)
        self.query.exec_(self.queryStrings['get_last_rowid'])
        self.processSqlError(self.query)
        if self.query.next():
            plotId = self.query.value(0)
        else:
            return

        for icln, className in enumerate(["XYCPlot"] + ["XYCAxis"]*3):
            self.query.exec_("""SELECT param_name, param_value FROM default_params
                             WHERE class_name='{}'""".format(className))
            self.processSqlError(self.query)
            while self.query.next():
                paramName = self.query.value(0)
                if paramName in self.axesIndex.keys():
                    continue
                defParamVal = self.query.value(1)
                paramValue = defParamVal
                if icln > 0:
                    paramType = icln+1
                else:
                    paramType = 0 if paramName in plotBeamParams else 1
                try:
                    paramValue = paramsDict[paramType][paramName]
                except (KeyError, TypeError):
                    pass
                q2 = qt.QSqlQuery()
                q2.exec_(self.queryStrings['plots_param_insert'].format(
                paramName, paramValue, defParamVal, paramType,
                'plots' if paramType < 2 else 'axes'))
                self.processSqlError(q2)

        obj = "{}.XYCPlot".format(xrtplot.__name__)
        self.plotsTable.model().select()
        self.initPythonObject('plots', plotId)
        self.plotWidgets[plotId] = xrtPlotWidget(
                parent=self, plotId=plotId)
        plotLayout = qt.QVBoxLayout()
        plotLayout.addWidget(self.objectsDict['plots'][plotId].canvas)
        self.plotWidgets[plotId].setLayout(plotLayout)
        self.plotWidgets[plotId].windowClosed.connect(
                self.updatePlotState)
        if bool(plotList[3]):
            self.plotWidgets[plotId].show()
        self.showDoc(obj)

    def addMethod(self, elName, methodDict, methodObj):
        for methKey, methVal in methodDict.items():
            for paramName, paramValue in methVal.items():
                self.query.exec_(
                    self.queryStrings['method_param_insert'].format(
                        paramName, paramValue, methKey, elName))
                self.processSqlError(self.query)
                # method input parameters. Type 0 stands for method name
#        self.showDoc(methodObj)

    def preparePlot(self, oeName=None, plotType='Flux Total'):
        pltLength = self.plotsTable.model().rowCount()
        pltName = "{0:2d} {1} - {2}".format(pltLength, oeName, plotType)
        plotList = [pltName, oeName, plotType, 1]

        paramDict = OrderedDict()
        paramDict[0] = dict()
        paramDict[1] = dict()
        for icln, className in enumerate(["XYCPlot"] + ["XYCAxis"]*3):
            paramType = icln+1
            self.query.exec_("""SELECT param_name, param_value FROM default_params
                             WHERE class_name='{}'""".format(className))
            self.processSqlError(self.query)
            if paramType > 1:
                paramDict[paramType] = dict()
            while self.query.next():
                paramName = self.query.value(0)
                paramValue = self.query.value(1)
                if paramName == 'beam':
                    paramValue = ""
                    q2 = qt.QSqlQuery()
                    q2.exec_("""SELECT beams.name FROM beams WHERE
                             beams.parent_id=(SELECT id FROM oes WHERE
                             name='{}')""".format(oeName))
                    if q2.next():
                        paramValue = q2.value(0)
                if className == "XYCPlot":
                    paramType = 0 if paramName in plotBeamParams else 1
                    if paramName in self.axesIndex.keys():
                        continue
                    elif paramName == "title":
                        paramValue = pltName
                else:
                    if paramType == 4:
                        if paramName == "unit":
                            paramValue = "eV"
                        if paramName == "label":
                            paramValue = "energy"
                    elif paramType == 3:
                        if paramName == "label":
                            paramValue = "y"
                    else:
                        if paramName == "label":
                            paramValue = "x"
                paramDict[paramType][paramName] = paramValue
        self.addPlot(plotList, paramDict)

    def autoAssignMethod(self, elClassStr, elName):
        elCls = eval(elClassStr)
        methodDict = OrderedDict()
        methInParams = OrderedDict()
        methOutParams = OrderedDict()
        if hasattr(elCls, 'hiddenMethods'):
            hmList = elCls.hiddenMethods
        else:
            hmList = []
        for namef, objf in inspect.getmembers(elCls):
            if (inspect.ismethod(objf) or inspect.isfunction(objf)) and\
                    not str(namef).startswith("_") and\
                    not str(namef) in hmList:
                fdoc = objf.__doc__
                if fdoc is not None:
                    objfNm = '{0}.{1}'.format(elClassStr,
                                              objf.__name__)
                    fdoc = re.findall(r"Returned values:.*", fdoc)
                    if len(fdoc) > 0 and (
                            str(objf.__name__) not in
                            self.experimentalModeFilter or
                            self.experimentalMode):
                        fdoc = fdoc[0].replace("Returned values: ", '').split(',')
                        methodName = objfNm.split('.')[-1] + '()'
                        break

        methodDict[0] = {'methodName': methodName}

        for arg, argVal in self.getParams(objfNm):
            if arg == 'bl':
                argVal = self.beamLineName
            elif 'beam' in arg:  # input parameters. taking the last global beam
                self.query.exec_("""SELECT beams.name FROM beams, oes WHERE
                              beams.type = 'beamGlobal' AND
                              beams.parent_id = IFNULL((SELECT oe_input FROM oes
                              WHERE name='{}'), 'None')""".format(elName))
                self.processSqlError(self.query)
                if self.query.next():
                    argVal = "None" if arg.lower() == 'accubeam' else\
                        self.query.value(0)
            methInParams[arg] = argVal
        methodDict[1] = methInParams

        for outstr in fdoc:
            outVal = outstr.strip()
            beamName = '{0}_{1}'.format(elName, outVal)
            methOutParams[outVal] = beamName
            queryStr = """INSERT INTO beams (name, type, parent_id) VALUES
                    ("{0}", "{1}", (SELECT MAX(id) FROM oes))""".format(
                    beamName, outVal.strip(('12')), elName)
            self.query.exec_(queryStr)
            self.processSqlError(self.query)

        methodDict[2] = methOutParams
        self.addMethod(elName, methodDict, objfNm)

    def initPythonObject(self, table, elementId):
        elParams = OrderedDict()
        queryStr = """SELECT "XYCPlot", 0, 0, "xrt.plotter" FROM {0}
                   WHERE id={1}""" if table == 'plots' else\
                   """SELECT  {0}.class, {0}.type, {0}.name, classes.module_name
                   FROM {0}, classes WHERE {0}.id={1} AND
                   {0}.class = classes.class_name"""
        self.query.exec_(queryStr.format(table, elementId))
        self.processSqlError(self.query)
        while self.query.next():
            elProps = [self.query.value(0), self.query.value(1),
                       self.query.value(2), self.query.value(3)]
            #  0-id, 1-class, 2-type, 3-name, 4-module,
        self.query.exec_("""SELECT  pname, pvalue FROM {0}_params WHERE
                         parent_id={1}{2}""".format(
                         table, elementId,
                         " AND ptype<2" if table == 'plots' else ""))
        self.processSqlError(self.query)
        while self.query.next():
            paramName = str(self.query.value(0))
            paramValue = str(self.query.value(1))
            if paramName == 'bl':
                elParams[paramName] = self.beamLine
            elif paramName == 'useQtWidget':
                elParams[paramName] = True
            elif paramName == 'center':
                paramValue = paramValue.strip('[]() ')
                paramValue =\
                    [self.getVal(c.strip())
                     for c in str.split(
                     paramValue, ',')]
                elParams[paramName] = paramValue
            elif paramName.lower() in ['tlayer', 'blayer', 'coating',
                                       'material', 'material2', 'substrate']:
                paramValue =\
                    self.objectsDict['materials'][paramValue][0]
                elParams[paramName] = paramValue
            else:
                elParams[paramName] = self.parametrize(paramValue)
        if table == 'plots':
            for iax, axis in enumerate(['xaxis', 'yaxis', 'caxis']):
                self.query.exec_("""SELECT  pname, pvalue FROM axes_params WHERE
                                 parent_id={0} AND ptype={1}""".format(
                                 elementId, iax+2))
                axParams = OrderedDict()
                while self.query.next():
                    paramName = self.query.value(0)
                    paramValue = self.query.value(1)
                    axParams[paramName] = self.parametrize(paramValue)
                axisObj = eval("{0}.{1}".format(
                        elProps[3], "XYCAxis"))(**axParams)
                elParams[axis] = axisObj
        try:
            elementInstance = eval("{0}.{1}".format(
                    elProps[3], elProps[0]))(**elParams)
            initState = 2
        except:
#            raise
            initState = 0
            elementInstance = None

        if table == 'plots':
            self.objectsDict[table][elementId] = elementInstance
        else:
            self.objectsDict[table][elProps[2]] = [elementInstance, elProps[1]]
            self.query.exec_("""UPDATE {0} SET state={1}
                             WHERE id={2}""".format(
                             table, initState, elementId))
            self.processSqlError(self.query)

    def getVal(self, value):
        if str(value) == 'round':
            return str(value)
        try:
            return eval(str(value))
        except:  # analysis:ignore
            return str(value)

    def quotize(self, value):
        try:
            dummy = unicode  # test for Python3 compatibility analysis:ignore
        except NameError:
            unicode = str
        value = self.getVal(value)
        if isinstance(value, (str, unicode)):
            if 'np.' not in value:
                value = 'r\'{}\''.format(value)
        if isinstance(value, tuple):
            value = list(value)
        return str(value)

    def quotizeAll(self, value):
        return str('r\'{}\''.format(value))

    def halfQuotes(self, value):
        if isinstance(value, raycing.basestring):
            return re.sub('\"', '\'', value)
        else:
            return value

    def parametrize(self, value):
        try:
            dummy = unicode  # test for Python3 compatibility analysis:ignore
        except NameError:
            unicode = str  # analysis:ignore
        if str(value) == 'round':
            return str(value)
        if str(value) == 'None':
            return None
        value = self.getVal(value)
#        print(value)
        if isinstance(value, tuple):
            value = list(value)
#        elif value in self.objectsDict['materials'].keys():
#            value = self.objectsDict['materials'][value]
        return value

    def populateDBPlotParams(self):
        queryStr = """INSERT INTO default_params (class_name, param_name,
            param_value) VALUES ('{0}', '{1}', '{2}')"""
        for name, obj in inspect.getmembers(xrtplot):
            if name in ["XYCPlot", "XYCAxis"] and inspect.isclass(obj):
                for namef, objf in inspect.getmembers(obj):
                    if inspect.ismethod(objf) or inspect.isfunction(objf):
                        if namef == "__init__" and\
                                inspect.getargspec(objf)[3] is not None:
                            for arg, argValue in zip(
                                    inspect.getargspec(objf)[0][1:],
                                    inspect.getargspec(objf)[3]):
                                self.query.exec_(queryStr.format(
                                        name, arg, argValue))
                                self.processSqlError(self.query)

    def getParams(self, obj):
        uArgs = OrderedDict()
        args = []
        argVals = []
        objRef = eval(str(obj))
        isMethod = False
        if hasattr(objRef, 'hiddenParams'):
            hpList = objRef.hiddenParams
        else:
            hpList = []
        if inspect.isclass(objRef):
            for parent in (inspect.getmro(objRef))[:-1]:
                for namef, objf in inspect.getmembers(parent):
                    if inspect.ismethod(objf) or inspect.isfunction(objf):
                        argSpec = inspect.getargspec(objf)
                        if namef == "__init__" and argSpec[3] is not None:
                            for arg, argVal in zip(argSpec[0][1:],
                                                   argSpec[3]):
                                if arg == 'bl':
                                    argVal = 'beamLine'
                                if arg not in args and arg not in hpList:
                                    uArgs[arg] = argVal
                        if namef == "__init__" or namef.endswith("pop_kwargs"):
                            kwa = re.findall("(?<=kwargs\.pop).*?\)",
                                             inspect.getsource(objf),
                                             re.S)
                            if len(kwa) > 0:
                                kwa = [re.split(
                                    ",", kwline.strip("\n ()"),
                                    maxsplit=1) for kwline in kwa]
                                for kwline in kwa:
                                    arg = kwline[0].strip("\' ")
                                    if len(kwline) > 1:
                                        argVal = kwline[1].strip("\' ")
                                    else:
                                        argVal = "None"
                                    if arg not in args and arg not in hpList:
                                        uArgs[arg] = argVal
                        if namef == "__init__" and\
                                str(argSpec.varargs) == 'None' and\
                                str(argSpec.keywords) == 'None':
                            break  # To prevent the parent class __init__
                else:
                    continue
                break
        elif inspect.ismethod(objRef) or inspect.isfunction(objRef):
            argList = inspect.getargspec(objRef)
            if argList[3] is not None:
                if objRef.__name__ == 'run_ray_tracing':
                    uArgs = OrderedDict(zip(argList[0], argList[3]))
                else:
                    isMethod = True
                    uArgs = OrderedDict(zip(argList[0][1:], argList[3]))
        moduleObj = eval(objRef.__module__)
        if hasattr(moduleObj, 'allArguments') and not isMethod:
            for argName in moduleObj.allArguments:
                if str(argName) in uArgs.keys():
                    args.append(argName)
                    argVals.append(uArgs[argName])
        else:
            args = list(uArgs.keys())
            argVals = list(uArgs.values())
        return zip(args, argVals)

    def getArgDescr(self, obj):
        argDesc = dict()
        objRef = eval(obj)

        def parseDescr(obji):
            fdoc = obji.__doc__
            if fdoc is not None:
                fdocRec = re.findall("\*.*?\n\n(?= *\*|\n)", fdoc, re.S | re.U)
                if len(fdocRec) > 0:
                    descs = [re.split(
                        "\*\:",
                        fdc.rstrip("\n")) for fdc in fdocRec]
                    for desc in descs:
                        if len(desc) == 2:
                            descName = desc[0].strip("\*")
                            descBody = desc[1].strip("\* ")
                            if descName not in argDesc.keys():
                                argDesc[descName] = descBody

        def sortDescr():
            retArgDescs = []
            retArgDescVals = []
            argKeys = argDesc.keys()
            for arg in argZip:
                for argKey in argKeys:
                    if argKey not in retArgDescs and str(arg) == str(argKey):
                        retArgDescs.append(argKey)
                        retArgDescVals.append(argDesc[argKey])
                        break
                else:
                    for argKey in argKeys:
                        if argKey not in retArgDescs and\
                                len(re.findall("{0}(?=[\,\s\*])".format(arg),
                                               argKey)) > 0:
                            retArgDescs.append(argKey)
                            retArgDescVals.append(argDesc[argKey])
                            break
            return retArgDescs, retArgDescVals

        argZip = OrderedDict(self.getParams(obj)).keys()

        if inspect.isclass(objRef):
            for parent in inspect.getmro(objRef)[:-1]:
                for namef, objf in inspect.getmembers(parent):
                    if (inspect.ismethod(objf) or inspect.isfunction(objf)):
                        if namef == "__init__" or namef.endswith("pop_kwargs"):
                            parseDescr(objf)
        else:
            parseDescr(objRef)

        return sortDescr()

    def importLayout(self):
        def loadParams(pRecord):
            paramsDict = OrderedDict()
            for cRecord in pRecord:
                if cRecord.attrib['type'] in ['prop', 'object']:
                    continue
                paramsDict[str(cRecord.tag)] = str(cRecord.text)
            return paramsDict

        def fixBeamStructure(bName, bType, oeName):
            self.query.exec_("""UPDATE beams SET parent_id=
                             (SELECT oes.id FROM oes
                             WHERE oes.name="{1}"), type="{2}"
                             WHERE name="{0}" """.format(
                                 bName, oeName, bType))
            self.processSqlError(self.query)

        def fixOEStructure():
            self.query.exec_("SELECT id FROM oes")
            while self.query.next():
                q3 = qt.QSqlQuery()
                q3.exec_("""UPDATE oes SET oe_input=(SELECT beams.parent_id
                            FROM beams, oes_methods WHERE
                            beams.name=oes_methods.pvalue
                            AND oes_methods.ptype=1
                            AND oes_methods.parent_id={0} LIMIT 1)
                            WHERE id={0}""".format(
                            self.query.value(0)))
                self.processSqlError(q3)

        self.isEmpty = True
        if not self.isEmpty:
            msgBox = qt.QMessageBox()
            if msgBox.warning(self,
                              'Warning',
                              'Current layout will be overwritten. Continue?',
                              buttons=qt.QMessageBox.Yes |
                              qt.QMessageBox.No,
                              defaultButton=qt.QMessageBox.No)\
                    == qt.QMessageBox.Yes:
                self.isEmpty = True
        if self.isEmpty:
            openFileName = ""
            openDialog = qt.QFileDialog()
            openDialog.setFileMode(qt.QFileDialog.ExistingFile)
            openDialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
            openDialog.setNameFilter("XML files (*.xml)")
            if (openDialog.exec_()):
                openFileName = openDialog.selectedFiles()[0]
                ldMsg = 'Loading layout from {}'.format(
                        os.path.basename(str(openFileName)))
                parseOK = False
                try:
                    treeImport = ET.parse(openFileName)
                    parseOK = True
                except (IOError, OSError, ET.ParseError) as errStr:
                    ldMsg = str(errStr)
                if parseOK:
                    self.blUpdateLatchOpen = False
                    root = treeImport.getroot()

                    for beamName in list(loadParams(
                            root.find('Beams')).keys()):
                        self.query.exec_(
                            self.queryStrings['beams_insert'].format(
                                    beamName, "None", -1))

                    for material in root.find('Materials'):
                        materialName = material.tag
                        materialObj = material.find('_object').text
                        materialParams = loadParams(
                                material.find('properties'))
                        self.addElement(materialName, materialObj, 'materials',
                                        materialParams)
                        time.sleep(0.1)  # Material properties file error

                    self.beamLineName = root[2].tag
                    for element in root[2]:
                        if element.attrib['type'] != 'value':
                            continue
                        elementName = element.tag
                        elementObj = element.find('_object').text
                        elementParams = loadParams(element.find('properties'))
                        self.addElement(elementName, elementObj, 'oes',
                                        elementParams)
                        for cRecord in element:
                            if cRecord.tag in ['_object', 'properties']:
                                continue
                            methodObj = cRecord.find('_object').text
                            methodDict = OrderedDict()
                            methodDict[0] = {'methodName': cRecord.tag + "()"}
                            methodDict[1] = loadParams(
                                    cRecord.find('parameters'))
                            methodDict[2] = loadParams(cRecord.find('output'))
                            for mpName, mpVal in methodDict[2].items():
                                fixBeamStructure(
                                    mpVal, str(mpName).strip('12'),
                                    elementName)
                            self.addMethod(elementName, methodDict, methodObj)

                    fixOEStructure()

                    for plot in root.find('plots'):
                        plotName = plot.tag
                        plotOe = -1
                        plotDict = OrderedDict()
                        plotDict[0] = dict()
                        plotDict[1] = dict()
                        plotParams = loadParams(plot)
                        plotType = plotParams['fluxKind']
                        plotName = plotParams['title'] if\
                            plotParams['title'] != "" else plotName
                        query = qt.QSqlQuery()
                        query.exec_("""SELECT oes.name FROM oes, beams WHERE
                                    oes.id=beams.parent_id AND
                                    beams.name="{}" LIMIT 1""".format(
                                        plotParams['beam']))
                        self.processSqlError(query)
                        if query.next():
                            plotOe = query.value(0)
                        plotList = [plotName, plotOe, plotType, 0]
#                        print(plotList)
                        for paramName, paramValue in plotParams.items():
                            pType = 0 if paramName in plotBeamParams else 1
                            plotDict[pType][paramName] = paramValue
                        for axis in plot:
                            if axis.attrib['type'] == 'prop':
                                plotDict[self.axesIndex[axis.tag]] =\
                                    loadParams(axis)
                        self.addPlot(plotList, plotDict)

                    self.layoutFileName = openFileName
                    self.fileDescription = root[5].text if\
                        len(root) > 5 else ""
                    self.descrEdit.setText(self.fileDescription)
                    self.showTutorial(
                        self.fileDescription,
                        "Descriprion",
                        os.path.join(os.path.dirname(os.path.abspath(str(
                            self.layoutFileName))), "_images"))
                    self.setWindowTitle(self.layoutFileName + " - xrtQook")



    def moveItem(self, oldPos, newPos):
        print("Start moving!")
        view = self.tree
        item = self.rootBLItem.child(oldPos, 0)
        parent = item.parent()
        item.model().blockSignals(True)
        self.flattenElement(view, item)
        item.model().blockSignals(False)
        tmpGlowUpdate = False
        if self.isGlowAutoUpdate:
            tmpGlowUpdate = True
            self.isGlowAutoUpdate = False
        newItem = parent.takeRow(oldPos)
        parent.insertRow(newPos, newItem)
        mvDir = np.sign(oldPos-newPos)
        substVec = [oldPos-1, newPos-1, ">=", "<", "+"] if mvDir > 0 else\
            [oldPos-1, newPos-1, "<=", ">", "-"]
#        substVec.append(mvDir)
        queryStr = """UPDATE oes SET position = CASE
            WHEN (position{3}{0} AND position{2}{1}) THEN position{4}1
            WHEN position={0} THEN {1}
            END WHERE position{3}={0} AND position{2}{1}""".format(*substVec)
#        print(queryStr)
        print(self.query.exec_(queryStr))
        print(self.query.exec_("""SELECT oe_name, position, id FROM oes"""))
##                    print(self.query.numRowsAffected())
        while self.query.next():
            print(self.query.value(0), self.query.value(1), self.query.value(2))

#        queryStr = """SELECT beam_name FROM beams, oes WHERE
#                    beams.parent_id = oes.id AND
#                    oes.position < (SELECT position FROM oes WHERE
#                    oe_name="{}") AND
#                    beams.beam_type = 'beamGlobal'""".format(
#                    str(self.rootBLItem.child(oldPos, 0).text()))
#        print(queryStr)
#        print(self.query.exec_(queryStr))
#        while self.query.next():
#            print(self.query.value(0))

        if tmpGlowUpdate:
            self.isGlowAutoUpdate = True
            startFrom = self.nameToFlowPos(newItem[0].text())
            if startFrom is not None:
                self.blPropagateFlow(startFrom)

    def showDoc(self, obj):
        if obj not in [self.curObj, None]:
            self.showObjHelp(obj)

    def showObjHelp(self, obj):
        self.curObj = obj
        argSpecStr = '('
        for arg, argVal in self.getParams(obj):
            showVal = self.quotize(argVal)
            try:
                showVal = showVal.strip('r') if showVal.startswith('r"') else\
                    showVal
                showVal = showVal.strip('"') if\
                    (arg == 'bl' and showVal is not None) else showVal
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    pass
            argSpecStr += '{0}={1}, '.format(arg, showVal)
        argSpecStr = argSpecStr.rstrip(', ') + ')'
        objP = self.getVal(obj)
        nameStr = objP.__name__
        noteStr = obj
        headerDoc = objP.__doc__

        if not inspect.isclass(obj) and headerDoc is not None:
            headerDocRip = re.findall('.+?(?= {4}\*)',
                                      headerDoc,
                                      re.S | re.U)
            if len(headerDocRip) > 0:
                headerDoc = headerDocRip[0].strip()

        argDocStr = u'{0}\n\n'.format(inspect.cleandoc(headerDoc)) if\
            objP.__doc__ is not None else "\n\n"
        dNames, dVals = self.getArgDescr(obj)
        if len(dNames) > 0:
            argDocStr += u'.. raw:: html\n\n   <div class="title"> '\
                u'<h3> Properties: </h3> </div>\n'
        for dName, dVal in zip(dNames, dVals):
            argDocStr += u'*{0}*: {1}\n\n'.format(dName, dVal)
        retValStr = re.findall(r"Returned values:.*", objP.__doc__)
        if len(retValStr) > 0:
            argDocStr += u'.. raw:: html\n\n   <div class="title"> '\
                u'<h3> Returns: </h3> </div>\n'
            retVals = retValStr[-1].replace("Returned values: ", '').split(',')
            argDocStr += ', '.join("*{0}*".format(v.strip()) for v in retVals)

        argDocStr = argDocStr.replace('imagezoom::', 'image::')
        if ext.isSphinx:
            self.webHelp.history().clear()
            self.webHelp.page().history().clear()
            self.renderLiveDoc(argDocStr, nameStr, argSpecStr, noteStr)
        else:
            argDocStr = u'{0}\nDefiniiton: {1}\n\nType: {2}\n\n\n'.format(
                nameStr.upper(), argSpecStr, noteStr) + argDocStr
            self.webHelp.setText(textwrap.dedent(argDocStr))
            self.webHelp.setReadOnly(True)

    def showOCLinfo(self):
        argDocStr = u""
        for iplatform, platform in enumerate(cl_platforms):
            argDocStr += 'Platform {0}: {1}\n'.format(iplatform, platform.name)
            argDocStr += '-' * 25 + '\n\n'
            argDocStr += ':Vendor:  {0}\n'.format(platform.vendor)
            argDocStr += ':Version:  {0}\n'.format(platform.version)
#            argDocStr += ':Extensions:  {0}\n'.format(platform.extensions)
            for idevice, device in enumerate(platform.get_devices()):
                maxFNLen = 0
                maxFVLen = 0
                argDocStr += '{0}**Device {1}**: {2}\n\n'.format(
                    '', idevice, device.name)
                fNames = ['*Type*',
                          '*Max Clock Speed*',
                          '*Compute Units*',
                          '*Local Memory*',
                          '*Constant Memory*',
                          '*Global Memory*',
                          '*FP64 Support*']
                isFP64 = bool(int(device.double_fp_config/63))
                strFP64 = str(isFP64)
                if not isFP64:
                    strFP64 = redStr.format(strFP64)
                fVals = [cl.device_type.to_string(device.type, "%d"),
                         str(device.max_clock_frequency) + ' MHz',
                         str(device.max_compute_units),
                         str(int(device.local_mem_size/1024)) + ' kB',
                         str(int(
                             device.max_constant_buffer_size/1024)) + ' kB',
                         '{0:.2f}'.format(
                             device.global_mem_size/1073741824.) + ' GB',
                         strFP64]
                for fieldName, fieldVal in zip(fNames, fVals):
                    if len(fieldName) > maxFNLen:
                        maxFNLen = len(fieldName)
                    if len(fieldVal) > maxFVLen:
                        maxFVLen = len(fieldVal)
                spacerH = '{0}+{1}+{2}+\n'.format(
                    myTab, (maxFNLen + 2) * '-', (maxFVLen + 4) * '-')
                argDocStr += spacerH
                for fName, fVal in zip(fNames, fVals):
                    argDocStr += '{0}| {1} |  {2}  |\n'.format(
                        myTab,
                        fName + (maxFNLen - len(fName)) * ' ',
                        fVal + (maxFVLen - len(fVal)) * ' ')
                    argDocStr += spacerH
        argDocStr += '\n'
        argDocStr = argDocStr.replace('imagezoom::', 'image::')
        if ext.isSphinx:
            self.webHelp.history().clear()
            self.webHelp.page().history().clear()
            self.renderLiveDoc(
                argDocStr, "OpenCL Platforms and Devices", "", "")
        else:
            argDocStr = "OpenCL Platforms and Devices\n\n" + argDocStr
            self.webHelp.setText(textwrap.dedent(argDocStr))
            self.webHelp.setReadOnly(True)

    def renderLiveDoc(self, doc, docName, docArgspec, docNote, img_path=""):
        self.sphinxWorker.prepare(doc, docName, docArgspec, docNote, img_path)
        self.sphinxThread.start()

    def _on_sphinx_thread_html_ready(self):
        """Set our sphinx documentation based on thread result"""
        self.webHelp.load(qt.QUrl(ext.xrtQookPage))

    def updateDescription(self):
        self.typingTimer.start(500)

    def updateDescriptionDelayed(self):
        self.fileDescription = self.descrEdit.toPlainText()
        img_path = "" if self.layoutFileName == "" else\
            os.path.join(
                os.path.dirname(os.path.abspath(str(self.layoutFileName))),
                "_images")
        self.showTutorial(self.fileDescription, "Description", img_path)
        self.descrEdit.setFocus()

    def updatePlotState(self, plotId):
        query = qt.QSqlQuery()
        query.exec_("""UPDATE plots SET visibility=0 WHERE id={}""".format(
                plotId))
        self.plotsTable.model().select()

    def showWelcomeScreen(self):
        Qt_version = qt.QT_VERSION_STR
        PyQt_version = qt.PYQT_VERSION_STR
        locos = pythonplatform.platform(terse=True)
        if 'Linux' in locos:
            locos = " ".join(pythonplatform.linux_distribution())
        if gl.isOpenGL:
            strOpenGL = '{0} {1}'.format(gl.__name__, gl.__version__)
            if not bool(gl.glutBitmapCharacter):
                strOpenGL += ' ' + redStr.format('but GLUT is not found')
        else:
            strOpenGL = 'OpenGL '+redStr.format('not found')
        if isOpenCL:
            vercl = cl.VERSION
            if isinstance(vercl, (list, tuple)):
                vercl = '.'.join(map(str, vercl))
        else:
            vercl = isOpenStatus
        strOpenCL = r'pyopencl {}'.format(vercl)
        strXrt = 'xrt {0} in {1}'.format(
            xrtversion, path_to_xrt).replace('\\', '\\\\')
        if type(self.xrt_pypi_version) is tuple:
            pypi_ver, cur_ver = self.xrt_pypi_version
            if cur_ver < pypi_ver:
                strXrt += \
                    ', **version {0} is available from** PyPI_'.format(pypi_ver)
            else:
                strXrt += ', this is the latest version at PyPI_'

        txt = u"""
.. image:: _images/qookSplash2.gif
   :scale: 80 %
   :target: http://xrt.rtfd.io

.. _PyPI: https://pypi.org/project/xrt

| xrtQook is a qt-based GUI for beamline layout manipulation and automated code generation.
| See a brief `startup tutorial <{0}>`_, `the documentation <http://xrt.rtfd.io>`_ and check the latest updates `at GitHub <https://github.com/kklmn/xrt>`_.

:Created by:
    Roman Chernikov (`Canadian Light Source <http://www.lightsource.ca>`_)\n
    Konstantin Klementiev (`MAX IV Laboratory <https://www.maxiv.lu.se/>`_)
:License:
    MIT License, March 2016
:Your system:
    {1}, Python {2}\n
    Qt {3}, {4} {5}\n
    {6}\n
    {7}\n
    {8} """.format(
            'tutorial',
            locos, pythonplatform.python_version(), Qt_version, qt.QtName,
            PyQt_version, strOpenGL, strOpenCL, strXrt)
        self.showTutorial(txt, "xrtQook")

    def showDescrByTab(self, tab):
        if tab == 4:
            self.updateDescriptionDelayed()

    def showTutorial(self, argDocStr, name, img_path=''):
        if argDocStr is None:
            return
        if not ext.isSphinx:
            return
        argDocStr = argDocStr.replace('imagezoom::', 'image::')
        self.webHelp.history().clear()
        self.webHelp.page().history().clear()
        self.renderLiveDoc(argDocStr, name, "", "", img_path)
        self.curObj = None

    def generateCode(self):
        def createMethodDict(elId, elStr):
            methodObj = None
            query = qt.QSqlQuery()
            query.exec_("""SELECT pvalue FROM oes_methods WHERE
                             parent_id={0} AND ptype=0""".format(elId))
            query.next()
            methodStr = str(query.value(0)).strip('()')

            for namef, objf in inspect.getmembers(eval(elStr)):
                if (inspect.ismethod(objf) or
                        inspect.isfunction(objf)) and\
                        namef == methodStr:
                    methodObj = objf
            inkwargs = OrderedDict()
            outkwargs = OrderedDict()
            query.exec_("""SELECT pname, pvalue FROM oes_methods WHERE
                             parent_id={0} AND ptype=1""".format(elId))
            while query.next():
                paraname = query.value(0)
                if paraname == "Incoming beam":
                    paraname = "beam"
                paravalue = self.parametrize(query.value(1))
                inkwargs[paraname] = paravalue
            query.exec_("""SELECT pname, pvalue FROM oes_methods WHERE
                             parent_id={0} AND ptype=2""".format(elId))
            while query.next():
                paraname = query.value(0)
                paravalue = self.parametrize(query.value(1))
                outkwargs[paraname] = paravalue

            if methodObj is not None:
                return methodObj, inkwargs, outkwargs
            else:
                return None, None, None

        def buildFlow(startFrom=0):
            blFlow = []
            query = qt.QSqlQuery()
            query.exec_("""SELECT classes.class_name, classes.module_name,
                        oes.id, oes.name FROM classes, oes
                        WHERE classes.class_name=oes.class
                        ORDER BY oes.position""")
            while query.next():
                oeCls = query.value(0)
                moduleName = query.value(1)
                oeId = query.value(2)
                oeName = query.value(3)
                methodObj, inkwArgs, outkwArgs = createMethodDict(
                    oeId, "{0}.{1}".format(moduleName, oeCls))
                blFlow.append([oeName, methodObj, inkwArgs, outkwArgs])
            return blFlow

        def run_process(beamLine):
            beamLine.propagate_flow()
            return beamLine.beamsDict

        if self.beamLine.alignE == 'auto':
            self.beamLine.alignE = (self.beamLine.sources[0].eMin +
                                    self.beamLine.sources[0].eMax) * 0.5
        self.beamLine.flow = buildFlow()
        self.beamLine.oesDict = self.objectsDict['oes']
        rrun.run_process = run_process

    def execCode(self):
        xrtrun.run_ray_tracing(
            plots=list(self.objectsDict['plots'].values()),
            backend=r"raycing",
#            processes=2,
#            threads=2,
            repeats=8,
            beamLine=self.beamLine)



    def closeEvent(self, event):
        self.db.close()
        for plot in self.plotWidgets.values():
            plot.close()
        event.accept()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    ex = XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())
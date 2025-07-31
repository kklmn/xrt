﻿# -*- coding: utf-8 -*-
u"""
xrtQook -- a GUI for creating a beamline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. imagezoom:: _images/xrtQook.png
   :align: right
   :alt: &ensp;A view of xrtQook with an empty beamline tree on the left and a
       help panel on the right.

The main interface to xrt is through a python script. Many examples of such
scripts can be found in the supplied folders `examples` and `tests`. The script
imports the modules of xrt, instantiates beamline parts, such as synchrotron or
geometric sources, various optical elements, apertures and screens, specifies
required materials for reflection, refraction or diffraction, defines plots and
sets job parameters.

The Qt tool :mod:`xrtQook` takes these ingredients as GUI elements and prepares
a ready to use script that can be run within the tool itself or in an external
Python context. :mod:`xrtQook` has a parallelly updated help panel that
provides a complete list of parameters for the used objects. :mod:`xrtQook`
writes/reads the recipes of beamlines into/from xml files.

In the present version, :mod:`xrtQook` does not provide automated generation of
*scans* and does not create *wave propagation* sequences. For these two tasks,
the corresponding script parts have to be written manually based on the
supplied examples and the present documentation.

See a brief :ref:`tutorial for xrtQook <qook_tutorial>`.

"""

from __future__ import print_function
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "18 Jan 2023"
__version__ = "1.4"

_DEBUG_ = False  # If False, exceptions inside the module are ignored
redStr = ':red:`{0}`'

import os
import sys
import textwrap
import numpy as np  # analysis:ignore , really needed
import time
from datetime import date
import inspect
if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec
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
import copy

from ..commons import ext  # analysis:ignore

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
from ...version import __version__ as xrtversion  # analysis:ignore
from ... import plotter as xrtplot  # analysis:ignore
from ... import runner as xrtrun  # analysis:ignore
from ..commons import qt  # analysis:ignore
from ..commons import gl  # analysis:ignore
from . import tutorial
if gl.isOpenGL:
    from .. import xrtGlow as xrtglow  # analysis:ignore

try:
    from ...backends.raycing import materials_elemental as rmatsel
    from ...backends.raycing import materials_compounds as rmatsco
    from ...backends.raycing import materials_crystals as rmatscr
    pdfMats = True
except ImportError:
    pdfMats = False
    raise("no predef mats")


path_to_xrt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
myTab = 4*" "

useSlidersInTree = False
withSlidersInTree = ['pitch', 'roll', 'yaw', 'bragg']
slidersInTreeScale = {'pitch': 0.1, 'roll': 0.1, 'yaw': 0.1, 'bragg': 1e-3}


class LevelRestrictedModel(qt.QStandardItemModel):
    def __init__(self):
        super().__init__()
        self._cached_drag_row = None
        self._cached_drag_parent = None

    def supportedDragActions(self):
        return qt.MoveAction

    def canDropMimeData(self, data, action, row, column, parent_index):
        if str(parent_index.data(role=qt.UserRole)) == 'top' and column == 0\
                and row > 1:
            return True
        return False

    def mimeData(self, indexes):
        mime = super().mimeData(indexes)
        if indexes:
            index = indexes[0]
            self._cached_drag_row = index.row()
            self._cached_drag_parent = self.itemFromIndex(index.parent()) if index.parent().isValid() else self.invisibleRootItem()
        return mime

    def dropMimeData(self, data, action, row, column, parent_index):
        if str(parent_index.data(role=qt.UserRole)) != 'top':
            return False

        if action != qt.MoveAction:
            return False

        # decode source
        if not data.hasFormat("application/x-qstandarditemmodeldatalist"):
            return False

        if self._cached_drag_row is None or self._cached_drag_parent is None:
            return False
        # locate original item
        # retrieve via internal stored property or last hovered index
        # OR reconstruct from data (advanced)

        # For now, assume we're tracking source during drag
        moved_row = self._cached_drag_row
        source_parent = self._cached_drag_parent

        # Move entire row (multi-column safe)
        items = source_parent.takeRow(moved_row)

        dest_parent = self.itemFromIndex(parent_index)

#        if parent_index.isValid():
#            dest_parent = self.itemFromIndex(parent_index)
#        else:
#            dest_parent = self.invisibleRootItem()

        if row < 0:
            dest_parent.appendRow(items)
        else:
            dest_parent.insertRow(row, items)

        self._cached_drag_row = None
        self._cached_drag_parent = None

        return True


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
    html_ready = qt.Signal()

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


class XrtQook(qt.QWidget):
    statusUpdate = qt.Signal(tuple)
    sig_resized = qt.Signal("QResizeEvent")
    sig_moved = qt.Signal("QMoveEvent")

    def __init__(self):
        super(XrtQook, self).__init__()
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

        self.xrtModules = ['rsources', 'rscreens',
                           'rmats', 'rmatsel', 'rmatsco', 'rmatscr',
                           'roes', 'rapts',
                           'rrun', 'raycing', 'xrtplot', 'xrtrun']

        self.objectFlag = qt.ItemFlags(0)
        self.paramFlag = qt.ItemFlags(qt.ItemIsEnabled | qt.ItemIsSelectable)
        self.valueFlag = qt.ItemFlags(
            qt.ItemIsEnabled | qt.ItemIsEditable | qt.ItemIsSelectable)
        self.checkFlag = qt.ItemFlags(
            qt.ItemIsEnabled | qt.ItemIsUserCheckable | qt.ItemIsSelectable)

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

        mainBox.addWidget(self.toolBar)
        tabsLayout = qt.QHBoxLayout()
        tabsLayout.addWidget(self.vToolBar)
        tabsLayout.addWidget(self.tabs)
#        mainBox.addWidget(self.tabs)
        mainBox.addItem(tabsLayout)
#        mainBox.addWidget(self.statusBar)
        mainBox.addWidget(self.progressBar)
        docBox.addWidget(self.webHelp)

        mainWidget.setLayout(mainBox)
        docWidget.setLayout(docBox)
#        docWidget.setStyleSheet("border:1px solid rgb(20, 20, 20);")
        self.helptab.addTab(docWidget, "Live Doc")
        self.helptab.tabBar().setVisible(False)
#        self.helptab.setTabsClosable(True)
#        self.helptab.setWidget(docWidget)

        canvasBox.addWidget(canvasSplitter)

        canvasSplitter.addWidget(mainWidget)
        canvasSplitter.addWidget(self.helptab)
        self.setLayout(canvasBox)
        self.initAllTrees()

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
        except Exception:
            pass

    def _addAction(self, module, elname, afunction, menu):
        objName = '{0}.{1}'.format(module.__name__, elname)
        elAction = qt.QAction(self)
        elAction.setText(elname)
        elAction.hovered.connect(
            partial(self.showObjHelp, objName))
        elAction.triggered.connect(
            partial(afunction, elname, objName))
        menu.addAction(elAction)

    def initToolBar(self):
        newBLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'filenew.png')),
            'New Beamline Layout',
            self)
        newBLAction.setShortcut('Ctrl+N')
        newBLAction.setIconText('New Beamline Layout')
        newBLAction.triggered.connect(self.newBL)

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
        saveBLAction.triggered.connect(self.exportLayout)

        saveBLAsAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'filesaveas.png')),
            'Save Beamline Layout As ...',
            self)
        saveBLAsAction.setShortcut('Ctrl+A')
        saveBLAsAction.triggered.connect(self.exportLayoutAs)

        generateCodeAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'pythonscript.png')),
            'Generate Python Script',
            self)
        generateCodeAction.setShortcut('Ctrl+G')
        generateCodeAction.triggered.connect(self.generateCode)

        saveScriptAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'pythonscriptsave.png')),
            'Save Python Script',
            self)
        saveScriptAction.setShortcut('Alt+S')
        saveScriptAction.triggered.connect(self.saveCode)

        saveScriptAsAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'pythonscriptsaveas.png')),
            'Save Python Script As ...',
            self)
        saveScriptAsAction.setShortcut('Alt+A')
        saveScriptAsAction.triggered.connect(self.saveCodeAs)

        runScriptAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'run.png')),
            'Save Python Script And Run',
            self)
        runScriptAction.setShortcut('Ctrl+R')
        runScriptAction.triggered.connect(self.execCode)

        glowAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, '3dg_256.png')),  #'eyeglasses7_128.png')),
            'Enable xrtGlow Live Update',
            self)
        if gl.isOpenGL:
            glowAction.setShortcut('CTRL+F1')
            glowAction.setCheckable(True)
            glowAction.setChecked(False)
            glowAction.toggled.connect(self.toggleGlow)

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

#        aboutAction = qt.QAction(
#            qt.QIcon(os.path.join(self.iconsDir, 'dialog-information.png')),
#            'About xrtQook',
#            self)
#        aboutAction.setShortcut('Ctrl+I')
#        aboutAction.triggered.connect(self.aboutCode)

        self.vToolBar = qt.QToolBar('Add Elements buttons')
        self.vToolBar.setOrientation(qt.QtCore.Qt.Vertical)
        self.vToolBar.setIconSize(qt.QtCore.QSize(56, 56))

        for menuName, amodule, afunction, aicon in zip(
                ['Add Source', 'Add OE', 'Add Aperture', 'Add Screen',
                 'Add Material', 'Add Plot'],
                [rsources, roes, rapts, rscreens, rmats, None],
                [self.addElement]*5 + [self.addPlot],
                ['add{0:1d}.png'.format(i+1) for i in range(6)]):
            amenuButton = qt.QToolButton()
            amenuButton.setIcon(qt.QIcon(os.path.join(self.iconsDir, aicon)))
            amenuButton.setToolTip(menuName)

            tmenu = qt.QMenu(amenuButton)
            if amodule is not None:
                if hasattr(amodule, '__allSectioned__'):
                    for sec, elnames in list(amodule.__allSectioned__.items()):
                        if isinstance(elnames, (tuple, list)):
                            smenu = tmenu.addMenu(sec)
                            for elname in elnames:
                                self._addAction(
                                    amodule, elname, afunction, smenu)
                        else:  # as single entry itself
                            self._addAction(amodule, sec, afunction, tmenu)
                    if menuName.endswith('Material'):
                        pdmmenu = tmenu.addMenu(self.tr("Predefined"))
                        for mlibname, mlib in zip(
                                ['Elemental', 'Compounds', 'Crystals'],
                                [rmatsel, rmatsco, rmatscr]):
                            pdflmenu = pdmmenu.addMenu(self.tr(mlibname))
                            for ssec, snames in list(
                                    mlib.__allSectioned__.items()):
                                if isinstance(snames, (tuple, list)):
                                    pdfsmenu = pdflmenu.addMenu(ssec)
                                    for sname in snames:
                                        self._addAction(
                                                mlib, sname, self.addElement,
                                                pdfsmenu)
                                else:
                                    self._addAction(
                                            mlib, ssec, self.addElement,
                                            pdflmenu)

                else:  # only with __all__
                    for elname in amodule.__all__:
                        self._addAction(amodule, elname, afunction, tmenu)
            else:
                for beamType in ['Local Beams', 'Global Beams']:
                    subAction = qt.QAction(self)
                    subAction.setText(beamType)
                    subAction.hovered.connect(partial(
                        self.populateBeamsMenu, subAction, beamType))
                    tmenu.addAction(subAction)
            amenuButton.setMenu(tmenu)
            amenuButton.setPopupMode(qt.QToolButton.InstantPopup)
            self.vToolBar.addWidget(amenuButton)
            if menuName in ['Add Screen', 'Add Material']:
                self.vToolBar.addSeparator()

        self.tabs = qt.QTabWidget()
        # compacting the default (wider) tabs: (doesn't work in macOS)
#        self.tabs.setStyleSheet("QTabBar::tab {padding: 5px 5px 5px 5px;}")
        self.toolBar = qt.QToolBar('Action buttons')

#        self.statusBar = qt.QStatusBar()
#        self.statusBar.setSizeGripEnabled(False)
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
        self.toolBar.addAction(saveScriptAsAction)
        self.toolBar.addAction(runScriptAction)
        self.toolBar.addSeparator()
        if gl.isOpenGL:
            self.toolBar.addAction(glowAction)
        if isOpenCL:
            self.toolBar.addAction(OCLAction)
        self.toolBar.addAction(tutorAction)
#        self.toolBar.addAction(aboutAction)
        bbl = qt.QShortcut(self)
        # bbl.setKey(qt.Key_F4)
        bbl.setKey("F4")
        bbl.activated.connect(self.catchViewer)
        amt = qt.QShortcut(self)
        amt.setKey(qt.CTRL + qt.Key_E)
        amt.activated.connect(self.toggleExperimentalMode)

    def catchViewer(self):
        if self.blViewer is not None:
            if self.helptab.count() < 2:
                self.helptab.tabBar().setVisible(True)
                self.blViewer.oldPos = [self.blViewer.x(), self.blViewer.y()]
                self.blViewer.dockToQook.setEnabled(False)
                self.helptab.addTab(self.blViewer, "Glow")
                self.blViewer.parentRef = None
                self.helptab.setCurrentIndex(1)
            else:
                self.blViewer.setParent(None)
                self.helptab.tabBar().setVisible(False)
                self.blViewer.show()
                self.blViewer.dockToQook.setEnabled(True)
                try:
                    self.blViewer.move(self.blViewer.oldPos[0],
                                       self.blViewer.oldPos[1])
                except:  # analysis:ignore
                    self.blViewer.move(100, 100)
                self.blViewer.parentRef = self

    def populateBeamsMenu(self, sender, beamType):
        # sender = self.sender()
        subMenu = qt.QMenu(self)
        for ibeam in range(self.rootBeamItem.rowCount()):
            if beamType[:5] in str(self.rootBeamItem.child(
                    ibeam, 1).text()):
                pAction = qt.QAction(self)
                beamName = self.rootBeamItem.child(ibeam, 0).text()
                pAction.setText(beamName)
                pAction.triggered.connect(
                    partial(self.addPlotBeam, beamName))
                subMenu.addAction(pAction)
        sender.setMenu(subMenu)

    def initTabs(self):
        self.tree = qt.QTreeView()
        self.matTree = qt.QTreeView()
        self.plotTree = qt.QTreeView()
        self.runTree = qt.QTreeView()

        self.defaultFont = qt.QFont("Courier New", 9)
        if ext.isSphinx:
            self.webHelp = QWebView()
            self.webHelp.page().setLinkDelegationPolicy(2)
            self.webHelp.setContextMenuPolicy(qt.CustomContextMenu)
            self.webHelp.customContextMenuRequested.connect(self.docMenu)

            self.lastBrowserLink = ''
            self.webHelp.page().linkClicked.connect(
                partial(self.linkClicked), type=qt.UniqueConnection)
        else:
            self.webHelp = qt.QTextEdit()
            self.webHelp.setFont(self.defaultFont)
            self.webHelp.setReadOnly(True)

        for itree in [self.tree, self.matTree, self.plotTree, self.runTree]:
            itree.setContextMenuPolicy(qt.CustomContextMenu)
            itree.clicked.connect(self.showDoc)

        self.plotTree.customContextMenuRequested.connect(self.plotMenu)
        self.matTree.customContextMenuRequested.connect(self.matMenu)
        self.tree.customContextMenuRequested.connect(self.openMenu)

        if ext.isSpyderlib:
            self.codeEdit = ext.codeeditor.CodeEditor(self)
            self.codeEdit.setup_editor(linenumbers=True, markers=True,
                                       tab_mode=False, language='py',
                                       font=self.defaultFont,
                                       color_scheme='Pydev')
            if qt.QtName == "PyQt5":
                self.codeEdit.zoom_in.connect(lambda: self.zoom(1))
                self.codeEdit.zoom_out.connect(lambda: self.zoom(-1))
                self.codeEdit.zoom_reset.connect(lambda: self.zoom(0))
            else:
                self.connect(self.codeEdit,
                             qt.SIGNAL('zoom_in()'),
                             lambda: self.zoom(1))
                self.connect(self.codeEdit,
                             qt.SIGNAL('zoom_out()'),
                             lambda: self.zoom(-1))
                self.connect(self.codeEdit,
                             qt.SIGNAL('zoom_reset()'),
                             lambda: self.zoom(0))
            qt.QShortcut(qt.QKeySequence.ZoomIn, self, lambda: self.zoom(1))
            qt.QShortcut(qt.QKeySequence.ZoomOut, self, lambda: self.zoom(-1))
            qt.QShortcut("Ctrl+0", self, lambda: self.zoom(0))
            for action in self.codeEdit.menu.actions()[-3:]:
                self.codeEdit.menu.removeAction(action)
        else:
            self.codeEdit = qt.QTextEdit()
            self.codeEdit.setFont(self.defaultFont)

        self.descrEdit = qt.QTextEdit()
        self.descrEdit.setFont(self.defaultFont)
        self.descrEdit.textChanged.connect(self.updateDescription)
        self.typingTimer = qt.QTimer(self)
        self.typingTimer.setSingleShot(True)
        self.typingTimer.timeout.connect(self.updateDescriptionDelayed)

        self.setGeometry(100, 100, 1200, 600)

        if ext.isSpyderConsole:
            self.codeConsole = ext.pythonshell.ExternalPythonShell(
                wdir=os.path.dirname(__file__))

        else:
            self.qprocess = qt.QProcess()
            self.qprocess.setProcessChannelMode(qt.QProcess.MergedChannels)
            self.qprocess.readyReadStandardOutput.connect(self.readStdOutput)
            qt.QShortcut("Ctrl+X", self, self.qprocess.kill)
            self.codeConsole = qt.QTextEdit()
            self.codeConsole.setFont(self.defaultFont)
            self.codeConsole.setReadOnly(True)

        self.tabs.addTab(self.tree, "Beamline")
        self.tabs.addTab(self.matTree, "Materials")
        self.tabs.addTab(self.plotTree, "Plots")
        self.tabs.addTab(self.runTree, "Job Settings")
        self.tabs.addTab(self.descrEdit, "Description")
        self.tabs.addTab(self.codeEdit, "Code")
        self.tabs.addTab(self.codeConsole, "Console")
        self.tabs.currentChanged.connect(self.showDescrByTab)

    def adjustUndockedPos(self, isFloating):
        if isFloating:
            self.sender().move(100, 100)
            self.sender().updateGeometry()

    def readStdOutput(self):
        output = bytes(self.qprocess.readAllStandardOutput()).decode()
        self.codeConsole.append(output.rstrip())

    def zoom(self, factor):
        """Zoom in/out/reset"""
        if factor == 0:
            self.codeEdit.set_font(self.defaultFont)
        else:
            font = self.codeEdit.font()
            size = font.pointSize() + factor
            if size > 0:
                font.setPointSize(size)
                self.codeEdit.set_font(font)

    def docMenu(self, position):
        menu = qt.QMenu()
        menu.addAction("Zoom In", lambda: self.zoomDoc(1))
        menu.addAction("Zoom Out", lambda: self.zoomDoc(-1))
        menu.addAction("Zoom reset", lambda: self.zoomDoc(0))
#        menu.addSeparator()
#        if str(self.webHelp.url().toString()).startswith('http:'):
#            menu.addAction("Back", self.goBack)
#            if self.webHelp.history().canGoForward():
#                menu.addAction("Forward", self.webHelp.forward)
        menu.exec_(self.webHelp.mapToGlobal(position))

#    def goBack(self):
#        if self.webHelp.history().canGoBack():
#            self.webHelp.back()
#        else:
#            self.webHelp.page().showHelp.emit()

    def zoomDoc(self, factor):
        """Zoom in/out/reset"""
        try:
            textSize = self.webHelp.textSizeMultiplier()
            zoomtype = 'textSizeMultiplier'
        except AttributeError:
            try:
                textSize = self.webHelp.zoomFactor()
                zoomtype = 'zoomFactor'
            except AttributeError:
                return

        if factor == 0:
            textSize = 1.
        elif factor == 1:
            if textSize < 2:
                textSize += 0.1
        elif textSize > 0.1:
            textSize -= 0.1

        if zoomtype == 'textSizeMultiplier':
            self.webHelp.setTextSizeMultiplier(textSize)
        elif zoomtype == 'zoomFactor':
            self.webHelp.setZoomFactor(textSize)

    def initAllTrees(self, blProps=None, runProps=None):
        self.blUpdateLatchOpen = False
        comboDelegate = qt.DynamicArgumentDelegate(mainWidget=self)

        # plotTree view
        self.plotTree.setModel(self.plotModel)
        self.plotTree.setAlternatingRowColors(True)
        self.plotTree.setSortingEnabled(False)
        self.plotTree.setHeaderHidden(False)
        self.plotTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.plotTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.plotTree.setItemDelegateForColumn(1, comboDelegate)

        # materialsTree view
        self.matTree.setModel(self.materialsModel)
        self.matTree.setAlternatingRowColors(True)
        self.matTree.setSortingEnabled(False)
        self.matTree.setHeaderHidden(False)
        self.matTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.matTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.matTree.setItemDelegateForColumn(1, comboDelegate)

        # BLTree view
        self.tree.setModel(self.beamLineModel)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)

        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(qt.QTreeView.InternalMove)
        self.tree.setDefaultDropAction(qt.MoveAction)

        self.tree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        headers = ['Parameter', 'Value']
        if useSlidersInTree:
            headers.append('Slider')
        self.tree.model().setHorizontalHeaderLabels(headers)

        self.curObj = None

        self.addElement(copyFrom=blProps,
                        isRoot=True)

        self.tree.expand(self.rootBLItem.index())
        self.tree.setColumnWidth(0, int(self.tree.width()/3))
        self.tree.setItemDelegateForColumn(1, comboDelegate)

        # runTree view
        self.runTree.setModel(self.runModel)
        self.runTree.setAlternatingRowColors(True)
        self.runTree.setSortingEnabled(False)
        self.runTree.setHeaderHidden(False)
        self.runTree.setAnimated(True)
        self.runTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
#        self.runTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.runTree.model().setHorizontalHeaderLabels(headers)

#        for name, obj in inspect.getmembers(xrtrun):
#            if inspect.isfunction(obj) and name == "run_ray_tracing":
#                if getargspec(obj)[3] is not None:
#                    runStr = '{0}.{1}'.format(xrtrun.__name__, name)
#                    self.addObject(self.runTree, self.rootRunItem, runStr)
#                    for arg, argVal in self.getParams(runStr):
#                        if arg.lower() == "plots":
#                            argVal = self.rootPlotItem.text()
#                        if arg.lower() == "beamline":
#                            argVal = self.rootBLItem.text()
#                        self.addParam(self.rootRunItem, arg, argVal)
        if runProps is None:
            runProps = dict(raycing.get_params('xrt.runner.run_ray_tracing'))
            runProps['plots'] = self.rootPlotItem.text()
            runProps['beamLine'] = self.rootBLItem.text()
            runProps['_object'] = 'xrt.runner.run_ray_tracing'

        for arg, argVal in runProps.items():
            if arg == '_object':
                self.addObject(self.runTree, self.rootRunItem, argVal)
            else:
                self.addParam(self.rootRunItem, arg, argVal)

        self.addCombo(self.runTree, self.rootRunItem)
        self.runTree.setColumnWidth(0, int(self.runTree.width()/3))
        index = self.runModel.indexFromItem(self.rootRunItem)
        self.runTree.setExpanded(index, True)

        self.tabs.tabBar().setTabTextColor(0, qt.black)
        self.tabs.tabBar().setTabTextColor(2, qt.black)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("New beamline")

        self.blColorCounter = 0
        self.pltColorCounter = 0
        self.fileDescription = ""
#        self.descrEdit.setText("")
        self.showWelcomeScreen()
        self.writeCodeBox("")
        self.setWindowTitle("xrtQook")
        self.prefixtab = "\t"
        self.ntab = 1
        self.cpChLevel = 0
        self.saveFileName = ""
        self.layoutFileName = ""
        self.glowOnly = False
        self.isEmpty = True
        if blProps is None:
            self.beamLine = raycing.BeamLine()
            self.beamLine.flowSource = 'Qook'
        self.updateBeamlineBeams(item=None)
        self.updateBeamlineMaterials(item=None)
        self.updateBeamline(item=None)
        self.rayPath = None
        if self.blViewer is not None:
            self.blViewer.customGlWidget.clearVScreen()
            self.blViewer.customGlWidget.selColorMin = None
            self.blViewer.customGlWidget.selColorMax = None
            self.blViewer.customGlWidget.tVec = np.array([0., 0., 0.])
            self.blViewer.customGlWidget.coordOffset = [0., 0., 0.]
        self.blUpdateLatchOpen = True

    def initAllModels(self):
        self.blUpdateLatchOpen = False
#        self.beamLineModel = qt.QStandardItemModel()
        self.beamLineModel = LevelRestrictedModel()

#        self.addValue(self.beamLineModel.invisibleRootItem(), "beamLine")

        self.beamLineModel.itemChanged.connect(self.beamLineItemChanged)
#        self.rootBLItem = self.beamLineModel.item(0, 0)
#        self.rootBLItem.setFlags(qt.ItemFlags(
#            qt.ItemIsEnabled | qt.ItemIsEditable |
#            qt.ItemIsSelectable | qt.ItemIsDropEnabled))
#        self.rootBLItem.setData("top", qt.UserRole)

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

        self.materialsModel = qt.QStandardItemModel()
        self.rootMatItem = self.materialsModel.invisibleRootItem()
        self.rootMatItem.setText("Materials")
        self.materialsModel.itemChanged.connect(self.beamLineItemChanged)
        self.addProp(self.materialsModel.invisibleRootItem(), "None")

        self.beamModel = qt.QStandardItemModel()
        self.beamModel.appendRow([qt.QStandardItem("None"),
                                  qt.QStandardItem("GlobalLocal"),
                                  qt.QStandardItem("None"),
                                  qt.QStandardItem('000')])
        self.rootBeamItem = self.beamModel.invisibleRootItem()
        self.rootBeamItem.setText("Beams")
#        self.beamModel.itemChanged.connect(self.updateBeamlineBeams)

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

        self.plotModel = qt.QStandardItemModel()
        self.addValue(self.plotModel.invisibleRootItem(), "plots")
        # self.plotModel.appendRow(qt.QStandardItem("plots"))
        self.rootPlotItem = self.plotModel.item(0, 0)
        self.plotModel.itemChanged.connect(self.colorizeChangedParam)
        self.plotModel.invisibleRootItem().setText("plots")

        self.runModel = qt.QStandardItemModel()
        self.addProp(self.runModel.invisibleRootItem(), "run_ray_tracing()")
        self.runModel.itemChanged.connect(self.colorizeChangedParam)
        self.rootRunItem = self.runModel.item(0, 0)

        self.comboModelDict = {'filamentBeam': self.boolModel,
                               'uniformRayDensity': self.boolModel,
                               'beam': self.beamModel,
                               'geom': self.matGeomModel}

        self.blUpdateLatchOpen = True

    def newBL(self):
        if not self.isEmpty:
            msgBox = qt.QMessageBox()
            if msgBox.warning(self,
                              'Warning',
                              'Current layout will be purged. Continue?',
                              buttons=qt.QMessageBox.Yes |
                              qt.QMessageBox.No,
                              defaultButton=qt.QMessageBox.No)\
                    == qt.QMessageBox.Yes:
                self.initAllModels()
                self.initAllTrees()
                self.tabs.setCurrentWidget(self.tree)

    def writeCodeBox(self, text):
        if ext.isSpyderlib:
            self.codeEdit.set_text(text)
        else:
            self.codeEdit.setText(text)

    def setIBold(self, item):
        eFont = item.font()
        eFont.setBold(True)
        item.setFont(eFont)

    def setIFontColor(self, item, color):
        eBrush = item.foreground()
        eBrush.setColor(color)
        item.setForeground(eBrush)

    def setIItalic(self, item):
        eFont = item.font()
        eFont.setItalic(True)
        item.setFont(eFont)

    def capitalize(self, view, item):
        self.setIBold(item)
        view.setCurrentIndex(item.index())
        view.expand(item.index())
        view.setColumnWidth(0, int(view.width()/3))

    def getObjStr(self, selItem, level):
        objRoot = None
        obj = None
        model = selItem.model()
        if model == self.beamLineModel:
            if selItem.hasChildren():
                if level < 3 and selItem.text() != 'properties':
                    objRoot = selItem
                else:
                    objRoot = selItem.parent()
            else:
                try:
                    objRoot = selItem.parent().parent()
                except AttributeError:
                    pass
        elif model == self.materialsModel:
            if level > 0:
                objRoot = selItem.parent() if selItem.hasChildren() else\
                    selItem.parent().parent()
            elif selItem.text() != "None":
                objRoot = selItem
        elif model == self.plotModel:
            if level > 0:
                objRoot = selItem if selItem.hasChildren() else\
                    selItem.parent()
        elif model == self.runModel:
            objRoot = selItem if selItem.hasChildren() else selItem.parent()

        if objRoot is not None:
            for i in range(objRoot.rowCount()):
                if objRoot.child(i, 0).text() == '_object':
                    obj = str(objRoot.child(i, 1).text())
                    break
        return obj

    def showDoc(self, selIndex):
        level = 0
        index = selIndex
        while index.parent().isValid():
            index = index.parent()
            level += 1
        selItem = selIndex.model().itemFromIndex(selIndex)
        obj = self.getObjStr(selItem, level)
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
            headerDocRip = re.findall(r'.+?(?= {4}\*)',
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
        retValStr = '' if objP.__doc__ is None else \
            re.findall(r"Returned values:.*", objP.__doc__)
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

    def showWelcomeScreen(self):
        # https://stackoverflow.com/a/69325836/2696065
        def isWin11():
            return True if sys.getwindowsversion().build > 22000 else False

        Qt_version = qt.QT_VERSION_STR
        PyQt_version = qt.PYQT_VERSION_STR
        locos = pythonplatform.platform(terse=True)
        if 'Linux' in locos:
            try:
                locos = " ".join(pythonplatform.linux_distribution())
            except AttributeError:  # no platform.linux_distribution in py3.8
                try:
                    import distro
                    locos = " ".join(distro.linux_distribution())
                except ImportError:
                    print("do 'pip install distro' for a better view of Linux"
                          " distro string")
        elif 'Windows' in locos:
            if isWin11():
                locos = 'Windows 11'
        if gl.isOpenGL:
            strOpenGL = '{0} {1}'.format(gl.__name__, gl.__version__)
#            if not bool(gl.glutBitmapCharacter):
#                strOpenGL += ' ' + redStr.format('but GLUT is not found')
        else:
            strOpenGL = 'OpenGL '+redStr.format('not found')
        if isOpenCL:
            vercl = cl.VERSION
            if isinstance(vercl, (list, tuple)):
                vercl = '.'.join(map(str, vercl))
        else:
            vercl = isOpenStatus
        strOpenCL = r'pyopencl {}'.format(vercl)
        if ext.isSphinx:
            strSphinx = 'Sphinx {0}'.format(ext.sphinx.__version__)
        else:
            strSphinx = 'Sphinx '+redStr.format('not found')
        strXrt = 'xrt {0} in {1}'.format(
            xrtversion, path_to_xrt).replace('\\', '\\\\')
        if type(self.xrt_pypi_version) is tuple:
            pypi_ver, cur_ver = self.xrt_pypi_version
            if cur_ver < pypi_ver:
                strXrt += ', **version {0} is available from** PyPI_'\
                    .format(pypi_ver)
            elif cur_ver == pypi_ver:
                strXrt += ', this is the latest version on PyPI_'
            elif cur_ver > pypi_ver:
                strXrt += ', this version is ahead of version {0} on PyPI_'\
                    .format(pypi_ver)

        txt = u"""
.. image:: _images/qookSplash2.gif
   :scale: 80 %
   :target: http://xrt.rtfd.io

.. _PyPI: https://pypi.org/project/xrt

| xrtQook is a qt-based GUI for beamline layout manipulation and automated code generation.
| See a brief `startup tutorial <{0}>`_, `the documentation <http://xrt.rtfd.io>`_ and check the latest updates on `GitHub <https://github.com/kklmn/xrt>`_.

:Created by:
    Roman Chernikov (`NSLS-II <http://https://www.bnl.gov/nsls2/>`_)\n
    Konstantin Klementiev (`MAX IV Laboratory <https://www.maxiv.lu.se/>`_)
:License:
    MIT License, March 2016
:Your system:
    {1}, Python {2}\n
    Qt {3}, {4} {5}\n
    {6}\n
    {7}\n
    {8}\n
    {9} """.format(
            'tutorial',
            locos, pythonplatform.python_version(), Qt_version, qt.QtName,
            PyQt_version, strOpenGL, strOpenCL, strSphinx, strXrt)
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

    def linkClicked(self, url):
        strURL = str(url.toString())
        if strURL.endswith('png'):
            return
        if strURL.endswith('tutorial.html') or strURL.endswith('tutorial'):
            self.showTutorial(tutorial.__doc__[60:],
                              "Using xrtQook for script generation")
        elif strURL.startswith('http') or strURL.startswith('ftp'):
            if self.lastBrowserLink == strURL:
                return
            webbrowser.open(strURL)
            self.lastBrowserLink = strURL

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

    def addObject(self, view, parent, obj):
        child0 = qt.QStandardItem('_object')
        child1 = qt.QStandardItem(str(obj))
        child0.setFlags(self.objectFlag)
        child1.setFlags(self.objectFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        child1.setDropEnabled(False)
        child1.setDragEnabled(False)
        parent.appendRow([child0, child1])
        view.setRowHidden(child0.index().row(), parent.index(), True)
        return child0, child1

    def addParam(self, parent, paramName, value, source=None):
        """Add a pair of Parameter-Value Items"""
        toolTip = None
        child0 = qt.QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem(str(value))
        child1.setFlags(self.paramFlag if str(paramName) == 'name' else
                        self.valueFlag)

        if str(paramName) == "center":
            toolTip = '\"x\" and \"z\" can be set to "auto"\
 for automatic alignment if \"y\" is known'
#        if str(paramName) == "pitch":
#            toolTip = 'For single OEs \"pitch\" can be set to "auto"\
# for automatic alignment with known \"roll\", \"yaw\"'
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        if toolTip is not None:
            child1.setToolTip(toolTip)
            # self.setIItalic(child0)
        row = [child0, child1]
        if useSlidersInTree:
            child2 = qt.QStandardItem()
            row.append(child2)
        if not isinstance(source, qt.QStandardItem):
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
        if source is None:
            parent.appendRow([child0, child1])
        else:
            parent.insertRow(source.row() + 1, [child0, child1])
        return child0

# useSlidersInTree
    def updateSlider(self, editItem, paramName, position):
        s = editItem.model().data(editItem.index(), qt.DisplayRole)
        withBrackets = False
        if paramName.lower() == 'bragg' and s[0] == '[' and s[-1] == ']':
            withBrackets = True
            s = s[1:-1]
        for pos in [s.rfind(" *"), s.rfind(" /")]:
            if pos > 0:
                s = s[:pos]
        if position:
            factor = 1. + abs(position)/10.*slidersInTreeScale[paramName]
            s += ' {0} {1:.8g}'.format("*" if position > 0 else "/", factor)
        if withBrackets:
            s = '[' + s + ']'
        editItem.model().setData(editItem.index(), s, qt.EditRole)
        editItem.model().dataChanged.emit(editItem.index(), editItem.index())

    def objToInstance(self, obj):  # Should we rather use class name?
        instanceStr = ''
        if obj is not None:
            instanceStr = 'Instance of {0}'.format(obj.split('.')[-1])
        return instanceStr

    def classNameToStr(self, name):
        className = str(name)
        if len(className) > 2:
            return '{0}{1}'.format(className[:2].lower(), className[2:])
        else:
            return className.lower()

    def addElement(self, name=None, obj=None, copyFrom=None, isRoot=False):
        """
        name: class name
        obj: class string including module
        copyFrom: dict with init kwargs (import from file) or another item

        """

        if isinstance(copyFrom, qt.QStandardItem):
            for i in range(copyFrom.rowCount()):
                if str(copyFrom.child(i, 0).text()) == '_object':
                    obj = str(copyFrom.child(i, 1).text())
                    name = obj.split('.')[-1]
                    break
        elif isinstance(copyFrom, dict):
            elementName = copyFrom['properties'].get('name')
            obj = copyFrom.get('_object')
            name = obj.split('.')[-1]

            methodProps = {}
            for field, val in copyFrom.items():
                if field in ['properties', '_object']:
                    continue
#                methodProps['_object'] = val.get('_object')
                methodProps['parameters'] = val
                break
        elif isRoot:
            elementName = 'BeamLine'
            obj = 'xrt.backends.raycing.BeamLine'

#        print(elementName, obj, name)

        if isRoot:
            tree = self.tree
            rootItem = self.beamLineModel.invisibleRootItem()
        elif 'materials' in obj:
            tree = self.matTree
            rootItem = self.rootMatItem
        else:
            tree = self.tree
            rootItem = self.rootBLItem

        if not isinstance(copyFrom, dict):  # None or another item
            if not isRoot:
                for i in range(99):
                    elementName = self.classNameToStr(name) + '{:02d}'.format(
                            i+1)
                    dupl = False
                    for ibm in range(rootItem.rowCount()):
                        if str(rootItem.child(ibm, 0).text()) ==\
                                str(elementName):
                            dupl = True
                    if not dupl:
                        break

        self.blUpdateLatchOpen = False
        elementItem, elementClassItem = self.addParam(rootItem,
                                                      elementName,
                                                      self.objToInstance(obj),
                                                      source=copyFrom)
        elementItem.model().blockSignals(True)
        elementClassItem.setFlags(self.objectFlag)
        if isRoot:
            self.rootBLItem = elementItem

        flags = qt.ItemFlags(
                qt.ItemIsEnabled | qt.ItemIsEditable |
                qt.ItemIsSelectable | qt.ItemIsDropEnabled)

        flags |= qt.ItemIsDropEnabled if isRoot else qt.ItemIsDragEnabled

        elementItem.setFlags(flags)

        if isinstance(copyFrom, qt.QStandardItem):
            propsDict = OrderedDict()
            for i in range(copyFrom.rowCount()):
                childLevel0 = copyFrom.child(i, 0)
                if str(childLevel0.text()) == 'properties':
                    for j in range(childLevel0.rowCount()):
                        propsDict[str(childLevel0.child(j, 0))] =\
                            str(childLevel0.child(j, 1))
                    break
            propsDict['uuid'] = raycing.uuid.uuid4()
        else:
            propsDict = dict(self.getParams(obj))
            propsDict['uuid'] = 'top' if isRoot else raycing.uuid.uuid4()

        if isinstance(copyFrom, dict):
            propsUpd = copyFrom.get('properties')
            if propsUpd is not None:
                propsDict.update(propsUpd)

        elementItem.setDragEnabled(True)
        elprops = self.addProp(elementItem, 'properties')
        self.addObject(tree, elementItem, obj)

        propsDict['name'] = elementName

        for arg, argVal in propsDict.items():
            if arg == 'uuid':
                elementItem.setData(argVal, qt.UserRole)
                continue
            if arg in ['material', 'material2', 'tlayer', 'blayer', 'coating',
                       'substrate']:
                for iMat in range(self.rootMatItem.rowCount()):
                    matItem = self.rootMatItem.child(iMat, 0)
                    if str(matItem.data(qt.UserRole)) == str(argVal):
                        argVal = str(matItem.text())
                        break
            self.addParam(elprops, arg, argVal)

        self.showDoc(elementItem.index())
        tree.expand(rootItem.index())
        self.capitalize(tree, elementItem)
        self.blUpdateLatchOpen = True
        elementItem.model().blockSignals(False)

        if not isinstance(copyFrom, dict):  # Not import from file
            if tree is self.tree:
                self.updateBeamline(elementItem, newElement=obj)
            else:
                self.updateBeamlineMaterials(elementItem, newElement=obj)

#        if not self.experimentalMode:

        if tree is self.tree and not isRoot:
            if isinstance(copyFrom, dict):
               self.autoAssignMethod(elementItem, methodProps)
            else:
                self.autoAssignMethod(elementItem)

        self.isEmpty = False
        self.tabs.setCurrentWidget(tree)
        tree.setCurrentIndex(elementItem.index())

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
                        argSpec = getargspec(objf)
                        if namef == "__init__" and argSpec[3] is not None:
                            for arg, argVal in zip(argSpec[0][1:],
                                                   argSpec[3]):
                                if arg == 'bl':
                                    argVal = self.rootBLItem.text()
                                if arg not in uArgs and arg not in hpList:
                                    uArgs[arg] = argVal
#                                    args.append(arg)
#                                    argVals.append(argVal)
                        if namef == "__init__" or namef.endswith("pop_kwargs"):
                            kwa = re.findall(r"(?<=kwargs\.pop).*?\)",
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
                                    if arg not in uArgs and arg not in hpList:
                                        uArgs[arg] = argVal
#                                        args.append(arg)
#                                        argVals.append(argVal)
                        attr = 'varkw' if hasattr(argSpec, 'varkw') else \
                            'keywords'
                        if namef == "__init__" and\
                                str(argSpec.varargs) == 'None' and\
                                str(getattr(argSpec, attr)) == 'None':
                            break  # To prevent the parent class __init__
                else:
                    continue
                break
        elif inspect.ismethod(objRef) or inspect.isfunction(objRef):
            argList = getargspec(inspect.unwrap(objRef))
            if argList[3] is not None:
                if objRef.__name__ == 'run_ray_tracing':
                    uArgs = OrderedDict(zip(argList[0], argList[3]))
                else:
                    isMethod = True
                    uArgs = OrderedDict(zip(argList[0][1:], argList[3]))
        try:
            moduleObj = eval(objRef.__module__)
        except NameError:
            import importlib
            moduleObj = importlib.import_module(objRef.__module__)
        if hasattr(moduleObj, 'allArguments') and not isMethod:
            for argName in moduleObj.allArguments:
                if str(argName) in uArgs.keys():
                    args.append(argName)
                    argVals.append(uArgs[argName])
        else:
            args = list(uArgs.keys())
            argVals = list(uArgs.values())
        return zip(args, argVals)

    def beamLineItemChanged(self, item):
        self.colorizeChangedParam(item)
        if self.blUpdateLatchOpen:
            if item.model() is self.beamLineModel:
                self.updateBeamline(item)
            elif item.model() is self.materialsModel:
#                if item.index().parent().isValid():
#                    matItem = item.parent().parent()
#                else:
#                    matItem = item
                self.updateBeamlineMaterials(item)

    def colorizeChangedParam(self, item):
        """ """
#        print("CCP", item.column() == 0, item.isEnabled())
        parent = item.parent()
        if parent is not None and\
                item.column() == 1 and\
                item.isEnabled():
            item.model().blockSignals(True)
            itemRow = item.row()
            obj = None
            for i in range(parent.rowCount()):
                if str(parent.child(i, 0).text()) == '_object':
                    obj = str(parent.child(i, 1).text())
                    break
            if obj is None:
                grandparent = parent.parent()
                if grandparent is not None:
                    for i in range(grandparent.rowCount()):
                        if str(grandparent.child(i, 0).text()) == '_object':
                            obj = str(grandparent.child(i, 1).text())
                            break
            if obj is not None and str(parent.text()) != 'output':
                if str(parent.child(itemRow, 0).text()) == 'beam':
                    color = None
                    if str(item.text()) == 'None':
                        color = qt.red
#                        counter = 1
                    elif parent.child(itemRow, 0).foreground().color() ==\
                            qt.red:
                        color = qt.black
#                        counter = -1
#                    else:
#                        counter = 0
#                    if item.model() == self.beamLineModel:
#                        self.blColorCounter += counter
#                    else:
#                        self.pltColorCounter += counter
#                    self.colorizeTabText(item)
                    if color is not None:
                        self.setIFontColor(parent.child(itemRow, 0), color)
                        if parent.parent() != self.rootPlotItem:
                            self.setIFontColor(parent.parent(), color)
                if parent.child(itemRow, 0).foreground().color() !=\
                        qt.red:
                    for defArg, defArgVal in self.getParams(obj):
                        if str(defArg) == str(parent.child(itemRow, 0).text()):
                            if str(defArgVal) != str(item.text()):
                                color = qt.blue
                            else:
                                color = qt.black
                            self.setIFontColor(parent.child(itemRow, 0), color)
                            break
            item.model().blockSignals(False)
        elif item.column() == 0 and item.isEnabled():  # TODO: Move to method. Rename only
            for i in range(item.rowCount()):
                child0 = item.child(i, 0)
                if str(child0.text()) == 'properties':
                    pyname = raycing.to_valid_var_name(item.text())
                    item.model().blockSignals(True)
                    item.setText(pyname)
                    buuid = str(item.data(qt.UserRole))
                    item.model().blockSignals(False)

                    if item.model() is self.materialsModel:
                        mat = self.beamLine.materialsDict.get(buuid)
                        oldname = mat.name
#                        print(pyname, oldname)
                        self.iterateRename(self.rootMatItem, oldname, pyname,
                                           ['tlay', 'blay', 'coat', 'substrate'])
                        self.iterateRename(self.rootBLItem, oldname, pyname,
                                           ['material'])
                    else:
                        for j in range(self.beamModel.rowCount()):
                            beams = self.beamModel.findItems(buuid, column=2)
                            for bItem in beams:
                                row = bItem.row()
                                btype = self.beamModel.item(row, 1).text()
                                bname = "{}_{}".format(pyname, btype[4:].lower())
                                oldname = self.beamModel.item(row, 0).text()
                                self.beamModel.item(row, 0).setText(bname)
                                self.iterateRename(self.rootBLItem, oldname,
                                                   bname, ['beam'])
                                self.iterateRename(self.rootPlotItem, oldname,
                                                   bname, ['beam'])

                    for j in range(child0.rowCount()):
                        if str(child0.child(j, 0).text()) == 'name':
                            child0.child(j, 1).setText(pyname)
                            break

                    break

#    def colorizeTabText(self, item):
#        if item.model() == self.beamLineModel:
#            color = qt.red if self.blColorCounter > 0 else\
#                qt.black
#            self.tabs.tabBar().setTabTextColor(0, color)
#        elif item.model() == self.plotModel:
#            color = qt.red if self.pltColorCounter > 0 else\
#                qt.black
#            self.tabs.tabBar().setTabTextColor(2, color)

    def iterateRename(self, rootItem, old_name, new_name, mask):
        rootItem.model().blockSignals(True)
        def recurse(item):
            for row in range(item.rowCount()):
                argName = item.child(row, 0)
                argValue = item.child(row, 1)
                if argValue is not None and argValue.text() == old_name and\
                        any([m in argName.text() for m in mask]):
                    argValue.setText(new_name)
                recurse(item.child(row, 0))
        recurse(rootItem)
        rootItem.model().blockSignals(False)

#    def addMethodFromDict(self, parentItem, paramsDict):
#        pass

    def addMethod(self, name, parentItem, outBeams, methProps=None):
        self.beamModel.sort(3)

        elstr = str(parentItem.text())
        eluuid = parentItem.data(qt.UserRole)

        methodOutputDict = OrderedDict()

        if methProps is not None:
#            name = methProps.get('_object')
            methodInputDict = methProps.get('parameters')
            if 'beam' in methodInputDict:
                fModel0 = qt.MultiColumnFilterProxy(
                        {1: 'Global', 2: methodInputDict['beam']})
                fModel0.setSourceModel(self.beamModel)
                beamName = fModel0.data(fModel0.index(0, 0))
                methodInputDict['beam'] = beamName
#            methodOutputDict = methProps.get('output')
        else:
            methodInputDict = OrderedDict()
            for pName, pVal in self.getParams(name):
                if pName == 'bl':
                    pVal = self.rootBLItem.text()  # Ever a case?
                elif 'beam' in pName:
                    fModel0 = qt.QSortFilterProxyModel()
                    fModel0.setSourceModel(self.beamModel)
                    fModel0.setFilterKeyColumn(1)
                    fModel0.setFilterRegExp('Global')
                    fModel = qt.QSortFilterProxyModel()
                    fModel.setSourceModel(fModel0)
                    fModel.setFilterKeyColumn(3)
                    regexp = self.intToRegexp(
                        self.nameToBLPos(eluuid))
                    fModel.setFilterRegExp(regexp)
                    lastIndex = fModel.rowCount() - 1
                    if pName.lower() == 'accubeam':
                        lastIndex = 0
                    pVal = fModel.data(fModel.index(lastIndex, 0))
                methodInputDict[pName] = pVal

        for outstr in outBeams:  # Will always auto-generate with new naming scheme
            outval = outstr.strip()
            beamName = '{0}_{1}'.format(elstr, outval[4:].lower())
            methodOutputDict[outval] = beamName

        self.blUpdateLatchOpen = False
        methodItem = self.addProp(parentItem, name.split('.')[-1] + '()')
        self.setIItalic(methodItem)
        methodProps = self.addProp(methodItem, 'parameters')
        self.addObject(self.tree, methodItem, name)


        for arg, argVal in methodInputDict.items():
            child0, child1 = self.addParam(methodProps, arg, argVal)

        methodOut = self.addProp(methodItem, 'output')

        for outval, beamName in methodOutputDict.items():
            child0, child1 = self.addParam(methodOut, outval, beamName)

            self.beamModel.appendRow([qt.QStandardItem(beamName),
                                      qt.QStandardItem(outval),
                                      qt.QStandardItem(str(eluuid)),
                                      qt.QStandardItem(str(self.nameToBLPos(
                                          eluuid)))])
            try:
                self.beamLine.beamsDict[beamName] = None
            except KeyError:
                if _DEBUG_:
                    raise
                else:
                    pass

        self.showDoc(methodItem.index())
        self.addCombo(self.tree, methodItem)
#        self.tree.expand(methodItem.index())
#        self.tree.expand(methodOut.index())
#        self.tree.expand(methodProps.index())
#        self.tree.setCurrentIndex(methodProps.index())
#        self.tree.setColumnWidth(0, int(self.tree.width()/2))
        self.blUpdateLatchOpen = True
        self.updateBeamline(methodItem, newElement=True)  # TODO:
        self.isEmpty = False

    def addPlot(self, copyFrom=None, plotName=None):
        if plotName is None:
            for i in range(99):
                plotName = 'plot{:02d}'.format(i+1)
                dupl = False
                for ibm in range(self.rootPlotItem.rowCount()):
                    if str(self.rootPlotItem.child(ibm, 0).text()) ==\
                            str(plotName):
                        dupl = True
                if not dupl:
                    break



        plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
        axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))
        plotProps = plotDefArgs

        for pname in ['xaxis', 'yaxis', 'caxis']:
            plotProps[pname] = copy.deepcopy(axDefArgs)
            if isinstance(copyFrom, dict):
                pval = copyFrom.pop(pname, None)
                if pval is not None:
                    plotProps[pname].update(pval)
            plotProps[pname]['_object'] = "xrt.plotter.XYCAxis"                    
            
        if isinstance(copyFrom, dict):
            plotProps.update(copyFrom)
            plotItem = self.addValue(self.rootPlotItem, plotName)
        else:
            plotItem = self.addValue(self.rootPlotItem, plotName, source=copyFrom)

        plotProps['_object'] = "xrt.plotter.XYCPlot"

        if isinstance(copyFrom, qt.QStandardItem):
            self.cpChLevel = 0
            self.copyChildren(plotItem, copyFrom)
        else:
            for pname, pval in plotProps.items():
                if pname in ['_object']:
                    self.addObject(self.plotTree, plotItem,
                                   "xrt.plotter.XYCPlot")
                elif pname in ['xaxis', 'yaxis', 'caxis']:
                    child0 = self.addProp(plotItem, pname)
                    for axname, axval in pval.items():
                        if axname == '_object':
                            self.addObject(self.plotTree, child0,
                                           "xrt.plotter.XYCAxis")
                            continue
                        self.addParam(child0, axname, axval)
                else:
                    if str(pname) == 'title':
                        arg_value = plotItem.text()
                    else:
                        arg_value = pval
                    self.addParam(plotItem, pname, arg_value)

        self.showDoc(plotItem.index())
        self.addCombo(self.plotTree, plotItem)
        self.capitalize(self.plotTree, plotItem)
        self.plotTree.expand(self.rootPlotItem.index())
        self.plotTree.setColumnWidth(0, int(self.plotTree.width()/3))
        self.isEmpty = False
        self.tabs.setCurrentWidget(self.plotTree)

    def addPlotBeam(self, beamName):
        self.addPlot()
        tItem = self.rootPlotItem.child(self.rootPlotItem.rowCount() - 1, 0)
        for ie in range(tItem.rowCount()):
            if tItem.child(ie, 0).text() == 'beam':
                child1 = tItem.child(ie, 1)
                if child1 is not None:
                    child1.setText(beamName)
                    iWidget = self.plotTree.indexWidget(child1.index())
                    if iWidget is not None:
                        iWidget.setCurrentIndex(iWidget.findText(beamName))
                break

    def getArgDescr(self, obj):
        argDesc = dict()
        objRef = eval(obj)

        def parseDescr(obji):
            fdoc = obji.__doc__
            if fdoc is not None:
                fdocRec = re.findall(
                    r"\*.*?\n\n(?= *\*|\n)", fdoc, re.S | re.U)
                if len(fdocRec) > 0:
                    descs = [re.split(
                        r"\*\:",
                        fdc.rstrip("\n")) for fdc in fdocRec]
                    for desc in descs:
                        if len(desc) == 2:
                            descName = desc[0].strip(r"\*")
                            descBody = desc[1].strip(r"\* ")
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
                                len(re.findall(r"{0}(?=[\,\s\*])".format(arg),
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

#    def addMaterial(self, name, obj, copyFrom=None):
#        print("Add material", name)
#        for i in range(99):
#            matName = self.classNameToStr(name) + '{:02d}'.format(i+1)
#            dupl = False
#            for ibm in range(self.rootMatItem.rowCount()):
#                if str(self.rootMatItem.child(ibm, 0).text()) == str(matName):
#                    dupl = True
#            if not dupl:
#                break
#        self.blUpdateLatchOpen = False
#        self.materialsModel().blockSignals(True)
#        matItem, matClass = self.addParam(self.rootMatItem,
#                                          matName,
#                                          self.objToInstance(obj))
#        matClass.setFlags(self.objectFlag)
#        matItem.setFlags(self.valueFlag)
#        matProps = self.addProp(matItem, 'properties')
#        self.addObject(self.matTree, matItem, obj)
#
#        for arg, argVal in self.getParams(obj):
#            self.addParam(matProps, arg, argVal)
#        self.showDoc(matItem.index())
#        self.addCombo(self.matTree, matItem)
#        self.capitalize(self.matTree, matItem)
#        self.materialsModel().blockSignals(False)
#        self.blUpdateLatchOpen = True
#        self.updateBeamlineMaterials(matItem, newMat=True)
#        self.isEmpty = False
#        self.tabs.setCurrentWidget(self.matTree)

    def moveItem(self, mvDir, view, item):
        oldRowNumber = item.index().row()
        statusExpanded = view.isExpanded(item.index())
        parent = item.parent()
        item.model().blockSignals(True)
        self.flattenElement(view, item)
        item.model().blockSignals(False)
        tmpGlowUpdate = False
        if self.isGlowAutoUpdate:
            tmpGlowUpdate = True
            self.isGlowAutoUpdate = False
        newItem = parent.takeRow(oldRowNumber)
        parent.insertRow(oldRowNumber + mvDir, newItem)
        self.addCombo(view, newItem[0])
        view.setExpanded(newItem[0].index(), statusExpanded)
        self.updateBeamline(newItem[0], newOrder=True)
        if tmpGlowUpdate:
            self.isGlowAutoUpdate = True
            startFrom = self.nameToFlowPos(newItem[0].text())
            if startFrom is not None:
                self.blPropagateFlow(startFrom)

    def copyChildren(self, itemTo, itemFrom):
        if itemFrom.hasChildren():
            self.cpChLevel += 1
            for ii in range(itemFrom.rowCount()):
                child0 = itemFrom.child(ii, 0)
                if itemFrom.model() == self.beamLineModel and\
                        str(child0.text()) != 'properties' and\
                        str(child0.text()) != '_object' and\
                        self.cpChLevel == 1:
                    pass
                else:
                    child1 = itemFrom.child(ii, 1)
                    child0n = qt.QStandardItem(child0.text())
                    child0n.setFlags(child0.flags())
                    child0n.setForeground(child0.foreground())
                    if child1 is not None:
                        child1n = qt.QStandardItem(child1.text())
                        child1n.setFlags(child1.flags())
                        itemTo.appendRow([child0n, child1n])
                    else:
                        itemTo.appendRow(child0n)
                    self.copyChildren(child0n, child0)
            self.cpChLevel -= 1
        else:
            pass

    def deleteElement(self, view, item):
        if item.model() == self.materialsModel and\
                item.parent() is None:
            del self.beamLine.materialsDict[str(item.text())]
        if item.parent() == self.rootBLItem:
            self.blUpdateLatchOpen = False

        while item.hasChildren():
            iItem = item.child(0, 0)
            if item.child(0, 1) is not None:
                if str(iItem.text()) == 'beam' and\
                        str(item.child(0, 1).text()) == 'None':
                    if iItem.model() == self.beamLineModel:
                        self.blColorCounter -= 1
                    elif iItem.model() == self.plotModel:
                        self.pltColorCounter -= 1
                iWidget = view.indexWidget(item.child(0, 1).index())
                if iWidget is not None:
                    if item.text() == "output":
                        try:
                            del self.beamLine.beamsDict[str(
                                iWidget.currentText())]
                        except Exception:
                            print("Failed to delete", iWidget.currentText())
                            if _DEBUG_:
                                raise
                            else:
                                pass
                        beamInModel = self.beamModel.findItems(
                            iWidget.currentText())
                        if len(beamInModel) > 0:
                            self.beamModel.takeRow(beamInModel[0].row())

            self.deleteElement(view, iItem)
        else:
            if item.parent() == self.rootBLItem:
                del self.beamLine.oesDict[str(item.text())]
                self.blUpdateLatchOpen = True
                self.updateBeamModel()
                self.updateBeamline(item, newElement=None)
            if item.parent() is not None:
                item.parent().removeRow(item.index().row())
            else:
                item.model().invisibleRootItem().removeRow(item.index().row())

    def exportLayout(self):
        saveStatus = False
        self.beamModel.sort(3)
        if self.layoutFileName == "":
            saveDialog = qt.QFileDialog()
            saveDialog.setFileMode(qt.QFileDialog.AnyFile)
            saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
            saveDialog.setNameFilter("XML files (*.xml)")
            if (saveDialog.exec_()):
                self.layoutFileName = saveDialog.selectedFiles()[0]
        if self.layoutFileName != "":
            self.confText = "<?xml version=\"1.0\"?>\n"
            self.confText += "<Project>\n"
            for item, view in zip([self.rootBeamItem,
                                   self.rootMatItem,
                                   self.rootBLItem,
                                   self.rootPlotItem,
                                   self.rootRunItem],
                                  [None,
                                   self.matTree,
                                   self.tree,
                                   self.plotTree,
                                   self.runTree]):
                item.model().blockSignals(True)
                self.flattenElement(view, item)
                item.model().blockSignals(False)
                if item == self.rootBeamItem:
                    self.updateBeamModel()
                if item == self.rootPlotItem and\
                        self.rootPlotItem.rowCount() == 0:
                    item = self.plotModel.invisibleRootItem()
                    item.setEditable(True)
                self.exportModel(item)
            self.confText += '<description>{0}</description>\n'.format(
                self.fileDescription)
            self.confText += "</Project>\n"
            if not str(self.layoutFileName).endswith('.xml'):
                self.layoutFileName += '.xml'
            try:
                fileObject = open(self.layoutFileName, 'w')
                fileObject.write(self.confText)
                fileObject.close
                saveStatus = True
                self.setWindowTitle(self.layoutFileName + " - xrtQook")
                messageStr = 'Layout saved to {}'.format(
                    os.path.basename(str(self.layoutFileName)))
            except (IOError, OSError) as errs:
                messageStr = 'Failed to save layout to {0}, {1}'.format(
                    os.path.basename(str(self.layoutFileName)), str(errs))
                self.progressBar.setValue(0)
                self.progressBar.setFormat(messageStr)
#            self.statusBar.showMessage(messageStr, 3000)
        return saveStatus

    def exportLayoutAs(self):
        tmpName = self.layoutFileName
        self.layoutFileName = ""
        if not self.exportLayout():
            self.progressBar.setFormat(
                'Failed saving to {}'.format(
                    os.path.basename(str(self.layoutFileName))))
            self.layoutFileName = tmpName

    def exportModel(self, item):
        def despace(pStr):
            return re.sub(' ', '_', pStr)

        flatModel = False
        if item.model() in [self.beamModel]:
            flatModel = True
        if item.hasChildren():
            self.prefixtab = self.ntab * '\t'
            if item.isEditable():
                itemType = "value"
            else:
                itemType = "prop"
            self.confText += '{0}<{1} type=\"{2}\">\n'.format(
                self.prefixtab, despace(str(item.text()).strip('()')),
                itemType)
            self.ntab += 1
            self.prefixtab = self.ntab * '\t'
            for ii in range(item.rowCount()):
                child0 = item.child(ii, 0)
                self.exportModel(child0)
                child1 = item.child(ii, 1)
                if child1 is not None and item.model() not in [self.beamModel]:
                    if child1.flags() != self.paramFlag:
                        if child1.isEnabled():
                            itemType = "param"
                        else:
                            itemType = "object"
                        if int(child1.isEnabled()) == int(child0.isEnabled()):
                            self.confText +=\
                                '{0}<{1} type=\"{3}\">{2}</{1}>\n'.format(
                                    self.prefixtab,
                                    despace(str(child0.text())),
                                    child1.text(),
                                    itemType)
                elif flatModel:
                    self.confText +=\
                        '{0}<{1} type=\"flat\"></{1}>\n'.format(
                            self.prefixtab, despace(str(child0.text())))

            self.ntab -= 1
            self.prefixtab = self.ntab * '\t'
            self.confText += '{0}</{1}>\n'.format(
                self.prefixtab, despace(str(item.text()).strip('()')))
        else:
            pass

    def importLayout(self):
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
                parseOK = True  # False
#                try:
#                    treeImport = ET.parse(openFileName)
#                    parseOK = True
#                except (IOError, OSError, ET.ParseError) as errStr:
#                    ldMsg = str(errStr)
                if parseOK:
                    self.beamLine.load_from_xml(openFileName)
                    project = self.beamLine.export_to_json()
                    beamlineName = next(raycing.islice(project.keys(), 2, 3))
                    self.beamLine.name = beamlineName

                    beamlineInitKWargs = project[beamlineName]['properties']
                    beamlineInitKWargs['name'] = beamlineName
                    self.blUpdateLatchOpen = False

                    self.initAllModels()
                    self.initAllTrees(blProps=
                           {'properties': beamlineInitKWargs,
                            '_object': 'xrt.backends.raycing.BeamLine'},
                            runProps=project.get('run_ray_tracing'))

                    for branch in ['Materials', beamlineName]:
                        for element, elementDict in project.get(branch).items():
                            if str(element) in ['properties', '_object']:
                                continue

                            if branch == beamlineName and 'flow' in project:
                                methDict = project['flow'].get(element)
                                elementDict.update(methDict)
                            elementDict['properties']['uuid'] = element
                            self.addElement(copyFrom=elementDict)

                    for plotName, plotDict in project.get('plots').items():
                        bName = plotDict.get('beam')
                        if isinstance(bName, tuple):  # Replace name in plots
                            beams = self.beamModel.findItems(bName[0],
                                                             column=2)
                            for bItem in beams:
                                row = bItem.row()
                                btype = str(self.beamModel.item(row, 1).text())
                                if btype == bName[-1]:
                                    plotDict['beam'] = self.beamModel.item(
                                            row, 0).text()
                                    break

                        self.addPlot(copyFrom=plotDict, plotName=plotName)

                    self.layoutFileName = openFileName
                    self.fileDescription = project.get('description')
                    if self.fileDescription is not None:
                        self.descrEdit.setText(self.fileDescription)
                        self.showTutorial(
                            self.fileDescription,
                            "Descriprion",
                            os.path.join(os.path.dirname(os.path.abspath(str(
                                self.layoutFileName))), "_images"))
                    self.setWindowTitle(self.layoutFileName + " - xrtQook")
                    self.writeCodeBox("")

                    self.blUpdateLatchOpen = True
                    self.progressBar.setValue(100)
                    self.progressBar.setFormat('Loaded layout from {}'.format(
                            os.path.basename(str(self.layoutFileName))))


    def importLayout_old(self):
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
            self.initAllModels()
            self.initAllTrees()
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

                    self.beamLine.load_from_xml(openFileName)


                    self.blUpdateLatchOpen = False
                    root = treeImport.getroot()
                    self.ntab = 0
                    for (i, rootModel), tree in zip(enumerate(
                                                    [self.rootBeamItem,
                                                     self.rootMatItem,
                                                     self.rootBLItem,
                                                     self.rootPlotItem,
                                                     self.rootRunItem]),
                                                    [None,
                                                     self.matTree,
                                                     self.tree,
                                                     self.plotTree,
                                                     self.runTree]):
                        for ie in range(rootModel.rowCount()):
                            rootModel.removeRow(0)
                        if rootModel == self.rootMatItem:
                            self.addProp(
                                self.materialsModel.invisibleRootItem(),
                                "None")
                        if rootModel in [self.rootBLItem, self.rootPlotItem]:
                            rootModel.setText(root[i].tag)

                        self.iterateImport(tree, rootModel, root[i])

                        self.beamModel.sort(3)
                        if rootModel == self.rootBLItem:
                            for ie in range(self.rootBLItem.rowCount()):
                                if self.rootBLItem.child(ie, 0).text() != "properties" and\
                                        self.rootBLItem.child(ie, 0).text() != "_object":
                                    tItem = self.rootBLItem.child(ie, 0)
                                    elName = str(tItem.text())
                                    uuid = self.beamLine.oenamesToUUIDs[elName]
                                    tItem.setData(uuid, qt.UserRole)
                            self.updateBeamImport()
                        if tree is not None:
#                            self.checkDefaults(None, rootModel)
                            tmpBlColor = self.blColorCounter
                            tmpPltColor = self.pltColorCounter
                            self.addCombo(tree, rootModel)
                            self.blColorCounter = tmpBlColor
                            self.pltColorCounter = tmpPltColor
#                        self.colorizeTabText(rootModel)
#                        msgStr = " {0:d} percent done.".format(int(i*100/5))
                        self.progressBar.setFormat("Loading... %p%")
                        self.progressBar.setValue(int(i*100/10))
#                        self.statusBar.showMessage(ldMsg + msgStr)
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
                    self.writeCodeBox("")
                    self.plotTree.expand(self.rootPlotItem.index())
                    self.plotTree.setColumnWidth(
                        0, int(self.plotTree.width()/3))
                    self.tabs.setCurrentWidget(self.tree)
#                    self.statusBar.showMessage(
#                        'Loaded layout from {}'.format(
#                            os.path.basename(str(self.layoutFileName))), 3000)
                    self.isEmpty = False

#                    if rootModel == self.rootBLItem:
#                    self.updateBeamImport()
                    print("Beam Model")
                    for row in range(self.beamModel.rowCount()):
                        line = ""
                        for col in range(self.beamModel.columnCount()):
                            line += "{} ".format(self.beamModel.item(
                                    row, col).text())
                        print(line)
#                    print(self.beamLine.layoutStr)

#                    time.sleep(1.)
#                    print(self.beamLine.oesDict, '\n\n', self.beamLine.beamsDictU)
#                    print(self.beamLine.flow, '\n\n',self.beamLine.flowU)
#                    try:
#                        self.beamLine = raycing.BeamLine()
#                        self.beamLine.flowSource = 'Qook'
#                        self.progressBar.setFormat(
#                            "Populating the beams... %p%")
#                        self.updateBeamlineBeams(item=None)
#                        self.progressBar.setValue(60)
#                        self.progressBar.setFormat(
#                            "Populating the materials... %p%")
#                        time.sleep(0.5)  # To prevent file read error in Py3
#                        self.updateBeamlineMaterials(item=None)
#                        self.progressBar.setValue(70)
#                        self.prbStart = 70
#                        self.prbRange = 30
#                        self.progressBar.setFormat(
#                            "Initializing optical elements... %p%")
#                        self.updateBeamline(item=None)
#                    except:  # analysis:ignore
#                        if _DEBUG_:
#                            raise
#                        else:
#                            pass
                    self.progressBar.setValue(100)
                    self.progressBar.setFormat('Loaded layout from {}'.format(
                            os.path.basename(str(self.layoutFileName))))
                    self.blUpdateLatchOpen = True
                else:
                    self.progressBar.setValue(0)
                    self.progressBar.setFormat(ldMsg)
#                    self.statusBar.showMessage(ldMsg)

    def iterateImport(self, view, rootModel, rootImport):
        if ET.iselement(rootImport):
            self.ntab += 1
            for childImport in rootImport:
                itemType = str(childImport.attrib['type'])
                itemTag = str(childImport.tag)
                itemText = str(childImport.text)
                child0 = qt.QStandardItem(itemTag)
                if itemType == "flat":
                    if rootModel.model() is not self.beamModel:
                        child0 = rootModel.appendRow(child0)
                    else:
                        rootModel.appendRow(
                            [child0, qt.QStandardItem("None"),
                             qt.QStandardItem("None"),
                             qt.QStandardItem("None")])
                elif itemType == "value":
                    child0 = self.addValue(rootModel, itemTag)
                    if self.ntab == 1:
                        self.capitalize(view, child0)
                        if rootModel is self.beamLineModel:
                            child0.setFlags(qt.ItemFlags(
                                qt.ItemIsEnabled | qt.ItemIsEditable |
                                qt.ItemIsSelectable | qt.ItemIsDragEnabled))
                elif itemType == "prop":
                    child0 = self.addProp(rootModel, itemTag)
                elif itemType == "object":
                    child0, child1 = self.addObject(view,
                                                    rootModel,
                                                    itemText)
                elif itemType == "param":
                    child0, child1 = self.addParam(rootModel,
                                                   itemTag,
                                                   itemText)
                self.iterateImport(view, child0, childImport)
            self.ntab -= 1
        else:
            pass

    def checkDefaults(self, obj, item):
        if item.hasChildren():
            paraProp = -1
            for ii in range(item.rowCount()):
                if item.child(ii, 0).text() == '_object':
                    obj = item.child(ii, 1).text()
                    if item.parent() is not None and\
                            item.model() != self.plotModel:
                        neighbour = item.parent().child(item.row(), 1)
                        if item.isEditable():  # Beamline Element or Material
                            neighbour.setText(self.objToInstance(obj))
                        else:  # Beamline Element Method
                            item.setText(item.text()+'()')
                            self.setIItalic(item)
                        neighbour.setFlags(self.objectFlag)
                    elif item.model() == self.materialsModel:
                        neighbour = self.rootMatItem.child(item.row(), 1)
                        neighbour.setText(self.objToInstance(obj))
                        neighbour.setFlags(self.objectFlag)
                if item.child(ii, 0).text() in ['properties', 'parameters']:
                    paraProp = ii
            if paraProp >= 0:
                for ii in range(item.rowCount()):
                    self.checkDefaults(obj, item.child(ii, 0))
            else:
                if obj is not None and str(item.text()) != 'output':
                    loadedParams = []
                    for ii in range(item.rowCount()):
                        loadedParams.extend([[item.child(ii, 0).text(),
                                             item.child(ii, 1).text(),
                                             item.child(ii, 0).hasChildren()]])
                    counter = -1
                    for argName, argVal in self.getParams(obj):
                        if counter < len(loadedParams) - 1:
                            counter += 1
                            if counter < len(loadedParams) - 1 and\
                                    str(loadedParams[counter][0]) == '_object':
                                counter += 1
                            child0 = item.child(counter, 0)
                            child1 = item.child(counter, 1)
                            if str(argName) == str(loadedParams[counter][0]):
                                if str(argVal) != str(
                                        loadedParams[counter][1]) and\
                                        not loadedParams[counter][2]:
                                    self.setIFontColor(child0,
                                                       qt.blue)
                            else:
                                for ix in range(len(loadedParams)):
                                    if str(argName) == str(
                                            loadedParams[ix][0]):
                                        if str(argVal) != str(
                                                loadedParams[ix][1]) and\
                                                not loadedParams[ix][2]:
                                            argVal = loadedParams[ix][1]
                                            self.setIFontColor(
                                                child0,
                                                qt.blue)
                                        break
                                child0.setText(str(argName))
                                child0.setFlags(self.paramFlag)
                                child1.setText(str(argVal))
                                child1.setFlags(self.valueFlag)
                        else:
                            counter += 1
                            child0, child1 = self.addParam(item,
                                                           argName,
                                                           argVal)
                        if str(child0.text()) == 'beam' and\
                                str(child1.text()) == 'None':
                            if child0.model() == self.beamLineModel:
                                self.blColorCounter += 1
                            elif child0.model() == self.plotModel:
                                self.pltColorCounter += 1
                    if item.rowCount() > counter + 1:
                        item.removeRows(counter + 1, item.rowCount()-counter-1)
                for ii in range(item.rowCount()):
                    if item.child(ii, 0).hasChildren():
                        self.checkDefaults(obj, item.child(ii, 0))
        else:
            pass

    def flattenElement(self, view, item):
        if item.hasChildren():
            for ii in range(item.rowCount()):
                iItem = item.child(ii, 0)
                if item.child(ii, 1) is not None and view is not None:
                    iWidget = view.indexWidget(item.child(ii, 1).index())
                    if iWidget is not None:
                        if iWidget.staticMetaObject.className() == 'QComboBox':  # analysis:ignore
                            if str(iItem.text()) == "targetOpenCL":
                                chItemText = self.OCLModel.item(
                                            iWidget.currentIndex(), 1).text()
                            else:
                                chItemText = iWidget.currentText()
                            if str(item.child(ii, 1).text()) !=\
                                    str(chItemText):
                                item.child(ii, 1).setText(chItemText)
                        elif iWidget.staticMetaObject.className() ==\
                                'QListWidget':
                            chItemText = "("
                            for rState in range(iWidget.count()):
                                if int(iWidget.item(rState).checkState()) == 2:
                                    chItemText += str(rState+1) + ","
                            else:
                                chItemText += ")"
                            item.child(ii, 1).setText(chItemText)
                self.flattenElement(view, iItem)
        else:
            pass

    def updateBeamImport(self):
        outBeams = ['None']
        self.rootBeamItem.setChild(
            0, 1,
            qt.QStandardItem("beamGlobalLocal"))
        self.rootBeamItem.setChild(
            0, 2,
            qt.QStandardItem("None"))
        self.rootBeamItem.setChild(
            0, 3,
            qt.QStandardItem('000'))
        for ibl in range(self.rootBLItem.rowCount()):
            elItem = self.rootBLItem.child(ibl, 0)
            elNameStr = str(elItem.text())
            eluuid = elItem.data(qt.UserRole)
#            print(elNameStr, eluuid)
            if elNameStr not in ['properties', '_object']:
                for iel in range(elItem.rowCount()):
                    if elItem.child(iel, 0) not in ['properties', '_object']:
                        for imet in range(elItem.child(iel, 0).rowCount()):
                            metChItem = elItem.child(iel, 0).child(imet, 0)
                            itemTxt = str(metChItem.text())
                            if itemTxt == 'output':
                                for ii in range(metChItem.rowCount()):
                                    child0 = metChItem.child(ii, 0)
                                    child1 = metChItem.child(ii, 1)
#                                    print(child0.text(), child1.text())
                                    if child1 is not None:
                                        outBeams.append(str(child1.text()))
                                    for irow in range(
                                            self.rootBeamItem.rowCount()):
                                        if str(child1.text()) ==\
                                                str(self.rootBeamItem.child(
                                                irow, 0).text()):
                                            self.rootBeamItem.setChild(
                                                irow, 1,
                                                qt.QStandardItem(child0.text()))  # analysis:ignore
                                            self.rootBeamItem.setChild(
                                                irow, 2,
                                                qt.QStandardItem(eluuid))
#                                                qt.QStandardItem(elNameStr))
                                            self.rootBeamItem.setChild(
                                                irow, 3,
                                                qt.QStandardItem(str(
                                                    self.nameToBLPos(
                                                        eluuid))))
#                                                        elNameStr))))
        for ibm in reversed(range(self.beamModel.rowCount())):
            beamName = str(self.beamModel.item(ibm, 0).text())
            if beamName not in outBeams:
                self.beamModel.takeRow(ibm)

    def intToRegexp(self, intStr):
        a = list(str(int(intStr)))
        oeClassStr = str(self.getClassName(
            self.rootBLItem.child(int(intStr), 0)))
        if 'source' in oeClassStr:
            return r'^(\d+)$'
        elif int(intStr) < 11:
            return '^([0][0][0-{}])$'.format(int(intStr)-1)
        else:
            return '^([0][0][0-9]|[0][0-{0}][0-9]{1}$'.format(
                int(a[0])-1, '|[0]{0}[0-{1}])'.format(
                    int(a[0]), int(a[1])-1) if int(a[1]) > 0 else ")")

    def addCombo(self, view, item):
        return
        self.beamModel.sort(3)
        if item.hasChildren():
            itemTxt = str(item.text())
            for ii in range(item.rowCount()):
                child0 = item.child(ii, 0)
                if child0 is None:
                    continue
                child1 = item.child(ii, 1)
                if str(child0.text()) == '_object':
                    view.setRowHidden(child0.index().row(), item.index(), True)
                if child1 is not None and not child0.isEditable():
                    value = child1.text()
                    paramName = str(child0.text())
                    iWidget = view.indexWidget(child1.index())
                    if iWidget is None:
                        combo = None
                        if isinstance(self.getVal(value), bool):
                            combo = self.addStandardCombo(
                                self.boolModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        if len(re.findall("beam", paramName.lower())) > 0\
                                and paramName.lower() != 'beamline'\
                                and paramName.lower() != 'filamentbeam':
                            if item.text() == 'parameters':  # input beam
                                combo = qt.QComboBox()
                                fModel0 = qt.QSortFilterProxyModel()
                                fModel0.setSourceModel(self.beamModel)
                                fModel0.setFilterKeyColumn(1)
                                fModel0.setFilterRegExp('Global')
                                fModel = qt.QSortFilterProxyModel()
                                fModel.setSourceModel(fModel0)
                                fModel.setFilterKeyColumn(3)
                                regexp = self.intToRegexp(
                                    self.nameToBLPos(item.parent(
                                    ).parent().data(qt.UserRole)))
#                                    self.nameToBLPos(str(item.parent(
#                                    ).parent().text())))
                                fModel.setFilterRegExp(regexp)
                                fModel.setDynamicSortFilter(True)
                            elif item.text() == 'output':  # output beam
                                fModel0 = qt.QSortFilterProxyModel()
                                fModel0.setSourceModel(self.beamModel)
                                fModel0.setFilterKeyColumn(1)
                                fModel0.setFilterRegExp(paramName)
                                fModel = qt.QSortFilterProxyModel()
                                fModel.setSourceModel(fModel0)
                                fModel.setFilterKeyColumn(3)
                                fModel.setFilterRegExp(
                                    self.nameToBLPos(item.parent(
                                    ).parent().data(qt.UserRole)))
#                                        self.nameToBLPos(str(
#                                    item.parent().parent().text())))
                                fModel.setDynamicSortFilter(True)
                            else:
                                fModel = self.beamModel
                            combo = self.addStandardCombo(fModel, value)
                            if combo.currentIndex() == -1:
                                combo.setCurrentIndex(0)
                                child1.setText(combo.currentText())
#                            lastIndex = combo.model().rowCount() - 1
#                            if paramName.lower() == 'accubeam':
#                                lastIndex = 0
#                            combo.setCurrentIndex(lastIndex)
#                            child1.setText(combo.currentText())
                            view.setIndexWidget(child1.index(), combo)
                            self.colorizeChangedParam(child1)
                            if itemTxt.lower() == "output":
                                combo.setEditable(False)
#                                combo.setEditable(True)
                                # combo.setInsertPolicy(2)
                                combo.setInsertPolicy(
                                    qt.QComboBox.InsertAtCurrent)
                        elif len(re.findall("wave", paramName.lower())) > 0:
                            if item.text() == 'parameters':  # input beam
                                combo = qt.QComboBox()
                                fModel0 = qt.QSortFilterProxyModel()
                                fModel0.setSourceModel(self.beamModel)
                                fModel0.setFilterKeyColumn(1)
                                fModel0.setFilterRegExp('Local')
                                fModel = qt.QSortFilterProxyModel()
                                fModel.setSourceModel(fModel0)
                                fModel.setFilterKeyColumn(3)
                                regexp = self.intToRegexp(
                                    self.nameToBLPos(item.parent(
                                    ).parent().data(qt.UserRole)))
#                                    self.nameToBLPos(str(item.parent(
#                                    ).parent().text())))
                                fModel.setFilterRegExp(regexp)
                                fModel.setDynamicSortFilter(True)
                            else:
                                fModel = self.beamModel
                            combo = self.addStandardCombo(fModel, value)
                            if combo.currentIndex() == -1:
                                combo.setCurrentIndex(0)
                                child1.setText(combo.currentText())
#                            lastIndex = combo.model().rowCount() - 1
#                            if paramName.lower() == 'accubeam':
#                                lastIndex = 0
#                            combo.setCurrentIndex(lastIndex)
#                            child1.setText(combo.currentText())
                            view.setIndexWidget(child1.index(), combo)
                            self.colorizeChangedParam(child1)
                            if itemTxt.lower() == "output":
#                                combo.setEditable(True)
                                combo.setEditable(False)
                                # combo.setInsertPolicy(2)
                                combo.setInsertPolicy(
                                    qt.QComboBox.InsertAtCurrent)

                        elif paramName == "bl" or\
                                paramName == "beamLine":
                            combo = self.addEditableCombo(
                                self.beamLineModel, value)
                            # combo.setInsertPolicy(2)
                            combo.setInsertPolicy(qt.QComboBox.InsertAtCurrent)
                            view.setIndexWidget(child1.index(), combo)
                        elif paramName == "plots":
                            combo = self.addEditableCombo(
                                self.plotModel, value)
                            # combo.setInsertPolicy(2)
                            combo.setInsertPolicy(qt.QComboBox.InsertAtCurrent)
                            view.setIndexWidget(child1.index(), combo)
                        elif any(paraStr in paramName.lower() for paraStr in
                                 ['material', 'tlayer', 'blayer', 'coating',
                                  'substrate']):
                            combo = self.addStandardCombo(
                                self.materialsModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall(
                                "density", paramName)) > 0 and\
                                paramName != 'uniformRayDensity':
                            combo = self.addStandardCombo(
                                self.densityModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("polarization",
                                            paramName.lower())) > 0:
                            combo = self.addEditableCombo(
                                self.polarizationsModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("shape",
                                            paramName.lower())) > 0:
                            combo = self.addEditableCombo(
                                self.shapeModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("table",
                                            paramName.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.matTableModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("data",
                                 paramName.lower())) > 0 and\
                                len(re.findall("axis",
                                               itemTxt.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.fluxDataModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("label",
                                 paramName.lower())) > 0 and\
                                len(re.findall("axis",
                                               itemTxt.lower())) > 0:
                            if value == '':
                                if itemTxt.lower() == "xaxis":
                                    value = "x"
                                elif itemTxt.lower() == "yaxis":
                                    value = "z"
                                else:
                                    value = "energy"
                            combo = self.addEditableCombo(
                                self.plotAxisModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("fluxkind",
                                            paramName.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.fluxKindModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif view == self.matTree and len(
                                re.findall("kind", paramName.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.matKindModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("distE",
                                            paramName)) > 0:
                            combo = qt.QComboBox()
                            for icd in range(item.parent().rowCount()):
                                if item.parent().child(icd,
                                                       0).text() == '_object':
                                    if len(re.findall('Source',
                                           item.parent().child(
                                               icd, 1).text())) > 0:
                                        combo.setModel(self.distEModelG)
                                    else:
                                        combo.setModel(self.distEModelS)
                            combo.setCurrentIndex(combo.findText(value))
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("geom", paramName.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.matGeomModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("aspect",
                                            paramName.lower())) > 0:
                            combo = self.addEditableCombo(
                                self.aspectModel, value)
                            view.setIndexWidget(child1.index(), combo)
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
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("targetopencl",
                                            paramName.lower())) > 0:
                            combo = qt.QComboBox()
                            combo.setModel(self.OCLModel)
                            oclInd = self.OCLModel.findItems(
                                value, flags=qt.MatchExactly, column=1)
                            if len(oclInd) > 0:
                                oclInd = oclInd[0].row()
                            else:
                                oclInd = 1
                            combo.setCurrentIndex(oclInd)
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("precisionopencl",
                                            paramName.lower())) > 0:
                            combo = self.addStandardCombo(
                                self.oclPrecnModel, value)
                            view.setIndexWidget(child1.index(), combo)
                        if combo is not None:
                            if combo.staticMetaObject.className() ==\
                                    'QListWidget':
                                combo.clicked.connect(
                                    partial(
                                        self.processListWidget, combo, child1))
                            elif combo.staticMetaObject.className() ==\
                                    'QComboBox':
                                combo.currentIndexChanged['QString'].connect(
                                    child1.setText)
                self.addCombo(view, child0)
        else:
            pass

    def processListWidget(self, combo, item):
        chItemText = "("
        for rState in range(combo.count()):
            if int(combo.item(rState).checkState()) == 2:
                chItemText += str(rState+1) + ","
        else:
            chItemText += ")"
        item.setText(chItemText)

    def addEditableCombo(self, model, value):
        combo = qt.QComboBox()
        combo.setModel(model)
        if combo.findText(value) < 0:
            newItem = qt.QStandardItem(value)
            model.appendRow(newItem)
        combo.setCurrentIndex(combo.findText(value))
        combo.setEditable(True)
        # combo.setInsertPolicy(1)
        combo.setInsertPolicy(qt.QComboBox.InsertAtTop)
        return combo

    def addStandardCombo(self, model, value):
        combo = qt.QComboBox()
        combo.setModel(model)
        combo.setCurrentIndex(combo.findText(value))
        return combo

    def autoAssignMethod(self, elItem, methProps=None):
        elstr = self.getClassName(elItem)
        elcls = eval(elstr)
        if hasattr(elcls, 'hiddenMethods'):
            hmList = elcls.hiddenMethods
        else:
            hmList = []
        for namef, objf in inspect.getmembers(elcls):
            if (inspect.ismethod(objf) or inspect.isfunction(objf)) and\
                    not str(namef).startswith("_") and\
                    not str(namef) in hmList:
                fdoc = objf.__doc__
                if fdoc is not None:
                    objfNm = '{0}.{1}'.format(elstr,
                                              objf.__name__)
                    fdoc = re.findall(r"Returned values:.*", fdoc)
                    if len(fdoc) > 0 and (
                            str(objf.__name__) not in
                            self.experimentalModeFilter or
                            self.experimentalMode):
#                        print("fdoc:", fdoc)
                        outBeams = fdoc[0].replace(
                                "Returned values: ", '').split(',')
#                        print("outbeams:", outBeams)
#                        print(objfNm)
                        self.addMethod(objfNm, elItem, outBeams, methProps)
                        break

    def openMenu(self, position):
        indexes = self.tree.selectedIndexes()

        level = 100
        if len(indexes) > 0:
            level = 0
            selIndex = indexes[0]
            index = indexes[0]
            selectedItem = self.beamLineModel.itemFromIndex(selIndex)
            selText = selectedItem.text()

            while index.parent().isValid():
                index = index.parent()
                level += 1

        menu = qt.QMenu()
        if level == 0 or level == 100:
            menu.addAction("Load Layout", self.importLayout)
            menu.addAction("Save Layout", self.exportLayout)
            menu.addSeparator()
            menusrc = menu.addMenu(self.tr("Add Source"))
            menuoe = menu.addMenu(self.tr("Add OE"))
            menuapt = menu.addMenu(self.tr("Add Aperture"))
            menuscr = menu.addMenu(self.tr("Add Screen"))
            for tsubmenu, tmodule in zip([menusrc, menuoe, menuapt, menuscr],
                                         [rsources, roes, rapts, rscreens]):
                if hasattr(tmodule, '__allSectioned__'):
                    for sec, elnames in list(tmodule.__allSectioned__.items()):
                        if isinstance(elnames, (tuple, list)):
                            smenu = tsubmenu.addMenu(sec)
                            for elname in elnames:
                                self._addAction(
                                    tmodule, elname, self.addElement, smenu)
                        else:  # as single entry itself
                            self._addAction(
                                tmodule, sec, self.addElement, tsubmenu)
                else:  # only with __all__
                    for elname in tmodule.__all__:
                        self._addAction(
                            tmodule, elname, self.addElement, tsubmenu)
        elif level == 1 and selText != "properties":
            tsubmenu = menu.addMenu(self.tr("Add method"))
            menu.addSeparator()
            menu.addAction("Duplicate " + str(selectedItem.text()),
                           partial(self.addElement, copyFrom=selectedItem))
            menu.addSeparator()
            if selIndex.row() > 2:
                menu.addAction("Move Up", partial(self.moveItem, -1,
                                                  self.tree,
                                                  selectedItem))
            if selIndex.row() < selectedItem.parent().rowCount()-1:
                menu.addAction("Move Down", partial(self.moveItem, 1,
                                                    self.tree,
                                                    selectedItem))
            menu.addSeparator()
            deleteActionName = "Remove " + str(selectedItem.text())
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.tree,
                                                     selectedItem))

            for ic in range(selectedItem.rowCount()):
                if selectedItem.child(ic, 0).text() == "_object":
                    elstr = str(selectedItem.child(ic, 1).text())
                    break
            elcls = eval(elstr)
            if hasattr(elcls, 'hiddenMethods'):
                hmList = elcls.hiddenMethods
            else:
                hmList = []
            for namef, objf in inspect.getmembers(elcls):
                if (inspect.ismethod(objf) or inspect.isfunction(objf)) and\
                        not str(namef).startswith("_") and\
                        not str(namef) in hmList:
                    fdoc = objf.__doc__
                    if fdoc is not None:
                        objfNm = '{0}.{1}'.format(elstr,
                                                  objf.__name__)
                        fdoc = re.findall(r"Returned values:.*", fdoc)
                        if len(fdoc) > 0 and (
                                str(objf.__name__) not in
                                self.experimentalModeFilter or
                                self.experimentalMode):
                            methAction = qt.QAction(self)
                            methAction.setText(namef + '()')
                            methAction.hovered.connect(
                                partial(self.showObjHelp, objfNm))
                            outBeams = fdoc[0].replace(
                                    "Returned values: ", '').split(',')
                            methAction.triggered.connect(
                                partial(self.addMethod, objfNm,
                                        selectedItem, outBeams))
                            tsubmenu.addAction(methAction)
        elif level == 2 and selText != "properties":
            deleteActionName = "Remove " + str(selText)
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.tree,
                                                     selectedItem))
        elif level == 4:
            selParent = selectedItem.parent()
            selRow = selectedItem.row()
            child0 = selParent.child(selRow, 0)
            child1 = selParent.child(selRow, 1)
            if str(child0.text()).lower().startswith('beam') and\
                    str(selParent.text()).lower().startswith('output') and\
                    not str(child1.text()).lower().startswith('none') and\
                    self.tree.indexWidget(child1.index()) is not None:
                menu.addAction(self.tr("Plot " + child1.text()),
                               partial(self.addPlotBeam, child1.text()))

        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def plotMenu(self, position):
        indexes = self.plotTree.selectedIndexes()
        level = 100
        if len(indexes) > 0:
            level = 0
            index = indexes[0]
            selectedItem = self.plotModel.itemFromIndex(index)
            while index.parent().isValid():
                index = index.parent()
                level += 1

        menu = qt.QMenu()

        if level == 0 or level == 100:
            menu.addAction(self.tr("Add Plot"), self.addPlot)
        elif level == 1:
            copyActionName = "Duplicate " + str(selectedItem.text())
            menu.addAction(copyActionName, partial(self.addPlot,
                                                   selectedItem))
            deleteActionName = "Remove " + str(selectedItem.text())
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.plotTree,
                                                     selectedItem))
        else:
            pass

        menu.exec_(self.plotTree.viewport().mapToGlobal(position))

    def matMenu(self, position):
        indexes = self.matTree.selectedIndexes()
        level = 100
        if len(indexes) > 0:
            level = 0
            index = indexes[0]
            selectedItem = self.materialsModel.itemFromIndex(index)
            while index.parent().isValid():
                index = index.parent()
                level += 1

        menu = qt.QMenu()

        matMenu = menu.addMenu(self.tr("Add Material"))
        if hasattr(rmats, '__allSectioned__'):
            for sec, mNames in list(rmats.__allSectioned__.items()):
                if isinstance(mNames, (tuple, list)):
                    smenu = matMenu.addMenu(sec)
                    for mName in mNames:
                        self._addAction(rmats, mName, self.addElement, smenu)
                else:  # as single entry itself
                    self._addAction(rmats, sec, self.addElement, matMenu)
            pdmmenu = matMenu.addMenu(self.tr("Predefined"))
            for mlibname, mlib in zip(['Elemental', 'Compounds', 'Crystals'],
                                      [rmatsel, rmatsco, rmatscr]):
                pdflmenu = pdmmenu.addMenu(self.tr(mlibname))
                for ssec, snames in list(mlib.__allSectioned__.items()):
                    if isinstance(snames, (tuple, list)):
                        pdfsmenu = pdflmenu.addMenu(ssec)
                        for sname in snames:
                            self._addAction(mlib, sname, self.addElement,
                                            pdfsmenu)
                    else:
                        self._addAction(mlib, ssec, self.addElement, pdflmenu)
        else:  # only with __all__
            for mName in rmats.__all__:
                self._addAction(rmats, mName, self.addElement, matMenu)

        if level == 0 and selectedItem.text() != "None":
            menu.addSeparator()
            deleteActionName = "Remove " + str(selectedItem.text())
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.matTree,
                                                     selectedItem))
        else:
            pass

        menu.exec_(self.matTree.viewport().mapToGlobal(position))

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
            if 'np.' not in value and\
                    (str(self.rootBLItem.text())+'.') not in value:
                value = 'r\"{}\"'.format(value)
#        if str(value) == 'round':
#            value = 'r\"{}\"'.format(value)
        if isinstance(value, tuple):
            value = list(value)
        return str(value)

    def quotizeAll(self, value):
        return str('r\"{}\"'.format(value))

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
        if isinstance(value, tuple):
            value = list(value)
#        elif value in self.beamLine.materialsDict.keys():
#            value = self.beamLine.materialsDict[value]
        return value

    def getClassName(self, itemObject):
        for iel in range(itemObject.rowCount()):
            if itemObject.child(iel, 0).text() == '_object':
                return str(itemObject.child(iel, 1).text())
        return None

    def nameToFlowPos(self, elementNameStr):
        retVal = 0
        try:
            for isegment, segment in enumerate(self.beamLine.flow):
                if segment[0] == elementNameStr:
                    retVal = isegment
                    break
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                pass
        return retVal

    def nameToBLPos(self, eluuid):
        for iel in range(self.rootBLItem.rowCount()):
            if str(self.rootBLItem.child(iel, 0).data(qt.UserRole)) == str(eluuid):
                return '{:03d}'.format(iel)
        else:
            return '000'

    def updateBeamModel(self):
        """This function cleans the beam model. It will do nothing if
        move/delete OE procedures perform correctly."""
        outBeams = ['None']
        for ie in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ie, 0).text() != "properties" and\
                    self.rootBLItem.child(ie, 0).text() != "_object":
                tItem = self.rootBLItem.child(ie, 0)
                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() != '_object' and\
                            tItem.child(ieph, 0).text() != 'properties':
                        pItem = tItem.child(ieph, 0)
                        for imet in range(pItem.rowCount()):
                            if pItem.child(imet, 0).text() == 'output':
                                mItem = pItem.child(imet, 0)
                                for iep in range(mItem.rowCount()):
                                    outvalue = mItem.child(iep, 1).text()
                                    outBeams.append(str(outvalue))
        for ibm in reversed(range(self.beamModel.rowCount())):
            beamName = str(self.beamModel.item(ibm, 0).text())
            if beamName not in outBeams:
                self.beamModel.takeRow(ibm)

    def updateBeamlineBeams(self, item=None):
        # sender = self.sender()
        # if sender is not None:
        if True:
            if item is None:  # Create empty beam dict
                beamsDict = OrderedDict()
                for ib in range(self.rootBeamItem.rowCount()):
                    beamsDict[str(
                        self.rootBeamItem.child(ib, 0).text())] = None
                self.beamLine.beamsDict = beamsDict
#            elif sender.staticMetaObject.className() == 'QComboBox':
#                currentIndex = int(sender.currentIndex())
#                beamValues = list(self.beamLine.beamsDict.values())
#                beamKeys = list(self.beamLine.beamsDict.keys())
#                beamKeys[currentIndex] = item.text()
#                self.beamLine.beamsDict = OrderedDict(
#                    zip(beamKeys, beamValues))
            else:  # Beam renamed
                bList = []
                for ib in range(self.rootBeamItem.rowCount()):
                    bList.append(self.rootBeamItem.child(ib, 0).text())

                for key, value in self.beamLine.beamsDict.items():
                    if key not in bList:
                        del self.beamLine.beamsDict[key]
                        self.beamLine.beamsDict[str(item.text())] = value
                        break

    def updateBeamlineMaterials(self, item=None, newElement=None):

        kwargs = {}
        if item is None or (item.column() == 0 and newElement is None):
            return

        if item.column() == 1:
            matItem = item.parent().parent()
        else:
            matItem = item

        objStr = None
        matId = str(matItem.data(qt.UserRole))
        paintItem = self.rootMatItem.child(matItem.row(), 1)
#        print(item.column(), item.text(),  item.parent())

        if item.column() == 1 and item.text() == matItem.text():  # renaming existing
            self.beamLine.materialsDict[matId].name = item.text()
            return
            
        
        for itop in range(matItem.rowCount()):
            chitem = matItem.child(itop, 0)
            if chitem.text() in ['properties']:
                for iprop in range(chitem.rowCount()):
                    argName = chitem.child(iprop, 0).text()
                    argValue = raycing.parametrize(
                            chitem.child(iprop, 1).text())
                    kwargs[str(argName)] = argValue
            elif chitem.text() == '_object':
                objStr = str(matItem.child(itop, 1).text())
        kwargs['uuid'] = matId
        outDict = {'properties': kwargs, '_object': objStr}
        initStatus = 0
        try:
            initStatus = self.beamLine.init_material_from_json(matId, outDict)
        except:
            raise
        
        self.paintStatus(paintItem, initStatus)

    def paintStatus(self, item, status):
        updateStatus = item.model().signalsBlocked()
        if not updateStatus:
            item.model().blockSignals(True)
        if status:
            color = qt.QColor(255, 200, 200)  # pale red
        else:
            color = qt.QColor(200, 255, 200)  # pale green
        item.setBackground(qt.QBrush(color))
        item.model().blockSignals(updateStatus)

    def updateBeamline(self, item=None, newElement=None, newOrder=False):
        def beamToUuid(beamName):
            for ib in range(self.beamModel.rowCount()):
                if self.beamModel.item(ib, 0).text() == beamName:
                    return self.beamModel.item(ib, 2).text()

        oeid = None
        argName = 'None'
        argValue = 'None'
        argValue_str = ''
        kwargs = {}
        outDict = {}

        if item is not None:
            iindex = item.index()
            column = iindex.column()
            row = iindex.row()
            parent = item.parent()

            if parent is None:
                print("No parent")
                return
            else:
                print(str(parent.text()))  # TODO: print

            if str(parent.text()) in ['properties']:
                oeItem = parent.parent()
                oeid = str(oeItem.data(qt.UserRole))
            elif str(parent.text()) in ['parameters']:
                methItem = parent.parent()
                oeItem = methItem.parent()
                oeid = str(oeItem.data(qt.UserRole))
                methObjStr = methItem.text().strip('()')
                outDict = {'_object': methObjStr,
                           'parameters': kwargs}
            elif raycing.is_valid_uuid(item.data(qt.UserRole)):
                oeid = str(item.data(qt.UserRole))

            if column == 1:  # Existing Element
                argValue_str = item.text()
                argName = parent.child(row, 0).text()
                if any(argName.lower().startswith(v) for v in
                       ['mater', 'tlay', 'blay', 'coat', 'substrate']):
                    argValue = self.beamLine.matnamesToUUIDs.get(argValue_str)
                else:
                    argValue = raycing.parametrize(argValue_str)
                kwargs[argName] = argValue
                outDict = kwargs

            elif column == 0 and newElement is not None:  # New Element
                if raycing.is_valid_uuid(parent.data(qt.UserRole)):
                    oeid = str(parent.data(qt.UserRole))
                    methKWArgs = OrderedDict()
                    outKWArgs = OrderedDict()
                    methObjStr = ''
                    for mch in range(item.rowCount()):
                        mchi = item.child(mch, 0)
                        if mchi.text() == 'parameters':
                            for mchpi in range(mchi.rowCount()):
                                argName = mchi.child(mchpi, 0).text()
                                argValue = mchi.child(mchpi, 1).text()
                                if argName == 'beam':
                                    argValue = beamToUuid(argValue)
                                else:
                                    argValue = raycing.parametrize(
                                        argValue)
                                methKWArgs[str(argName)] = argValue
                        elif mchi.text() == 'output':
                            for mchpi in range(mchi.rowCount()):
                                argName = mchi.child(mchpi, 0).text()
                                argValue = mchi.child(mchpi, 1).text()
                                if argName == 'beam':
                                    argValue = beamToUuid(argValue)
                                else:
                                    argValue = raycing.parametrize(
                                        argValue)
                                outKWArgs[str(argName)] = argValue
                        elif mchi.text() == '_object':
                            methObjStr = str(item.child(mch, 1).text())
                    outDict = {'_object': methObjStr,
                                'parameters': methKWArgs,
                                'output': outKWArgs}

                    methStr = methObjStr.split('.')[-1]
                    self.beamLine.update_flow_from_json(oeid,
                                                        {methStr: outDict})
                    self.beamLine.sort_flow()
#                    print("flow U", self.beamLine.flowU)

                else:
                    for itop in range(item.rowCount()):
                        chitem = item.child(itop, 0)
                        if chitem.text() in ['properties']:
                            for iprop in range(chitem.rowCount()):
                                argName = chitem.child(iprop, 0).text()
                                argValue = raycing.parametrize(
                                        chitem.child(iprop, 1).text())
                                kwargs[str(argName)] = argValue
                        elif chitem.text() == '_object':
                            continue
                    kwargs['uuid'] = oeid
                    outDict = {'properties': kwargs, '_object': newElement}
                    initStatus = self.beamLine.init_oe_from_json(outDict)

                    paintItem = item.parent().child(item.row(), 1)
                    self.paintStatus(paintItem, initStatus)

            if self.blViewer is None or not outDict:
                return

            self.blViewer.customGlWidget.update_beamline(oeid, outDict)

#            print(item.text(), row, column, parent.text())

#        self.blRunGlow()
#        def createParamDict(parentItem, elementString):
#            kwargs = dict()
#            for iep, arg_def in zip(range(
#                    parentItem.rowCount()),
#                    list(zip(*self.getParams(elementString)))[1]):
#                paraname = str(parentItem.child(iep, 0).text())
#                paravalue = str(parentItem.child(iep, 1).text())
#                if paravalue != str(arg_def) or\
#                        paraname == 'bl':
#                    if paraname == 'center':
#                        paravalue = paravalue.strip('[]() ')
#                        paravalue =\
#                            [self.getVal(c.strip())
#                             for c in str.split(
#                             paravalue, ',')]
#                    elif paraname.startswith('material'):
#                        paravalue =\
#                            self.beamLine.materialsDict[paravalue]
#                    elif paraname == 'bl':
#                        paravalue = self.beamLine
#                    else:
#                        paravalue = self.parametrize(paravalue)
#                    kwargs[paraname] = paravalue
#            return kwargs
#
#        def createMethodDict(elementItem, elementString):
#            methodObj = None
#            for ieph in range(elementItem.rowCount()):
#                pItem = elementItem.child(ieph, 0)
#                methodObj = None
#                if str(pItem.text()) not in ['_object', 'properties']:
#                    for namef, objf in inspect.getmembers(eval(elementString)):
#                        if (inspect.ismethod(objf) or
#                                inspect.isfunction(objf)) and\
#                                namef == str(pItem.text()).strip('()'):
#                            methodObj = objf
#                    inkwargs = {}
#                    outkwargs = OrderedDict()
#                    for imet in range(pItem.rowCount()):
#                        mItem = pItem.child(imet, 0)
#                        if str(mItem.text()) == 'parameters':
#                            for iep, arg_def in\
#                                zip(range(mItem.rowCount()),
#                                    getargspec(methodObj)[3]):
#                                paraname = str(mItem.child(
#                                    iep, 0).text())
#                                paravalue = self.parametrize(str(mItem.child(
#                                    iep, 1).text()))
#                                if len(re.findall('beam', paraname)) > 0 and\
#                                    self.beamLine.oesDict[str(elementItem.text(
#                                        ))][1] and paravalue is None:
#                                    return None, None, None
#                                inkwargs[paraname] = paravalue
#                        elif str(mItem.text()) == 'output':
#                            for iep in range(mItem.rowCount()):
#                                paraname = str(mItem.child(
#                                    iep, 0).text())
#                                paravalue = str(mItem.child(iep, 1).text())
#                                outkwargs[paraname] = paravalue
#            if methodObj is not None:
#                return methodObj, inkwargs, outkwargs
#            else:
#                return None, None, None
#
#        def buildFlow(startFrom=0):
#            blFlow = []
#            for ie in range(self.rootBLItem.rowCount()):
#                elItem = self.rootBLItem.child(ie, 0)
#                elName = str(elItem.text())
#                if elName not in ["properties", "_object"]:
#                    elStr = self.getClassName(elItem)
#                    methodObj, inkwArgs, outkwArgs = createMethodDict(
#                        elItem, elStr)
#                    if methodObj is not None:
#                        blFlow.append([elName, methodObj,
#                                       inkwArgs, outkwArgs])
#            return blFlow
#
#        def updateRegexp():
#            for iElement in range(self.rootBLItem.rowCount()):
#                elItem = self.rootBLItem.child(iElement, 0)
#                if str(elItem.text()) not in ['properties', '_object']:
#                    for iProp in range(elItem.rowCount()):
#                        propItem = elItem.child(iProp, 0)
#                        if str(propItem.text()) not in ['properties',
#                                                        '_object']:
#                            for iMeth in range(propItem.rowCount()):
#                                methItem = propItem.child(iMeth, 0)
#                                if str(methItem.text()) == 'parameters':
#                                    for iBeam in range(methItem.rowCount()):
#                                        bItem = methItem.child(iBeam, 0)
#                                        if len(re.findall(
#                                                'beam', str(
#                                                bItem.text()))) > 0:
#                                            vItem = methItem.child(iBeam, 1)
#                                            iWidget = self.tree.indexWidget(
#                                                vItem.index())
#                                            if iWidget is not None:
#                                                try:
#                                                    regexp =\
#                                                        self.intToRegexp(
#                                                            iElement)
#                                                    iWidget.model(
#                                                        ).setFilterRegExp(
#                                                            regexp)
#                                                except:  # analysis:ignore
#                                                    if _DEBUG_:
#                                                        raise
#                                                    else:
#                                                        continue
#                                elif str(methItem.text()) == 'output':
#                                    for iBeam in range(methItem.rowCount()):
#                                        bItem = methItem.child(iBeam, 0)
#                                        if len(re.findall(
#                                                'beam', str(
#                                                bItem.text()))) > 0:
#                                            vItem = methItem.child(iBeam, 1)
#                                            iWidget = self.tree.indexWidget(
#                                                vItem.index())
#                                            if iWidget is not None:
#                                                try:
#                                                    regexp = '{:03d}'.format(
#                                                        iElement)
#                                                    iWidget.model(
#                                                        ).setFilterRegExp(
#                                                            regexp)
#                                                except:  # analysis:ignore
#                                                    if _DEBUG_:
#                                                        raise
#                                                    else:
#                                                        continue
#        self.rootBLItem.model().blockSignals(True)
#        self.flattenElement(self.tree,
#                            self.rootBLItem if item is None else item)
#        self.rootBLItem.model().blockSignals(False)
#
#        if item is not None:
#            if item.index().parent().isValid():  # not the Beamline root
#                iCol = item.index().column()
#                pText = str(item.parent().text())
#                if pText == str(self.rootBLItem.text()):
#                    if newElement:  # New element added
#                        elNameStr = str(item.text())
#                        elClassStr = self.getClassName(item)
#                        for iep in range(item.rowCount()):
#                            if item.child(iep, 0).text() == 'properties':
#                                propItem = item.child(iep, 0)
#                                break
#                        oeType = 0 if len(re.findall(
#                            'raycing.sou', elClassStr)) > 0 else 1
#                        try:
#                            kwArgs = createParamDict(propItem, elClassStr)
#                            self.beamLine.oesDict[elNameStr] =\
#                                [eval(elClassStr)(**kwArgs), oeType]
#                            self.progressBar.setFormat(
#                                "Class {} successfully initialized.".format(
#                                    elNameStr))
#                            print("Class", elNameStr,
#                                  "successfully initialized.")
#                        except:  # analysis:ignore
#                            self.beamLine.oesDict[elNameStr] =\
#                                [None, oeType]
#                            self.progressBar.setFormat(
#                                "Incorrect parameters. Class {} not initialized.".format(  # analysis:ignore
#                                    elNameStr))
#                            print("Incorrect parameters. Class", elNameStr,
#                                  "not initialized.")
#                        self.beamLine.flow = buildFlow(startFrom=elNameStr)
#                        startFrom = self.nameToFlowPos(elNameStr)
#                    else:  # Element renamed or moved
#                        oesValues = list(self.beamLine.oesDict.values())
#                        oesKeys = list(self.beamLine.oesDict.keys())
#                        wasDeleted = True if\
#                            len(oesKeys) + 2 < self.rootBLItem.rowCount()\
#                            else False
#                        if not wasDeleted:
#                            newDict = OrderedDict()
#                            counter = 0
#                            startElement = None
#                            rbi = self.rootBeamItem
#                            for ie in range(self.rootBLItem.rowCount()):
#                                elNameStr =\
#                                    str(self.rootBLItem.child(ie, 0).text())
#                                if elNameStr not in ["properties", "_object"]:
#                                    if newOrder:
#                                        newDict[elNameStr] =\
#                                            self.beamLine.oesDict[elNameStr]
#                                        if elNameStr != oesKeys[counter] and\
#                                                startElement is None:
#                                            startElement = elNameStr
#                                            newElName = elNameStr
#                                    else:
#                                        newDict[elNameStr] = oesValues[counter]
#                                        if elNameStr != oesKeys[counter]:
#                                            startElement = oesKeys[counter]
#                                            for ibeam in range(
#                                                    rbi.rowCount()):
#                                                if str(rbi.child(
#                                                        ibeam, 2).text()) ==\
#                                                        startElement:
#                                                    rbi.child(
#                                                        ibeam, 2).setText(
#                                                            elNameStr)
#                                            self.progressBar.setFormat(
#                                                "Element {0} renamed to {1}".format(  # analysis:ignore
#                                                    startElement, elNameStr))
#                                            print("Element", startElement,
#                                                  "renamed to", elNameStr)
#                                            newElName = elNameStr
#                                    counter += 1
#                            self.beamLine.oesDict = newDict
#                        if newOrder:
#                            self.progressBar.setFormat(
#                                "Element {} moved to new position".format(
#                                    item.text()))
#                            print("Element", item.text(),
#                                  "moved to new position")
#                            for ibeam in range(rbi.rowCount()):
#                                rbi.child(ibeam, 3).setText(str(
#                                    self.nameToBLPos(str(rbi.child(
#                                        ibeam, 2).text()))))
#                            self.beamModel.sort(3)
#                            updateRegexp()
#                        elif wasDeleted:
#                            self.progressBar.setFormat(
#                                "Element {} was removed".format(item.text()))
#                            print("Element", item.text(),
#                                  "was removed")
#                            startElement = str(item.text())
#                            startFrom = self.nameToFlowPos(startElement)
#                            self.beamLine.flow =\
#                                buildFlow(startFrom=startElement)
#                        else:
#                            for iel in range(len(self.beamLine.flow)):
#                                if self.beamLine.flow[iel][0] == startElement:
#                                    self.beamLine.flow[iel][0] =\
#                                        str(item.text())
#                        if not wasDeleted:
#                            startFrom = self.nameToFlowPos(newElName)
#                elif pText in ['properties'] and iCol > 0:
#                    elItem = item.parent().parent()
#                    elNameStr = str(elItem.text())
#                    elClassStr = self.getClassName(elItem)
#                    if len(re.findall('.BeamLine', elClassStr)) > 0:  # BL
#                        paramName = str(item.parent().child(item.index().row(),
#                                                            0).text())
#                        paramValue = self.parametrize(str(item.text()))
#                        setattr(self.beamLine, paramName, paramValue)
#                        startFrom = 0
#                    else:  # Setters and getters not implemented yet for OEs
#                        oeType = 0 if len(re.findall(
#                            'raycing.sou', elClassStr)) > 0 else 1
#                        try:
#                            kwargs = createParamDict(item.parent(),
#                                                     elClassStr)
#                            if self.beamLine.oesDict[elNameStr][0] is None:
#                                self.beamLine.oesDict[elNameStr] =\
#                                    [eval(elClassStr)(**kwargs), oeType]
#                                self.progressBar.setFormat(
#                                    "Class {} successfully initialized.".format(  # analysis:ignore
#                                        elNameStr))
#                                print("Class", elNameStr,
#                                      "successfully initialized.")
#                            else:
#                                self.beamLine.oesDict[elNameStr][0].__init__(**kwargs)  # analysis:ignore
#                                self.progressBar.setFormat(
#                                    "Class {} successfully re-initialized.".format(  # analysis:ignore
#                                        elNameStr))
#                                print("Class", elNameStr,
#                                      "successfully re-initialized.")
#                        except:  # analysis:ignore
#                            self.beamLine.oesDict[elNameStr] =\
#                                [None, oeType]
#                            self.progressBar.setFormat(
#                                "Incorrect parameters. Class {} not initialized.".format(elNameStr))  # analysis:ignore
#                            print("Incorrect parameters. Class", elNameStr,
#                                  "not initialized.")
#                        if len(re.findall('raycing.aper', elClassStr)) > 0:
#                            if self.rayPath is not None:
#                                for segment in self.rayPath[0]:
#                                    if segment[2] == elNameStr:
#                                        startFrom =\
#                                            self.nameToFlowPos(segment[0])
#                                        break
#                                else:
#                                    startFrom =\
#                                        self.nameToFlowPos(elNameStr)
#                            else:
#                                startFrom = self.nameToFlowPos(elNameStr)
#                        else:
#                            startFrom = self.nameToFlowPos(elNameStr)
#                elif pText in ['parameters', 'output'] and iCol > 0:
#                    elItem = item.parent().parent().parent()
#                    elNameStr = str(elItem.text())
#                    self.progressBar.setFormat(
#                        "Method of {} was modified".format(elNameStr))
#                    print("Method of", elNameStr, "was modified")
#                    self.beamLine.flow = buildFlow(startFrom=elNameStr)
#                    startFrom = self.nameToFlowPos(elNameStr)
#                elif item.parent().parent() == self.rootBLItem and newElement:
#                    elItem = item.parent()
#                    elNameStr = str(elItem.text())
#                    self.progressBar.setFormat(
#                        "Method {0} was added to {1}".format(
#                            item.text(), elNameStr))
#                    print("Method", item.text(), "was added to", elNameStr)
#                    self.beamLine.flow = buildFlow(startFrom=elNameStr)
#                    startFrom = self.nameToFlowPos(elNameStr)
#        else:  # Rebuild beamline
#            for ie in range(self.rootBLItem.rowCount()):
#                elItem = self.rootBLItem.child(ie, 0)
#                elNameStr = str(elItem.text())
#                if elNameStr == 'properties':  # Beamline properties
#                    for iprop in range(elItem.rowCount()):
#                        paramName = str(elItem.child(iprop, 0).text())
#                        paramValue = self.parametrize(str(elItem.child(
#                            iprop, 1).text()))
#                        setattr(self.beamLine, paramName, paramValue)
#                elif elNameStr != '_object':  # Beamline element
#                    for iprop in range(elItem.rowCount()):
#                        pItem = elItem.child(iprop, 0)
#                        pText = str(pItem.text())
#                        if pText == 'properties':  # OE properties
#                            elClassStr = self.getClassName(elItem)
#                            oeType = 0 if len(re.findall(
#                                'raycing.sou', elClassStr)) > 0 else 1
#                            try:
#                                kwArgs = createParamDict(pItem, elClassStr)
#                                self.beamLine.oesDict[elNameStr] =\
#                                    [eval(elClassStr)(**kwArgs), oeType]
#                                self.progressBar.setFormat(
#                                    "Class {} successfully initialized.".format(  # analysis:ignore
#                                        elNameStr))
#                                print("Class", elNameStr,
#                                      "successfully initialized.")
#                            except:  # analysis:ignore
#                                self.beamLine.oesDict[elNameStr] =\
#                                    [None, oeType]
#                                self.progressBar.setFormat(
#                                    "Incorrect parameters. Class {} not initialized.".format(elNameStr))  # analysis:ignore
#                                print("Incorrect parameters. Class", elNameStr,
#                                      "not initialized.")
#            self.beamLine.flow = buildFlow()
#            startFrom = 0
#
#        if self.isGlowAutoUpdate:
#            if startFrom is not None:
#                self.blPropagateFlow(startFrom)

    def blPropagateFlow(self, startFrom):
#        self.blRunGlow()
        pass
#        objThread = qt.QThread(self)
#        obj = PropagationConnect()
#        obj.emergencyRetrace.connect(partial(self.reTrace, objThread))
#        obj.finished.connect(objThread.quit)
#        obj.rayPathReady.connect(self.blRunGlow)
#        obj.propagationProceed.connect(self.updateProgressBar)
#        obj.moveToThread(objThread)
#        propagateFlowInThread = partial(
#            obj.propagateFlowThread, self.beamLine, startFrom)
#        objThread.started.connect(propagateFlowInThread)
#        objThread.start()

    def reTrace(self, objThread):
        print('Beam structure integrity corrupted. Retracing from the Source.')
        try:
            objThread.quit()
        except Exception:
            if _DEBUG_:
                raise
            else:
                pass
        self.populateBeamline(item=None)

    def populateBeamline(self, item=None):
        self.blUpdateLatchOpen = False
        self.prbStart = 0
        self.prbRange = 100
        try:
            self.beamLine = raycing.BeamLine()
            self.beamLine.flowSource = 'Qook'
            self.updateBeamlineBeams(item=None)
            self.updateBeamlineMaterials(item=None)
            self.updateBeamline(item=None)
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                pass
        self.blUpdateLatchOpen = True

    def toggleGlow(self, status):
        self.isGlowAutoUpdate = status
        self.blRunGlow()
#        if self.isGlowAutoUpdate:
#            self.populateBeamline()

#    def blRunGlow(self, rayPath):
    def blRunGlow(self, kwargs={}):
#        self.rayPath = rayPath
#        if hasattr(self.beamLine, 'layoutStr'):
#            print(self.beamLine.layoutStr)
        if self.blViewer is None:
            try:
                _ = self.beamLine.export_to_json()
                print(self.beamLine.layoutStr)
                self.blViewer = xrtglow.xrtGlow(layout=self.beamLine.layoutStr,
                                                **kwargs)
                self.blViewer.setWindowTitle("xrtGlow")
                self.blViewer.show()
                self.blViewer.parentRef = self
                self.blViewer.parentSignal = self.statusUpdate
            except Exception as e:  # TODO: Handle exceptions
                raise(e)
        else:
#            self.blViewer.updateOEsList(self.rayPath)
            if self.blViewer.isHidden():
                self.blViewer.show()

    def updateProgressBar(self, dataTuple):
        self.progressBar.setValue(self.prbStart +
                                  int(dataTuple[0] * self.prbRange))
        self.progressBar.setFormat(dataTuple[1])

    def generateCode(self):
        self.progressBar.setValue(0)
        self.progressBar.setFormat("Flattening structure.")
        for tree, item in zip([self.tree, self.matTree,
                               self.plotTree, self.runTree],
                              [self.rootBLItem, self.rootMatItem,
                               self.rootPlotItem, self.rootRunItem]):
            item.model().blockSignals(True)
            self.flattenElement(tree, item)
            item.model().blockSignals(False)
        self.progressBar.setValue(10)
        BLName = str(self.rootBLItem.text())
        e0str = "{}E0 = 5000\n".format(myTab)
        fullCode = ""
        codeHeader = """# -*- coding: utf-8 -*-\n\"\"\"\n
__author__ = \"Konstantin Klementiev\", \"Roman Chernikov\"
__date__ = \"{0}\"\n\nCreated with xrtQook\n\n\n{2}\n\n"\"\"\n
import numpy as np\nimport sys\nsys.path.append(r\"{1}\")\n""".format(
            str(date.today()), path_to_xrt, self.fileDescription)
        codeDeclarations = """\n"""
        codeBuildBeamline = "\ndef build_beamline():\n"
        codeBuildBeamline += '{2}{0} = {1}.BeamLine('.format(
            BLName, raycing.__name__, myTab)
        self.progressBar.setFormat("Defining the beamline.")
        for ib in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ib, 0).text() == '_object':
                blstr = str(self.rootBLItem.child(ib, 1).text())
                break
        for ib in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ib, 0).text() == 'properties':
                blPropItem = self.rootBLItem.child(ib, 0)
                for iep, arg_def in zip(range(
                        blPropItem.rowCount()),
                        list(zip(*self.getParams(blstr)))[1]):
                    paraname = str(blPropItem.child(iep, 0).text())
                    paravalue = str(blPropItem.child(iep, 1).text())
                    if paravalue != str(arg_def):
                        paravalue = self.quotize(paravalue)
                        codeBuildBeamline += '\n{2}{0}={1},'.format(
                            paraname, paravalue, myTab*2)
        codeBuildBeamline = codeBuildBeamline.rstrip(',') + ')\n\n'

        codeRunProcess = '\ndef run_process({}):\n'.format(BLName)

        codeMain = "\ndef main():\n"
        codeMain += '{0}{1} = build_beamline()\n'.format(myTab, BLName)

        codeFooter = """\n
if __name__ == '__main__':
    main()\n"""
        self.progressBar.setValue(20)
        self.progressBar.setFormat("Defining materials.")
        for ie in range(self.rootMatItem.rowCount()):
            if str(self.rootMatItem.child(ie, 0).text()) != "None":
                matItem = self.rootMatItem.child(ie, 0)
                ieinit = ""
                for ieph in range(matItem.rowCount()):
                    if matItem.child(ieph, 0).text() == '_object':
                        elstr = str(matItem.child(ieph, 1).text())
                        klass = eval(elstr)
                        if klass.__module__.startswith('xrt'):
                            ieinit = elstr + "(" + ieinit
                        else:
                            # import of custom materials
                            importStr = 'import {0}'.format(klass.__module__)
                            # if importStr not in codeHeader:
                            codeHeader += importStr + '\n'
                            ieinit = "{0}.{1}({2}".format(
                                klass.__module__, klass.__name__, ieinit)
                for ieph in range(matItem.rowCount()):
                    if matItem.child(ieph, 0).text() != '_object':
                        if matItem.child(ieph, 0).text() == 'properties':
                            pItem = matItem.child(ieph, 0)
                            for iep, arg_def in zip(range(
                                    pItem.rowCount()),
                                    list(zip(*self.getParams(elstr)))[1]):
                                paraname = str(pItem.child(iep, 0).text())
                                paravalue = str(pItem.child(iep, 1).text())
                                if paravalue != str(arg_def) or\
                                        paravalue == 'bl':
                                    if paraname.lower() not in\
                                            ['tlayer', 'blayer',
                                             'coating', 'substrate']:
                                        paravalue = self.quotize(paravalue)
                                    ieinit += '\n{2}{0}={1},'.format(
                                        paraname, paravalue, myTab)
                codeDeclarations += '{0} = {1})\n\n'.format(
                    matItem.text(), str.rstrip(ieinit, ","))
        self.progressBar.setValue(30)
        self.progressBar.setFormat("Adding optical elements.")
        outBeams = ['None']
        for ie in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ie, 0).text() != "properties" and\
                    self.rootBLItem.child(ie, 0).text() != "_object":
                tItem = self.rootBLItem.child(ie, 0)
                ieinit = ""
                ierun = ""
                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() == '_object':
                        elstr = str(tItem.child(ieph, 1).text())
                        klass = eval(elstr)
                        if klass.__module__.startswith('xrt'):
                            ieinit = elstr + "(" + ieinit
                        else:
                            # import of custom OEs
                            importStr = 'import {0}'.format(klass.__module__)
                            # if importStr not in codeHeader:
                            codeHeader += importStr + '\n'
                            ieinit = "{0}.{1}({2}".format(
                                klass.__module__, klass.__name__, ieinit)

                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() == 'properties':
                        pItem = tItem.child(ieph, 0)
                        for iep, arg_def in zip(range(
                                pItem.rowCount()),
                                list(zip(*self.getParams(elstr)))[1]):
                            paraname = str(pItem.child(iep, 0).text())
                            paravalue = str(pItem.child(iep, 1).text())
                            if paraname == 'center':
                                if paravalue.startswith('['):
                                    paravalue = re.findall(r'\[(.*)\]',
                                                           paravalue)[0]
                                elif paravalue.startswith('('):
                                    paravalue = re.findall(r'\((.*)\)',
                                                           paravalue)[0]
                                cCoord = [self.getVal(c.strip()) for c in
                                          str.split(paravalue, ',')]
                                paravalue = re.sub('\'', '', str(
                                    [self.quotize(c) for c in cCoord]))
                            if paravalue != str(arg_def) or\
                                    paraname == 'bl':
                                if paraname.lower() not in\
                                        ['bl', 'center', 'material',
                                         'material2']:
                                    paravalue = self.quotize(paravalue)
                                ieinit += '\n{2}{0}={1},'.format(
                                    paraname, paravalue, myTab*2)
                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() != '_object' and\
                            tItem.child(ieph, 0).text() != 'properties':
                        pItem = tItem.child(ieph, 0)
                        tmpSourceName = ""
                        for namef, objf in inspect.getmembers(eval(elstr)):
                            if (inspect.ismethod(objf) or
                                    inspect.isfunction(objf)) and\
                                    namef == str(pItem.text()).strip('()'):
                                methodObj = inspect.unwrap(objf)
                        for imet in range(pItem.rowCount()):
                            if str(pItem.child(imet, 0).text()) ==\
                                    'parameters':
                                mItem = pItem.child(imet, 0)
                                for iep, arg_def in\
                                    zip(range(mItem.rowCount()),
                                        getargspec(methodObj)[3]):
                                    paraname = str(mItem.child(iep, 0).text())
                                    paravalue = str(mItem.child(iep, 1).text())
                                    if paravalue != str(arg_def):
                                        ierun += '\n{2}{0}={1},'.format(
                                            paraname, paravalue, myTab*2)
                            elif pItem.child(imet, 0).text() == 'output':
                                mItem = pItem.child(imet, 0)
                                paraOutput = ""
                                paraOutBeams = []
                                for iep in range(mItem.rowCount()):
                                    paravalue = mItem.child(iep, 1).text()
                                    paraOutBeams.append(str(paravalue))
                                    outBeams.append(str(paravalue))
                                    paraOutput += str(paravalue)+", "
                                    if len(re.findall('sources', elstr)) > 0\
                                            and tmpSourceName == "":
                                        tmpSourceName = str(paravalue)
                                        if len(re.findall('Source',
                                                          elstr)) > 0:
                                            e0str = '{2}E0 = list({0}.{1}.energies)[0]\n'.format( # analysis:ignore
                                                BLName, tItem.text(), myTab)
                                        else:
                                            e0str = '{2}E0 = 0.5 * ({0}.{1}.eMin +\n{3}{0}.{1}.eMax)\n'.format( # analysis:ignore
                                                BLName, tItem.text(), myTab,
                                                myTab*4)
                        codeRunProcess += '{5}{0} = {1}.{2}.{3}({4})\n\n'.format( # analysis:ignore
                            paraOutput.rstrip(', '), BLName, tItem.text(),
                            str(pItem.text()).strip('()'),
                            ierun.rstrip(','), myTab)

                codeBuildBeamline += '{3}{0}.{1} = {2})\n\n'.format(
                    BLName, str(tItem.text()), ieinit.rstrip(','), myTab)
        codeBuildBeamline += "{0}return {1}\n\n".format(myTab, BLName)
        codeRunProcess += r"{0}outDict = ".format(myTab) + "{"
        self.progressBar.setValue(60)
        self.progressBar.setFormat("Defining the propagation.")
        for ibm in reversed(range(self.beamModel.rowCount())):
            beamName = str(self.beamModel.item(ibm, 0).text())
            if beamName not in outBeams:
                self.beamModel.takeRow(ibm)
        for ibm in range(self.beamModel.rowCount()):
            beamName = str(self.beamModel.item(ibm, 0).text())
            if beamName != "None":
                codeRunProcess += '\n{1}{1}\'{0}\': {0},'.format(
                    beamName, myTab)
        codeRunProcess = codeRunProcess.rstrip(',') + "}\n"
        codeRunProcess += "{0}return outDict\n\n\n".format(myTab)
        codeRunProcess +=\
            '{}.run_process = run_process\n\n\n'.format(rrun.__name__)

        codeMain += e0str
        codeMain += '{1}{0}.alignE=E0\n'.format(BLName, myTab)
        if not self.glowOnly:
            codeMain += '{0}{1} = define_plots()\n'.format(
                myTab, self.rootPlotItem.text())
        codePlots = '\ndef define_plots():\n{0}{1} = []\n'.format(
            myTab, self.rootPlotItem.text())
        self.progressBar.setValue(70)
        self.progressBar.setFormat("Adding plots.")
        plotNames = []
        plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
        axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))
        for ie in range(self.rootPlotItem.rowCount()):
            tItem = self.rootPlotItem.child(ie, 0)
            ieinit = "\n{0}{1} = ".format(myTab, tItem.text())
            plotNames.append(str(tItem.text()))
            for ieph in range(tItem.rowCount()):
                if tItem.child(ieph, 0).text() == '_object':
                    elstr = str(tItem.child(ieph, 1).text())
#                    print("xx0", elstr)
                    ieinit += elstr + "("
#                    for parent in (inspect.getmro(eval(elstr)))[:-1]:
#                        for namef, objf in inspect.getmembers(parent):
#                            if (inspect.ismethod(objf) or
#                                    inspect.isfunction(objf)):
#                                if namef == "__init__" and\
#                                        getargspec(objf)[3] is not None:
#                                    obj = objf
#            for iepm, arg_def in zip(range(tItem.rowCount()-1),
#                                     getargspec(obj)[3]):
            for iep in range(tItem.rowCount()):
#                iep = iepm + 1
                if tItem.child(iep, 0).text() != '_object':
                    pItem = tItem.child(iep, 0)
                    if pItem.hasChildren():
                        for ieax in range(pItem.rowCount()):
                            if pItem.child(ieax, 0).text() == '_object':
                                axstr = str(pItem.child(ieax, 1).text())
                                # ieinit = ieinit.rstrip("\n\t\t")
                                ieinit += "\n{2}{0}={1}(".format(
                                    str(tItem.child(iep, 0).text()), axstr,
                                    myTab*2)
#                                for parentAx in\
#                                        (inspect.getmro(eval(axstr)))[:-1]:
#                                    for namefAx, objfAx in\
#                                            inspect.getmembers(parentAx):
#                                        if (inspect.ismethod(objfAx) or
#                                                inspect.isfunction(objfAx)):
#                                            if namefAx == "__init__" and\
#                                                getargspec(
#                                                    objfAx)[3] is not None:
#                                                objAx = objfAx
                       
#                        for ieaxm, arg_defAx in zip(range(pItem.rowCount()-1),
#                                                    getargspec(objAx)[3]):
                        for ieax in range(pItem.rowCount()):
                            paraname = str(pItem.child(ieax, 0).text())
                            if paraname in ['_object']:
                                continue
                            paravalue = str(pItem.child(ieax, 1).text())
                            arg_defAx = str(axDefArgs.get(paraname))
                            
                            
                            if paraname == "data" and paravalue != "auto":
                                paravalue = '{0}.get_{1}'.format(
                                        raycing.__name__, paravalue)
                                ieinit += '\n{2}{0}={1},'.format(
                                        paraname, paravalue, myTab*3)
                            elif paravalue != arg_defAx:
                                # code below parses the properties of other plots
                                # do we need it?
                                tmpParavalue = paravalue * 2
                                while any(str(pltName + '.') in paravalue for
                                          pltName in plotNames) and\
                                        tmpParavalue != paravalue:
                                    tmpParavalue = paravalue
                                    for ipn in range(
                                            self.rootPlotItem.rowCount()):
                                        ipnItem = self.rootPlotItem.child(ipn,
                                                                          0)
                                        for ipp in range(ipnItem.rowCount()):
                                            ippItem = ipnItem.child(ipp, 0)
                                            if ippItem.hasChildren():
                                                for ipx in range(
                                                        ippItem.rowCount()):
                                                    paravalue = re.sub(
                                                        '{0}.{1}.{2}'.format(
                                                            ipnItem.text(),
                                                            ippItem.text(),
                                                            ippItem.child(
                                                                ipx,
                                                                0).text()),
                                                        '{}'.format(
                                                            ippItem.child(
                                                                ipx,
                                                                1).text()),
                                                        paravalue)

                                ieinit += '\n{2}{0}={1},'.format(
                                    paraname, self.quotize(paravalue), myTab*3)
                        ieinit = ieinit.rstrip(",") + "),"
                    else:
                        paraname = str(tItem.child(iep, 0).text())
                        paravalue = str(tItem.child(iep, 1).text())
                        arg_def = str(plotDefArgs.get(paraname))
                        if paravalue != arg_def:
                            if paraname == "fluxKind":
                                ieinit += '\n{2}{0}={1},'.format(
                                    paraname, self.quotizeAll(paravalue),
                                    myTab*2)
                            else:
                                tmpParavalue = paravalue * 2
                                while any(str(pltName + '.') in paravalue for
                                          pltName in plotNames) and\
                                        tmpParavalue != paravalue:
                                    tmpParavalue = paravalue
                                    for ipn in range(
                                            self.rootPlotItem.rowCount()):
                                        ipnItem = self.rootPlotItem.child(ipn,
                                                                          0)
                                        for ipp in range(ipnItem.rowCount()):
                                            paravalue = re.sub(
                                                '{0}.{1}'.format(
                                                    ipnItem.text(),
                                                    ipnItem.child(ipp,
                                                                  0).text()),
                                                '{}'.format(
                                                    ipnItem.child(ipp,
                                                                  1).text()),
                                                paravalue)
                                ieinit += '\n{2}{0}={1},'.format(
                                    paraname, self.quotize(paravalue), myTab*2)
            codePlots += ieinit.rstrip(",") + ")\n"
            codePlots += "{0}{2}.append({1})\n".format(
                myTab, tItem.text(), self.rootPlotItem.text())
        codePlots += "{0}return {1}\n\n".format(
            myTab, self.rootPlotItem.text())
        
        self.progressBar.setValue(90)
        self.progressBar.setFormat("Preparing the main() function.")
        if not self.glowOnly:
            for ie in range(self.rootRunItem.rowCount()):
                if self.rootRunItem.child(ie, 0).text() == '_object':
                    elstr = str(self.rootRunItem.child(ie, 1).text())
                    codeMain += "{0}{1}(\n".format(myTab, elstr)
                    break

            ieinit = ""
            for iem, (argNm, argVal) in zip(range(
                    self.rootRunItem.rowCount() - 1), self.getParams(elstr)):
                ie = iem + 1
                if self.rootRunItem.child(ie, 0).text() != '_object':
                    paraname = self.rootRunItem.child(ie, 0).text()
                    paravalue = self.rootRunItem.child(ie, 1).text()
                    if paraname == "plots":
                        paravalue = self.rootPlotItem.text()
                    if paraname == "backend":
                        paravalue = 'r\"{0}\"'.format(paravalue)
                    if str(paravalue) != str(argVal):
                        if paravalue == 'auto':
                            paravalue = self.quotize(paravalue)
                        ieinit += "{0}{1}={2},\n".format(
                            myTab*2, paraname, paravalue)
            codeMain += ieinit.rstrip(",\n") + ")\n"

        fullCode = codeDeclarations + codeBuildBeamline +\
            codeRunProcess + codePlots + codeMain + codeFooter
        for xrtAlias in self.xrtModules:
            fullModName = (eval(xrtAlias)).__name__
            fullCode = fullCode.replace(fullModName+".", xrtAlias+".")
            codeHeader += 'import {0} as {1}\n'.format(fullModName, xrtAlias)
        fullCode = codeHeader + fullCode
        if self.glowOnly:
            self.glowCode = fullCode
        else:
            if ext.isSpyderlib:
                self.codeEdit.set_text(fullCode)
            else:
                self.codeEdit.setText(fullCode)
                self.tabs.setCurrentWidget(self.codeEdit)
            self.progressBar.setValue(100)
            self.progressBar.setFormat(
                'Python code successfully generated')
#                self.statusBar.showMessage(
#                    'Python code successfully generated', 5000)

    def saveCode(self):
        saveStatus = False
        if self.saveFileName == "":
            saveDialog = qt.QFileDialog()
            saveDialog.setFileMode(qt.QFileDialog.AnyFile)
            saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
            saveDialog.setNameFilter("Python files (*.py)")
            if (saveDialog.exec_()):
                self.saveFileName = saveDialog.selectedFiles()[0]
        if self.saveFileName != "":
            if not str(self.saveFileName).endswith('.py'):
                self.saveFileName += '.py'
            try:
                fileObject = open(self.saveFileName, 'w')
                fileObject.write(self.codeEdit.toPlainText())
                fileObject.close
                saveStatus = True
                saveMsg = 'Script saved to {}'.format(
                    os.path.basename(str(self.saveFileName)))
                self.tabs.setTabText(
                    5, os.path.basename(str(self.saveFileName)))
                if ext.isSpyderConsole:
                    self.codeConsole.wdir = os.path.dirname(
                        str(self.saveFileName))
                else:
                    self.qprocess.setWorkingDirectory(
                        os.path.dirname(str(self.saveFileName)))
            except (OSError, IOError) as errStr:
                saveMsg = str(errStr)
                self.progressBar.setFormat(saveMsg)
#            self.statusBar.showMessage(saveMsg, 5000)
        return saveStatus

    def saveCodeAs(self):
        tmpName = self.saveFileName
        self.saveFileName = ""
        if not self.saveCode():
            self.progressBar.setValue(0)
            self.progressBar.setFormat('Failed saving code to {}'.format(
                    os.path.basename(str(self.saveFileName))))
#                'Failed saving code to {}'.format(
#                    os.path.basename(str(self.saveFileName))), 5000)
            self.saveFileName = tmpName

    def execCode(self):
        self.saveCode()
        self.tabs.setCurrentWidget(self.codeConsole)
        if ext.isSpyderConsole:
            self.codeConsole.fname = str(self.saveFileName)
            self.codeConsole.create_process()
        else:
            self.codeConsole.clear()
            self.codeConsole.append('Starting {}\n\n'.format(
                    os.path.basename(str(self.saveFileName))))
            self.codeConsole.append('Press Ctrl+X to terminate process\n\n')
            self.qprocess.start(sys.executable, ['-u', str(self.saveFileName)])

    def toggleExperimentalMode(self):
        self.experimentalMode = not self.experimentalMode
        self.progressBar.setFormat("Experimental Mode {}abled".format(
            "en" if self.experimentalMode else "dis"))

    def closeEvent(self, event):
        if self.blViewer is not None:
            self.blViewer.close()
        super().closeEvent(event)


class PropagationConnect(qt.QObject):
    propagationProceed = qt.Signal(tuple)
    rayPathReady = qt.Signal(list)
    finished = qt.Signal()
    emergencyRetrace = qt.Signal()

    def propagateFlowThread(self, blRef, startFrom):
        try:
            self.propagationProceed.emit((0.5, "Starting propagation"))
            blRef.propagate_flow(
                startFrom=startFrom,
                signal=self.propagationProceed)
            self.propagationProceed.emit((1, "Preparing data for Glow"))
            rayPath = blRef.export_to_glow(signal=self.propagationProceed)
            self.propagationProceed.emit((1, "Done"))
            self.rayPathReady.emit(rayPath)
            self.finished.emit()
        except Exception:
            if _DEBUG_:
                print('Propagation impossible', startFrom)
                raise
            else:
                if startFrom != 0:
                    self.emergencyRetrace.emit()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    ex = XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())

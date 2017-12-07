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

See a short :ref:`tutorial for xrtQook <qook_tutorial>`.

.. imagezoom:: _images/xrtQook.png
   :alt: &ensp;A view of xrtQook with an empty beamline tree on the left and a
       help panel on the right.

"""

from __future__ import print_function
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "25 Jun 2017"
__version__ = "1.3"

import os
import sys
import textwrap
import numpy as np  # analysis:ignore , really needed

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
except ImportError:
    isOpenCL = False

from ..commons import myspyder as spyder  # analysis:ignore

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
from ... import plotter as xrtplot  # analysis:ignore
from ... import runner as xrtrun  # analysis:ignore
from ..commons import qt  # analysis:ignore
from ..commons import gl  # analysis:ignore
if gl.isOpenGL:
    from .. import xrtGlow as xrtglow  # analysis:ignore

path_to_xrt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
myTab = 4*" "

useSlidersInTree = False
withSlidersInTree = ['pitch', 'roll', 'yaw', 'bragg']
slidersInTreeScale = {'pitch': 0.1, 'roll': 0.1, 'yaw': 0.1, 'bragg': 1e-3}

try:
    class WebPage(qt.QtWeb.QWebPage):
        """
        Web page subclass to manage hyperlinks like in WebEngine
        """
        showHelp = qt.Signal()

    class QWebView(qt.QtWeb.QWebView):
        """Web view"""
        def __init__(self):
            qt.QtWeb.QWebView.__init__(self)
            web_page = WebPage(self)
            self.setPage(web_page)

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
        showHelp = qt.Signal()
        linkDelegationPolicy = 0

        def setLinkDelegationPolicy(self, policy):
            self.linkDelegationPolicy = policy

        def acceptNavigationRequest(self, url, navigation_type, isMainFrame):
            """
            Overloaded method to handle links ourselves
            """
            if navigation_type in\
                    [qt.QtWeb.QWebEnginePage.NavigationTypeLinkClicked] and\
                    str(url.toString()).startswith('file:'):
                if self.linkDelegationPolicy == 1 and\
                        '.png' not in url.toString():
                    self.linkClicked.emit(url)
                return False
            elif navigation_type in\
                    [qt.QtWeb.QWebEnginePage.NavigationTypeBackForward] and\
                    self.linkDelegationPolicy == 0:
                if str(qt.QUrl(spyder.CSS_PATH).toString()).lower() in\
                        str(url.toString()).lower():
                    self.showHelp.emit()
                    return False
            return True

    class QWebView(qt.QtWeb.QWebEngineView):
        """Web view"""
        def __init__(self):
            qt.QtWeb.QWebEngineView.__init__(self)
            web_page = WebPage(self)
            self.setPage(web_page)


class XrtQook(qt.QWidget):
    statusUpdate = qt.pyqtSignal(tuple)
    sig_resized = qt.Signal("QResizeEvent")
    sig_moved = qt.Signal("QMoveEvent")

    def __init__(self):
        super(XrtQook, self).__init__()
        self.xrtQookDir = os.path.dirname(os.path.abspath(__file__))
        self.setAcceptDrops(True)
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
        mainWidget.setMinimumWidth(400)
        mainBox = qt.QVBoxLayout()
        mainBox.setContentsMargins(0, 0, 0, 0)
        docBox = qt.QVBoxLayout()
        docBox.setContentsMargins(0, 0, 0, 0)

        self.helptab = qt.QTabWidget()
        docWidget = qt.QWidget()
        docWidget.setMinimumWidth(500)

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
        docWidget.setStyleSheet("border:1px solid rgb(20, 20, 20);")
        self.helptab.addTab(docWidget, "Live Doc")
        self.helptab.tabBar().setVisible(False)
#        self.helptab.setTabsClosable(True)
#        self.helptab.setWidget(docWidget)

        canvasBox.addWidget(canvasSplitter)

        canvasSplitter.addWidget(mainWidget)
        canvasSplitter.addWidget(self.helptab)
        self.setLayout(canvasBox)
        self.initAllTrees()

    def _addAction(self, module, elname, afunction, menu):
        objName = '{0}.{1}'.format(module.__name__, elname)
        elAction = qt.QAction(self)
        elAction.setText(elname)
        elAction.hovered.connect(
            partial(self.showObjHelp, objName))
        elAction.triggered.connect(
            partial(afunction, elname, objName, None))
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
            qt.QIcon(os.path.join(self.iconsDir, 'eyeglasses7_128.png')),
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

        aboutAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'dialog-information.png')),
            'About xrtQook',
            self)
        aboutAction.setShortcut('Ctrl+I')
        aboutAction.triggered.connect(self.aboutCode)

        self.vToolBar = qt.QToolBar('Add Elements buttons')
        self.vToolBar.setOrientation(qt.QtCore.Qt.Vertical)
        self.vToolBar.setIconSize(qt.QtCore.QSize(56, 56))

        for menuName, amodule, afunction, aicon in zip(
                ['Add Source', 'Add OE', 'Add Aperture', 'Add Screen',
                 'Add Material', 'Add Plot'],
                [rsources, roes, rapts, rscreens, rmats, None],
                [self.addElement]*4 + [self.addMaterial, self.addPlot],
                ['add{0:1d}'.format(i+1) for i in range(6)]):
            amenuButton = qt.QToolButton()
            amenuButton.setIcon(qt.QIcon(os.path.join(
                self.iconsDir, '{}.png'.format(aicon))))
            amenuButton.setToolTip(menuName)

            tmenu = qt.QMenu()
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
                else:  # only with __all__
                    for elname in amodule.__all__:
                        self._addAction(amodule, elname, afunction, tmenu)
            else:
                for beamType in ['Local Beams', 'Global Beams']:
                    subAction = qt.QAction(self)
                    subAction.setText(beamType)
                    subAction.hovered.connect(partial(
                        self.populateBeamsMenu, beamType))
                    tmenu.addAction(subAction)
            amenuButton.setMenu(tmenu)
            amenuButton.setPopupMode(qt.QToolButton.InstantPopup)
            self.vToolBar.addWidget(amenuButton)
            if menuName in ['Add Screen', 'Add Material']:
                self.vToolBar.addSeparator()

        self.tabs = qt.QTabWidget()
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
        self.toolBar.addAction(aboutAction)
        bbl = qt.QShortcut(self)
        bbl.setKey(qt.Key_F4)
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

    def populateBeamsMenu(self, beamType):
        sender = self.sender()
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
        if spyder.isSphinx:
            self.webHelp = QWebView()
            self.webHelp.setContextMenuPolicy(qt.CustomContextMenu)
            self.webHelp.customContextMenuRequested.connect(self.docMenu)
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

        if spyder.isSpyderlib:
            self.codeEdit = spyder.codeeditor.CodeEditor(self)
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

        if spyder.isSpyderConsole:
            self.codeConsole = spyder.pythonshell.ExternalPythonShell(
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
        menu.addAction("Zoom In",
                       lambda: self.zoomDoc(1))
        menu.addAction("Zoom Out", lambda: self.zoomDoc(-1))
        menu.addAction("Zoom reset", lambda: self.zoomDoc(0))
        menu.addSeparator()
        if str(self.webHelp.url().toString()).startswith('http:'):
            menu.addAction("Back", self.goBack)
            if self.webHelp.history().canGoForward():
                menu.addAction("Forward", self.webHelp.forward)
        menu.exec_(self.webHelp.mapToGlobal(position))

    def goBack(self):
        if self.webHelp.history().canGoBack():
            self.webHelp.back()
        else:
            self.webHelp.page().showHelp.emit()

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

    def initAllTrees(self):
        self.blUpdateLatchOpen = False
        # runTree view
        self.runTree.setModel(self.runModel)
        self.runTree.setAlternatingRowColors(True)
        self.runTree.setSortingEnabled(False)
        self.runTree.setHeaderHidden(False)
        self.runTree.setAnimated(True)
        self.runTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.runTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        for name, obj in inspect.getmembers(xrtrun):
            if inspect.isfunction(obj) and name == "run_ray_tracing":
                if inspect.getargspec(obj)[3] is not None:
                    runStr = '{0}.{1}'.format(xrtrun.__name__, name)
                    self.addObject(self.runTree, self.rootRunItem, runStr)
                    for arg, argVal in self.getParams(runStr):
                        if arg.lower() == "plots":
                            argVal = self.rootPlotItem.text()
                        if arg.lower() == "beamline":
                            argVal = self.rootBLItem.text()
                        self.addParam(self.rootRunItem, arg, argVal)
        self.addCombo(self.runTree, self.rootRunItem)
        self.runTree.setColumnWidth(0, int(self.runTree.width()/3))
        index = self.runModel.indexFromItem(self.rootRunItem)
        self.runTree.setExpanded(index, True)

        # plotTree view
        self.plotTree.setModel(self.plotModel)
        self.plotTree.setAlternatingRowColors(True)
        self.plotTree.setSortingEnabled(False)
        self.plotTree.setHeaderHidden(False)
        self.plotTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.plotTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])

        # materialsTree view
        self.matTree.setModel(self.materialsModel)
        self.matTree.setAlternatingRowColors(True)
        self.matTree.setSortingEnabled(False)
        self.matTree.setHeaderHidden(False)
        self.matTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.matTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])

        # BLTree view
        self.tree.setModel(self.beamLineModel)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)
        self.tree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        headers = ['Parameter', 'Value']
        if useSlidersInTree:
            headers.append('Slider')
        self.runTree.model().setHorizontalHeaderLabels(headers)
        self.tree.model().setHorizontalHeaderLabels(headers)

        elprops = self.addProp(self.rootBLItem, 'properties')
        for name, obj in inspect.getmembers(raycing):
            if inspect.isclass(obj) and name == "BeamLine":
                for namef, objf in inspect.getmembers(obj):
                    if (inspect.ismethod(objf) or
                        inspect.isfunction(objf)) and\
                       namef == "__init__" and\
                       inspect.getargspec(objf)[3] is not None:
                        self.addObject(self.tree,
                                       self.rootBLItem,
                                       '{0}.{1}'.format(
                                           raycing.__name__, name))
                        for arg, argVal in zip(inspect.getargspec(objf)[0][1:],
                                               inspect.getargspec(objf)[3]):
                            self.addParam(elprops, arg, argVal)

        # self.showDoc(self.rootBLItem.index())
        self.tree.expand(self.rootBLItem.index())
        self.tree.setColumnWidth(0, int(self.tree.width()/3))
        self.tabs.tabBar().setTabTextColor(0, qt.black)
        self.tabs.tabBar().setTabTextColor(2, qt.black)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("New beamline")
        self.curObj = None
        self.blColorCounter = 0
        self.pltColorCounter = 0
        self.fileDescription = ""
#        self.descrEdit.setText("")
        self.currHtml = ""
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
        self.beamLine = raycing.BeamLine()
        self.beamLine.flowSource = 'Qook'
        self.updateBeamlineBeams(item=None)
        self.updateBeamlineMaterials(item=None)
        self.updateBeamline(item=None)
        self.rayPath = None
        self.blUpdateLatchOpen = True

    def initAllModels(self):
        self.blUpdateLatchOpen = False
        self.beamLineModel = qt.QStandardItemModel()
        self.addValue(self.beamLineModel.invisibleRootItem(), "beamLine")
        self.beamLineModel.itemChanged.connect(self.beamLineItemChanged)
        self.rootBLItem = self.beamLineModel.item(0, 0)

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
                        oclToolTip = 'Platform: {0}\nDevice: {1}\nType: {2}\n\
Compute Units: {3}\nFP64 Support: {4}'.format(platform.name,
                                              device.name,
                                              cl.device_type.to_string(
                                                  device.type),
                                              device.max_compute_units,
                                              bool(device.double_fp_config))
                        oclItem, oclItemStr = self.addParam(self.OCLModel,
                                                            device.name,
                                                            oclDev)
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
        self.beamModel.itemChanged.connect(self.updateBeamlineBeams)

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
        if spyder.isSpyderlib:
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
        if spyder.isSphinx:
            self.webHelp.page().setLinkDelegationPolicy(0)
        argSpecStr = '('
        for arg, argVal in self.getParams(obj):
            showVal = self.quotize(argVal)
            try:
                showVal = showVal.strip('r') if showVal.startswith('r"') else\
                    showVal
                showVal = showVal.strip('"') if\
                    (arg == 'bl' and showVal is not None) else showVal
            except:  # analysis:ignore
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

        argDocStr = u'{0}{1}\n\n'.format(myTab, headerDoc) if\
            objP.__doc__ is not None else "\n\n"
        dNames, dVals = self.getArgDescr(obj)
        if len(dNames) > 0:
            argDocStr += '{0}Properties\n{0}{1}\n\n'.format(myTab, 10*'-')
        for dName, dVal in zip(dNames, dVals):
            argDocStr += u'{2}*{0}*: {1}\n\n'.format(dName, dVal, myTab)

        if spyder.isSphinx:
            self.webHelp.history().clear()
            self.webHelp.page().history().clear()
            err = None
            try:
                cntx = spyder.generate_context(
                    name=nameStr,
                    argspec=argSpecStr,
                    note=noteStr,
                    img_path=self.xrtQookDir,
                    math=True)
            except TypeError as err:
                cntx = spyder.generate_context(
                    name=nameStr,
                    argspec=argSpecStr,
                    note=noteStr,
                    math=True)

            argDocStr = argDocStr.replace('imagezoom::', 'image::')
            html_text = spyder.sphinxify(textwrap.dedent(argDocStr), cntx)
            if err is None:
                html2 = re.findall(' {4}return.*', html_text)[0]
                sbsPath = re.sub('img_name',
                                 'attr',
                                 re.sub('\\\\', '/', html2))
                if 'file://' not in sbsPath:
                    sbsPath = re.sub('return \'', 'return \'file:///', sbsPath)
                new_html = re.sub(' {4}return.*', sbsPath, html_text, 1)
            else:
                spyder_crutch = "<script>\n$(document).ready(\
    function () {\n    $('img').attr\
    ('src', function(index, attr){\n     return \'file:///"
                spyder_crutch += "{0}\' + \'/\' + attr\n".format(
                    re.sub('\\\\', '/', os.path.join(path_to_xrt,
                                                     'xrt',
                                                     'xrtQook')))
                spyder_crutch += "    });\n});\n</script>\n<body>"
                new_html = re.sub('<body>', spyder_crutch, html_text, 1)
            self.webHelp.setHtml(new_html, qt.QUrl(spyder.CSS_PATH))
            self.currHtml = new_html
            self.webHelp.page().showHelp.connect(partial(
                self.webHelp.setHtml, self.currHtml, qt.QUrl(spyder.CSS_PATH)))
        else:
            argDocStr = u'{0}\nDefiniiton: {1}\n\nType: {2}\n\n\n'.format(
                nameStr.upper(), argSpecStr, noteStr) + argDocStr
            self.webHelp.setText(textwrap.dedent(argDocStr))
            self.webHelp.setReadOnly(True)

    def updateDescription(self):
        self.typingTimer.start(500)

    def updateDescriptionDelayed(self):
        self.fileDescription = self.descrEdit.toPlainText()
        img_path = __file__ if self.layoutFileName == "" else\
            self.layoutFileName
        self.showTutorial(self.fileDescription,
                          "Description",
                          os.path.dirname(os.path.abspath(str(img_path))))
        self.descrEdit.setFocus()

    def showWelcomeScreen(self):
        argDescr = u"""

        .. image:: _images/qookSplash2.gif
           :scale: 75 %

        xrtQook is a qt-based GUI for using xrt without having to write python
        scripts. See a short startup `tutorial <tutorial>`_.

        """
        self.showTutorial(argDescr,
                          "xrtQook",
                          os.path.dirname(os.path.abspath(__file__)),
                          delegateLink=True)

    def showDescrByTab(self, tab):
        if tab == 4:
            self.updateDescriptionDelayed()

    def showTutorial(self, argDocStr, name, img_path, delegateLink=False):
        if argDocStr is None:
            return
        if not spyder.isSphinx:
            return
        err = None
        try:
            cntx = spyder.generate_context(
                name=name,
                argspec="",
                note="",
                img_path=img_path,
                math=True)
        except TypeError as err:
            cntx = spyder.generate_context(
                name=name,
                argspec="",
                note="",
                math=True)
        argDocStr = argDocStr.replace('imagezoom::', 'image::')
        html_text = spyder.sphinxify(textwrap.dedent(argDocStr), cntx)
        if err is None:
            html2 = re.findall(' {4}return.*', html_text)[0]
            sbsPath = re.sub('img_name',
                             'attr',
                             re.sub('\\\\', '/', html2))
            if 'file://' not in sbsPath:
                sbsPath = re.sub('return \'', 'return \'file:///', sbsPath)
            new_html = re.sub(' {4}return.*', sbsPath, html_text, 1)
        else:
            spyder_crutch = "<script>\n$(document).ready(\
    function () {\n    $('img').attr\
    ('src', function(index, attr){\n     return \'file:///"
            spyder_crutch += "{0}\' + \'/\' + attr\n".format(
                re.sub('\\\\', '/', os.path.join(path_to_xrt,
                                                 'xrt',
                                                 'xrtQook')))
            spyder_crutch += "    });\n});\n</script>\n<body>"
            new_html = re.sub('<body>', spyder_crutch, html_text, 1)
        self.webHelp.setHtml(new_html, qt.QUrl(spyder.CSS_PATH))
        if delegateLink:
            from . import tutorial
            self.webHelp.page().setLinkDelegationPolicy(1)
            self.webHelp.page().linkClicked.connect(partial(
                self.showTutorial,
                tutorial.__doc__[229:],
                "Using xrtQook for script generation",
                self.xrtQookDir))
        else:
            self.webHelp.page().setLinkDelegationPolicy(0)
        self.curObj = None

    def showOCLinfo(self):
        argDocStr = u""
        for iplatform, platform in enumerate(cl_platforms):
            argDocStr += '=' * 25 + '\n'
            argDocStr += 'Platform {0}: {1}\n'.format(iplatform, platform.name)
            argDocStr += '=' * 25 + '\n'
            argDocStr += '**Vendor**:  {0}\n\n'.format(platform.vendor)
            argDocStr += '**Version**:  {0}\n\n'.format(platform.version)
            # argDocStr += '**Extensions**:  {0}\n\n'.format(
            #    platform.extensions)
            for idevice, device in enumerate(platform.get_devices()):
                maxFNLen = 0
                maxFVLen = 0
                argDocStr += '{0}**DEVICE {1}**: {2}\n\n'.format(
                    myTab, idevice, device.name)
                fNames = ['*Type*',
                          '*Max Clock Speed*',
                          '*Compute Units*',
                          '*Local Memory*',
                          '*Constant Memory*',
                          '*Global Memory*',
                          '*FP64 Support*']
                fVals = [cl.device_type.to_string(device.type, "%d"),
                         str(device.max_clock_frequency) + ' MHz',
                         str(device.max_compute_units),
                         str(int(device.local_mem_size/1024)) + ' kB',
                         str(int(
                             device.max_constant_buffer_size/1024)) + ' kB',
                         '{0:.2f}'.format(
                             device.global_mem_size/1073741824.) + ' GB',
                         str(bool(int(device.double_fp_config/63)))]
                for fieldName, fieldVal in zip(fNames, fVals):
                    if len(fieldName) > maxFNLen:
                        maxFNLen = len(fieldName)
                    if len(fieldVal) > maxFVLen:
                        maxFVLen = len(fieldVal)
                spacerH = '{0}+{1}+{2}+\n'.format(
                    myTab, (maxFNLen + 2) * '-', (maxFVLen + 2) * '-')
                argDocStr += spacerH
                for fName, fVal in zip(fNames, fVals):
                    argDocStr += '{0}| {1} | {2} |\n'.format(
                        myTab,
                        fName + (maxFNLen - len(fName)) * ' ',
                        fVal + (maxFVLen - len(fVal)) * ' ')
                    argDocStr += spacerH
        argDocStr += '\n'
        if spyder.isSphinx:
            self.webHelp.page().setLinkDelegationPolicy(0)
            cntx = spyder.generate_context(
                name="OpenCL Platforms and Devices",
                argspec="",
                note="",
                math=True)
            argDocStr = argDocStr.replace('imagezoom::', 'image::')
            html_text = spyder.sphinxify(textwrap.dedent(argDocStr), cntx)
            self.webHelp.setHtml(html_text, qt.QUrl(spyder.CSS_PATH))
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
        child1.setFlags(self.valueFlag)
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

    def objToInstance(self, obj):
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

    def addElement(self, name, obj, copyFrom=None):
        if copyFrom is not None:
            for i in range(copyFrom.rowCount()):
                if str(copyFrom.child(i, 0).text()) == '_object':
                    obj = self.getVal(copyFrom.child(i, 1).text())
                    name = obj.__name__
                    obj = "{0}.{1}".format(obj.__module__, name)
                    break

        for i in range(99):
            elementName = self.classNameToStr(name) + '{:02d}'.format(i+1)
            dupl = False
            for ibm in range(self.rootBLItem.rowCount()):
                if str(self.rootBLItem.child(ibm, 0).text()) ==\
                        str(elementName):
                    dupl = True
            if not dupl:
                break
        self.blUpdateLatchOpen = False
        elementItem, elementClass = self.addParam(self.rootBLItem,
                                                  elementName,
                                                  self.objToInstance(obj),
                                                  source=copyFrom)
        elementClass.setFlags(self.objectFlag)
        elementItem.setFlags(self.valueFlag)
        if copyFrom is not None:
            self.cpChLevel = 0
            self.copyChildren(elementItem, copyFrom)
        else:
            elementItem.setDragEnabled(True)
            elprops = self.addProp(elementItem, 'properties')
            self.addObject(self.tree, elementItem, obj)
            for arg, argVal in self.getParams(obj):
                self.addParam(elprops, arg, argVal)
        self.showDoc(elementItem.index())
        self.addCombo(self.tree, elementItem)
        self.tree.expand(self.rootBLItem.index())
        self.capitalize(self.tree, elementItem)
        self.blUpdateLatchOpen = True
        self.updateBeamline(elementItem, newElement=True)
        if not self.experimentalMode:
            self.autoAssignMethod(elementItem)
        self.isEmpty = False
        self.tabs.setCurrentWidget(self.tree)

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
                                    argVal = self.rootBLItem.text()
                                if arg not in args and arg not in hpList:
                                    uArgs[arg] = argVal
#                                    args.append(arg)
#                                    argVals.append(argVal)
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
#                                        args.append(arg)
#                                        argVals.append(argVal)
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

    def beamLineItemChanged(self, item):
        self.colorizeChangedParam(item)
        if self.blUpdateLatchOpen:
            if item.model() == self.beamLineModel:
                self.updateBeamline(item)
            elif item.model() == self.materialsModel:
                self.updateBeamlineMaterials(item)

    def colorizeChangedParam(self, item):
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

#    def colorizeTabText(self, item):
#        if item.model() == self.beamLineModel:
#            color = qt.red if self.blColorCounter > 0 else\
#                qt.black
#            self.tabs.tabBar().setTabTextColor(0, color)
#        elif item.model() == self.plotModel:
#            color = qt.red if self.pltColorCounter > 0 else\
#                qt.black
#            self.tabs.tabBar().setTabTextColor(2, color)

    def addMethod(self, name, parentItem, fdoc):
        self.beamModel.sort(3)
        elstr = str(parentItem.text())
        fdoc = fdoc[0].replace("Returned values: ", '').split(',')
        self.blUpdateLatchOpen = False
        methodItem = self.addProp(parentItem, name.split('.')[-1] + '()')
        self.setIItalic(methodItem)
        methodProps = self.addProp(methodItem, 'parameters')
        self.addObject(self.tree, methodItem, name)
        for arg, argVal in self.getParams(name):
                if arg == 'bl':
                    argVal = self.rootBLItem.text()
                elif 'beam' in arg:
                    fModel0 = qt.QSortFilterProxyModel()
                    fModel0.setSourceModel(self.beamModel)
                    fModel0.setFilterKeyColumn(1)
                    fModel0.setFilterRegExp('Global')
                    fModel = qt.QSortFilterProxyModel()
                    fModel.setSourceModel(fModel0)
                    fModel.setFilterKeyColumn(3)
                    regexp = self.intToRegexp(
                        self.nameToBLPos(elstr))
                    fModel.setFilterRegExp(regexp)
                    lastIndex = fModel.rowCount() - 1
                    if arg.lower() == 'accubeam':
                        lastIndex = 0
                    argVal = fModel.data(fModel.index(lastIndex, 0))
                child0, child1 = self.addParam(methodProps, arg, argVal)

        methodOut = self.addProp(methodItem, 'output')
        for outstr in fdoc:
            outval = outstr.strip()

            for i in range(99):
                beamName = '{0}{1}{2:02d}'.format(elstr, outval, i+1)
                dupl = False
                for ibm in range(self.beamModel.rowCount()):
                    if str(self.beamModel.index(ibm, 0).data(0)) ==\
                            str(beamName):
                        dupl = True
                if not dupl:
                    break

            child0, child1 = self.addParam(methodOut, outval, beamName)
            if 'shine' in name:
                outval += 'Local'
            self.beamModel.appendRow([qt.QStandardItem(beamName),
                                      qt.QStandardItem(outval),
                                      qt.QStandardItem(elstr),
                                      qt.QStandardItem(str(self.nameToBLPos(
                                          elstr)))])
            try:
                self.beamLine.beamsDict[beamName] = None
            except KeyError:
                pass

        self.showDoc(methodItem.index())
        self.addCombo(self.tree, methodItem)
        self.tree.expand(methodItem.index())
        self.tree.expand(methodOut.index())
        self.tree.expand(methodProps.index())
        self.tree.setCurrentIndex(methodProps.index())
        self.tree.setColumnWidth(0, int(self.tree.width()/3))
        self.blUpdateLatchOpen = True
        self.updateBeamline(methodItem, newElement=True)
        self.isEmpty = False

    def addPlot(self, copyFrom=None):
        for i in range(99):
            plotName = 'plot{:02d}'.format(i+1)
            dupl = False
            for ibm in range(self.rootPlotItem.rowCount()):
                if str(self.rootPlotItem.child(ibm, 0).text()) ==\
                        str(plotName):
                    dupl = True
            if not dupl:
                break

        plotItem = self.addValue(self.rootPlotItem, plotName, source=copyFrom)

        if copyFrom is not None:
            self.cpChLevel = 0
            self.copyChildren(plotItem, copyFrom)
        else:
            for name, obj in inspect.getmembers(xrtplot):
                if name == "XYCPlot" and inspect.isclass(obj):
                    self.addObject(self.plotTree, plotItem,
                                   '{0}.{1}'.format(xrtplot.__name__, name))
                    for namef, objf in inspect.getmembers(obj):
                        if inspect.ismethod(objf) or inspect.isfunction(objf):
                            if namef == "__init__" and\
                                    inspect.getargspec(objf)[3] is not None:
                                for arg, arg_def in zip(
                                        inspect.getargspec(objf)[0][1:],
                                        inspect.getargspec(objf)[3]):
                                    # child0 = qt.QStandardItem(str(arg))
                                    # child0.setFlags(self.paramFlag)
                                    if len(re.findall("axis",
                                                      arg.lower())) > 0:
                                        child0 = self.addProp(plotItem,
                                                              str(arg))
                                        for name2, obj2 in\
                                                inspect.getmembers(xrtplot):
                                            if name2 == "XYCAxis" and\
                                                    inspect.isclass(obj2):
                                                self.addObject(self.plotTree,
                                                               child0,
                                                               '{0}.{1}'.format( # analysis:ignore
                                                                  xrtplot.__name__, # analysis:ignore
                                                                  name2))
                                                for namef2, objf2 in\
                                                        inspect.getmembers(
                                                            obj2):
                                                    if (inspect.ismethod(
                                                        objf2) or
                                                        inspect.isfunction(
                                                            objf2)):
                                                        argsList =\
                                                            inspect.getargspec(
                                                                objf2)
                                                        if namef2 ==\
                                                            "__init__" and\
                                                            argsList[3] is\
                                                                not None:
                                                            for arg2, arg_def2\
                                                                in zip(
                                                                    argsList[0][1:], # analysis:ignore
                                                                    argsList[3]): # analysis:ignore
                                                                if str(arg) == 'caxis' and str(arg2) == 'unit': # analysis:ignore
                                                                    arg_def2 = 'eV' # analysis:ignore
                                                                self.addParam(child0, arg2, arg_def2) # analysis:ignore
                                    else:
                                        if str(arg) == 'title':
                                            arg_value = plotItem.text()
                                        else:
                                            arg_value = arg_def
                                        self.addParam(plotItem, arg, arg_value)
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

    def addMaterial(self, name, obj, *args):
        for i in range(99):
            matName = self.classNameToStr(name) + '{:02d}'.format(i+1)
            dupl = False
            for ibm in range(self.rootMatItem.rowCount()):
                if str(self.rootMatItem.child(ibm, 0).text()) == str(matName):
                    dupl = True
            if not dupl:
                break
        self.blUpdateLatchOpen = False
        matItem, matClass = self.addParam(self.rootMatItem,
                                          matName,
                                          self.objToInstance(obj))
        matClass.setFlags(self.objectFlag)
        matItem.setFlags(self.valueFlag)
        matProps = self.addProp(matItem, 'properties')
        self.addObject(self.matTree, matItem, obj)

        for arg, argVal in self.getParams(obj):
            self.addParam(matProps, arg, argVal)
        self.showDoc(matItem.index())
        self.addCombo(self.matTree, matItem)
        self.capitalize(self.matTree, matItem)
        self.blUpdateLatchOpen = True
        self.updateBeamlineMaterials(matItem, newMat=True)
        self.isEmpty = False
        self.tabs.setCurrentWidget(self.matTree)

    def moveItem(self, mvDir, view, item):
        oldRowNumber = item.index().row()
        statusExpanded = view.isExpanded(item.index())
        parent = item.parent()
        item.model().blockSignals(True)
        self.flattenElement(view, item)
        item.model().blockSignals(False)
        newItem = parent.takeRow(oldRowNumber)
        parent.insertRow(oldRowNumber + mvDir, newItem)
        self.addCombo(view, newItem[0])
        view.setExpanded(newItem[0].index(), statusExpanded)
        self.updateBeamline(newItem[0], newOrder=True)

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
                        except:
                            print("Failed to delete", iWidget.currentText())
                            raise
                        beamInModel = self.beamModel.findItems(
                            iWidget.currentText())
                        if len(beamInModel) > 0:
                            self.beamModel.takeRow(beamInModel[0].row())

            self.deleteElement(view, iItem)
        else:
            if item.parent() == self.rootBLItem:
                del self.beamLine.oesDict[str(item.text())]
                self.blUpdateLatchOpen = True
                self.updateBeamline(item, newElement=False)
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
            self.statusBar.showMessage(
                'Failed saving to {}'.format(
                    os.path.basename(str(self.layoutFileName))), 3000)
            self.layoutFileName = tmpName

    def exportModel(self, item):
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
                self.prefixtab, str(item.text()).strip('()'), itemType)
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
                                    child0.text(),
                                    child1.text(),
                                    itemType)
                elif flatModel:
                        self.confText +=\
                            '{0}<{1} type=\"flat\"></{1}>\n'.format(
                                self.prefixtab, child0.text())

            self.ntab -= 1
            self.prefixtab = self.ntab * '\t'
            self.confText += '{0}</{1}>\n'.format(
                self.prefixtab, str(item.text()).strip('()'))
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
                            self.updateBeamImport()
                        if tree is not None:
                            self.checkDefaults(None, rootModel)
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
                        os.path.dirname(os.path.abspath(str(
                            self.layoutFileName))))
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
                    try:
                        self.beamLine = raycing.BeamLine()
                        self.beamLine.flowSource = 'Qook'
                        self.progressBar.setFormat(
                            "Populating the beams... %p%")
                        self.updateBeamlineBeams(item=None)
                        self.progressBar.setValue(60)
                        self.progressBar.setFormat(
                            "Populating the materials... %p%")
                        self.updateBeamlineMaterials(item=None)
                        self.progressBar.setValue(70)
                        self.prbStart = 70
                        self.prbRange = 30
                        self.progressBar.setFormat(
                            "Initializing optical elements... %p%")
                        self.updateBeamline(item=None)
                    except:  # analysis:ignore
                        raise
                        pass
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
                    if rootModel.model() != self.beamModel:
                        child0 = rootModel.appendRow(child0)
                    else:
                        rootModel.appendRow([child0, None, None, None])
                elif itemType == "value":
                    child0 = self.addValue(rootModel, itemTag)
                    if self.ntab == 1:
                        self.capitalize(view, child0)
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
                                                qt.QStandardItem(elNameStr))
                                            self.rootBeamItem.setChild(
                                                irow, 3,
                                                qt.QStandardItem(str(
                                                    self.nameToBLPos(
                                                        elNameStr))))

    def intToRegexp(self, intStr):
        a = list(str(int(intStr)))
        oeClassStr = str(self.getClassName(
            self.rootBLItem.child(int(intStr), 0)))
        if 'source' in oeClassStr:
            return '^(\d+)$'
        elif int(intStr) < 11:
            return '^([0][0][0-{}])$'.format(int(intStr)-1)
        else:
            return '^([0][0][0-9]|[0][0-{0}][0-9]{1}$'.format(
                int(a[0])-1, '|[0]{0}[0-{1}])'.format(
                    int(a[0]), int(a[1])-1) if int(a[1]) > 0 else ")")

    def addCombo(self, view, item):
        self.beamModel.sort(3)
        if item.hasChildren():
            itemTxt = str(item.text())
            for ii in range(item.rowCount()):
                child0 = item.child(ii, 0)
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
                                    self.nameToBLPos(str(item.parent(
                                    ).parent().text())))
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
                                fModel.setFilterRegExp(self.nameToBLPos(str(
                                    item.parent().parent().text())))
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
                                combo.setEditable(True)
                                combo.setInsertPolicy(2)
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
                                    self.nameToBLPos(str(item.parent(
                                    ).parent().text())))
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
                                combo.setEditable(True)
                                combo.setInsertPolicy(2)

                        elif paramName == "bl" or\
                                paramName == "beamLine":
                            combo = self.addEditableCombo(
                                self.beamLineModel, value)
                            combo.setInsertPolicy(2)
                            view.setIndexWidget(child1.index(), combo)
                        elif paramName == "plots":
                            combo = self.addEditableCombo(
                                self.plotModel, value)
                            combo.setInsertPolicy(2)
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
                                    partial(self.processListWidget, child1))
                            elif combo.staticMetaObject.className() ==\
                                    'QComboBox':
                                combo.currentIndexChanged['QString'].connect(
                                    child1.setText)
                self.addCombo(view, child0)
        else:
            pass

    def processListWidget(self, item):
        iWidget = self.sender()
        chItemText = "("
        for rState in range(iWidget.count()):
            if int(iWidget.item(rState).checkState()) == 2:
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
        combo.setInsertPolicy(1)
        return combo

    def addStandardCombo(self, model, value):
        combo = qt.QComboBox()
        combo.setModel(model)
        combo.setCurrentIndex(combo.findText(value))
        return combo

    def autoAssignMethod(self, elItem):
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
                        self.addMethod(objfNm, elItem, fdoc)
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
                           partial(self.addElement, None, None, selectedItem))
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
                            methAction.triggered.connect(
                                partial(self.addMethod, objfNm,
                                        selectedItem, fdoc))
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
                        self._addAction(rmats, mName, self.addMaterial, smenu)
                else:  # as single entry itself
                    self._addAction(rmats, sec, self.addMaterial, matMenu)
        else:  # only with __all__
            for mName in rmats.__all__:
                self._addAction(rmats, mName, self.addMaterial, matMenu)

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
        try:
            return eval(str(value))
        except:  # analysis:ignore
            return str(value)

    def quotize(self, value):
        try:
            dummy = unicode  # test for Python3 compatibility analysis:ignore
        except NameError:
            unicode = str
        if isinstance(self.getVal(value), (str, unicode)):
            if 'np.' not in value and\
                    (str(self.rootBLItem.text())+'.') not in value:
                value = 'r\"{}\"'.format(value)
        if str(value) == 'round':
            value = 'r\"{}\"'.format(value)
        if isinstance(self.getVal(value), tuple):
            value = list(self.getVal(value))
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

    def updateBeamlineBeams(self, item):
        sender = self.sender()
        if sender is not None:
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
                    bList.append(self.rootBeamItem.child(ib, 0))
                for key, value in self.beamLine.beamsDict.items():
                    if key not in bList:
                        del self.beamLine.beamsDict[key]
                        self.beamLine.beamsDict[str(item.text())] = value
                        break

    def updateBeamlineMaterials(self, item, newMat=False):
        def createParamDict(parentItem, elementString):
            kwargs = dict()
            for iep, arg_def in zip(range(parentItem.rowCount()), list(zip(
                    *self.getParams(elementString)))[1]):
                paraname = str(parentItem.child(iep, 0).text())
                paravalue = str(parentItem.child(iep, 1).text())
                if paravalue != str(arg_def) or\
                        paravalue == 'bl':
                    if paraname.lower() in ['tlayer', 'blayer', 'coating',
                                            'substrate']:
                        paravalue =\
                            self.beamLine.materialsDict[paravalue]
                    else:
                        paravalue = self.parametrize(paravalue)
                    kwargs[paraname] = paravalue
            return kwargs

        if item is None:
            self.beamLine.materialsDict = OrderedDict({'None': None})
            for ie in range(self.rootMatItem.rowCount()):
                matItem = self.rootMatItem.child(ie, 0)
                matName = str(matItem.text())
                if matName != "None":
                    matClassStr = self.getClassName(matItem)
                    for ieph in range(matItem.rowCount()):
                        if matItem.child(ieph, 0).text() == 'properties':
                            kwArgs = createParamDict(
                                matItem.child(ieph, 0), matClassStr)
                            break
                    try:
                        self.beamLine.materialsDict[matName] =\
                            eval(matClassStr)(**kwArgs)
                        self.progressBar.setFormat(
                            "Class {} successfully initialized.".format(
                                matName))
                        print("Class", matName, "successfully initialized.")
                    except:  # analysis:ignore
                        self.beamLine.materialsDict[matName] = None
                        self.progressBar.setFormat(
                            "Incorrect parameters. Class {} not initialized.".format(  # analysis:ignore
                                matName))
                        print("Incorrect parameters. Class", matName,
                              "not initialized.")
        else:
            if item.index().column() == 0 and not newMat:  # Rename material
                matValues = list(self.beamLine.materialsDict.values())
                matKeys = list(self.beamLine.materialsDict.keys())
                blMats = OrderedDict({'None': None})
                counter = 0
                for ie in range(self.rootMatItem.rowCount()):
                    matItemStr = str(self.rootMatItem.child(
                        ie, 0).text())
                    if matItemStr != 'None':
                        blMats[matItemStr] = matValues[counter]
                        if matItemStr != matKeys[counter]:
                            print("Material", matKeys[counter], "renamed to",
                                  matItemStr)
                    counter += 1
                self.beamLine.materialsDict = blMats
            else:  # New material or property changed, init one material
                if newMat:
                    matItem = item
                    for ie in range(item.rowCount()):
                        if item.child(ie, 0).text() == 'properties':
                            propItem = item.child(ie, 0)
                            break
                else:
                    matItem = item.parent().parent()
                    propItem = item.parent()
                matClassStr = self.getClassName(matItem)
                kwArgs = createParamDict(propItem, matClassStr)
                matName = str(matItem.text())
                try:
                    if newMat:
                        self.beamLine.materialsDict[matName] = eval(
                            matClassStr)(**kwArgs)
                    elif self.beamLine.materialsDict[matName] is None:
                        self.beamLine.materialsDict[matName] = eval(
                            matClassStr)(**kwArgs)
                    else:
                        self.beamLine.materialsDict[matName].__init__(**kwArgs)
                    self.progressBar.setFormat(
                        "Class {} successfully initialized.".format(matName))
                    print("Class", matName, "successfully initialized.")
                except:  # analysis:ignore
                    self.beamLine.materialsDict[matName] = None
                    self.progressBar.setFormat(
                        "Incorrect parameters. Class {} not initialized.".format(  # analysis:ignore
                            matName))
                    print("Incorrect parameters. Class", matName,
                          "not initialized.")
                startFrom = None
                if matName is not None:
                    for iel in range(self.rootBLItem.rowCount()):
                        eItem = self.rootBLItem.child(iel, 0)
                        elNameStr = str(eItem.text())
                        for ich1 in range(eItem.rowCount()):
                            if str(eItem.child(ich1, 0).text()) ==\
                                    'properties':
                                pItem = eItem.child(ich1, 0)
                                for ich2 in range(pItem.rowCount()):
                                    if str(pItem.child(ich2, 1).text()) ==\
                                            matName:
                                        startFrom =\
                                            self.nameToFlowPos(
                                                elNameStr)
                                        break
                                else:
                                    continue
                                break
                        else:
                            continue
                        break
                    if startFrom is not None:
                        self.blPropagateFlow(startFrom)

    def nameToFlowPos(self, elementNameStr):
        retVal = 0
        try:
            for isegment, segment in enumerate(self.beamLine.flow):
                if segment[0] == elementNameStr:
                    retVal = isegment
                    break
        except:  # analysis:ignore
            pass
        return retVal

    def nameToBLPos(self, elname):
        for iel in range(self.rootBLItem.rowCount()):
            if str(self.rootBLItem.child(iel, 0).text()) == elname:
                return '{:03d}'.format(iel)
        else:
            return '000'

    def updateBeamline(self, item=None, newElement=False, newOrder=False):
        def createParamDict(parentItem, elementString):
            kwargs = dict()
            for iep, arg_def in zip(range(
                    parentItem.rowCount()),
                    list(zip(*self.getParams(elementString)))[1]):
                paraname = str(parentItem.child(iep, 0).text())
                paravalue = str(parentItem.child(iep, 1).text())
                if paravalue != str(arg_def) or\
                        paraname == 'bl':
                    if paraname == 'center':
                        paravalue = paravalue.strip('[]() ')
                        paravalue =\
                            [self.getVal(c.strip())
                             for c in str.split(
                             paravalue, ',')]
                    elif paraname.startswith('material'):
                        paravalue =\
                            self.beamLine.materialsDict[paravalue]
                    elif paraname == 'bl':
                        paravalue = self.beamLine
                    else:
                        paravalue = self.parametrize(paravalue)
                    kwargs[paraname] = paravalue
            return kwargs

        def createMethodDict(elementItem, elementString):
            methodObj = None
            for ieph in range(elementItem.rowCount()):
                pItem = elementItem.child(ieph, 0)
                methodObj = None
                if str(pItem.text()) not in ['_object', 'properties']:
                    for namef, objf in inspect.getmembers(eval(elementString)):
                        if (inspect.ismethod(objf) or
                                inspect.isfunction(objf)) and\
                                namef == str(pItem.text()).strip('()'):
                            methodObj = objf
                    inkwargs = {}
                    outkwargs = OrderedDict()
                    for imet in range(pItem.rowCount()):
                        mItem = pItem.child(imet, 0)
                        if str(mItem.text()) == 'parameters':
                            for iep, arg_def in\
                                zip(range(mItem.rowCount()),
                                    inspect.getargspec(methodObj)[3]):
                                paraname = str(mItem.child(
                                    iep, 0).text())
                                paravalue = self.parametrize(str(mItem.child(
                                    iep, 1).text()))
                                if len(re.findall('beam', paraname)) > 0 and\
                                    self.beamLine.oesDict[str(elementItem.text(
                                        ))][1] and paravalue is None:
                                    return None, None, None
                                inkwargs[paraname] = paravalue
                        elif str(mItem.text()) == 'output':
                            for iep in range(mItem.rowCount()):
                                paraname = str(mItem.child(
                                    iep, 0).text())
                                paravalue = str(mItem.child(iep, 1).text())
                                outkwargs[paraname] = paravalue
            if methodObj is not None:
                return methodObj, inkwargs, outkwargs
            else:
                return None, None, None

        def buildFlow(startFrom=0):
            blFlow = []
            for ie in range(self.rootBLItem.rowCount()):
                elItem = self.rootBLItem.child(ie, 0)
                elName = str(elItem.text())
                if elName not in ["properties", "_object"]:
                    elStr = self.getClassName(elItem)
                    methodObj, inkwArgs, outkwArgs = createMethodDict(
                        elItem, elStr)
                    if methodObj is not None:
                        blFlow.append([elName, methodObj,
                                       inkwArgs, outkwArgs])
            return blFlow

        def updateRegexp():
            for iElement in range(self.rootBLItem.rowCount()):
                elItem = self.rootBLItem.child(iElement, 0)
                if str(elItem.text()) not in ['properties', '_object']:
                    for iProp in range(elItem.rowCount()):
                        propItem = elItem.child(iProp, 0)
                        if str(propItem.text()) not in ['properties',
                                                        '_object']:
                            for iMeth in range(propItem.rowCount()):
                                methItem = propItem.child(iMeth, 0)
                                if str(methItem.text()) == 'parameters':
                                    for iBeam in range(methItem.rowCount()):
                                        bItem = methItem.child(iBeam, 0)
                                        if len(re.findall(
                                                'beam', str(
                                                bItem.text()))) > 0:
                                            vItem = methItem.child(iBeam, 1)
                                            iWidget = self.tree.indexWidget(
                                                vItem.index())
                                            if iWidget is not None:
                                                try:
                                                    regexp =\
                                                        self.intToRegexp(
                                                            iElement)
                                                    iWidget.model(
                                                        ).setFilterRegExp(
                                                            regexp)
                                                except:  # analysis:ignore
                                                    continue
                                elif str(methItem.text()) == 'output':
                                    for iBeam in range(methItem.rowCount()):
                                        bItem = methItem.child(iBeam, 0)
                                        if len(re.findall(
                                                'beam', str(
                                                bItem.text()))) > 0:
                                            vItem = methItem.child(iBeam, 1)
                                            iWidget = self.tree.indexWidget(
                                                vItem.index())
                                            if iWidget is not None:
                                                try:
                                                    regexp = '{:03d}'.format(
                                                        iElement)
                                                    iWidget.model(
                                                        ).setFilterRegExp(
                                                            regexp)
                                                except:  # analysis:ignore
                                                    continue
        self.rootBLItem.model().blockSignals(True)
        self.flattenElement(self.tree,
                            self.rootBLItem if item is None else item)
        self.rootBLItem.model().blockSignals(False)

        if item is not None:
            if item.index().parent().isValid():  # not the Beamline root
                iCol = item.index().column()
                pText = str(item.parent().text())
                if pText == str(self.rootBLItem.text()):
                    if newElement:  # New element added
                        elNameStr = str(item.text())
                        elClassStr = self.getClassName(item)
                        for iep in range(item.rowCount()):
                            if item.child(iep, 0).text() == 'properties':
                                propItem = item.child(iep, 0)
                                break
                        oeType = 0 if len(re.findall(
                            'raycing.sou', elClassStr)) > 0 else 1
                        try:
                            kwArgs = createParamDict(propItem, elClassStr)
                            self.beamLine.oesDict[elNameStr] =\
                                [eval(elClassStr)(**kwArgs), oeType]
                            self.progressBar.setFormat(
                                "Class {} successfully initialized.".format(
                                    elNameStr))
                            print("Class", elNameStr,
                                  "successfully initialized.")
                        except:  # analysis:ignore
                            self.beamLine.oesDict[elNameStr] =\
                                [None, oeType]
                            self.progressBar.setFormat(
                                "Incorrect parameters. Class {} not initialized.".format(  # analysis:ignore
                                    elNameStr))
                            print("Incorrect parameters. Class", elNameStr,
                                  "not initialized.")
                        self.beamLine.flow = buildFlow(startFrom=elNameStr)
                        startFrom = self.nameToFlowPos(elNameStr)
                    else:  # Element renamed or moved
                        oesValues = list(self.beamLine.oesDict.values())
                        oesKeys = list(self.beamLine.oesDict.keys())
                        wasDeleted = True if\
                            len(oesKeys) + 2 < self.rootBLItem.rowCount()\
                            else False
                        if not wasDeleted:
                            newDict = OrderedDict()
                            counter = 0
                            startElement = None
                            rbi = self.rootBeamItem
                            for ie in range(self.rootBLItem.rowCount()):
                                elNameStr =\
                                    str(self.rootBLItem.child(ie, 0).text())
                                if elNameStr not in ["properties", "_object"]:
                                    if newOrder:
                                        newDict[elNameStr] =\
                                            self.beamLine.oesDict[elNameStr]
                                        if elNameStr != oesKeys[counter] and\
                                                startElement is None:
                                            startElement = elNameStr
                                    else:
                                        newDict[elNameStr] = oesValues[counter]
                                        if elNameStr != oesKeys[counter]:
                                            startElement = oesKeys[counter]
                                            for ibeam in range(
                                                    rbi.rowCount()):
                                                if str(rbi.child(
                                                        ibeam, 2).text()) ==\
                                                        startElement:
                                                    rbi.child(
                                                        ibeam, 2).setText(
                                                            elNameStr)
                                            self.progressBar.setFormat(
                                                "Element {0} renamed to {1}".format(  # analysis:ignore
                                                    startElement, elNameStr))
                                            print("Element", startElement,
                                                  "renamed to", elNameStr)
                                    counter += 1
                            self.beamLine.oesDict = newDict
                        if newOrder:
                            self.progressBar.setFormat(
                                "Element {} moved to new position".format(
                                    item.text()))
                            print("Element", item.text(),
                                  "moved to new position")
                            for ibeam in range(rbi.rowCount()):
                                rbi.child(ibeam, 3).setText(str(
                                    self.nameToBLPos(str(rbi.child(
                                        ibeam, 2).text()))))
                            self.beamModel.sort(3)
                            updateRegexp()
                        elif wasDeleted:
                            self.progressBar.setFormat(
                                "Element {} was removed".format(item.text()))
                            print("Element", item.text(),
                                  "was removed")
                            startElement = str(item.text())
                            self.beamLine.flow =\
                                buildFlow(startFrom=startElement)
                        else:
                            for iel in range(len(self.beamLine.flow)):
                                if self.beamLine.flow[iel][0] == startElement:
                                    self.beamLine.flow[iel][0] =\
                                        str(item.text())
                        startFrom = self.nameToFlowPos(startElement)
                elif pText in ['properties'] and iCol > 0:
                    elItem = item.parent().parent()
                    elNameStr = str(elItem.text())
                    elClassStr = self.getClassName(elItem)
                    if len(re.findall('.BeamLine', elClassStr)) > 0:  # BL
                        paramName = str(item.parent().child(item.index().row(),
                                                            0).text())
                        paramValue = self.parametrize(str(item.text()))
                        setattr(self.beamLine, paramName, paramValue)
                        startFrom = 0
                    else:  # Setters and getters not implemented yet for OEs
                        oeType = 0 if len(re.findall(
                            'raycing.sou', elClassStr)) > 0 else 1
                        try:
                            kwargs = createParamDict(item.parent(),
                                                     elClassStr)
                            if self.beamLine.oesDict[elNameStr][0] is None:
                                self.beamLine.oesDict[elNameStr] =\
                                    [eval(elClassStr)(**kwargs), oeType]
                                self.progressBar.setFormat(
                                    "Class {} successfully initialized.".format(  # analysis:ignore
                                        elNameStr))
                                print("Class", elNameStr,
                                      "successfully initialized.")
                            else:
                                self.beamLine.oesDict[elNameStr][0].__init__(**kwargs)  # analysis:ignore
                                self.progressBar.setFormat(
                                    "Class {} successfully re-initialized.".format(  # analysis:ignore
                                        elNameStr))
                                print("Class", elNameStr,
                                      "successfully re-initialized.")
                        except:  # analysis:ignore
                            self.beamLine.oesDict[elNameStr] =\
                                [None, oeType]
                            self.progressBar.setFormat(
                                "Incorrect parameters. Class {} not initialized.".format(elNameStr))  # analysis:ignore
                            print("Incorrect parameters. Class", elNameStr,
                                  "not initialized.")
                        if len(re.findall('raycing.aper', elClassStr)) > 0:
                            if self.rayPath is not None:
                                for segment in self.rayPath[0]:
                                    if segment[2] == elNameStr:
                                        startFrom =\
                                            self.nameToFlowPos(segment[0])
                                        break
                                else:
                                    startFrom =\
                                        self.nameToFlowPos(elNameStr)
                            else:
                                startFrom = self.nameToFlowPos(elNameStr)
                        else:
                            startFrom = self.nameToFlowPos(elNameStr)
                elif pText in ['parameters', 'output'] and iCol > 0:
                    elItem = item.parent().parent().parent()
                    elNameStr = str(elItem.text())
                    self.progressBar.setFormat(
                        "Method of {} was modified".format(elNameStr))
                    print("Method of", elNameStr, "was modified")
                    self.beamLine.flow = buildFlow(startFrom=elNameStr)
                    startFrom = self.nameToFlowPos(elNameStr)
                elif item.parent().parent() == self.rootBLItem and newElement:
                    elItem = item.parent()
                    elNameStr = str(elItem.text())
                    self.progressBar.setFormat(
                        "Method {0} was added to {1}".format(
                            item.text(), elNameStr))
                    print("Method", item.text(), "was added to", elNameStr)
                    self.beamLine.flow = buildFlow(startFrom=elNameStr)
                    startFrom = self.nameToFlowPos(elNameStr)
        else:  # Rebuild beamline
            for ie in range(self.rootBLItem.rowCount()):
                elItem = self.rootBLItem.child(ie, 0)
                elNameStr = str(elItem.text())
                if elNameStr == 'properties':  # Beamline properties
                    for iprop in range(elItem.rowCount()):
                        paramName = str(elItem.child(iprop, 0).text())
                        paramValue = self.parametrize(str(elItem.child(
                            iprop, 1).text()))
                        setattr(self.beamLine, paramName, paramValue)
                elif elNameStr != '_object':  # Beamline element
                    for iprop in range(elItem.rowCount()):
                        pItem = elItem.child(iprop, 0)
                        pText = str(pItem.text())
                        if pText == 'properties':  # OE properties
                            elClassStr = self.getClassName(elItem)
                            oeType = 0 if len(re.findall(
                                'raycing.sou', elClassStr)) > 0 else 1
                            try:
                                kwArgs = createParamDict(pItem, elClassStr)
                                self.beamLine.oesDict[elNameStr] =\
                                    [eval(elClassStr)(**kwArgs), oeType]
                                self.progressBar.setFormat(
                                    "Class {} successfully initialized.".format(  # analysis:ignore
                                        elNameStr))
                                print("Class", elNameStr,
                                      "successfully initialized.")
                            except:  # analysis:ignore
                                self.beamLine.oesDict[elNameStr] =\
                                    [None, oeType]
                                self.progressBar.setFormat(
                                    "Incorrect parameters. Class {} not initialized.".format(elNameStr))  # analysis:ignore
                                print("Incorrect parameters. Class", elNameStr,
                                      "not initialized.")
            self.beamLine.flow = buildFlow()
            startFrom = 0

        if self.isGlowAutoUpdate:
            if startFrom is not None:
                self.blPropagateFlow(startFrom)

    def blPropagateFlow(self, startFrom):
        objThread = qt.QThread(self)
        obj = PropagationConnect()
        obj.finished.connect(objThread.quit)
        obj.rayPathReady.connect(self.blRunGlow)
        obj.propagationProceed.connect(self.updateProgressBar)
        obj.moveToThread(objThread)
        propagateFlowInThread = partial(
            obj.propagateFlowThread, self.beamLine, startFrom)
        objThread.started.connect(propagateFlowInThread)
        objThread.start()

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
            pass
        self.blUpdateLatchOpen = True

    def toggleGlow(self, status):
        self.isGlowAutoUpdate = status
        if self.isGlowAutoUpdate:
            self.populateBeamline()

    def blRunGlow(self, rayPath):
        self.rayPath = rayPath
        if self.blViewer is None:
            self.blViewer = xrtglow.xrtGlow(self.rayPath, self)
            self.blViewer.setWindowTitle("xrtGlow")
            self.blViewer.show()
            self.blViewer.parentRef = self
            self.blViewer.parentSignal = self.statusUpdate
        else:
            self.blViewer.updateOEsList(self.rayPath)
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
                        ieinit = elstr + "(" + ieinit
                        break
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
        for ie in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ie, 0).text() != "properties" and\
                    self.rootBLItem.child(ie, 0).text() != "_object":
                tItem = self.rootBLItem.child(ie, 0)
                ieinit = ""
                ierun = ""
                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() == '_object':
                        elstr = str(tItem.child(ieph, 1).text())
                        ieinit = elstr + "(" + ieinit

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
                                    paravalue = re.findall('\[(.*)\]',
                                                           paravalue)[0]
                                elif paravalue.startswith('('):
                                    paravalue = re.findall('\((.*)\)',
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
                                methodObj = objf
                        for imet in range(pItem.rowCount()):
                            if str(pItem.child(imet, 0).text()) ==\
                                    'parameters':
                                mItem = pItem.child(imet, 0)
                                for iep, arg_def in\
                                    zip(range(mItem.rowCount()),
                                        inspect.getargspec(methodObj)[3]):
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
        for ie in range(self.rootPlotItem.rowCount()):
            tItem = self.rootPlotItem.child(ie, 0)
            ieinit = "\n{0}{1} = ".format(myTab, tItem.text())
            plotNames.append(str(tItem.text()))
            for ieph in range(tItem.rowCount()):
                if tItem.child(ieph, 0).text() == '_object':
                    elstr = str(tItem.child(ieph, 1).text())
                    ieinit += elstr + "("
                    for parent in (inspect.getmro(eval(elstr)))[:-1]:
                        for namef, objf in inspect.getmembers(parent):
                            if (inspect.ismethod(objf) or
                                    inspect.isfunction(objf)):
                                if namef == "__init__" and\
                                        inspect.getargspec(
                                        objf)[3] is not None:
                                    obj = objf
            for iepm, arg_def in zip(range(tItem.rowCount()-1),
                                     inspect.getargspec(obj)[3]):
                iep = iepm + 1
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
                                for parentAx in\
                                        (inspect.getmro(eval(axstr)))[:-1]:
                                    for namefAx, objfAx in\
                                            inspect.getmembers(parentAx):
                                        if (inspect.ismethod(objfAx) or
                                                inspect.isfunction(objfAx)):
                                            if namefAx == "__init__" and\
                                                inspect.getargspec(
                                                    objfAx)[3] is not None:
                                                objAx = objfAx
                        for ieaxm, arg_defAx in zip(
                            range(pItem.rowCount()-1),
                                inspect.getargspec(objAx)[3]):
                            ieax = ieaxm + 1
                            paraname = pItem.child(ieax, 0).text()
                            paravalue = pItem.child(ieax, 1).text()
                            if paraname == "data" and paravalue != "auto":
                                paravalue = '{0}.get_{1}'.format(
                                        raycing.__name__, paravalue)
                                ieinit += '\n{2}{0}={1},'.format(
                                        paraname, paravalue, myTab*3)
                            elif str(paravalue) != str(arg_defAx):
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
                        if str(paravalue) != str(arg_def):
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
            fullCode = fullCode.replace(fullModName, xrtAlias)
            codeHeader += 'import {0} as {1}\n'.format(fullModName, xrtAlias)
        fullCode = codeHeader + fullCode
        if self.glowOnly:
            self.glowCode = fullCode
        else:
            if spyder.isSpyderlib:
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
                if spyder.isSpyderConsole:
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
        if spyder.isSpyderConsole:
            self.codeConsole.fname = str(self.saveFileName)
            self.codeConsole.create_process()
        else:
            self.codeConsole.clear()
            self.codeConsole.append('Starting {}\n\n'.format(
                    os.path.basename(str(self.saveFileName))))
            self.codeConsole.append('Press Ctrl+X to terminate process\n\n')
            self.qprocess.start("python", ['-u', str(self.saveFileName)])

    def toggleExperimentalMode(self):
        self.experimentalMode = not self.experimentalMode
        self.progressBar.setFormat("Experimental Mode {}abled".format(
            "en" if self.experimentalMode else "dis"))

    def aboutCode(self):
        import platform
        from ...version import __version__ as xrtversion
#        if use_pyside:
#            Qt_version = qt.__version__
#            PyQt_version = PySide.__version__
#        else:
        Qt_version = qt.QT_VERSION_STR
        PyQt_version = qt.PYQT_VERSION_STR

        msgBox = qt.QMessageBox()
        msgBox.setWindowIcon(qt.QIcon(
            os.path.join(self.xrtQookDir, '_icons', 'xrQt1.ico')))
        msgBox.setWindowTitle("About xrtQook")
        msgBox.setIconPixmap(qt.QPixmap(
            os.path.join(self.xrtQookDir, '_icons', 'logo-xrtQt.png')))
        msgBox.setTextFormat(qt.RichText)
        msgBox.setText("Beamline layout manipulation and automated code\
 generation tool for the <a href='http://xrt.rtfd.io'>xrt ray tracing\
 package</a>.\nFor a quick start see this short \
 <a href='http://xrt.rtfd.io/qook_tutorial.html'>tutorial</a>.")
        locos = platform.platform(terse=True)
        if 'Linux' in locos:
            locos = " ".join(platform.linux_distribution())
        infText = """Created by:\
\nRoman Chernikov (Canadian Light Source)\
\nKonstantin Klementiev (MAX IV Laboratory)\
\nLicensed under the terms of the MIT License\nMarch 2016\
\n\nYour system:\n{0}\nPython {1}\nQt {2}\n{3} {4}""".format(
                locos, platform.python_version(),
                Qt_version, qt.QtName, PyQt_version)
        if isOpenCL:
            vercl = cl.VERSION
            if isinstance(vercl, (list, tuple)):
                vercl = '.'.join(map(str, vercl))
        else:
            vercl = 'not found'
        infText += '\npyopencl {}'.format(vercl)
        if gl.isOpenGL:
            infText += '\n{0} {1}'.format(gl.__name__, gl.__version__)
        infText += '\nxrt {0} in {1}'.format(xrtversion, path_to_xrt)
        msgBox.setInformativeText(infText)
        msgBox.setStandardButtons(qt.QMessageBox.Ok)
        msgBox.exec_()

    def closeEvent(self, event):
        ret = qt.QMessageBox.question(
            self, 'Exit',
            "Do you want to save Beamline Layout before exit?",
            qt.QMessageBox.Yes | qt.QMessageBox.No | qt.QMessageBox.Cancel,
            qt.QMessageBox.Cancel)

        if ret == qt.QMessageBox.Yes:
            if self.exportLayout():
                event.accept()
            else:
                event.ignore()
        elif ret == qt.QMessageBox.Cancel:
            event.ignore()
        else:
            event.accept()


class PropagationConnect(qt.QObject):
    propagationProceed = qt.pyqtSignal(tuple)
    rayPathReady = qt.pyqtSignal(list)
    finished = qt.pyqtSignal()

    def propagateFlowThread(self, blRef, startFrom):
        self.propagationProceed.emit((0.5, "Starting propagation"))
        blRef.propagate_flow(
            startFrom=startFrom,
            signal=self.propagationProceed)
        self.propagationProceed.emit((1, "Preparing data for Glow"))
        rayPath = blRef.export_to_glow(signal=self.propagationProceed)
        self.propagationProceed.emit((1, "Done"))
        self.rayPathReady.emit(rayPath)
        self.finished.emit()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    ex = XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())

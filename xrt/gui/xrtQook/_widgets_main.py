# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:59:25 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import os  # analysis:ignore
import sys  # analysis:ignore
import textwrap  # analysis:ignore
import numpy as np  # analysis:ignore
import re  # analysis:ignore
from datetime import date  # analysis:ignore
import inspect  # analysis:ignore

from functools import partial  # analysis:ignore
from collections import OrderedDict  # analysis:ignore

from ._constants import (redStr, isUnitsEnabled, useSlidersInTree, path_to_xrt,  # analysis:ignore
                         myTab, withSlidersInTree, slidersInTreeScale, _DEBUG_)
from ._objects_custom import (SphinxWorker, LevelRestrictedModel,  # analysis:ignore
                              BusyIconWorker)
from ._widgets_custom import (QWebView, TreeViewEx, PlotViewer,  # analysis:ignore
                              QDockWidgetNoClose)

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
from ...backends.raycing import figure_error as rfe  # analysis:ignore
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

from .. import xrtGlow as xrtglow  # analysis:ignore
from ..xrtGlow import InstanceInspector, is_screen, is_aperture

try:
    from ...backends.raycing import materials_elemental as rmatsel
    from ...backends.raycing import materials_compounds as rmatsco
    from ...backends.raycing import materials_crystals as rmatscr
    pdfMats = True
except ImportError:
    pdfMats = False
    raise ImportError("no predef mats")

if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec


class XrtQook(qt.QMainWindow):
    plotParamUpdate = qt.Signal(tuple)
    statusUpdate = qt.Signal(tuple)
    newElementCreated = qt.Signal(str)
    sig_resized = qt.Signal("QResizeEvent")
    sig_moved = qt.Signal("QMoveEvent")

    TABICONSIZE = 36

    def __init__(self, parent=None, loadLayout=None, projectFile=None):
        super().__init__(parent)
        self.xrtQookDir = os.path.dirname(os.path.abspath(__file__))
        self.setAcceptDrops(True)
        self.xrt_pypi_version = self.check_pypi_version()  # pypi_ver, cur_ver

        self.prbStart = 0
        self.prbRange = 100
        self.busyIconThread = None

        self.prepareViewer = False
        self.callWizard = True
        self.isGlowAutoUpdate = True
#        self.isGlowAutoUpdate = False
        self.experimentalMode = False
        self.experimentalModeFilter = ['propagate_wave',
                                       'diffract', 'expose_wave']
        self.statusUpdate.connect(self.updateProgressBar)
        self.iconsDir = os.path.join(self.xrtQookDir, '_icons')
        self.setWindowIcon(qt.QIcon(os.path.join(self.iconsDir, 'xrQt1.ico')))

        self.xrtModules = ['rsources', 'rscreens',
                           'rmats', 'rmatsel', 'rmatsco', 'rmatscr',
                           'roes', 'rapts', 'rfe',
                           'rrun', 'raycing', 'xrtplot', 'xrtrun']

        self.objectFlag = qt.Qt.ItemFlags(qt.Qt.ItemIsEnabled)
        self.paramFlag = qt.Qt.ItemFlags(qt.Qt.ItemIsEnabled |
                                         qt.Qt.ItemIsSelectable)
        self.valueFlag = qt.Qt.ItemFlags(qt.Qt.ItemIsEnabled |
                                         qt.Qt.ItemIsEditable |
                                         qt.Qt.ItemIsSelectable)
        self.checkFlag = qt.Qt.ItemFlags(qt.Qt.ItemIsEnabled |
                                         qt.Qt.ItemIsUserCheckable |
                                         qt.Qt.ItemIsSelectable)

        self.initAllModels()
        self.initToolBar()
        self.initTabs()

        self.blViewer = None

        # Add worker thread for handling rich text rendering
        self.sphinxThread = qt.QThread(self)
        self.sphinxWorker = SphinxWorker()
        self.sphinxWorker.moveToThread(self.sphinxThread)
        self.sphinxThread.started.connect(self.sphinxWorker.render)
        self.sphinxWorker.html_ready.connect(self._on_sphinx_thread_html_ready)

        mainBox = qt.QVBoxLayout()
        mainBox.setContentsMargins(0, 0, 0, 0)
        mainBox.addWidget(self.toolBar)
        tabsLayout = qt.QHBoxLayout()
        tabsLayout.addWidget(self.vToolBar)
        tabsLayout.addWidget(self.tabs)
#        mainBox.addWidget(self.tabs)
        mainBox.addItem(tabsLayout)
#        mainBox.addWidget(self.statusBar)
        mainBox.addWidget(self.progressBar)
        mainWidget = qt.QWidget()
        mainWidget.setMinimumWidth(240)
        mainWidget.setLayout(mainBox)

        if ext.isSphinx:
            self.webHelp = QWebView()
            self.webHelp.page().setLinkDelegationPolicy(2)
            self.webHelp.setContextMenuPolicy(qt.Qt.CustomContextMenu)
            self.webHelp.customContextMenuRequested.connect(self.docMenu)

            self.lastBrowserLink = ''
            self.webHelp.page().linkClicked.connect(
                partial(self.linkClicked), type=qt.Qt.UniqueConnection)
        else:
            self.webHelp = qt.QTextEdit()
            self.webHelp.setFont(self.defaultFont)
            self.webHelp.setReadOnly(True)
        self.webHelp.setMinimumWidth(240)
        self.webHelp.setMinimumHeight(620)

        self.setCentralWidget(mainWidget)
        self.initAllTrees()
        self.blRunGlow()
        self.initDocWidgets()
        style = "QMainWindow::separator {width: 7px;} " \
            "QMainWindow::separator:hover {background-color: #6087cefa;}"
        self.setStyleSheet(style)

        self.newElementCreated.connect(self.runElementViewer)

        self.showWelcomeScreen()

        if loadLayout is not None or projectFile is not None:
            self.importLayout(layoutJSON=loadLayout, filename=projectFile)

    def initDocWidgets(self):
        self.setTabPosition(qt.Qt.AllDockWidgetAreas,
                            qt.QTabWidget.North)
        dockFeatures = (qt.QDockWidget.DockWidgetMovable |
                        qt.QDockWidget.DockWidgetFloatable)

        self.tabNames = "Live Doc", "xrtGlow"
        self.tabNameGlow = self.tabNames[1]
        tabWidgets = self.webHelp, self.blViewer
        tabIcons = "icon-help.png", "3dg_256.png"
        self.docks = []
        for i, (tabName, w, tabIcon) in enumerate(zip(
                self.tabNames, tabWidgets, tabIcons)):
            dock = QDockWidgetNoClose(tabName, self)
            dock.setAllowedAreas(qt.Qt.RightDockWidgetArea)
            dock.setFeatures(dockFeatures)
            dock.topLevelChanged.connect(dock.changeWindowFlags)
            self.addDockWidget(qt.Qt.RightDockWidgetArea, dock)
            dock.setWidget(w)
            dock.dockIcon = qt.QIcon(os.path.join(self.iconsDir, tabIcon))
            if i == 0:
                dock0 = dock
            else:
                self.tabifyDockWidget(dock0, dock)
            self.docks.append(dock)

        self.tabWidget = None
        for tab in self.findChildren(qt.QTabBar):
            if tab.tabText(0) == self.tabNames[0]:
                self.tabWidget = tab
                break
        style = "QTabWidget>QWidget>QWidget {background: palette(window);}"\
            "QTabBar::tab {padding: 2px 10px 0px 10px;"\
            "margin-left: 1px; margin-right: 1px; IB} "\
            "QTabBar::tab:hover {background: #6087cefa;}"\
            "QTabBar::tab:selected {border-top: 3px solid lightblue; "\
            "font-weight: 700; AB}"
        # if csi.onMac:
        AB = f"background: white; height: {self.TABICONSIZE};"
        IB = f"height: {self.TABICONSIZE};"
        style = style.replace("AB", AB).replace("IB", IB)
        self.tabWidget.setStyleSheet(style)
        iconSize = int(self.TABICONSIZE)
        self.tabWidget.setIconSize(qt.QSize(iconSize, iconSize))

        self.setTabIcons()

    def setTabIcons(self):
        for dock, tabName in zip(self.docks, self.tabNames):
            # tab order is unknown:
            for itab in range(self.tabWidget.count()):
                if self.tabWidget.tabText(itab) == tabName:
                    break
            else:
                continue
            self.tabWidget.setTabIcon(itab, dock.dockIcon)

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
        elAction.hovered.connect(partial(self.showObjHelp, objName))
        elAction.triggered.connect(partial(afunction, elname, objName))
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
        loadBLAction.triggered.connect(partial(self.importLayout, None, None))

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
            qt.QIcon(os.path.join(self.iconsDir, '3dg_256.png')),
            'Enable xrtGlow Live Update',
            self)
        if gl.isOpenGL:
            glowAction.setShortcut('CTRL+F1')
            glowAction.setCheckable(True)
            glowAction.setChecked(True)
            glowAction.toggled.connect(self.toggleGlow)

        OCLAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'GPU4.png')),
            'OpenCL Info',
            self)
        if isOpenCL:
            OCLAction.setShortcut('Alt+I')
            OCLAction.triggered.connect(self.showOCLinfo)

        tutorAction = qt.QAction(
            qt.QIcon(os.path.join(self.iconsDir, 'icon-info.png')),
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
        self.vToolBar.setOrientation(qt.Qt.Vertical)
        self.vToolBar.setIconSize(qt.QSize(56, 56))

        menuNames = ['Add Source', 'Add Optic', 'Add Aperture', 'Add Screen',
                     'Add Material', 'Add Figure Error', 'Add Plot']
        tabNames = ['Beamline']*4 + ['Materials', 'Figure Error', 'Plots']
        modules = [rsources, roes, rapts, rscreens, rmats, rfe, None]
        methods = [self.addElement]*6 + [self.addPlot]
        pngs = ['add-source.png', 'add-oe.png', 'add-aperture.png',
                'add-screen.png', 'add-material.png', 'add-figure-error.png',
                'add-plot.png']
        for menuName, amodule, afunction, aicon, tabName in zip(
                menuNames, modules, methods, pngs, tabNames):
            amenuButton = qt.QToolButton()
            amenuButton.setIcon(qt.QIcon(os.path.join(self.iconsDir, aicon)))
            amenuButton.setToolTip(menuName + ' to tab ' + tabName)

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
            if menuName in ['Add Screen', 'Add Figure Error']:
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
        self.progressBar.setAlignment(qt.Qt.AlignCenter)
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

        amt = qt.QShortcut(self)
        amt.setKey(qt.Qt.CTRL + qt.Qt.Key_E)
        amt.activated.connect(self.toggleExperimentalMode)

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
        self.tree = TreeViewEx()
        self.matTree = TreeViewEx()
        self.plotTree = TreeViewEx()
        self.feTree = TreeViewEx()
#        self.tree = qt.QTreeView()
#        self.matTree = qt.QTreeView()
#        self.plotTree = qt.QTreeView()
        self.runTree = qt.QTreeView()

        self.defaultFont = qt.QFont("Courier New", 9)

        for itree in [self.tree, self.matTree, self.feTree, self.plotTree,
                      self.runTree]:
            itree.setContextMenuPolicy(qt.Qt.CustomContextMenu)
            itree.clicked.connect(self.showDoc)

        self.plotTree.customContextMenuRequested.connect(self.plotMenu)
        self.plotTree.objDoubleClicked.connect(self.runPlotViewer)
        self.matTree.customContextMenuRequested.connect(self.matMenu)
        self.matTree.objDoubleClicked.connect(self.runMaterialViewer)
        self.feTree.customContextMenuRequested.connect(self.feMenu)
        self.feTree.objDoubleClicked.connect(self.runSurfViewer)
        self.tree.customContextMenuRequested.connect(self.openMenu)
        self.tree.objDoubleClicked.connect(self.runElementViewer)

        if ext.isSpyderlib:
            self.codeEdit = ext.codeeditor.CodeEditor(self)
            self.codeEdit.setup_editor(linenumbers=True, markers=True,
                                       tab_mode=False, language='py',
                                       font=self.defaultFont,
                                       color_scheme='Pydev')
            if qt.QtName == "PyQt5":
                self.codeEdit.zoom_in.connect(partial(self.zoom, 1))
                self.codeEdit.zoom_out.connect(partial(self.zoom, -1))
                self.codeEdit.zoom_reset.connect(partial(self.zoom, 0))
            elif qt.QtName == "PyQt4":
                self.connect(self.codeEdit, qt.Signal('zoom_in()'),
                             partial(self.zoom, 1))
                self.connect(self.codeEdit, qt.Signal('zoom_out()'),
                             partial(self.zoom, -1))
                self.connect(self.codeEdit, qt.Signal('zoom_reset()'),
                             partial(self.zoom, 0))
            qt.QShortcut(qt.QKeySequence.ZoomIn, self, partial(self.zoom, 1))
            qt.QShortcut(qt.QKeySequence.ZoomOut, self, partial(self.zoom, -1))
            qt.QShortcut("Ctrl+0", self, partial(self.zoom, 0))
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
        self.tabs.addTab(self.feTree, "Figure Error")
        self.tabs.addTab(self.plotTree, "Plots")
        self.tabs.addTab(self.runTree, "Job Settings")
        self.tabs.addTab(self.descrEdit, "Description")
        self.tabs.addTab(self.codeEdit, "Code")
        self.tabs.addTab(self.codeConsole, "Console")
        self.tabs.currentChanged.connect(self.showDescrByTab)

    def runElementViewer(self, oeuuid=None):
        oe = self.beamLine.oesDict.get(oeuuid)
        if oe is None:
            return
        oeObj = oe[0]
        oeType = oe[-1]
        blName = self.beamLine.name
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
                argMat = self.beamLine.materialsDict.get(argValue)
                if hasattr(argMat, 'name'):
                    oeProps[argName] = argMat.name
            if any(argName.lower().startswith(v) for v in
                    ['figureerror']) and\
                    raycing.is_valid_uuid(argValue):
                argFE = self.beamLine.fesDict.get(argValue)
                if hasattr(argFE, 'name'):
                    oeProps[argName] = argFE.name

        catDict = {'Position': raycing.orientationArgSet}
        if oeType == 0:  # source
            if hasattr(oeObj, 'eE'):
                catDict.update({
                    'Electron Beam': rsources.electronBeamArgSet,
                    'Magnetic Structure': rsources.magneticStructureArgSet})

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

        glowObj = getattr(self, 'blViewer', None)
        glWidget = getattr(glowObj, 'customGlWidget', None)
        elViewer = InstanceInspector(self, oeProps,
                                     initDict=oeInitProps,
                                     epicsDict=getattr(
                                             glWidget, 'epicsInterface', None),
                                     viewOnly=False,
                                     beamLine=self.beamLine,
                                     categoriesDict=catDict)

        if glWidget is not None:
            glWidget.beamUpdated.connect(elViewer.update_beam)
            glWidget.oePropsUpdated.connect(elViewer.update_param)
            # TODO: update tree
            elViewer.propertiesChanged.connect(
                    partial(glWidget.update_beamline, oeuuid,
                            sender='OEE'))
#            elViewer.propertiesChanged.connect(
#                    partial(self.updateBeamlineModel, oeuuid))
#        if (elViewer.exec_()):
        elViewer.show()

    def runMaterialViewer(self, matuuid=None):
        matobj = self.beamLine.materialsDict.get(matuuid)
        if matobj is None:
            return
        blName = self.beamLine.name
        matProps = raycing.get_init_kwargs(matobj, compact=False,
                                           blname=blName)
        matProps.update({'uuid': matuuid})
        for argName, argValue in matProps.items():
            if any(argName.lower().startswith(v) for v in
                    ['mater', 'tlay', 'blay', 'coat', 'substrate']) and\
                   raycing.is_valid_uuid(str(argValue)):
                argMat = self.beamLine.materialsDict.get(argValue)
                if hasattr(argMat, 'name'):
                    matProps[argName] = argMat.name
        glowObj = getattr(self, 'blViewer', None)
        glWidget = getattr(glowObj, 'customGlWidget', None)
        matViewer = InstanceInspector(self, matProps, beamLine=self.beamLine,
                                      viewOnly=False)
        if glWidget is not None:
            matViewer.propertiesChanged.connect(
                    partial(glWidget.update_beamline, matuuid, sender='OEE'))
            matViewer.propertiesChanged.connect(
                matViewer.dynamicPlotWidget.calculate)
        matViewer.show()

    def runPlotViewer(self, plotName):

        for i in range(self.rootPlotItem.rowCount()):
            plotItem = self.rootPlotItem.child(i, 0)
            if str(plotItem.text()) == plotName:
                break
        else:
            return

        plotsDict = self.treeToDict(plotItem)
        plotId = plotItem.data(qt.Qt.UserRole)
        plotsDict['beam'] = self.getBeamTag(plotsDict.get('beam'))

        plotViewer = PlotViewer(plotsDict, self, viewOnly=False,
                                beamLine=self.beamLine, plotId=plotId)
        self.plotParamUpdate.connect(plotViewer.update_plot_param)
        if hasattr(self, 'blViewer') and self.blViewer is not None:
            self.blViewer.customGlWidget.beamUpdated.connect(
                    plotViewer.update_beam)
        if (plotViewer.show()):
            pass

    def runSurfViewer(self, surfuuid=None):
        surfobj = self.beamLine.fesDict.get(surfuuid)
        if surfobj is None:
            return
        blName = self.beamLine.name
        surfProps = raycing.get_init_kwargs(surfobj, compact=False,
                                            blname=blName)
        surfProps.update({'uuid': surfuuid})

        for argName, argValue in surfProps.items():
            if any(argName.lower().startswith(v) for v in
                    ['basefe']) and\
                   raycing.is_valid_uuid(str(argValue)):
                argFE = self.beamLine.fesDict.get(argValue)
                if hasattr(argFE, 'name'):
                    surfProps[argName] = argFE.name
        glowObj = getattr(self, 'blViewer', None)
        glWidget = getattr(glowObj, 'customGlWidget', None)
        surfViewer = InstanceInspector(self, surfProps, beamLine=self.beamLine,
                                       viewOnly=False)
        if glWidget is not None:

            surfViewer.propertiesChanged.connect(
                    partial(glWidget.update_beamline, surfuuid,
                            sender='OEE'))
            surfViewer.propertiesChanged.connect(
                    surfViewer.dynamicPlotWidget.update_surface)
        surfViewer.show()

    def getBeamTag(self, beamName):
        beams = self.beamModel.findItems(beamName, column=0)
        beamTag = []
        for bItem in beams:
            row = bItem.row()
            btype = self.beamModel.item(row, 1).text()
            oeid = self.beamModel.item(row, 2).text()
            beamTag = (oeid, btype)
            break
        return beamTag

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
        menu.addAction("Zoom In", partial(self.zoomDoc, 1))
        menu.addAction("Zoom Out", partial(self.zoomDoc, -1))
        menu.addAction("Zoom reset", partial(self.zoomDoc, 0))
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

        # figureErrorsTree view
        self.feTree.setModel(self.fesModel)
        self.feTree.setAlternatingRowColors(True)
        self.feTree.setSortingEnabled(False)
        self.feTree.setHeaderHidden(False)
        self.feTree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        self.feTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.feTree.setItemDelegateForColumn(1, comboDelegate)

        # BLTree view
        self.tree.setModel(self.beamLineModel)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)

        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(qt.QTreeView.InternalMove)
        self.tree.setDefaultDropAction(qt.Qt.MoveAction)
        # self.tree.setUniformRowHeights(False)

        self.tree.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
        headers = ['Parameter', 'Value']
        if isUnitsEnabled:
            headers.append('Unit')
        if useSlidersInTree:
            headers.append('Slider')
        self.tree.model().setHorizontalHeaderLabels(headers)

        self.curObj = None

        self.addElement(copyFrom=blProps,
                        isRoot=True)

        # self.tree.expand(self.rootBLItem.index())
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

#        self.addCombo(self.runTree, self.rootRunItem)
        self.runTree.setColumnWidth(0, int(self.runTree.width()/3))
        index = self.runModel.indexFromItem(self.rootRunItem)
        self.runTree.setExpanded(index, True)

        self.tabs.tabBar().setTabTextColor(0, qt.Qt.black)
        self.tabs.tabBar().setTabTextColor(2, qt.Qt.black)
        self.progressBar.setValue(0)
        self.progressBar.setFormat("New beamline")

        self.blColorCounter = 0
        self.pltColorCounter = 0
        self.fileDescription = ""
#        self.descrEdit.setText("")
        self.writeCodeBox("")
        self.setWindowTitle("xrtQook")
        self.prefixtab = "\t"
        self.ntab = 1
        self.cpChLevel = 0
        self.saveFileName = ""
        self.layoutFileName = ""
        self.glowOnly = False
        self.isEmpty = True
        if getattr(self, 'beamLine', None) is None:
            self.beamLine = raycing.BeamLine()
            self.beamLine.flowSource = 'Qook'
#        self.updateBeamlineBeams(item=None)
        self.updateBeamlineMaterials(item=None)
        self.updateBeamline(item=None)
        self.rayPath = None
        if self.blViewer is not None:
            self.blViewer.customGlWidget.clearVScreen()
            self.blViewer.customGlWidget.selColorMin = None
            self.blViewer.customGlWidget.selColorMax = None
            self.blViewer.customGlWidget.tVec = np.array([0., 0., 0.])
            self.blViewer.customGlWidget.coordOffset = [0., 0., 0.]
            self.blViewer.customGlWidget.iMax = -1e20
            self.blViewer.customGlWidget.colorMax = -1e20
            self.blViewer.customGlWidget.colorMin = 1e20
        self.blUpdateLatchOpen = True

    def initAllModels(self):
        self.blUpdateLatchOpen = False

        self.beamLineModel = LevelRestrictedModel()
        self.beamLineModel.itemChanged.connect(self.beamLineItemChanged)

        self.boolModel = qt.QStandardItemModel()
        self.boolModel.appendRow(qt.QStandardItem('False'))
        self.boolModel.appendRow(qt.QStandardItem('True'))

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

        self.fesModel = qt.QStandardItemModel()
        self.rootFEItem = self.fesModel.invisibleRootItem()
        self.rootFEItem.setText("FigureErrors")
        self.fesModel.itemChanged.connect(self.beamLineItemChanged)
        self.addProp(self.fesModel.invisibleRootItem(), "None")

        self.beamModel = qt.QStandardItemModel()
        self.beamModel.appendRow([qt.QStandardItem("None"),  # name
                                  qt.QStandardItem("GlobalLocal"),  # type
                                  qt.QStandardItem("None"),  # OE uuid
                                  qt.QStandardItem('000')])  # Ordinal number
        self.rootBeamItem = self.beamModel.invisibleRootItem()
        self.rootBeamItem.setText("Beams")
#        self.beamModel.itemChanged.connect(self.updateBeamlineBeams)

#        self.matKindModel = qt.QStandardItemModel()
#        for mtKind in ['mirror', 'thin mirror',
#                       'plate', 'lens', 'grating', 'FZP', 'auto']:
#            mtItem = qt.QStandardItem(mtKind)
#            self.matKindModel.appendRow(mtItem)

#        self.matTableModel = qt.QStandardItemModel()
#        for mtTable in ['Chantler', 'Chantler total', 'Henke', 'BrCo']:
#            mtTItem = qt.QStandardItem(mtTable)
#            self.matTableModel.appendRow(mtTItem)

#        self.shapeModel = qt.QStandardItemModel()
#        for shpEl in ['rect', 'round']:
#            shpItem = qt.QStandardItem(shpEl)
#            self.shapeModel.appendRow(shpItem)

        self.matGeomModel = qt.QStandardItemModel()
        for mtGeom in ['Bragg reflected', 'Bragg transmitted',
                       'Laue reflected', 'Laue transmitted',
                       'Fresnel']:
            mtGItem = qt.QStandardItem(mtGeom)
            self.matGeomModel.appendRow(mtGItem)

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
            rayItem.setCheckState(qt.Qt.Checked)

        self.plotModel = qt.QStandardItemModel()
        self.addValue(self.plotModel.invisibleRootItem(), "plots")
        # self.plotModel.appendRow(qt.QStandardItem("plots"))
        self.rootPlotItem = self.plotModel.item(0, 0)
#        self.plotModel.itemChanged.connect(self.colorizeChangedParam)
        self.plotModel.itemChanged.connect(self.plotItemChanged)
        self.plotModel.invisibleRootItem().setText("plots")
        self.runModel = qt.QStandardItemModel()
        self.addProp(self.runModel.invisibleRootItem(), "run_ray_tracing()")
        self.runModel.itemChanged.connect(self.colorizeChangedParam)
        self.rootRunItem = self.runModel.item(0, 0)

        self.comboModelDict = {'filamentBeam': self.boolModel,
                               'uniformRayDensity': self.boolModel,
                               'beam': self.beamModel,
                               'geom': self.matGeomModel}

        self.fluxLabelList = list(raycing.allBeamFields)
        self.fluxDataList = ['auto'] + list(self.fluxLabelList)
        self.lengthUnitList = list(raycing.allUnitsLenStr.values())
        self.angleUnitList = list(raycing.allUnitsAngStr.values())
        self.energyUnitList = list(raycing.allUnitsEnergyStr.values())

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
                for row in reversed(range(self.rootBLItem.rowCount())):
                    oeItem = self.rootBLItem.child(row, 0)
                    self.deleteElement(self.tree, oeItem)
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
        # view.expand(item.index())
        view.setColumnWidth(0, int(view.width()/3))

    def getObjStr(self, selItem, level):
        objRoot = None
        obj = None
        model = selItem.model()
        if selItem.column() > 0:
            ind = selItem.index().siblingAtColumn(0)
            testItem = model.itemFromIndex(ind)
            if testItem is not None:
                selItem = testItem
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
        elif model in [self.materialsModel, self.fesModel]:
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
            self.renderLiveDoc(argDocStr, nameStr, argSpecStr, noteStr,
                               img_path='../_images')
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
        self.showTutorial(txt, "xrtQook", img_path='../_images')
        self.docks[0].raise_()

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
                              "Using xrtQook for script generation",
                              img_path='../_images')
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
        self.docks[0].raise_()

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

    def addParam(self, parent, paramName, value, source=None, unit=None):
        """Add a pair of Parameter-Value Items"""
        toolTip = None
        child0 = qt.QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem(str(value))
        if str(paramName) == 'name':
            ch1flag = self.paramFlag
        elif isinstance(parent, qt.QStandardItem) and\
                str(parent.text()) == 'output':
            ch1flag = self.paramFlag
        else:
            ch1flag = self.valueFlag

        child1.setFlags(ch1flag)

        if unit is not None:
            child1u = qt.QStandardItem(str(unit))
            child1u.setFlags(self.valueFlag)
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
        if unit is not None:
            row.append(child1u)
        if useSlidersInTree:
            child2 = qt.QStandardItem()
            row.append(child2)

        if not isinstance(source, qt.QStandardItem):
            parent.appendRow(row)
        else:
            parent.insertRow(source.row() + 1, row)

        if useSlidersInTree:  # to be replaced with delegates
            if paramName in withSlidersInTree:
                ind = child0.index().sibling(child0.index().row(), 2)
                slider = qt.QSlider(qt.Qt.Horizontal)
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
        s = editItem.model().data(editItem.index(), qt.Qt.DisplayRole)
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
        editItem.model().setData(editItem.index(), s, qt.Qt.EditRole)
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
        elif 'figure' in obj:
            tree = self.feTree
            rootItem = self.rootFEItem
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
        elementClassItem.setToolTip(
                "Double click to see live object properties")
        if isRoot:
            self.rootBLItem = elementItem

        flags = qt.Qt.ItemFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsEditable |
                                qt.Qt.ItemIsSelectable |
                                qt.Qt.ItemIsDropEnabled)

        flags |= qt.Qt.ItemIsDropEnabled if isRoot else qt.Qt.ItemIsDragEnabled

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
            propsDict['uuid'] = str(raycing.uuid.uuid4())
        else:
            propsDict = dict(self.getParams(obj))
            propsDict['uuid'] = 'top' if isRoot else str(raycing.uuid.uuid4())

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
                elementItem.setData(argVal, qt.Qt.UserRole)
                continue
            if arg in ['material', 'material2', 'tlayer', 'blayer', 'coating',
                       'substrate']:
                for iMat in range(self.rootMatItem.rowCount()):
                    matItem = self.rootMatItem.child(iMat, 0)
                    if str(matItem.data(qt.Qt.UserRole)) == str(argVal):
                        argVal = str(matItem.text())
                        break
            elif arg in ['figureError', 'baseFE']:
                for iFe in range(self.rootFEItem.rowCount()):
                    feItem = self.rootFEItem.child(iFe, 0)
                    if str(feItem.data(qt.Qt.UserRole)) == str(argVal):
                        argVal = str(feItem.text())
                        break
            self.addParam(elprops, arg, argVal)

        if not isRoot:
            self.showDoc(elementItem.index())

        tree.expand(rootItem.index())
        self.capitalize(tree, elementItem)
        self.blUpdateLatchOpen = True
        elementItem.model().blockSignals(False)

#        if not isinstance(copyFrom, dict):  # Not import from file
#       TODO: load all elements first, then run propagation
        if tree is self.tree:
            self.updateBeamline(elementItem, newElement=obj)
        elif tree is self.feTree:
            self.updateBeamlineFEs(elementItem, newElement=obj)
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
        tree.resizeColumnToContents(0)

        if not copyFrom and not isRoot and tree is self.tree and \
                self.callWizard:
            self.newElementCreated.emit(propsDict['uuid'])

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
                self.updateBeamlineMaterials(item)
            elif item.model() is self.fesModel:
                self.updateBeamlineFEs(item)

    def plotItemChanged(self, item):
        parent = item.parent()
        if item.column() == 0:
            return
        elif str(parent.text()) == 'plots' and\
                str(item.text()).startswith('Preview'):
            return
        elif item.column() == 1 and item.isEnabled():
            paramValue = raycing.parametrize(item.text())
            objChng = str(parent.text())
            if objChng.endswith('axis'):
                plotParent = parent.parent()
                plotId = str(plotParent.data(qt.Qt.UserRole))
            else:
                plotId = str(parent.data(qt.Qt.UserRole))
            row = item.row()
            paramName = str(parent.child(row, 0).text())
            if paramName == 'beam':
                paramValue = self.getBeamTag(paramValue)
            plotParamTuple = plotId, objChng, paramName, paramValue
            self.plotParamUpdate.emit(plotParamTuple)

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
                        color = qt.Qt.red
#                        counter = 1
                    elif parent.child(itemRow, 0).foreground().color() ==\
                            qt.Qt.red:
                        color = qt.Qt.black
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
                if parent.child(itemRow, 0).foreground().color() != qt.Qt.red:
                    for defArg, defArgVal in self.getParams(obj):
                        if str(defArg) == str(parent.child(itemRow, 0).text()):
                            if str(defArgVal) != str(item.text()):
                                color = qt.Qt.blue
                            else:
                                color = qt.Qt.black
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
                    buuid = str(item.data(qt.Qt.UserRole))
                    item.model().blockSignals(False)

                    if item.model() is self.materialsModel:
                        mat = self.beamLine.materialsDict.get(buuid)
                        oldname = mat.name
#                        print(pyname, oldname)
                        self.iterateRename(
                                self.rootMatItem, oldname, pyname,
                                ['tlay', 'blay', 'coat', 'substrate'])
                        self.iterateRename(self.rootBLItem, oldname, pyname,
                                           ['material'])
                    elif item.model() is self.fesModel:
                        fe = self.beamLine.fesDict.get(buuid)
                        oldname = fe.name
#                        print(pyname, oldname)
                        self.iterateRename(
                                self.rootFEItem, oldname, pyname,
                                ['baseFE'])
                        self.iterateRename(self.rootBLItem, oldname, pyname,
                                           ['figureError'])
                    else:
                        for j in range(self.beamModel.rowCount()):
                            beams = self.beamModel.findItems(buuid, column=2)
                            for bItem in beams:
                                row = bItem.row()
                                btype = self.beamModel.item(row, 1).text()
                                bname = "{}_{}".format(
                                        pyname, btype[4:].lower())
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
#            color = qt.Qt.red if self.blColorCounter > 0 else\
#                qt.Qt.black
#            self.tabs.tabBar().setTabTextColor(0, color)
#        elif item.model() == self.plotModel:
#            color = qt.Qt.red if self.pltColorCounter > 0 else\
#                qt.Qt.black
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

    def addMethod(self, name, parentItem, outBeams, methProps=None):
        self.beamModel.sort(3)

        elstr = str(parentItem.text())
        eluuid = parentItem.data(qt.Qt.UserRole)

        methodOutputDict = OrderedDict()

        if methProps is not None:
            methodInputDict = methProps.get('parameters')
            if 'beam' in methodInputDict:
                fModel0 = qt.MultiColumnFilterProxy(
                        {1: 'Global', 2: methodInputDict['beam']})
                fModel0.setSourceModel(self.beamModel)
                beamName = fModel0.data(fModel0.index(0, 0))
                methodInputDict['beam'] = beamName
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
                    pVal = fModel.data(fModel.index(lastIndex, 0)) if\
                        fModel.rowCount() > 0 else "None"
                methodInputDict[pName] = pVal

        # Will always auto-generate with new naming scheme
        for outstr in outBeams:
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
#        self.addCombo(self.tree, methodItem)
#        self.tree.expand(methodItem.index())
#        self.tree.expand(methodOut.index())
#        self.tree.expand(methodProps.index())
#        self.tree.setCurrentIndex(methodProps.index())
#        self.tree.setColumnWidth(0, int(self.tree.width()/2))
        self.blUpdateLatchOpen = True
        self.updateBeamline(methodItem, newElement=True)  # TODO:
        self.isEmpty = False

    def addPlot(self, copyFrom=None, plotName=None, beamName=None):
        plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
        axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))
        plotProps = plotDefArgs

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

        plotProps['title'] = plotName
        axHints = {'xaxis': {'label': 'x', 'unit': 'mm'},
                   'yaxis': {'label': 'y', 'unit': 'mm'},
                   'caxis': {'label': 'energy', 'unit': 'eV'}}

        if beamName is not None:
            plotProps['beam'] = beamName
            oeid = None
            oeobj = None
            beamType = 'local'
            beams = self.beamModel.findItems(beamName, column=0)

            for bItem in beams:
                row = bItem.row()
                oeid = str(self.beamModel.item(row, 2).text())
                beamType = str(self.beamModel.item(row, 1).text())
                break

            if oeid is not None:
                oeLine = self.beamLine.oesDict.get(oeid)
                if oeLine is not None:
                    oeobj = oeLine[0]

            if beamType.endswith('lobal') or is_screen(oeobj) or\
                    is_aperture(oeobj):
                axHints['yaxis']['label'] = 'z'

            plotProps['title'] =\
                f'{plotName}-{beamName}-{axHints["caxis"]["label"]}'

        for pname in ['xaxis', 'yaxis', 'caxis']:
            plotProps[pname] = copy.deepcopy(axDefArgs)
            if isinstance(copyFrom, dict):
                pval = copyFrom.pop(pname, None)
                if pval is not None:
                    plotProps[pname].update(pval)
            else:
                plotProps[pname].update(axHints[pname])

            plotProps[pname]['_object'] = "xrt.plotter.XYCAxis"

#        if isinstance(copyFrom, dict):
#            plotProps.update(copyFrom)
#            plotItem = self.addValue(self.rootPlotItem, plotName)
#        else:
#            plotItem = self.addValue(
#                    self.rootPlotItem, plotName, source=copyFrom)

        if isinstance(copyFrom, dict):
            plotProps.update(copyFrom)
            plotItem, plotViewItem = self.addParam(
                    self.rootPlotItem, plotName, "Preview plot")
        else:
            plotItem, plotViewItem = self.addParam(
                    self.rootPlotItem, plotName, "Preview plot",
                    source=copyFrom)
        plotItem.setData(str(raycing.uuid.uuid4()), qt.Qt.UserRole)
        self.paintStatus(plotViewItem, 0)
        plotViewItem.setToolTip("Double click to preview")
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
                    arg_value = pval
                    self.addParam(plotItem, pname, arg_value)

        self.showDoc(plotItem.index())
#        self.addCombo(self.plotTree, plotItem)
        self.capitalize(self.plotTree, plotItem)
#        self.plotTree.expand(self.rootPlotItem.index())
        self.plotTree.resizeColumnToContents(0)
#        self.plotTree.setColumnWidth(0, int(self.plotTree.width()/3))
        self.isEmpty = False
        self.tabs.setCurrentWidget(self.plotTree)

    def addPlotBeam(self, beamName):
        self.addPlot(beamName=beamName)

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
        """
        call beamline.delete_element_by_id(item.uuid)
        then item.parent.removeRow(item.index().row())

        """
        objuuid = item.data(qt.Qt.UserRole)
        if self.blViewer is not None:
            if view is self.tree:
                self.blViewer.customGlWidget.deletionQueue.append(objuuid)
                # only delete oe when it is safe
            self.blViewer.customGlWidget.delete_object(objuuid)

        oldname = str(item.text())

        if item.parent() is not None:
            item.parent().removeRow(item.index().row())
            beams = self.beamModel.findItems(objuuid, column=2)
            bRows = []
            for bItem in beams:
                bRows.append(bItem.row())

            for row in sorted(bRows, reverse=True):
                self.beamModel.removeRow(row)
        elif view is self.feTree:
            self.iterateRename(
                    self.rootFEItem, oldname, "None",
                    ['baseFE'])
            self.iterateRename(self.rootBLItem, oldname, "None",
                               ['figureError'])
            item.model().invisibleRootItem().removeRow(item.index().row())

        else:
            self.iterateRename(
                    self.rootMatItem, oldname, "None",
                    ['tlay', 'blay', 'coat', 'substrate'])
            self.iterateRename(self.rootBLItem, oldname, "None",
                               ['material'])
            item.model().invisibleRootItem().removeRow(item.index().row())

        # TODO: consider non-glow case, beamline belongs to Qook widget?

#                item.parent() is None:
#            del self.beamLine.materialsDict[str(item.text())]
#        if item.parent() == self.rootBLItem:
#            self.blUpdateLatchOpen = False
#
#        while item.hasChildren():
#            iItem = item.child(0, 0)
#            if item.child(0, 1) is not None:
#                if str(iItem.text()) == 'beam' and\
#                        str(item.child(0, 1).text()) == 'None':
#                    if iItem.model() == self.beamLineModel:
#                        self.blColorCounter -= 1
#                    elif iItem.model() == self.plotModel:
#                        self.pltColorCounter -= 1
#                iWidget = view.indexWidget(item.child(0, 1).index())
#                if iWidget is not None:
#                    if item.text() == "output":
#                        try:
#                            del self.beamLine.beamsDict[str(
#                                iWidget.currentText())]
#                        except Exception:
#                            print("Failed to delete", iWidget.currentText())
#                            if _DEBUG_:
#                                raise
#                            else:
#                                pass
#                        beamInModel = self.beamModel.findItems(
#                            iWidget.currentText())
#                        if len(beamInModel) > 0:
#                            self.beamModel.takeRow(beamInModel[0].row())
#
#            self.deleteElement(view, iItem)
#        else:
#            if item.parent() == self.rootBLItem:
#                del self.beamLine.oesDict[str(item.text())]
#                self.blUpdateLatchOpen = True
#                self.updateBeamModel()
#                self.updateBeamline(item, newElement=None)
#            if item.parent() is not None:
#                item.parent().removeRow(item.index().row())
#            else:
#                item.model().invisibleRootItem().removeRow(item.index().row())

    def exportLayout(self):
        saveStatus = False
        self.beamModel.sort(3)
        if self.layoutFileName == "":
            saveDialog = qt.QFileDialog()
            saveDialog.setFileMode(qt.QFileDialog.AnyFile)
            saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
            saveDialog.setNameFilter("XML files (*.xml);;JSON files (*.json)")
            if (saveDialog.exec_()):
                self.layoutFileName = saveDialog.selectedFiles()[0]
        if self.layoutFileName != "":
            if self.layoutFileName.lower().endswith("json"):
                _ = self.beamLine.export_to_json()
                plotsDict = self.treeToDict(self.rootPlotItem)
                runDict = self.treeToDict(self.rootRunItem)
#                elementsDict = self.treeToDict(self.rootBLItem)
#                print(elementsDict)
                if 'Project' in self.beamLine.layoutStr:
                    self.beamLine.layoutStr['Project']['plots'] = plotsDict
                    self.beamLine.layoutStr['Project']['run_ray_tracing'] = \
                        runDict
                    self.beamLine.layoutStr['Project']['description'] = \
                        self.fileDescription

                with open(self.layoutFileName, 'w',
                          encoding="utf-8") as json_file:
                    raycing.json.dump(
                        self.beamLine.layoutStr, json_file, indent=4)
                    # TODO: plots, run, description
            elif self.layoutFileName.lower().endswith("xml"):
                self.confText = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                self.confText += "<Project>\n"
                for item, view in zip([self.rootBeamItem,
                                       self.rootMatItem,
                                       self.rootBLItem,
                                       self.rootFEItem,
                                       self.rootPlotItem,
                                       self.rootRunItem],
                                      [None,
                                       self.matTree,
                                       self.tree,
                                       self.feTree,
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
                    fileObject = open(self.layoutFileName, 'w',
                                      encoding="utf-8")
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

    def treeToDict(self, rootItem):
        outDict = OrderedDict()
        for pn in range(rootItem.rowCount()):
            pnItem = rootItem.child(pn, 0)
            itemDict = OrderedDict()
            if not pnItem.hasChildren():
                outDict[str(pnItem.text())] = str(rootItem.child(pn, 1).text())
                continue
            for pnp in range(pnItem.rowCount()):
                pltPropItem = pnItem.child(pnp, 0)
                if pltPropItem.hasChildren():
                    axDict = OrderedDict()
                    for axn in range(pltPropItem.rowCount()):
                        axPropName = pltPropItem.child(axn, 0).text()
                        axPropValue = pltPropItem.child(axn, 1).text()
                        axDict[axPropName] = axPropValue
                    itemDict[str(pltPropItem.text())] = axDict
                else:
                    itemDict[str(pltPropItem.text())] = str(
                            pnItem.child(pnp, 1).text())
            outDict[str(pnItem.text())] = itemDict
        return outDict

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
                    if child1.flags() != self.paramFlag or\
                            str(child0.text()) == "name" or\
                            str(child0.text()).startswith('beam'):
                        if child1.isEnabled():
                            itemType = "param"
                        else:
                            itemType = "object"
                        if int(child1.isEnabled()) == int(child0.isEnabled()):
                            if not str(child1.text()).endswith('plot') and not\
                                    str(child1.text()).startswith('Instance'):
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

    def importLayout(self, layoutJSON=None, filename=None):
        project = None
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

            if layoutJSON is not None:
                project = layoutJSON
            else:
                if filename is not None:
                    openFileName = filename
                else:
                    openDialog = qt.QFileDialog()
                    openDialog.setFileMode(qt.QFileDialog.ExistingFile)
                    openDialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
                    openDialog.setNameFilter(
                            "XML and JSON files (*.xml *.json)")
                    if (openDialog.exec_()):
                        openFileName = openDialog.selectedFiles()[0]

                if not openFileName:
                    self.progressBar.setFormat(
                            "Error opening layout from file")
                    return
                ldMsg = 'Loading layout from {}'.format(
                        os.path.basename(str(openFileName)))

                self.progressBar.setFormat(ldMsg)

                parseOK = True  # False
                if parseOK:
                    tmpBL = raycing.BeamLine(fileName=openFileName)
                    project = tmpBL.export_to_json()

            if project is None:
                self.progressBar.setFormat(
                            "No layout")
                return

            tmpAutoUpdate = self.isGlowAutoUpdate
            if tmpAutoUpdate:
                self.toggleGlow(False)
                self.docks[1].raise_()

            # Deleting existing elements

            for row in reversed(range(self.rootBLItem.rowCount())):
                oeItem = self.rootBLItem.child(row, 0)
                self.deleteElement(self.tree, oeItem)

            beamlineName = next(raycing.islice(project.keys(), 2, 3))
            self.beamLine.name = beamlineName

            beamlineInitKWargs = project[beamlineName]['properties']
            beamlineInitKWargs['name'] = beamlineName
            self.blUpdateLatchOpen = False

            # INIT THE BEAMLINE HERE

            self.initAllModels()
            self.initAllTrees(
                   blProps={'properties': beamlineInitKWargs,
                            '_object': 'xrt.backends.raycing.BeamLine'},
                   runProps=project.get('run_ray_tracing'))
            for branch in ['Materials', 'FigureErrors', beamlineName]:
                for element, elementDict in project.get(branch).items():
                    if str(element) in ['properties', '_object']:
                        continue

                    if branch == beamlineName and 'flow' in project:
                        methDict = project['flow'].get(element)
                        if methDict is None:  # VirtualScreen
                            continue
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
                elif isinstance(bName, str):
                    beams = self.beamModel.findItems(bName, column=0)
                    if len(beams) < 1:
                        plotDict['beam'] = 'None'

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

#            self.plotTree.setColumnWidth(
#                0, int(self.plotTree.width()/3))
            self.tabs.setCurrentWidget(self.tree)
            self.prbStart = 0
            self.prbRange = 100
            self.statusUpdate.emit((0., 'Running propagation'))

            if tmpAutoUpdate:
                self.toggleGlow(True)
                self.docks[1].raise_()

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
        self.rootBeamItem.setChild(0, 1, qt.QStandardItem("beamGlobalLocal"))
        self.rootBeamItem.setChild(0, 2, qt.QStandardItem("None"))
        self.rootBeamItem.setChild(0, 3, qt.QStandardItem('000'))
        for ibl in range(self.rootBLItem.rowCount()):
            elItem = self.rootBLItem.child(ibl, 0)
            elNameStr = str(elItem.text())
            eluuid = elItem.data(qt.Qt.UserRole)
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

                        outBeams = fdoc[0].replace(
                                "Returned values: ", '').split(',')

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
#            if selIndex.row() > 2:
#                menu.addAction("Move Up", partial(self.moveItem, -1,
#                                                  self.tree,
#                                                  selectedItem))
#            if selIndex.row() < selectedItem.parent().rowCount()-1:
#                menu.addAction("Move Down", partial(self.moveItem, 1,
#                                                    self.tree,
#                                                    selectedItem))
#            menu.addSeparator()
            deleteActionName = "Remove " + str(selectedItem.text())
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.tree,
                                                     selectedItem))

            for ic in range(selectedItem.rowCount()):
                if selectedItem.child(ic, 0).text() == "_object":
                    elstr = str(selectedItem.child(ic, 1).text())
                    break
            else:
                return
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

    def feMenu(self, position):
        indexes = self.feTree.selectedIndexes()
        level = 100
        if len(indexes) > 0:
            level = 0
            index = indexes[0]
            selectedItem = self.fesModel.itemFromIndex(index)
            while index.parent().isValid():
                index = index.parent()
                level += 1

        menu = qt.QMenu()

        feMenu = menu.addMenu(self.tr("Add Figure Error"))
        if True:  # __all__
            for mName in rfe.__all__:
                self._addAction(rfe, mName, self.addElement, feMenu)

        if level == 0 and selectedItem.text() != "None":
            menu.addSeparator()
            deleteActionName = "Remove " + str(selectedItem.text())
            menu.addAction(deleteActionName, partial(self.deleteElement,
                                                     self.feTree,
                                                     selectedItem))
        else:
            pass

        menu.exec_(self.feTree.viewport().mapToGlobal(position))

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
            if str(self.rootBLItem.child(iel, 0).data(
                    qt.Qt.UserRole)) == str(eluuid):
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

    def updateBeamlineModel(self, data):
        oeid, kwargs = data

        if oeid in self.beamLine.oesDict:
            model = self.beamLineModel
            tree = self.tree
            rootItem = self.rootBLItem
        elif oeid in self.beamLine.materialsDict:
            model = self.materialsModel
            tree = self.matTree
            rootItem = self.rootMatItem
        elif oeid in self.beamLine.fesDict:
            model = self.fesModel
            tree = self.feTree
            rootItem = self.rootFEItem
        else:
            return

        model.blockSignals(True)
        for i in range(rootItem.rowCount()):
            elItem = rootItem.child(i, 0)
            elUUID = str(elItem.data(qt.Qt.UserRole))
            if elUUID == oeid:
                for j in range(elItem.rowCount()):
                    pItem = elItem.child(0, j)
                    if str(pItem.text()) in 'properties':
                        for k in range(pItem.rowCount()):
                            pNItem = pItem.child(k, 0)
                            for argName, argValue in kwargs.items():
                                if str(pNItem.text()) == argName:
                                    if any(argName.lower().startswith(v) for v in
                                           ['mater', 'tlay', 'blay', 'coat', 'substrate']) and\
                                        raycing.is_valid_uuid(argValue):
                                            matObj = self.beamLine.materialsDict.get(argValue)
                                            argValue = matObj.name
                                    elif any(argName.lower().startswith(v) for v in
                                           ['figureerr', 'basefe']):
                                        if raycing.is_valid_uuid(argValue):
                                            feObj = self.beamLine.fenamesToUUIDs.get(argValue)
                                            argValue = feObj.name

                                    pVItem = pItem.child(k, 1)
                                    pVItem.setText(str(argValue))
                        break
                break
        model.blockSignals(False)
        tree.update()

    def updateBeamlineMaterials(self, item=None, newElement=None):
        # TODO: move deletion here
        kwargs = {}
        if item is None or (item.column() == 0 and newElement is None):
            return

        if item.column() == 1:
            matItem = item.parent().parent()
        else:
            matItem = item

        objStr = None
        matId = str(matItem.data(qt.Qt.UserRole))
        paintItem = self.rootMatItem.child(matItem.row(), 1)
        # renaming existing
        if item.column() == 1 and item.text() == matItem.text():
            self.beamLine.materialsDict[matId].name = item.text()
            return

        parent = item.parent()

        if item.column() == 1:  # Existing Element
            argValue_str = item.text()
            argName = parent.child(item.row(), 0).text()
            argValue = raycing.parametrize(argValue_str)

            kwargs[argName] = argValue
            outDict = kwargs

        elif item.column() == 0:  # New Element
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
                initStatus = self.beamLine.init_material_from_json(
                        matId, outDict)
            except Exception:
                raise

            self.paintStatus(paintItem, initStatus)

        if self.blViewer is None or not outDict:
            return

        self.blViewer.customGlWidget.update_beamline(
                matId, outDict, sender='Qook')

    def updateBeamlineFEs(self, item=None, newElement=None):
        kwargs = {}
        if item is None or (item.column() == 0 and newElement is None):
            return

        if item.column() == 1:
            feItem = item.parent().parent()
        else:
            feItem = item

        objStr = None
        feId = str(feItem.data(qt.Qt.UserRole))
        paintItem = self.rootFEItem.child(feItem.row(), 1)
        # renaming existing
        if item.column() == 1 and item.text() == feItem.text():
            self.beamLine.fesDict[feId].name = item.text()
            return

        parent = item.parent()

        if item.column() == 1:  # Existing Element
            argValue_str = item.text()
            argName = parent.child(item.row(), 0).text()
            argValue = raycing.parametrize(argValue_str)

            kwargs[argName] = argValue
            outDict = kwargs

        elif item.column() == 0:  # New Element
            for itop in range(feItem.rowCount()):
                chitem = feItem.child(itop, 0)
                if chitem.text() in ['properties']:
                    for iprop in range(chitem.rowCount()):
                        argName = chitem.child(iprop, 0).text()
                        argValue = raycing.parametrize(
                                chitem.child(iprop, 1).text())
                        kwargs[str(argName)] = argValue
                elif chitem.text() == '_object':
                    objStr = str(feItem.child(itop, 1).text())
            kwargs['uuid'] = feId
            outDict = {'properties': kwargs, '_object': objStr}
            initStatus = 0
            try:
                initStatus = self.beamLine.init_fe_from_json(feId, outDict)
            except Exception:
                raise

            self.paintStatus(paintItem, initStatus)

        if self.blViewer is None or not outDict:
            return

        self.blViewer.customGlWidget.update_beamline(
                feId, outDict, sender='Qook')

    def updateBeamline(self, item=None, newElement=None, newOrder=False):
        def beamToUuid(beamName):
            for ib in range(self.beamModel.rowCount()):
                if self.beamModel.item(ib, 0).text() == beamName:
                    return self.beamModel.item(ib, 2).text()

        def buildMethodDict(mItem):
            methKWArgs = OrderedDict()
            outKWArgs = OrderedDict()
            methObjStr = ''
            for mch in range(mItem.rowCount()):
                mchi = mItem.child(mch, 0)
                if str(mchi.text()) == 'parameters':
                    for mchpi in range(mchi.rowCount()):
                        argName = str(mchi.child(mchpi, 0).text())
                        argValue = str(mchi.child(mchpi, 1).text())
                        if argName == 'beam':
                            argValue = beamToUuid(argValue)
                        else:
                            argValue = raycing.parametrize(
                                argValue)
                        methKWArgs[argName] = argValue
                elif str(mchi.text()) == 'output':
                    for mchpi in range(mchi.rowCount()):
                        argName = str(mchi.child(mchpi, 0).text())
                        argValue = str(mchi.child(mchpi, 1).text())
                        if argName == 'beam':
                            argValue = beamToUuid(argValue)
                        else:
                            argValue = raycing.parametrize(
                                argValue)
                        outKWArgs[argName] = argValue
                elif str(mchi.text()) == '_object':
                    methObjStr = str(mItem.child(mch, 1).text())
            outDict = {'_object': methObjStr,
                       'parameters': methKWArgs,
                       'output': outKWArgs}
            return outDict

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
                # print("No parent")
                return
            # else:
            #     print(str(parent.text()))  # TODO: print

            if str(parent.text()) in ['properties']:
                oeItem = parent.parent()
                oeid = str(oeItem.data(qt.Qt.UserRole))
            elif str(parent.text()) in ['parameters']:
                methItem = parent.parent()
                oeItem = methItem.parent()
                oeid = str(oeItem.data(qt.Qt.UserRole))
                methObjStr = methItem.text().strip('()')
                outDict = {'_object': methObjStr}

            elif raycing.is_valid_uuid(item.data(qt.Qt.UserRole)):
                oeid = str(item.data(qt.Qt.UserRole))

            if column == 1:  # Existing Element
                argValue_str = item.text()
                argName = parent.child(row, 0).text()

                if argName == 'beam':
                    argValue = beamToUuid(argValue_str)
                else:
                    argValue = raycing.parametrize(argValue_str)

                kwargs[argName] = argValue

                if outDict:  # updating flow
                    flowRec = self.beamLine.flowU.get(oeid)

                    if flowRec is None:
                        outDict = buildMethodDict(methItem)
                    else:
                        for methParams in flowRec.values():
                            methParams.update(kwargs)
                        outDict['parameters'] = methParams

                    self.beamLine.update_flow_from_json(
                            oeid, {methObjStr: outDict})
                    self.beamLine.sort_flow()

                else:
                    outDict = kwargs

            elif column == 0 and newElement is not None:  # New Element
                if raycing.is_valid_uuid(parent.data(qt.Qt.UserRole)):  # flow
                    oeid = str(parent.data(qt.Qt.UserRole))
                    outDict = buildMethodDict(item)
                    methStr = outDict['_object'].split('.')[-1]
                    self.beamLine.update_flow_from_json(
                            oeid, {methStr: outDict})
                    self.beamLine.sort_flow()

                else:  # element
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

            self.blViewer.customGlWidget.update_beamline(
                    oeid, outDict, sender='Qook')

    def paintStatus(self, item, status):
        updateStatus = item.model().signalsBlocked()
        if not updateStatus:
            item.model().blockSignals(True)
        if status:
            color = qt.QColor(255, 200, 200)  # pale red
        else:
            color = qt.QColor(200, 255, 192)  # pale green
        item.setBackground(qt.QBrush(color))
        item.setIcon(qt.QIcon(os.path.join(self.iconsDir, 'double-click.png')))
        item.model().blockSignals(updateStatus)

    def updateOrder(self, *args, **kwargs):
        if not hasattr(self, 'rootBLItem'):
            return

        try:
            for iel in range(self.rootBLItem.rowCount()):
                elItem = self.rootBLItem.child(iel, 0)
                if elItem.text() == "properties":
                    continue
                eluuid = str(elItem.data(qt.Qt.UserRole))
                iBeams = self.beamModel.findItems(eluuid, column=2)
                for bItem in iBeams:
                    row = bItem.row()
                    self.beamModel.item(row, 3).setText(f"{iel:03d}")
            self.beamModel.sort(3)
#            for iel in range(self.rootBeamItem.rowCount()):
#                for n in range(4):
#                    print(self.rootBeamItem.child(iel, n).text())
        except RuntimeError:
            pass

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

    def toggleGlow(self, status):
        self.isGlowAutoUpdate = status
        self.blRunGlow()
        if self.blViewer is not None:
            if hasattr(self.blViewer.customGlWidget, 'input_queue'):
                self.blViewer.customGlWidget.input_queue.put({
                            "command": "auto_update",
                            "object_type": "beamline",
                            "kwargs": {"value": int(status)}
                            })

    def blRunGlow(self, kwargs={}):
        if self.blViewer is None:
            try:
                _ = self.beamLine.export_to_json()  # Init Gl.bl and move there
                self.blViewer = xrtglow.xrtGlow(
                        layout=self.beamLine.layoutStr,
                        progressSignal=self.statusUpdate,
                        **kwargs)
                self.blViewer.setWindowTitle("xrtGlow")
                # self.blViewer.show()
                self.blViewer.parentRef = self
#                self.blViewer.parentSignal = self.statusUpdate
                self.beamLine = self.blViewer.customGlWidget.beamline
                self.blViewer.customGlWidget.updateQookTree.connect(
                    self.updateBeamlineModel)
            except AttributeError:
                pass
            except Exception as e:
                print('Cannot create xrtGlow')
                raise e

    def updateProgressBar(self, dataTuple):
        self.progressBar.setValue(self.prbStart +
                                  int(dataTuple[0] * self.prbRange))
        self.progressBar.setFormat(dataTuple[1])

        if dataTuple[0] <= 0:
            self.busyIconThread = qt.QThread(self)
            self.busyIconWorker = BusyIconWorker()
            self.busyIconWorker.moveToThread(self.busyIconThread)
            self.busyIconWorker.prepare(self)
            self.busyIconThread.started.connect(self.busyIconWorker.render)
            self.busyIconThread.finished.connect(self.busyIconWorker.halt)
            self.busyIconThread.start()
        elif dataTuple[0] >= 1:
            if self.busyIconThread is not None:
                self.busyIconWorker.shouldRedraw = False
                self.busyIconThread.quit()
                self.busyIconThread.deleteLater()
                self.busyIconThread = None
                self.setTabIcons()

    def generateCode(self):
        self.progressBar.setValue(0)
        self.progressBar.setFormat("Flattening structure.")
        for tree, item in zip([self.tree, self.matTree, self.feTree,
                               self.plotTree, self.runTree],
                              [self.rootBLItem, self.rootMatItem,
                               self.rootFEItem,
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

        for matId in self.beamLine.sort_materials():
            for ie in range(self.rootMatItem.rowCount()):
                if str(self.rootMatItem.child(ie, 0).data(qt.Qt.UserRole)) == \
                        matId:
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
                                importStr = 'import {0}'.format(
                                    klass.__module__)
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

        self.progressBar.setValue(25)
        self.progressBar.setFormat("Defining figure errors.")

        for feId in self.beamLine.sort_figerrors():
            print(feId, self.beamLine.fesDict[feId].name)
            for ie in range(self.rootFEItem.rowCount()):
                if str(self.rootFEItem.child(ie, 0).data(qt.Qt.UserRole)) == \
                        feId:
                    feItem = self.rootFEItem.child(ie, 0)
                    ieinit = ""
                    for ieph in range(feItem.rowCount()):
                        if feItem.child(ieph, 0).text() == '_object':
                            elstr = str(feItem.child(ieph, 1).text())
                            klass = eval(elstr)
                            ieinit = elstr + "(" + ieinit
                    for ieph in range(feItem.rowCount()):
                        if feItem.child(ieph, 0).text() != '_object':
                            if feItem.child(ieph, 0).text() == 'properties':
                                pItem = feItem.child(ieph, 0)
                                for iep, arg_def in zip(range(
                                        pItem.rowCount()),
                                        list(zip(*self.getParams(elstr)))[1]):
                                    paraname = str(pItem.child(iep, 0).text())
                                    paravalue = str(pItem.child(iep, 1).text())
                                    if paravalue != str(arg_def) or\
                                            paravalue == 'bl':
                                        if paraname.lower() not in\
                                                ['basefe']:
                                            paravalue = self.quotize(paravalue)
                                        ieinit += '\n{2}{0}={1},'.format(
                                            paraname, paravalue, myTab)
                    codeDeclarations += '{0} = {1})\n\n'.format(
                        feItem.text(), str.rstrip(ieinit, ","))
#                    break

        self.progressBar.setValue(30)
        self.progressBar.setFormat("Adding optical elements.")
        outBeams = ['None']

        for oeId in self.beamLine.flowU:
            for ie in range(self.rootBLItem.rowCount()):
                if str(self.rootBLItem.child(ie, 0).data(qt.Qt.UserRole)) == \
                        oeId:
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
                                importStr = 'import {0}'.format(
                                    klass.__module__)
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
                                             'material2', 'figureerror']:
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
                                        paraname = str(
                                            mItem.child(iep, 0).text())
                                        paravalue = str(
                                            mItem.child(iep, 1).text())
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
                                        if len(re.findall(
                                                'sources', elstr)) > 0\
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
                    ieinit += elstr + "("
            for iep in range(tItem.rowCount()):
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
                                # code below parses the properties of other
                                # plots. do we need it?
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

                                ieinit += u'\n{2}{0}={1},'.format(
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

            runParams = dict(self.getParams(elstr))

            for ie in range(self.rootRunItem.rowCount()):
                if self.rootRunItem.child(ie, 0).text() != '_object':
                    paraname = str(self.rootRunItem.child(ie, 0).text())
                    paravalue = str(self.rootRunItem.child(ie, 1).text())
                    if paraname == "plots":
                        paravalue = str(self.rootPlotItem.text())
                    if paraname == "backend":
                        paravalue = 'r\"{0}\"'.format(paravalue)
                    argVal = runParams.get(paraname)
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
                fileObject = open(self.saveFileName, 'w',
                                  encoding="utf-8")
                fileObject.write(self.codeEdit.toPlainText())
                fileObject.close
                saveStatus = True
                saveMsg = 'Script saved to {}'.format(
                    os.path.basename(str(self.saveFileName)))
                self.tabs.setTabText(
                    6, os.path.basename(str(self.saveFileName)))
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

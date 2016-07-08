# -*- coding: utf-8 -*-
u"""
Using xrtQook for script generation
-----------------------------------

- Start xrtQook: type ``python xrtQook.pyw`` from xrt/xrtQook or, if you have
  installed xrt by running setup.py, type ``xrtQook.pyw`` from any location.
- Rename beamLine to myTestBeamline by double clicking on it (you do not have
  to, only for demonstration).
- Right-click on myTestBeamline and Add Source → BendingMagnet.

  .. image:: _images/qookTutor1.png
     :scale: 60 %

- In its properties change eMin to 10000-1 and eMax to 10000+1. The middle of
  this range will be used to automatically align crystals (one crystal in this
  example). Blue color indicates non-default values. These will be necessarily
  included into the generated script. All the default-valued parameters do not
  propagate into the script.

  .. image:: _images/qookTutor2.png
     :scale: 60 %

- In Materials tab create a crystalline material:
  right click → Add Material → CrystalSi. This will create a Si111 crystal at
  room temperature.

  .. image:: _images/qookTutor3.png
     :scale: 60 %

- In Beamline tab right click → Add OE → OE. This will add an optical element
  with a flat surface.

  .. note::
     The sequence of the inserted optical elements does matter! This sequence
     determines the order of beam propagation.

  .. image:: _images/qookTutor4.png
     :scale: 60 %

- In its properties select the created crystal as 'material', put [0, 20000, 0]
  as 'center' (i.e. 20 m from source) and “auto” (with or without quotes) as
  'pitch'.
- Add a screen to the beamline. Give it [0, 21000, “auto”] as 'center'. Its
  height -- the last coordinate -- will be automatically calculated from the
  previous elements.

  .. image:: _images/qookTutor5.png
     :scale: 60 %

- Add methods to the beamline elements (with right click):

  a) shine() to the source,
  b) reflect() to the optical element and select a proper beam for it – the
     global beam from the source, it has a default name but you may rename it
     in the shine();
  c) expose() to the screen and select a proper beam for it – the global beam
     from the OE, it has a default name but you may rename it in the reflect();

  Red colored words indicate None as a selected beam. If you continue with the
  script generation, the script will result in a run time error.

  .. image:: _images/qookTutor6.png
     :scale: 60 %

- Add a plot in Plots tab and select the local screen beam.
- Save the beamline layout as xml.
- Generate python script (the button with a code page and the python logo),
  save the script and run it.
- In the console output you can read the actual pitch (Bragg angle) for the
  crystal and the screen position.

  .. image:: _images/qookTutor7.png
     :scale: 60 %

"""
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "07 Mar 2016"
__version__ = "1.0"

import os
import sys
import textwrap

from datetime import date
import inspect
import re
import xml.etree.ElementTree as ET
from functools import partial

try:
    from spyderlib.widgets.sourcecode import codeeditor
    isSpyderlib = True
except ImportError:
    isSpyderlib = False

try:
    import pyopencl as cl
    isOpenCL = True
except ImportError:
    isOpenCL = False

try:
    from spyderlib.widgets.externalshell import pythonshell
    isSpyderConsole = True
except ImportError:
    isSpyderConsole = False

try:
    from spyderlib.utils.inspector.sphinxify import (CSS_PATH, sphinxify,
                                                     generate_context)
    isSphinx = True
except (ImportError, TypeError):
    sphinxify = sphinx_version = None  # analysis:ignore
    isSphinx = False
if not isSphinx:
    try:
        from spyderlib.utils.help.sphinxify import (CSS_PATH, sphinxify,  # analysis:ignore
                                                    generate_context)  # analysis:ignore
        isSphinx = True
    except (ImportError, TypeError):
        pass

try:
    from matplotlib.backends import qt_compat
except ImportError:
    from matplotlib.backends import qt4_compat
    qt_compat = qt4_compat

if qt_compat.QT_API == 'PySide':
    QtName = "PySide"
    import PySide
    from PySide import QtGui, QtCore
    import PySide.QtGui as myQtGUI
    import PySide.QtWebKit as myQtWeb
elif qt_compat.QT_API == 'PyQt5':
    QtName = "PyQt5"
    from PyQt5 import QtGui, QtCore
    import PyQt5.QtWidgets as myQtGUI
    import PyQt5.QtWebKitWidgets as myQtWeb
elif qt_compat.QT_API == 'PyQt4':
    QtName = "PyQt4"
    from PyQt4 import QtGui, QtCore
    import PyQt4.QtGui as myQtGUI
    import PyQt4.QtWebKit as myQtWeb
else:
    raise ImportError

QWidget, QApplication, QAction, QTabWidget, QToolBar, QStatusBar, QTreeView,\
    QShortcut, QAbstractItemView, QHBoxLayout, QVBoxLayout, QSplitter,\
    QComboBox, QMenu, QListWidget, QTextEdit, QMessageBox, QFileDialog,\
    QListWidgetItem = (
        myQtGUI.QWidget, myQtGUI.QApplication, myQtGUI.QAction,
        myQtGUI.QTabWidget, myQtGUI.QToolBar, myQtGUI.QStatusBar,
        myQtGUI.QTreeView, myQtGUI.QShortcut, myQtGUI.QAbstractItemView,
        myQtGUI.QHBoxLayout, myQtGUI.QVBoxLayout, myQtGUI.QSplitter,
        myQtGUI.QComboBox, myQtGUI.QMenu, myQtGUI.QListWidget,
        myQtGUI.QTextEdit, myQtGUI.QMessageBox, myQtGUI.QFileDialog,
        myQtGUI.QListWidgetItem)
QIcon, QFont, QKeySequence, QStandardItemModel, QStandardItem, QPixmap =\
    (QtGui.QIcon, QtGui.QFont, QtGui.QKeySequence, QtGui.QStandardItemModel,
     QtGui.QStandardItem, QtGui.QPixmap)
QWebView = myQtWeb.QWebView

sys.path.append(os.path.join('..', '..'))

import xrt
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

path_to_xrt = os.path.dirname(os.path.dirname(
    os.path.abspath(xrt.__file__)))
myTab = 4*" "


class XrtQook(QWidget):
    def __init__(self):
        super(XrtQook, self).__init__()
        self.xrtQookDir = os.path.dirname(os.path.abspath(__file__))
        self.setAcceptDrops(True)
        iconsDir = os.path.join(self.xrtQookDir, '_icons')

        self.setWindowIcon(QIcon(os.path.join(iconsDir, 'xrQt1.ico')))

        newBLAction = QAction(
            QIcon(os.path.join(iconsDir, 'filenew.png')),
            'New Beamline Layout',
            self)
        newBLAction.setShortcut('Ctrl+N')
        newBLAction.setIconText('New Beamline Layout')
        newBLAction.triggered.connect(self.newBL)

        loadBLAction = QAction(
            QIcon(os.path.join(iconsDir, 'fileopen.png')),
            'Load Beamline Layout',
            self)
        loadBLAction.setShortcut('Ctrl+L')
        loadBLAction.triggered.connect(self.importLayout)

        saveBLAction = QAction(
            QIcon(os.path.join(iconsDir, 'filesave.png')),
            'Save Beamline Layout',
            self)
        saveBLAction.setShortcut('Ctrl+S')
        saveBLAction.triggered.connect(self.exportLayout)

        saveBLAsAction = QAction(
            QIcon(os.path.join(iconsDir, 'filesaveas.png')),
            'Save Beamline Layout As ...',
            self)
        saveBLAsAction.setShortcut('Ctrl+A')
        saveBLAsAction.triggered.connect(self.exportLayoutAs)

        generateCodeAction = QAction(
            QIcon(os.path.join(iconsDir, 'pythonscript.png')),
            'Generate Python Script',
            self)
        generateCodeAction.setShortcut('Ctrl+G')
        generateCodeAction.triggered.connect(self.generateCode)

        saveScriptAction = QAction(
            QIcon(os.path.join(iconsDir, 'pythonscriptsave.png')),
            'Save Python Script',
            self)
        saveScriptAction.setShortcut('Alt+S')
        saveScriptAction.triggered.connect(self.saveCode)

        saveScriptAsAction = QAction(
            QIcon(os.path.join(iconsDir, 'pythonscriptsaveas.png')),
            'Save Python Script As ...',
            self)
        saveScriptAsAction.setShortcut('Alt+A')
        saveScriptAsAction.triggered.connect(self.saveCodeAs)

        runScriptAction = QAction(
            QIcon(os.path.join(iconsDir, 'run.png')),
            'Save Python Script And Run',
            self)
        runScriptAction.setShortcut('Ctrl+R')
        runScriptAction.triggered.connect(self.execCode)

        OCLAction = QAction(
            QIcon(os.path.join(iconsDir, 'GPU4.png')),
            'OpenCL Info',
            self)
        if isOpenCL:
            OCLAction.setShortcut('Alt+I')
            OCLAction.triggered.connect(self.showOCLinfo)

        tutorAction = QAction(
            QIcon(os.path.join(iconsDir, 'home.png')),
            'Show Welcome Screen',
            self)
        tutorAction.setShortcut('Ctrl+H')
        tutorAction.triggered.connect(self.showWelcomeScreen)

        aboutAction = QAction(
            QIcon(os.path.join(iconsDir, 'readme.png')),
            'About xrtQook',
            self)
        aboutAction.setShortcut('Ctrl+I')
        aboutAction.triggered.connect(self.aboutCode)

        self.tabs = QTabWidget()
        self.toolBar = QToolBar('File')
        self.statusBar = QStatusBar()
        self.statusBar.setSizeGripEnabled(False)
        # self.statusBar.setStyleSheet("border:1px solid rgb(0, 0, 0);")

        self.toolBar.addAction(newBLAction)
        self.toolBar.addAction(loadBLAction)
        self.toolBar.addAction(saveBLAction)
        self.toolBar.addAction(saveBLAsAction)
        self.toolBar.addSeparator()
        self.toolBar.addSeparator()
        self.toolBar.addAction(generateCodeAction)
        self.toolBar.addAction(saveScriptAction)
        self.toolBar.addAction(saveScriptAsAction)
        self.toolBar.addAction(runScriptAction)
        self.toolBar.addSeparator()
        self.toolBar.addSeparator()
        if isOpenCL:
            self.toolBar.addAction(OCLAction)
        self.toolBar.addAction(tutorAction)
        self.toolBar.addAction(aboutAction)

        self.xrtModules = ['rsources', 'rscreens', 'rmats', 'roes', 'rapts',
                           'rrun', 'raycing', 'xrtplot', 'xrtrun']

        self.objectFlag = QtCore.Qt.ItemFlags(-33)
        self.paramFlag = QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsEnabled | ~
                                             QtCore.Qt.ItemIsEditable |
                                             QtCore.Qt.ItemIsSelectable)
        self.valueFlag = QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsEnabled |
                                             QtCore.Qt.ItemIsEditable |
                                             QtCore.Qt.ItemIsSelectable)
        self.checkFlag = QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsEnabled | ~
                                             QtCore.Qt.ItemIsEditable |
                                             QtCore.Qt.ItemIsUserCheckable |
                                             QtCore.Qt.ItemIsSelectable)

        self.tree = QTreeView()
        self.matTree = QTreeView()
        self.plotTree = QTreeView()
        self.runTree = QTreeView()

        self.defaultFont = QFont("Courier New", 9)
        if isSphinx:
            self.webHelp = QWebView()
            self.webHelp.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.webHelp.customContextMenuRequested.connect(self.docMenu)
        else:
            self.webHelp = QTextEdit()
            self.webHelp.setFont(self.defaultFont)
            self.webHelp.setReadOnly(True)

        for itree in [self.tree, self.matTree, self.plotTree, self.runTree]:
            itree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            itree.clicked.connect(self.showDoc)

        self.plotTree.customContextMenuRequested.connect(self.plotMenu)
        self.matTree.customContextMenuRequested.connect(self.matMenu)
        self.tree.customContextMenuRequested.connect(self.openMenu)

        if isSpyderlib:
            self.codeEdit = codeeditor.CodeEditor(self)
            self.codeEdit.setup_editor(linenumbers=True, markers=True,
                                       tab_mode=False, language='py',
                                       font=self.defaultFont,
                                       color_scheme='Pydev')
            if QtName == "PyQt5":
                self.codeEdit.zoom_in.connect(lambda: self.zoom(1))
                self.codeEdit.zoom_out.connect(lambda: self.zoom(-1))
                self.codeEdit.zoom_reset.connect(lambda: self.zoom(0))
            else:
                self.connect(self.codeEdit,
                             QtCore.SIGNAL('zoom_in()'),
                             lambda: self.zoom(1))
                self.connect(self.codeEdit,
                             QtCore.SIGNAL('zoom_out()'),
                             lambda: self.zoom(-1))
                self.connect(self.codeEdit,
                             QtCore.SIGNAL('zoom_reset()'),
                             lambda: self.zoom(0))
            QShortcut(QKeySequence.ZoomIn, self, lambda: self.zoom(1))
            QShortcut(QKeySequence.ZoomOut, self, lambda: self.zoom(-1))
            QShortcut("Ctrl+0", self, lambda: self.zoom(0))
            for action in self.codeEdit.menu.actions()[-3:]:
                self.codeEdit.menu.removeAction(action)
        else:
            self.codeEdit = QTextEdit()
            self.codeEdit.setFont(self.defaultFont)

        self.descrEdit = QTextEdit()
        self.descrEdit.setFont(self.defaultFont)
        self.descrEdit.textChanged.connect(self.updateDescription)

        self.setGeometry(100, 100, 1100, 600)

        if isSpyderConsole:
            self.codeConsole = pythonshell.ExternalPythonShell(
                wdir=os.path.dirname(__file__))

        else:
            self.qprocess = QtCore.QProcess()
            self.qprocess.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.qprocess.readyReadStandardOutput.connect(self.readStdOutput)
            QShortcut("Ctrl+X", self, self.qprocess.kill)
            self.codeConsole = QTextEdit()
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

        canvasBox = QHBoxLayout()
        canvasSplitter = QSplitter()
        canvasSplitter.setChildrenCollapsible(False)

        mainWidget = QWidget()
        mainWidget.setMinimumWidth(465)
        docWidget = QWidget()
        docWidget.setMinimumWidth(300)
        mainBox = QVBoxLayout()
        docBox = QVBoxLayout()

        mainBox.addWidget(self.toolBar)
        mainBox.addWidget(self.tabs)
        mainBox.addWidget(self.statusBar)
        docBox.addWidget(self.webHelp)

        mainWidget.setLayout(mainBox)
        docWidget.setLayout(docBox)
        docWidget.setStyleSheet("border:1px solid rgb(20, 20, 20);")

        canvasBox.addWidget(canvasSplitter)

        canvasSplitter.addWidget(mainWidget)
        canvasSplitter.addWidget(docWidget)
        self.setLayout(canvasBox)
        self.initAllModels()
        self.initAllTrees()

    def readStdOutput(self):
        output = self.qprocess.readAllStandardOutput()
        self.codeConsole.append(str(output))

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
        menu = QMenu()
        menu.addAction("Zoom In",
                       lambda: self.zoomDoc(1))
        menu.addAction("Zoom Out", lambda: self.zoomDoc(-1))
        menu.addAction("Zoom reset", lambda: self.zoomDoc(0))
        menu.exec_(self.webHelp.mapToGlobal(position))

    def zoomDoc(self, factor):
        """Zoom in/out/reset"""
        try:
            textSize = self.webHelp.textSizeMultiplier()
            if factor == 0:
                textSize = 1.
            elif factor == 1:
                if textSize < 2:
                    textSize += 0.1
            elif textSize > 0.1:
                textSize -= 0.1

            self.webHelp.setTextSizeMultiplier(textSize)
        except AttributeError:
            pass

    def initAllTrees(self):
        # runTree view
        self.runTree.setModel(self.runModel)
        self.runTree.setAlternatingRowColors(True)
        self.runTree.setSortingEnabled(False)
        self.runTree.setHeaderHidden(False)
        self.runTree.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.runTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        for name, obj in inspect.getmembers(xrtrun):
            if inspect.isfunction(obj) and name == "run_ray_tracing":
                if inspect.getargspec(obj)[3] is not None:
                    self.addObject(self.runTree,
                                   self.rootRunItem,
                                   '{0}.{1}'.format(xrtrun.__name__, name))
                    for arg, argVal in zip(inspect.getargspec(obj)[0],
                                           inspect.getargspec(obj)[3]):
                        if arg.lower() == "plots":
                            argVal = self.rootPlotItem.text()
                        if arg.lower() == "beamline":
                            argVal = self.rootBLItem.text()
                        self.addParam(self.rootRunItem, arg, argVal)
        self.addCombo(self.runTree, self.rootRunItem)
        self.runTree.expand(self.rootRunItem.index())
        self.runTree.setColumnWidth(0, int(self.runTree.width()/3))

        # plotTree view
        self.plotTree.setModel(self.plotModel)
        self.plotTree.setAlternatingRowColors(True)
        self.plotTree.setSortingEnabled(False)
        self.plotTree.setHeaderHidden(False)
        self.plotTree.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.plotTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        # materialsTree view
        self.matTree.setModel(self.materialsModel)
        self.matTree.setAlternatingRowColors(True)
        self.matTree.setSortingEnabled(False)
        self.matTree.setHeaderHidden(False)
        self.matTree.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.matTree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
        # BLTree view
        self.tree.setModel(self.beamLineModel)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tree.model().setHorizontalHeaderLabels(['Parameter', 'Value'])
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
        self.tabs.tabBar().setTabTextColor(0, QtCore.Qt.black)
        self.tabs.tabBar().setTabTextColor(2, QtCore.Qt.black)

    def initAllModels(self):
        self.beamLineModel = QStandardItemModel()
        self.addValue(self.beamLineModel.invisibleRootItem(), "beamLine")
        self.beamLineModel.itemChanged.connect(self.colorizeChangedParam)
        self.rootBLItem = self.beamLineModel.item(0, 0)

        self.boolModel = QStandardItemModel()
        self.boolModel.appendRow(QStandardItem('False'))
        self.boolModel.appendRow(QStandardItem('True'))

        self.densityModel = QStandardItemModel()
        self.densityModel.appendRow(QStandardItem('histogram'))
        self.densityModel.appendRow(QStandardItem('kde'))

        self.oclPrecnModel = QStandardItemModel()
        for opm in ['auto', 'float32', 'float64']:
            self.oclPrecnModel.appendRow(QStandardItem(opm))

        self.plotAxisModel = QStandardItemModel()
        for ax in ['x', 'y', 'z', 'x\'', 'z\'', 'energy']:
            self.addValue(self.plotAxisModel, ax)

        self.OCLModel = QStandardItemModel()
        oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                     "None",
                                                     "None")
        oclNoneItem, oclNoneItemName = self.addParam(self.OCLModel,
                                                     "auto",
                                                     "auto")
        if isOpenCL:
            iDeviceCPU = []
            iDeviceGPU = []
            for platform in cl.get_platforms():
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

            for iplatform, platform in enumerate(cl.get_platforms()):
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

        self.materialsModel = QStandardItemModel()
        self.rootMatItem = self.materialsModel.invisibleRootItem()
        self.rootMatItem.setText("Materials")
        self.materialsModel.itemChanged.connect(self.colorizeChangedParam)
        self.addProp(self.materialsModel.invisibleRootItem(), "None")

        self.beamModel = QStandardItemModel()
        self.beamModel.appendRow(QStandardItem("None"))
        self.rootBeamItem = self.beamModel.invisibleRootItem()
        self.rootBeamItem.setText("Beams")

        self.fluxDataModel = QStandardItemModel()
        self.fluxDataModel.appendRow(QStandardItem("auto"))
        for rfName, rfObj in inspect.getmembers(raycing):
            if rfName.startswith('get_') and\
                    rfName != "get_output":
                flItem = QStandardItem(rfName.replace("get_", ''))
                self.fluxDataModel.appendRow(flItem)

        self.fluxKindModel = QStandardItemModel()
        for flKind in ['total', 'power', 's', 'p',
                       '+45', '-45', 'left', 'right']:
            flItem = QStandardItem(flKind)
            self.fluxKindModel.appendRow(flItem)

        self.polarizationsModel = QStandardItemModel()
        for pol in ['horizontal', 'vertical',
                    '+45', '-45', 'left', 'right', 'None']:
            polItem = QStandardItem(pol)
            self.polarizationsModel.appendRow(polItem)

        self.matKindModel = QStandardItemModel()
        for mtKind in ['mirror', 'thin mirror',
                       'plate', 'lens', 'grating', 'FZP']:
            mtItem = QStandardItem(mtKind)
            self.matKindModel.appendRow(mtItem)

        self.matTableModel = QStandardItemModel()
        for mtTable in ['Chantler', 'Henke', 'BrCo']:
            mtTItem = QStandardItem(mtTable)
            self.matTableModel.appendRow(mtTItem)

        self.shapeModel = QStandardItemModel()
        for shpEl in ['rect', 'round']:
            shpItem = QStandardItem(shpEl)
            self.shapeModel.appendRow(shpItem)

        self.matGeomModel = QStandardItemModel()
        for mtGeom in ['Bragg reflected', 'Bragg transmitted',
                       'Laue reflected', 'Laue transmitted',
                       'Fresnel']:
            mtGItem = QStandardItem(mtGeom)
            self.matGeomModel.appendRow(mtGItem)

        self.aspectModel = QStandardItemModel()
        for aspect in ['equal', 'auto']:
            aspItem = QStandardItem(aspect)
            self.aspectModel.appendRow(aspItem)

        self.distEModelG = QStandardItemModel()
        for distEMod in ['None', 'normal', 'flat', 'lines']:
            dEItem = QStandardItem(distEMod)
            self.distEModelG.appendRow(dEItem)

        self.distEModelS = QStandardItemModel()
        for distEMod in ['eV', 'BW']:
            dEItem = QStandardItem(distEMod)
            self.distEModelS.appendRow(dEItem)

        self.rayModel = QStandardItemModel()
        for iray, ray in enumerate(['Good', 'Out', 'Over', 'Alive']):
            rayItem, rItemStr = self.addParam(self.rayModel, ray, iray+1)
            rayItem.setCheckable(True)
            rayItem.setCheckState(QtCore.Qt.Checked)

        self.plotModel = QStandardItemModel()
        self.addValue(self.plotModel.invisibleRootItem(), "plots")
        # self.plotModel.appendRow(QStandardItem("plots"))
        self.rootPlotItem = self.plotModel.item(0, 0)
        self.plotModel.itemChanged.connect(self.colorizeChangedParam)
        self.plotModel.invisibleRootItem().setText("plots")

        self.runModel = QStandardItemModel()
        self.addProp(self.runModel.invisibleRootItem(),
                     "run_ray_tracing()")
        # self.runModel.appendRow(QStandardItem("RayTracingParameters"))
        self.runModel.itemChanged.connect(self.colorizeChangedParam)
        self.rootRunItem = self.runModel.item(0, 0)

        self.prefixtab = "\t"
        self.ntab = 1
        self.cpChLevel = 0
        self.saveFileName = ""
        self.layoutFileName = ""
        self.writeCodeBox("")
        self.setWindowTitle("xrtQook")
        self.isEmpty = True
        self.curObj = None
        self.blColorCounter = 0
        self.pltColorCounter = 0
        self.fileDescription = ""
        self.descrEdit.setText("")
        self.showWelcomeScreen()

    def newBL(self):
        if not self.isEmpty:
            msgBox = QMessageBox()
            if msgBox.warning(self,
                              'Warning',
                              'Current layout will be purged. Continue?',
                              buttons=QMessageBox.Yes |
                              QMessageBox.No,
                              defaultButton=QMessageBox.No)\
                    == QMessageBox.Yes:
                self.initAllModels()
                self.initAllTrees()
                self.tabs.setCurrentWidget(self.tree)

    def writeCodeBox(self, text):
        if isSpyderlib:
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
        self.webHelp.page().setLinkDelegationPolicy(0)
        argSpecStr = '('
        for arg, argVal in self.getParams(obj):
            argSpecStr += '{0}={1}, '.format(arg, argVal)
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

        argDocStr = '{0}{1}\n\n'.format(myTab, headerDoc) if\
            objP.__doc__ is not None else "\n\n"
        dNames, dVals = self.getArgDescr(obj)
        if len(dNames) > 0:
            argDocStr += '{0}Properties\n{0}{1}\n\n'.format(myTab, 10*'-')
        for dName, dVal in zip(dNames, dVals):
            argDocStr += u'{2}{0}*: {1}\n\n'.format(dName, dVal, myTab)

        if isSphinx:
            err = None
            try:
                cntx = generate_context(
                    name=nameStr,
                    argspec=argSpecStr,
                    note=noteStr,
                    img_path=self.xrtQookDir,
                    math=True)
            except TypeError as err:
                cntx = generate_context(
                    name=nameStr,
                    argspec=argSpecStr,
                    note=noteStr,
                    math=True)

            html_text = sphinxify(textwrap.dedent(argDocStr), cntx)
            if err is None:
                html2 = re.findall(' {4}return.*', html_text)[0]
                sbsPath = re.sub('img_name',
                                 'attr',
                                 re.sub('\\\\', '/', html2))
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
            self.webHelp.setHtml(new_html, QtCore.QUrl(CSS_PATH))
        else:
            argDocStr = u'{0}\nDefiniiton: {1}\n\nType: {2}\n\n\n'.format(
                nameStr.upper(), argSpecStr, noteStr) + argDocStr
            self.webHelp.setText(textwrap.dedent(argDocStr))
            self.webHelp.setReadOnly(True)

    def updateDescription(self):
        self.fileDescription = self.descrEdit.toPlainText()
        img_path = __file__ if self.layoutFileName == "" else\
            self.layoutFileName
        self.showTutorial(self.fileDescription,
                          "Description",
                          os.path.dirname(os.path.abspath(img_path)))

    def showWelcomeScreen(self):
        argDescr = """

        .. image:: _images/qookSplashSmall_ani.gif

        xrtQook is a qt-based GUI for using xrt without having to write python
        scripts. See a short startup `tutorial <tutorial>`_.

        """
        self.showTutorial(argDescr,
                          "xrtQook",
                          os.path.dirname(os.path.abspath(__file__)),
                          delegateLink=True)

    def showDescrByTab(self, tab):
        if tab == 4:
            self.updateDescription()

    def showTutorial(self, argDocStr, name, img_path, delegateLink=False):
        if isSphinx:
            err = None
            try:
                cntx = generate_context(
                    name=name,
                    argspec="",
                    note="",
                    img_path=img_path,
                    math=True)
            except TypeError as err:
                cntx = generate_context(
                    name=name,
                    argspec="",
                    note="",
                    math=True)
            html_text = sphinxify(textwrap.dedent(argDocStr), cntx)
            if err is None:
                html2 = re.findall(' {4}return.*', html_text)[0]
                sbsPath = re.sub('img_name',
                                 'attr',
                                 re.sub('\\\\', '/', html2))
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
            self.webHelp.setHtml(new_html, QtCore.QUrl(CSS_PATH))
            if delegateLink:
                self.webHelp.page().setLinkDelegationPolicy(1)
                self.webHelp.page().linkClicked.connect(partial(
                    self.showTutorial,
                    __doc__[229:],
                    "Using xrtQook for script generation",
                    self.xrtQookDir))
            else:
                self.webHelp.page().setLinkDelegationPolicy(0)
            self.curObj = None

    def showOCLinfo(self):
        argDocStr = ""
        for iplatform, platform in enumerate(cl.get_platforms()):
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
                fVals = [cl.device_type.to_string(device.type),
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
        self.webHelp.page().setLinkDelegationPolicy(0)
        if isSphinx:
            cntx = generate_context(
                name="OpenCL Platforms and Devices",
                argspec="",
                note="",
                math=True)
            html_text = sphinxify(textwrap.dedent(argDocStr), cntx)
            self.webHelp.setHtml(html_text, QtCore.QUrl(CSS_PATH))
        else:
            argDocStr = "OpenCL Platforms and Devices\n\n" + argDocStr
            self.webHelp.setText(textwrap.dedent(argDocStr))
            self.webHelp.setReadOnly(True)

    def addObject(self, view, parent, obj):
        child0 = QStandardItem('_object')
        child1 = QStandardItem(str(obj))
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
        child0 = QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = QStandardItem(str(value))
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
        if source is None:
            parent.appendRow([child0, child1])
        else:
            parent.insertRow(source.row() + 1, [child0, child1])
        return child0, child1

    def addProp(self, parent, propName):
        """Add non-editable Item"""
        child0 = QStandardItem(str(propName))
        child0.setFlags(self.paramFlag)
        child1 = QStandardItem()
        child1.setFlags(self.paramFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        parent.appendRow([child0, child1])
        return child0

    def addValue(self, parent, value, source=None):
        """Add editable Item"""
        child0 = QStandardItem(str(value))
        child0.setFlags(self.valueFlag)
        child1 = QStandardItem()
        child1.setFlags(self.paramFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        if source is None:
            parent.appendRow([child0, child1])
        else:
            parent.insertRow(source.row() + 1, [child0, child1])
        return child0

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
        self.isEmpty = False

    def getParams(self, obj):
        args = []
        argVals = []
        objRef = eval(str(obj))
        if inspect.isclass(objRef):
            for parent in (inspect.getmro(objRef))[:-1]:
                for namef, objf in inspect.getmembers(parent):
                    if inspect.ismethod(objf) or inspect.isfunction(objf):
                        if namef == "__init__" and\
                                inspect.getargspec(objf)[3] is not None:
                            for arg, argVal in zip(
                                    inspect.getargspec(objf)[0][1:],
                                    inspect.getargspec(objf)[3]):
                                if arg == 'bl':
                                    argVal = self.rootBLItem.text()
                                if arg not in args:
                                    args.append(arg)
                                    argVals.append(argVal)
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
                                    if arg not in args:
                                        args.append(arg)
                                        argVals.append(argVal)
#        elif inspect.ismethod(objRef):
#            argList = inspect.getargspec(objRef)
#            if argList[3] is not None:
#                args = argList[0][1:]
#                argVals = argList[3]
#        else:
#            argList = inspect.getargspec(objRef)
#            if argList[3] is not None:
#                args = argList[0]
#                argVals = argList[3]
        elif inspect.ismethod(objRef) or inspect.isfunction(objRef):
            argList = inspect.getargspec(objRef)
            if argList[3] is not None:
                if objRef.__name__ == 'run_ray_tracing':
                    args = argList[0]
                    argVals = argList[3]
                else:
                    args = argList[0][1:]
                    argVals = argList[3]
        return zip(args, argVals)

    def colorizeChangedParam(self, item):
        parent = item.parent()
        if parent is not None and\
                item.column() == 1 and\
                item.isEnabled():
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
                        color = QtCore.Qt.red
                        counter = 1
                    elif parent.child(itemRow, 0).foreground().color() ==\
                            QtCore.Qt.red:
                        color = QtCore.Qt.black
                        counter = -1
                    else:
                        counter = 0
                    if item.model() == self.beamLineModel:
                        self.blColorCounter += counter
                    else:
                        self.pltColorCounter += counter
                    self.colorizeTabText(item)
                    if color is not None:
                        self.setIFontColor(parent.child(itemRow, 0), color)
                        if parent.parent() != self.rootPlotItem:
                            self.setIFontColor(parent.parent(), color)
                if parent.child(itemRow, 0).foreground().color() !=\
                        QtCore.Qt.red:
                    for defArg, defArgVal in self.getParams(obj):
                        if str(defArg) == str(parent.child(itemRow, 0).text()):
                            if str(defArgVal) != str(item.text()):
                                color = QtCore.Qt.blue
                            else:
                                color = QtCore.Qt.black
                            self.setIFontColor(parent.child(itemRow, 0), color)
                            break

    def colorizeTabText(self, item):
        if item.model() == self.beamLineModel:
            color = QtCore.Qt.red if self.blColorCounter > 0 else\
                QtCore.Qt.black
            self.tabs.tabBar().setTabTextColor(0, color)
        elif item.model() == self.plotModel:
            color = QtCore.Qt.red if self.pltColorCounter > 0 else\
                QtCore.Qt.black
            self.tabs.tabBar().setTabTextColor(2, color)

    def addMethod(self, name, parentItem, fdoc):
        elstr = str(parentItem.text())
        fdoc = fdoc[0].replace("Returned values: ", '').split(',')

        methodItem = self.addProp(parentItem, name.split('.')[-1] + '()')
        self.setIItalic(methodItem)
        methodProps = self.addProp(methodItem, 'parameters')
        self.addObject(self.tree, methodItem, name)

        for arg, argVal in self.getParams(name):
                if arg == 'bl':
                    argVal = self.rootBLItem.text()
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
            self.beamModel.appendRow(QStandardItem(beamName))

        self.showDoc(methodItem.index())
        self.addCombo(self.tree, methodItem)
        self.tree.expand(methodItem.index())
        self.tree.expand(methodOut.index())
        self.tree.expand(methodProps.index())
        self.tree.setCurrentIndex(methodProps.index())
        self.tree.setColumnWidth(0, int(self.tree.width()/3))
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
                                    # child0 = QStandardItem(str(arg))
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
        argDescName = []
        argDescValue = []
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
                            descName = desc[0]
                            descBody = desc[1].strip("\* ")
                            if descName not in argDescName:
                                argDescName.append(descName)
                                argDescValue.append(descBody)

        if inspect.isclass(objRef):
            for parent in inspect.getmro(objRef)[:-1]:
                for namef, objf in inspect.getmembers(parent):
                    if (inspect.ismethod(objf) or inspect.isfunction(objf)):
                        if namef == "__init__" or namef.endswith("pop_kwargs"):
                            parseDescr(objf)
        else:
            parseDescr(objRef)

        return argDescName, argDescValue

    def addMaterial(self, name, obj):
        for i in range(99):
            matName = self.classNameToStr(name) + '{:02d}'.format(i+1)
            dupl = False
            for ibm in range(self.rootMatItem.rowCount()):
                if str(self.rootMatItem.child(ibm, 0).text()) == str(matName):
                    dupl = True
            if not dupl:
                break

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
        self.isEmpty = False

    def moveItem(self, mvDir, view, item):
        oldRowNumber = item.index().row()
        statusExpanded = view.isExpanded(item.index())
        parent = item.parent()
        self.flattenElement(view, item)
        newItem = parent.takeRow(oldRowNumber)
        parent.insertRow(oldRowNumber + mvDir, newItem)
        self.addCombo(view, newItem[0])
        view.setExpanded(newItem[0].index(), statusExpanded)

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
                    child0n = QStandardItem(child0.text())
                    child0n.setFlags(child0.flags())
                    child0n.setForeground(child0.foreground())
                    if child1 is not None:
                        child1n = QStandardItem(child1.text())
                        child1n.setFlags(child1.flags())
                        itemTo.appendRow([child0n, child1n])
                    else:
                        itemTo.appendRow(child0n)
                    self.copyChildren(child0n, child0)
            self.cpChLevel -= 1
        else:
            pass

    def deleteElement(self, view, item):
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
                    if item.text() == "output" and\
                            iWidget.model() == self.beamModel:
                        self.beamModel.takeRow(iWidget.currentIndex())
            self.deleteElement(view, iItem)
        else:
            self.colorizeTabText(item)
            if item.parent() is not None:
                item.parent().removeRow(item.index().row())
            else:
                item.model().invisibleRootItem().removeRow(item.index().row())

    def exportLayout(self):
        saveStatus = False
        if self.layoutFileName == "":
            saveDialog = QFileDialog()
            saveDialog.setFileMode(QFileDialog.AnyFile)
            saveDialog.setAcceptMode(QFileDialog.AcceptSave)
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
                self.flattenElement(view, item)
                if item == self.rootPlotItem and\
                        self.rootPlotItem.rowCount() == 0:
                    item = self.plotModel.invisibleRootItem()
                    item.setEditable(True)
                self.exportModel(item)
            self.confText += '<description>\n{0}\n</description>\n'.format(
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
            self.statusBar.showMessage(messageStr, 3000)
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
                if child1 is not None:
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
            msgBox = QMessageBox()
            if msgBox.warning(self,
                              'Warning',
                              'Current layout will be overwritten. Continue?',
                              buttons=QMessageBox.Yes |
                              QMessageBox.No,
                              defaultButton=QMessageBox.No)\
                    == QMessageBox.Yes:
                self.isEmpty = True
        if self.isEmpty:
            self.initAllModels()
            self.initAllTrees()
            openFileName = ""
            openDialog = QFileDialog()
            openDialog.setFileMode(QFileDialog.ExistingFile)
            openDialog.setAcceptMode(QFileDialog.AcceptOpen)
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
                        if tree is not None:
                            self.checkDefaults(None, rootModel)
                            tmpBlColor = self.blColorCounter
                            tmpPltColor = self.pltColorCounter
                            self.addCombo(tree, rootModel)
                            self.blColorCounter = tmpBlColor
                            self.pltColorCounter = tmpPltColor
                        self.colorizeTabText(rootModel)
                        msgStr = " {0:d} percent done.".format(int(i*100/5))
                        self.statusBar.showMessage(ldMsg + msgStr)
                    self.layoutFileName = openFileName
                    self.fileDescription = root[5].text if\
                        len(root) > 5 else ""
                    self.descrEdit.setText(self.fileDescription)
                    self.showTutorial(
                        self.fileDescription,
                        "Descriprion",
                        os.path.dirname(os.path.abspath(self.layoutFileName)))
                    self.setWindowTitle(self.layoutFileName + " - xrtQook")
                    self.writeCodeBox("")
                    self.plotTree.expand(self.rootPlotItem.index())
                    self.plotTree.setColumnWidth(
                        0, int(self.plotTree.width()/3))
                    self.tabs.setCurrentWidget(self.tree)
                    self.statusBar.showMessage(
                        'Loaded layout from {}'.format(
                            os.path.basename(str(self.layoutFileName))), 3000)
                    self.isEmpty = False
                else:
                    self.statusBar.showMessage(ldMsg)

    def iterateImport(self, view, rootModel, rootImport):
        if ET.iselement(rootImport):
            self.ntab += 1
            for childImport in rootImport:
                itemType = str(childImport.attrib['type'])
                itemTag = str(childImport.tag)
                itemText = str(childImport.text)
                child0 = QStandardItem(itemTag)
                if itemType == "flat":
                    child0 = rootModel.appendRow(QStandardItem(itemTag))
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
                                                       QtCore.Qt.blue)
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
                                                QtCore.Qt.blue)
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
                if item.child(ii, 1) is not None:
                    iWidget = view.indexWidget(item.child(ii, 1).index())
                    if iWidget is not None:
                        if iWidget.staticMetaObject.className() == 'QComboBox':
                            if str(iItem.text()) == "targetOpenCL":
                                chItemText = self.OCLModel.item(
                                            iWidget.currentIndex(), 1).text()
                            else:
                                chItemText = iWidget.currentText()
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

    def addCombo(self, view, item):
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
                            combo = self.addStandardCombo(
                                self.beamModel, value)
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
                        elif len(re.findall("material",
                                            paramName.lower())) > 0:
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
                            combo = QComboBox()
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
                            combo = QListWidget()
                            for iray, ray in enumerate(['Good',
                                                        'Out',
                                                        'Over',
                                                        'Alive']):
                                rayItem = QListWidgetItem(str(ray))
                                if len(re.findall(str(iray + 1),
                                                  str(value))) > 0:
                                    rayItem.setCheckState(QtCore.Qt.Checked)
                                else:
                                    rayItem.setCheckState(QtCore.Qt.Unchecked)
                                combo.addItem(rayItem)
                            combo.setMaximumHeight((
                                combo.sizeHintForRow(1) + 1) *
                                    self.rayModel.rowCount())
                            view.setIndexWidget(child1.index(), combo)
                        elif len(re.findall("targetopencl",
                                            paramName.lower())) > 0:
                            combo = QComboBox()
                            combo.setModel(self.OCLModel)
                            oclInd = self.OCLModel.findItems(
                                value, flags=QtCore.Qt.MatchExactly, column=1)
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
        combo = QComboBox()
        combo.setModel(model)
        if combo.findText(value) < 0:
            newItem = QStandardItem(value)
            model.appendRow(newItem)
        combo.setCurrentIndex(combo.findText(value))
        combo.setEditable(True)
        combo.setInsertPolicy(1)
        return combo

    def addStandardCombo(self, model, value):
        combo = QComboBox()
        combo.setModel(model)
        combo.setCurrentIndex(combo.findText(value))
        return combo

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

        menu = QMenu()
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
                for elname in tmodule.__all__:
                    objName = '{0}.{1}'.format(tmodule.__name__, elname)
                    elAction = QAction(self)
                    elAction.setText(elname)
                    elAction.hovered.connect(partial(self.showObjHelp,
                                                     objName))
                    elAction.triggered.connect(
                        partial(self.addElement, elname, objName, None))
                    tsubmenu.addAction(elAction)
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
            for namef, objf in inspect.getmembers(elcls):
                if (inspect.ismethod(objf) or inspect.isfunction(objf)) and\
                        not str(namef).startswith("_"):
                    fdoc = objf.__doc__
                    if fdoc is not None:
                        objfNm = '{0}.{1}'.format(elstr,
                                                  objf.__name__)
                        fdoc = re.findall(r"Returned values:.*", fdoc)
                        if len(fdoc) > 0:
                            methAction = QAction(self)
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

        menu = QMenu()

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

        menu = QMenu()

        matMenu = menu.addMenu(self.tr("Add Material"))
        for mName in rmats.__all__:
            objName = '{0}.{1}'.format(rmats.__name__, mName)
            matAction = QAction(self)
            matAction.setText(mName)
            matAction.hovered.connect(partial(self.showObjHelp, objName))
            matAction.triggered.connect(partial(self.addMaterial, mName,
                                                objName))
            matMenu.addAction(matAction)

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
        except:
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

    def whichCrystal(self, matName):
        material = self.materialsModel.findItems(matName)[0]
        rtype = "None"
        if material.text() != "None":
            for i in range(material.rowCount()):
                if material.child(i, 0).text() == '_object':
                    mstr = str(material.child(i, 1).text())
                    break
            for parent in (inspect.getmro(eval(mstr)))[:-1]:
                pname = str(parent.__name__)
                if len(re.findall("crystal", str.lower(pname))) > 0:
                    rtype = "crystal"
                    break
                elif len(re.findall("multilayer", str.lower(pname))) > 0:
                    rtype = "mlayer"
                    break
        return rtype

    def generateCode(self):
        self.flattenElement(self.tree, self.rootBLItem)
        self.flattenElement(self.matTree, self.rootMatItem)
        self.flattenElement(self.plotTree, self.rootPlotItem)
        self.flattenElement(self.runTree, self.rootRunItem)

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
        codeAlignBL = ""
        codeMain = "\ndef main():\n"
        codeMain += '{0}{1} = build_beamline()\n'.format(myTab, BLName)

        codeFooter = """\n
if __name__ == '__main__':
    main()\n"""
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
                                    paravalue = self.quotize(paravalue)
                                    ieinit += '\n{2}{0}={1},'.format(
                                        paraname, paravalue, myTab)
                codeDeclarations += '{0} = {1})\n\n'.format(
                    matItem.text(), str.rstrip(ieinit, ","))

        for ie in range(self.rootBLItem.rowCount()):
            if self.rootBLItem.child(ie, 0).text() != "properties" and\
                    self.rootBLItem.child(ie, 0).text() != "_object":
                tItem = self.rootBLItem.child(ie, 0)
                ieinit = ""
                ierun = ""
                autoX = False
                autoZ = False
                autoPitch = False
                autoBragg = False
                matType = 'None'
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
                                if 'auto' in str(cCoord[0]):
                                    autoX = True
                                    cCoord[0] = '0.0'
                                if 'auto' in str(cCoord[2]):
                                    autoZ = True
                                    cCoord[2] = '0.0'
                                paravalue = re.sub('\'', '', str(
                                    [self.getVal(c) for c in cCoord]))
                            # if paraname == "yaw" and paravalue == "auto":
                            #    autoYaw = True
                            #    paravalue = '0.0'
                            if paraname == "pitch" and ('auto' in paravalue):
                                autoPitch = True
                                paravalue = '0.0'
                            if paraname == "bragg" and ('auto' in paravalue):
                                autoBragg = True
                                paravalue = '0.0'
                            if paraname == "material":
                                matType = self.whichCrystal(paravalue)
                                matName = paravalue
                            if BLName in paravalue and paraname != 'bl':
                                codeAlignBL += '{0}{1}.{2}.{3}={4}\n'.format(
                                    myTab, BLName, tItem.text(), paraname,
                                    paravalue)
                            if paravalue != str(arg_def) or\
                                    paraname == 'bl':
                                if paraname not in ['bl',
                                                    'material',
                                                    'material2']:
                                    paravalue = self.quotize(paravalue)
                                ieinit += '\n{2}{0}={1},'.format(
                                    paraname, paravalue, myTab*2)
                for ieph in range(tItem.rowCount()):
                    if tItem.child(ieph, 0).text() != '_object' and\
                            tItem.child(ieph, 0).text() != 'properties':
                        pItem = tItem.child(ieph, 0)
                        tmpBeamName = ""
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
                                    if len(re.findall(
                                            "beam", paraname.lower())) > 0:
                                        tmpBeamName = paravalue
                            elif pItem.child(imet, 0).text() == 'output':
                                mItem = pItem.child(imet, 0)
                                paraOutput = ""
                                for iep in range(mItem.rowCount()):
                                    paravalue = mItem.child(iep, 1).text()
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
                        if len(re.findall('sources', elstr)) > 0:
                            codeAlignBL += '{3}{0} = {1}.{2}\n'.format(
                                tmpSourceName,
                                rsources.__name__,
                                'Beam(nrays=2)', myTab)
                            codeAlignBL += '{1}{0}.a[:] = 0\n{1}{0}.b[:] = 1\n{1}{0}.c[:] = 0\n\
{1}{0}.x[:] = 0\n{1}{0}.y[:] = 0\n{1}{0}.z[:] = 0\n{1}{0}.state[:] = 1\n\n'.format(tmpSourceName, myTab) # analysis:ignore
                        else:
                            codeAlignBL += '{2}tmpy = {0}.{1}.center[1]\n'.format(BLName, tItem.text(), myTab) # analysis:ignore
                            if autoX:
                                codeAlignBL += '{1}newx = {0}.x[0] +\\\n{1}{1}{0}.a[0] * (tmpy - {0}.y[0]) /\\\n{1}{1}{0}.b[0]\n'.format(tmpBeamName, myTab) # analysis:ignore
                            else:
                                codeAlignBL += '{2}newx = {0}.{1}.center[0]\n'.format(BLName, tItem.text(), myTab) # analysis:ignore
                            if autoZ:
                                codeAlignBL += '{1}newz = {0}.z[0] +\\\n{1}{1}{0}.c[0] * (tmpy - {0}.y[0]) /\\\n{1}{1}{0}.b[0]\n'.format(tmpBeamName, myTab) # analysis:ignore
                            else:
                                codeAlignBL += '{2}newz = {0}.{1}.center[2]\n'.format(BLName, tItem.text(), myTab) # analysis:ignore
                            codeAlignBL += '{2}{0}.{1}.center = (newx, tmpy, newz)\n'.format( # analysis:ignore
                                BLName, tItem.text(), myTab)
                            codeAlignBL += '{2}print(\"{1}.center:\", {0}.{1}.center)\n\n'.format( # analysis:ignore
                                BLName, tItem.text(), myTab)
                            if autoPitch or autoBragg:
                                if matType != 'None':
                                    codeAlignBL += '{1}braggT = {0}.get_Bragg_angle(energy)\n'.format( # analysis:ignore
                                        matName, myTab)
                                    codeAlignBL += '{2}alphaT = 0 if {0}.{1}.alpha is None else {0}.{1}.alpha\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab)
                                    codeAlignBL += '{0}lauePitch = 0\n'.format( # analysis:ignore
                                        myTab)
                                    codeAlignBL += '{2}print(\"bragg, alpha:\", np.degrees(braggT), np.degrees(alphaT), \"degrees\")\n\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab)
                                if matType == 'crystal':
                                    codeAlignBL += '{1}braggT += -{0}.get_dtheta(energy, alphaT)\n'.format( # analysis:ignore
                                        matName, myTab)
                                    codeAlignBL += '{1}if {0}.geom.startswith(\'Laue\'):\n{1}{1}lauePitch = 0.5 * np.pi\n'.format( # analysis:ignore
                                        matName, myTab)
                                    codeAlignBL += '{2}print(\"braggT:\", np.degrees(braggT), \"degrees\")\n\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab)
                                if matType != 'None':
                                    codeAlignBL += '{1}loBeam = rsources.Beam(copyFrom={0})\n'.format( # analysis:ignore
                                        tmpBeamName, myTab)
                                    codeAlignBL += '{3}raycing.global_to_virgin_local(\n{3}{3}{0},\n{3}{3}{1},\n{3}{3}loBeam,\n{3}{3}center={0}.{2}.center)\n'.format( # analysis:ignore
                                        BLName, tmpBeamName,
                                        tItem.text(), myTab)
                                    codeAlignBL += '{2}raycing.rotate_beam(\n{2}{2}loBeam,\n{2}{2}roll=-({0}.{1}.positionRoll + {0}.{1}.roll),\n{2}{2}yaw=-{0}.{1}.yaw,\n{2}{2}pitch=0)\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab)
                                    codeAlignBL += '{0}theta0 = np.arctan2(-loBeam.c[0], loBeam.b[0])\n'.format(myTab) # analysis:ignore
                                    codeAlignBL += '{0}th2pitch = np.sqrt(1. - loBeam.a[0]**2)\n'.format(myTab) # analysis:ignore
                                    codeAlignBL += '{0}targetPitch = np.arcsin(np.sin(braggT) / th2pitch) -\\\n{0}{0}theta0\n'.format(myTab) # analysis:ignore
                                    codeAlignBL += '{0}targetPitch += alphaT + lauePitch\n'.format(myTab) # analysis:ignore
                                    if autoBragg:
                                        strPitch = 'bragg'
                                        addPitch = '-{0}.{1}.pitch'.format(
                                            BLName, tItem.text())
                                    else:
                                        strPitch = 'pitch'
                                        addPitch = ''
                                    codeAlignBL += '{2}{0}.{1}.{3} = targetPitch{4}\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab,
                                        strPitch, addPitch)
                                    codeAlignBL += '{2}print(\"{1}.{3}:\", np.degrees({0}.{1}.{3}), \"degrees\")\n\n'.format( # analysis:ignore
                                        BLName, tItem.text(), myTab, strPitch)
                            codeAlignBL += '{5}{0} = {1}.{2}.{3}({4})\n'.format( # analysis:ignore
                                paraOutput.rstrip(', '),
                                BLName, tItem.text(),
                                str(pItem.text()).strip('()'),
                                ierun.rstrip(','), myTab)
                codeBuildBeamline += '{3}{0}.{1} = {2})\n\n'.format(
                    BLName, str(tItem.text()), ieinit.rstrip(','), myTab)
        codeBuildBeamline += "{0}return {1}\n\n".format(myTab, BLName)
        codeRunProcess += r"{0}outDict = ".format(myTab) + "{"
        codeAlignBL = 'def align_beamline({0}, energy):\n'.format(BLName) +\
            codeAlignBL + "{}\n".format(
                myTab + "pass" if codeAlignBL == '' else '')

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
        codeMain += '{1}align_beamline({0}, E0)\n'.format(BLName, myTab)
        codeMain += '{0}{1} = define_plots()\n'.format(
            myTab, self.rootPlotItem.text())
        codePlots = '\ndef define_plots():\n{0}{1} = []\n'.format(
            myTab, self.rootPlotItem.text())

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
                        paravalue = tItem.child(iep, 1).text()
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

        for ie in range(self.rootRunItem.rowCount()):
            if self.rootRunItem.child(ie, 0).text() == '_object':
                elstr = str(self.rootRunItem.child(ie, 1).text())
                codeMain += "{0}{1}(\n".format(myTab, elstr)
                objrr = eval(elstr)
                break

        ieinit = ""
        for iem, argVal in zip(range(self.rootRunItem.rowCount()-1),
                               inspect.getargspec(objrr)[3]):
            ie = iem + 1
            if self.rootRunItem.child(ie, 0).text() != '_object':
                paraname = self.rootRunItem.child(ie, 0).text()
                paravalue = self.rootRunItem.child(ie, 1).text()
                if paraname == "plots":
                    paravalue = self.rootPlotItem.text()
                if paraname == "backend":
                    paravalue = 'r\"{0}\"'.format(paravalue)
                if str(paravalue) != str(argVal):
                    ieinit += "{0}{1}={2},\n".format(
                        myTab*2, paraname, paravalue)

        codeMain += ieinit.rstrip(",\n") + ")\n"

        fullCode = codeDeclarations + codeBuildBeamline +\
            codeRunProcess + codeAlignBL + codePlots + codeMain + codeFooter
        for xrtAlias in self.xrtModules:
            fullModName = (eval(xrtAlias)).__name__
            fullCode = fullCode.replace(fullModName, xrtAlias)
            codeHeader += 'import {0} as {1}\n'.format(fullModName, xrtAlias)
        fullCode = codeHeader + fullCode
        if isSpyderlib:
            self.codeEdit.set_text(fullCode)
        else:
            self.codeEdit.setText(fullCode)
        self.tabs.setCurrentWidget(self.codeEdit)
        self.statusBar.showMessage('Python code successfully generated', 5000)

    def saveCode(self):
        saveStatus = False
        if self.saveFileName == "":
            saveDialog = QFileDialog()
            saveDialog.setFileMode(QFileDialog.AnyFile)
            saveDialog.setAcceptMode(QFileDialog.AcceptSave)
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
                if isSpyderConsole:
                    self.codeConsole.wdir = os.path.dirname(
                        str(self.saveFileName))
                else:
                    self.qprocess.setWorkingDirectory(
                        os.path.dirname(str(self.saveFileName)))
            except (OSError, IOError) as errStr:
                saveMsg = str(errStr)
            self.statusBar.showMessage(saveMsg, 5000)
        return saveStatus

    def saveCodeAs(self):
        tmpName = self.saveFileName
        self.saveFileName = ""
        if not self.saveCode():
            self.statusBar.showMessage(
                'Failed saving code to {}'.format(
                    os.path.basename(str(self.saveFileName))), 5000)
            self.saveFileName = tmpName

    def execCode(self):
        self.saveCode()
        self.tabs.setCurrentWidget(self.codeConsole)
        if isSpyderConsole:
            self.codeConsole.fname = str(self.saveFileName)
            self.codeConsole.create_process()
        else:
            self.codeConsole.clear()
            self.codeConsole.append('Starting {}\n\n'.format(
                    os.path.basename(str(self.saveFileName))))
            self.codeConsole.append('Press Ctrl+X to terminate process\n\n')
            self.qprocess.start("python", ['-u', str(self.saveFileName)])

    def aboutCode(self):
        import platform
        if use_pyside:
            Qt_version = QtCore.__version__
            PyQt_version = PySide.__version__
        else:
            Qt_version = QtCore.QT_VERSION_STR
            PyQt_version = QtCore.PYQT_VERSION_STR
        msgBox = QMessageBox()
        msgBox.setWindowIcon(QIcon(
            os.path.join(self.xrtQookDir, '_icons', 'xrQt1.ico')))
        msgBox.setWindowTitle("About xrtQook")
        msgBox.setIconPixmap(QPixmap(
            os.path.join(self.xrtQookDir, '_icons', 'logo-xrtQt.png')))
        msgBox.setTextFormat(QtCore.Qt.RichText)
        msgBox.setText("Beamline layout manipulation and automated code\
 generation tool for the <a href='http://pythonhosted.org/xrt'>xrt ray tracing\
 package</a>.\nFor a quick start see this short \
 <a href='http://pythonhosted.org/xrt/qook.html'>tutorial</a>.")
        infText = """Created by:\
\nRoman Chernikov (DESY Photon Science)\
\nKonstantin Klementiev (MAX IV Laboratory)\
\nLicensed under the terms of the MIT License\nMarch 2016\
\n\nYour system:\n{0}\nPython {1}\nQt {2}\n{3} {4}""".format(
                platform.platform(terse=True), platform.python_version(),
                Qt_version, QtName, PyQt_version)
        infText += '\npyopencl {}'.format(
            cl.VERSION if isOpenCL else 'not found')
        infText += '\nxrt {0} in {1}'.format(
            xrt.__version__, path_to_xrt)
        msgBox.setInformativeText(infText)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = XrtQook()
    ex.setWindowTitle("xrtQook")
    ex.show()
    sys.exit(app.exec_())

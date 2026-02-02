# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:03:13 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import os  # analysis:ignore
import re  # analysis:ignore
import numpy as np  # analysis:ignore
from functools import partial  # analysis:ignore
import matplotlib as mpl  # analysis:ignore
from matplotlib.colors import hsv_to_rgb  # analysis:ignore
from matplotlib.figure import Figure  # analysis:ignore
from matplotlib.widgets import RectangleSelector  # analysis:ignore

from .._constants import _DEBUG_  # analysis:ignore
from .._utils import is_source, is_oe, is_aperture, is_screen, basis_rotation_q  # analysis:ignore
from .inspector import InstanceInspector  # analysis:ignore
from .opengl import xrtGlWidget  # analysis:ignore

from ...commons import qt  # analysis:ignore

from ....backends import raycing  # analysis:ignore
from ....backends.raycing import sources as rsources  # analysis:ignore
from ....plotter import colorFactor, colorSaturation  # analysis:ignore


class xrtGlow(qt.QWidget):
    def __init__(self, arrayOfRays=None, parent=None, progressSignal=None,
                 layout=None, epicsPrefix=None):
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
                         'epicsPrefix': epicsPrefix, 'signal': progressSignal}

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
        self.customGlWidget.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customGlWidget.customContextMenuRequested.connect(self.glMenu)
        self.customGlWidget.openElViewer.connect(self.runElementViewer)

        self.makeNavigationPanel()
        self.makeTransformationPanel()
        self.makeColorsPanel()
        self.makeGridAndProjectionsPanel()
        self.makeScenePanel()

        mainLayout = qt.QHBoxLayout()
        sideLayout = qt.QVBoxLayout()

        tabs = qt.QTabWidget()
        tabs.addTab(self.navigationPanel, "Navigation")
        tabs.addTab(self.transformationPanel, "Transformations")
        tabs.addTab(self.colorOpacityPanel, "Colors")
        tabs.addTab(self.projectionPanel, "Grid/Projections")
        tabs.addTab(self.scenePanel, "Scene")
#        tabs.setTabPosition(qt.QTabWidget.West)
        sideLayout.addWidget(tabs)
        self.canvasSplitter = qt.QSplitter()
        self.canvasSplitter.setChildrenCollapsible(False)
        self.canvasSplitter.setOrientation(qt.Qt.Horizontal)
        mainLayout.addWidget(self.canvasSplitter)
        sideWidget = qt.QWidget()
        sideWidget.setLayout(sideLayout)
        self.canvasSplitter.addWidget(self.customGlWidget)
        self.canvasSplitter.addWidget(sideWidget)

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

    def closeEvent(self, event):
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

            elViewer = InstanceInspector(
                    self, oeProps,
                    initDict=oeInitProps,
                    epicsDict=getattr(self.customGlWidget,
                                      'epicsInterface', None),
                    viewOnly=False,
                    beamLine=self.customGlWidget.beamline,
                    categoriesDict=catDict)

            self.customGlWidget.beamUpdated.connect(elViewer.update_beam)
            self.customGlWidget.oePropsUpdated.connect(elViewer.update_param)
            # TODO: update tree
            elViewer.propertiesChanged.connect(
                    partial(self.customGlWidget.update_beamline, oeuuid,
                            sender='OEE'))
    #        if (elViewer.exec_()):
            if (elViewer.show()):
                pass

    def makeNavigationPanel(self):
        self.navigationLayout = qt.QVBoxLayout()

        centerCBLabel = qt.QLabel('Center view at:')
        self.centerCB = qt.QComboBox()
        self.centerCB.setMaxVisibleItems(48)
        self.centerCB.setSizeAdjustPolicy(qt.QComboBox.AdjustToContents)
        proxy_model = qt.ComboBoxFilterProxyModel()
        proxy_model.setSourceModel(self.segmentsModel)
        self.centerCB.setModel(proxy_model)
        self.centerCB.setModelColumn(0)
        self.centerCB.currentIndexChanged['int'].connect(
                lambda elementid: self.centerEl(
                        self.centerCB.itemData(elementid,
                                               role=qt.Qt.UserRole)))
        self.centerCB.setCurrentIndex(0)

        layout = qt.QHBoxLayout()
        layout.addWidget(centerCBLabel)
        layout.addWidget(self.centerCB)
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
        self.paletteWidget.setSizePolicy(qt.QSizePolicy.Maximum,
                                         qt.QSizePolicy.Maximum)
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

        glNormCB = qt.QCheckBox('Global Normalization')
        glNormCB.setChecked(True)
        glNormCB.stateChanged.connect(self.checkGNorm)
        colorLayout.addWidget(glNormCB)
        self.glNormCB = glNormCB

        iHSVCB = qt.QCheckBox('Intensity as HSV Value')
        iHSVCB.setChecked(False)
        iHSVCB.stateChanged.connect(self.checkHSV)
        colorLayout.addWidget(iHSVCB)
        self.iHSVCB = iHSVCB

        self.colorPanel.setLayout(colorLayout)

        self.colorOpacityPanel = qt.QWidget(self)
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
        self.gridPanel.toggled.connect(self.checkDrawGrid)

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
        checkBox.stateChanged.connect(self.checkFineGrid)
        xyzGridLayout.addWidget(checkBox)
        self.checkBoxFineGrid = checkBox
        self.gridControls = []

        projectionLayout = qt.QVBoxLayout()
        checkBox = qt.QCheckBox('Perspective')
        checkBox.setChecked(True)
        checkBox.stateChanged.connect(self.checkPerspect)
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

        self.sceneControls = []
        for iCB, (cbText, cbFunc) in enumerate(zip(
                ['Enable antialiasing',
                 'Use global colors',
                 'Depth test for Lines',
                 'Depth test for Points',
                 'Invert scene color',
                 'Use scalable font',
                 'Show Virtual Screen label',
                 'OE size match beam',
                 'Show lost rays',
                 'Show local axes'],
                [self.checkAA,
                 self.checkGlobalColors,
                 self.checkLineDepthTest,
                 self.checkPointDepthTest,
                 self.invertSceneColor,
                 self.checkScalableFont,
                 self.checkShowLabels,
                 self.checkOEAutoSize,
                 self.checkShowLost,
                 self.checkShowLocalAxes])):
            aaCheckBox = qt.QCheckBox(cbText)
            aaCheckBox.setChecked(iCB in [1])
            aaCheckBox.stateChanged.connect(cbFunc)
            self.sceneControls.append(aaCheckBox)
            sceneLayout.addWidget(aaCheckBox)

        for it, (what, tt, defv) in enumerate(zip(
                ['Default OE thickness, mm',
                 'Force OE thickness, mm',
                 'Aperture frame size, %',
                 'Scene limit, mm'],
                ['For OEs that do not have thickness',
                 'For OEs that have thickness, e.g. plates or lenses',
                 '', ''],
                [self.customGlWidget.oeThickness,
                 self.customGlWidget.oeThicknessForce,
                 self.customGlWidget.slitThicknessFraction,
                 self.customGlWidget.maxLen])):
            tLabel = qt.QLabel(what)
            tLabel.setToolTip(tt)
            tEdit = qt.QLineEdit()
            tEdit.setToolTip(tt)
            if defv is not None:
                tEdit.setText(str(defv))
            tEdit.editingFinished.connect(
                partial(self.updateThicknessFromQLE, tEdit, it))
            if it == 3:
                self.maxLenEditor = tEdit

            layout = qt.QHBoxLayout()
            tLabel.setMinimumWidth(145)
            layout.addWidget(tLabel)
            tEdit.setMaximumWidth(96)
            layout.addWidget(tEdit)
            layout.addStretch()
            sceneLayout.addLayout(layout)

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

            layout = qt.QHBoxLayout()
            axLabel.setMinimumWidth(100)
            layout.addWidget(axLabel)
            axEdit.setMaximumWidth(48)
            layout.addWidget(axEdit)
            layout.addStretch()
            sceneLayout.addLayout(layout)

        self.scenePanel = qt.QWidget(self)
        sceneLayout.addStretch()
        self.scenePanel.setLayout(sceneLayout)

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

    def updateSegmentsModel(self, arrayOfRays):
        def copyRow(item, row):
            newRow = []
            for iCol in range(4):
                oldItem = item.child(row, iCol)
                newItem = qt.QStandardItem(str(oldItem.text()))
                newItem.setCheckable(oldItem.isCheckable())
                if newItem.isCheckable():
                    newItem.setCheckState(oldItem.checkState())
                newItem.setEditable(oldItem.isEditable())
                newRow.append(newItem)
            return newRow

        newSegmentsModel = self.initSegmentsModel(isNewModel=False)
        newSegmentsModel.invisibleRootItem().appendRow(
            copyRow(self.segmentsModelRoot, 0))
        for element, elRecord in self.oesList.items():
            for iel in range(self.segmentsModelRoot.rowCount()):
                elItem = self.segmentsModelRoot.child(iel, 0)
                elName = str(elItem.text())
                if str(element) == elName:
                    elRow = copyRow(self.segmentsModelRoot, iel)
                    for segment in arrayOfRays[0]:
                        if segment[3] is not None:
                            endBeamText = "to {}".format(
                                self.beamsToElements[segment[3]])
                            if str(segment[1]) == str(elRecord[1]):
                                if elItem.hasChildren():
                                    for ich in range(elItem.rowCount()):
                                        if str(elItem.child(ich, 0).text()) ==\
                                                endBeamText:
                                            elRow[0].appendRow(
                                                copyRow(elItem, ich))
                                            break
                                    else:
                                        elRow[0].appendRow(self.createRow(
                                            endBeamText, 3))
                                else:
                                    elRow[0].appendRow(self.createRow(
                                        endBeamText, 3))
                    newSegmentsModel.invisibleRootItem().appendRow(elRow)
                    break
            else:
                elRow = self.createRow(str(element), 1)
                for segment in arrayOfRays[0]:
                    if str(segment[1]) == str(elRecord[1]) and\
                            segment[3] is not None:
                        endBeamText = "to {}".format(
                            self.beamsToElements[segment[3]])
                        elRow[0].appendRow(self.createRow(endBeamText, 3))
                newSegmentsModel.invisibleRootItem().appendRow(elRow)
        self.segmentsModel = newSegmentsModel
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()
        self.oeTree.setModel(self.segmentsModel)

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

#            for eluuid, operations in self.customGlWidget.beamline.flowU.items():
#                element = self.customGlWidget.beamline.oesDict[eluuid][0]
#                elname = element.name
#                newRow = self.createRow(elname, 1, uuid=eluuid,
#                                        icon=self.getIcon(element))
#                for targetuuid, targetoperations in self.customGlWidget.beamline.flowU.items():
#                    for kwargset in targetoperations.values():
#                        if kwargset.get('beam', 'none') == eluuid:
#                            try:
#                                targetName = self.customGlWidget.beamline.oesDict[targetuuid][0].name
#                                endBeamText = "to {}".format(targetName)
#                                newRow[0].appendRow(self.createRow(
#                                        endBeamText, 3, uuid=targetuuid))
#                            except:  # analysis:ignore
#                                continue

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
                    except:  # analysis:ignore
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

    def updateTargets(self):

        tmpDict = dict()
        self.centerCB.blockSignals(True)
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
        self.centerCB.blockSignals(False)

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
            except:  # analysis:ignore
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

    def checkGNorm(self, state):
        self.customGlWidget.globalNorm = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def checkHSV(self, state):
        self.customGlWidget.iHSV = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def checkDrawGrid(self, state):
        self.customGlWidget.drawGrid = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkFineGrid(self, state):
        self.customGlWidget.fineGridEnabled = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkPerspect(self, state):
        self.customGlWidget.perspectiveEnabled = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkAA(self, state):
        self.customGlWidget.enableAA = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkGlobalColors(self, state):
        self.customGlWidget.globalColors = True if state > 0 else False
        self.customGlWidget.newColorAxis = True
        self.customGlWidget.change_beam_colorax()
#        self.customGlWidget.enableBlending = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkLineDepthTest(self, state):
        self.customGlWidget.linesDepthTest = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkPointDepthTest(self, state):
        self.customGlWidget.pointsDepthTest = True if state > 0 else False
        self.customGlWidget.glDraw()

    def invertSceneColor(self, state):
        self.customGlWidget.invertColors = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkScalableFont(self, state):
        self.customGlWidget.useScalableFont = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkShowLabels(self, state):
        self.customGlWidget.showOeLabels = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkOEAutoSize(self, state):
        self.customGlWidget.autoSizeOe = True if state > 0 else False
        self.customGlWidget.needMeshUpdate.extend(
                list(self.customGlWidget.beamline.oesDict.keys()))
        self.customGlWidget.glDraw()

    def checkShowLost(self, state):
        self.customGlWidget.showLostRays = True if state > 0 else False
        self.customGlWidget.glDraw()

    def checkShowLocalAxes(self, state):
        self.customGlWidget.showLocalAxes = True if state > 0 else False
        self.customGlWidget.glDraw()

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
            self.customGlWidget.getColor = getattr(
                raycing, 'get_{}'.format(selAxis))
            self.customGlWidget.newColorAxis = True
        oldColorMin = self.customGlWidget.colorMin
        oldColorMax = self.customGlWidget.colorMax
        self.customGlWidget.change_beam_colorax()
#        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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
        except:  # analysis:ignore
            pass

    def updateColorSel(self, slider, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
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
        except:  # analysis:ignore
            pass

    def updateColorSelFromQLE(self, editor, icSel):
        try:
            # editor = self.sender()
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
        except:  # analysis:ignore
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
            except:  # analysis:ignore
                pass
        editor.setText("{0:.2f}".format(position))
        self.customGlWidget.rotations[iax] = np.float32(position)
        self.customGlWidget.updateQuats()
        self.customGlWidget.glDraw()

    def updateRotationFromGL(self, actPos):
        for iaxis, (slider, editor) in\
                enumerate(zip(self.rotationSliders, self.rotationEditors)):
            value = actPos[iaxis]
            slider.setValue(value)
            editor.setText("{0:.2f}".format(value))

    def updateRotationFromQLE(self, editor, slider):
        # editor = self.sender()
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateScale(self, slider, iax, editor, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                pass
        editor.setText("{0:.2f}".format(position))
        self.customGlWidget.scaleVec[iax] = np.float32(np.power(10, position))
        try:
            self.customGlWidget.cBox.update_grid()
        except AttributeError:
            pass
        self.customGlWidget.glDraw()

    def updateScaleFromGL(self, scale):
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        for iaxis, (slider, editor) in \
                enumerate(zip(self.zoomSliders, self.zoomEditors)):
            value = np.log10(scale[iaxis])
            slider.setValue(value)
            editor.setText("{0:.2f}".format(value))

    def updateScaleFromQLE(self, editor, slider):
        # editor = self.sender()
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateMaxLenFromGL(self, value):
        self.maxLenEditor.setText("{0:.2f}".format(float(value)))
#        if isinstance(scale, (int, float)):
#            scale = [scale, scale, scale]
#        for iaxis, (slider, editor) in \
#                enumerate(zip(self.zoomSliders, self.zoomEditors)):
#            value = np.log10(scale[iaxis])
#            slider.setValue(value)
#            editor.setText("{0:.2f}".format(value))

    def updateFontSize(self, slider, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
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
            menu.addAction('to Beam Local',
                           partial(self.toBeamLocal,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('restore Global',
                           partial(self.toGlobal,
                                   str(selectedItem.data(qt.Qt.UserRole))))
            menu.addAction('View Properties',
                           partial(self.runElementViewer,
                                   str(selectedItem.data(qt.Qt.UserRole))))

            menu.exec_(self.oeTree.viewport().mapToGlobal(position))
        else:
            pass

    def updateGrid(self, slider, iax, editor, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                pass
        editor.setText("{0:.2f}".format(position))
        if position != 0:
            self.customGlWidget.aPos[iax] = np.float32(position)
            self.customGlWidget.cBox.update_grid()
            self.customGlWidget.glDraw()

    def updateGridFromQLE(self, editor, slider):
        # editor = self.sender()
        value = float(re.sub(',', '.', str(editor.text())))
        slider.setValue(value)

    def updateGridFromGL(self, aPos):
        for iaxis, (slider, editor) in\
                enumerate(zip(self.gridSliders, self.gridEditors)):
            value = aPos[iaxis]
            slider.setValue(value)
            editor.setText("{0:.2f}".format(value))

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
        for iAction, actCnt in enumerate(self.sceneControls):
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
        for iAction, actCnt in enumerate(self.sceneControls):
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
            menu.addAction('Transform to {} Beam Local'.format(oeName),
                           partial(self.toBeamLocal, oeuuid))
            menu.addAction('Restore Global at {}.center'.format(oeName),
                           partial(self.toGlobal, oeuuid))
            menu.addAction('View Properties',
                           partial(self.runElementViewer, oeuuid))
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
#        saveDialog.selectFile("oe_name.stl")
        saveDialog.setNameFilter("STL files (*.stl)")
        saveDialog.selectNameFilter("STL files (*.stl)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            mesh = self.customGlWidget.meshDict.get(oeid)
            if mesh is not None:
                mesh.export_to_stl(filename)

    def saveSceneDialog(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("Numpy files (*.npy)")  # analysis:ignore
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
        for param in ['aspect', 'cameraAngle', 'projectionsVisibility',
                      'lineOpacity', 'lineWidth', 'pointOpacity', 'pointSize',
                      'lineProjectionOpacity', 'lineProjectionWidth',
                      'pointProjectionOpacity', 'pointProjectionSize',
                      'coordOffset', 'cutoffI', 'drawGrid', 'aPos', 'scaleVec',
                      'tVec', 'cameraPos', 'rotations',
                      'visibleAxes', 'signs', 'selColorMin', 'selColorMax',
                      'colorMin', 'colorMax', 'fineGridEnabled',
                      'useScalableFont', 'invertColors', 'perspectiveEnabled',
                      'globalNorm', 'viewPortGL', 'iHSV']:
            params[param] = getattr(self.customGlWidget, param)
        params['size'] = self.geometry()
        params['sizeGL'] = self.canvasSplitter.sizes()
        params['colorAxis'] = str(self.colorControls[0].currentText())
        try:
            np.save(filename, params)
        except:  # analysis:ignore
            print('Error saving file')
            return
        print('Saved scene to {}'.format(filename))

    def loadScene(self, filename):
        try:
            params = np.load(filename, allow_pickle=True).item()
        except:  # analysis:ignore
            print('Error loading file')
            return

        for param in ['aspect', 'cameraAngle', 'projectionsVisibility',
                      'lineOpacity', 'lineWidth', 'pointOpacity', 'pointSize',
                      'lineProjectionOpacity', 'lineProjectionWidth',
                      'pointProjectionOpacity', 'pointProjectionSize',
                      'coordOffset', 'cutoffI', 'drawGrid', 'aPos', 'scaleVec',
                      'tVec', 'cameraPos', 'rotations',
                      'visibleAxes', 'signs', 'selColorMin', 'selColorMax',
                      'colorMin', 'colorMax', 'fineGridEnabled',
                      'useScalableFont', 'invertColors', 'perspectiveEnabled',
                      'globalNorm', 'viewPortGL', 'iHSV']:
            setattr(self.customGlWidget, param, params[param])
        self.setGeometry(params['size'])
        self.canvasSplitter.setSizes(params['sizeGL'])
        self.updateScaleFromGL(self.customGlWidget.scaleVec)
        self.blockSignals(True)
        self.updateRotationFromGL(self.customGlWidget.rotations)
        self.updateOpacityFromGL([self.customGlWidget.lineOpacity,
                                  self.customGlWidget.lineWidth,
                                  self.customGlWidget.pointOpacity,
                                  self.customGlWidget.pointSize])
        for iax, checkBox in enumerate(self.projectionControls):
            checkBox.setChecked(self.customGlWidget.projectionsVisibility[iax])
        self.gridPanel.setChecked(self.customGlWidget.drawGrid)
        self.checkBoxFineGrid.setChecked(self.customGlWidget.fineGridEnabled)
        self.checkBoxPerspective.setChecked(
            self.customGlWidget.perspectiveEnabled)
        self.updateProjectionOpacityFromGL(
            [self.customGlWidget.lineProjectionOpacity,
             self.customGlWidget.lineProjectionWidth,
             self.customGlWidget.pointProjectionOpacity,
             self.customGlWidget.pointProjectionSize])
        self.updateGridFromGL(self.customGlWidget.aPos)
        self.sceneControls[4].setChecked(self.customGlWidget.invertColors)
        self.sceneControls[5].setChecked(self.customGlWidget.useScalableFont)
        self.sceneControls[4].setChecked(self.customGlWidget.invertColors)
        self.glNormCB.setChecked(self.customGlWidget.globalNorm)
        self.iHSVCB.setChecked(self.customGlWidget.iHSV)

        self.blockSignals(False)
        self.mplFig.canvas.draw()
        colorCB = self.colorControls[0]
        colorCB.setCurrentIndex(colorCB.findText(params['colorAxis']))
        newExtents = list(self.paletteWidget.span.extents)
        newExtents[0] = params['selColorMin']
        newExtents[1] = params['selColorMax']

        try:
            self.paletteWidget.span.extents = newExtents
        except:  # analysis:ignore
            pass
        self.updateColorSelFromMPL(0, 0)

        print('Loaded scene from {}'.format(filename))

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
        off0 = np.array(oeLine[0].center) - np.array(
            self.customGlWidget.tmpOffset)  # TODO: may fail on raw 'auto'
        cOffset = qt.QVector4D(off0[0], off0[1], off0[2], 0)
        off1 = self.customGlWidget.mModLocal * cOffset
        self.customGlWidget.coordOffset = np.array(
            [off1.x(), off1.y(), off1.z()])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        if hasattr(self.customGlWidget, 'cBox'):
            self.customGlWidget.cBox.update_grid()
        self.customGlWidget.glDraw()

    def toLocal(self, oeuuid):
        oe = self.customGlWidget.beamline.oesDict[oeuuid][0]
        self.customGlWidget.mModLocal =\
            self.customGlWidget.meshDict[oeuuid].transMatrix[0].inverted()[0]
        self.customGlWidget.coordOffset = np.float32([0, 0, 0])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.tmpOffset = oe.center
        self.customGlWidget.cBox.update_grid()
        self.customGlWidget.glDraw()

    def toGlobal(self, oeuuid):
        self.customGlWidget.mModLocal = qt.QMatrix4x4()
        self.customGlWidget.tmpOffset = np.float32([0, 0, 0])
        self.customGlWidget.coordOffset = list(
                self.customGlWidget.beamline.oesDict[oeuuid][0].center)
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.cBox.update_grid()
        self.customGlWidget.glDraw()

    def toBeamLocal(self, oeuuid):
        beamDict = self.customGlWidget.beamline.beamsDictU[oeuuid]
        oe = self.customGlWidget.beamline.oesDict[oeuuid][0]
        off0 = oe.center if isinstance(oe.center, list) else oe.center.tolist()
        mTranslation = qt.QMatrix4x4()
        mTranslation.translate(*off0)
        self.customGlWidget.coordOffset = np.float32([0, 0, 0])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        if 'beamLocal1' in beamDict:
            beam = beamDict['beamLocal1']
        elif 'beamLocal' in beamDict:
            beam = beamDict['beamLocal']
        else:
            beam = beamDict['beamGlobal']

        if hasattr(beam, 'basis'):
            rotationQ = basis_rotation_q(np.identity(3), beam.basis.T)
            mRotation = qt.QMatrix4x4()
            mRotation.rotate(qt.QQuaternion(*rotationQ))
            posMatrix = mTranslation*mRotation
            self.customGlWidget.mModLocal = posMatrix.inverted()[0]
        self.customGlWidget.cBox.update_grid()
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

    def updateThicknessFromQLE(self, editor, ia):
        # editor = self.sender()
        value = float(str(editor.text())) if editor.text() else None
        if ia == 0:
            self.customGlWidget.oeThickness = value
        elif ia == 1:
            self.customGlWidget.oeThicknessForce = value
        elif ia == 2:
            self.customGlWidget.slitThicknessFraction = value
        elif ia == 3:
            self.customGlWidget.maxLen = value
        else:
            return
        for oeid, oeLine in self.customGlWidget.beamline.oesDict.items():
            if is_oe(oeLine[0]) and\
                    oeid not in self.customGlWidget.needMeshUpdate:
                self.customGlWidget.needMeshUpdate.append(oeid)
        self.customGlWidget.glDraw()

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

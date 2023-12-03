# -*- coding: utf-8 -*-
u"""
xrtGlow -- an interactive 3D beamline viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The beamline created in xrtQook or in a python script can be interactively
viewed in an OpenGL based widget xrtGlow. It visualizes beams, footprints,
surfaces, apertures and screens. The brightness represents intensity and the
color represents an auxiliary user-selected distribution, typically energy.
A virtual screen can be put at any position and dragged by mouse with
simultaneous observation of the beam distribution on it. See two example
screenshots below (click to expand and read the captions).

The primary purpose of xrtGlow is to demonstrate the alignment correctness
given the fact that xrtQook can automatically calculate several positional and
angular parameters.

See aslo :ref:`Notes on using xrtGlow <glow_notes>`.

+-------------+-------------+
|   |glow1|   |   |glow2|   |
+-------------+-------------+

.. |glow1| imagezoom:: _images/xrtGlow1.png
   :alt: &ensp;A view of xrtQook with embedded xrtGlow. Visible is a virtual
       screen draggable by mouse, a curved mirror surface with a footprint on
       it and the color (energy) distribution on the virtual screen. The scale
       along the beamline is compressed by a factor of 100.

.. |glow2| imagezoom:: _images/xrtGlow2.png
   :loc: upper-right-corner
   :alt: &ensp; xrtGlow with three double-paraboloid lenses. The scaling on
       this image is isotropic. The source (on the left) is a parallel
       geometric source. The coloring is by axial divergence (red=0), showing
       the effect of refractive focusing.

"""
from __future__ import print_function
__author__ = "Roman Chernikov, Konstantin Klementiev"

# import sys
import os
import numpy as np
from functools import partial
import matplotlib as mpl
# import inspect
import re
import copy
# import time

from collections import OrderedDict

from ...backends import raycing
from ...backends.raycing import sources as rsources
from ...backends.raycing import screens as rscreens
from ...backends.raycing import oes as roes
from ...backends.raycing import apertures as rapertures
from ...backends.raycing import materials as rmats
from ..commons import qt
from ..commons import gl
from ...plotter import colorFactor, colorSaturation
_DEBUG_ = False  # If False, exceptions inside the module are ignored


class xrtGlow(qt.QWidget):
    def __init__(self, arrayOfRays, parent=None, progressSignal=None):
        super(xrtGlow, self).__init__()
        self.parentRef = parent
        self.cAxisLabelSize = 10
        mplFont = {'size': self.cAxisLabelSize}
        mpl.rc('font', **mplFont)
        self.setWindowTitle('xrtGlow')
        iconsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '_icons')
        self.setWindowIcon(qt.QIcon(os.path.join(iconsDir, 'icon-GLow.ico')))
        self.populateOEsList(arrayOfRays)

        self.segmentsModel = self.initSegmentsModel()
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()

        self.populateSegmentsModel(arrayOfRays)

        self.fluxDataModel = qt.QStandardItemModel()

        for colorField in raycing.allBeamFields:
            self.fluxDataModel.appendRow(qt.QStandardItem(colorField))

        self.customGlWidget = xrtGlWidget(self, arrayOfRays,
                                          self.segmentsModelRoot,
                                          self.oesList,
                                          self.beamsToElements,
                                          progressSignal)
        self.customGlWidget.rotationUpdated.connect(self.updateRotationFromGL)
        self.customGlWidget.scaleUpdated.connect(self.updateScaleFromGL)
        self.customGlWidget.histogramUpdated.connect(self.updateColorMap)
        self.customGlWidget.setContextMenuPolicy(qt.CustomContextMenu)
        self.customGlWidget.customContextMenuRequested.connect(self.glMenu)

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
        sideLayout.addWidget(tabs)
        self.canvasSplitter = qt.QSplitter()
        self.canvasSplitter.setChildrenCollapsible(False)
        self.canvasSplitter.setOrientation(qt.Horizontal)
        mainLayout.addWidget(self.canvasSplitter)
        sideWidget = qt.QWidget()
        sideWidget.setLayout(sideLayout)
        self.canvasSplitter.addWidget(self.customGlWidget)
        self.canvasSplitter.addWidget(sideWidget)

        self.setLayout(mainLayout)
        self.customGlWidget.oesList = self.oesList
        toggleHelp = qt.QShortcut(self)
        toggleHelp.setKey("F1")
        toggleHelp.activated.connect(self.openHelpDialog)
#        toggleHelp.activated.connect(self.customGlWidget.toggleHelp)
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
        self.dockToQook = qt.QShortcut(self)
        self.dockToQook.setKey("F4")
        self.dockToQook.activated.connect(self.toggleDock)
        tiltScreen = qt.QShortcut(self)
        tiltScreen.setKey("Ctrl+T")
        tiltScreen.activated.connect(self.customGlWidget.switchVScreenTilt)

    def closeEvent(self, event):
        if self.parentRef is not None:
            try:
                parentAlive = self.parentRef.isVisible()
                if parentAlive:
                    event.ignore()
                    self.setVisible(False)
                else:
                    event.accept()
            except:
                event.accept()
        else:
            event.accept()

    def makeNavigationPanel(self):
        self.navigationLayout = qt.QVBoxLayout()

        centerCBLabel = qt.QLabel('Center view at:')
        self.centerCB = qt.QComboBox()
        self.centerCB.setMaxVisibleItems(48)
        for key in self.oesList.keys():
            self.centerCB.addItem(str(key))
#        centerCB.addItem('customXYZ')
        self.centerCB.currentIndexChanged['QString'].connect(self.centerEl)
        self.centerCB.setCurrentIndex(0)

        layout = qt.QHBoxLayout()
        layout.addWidget(centerCBLabel)
        layout.addWidget(self.centerCB)
        layout.addStretch()
        self.navigationLayout.addLayout(layout)
        self.oeTree = qt.QTreeView()
        self.oeTree.setModel(self.segmentsModel)
        self.oeTree.setContextMenuPolicy(qt.CustomContextMenu)
        self.oeTree.customContextMenuRequested.connect(self.oeTreeMenu)
        self.oeTree.resizeColumnToContents(0)
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
            axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)
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
#        rotModeCB = qt.QCheckBox('Use Eulerian rotation')
#        rotModeCB.setCheckState(qt.Checked)
#        rotModeCB.stateChanged.connect(self.checkEulerian)
#        rotationLayout.addWidget(rotModeCB, 0, 0)

        rotValidator = qt.QDoubleValidator()
        rotValidator.setRange(-180., 180., 9)
        self.rotationSliders = []
        self.rotationEditors = []
        for iaxis, axis in enumerate(['pitch (Rx)', 'roll (Ry)', 'yaw (Rz)']):
            axLabel = qt.QLabel(axis)
            axEdit = qt.QLineEdit("0.")
            axEdit.setValidator(rotValidator)
            axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)
            axSlider.setRange(-180, 180, 0.01)
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
                                [[[0.], [0.], [0.]],
                                 [[0.], [0.], [90.]],
                                 [[0.], [90.], [0.]],
                                 [[0.], [35.264], [-45.]]]):
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
        for dim in dims:
            dimMin = np.min(self.customGlWidget.footprintsArray[:, dim])
            dimMax = np.max(self.customGlWidget.footprintsArray[:, dim])
            newScale = 1.9 * self.customGlWidget.aPos[dim] /\
                (dimMax - dimMin) * self.customGlWidget.maxLen
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
                (0.2, 2., 0.25, 3.))):
            axLabel = qt.QLabel(axis)
            opacityValidator = qt.QDoubleValidator()
            axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)

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
        self.mplFig = mpl.figure.Figure(dpi=self.logicalDpiX()*0.8)
        self.mplFig.patch.set_alpha(0.)
        self.mplFig.subplots_adjust(left=0.15, bottom=0.15, top=0.92)
        self.mplAx = self.mplFig.add_subplot(111)
        self.mplFig.suptitle("")

        self.drawColorMap('energy')
        self.paletteWidget = qt.FigCanvas(self.mplFig)
        self.paletteWidget.setSizePolicy(qt.QSizePolicy.Maximum,
                                         qt.QSizePolicy.Maximum)
        self.paletteWidget.span = mpl.widgets.RectangleSelector(
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

        selSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)
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
#            self, qt.Horizontal, qt.glowTopScale)
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
            axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)
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
            axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)

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
                 'Enable blending',
                 'Depth test for Lines',
                 'Depth test for Points',
                 'Invert scene color',
                 'Use scalable font',
                 'Show Virtual Screen label',
                 'Virtual Screen for Indexing',
                 'Show lost rays',
                 'Show local axes'],
                [self.checkAA,
                 self.checkBlending,
                 self.checkLineDepthTest,
                 self.checkPointDepthTest,
                 self.invertSceneColor,
                 self.checkScalableFont,
                 self.checkShowLabels,
                 self.checkVSColor,
                 self.checkShowLost,
                 self.checkShowLocalAxes])):
            aaCheckBox = qt.QCheckBox(cbText)
            aaCheckBox.setChecked(iCB in [1, 2])
            aaCheckBox.stateChanged.connect(cbFunc)
            self.sceneControls.append(aaCheckBox)
            sceneLayout.addWidget(aaCheckBox)

        for it, (what, tt, defv) in enumerate(zip(
                ['Default OE thickness, mm',
                 'Force OE thickness, mm',
                 'Aperture frame size, %'],
                ['For OEs that do not have thickness',
                 'For OEs that have thickness, e.g. plates or lenses',
                 ''],
                [self.customGlWidget.oeThickness,
                 self.customGlWidget.oeThicknessForce,
                 self.customGlWidget.slitThicknessFraction])):
            tLabel = qt.QLabel(what)
            tLabel.setToolTip(tt)
            tEdit = qt.QLineEdit()
            tEdit.setToolTip(tt)
            if defv is not None:
                tEdit.setText(str(defv))
            tEdit.editingFinished.connect(
                partial(self.updateThicknessFromQLE, tEdit, it))

            layout = qt.QHBoxLayout()
            tLabel.setMinimumWidth(145)
            layout.addWidget(tLabel)
            tEdit.setMaximumWidth(48)
            layout.addWidget(tEdit)
            layout.addStretch()
            sceneLayout.addLayout(layout)

        axLabel = qt.QLabel('Font Size')
        axSlider = qt.glowSlider(self, qt.Horizontal, qt.glowTopScale)
        axSlider.setRange(1, 20, 0.5)
        axSlider.setValue(5)
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
        oeTileValidator.setRange(1, 20)
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

    def toggleDock(self):
        if self.parentRef is not None:
            self.parentRef.catchViewer()
            self.parentRef = None

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
                child.setCheckState(qt.Checked if i < 2 else qt.Unchecked)
                headerRow.append(child)
            newModel.invisibleRootItem().appendRow(headerRow)
        newModel.itemChanged.connect(self.updateRaysList)
        return newModel

    def updateOEsList(self, arrayOfRays):
        self.oesList = None
        self.beamsToElements = None
        self.populateOEsList(arrayOfRays)
        self.updateSegmentsModel(arrayOfRays)
        self.oeTree.resizeColumnToContents(0)
        self.centerCB.blockSignals(True)
        tmpIndex = self.centerCB.currentIndex()
        for i in range(self.centerCB.count()):
            self.centerCB.removeItem(0)
        for key in self.oesList.keys():
            self.centerCB.addItem(str(key))
#        self.segmentsModel.layoutChanged.emit()
        try:
            self.centerCB.setCurrentIndex(tmpIndex)
        except:  # analysis:ignore
            pass
        self.centerCB.blockSignals(False)
        self.customGlWidget.arrayOfRays = arrayOfRays
        self.customGlWidget.beamsDict = arrayOfRays[1]
        self.customGlWidget.oesList = self.oesList
        self.customGlWidget.beamsToElements = self.beamsToElements
#        self.customGlWidget.newColorAxis = True
#        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.changeColorAxis(None)
        self.customGlWidget.positionVScreen()
        self.customGlWidget.glDraw()

    def populateOEsList(self, arrayOfRays):
        self.oesList = OrderedDict()
        self.beamsToElements = OrderedDict()
        oesList = arrayOfRays[2]
        for segment in arrayOfRays[0]:
            if segment[0] == segment[2]:
                oesList[segment[0]].append(segment[1])
                oesList[segment[0]].append(segment[3])

        for segOE, oeRecord in oesList.items():
            if len(oeRecord) > 2:  # DCM
                elNames = [segOE+'_Entrance', segOE+'_Exit']
            else:
                elNames = [segOE]

            for elName in elNames:
                self.oesList[elName] = [oeRecord[0]]  # pointer to object
                if len(oeRecord) < 3 or elName.endswith('_Entrance'):
                    center = list(oeRecord[0].center)
                    is2ndXtal = False
                else:
                    is2ndXtal = True
#                    center = arrayOfRays[1][oeRecord[3]].wCenter
                    gb = self.oesList[elName][0].local_to_global(
                        rsources.Beam(nrays=2), returnBeam=True,
                        is2ndXtal=is2ndXtal)
                    center = [gb.x[0], gb.y[0], gb.z[0]]

                for segment in arrayOfRays[0]:
                    ind = oeRecord[1]*2
                    if str(segment[ind]) == str(segOE):
                        if len(oeRecord) < 3 or\
                            (elName.endswith('Entrance') and
                                str(segment[3]) == str(oeRecord[2])) or\
                            (elName.endswith('Exit') and
                                str(segment[3]) == str(oeRecord[3])):
                            if len(self.oesList[elName]) < 2:
                                self.oesList[elName].append(
                                    str(segment[ind+1]))
                                self.beamsToElements[segment[ind+1]] =\
                                    elName
                                break
                else:
                    self.oesList[elName].append(None)
                self.oesList[elName].append(center)
                self.oesList[elName].append(is2ndXtal)

    def createRow(self, text, segMode):
        newRow = []
        for iCol in range(4):
            newItem = qt.QStandardItem(str(text) if iCol == 0 else "")
            newItem.setCheckable(True if (segMode == 3 and iCol == 0) or
                                 (segMode == 1 and iCol > 0) else False)
            if newItem.isCheckable():
                newItem.setCheckState(qt.Checked if iCol < 2 else qt.Unchecked)
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

    def populateSegmentsModel(self, arrayOfRays):
        for element, elRecord in self.oesList.items():
            newRow = self.createRow(element, 1)
            for segment in arrayOfRays[0]:
                cond = str(segment[1]) == str(elRecord[1])  # or\
#                    str(segment[0])+"_Entrance" == element
                if cond:
                    try:  # if segment[3] is not None:
                        endBeamText = "to {}".format(
                            self.beamsToElements[segment[3]])
                        newRow[0].appendRow(self.createRow(endBeamText, 3))
                    except:  # analysis:ignore
                        continue
            self.segmentsModelRoot.appendRow(newRow)

    def drawColorMap(self, axis):
        xv, yv = np.meshgrid(np.linspace(0, colorFactor, 200),
                             np.linspace(0, 1, 200))
        xv = xv.flatten()
        yv = yv.flatten()
        self.im = self.mplAx.imshow(mpl.colors.hsv_to_rgb(np.vstack((
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
                histImage[0:histVals[col], col, :] = mpl.colors.hsv_to_rgb(
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
            if self.paletteWidget.span.visible:
                self.paletteWidget.span.extents =\
                    self.paletteWidget.span.extents
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

    def checkBlending(self, state):
        self.customGlWidget.enableBlending = True if state > 0 else False
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

    def checkVSColor(self, state):
        self.customGlWidget.vScreenForColors = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def checkShowLost(self, state):
        self.customGlWidget.showLostRays = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                pass
        editor.setText("{0:.2f}".format(position))
        self.customGlWidget.rotations[iax][0] = np.float32(position)
        self.customGlWidget.updateQuats()
        self.customGlWidget.glDraw()

    def updateRotationFromGL(self, actPos):
        for iaxis, (slider, editor) in\
                enumerate(zip(self.rotationSliders, self.rotationEditors)):
            value = actPos[iaxis][0]
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

    def updateFontSize(self, slider, position):
        # slider = self.sender()
        if isinstance(position, int):
            try:
                position /= slider.scale
            except:  # analysis:ignore
                pass
        self.customGlWidget.fontSize = position
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

        if item.column() == 3:
            self.customGlWidget.labelsToPlot = []
            for ioe in range(self.segmentsModelRoot.rowCount() - 1):
                if self.segmentsModelRoot.child(ioe + 1, 3).checkState() == 2:
                    self.customGlWidget.labelsToPlot.append(str(
                        self.segmentsModelRoot.child(ioe + 1, 0).text()))
        else:
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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
                           partial(self.centerEl, str(selectedItem.text())))
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
        mAction.setChecked(False if self.customGlWidget.virtScreen is None
                           else True)
        mAction.triggered.connect(self.customGlWidget.toggleVScreen)
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
            image = self.customGlWidget.grabFrameBuffer(withAlpha=True)
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            image.save(filename)

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
            params = np.load(filename).item()
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
        self.customGlWidget.coordOffset = list(self.oesList[str(oeName)][2])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def updateCutoffFromQLE(self, editor):
        try:
            # editor = self.sender()
            value = float(re.sub(',', '.', str(editor.text())))
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.cutoffI = np.float32(value)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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
        else:
            return
        self.customGlWidget.glDraw()

    def updateTileFromQLE(self, editor, ia):
        # editor = self.sender()
        value = float(str(editor.text()))
        self.customGlWidget.tiles[ia] = np.int32(value)
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


class xrtGlWidget(qt.QGLWidget):
    rotationUpdated = qt.Signal(np.ndarray)
    scaleUpdated = qt.Signal(np.ndarray)
    histogramUpdated = qt.Signal(tuple)

    def __init__(self, parent, arrayOfRays, modelRoot, oesList, b2els, signal):
        qt.QGLWidget.__init__(self, parent)
        self.QookSignal = signal
        self.virtScreen = None
        self.virtBeam = None
        self.virtDotsArray = None
        self.virtDotsColor = None
        self.vScreenForColors = False
        self.globalColorIndex = None
        self.isVirtScreenNormal = False
        self.segmentModel = modelRoot
        self.vScreenSize = 0.5
        self.setMinimumSize(400, 400)
        self.aspect = 1.
        self.depthScaler = 0.
        self.viewPortGL = [0, 0, 500, 500]
        self.perspectiveEnabled = True
        self.cameraAngle = 60
        self.setMouseTracking(True)
        self.surfCPOrder = 4
        self.oesToPlot = []
        self.labelsToPlot = []
        self.tiles = [2, 2]
        self.arrayOfRays = arrayOfRays
        self.beamsDict = arrayOfRays[1]
        self.oesList = oesList
        self.oeContour = dict()
        self.slitEdges = dict()
        self.beamsToElements = b2els
        self.oeThickness = 5  # mm
        self.oeThicknessForce = None
        self.slitThicknessFraction = 50
        self.contourWidth = 2

        self.projectionsVisibility = [0, 0, 0]
        self.lineOpacity = 0.1
        self.lineWidth = 1
        self.pointOpacity = 0.1
        self.pointSize = 1
        self.linesDepthTest = True
        self.pointsDepthTest = False
        self.labelCoordPrec = 1

        self.lineProjectionOpacity = 0.1
        self.lineProjectionWidth = 1
        self.pointProjectionOpacity = 0.1
        self.pointProjectionSize = 1

        self.coordOffset = [0., 0., 0.]
        self.enableAA = False
        self.enableBlending = True
        self.cutoffI = 0.01
        self.getColor = raycing.get_energy
        self.globalNorm = True
        self.iHSV = False
        self.newColorAxis = True
        self.colorMin = -1e20
        self.colorMax = 1e20
        self.selColorMin = None
        self.selColorMax = None
        self.scaleVec = np.array([1e3, 1e1, 1e3])
        self.maxLen = 1.
        self.showLostRays = False
        self.showLocalAxes = False
        self.populateVerticesArray(modelRoot)

        self.drawGrid = True
        self.fineGridEnabled = False
        self.showOeLabels = False
        self.aPos = [0.9, 0.9, 0.9]
        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])
        self.coordinateGridLineWidth = 1
#        self.fixedFontType = 'GLUT_BITMAP_TIMES_ROMAN'
        self.fixedFontType = 'GLUT_BITMAP_HELVETICA'
        self.fixedFontSize = '12'  # 10, 12, 18 for Helvetica; 10, 24 for Roman
        self.fixedFont = getattr(gl, "{0}_{1}".format(self.fixedFontType,
                                                      self.fixedFontSize))
        self.useScalableFont = False
        self.fontSize = 5
        self.scalableFontType = gl.GLUT_STROKE_ROMAN
#        self.scalableFontType = gl.GLUT_STROKE_MONO_ROMAN
        self.scalableFontWidth = 1
        self.useFontAA = False
        self.tVec = np.array([0., 0., 0.])
        self.cameraTarget = [0., 0., 0.]
        self.cameraPos = np.float32([3.5, 0., 0.])
        self.isEulerian = False
        self.rotations = np.float32([[0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
        self.textOrientation = [0.5, 0.5, 0.5, 0.5]
        self.updateQuats()
        pModelT = np.identity(4)
        self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
        self.signs = np.ones_like(pModelT)
        self.invertColors = False
        self.showHelp = False
#        self.glDraw()

    def eulerToQ(self, rotMatrXYZ):
        hPitch = np.radians(rotMatrXYZ[0][0]) * 0.5
        hRoll = np.radians(rotMatrXYZ[1][0]) * 0.5
        hYaw = np.radians(rotMatrXYZ[2][0]) * 0.5

        cosPitch = np.cos(hPitch)
        sinPitch = np.sin(hPitch)
        cosRoll = np.cos(hRoll)
        sinRoll = np.sin(hRoll)
        cosYaw = np.cos(hYaw)
        sinYaw = np.sin(hYaw)

        return [cosPitch*cosRoll*cosYaw - sinPitch*sinRoll*sinYaw,
                sinRoll*sinYaw*cosPitch + sinPitch*cosRoll*cosYaw,
                sinRoll*cosPitch*cosYaw - sinPitch*sinYaw*cosRoll,
                sinYaw*cosPitch*cosRoll + sinPitch*sinRoll*cosYaw]

    def qToVec(self, quat):
        angle = 2 * np.arccos(quat[0])
        q2v = np.sin(angle * 0.5)
        qbt1 = quat[1] / q2v if q2v != 0 else 0
        qbt2 = quat[2] / q2v if q2v != 0 else 0
        qbt3 = quat[3] / q2v if q2v != 0 else 0
        return [np.degrees(angle), qbt1, qbt2, qbt3]

    def rotateZYX(self):
        if self.isEulerian:
            gl.glRotatef(*self.rotations[0])
            gl.glRotatef(*self.rotations[1])
            gl.glRotatef(*self.rotations[2])
        else:
            gl.glRotatef(*self.rotationVec)

    def updateQuats(self):
        self.qRot = self.eulerToQ(self.rotations)
        self.rotationVec = self.qToVec(self.qRot)
        self.qText = self.qToVec(
            self.quatMult([self.qRot[0], -self.qRot[1],
                           -self.qRot[2], -self.qRot[3]],
                          self.textOrientation))

    def vecToQ(self, vec, alpha):
        """ Quaternion from vector and angle"""
        return np.insert(vec*np.sin(alpha*0.5), 0, np.cos(alpha*0.5))

    def rotateVecQ(self, vec, q):
        qn = np.copy(q)
        qn[1:] *= -1
        return self.quatMult(self.quatMult(
            q, self.vecToQ(vec, np.pi*0.25)), qn)[1:]

    def setPointSize(self, pSize):
        self.pointSize = pSize
        self.glDraw()

    def setLineWidth(self, lWidth):
        self.lineWidth = lWidth
        self.glDraw()

    def populateVerticesOnly(self, segmentsModelRoot):
        if segmentsModelRoot is None:
            return
        self.segmentModel = segmentsModelRoot
        # signal = self.QookSignal
        self.verticesArray = None
        self.footprintsArray = None
        self.oesToPlot = []
        self.labelsToPlot = []
        self.footprints = dict()
        colorsRays = None
        alphaRays = None
        colorsDots = None
        alphaDots = None
        globalColorsDots = None
        globalColorsRays = None

        verticesArrayLost = None
        colorsRaysLost = None
        footprintsArrayLost = None
        colorsDotsLost = None
        maxLen = 1.
        tmpMax = -1.0e12 * np.ones(3)
        tmpMin = -1. * tmpMax

        if self.newColorAxis:
            newColorMax = -1e20
            newColorMin = 1e20
            if self.selColorMin is None:
                self.selColorMin = newColorMin
            if self.selColorMax is None:
                self.selColorMax = newColorMax
        else:
            newColorMax = self.colorMax
            newColorMin = self.colorMin

#        totalOEs = range(segmentsModelRoot.rowCount() - 2)
        for ioe in range(segmentsModelRoot.rowCount() - 1):
            ioeItem = segmentsModelRoot.child(ioe + 1, 0)
#            try:
#                if signal is not None:
#                    signalStr = "Plotting beams for {}, %p% done.".format(
#                        str(ioeItem.text()))
#                    signal.emit((float(ioe) / float(totalOEs),
#                                 signalStr))
#            except:
#                pass
            if segmentsModelRoot.child(ioe + 1, 2).checkState() == 2:
                self.oesToPlot.append(str(ioeItem.text()))
                self.footprints[str(ioeItem.text())] = None
            if segmentsModelRoot.child(ioe + 1, 3).checkState() == 2:
                self.labelsToPlot.append(str(ioeItem.text()))

            try:
                startBeam = self.beamsDict[
                    self.oesList[str(ioeItem.text())][1]]
#                lostNum = self.oesList[str(ioeItem.text())][0].lostNum
                # good = startBeam.state > 0
                good = (startBeam.state == 1) | (startBeam.state == 2)
                if len(startBeam.state[good]) > 0:
                    for tmpCoord, tAxis in enumerate(['x', 'y', 'z']):
                        axMin = np.min(getattr(startBeam, tAxis)[good])
                        axMax = np.max(getattr(startBeam, tAxis)[good])
                        if axMin < tmpMin[tmpCoord]:
                            tmpMin[tmpCoord] = axMin
                        if axMax > tmpMax[tmpCoord]:
                            tmpMax[tmpCoord] = axMax

                    newColorMax = max(np.max(
                        self.getColor(startBeam)[good]),
                        newColorMax)
                    newColorMin = min(np.min(
                        self.getColor(startBeam)[good]),
                        newColorMin)
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    continue

            if self.newColorAxis:
                if newColorMin != self.colorMin:
                    self.colorMin = newColorMin
                    self.selColorMin = self.colorMin
                if newColorMax != self.colorMax:
                    self.colorMax = newColorMax
                    self.selColorMax = self.colorMax

            if ioeItem.hasChildren():
                for isegment in range(ioeItem.rowCount()):
                    segmentItem0 = ioeItem.child(isegment, 0)
                    if segmentItem0.checkState() == 2:
                        endBeam = self.beamsDict[
                            self.oesList[str(segmentItem0.text())[3:]][1]]
                        # good = startBeam.state > 0
                        good = (startBeam.state == 1) | (startBeam.state == 2)
                        if len(startBeam.state[good]) == 0:
                            continue
                        intensity = startBeam.Jss + startBeam.Jpp
                        intensityAll = intensity / np.max(intensity[good])

                        good = np.logical_and(good,
                                              intensityAll >= self.cutoffI)
                        goodC = np.logical_and(
                            self.getColor(startBeam) <= self.selColorMax,
                            self.getColor(startBeam) >= self.selColorMin)

                        good = np.logical_and(good, goodC)

                        if self.vScreenForColors and\
                                self.globalColorIndex is not None:
                            good = np.logical_and(good, self.globalColorIndex)
                            globalColorsRays = np.repeat(
                                self.globalColorArray[good], 2, axis=0) if\
                                globalColorsRays is None else np.concatenate(
                                    (globalColorsRays,
                                     np.repeat(self.globalColorArray[good], 2,
                                               axis=0)))
                        else:
                            if self.globalNorm:
                                alphaMax = 1.
                            else:
                                if len(intensity[good]) > 0:
                                    alphaMax = np.max(intensity[good])
                                else:
                                    alphaMax = 1.
                            alphaMax = alphaMax if alphaMax != 0 else 1.

                            alphaRays = np.repeat(intensity[good] / alphaMax,
                                                  2).T\
                                if alphaRays is None else np.concatenate(
                                    (alphaRays.T,
                                     np.repeat(intensity[good] / alphaMax,
                                               2).T))
                            colorsRays = np.repeat(np.array(self.getColor(
                                startBeam)[good]), 2).T if\
                                colorsRays is None else np.concatenate(
                                    (colorsRays.T,
                                     np.repeat(np.array(self.getColor(
                                         startBeam)[good]), 2).T))
                        vertices = np.array(
                            [startBeam.x[good] - self.coordOffset[0],
                             endBeam.x[good] - self.coordOffset[0]]).flatten(
                                 'F')
                        vertices = np.vstack((vertices, np.array(
                            [startBeam.y[good] - self.coordOffset[1],
                             endBeam.y[good] - self.coordOffset[1]]).flatten(
                                 'F')))
                        vertices = np.vstack((vertices, np.array(
                            [startBeam.z[good] - self.coordOffset[2],
                             endBeam.z[good] - self.coordOffset[2]]).flatten(
                                 'F')))

                        self.verticesArray = vertices.T if\
                            self.verticesArray is None else\
                            np.vstack((self.verticesArray, vertices.T))

                        if self.showLostRays:
                            try:
                                lostNum = self.oesList[str(
                                    segmentItem0.text())[3:]][0].lostNum
                            except:  # analysis:ignore
                                lostNum = 1e3
                            lost = startBeam.state == lostNum
                            try:
                                lostOnes = len(startBeam.x[lost]) * 2
                            except:  # analysis:ignore
                                lostOnes = 0
                            colorsRaysLost = lostOnes if colorsRaysLost is\
                                None else colorsRaysLost + lostOnes
                            if lostOnes > 0:
                                verticesLost = np.array(
                                    [startBeam.x[lost] - self.coordOffset[0],
                                     endBeam.x[lost] -
                                     self.coordOffset[0]]).flatten('F')
                                verticesLost = np.vstack(
                                    (verticesLost, np.array(
                                        [startBeam.y[lost]-self.coordOffset[1],
                                         endBeam.y[lost] -
                                         self.coordOffset[1]]).flatten('F')))
                                verticesLost = np.vstack(
                                    (verticesLost, np.array(
                                        [startBeam.z[lost]-self.coordOffset[2],
                                         endBeam.z[lost] -
                                         self.coordOffset[2]]).flatten('F')))
                                verticesArrayLost = verticesLost.T if\
                                    verticesArrayLost is None else\
                                    np.vstack(
                                        (verticesArrayLost, verticesLost.T))

            if segmentsModelRoot.child(ioe + 1, 1).checkState() == 2:
                # good = startBeam.state > 0
                good = (startBeam.state == 1) | (startBeam.state == 2)
                if len(startBeam.state[good]) == 0:
                    continue
                intensity = startBeam.Jss + startBeam.Jpp
                try:
                    intensityAll = intensity / np.max(intensity[good])
                    good = np.logical_and(good, intensityAll >= self.cutoffI)
                    goodC = np.logical_and(
                        self.getColor(startBeam) <= self.selColorMax,
                        self.getColor(startBeam) >= self.selColorMin)

                    good = np.logical_and(good, goodC)
                except:  # analysis:ignore
                    if _DEBUG_:
                        raise
                    else:
                        continue
                if self.vScreenForColors and self.globalColorIndex is not None:
                    good = np.logical_and(good, self.globalColorIndex)
                    globalColorsDots = self.globalColorArray[good] if\
                        globalColorsDots is None else np.concatenate(
                            (globalColorsDots, self.globalColorArray[good]))
                else:
                    if self.globalNorm:
                        alphaMax = 1.
                    else:
                        if len(intensity[good]) > 0:
                            alphaMax = np.max(intensity[good])
                        else:
                            alphaMax = 1.

                    alphaMax = alphaMax if alphaMax != 0 else 1.
                    alphaDots = intensity[good].T / alphaMax if\
                        alphaDots is None else np.concatenate(
                            (alphaDots.T, intensity[good].T / alphaMax))
                    colorsDots = np.array(self.getColor(
                        startBeam)[good]).T if\
                        colorsDots is None else np.concatenate(
                            (colorsDots.T, np.array(self.getColor(
                                startBeam)[good]).T))

                vertices = np.array(startBeam.x[good] - self.coordOffset[0])
                vertices = np.vstack((vertices, np.array(
                    startBeam.y[good] - self.coordOffset[1])))
                vertices = np.vstack((vertices, np.array(
                    startBeam.z[good] - self.coordOffset[2])))
                self.footprintsArray = vertices.T if\
                    self.footprintsArray is None else\
                    np.vstack((self.footprintsArray, vertices.T))
                if self.showLostRays:
                    try:
                        lostNum = self.oesList[str(ioeItem.text())][0].lostNum
                    except:  # analysis:ignore
                        lostNum = 1e3
                    lost = startBeam.state == lostNum
                    try:
                        lostOnes = len(startBeam.x[lost])
                    except:  # analysis:ignore
                        lostOnes = 0
                    colorsDotsLost = lostOnes if\
                        colorsDotsLost is None else\
                        colorsDotsLost + lostOnes
                    if lostOnes > 0:
                        verticesLost = np.array(startBeam.x[lost] -
                                                self.coordOffset[0])
                        verticesLost = np.vstack((verticesLost, np.array(
                            startBeam.y[lost] - self.coordOffset[1])))
                        verticesLost = np.vstack((verticesLost, np.array(
                            startBeam.z[lost] - self.coordOffset[2])))
                        footprintsArrayLost = verticesLost.T if\
                            footprintsArrayLost is None else\
                            np.vstack((footprintsArrayLost, verticesLost.T))

        try:
            if self.colorMin == self.colorMax:
                if self.colorMax == 0:  # and self.colorMin == 0 too
                    self.colorMin, self.colorMax = -0.1, 0.1
                else:
                    self.colorMin = self.colorMax * 0.99
                    self.colorMax *= 1.01

            if self.vScreenForColors and self.globalColorIndex is not None:
                self.raysColor = globalColorsRays
            elif colorsRays is not None:
                colorsRays = colorFactor * (colorsRays-self.colorMin) /\
                    (self.colorMax - self.colorMin)
                colorsRays = np.dstack(
                    (colorsRays,
                     np.ones_like(alphaRays)*colorSaturation,
                     alphaRays if self.iHSV else
                     np.ones_like(alphaRays)))
                colorsRGBRays = np.squeeze(mpl.colors.hsv_to_rgb(colorsRays))
                if self.globalNorm and len(alphaRays) > 0:
                    alphaMax = np.max(alphaRays)
                else:
                    alphaMax = 1.
                alphaMax = alphaMax if alphaMax != 0 else 1.
                alphaColorRays = np.array([alphaRays / alphaMax]).T
                self.raysColor = np.float32(np.hstack([colorsRGBRays,
                                                       alphaColorRays]))
            if self.showLostRays:
                if colorsRaysLost is not None:
                    lostColor = np.zeros((colorsRaysLost, 4))
                    lostColor[:, 0] = 0.5
                    lostColor[:, 3] = 0.25
                    self.raysColor = np.float32(np.vstack((self.raysColor,
                                                           lostColor)))
                if verticesArrayLost is not None:
                    self.verticesArray = np.float32(np.vstack((
                        self.verticesArray, verticesArrayLost)))
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                pass

        try:
            if self.colorMin == self.colorMax:
                if self.colorMax == 0:  # and self.colorMin == 0 too
                    self.colorMin, self.colorMax = -0.1, 0.1
                else:
                    self.colorMin = self.colorMax * 0.99
                    self.colorMax *= 1.01
            if self.vScreenForColors and self.globalColorIndex is not None:
                self.dotsColor = globalColorsDots
            elif colorsDots is not None:
                colorsDots = colorFactor * (colorsDots-self.colorMin) /\
                    (self.colorMax - self.colorMin)
                colorsDots = np.dstack(
                    (colorsDots,
                     np.ones_like(alphaDots)*colorSaturation,
                     alphaDots if self.iHSV else
                     np.ones_like(alphaDots)))
                colorsRGBDots = np.squeeze(mpl.colors.hsv_to_rgb(colorsDots))

                if self.globalNorm and len(alphaDots) > 0:
                    alphaMax = np.max(alphaDots)
                else:
                    alphaMax = 1.
                alphaMax = alphaMax if alphaMax != 0 else 1.
                alphaColorDots = np.array([alphaDots / alphaMax]).T
                self.dotsColor = np.float32(np.hstack([colorsRGBDots,
                                                       alphaColorDots]))

            if self.showLostRays:
                if colorsDotsLost is not None:
                    lostColor = np.zeros((colorsDotsLost, 4))
                    lostColor[:, 0] = 0.5
                    lostColor[:, 3] = 0.25
                    self.dotsColor = np.float32(np.vstack((self.dotsColor,
                                                           lostColor)))
                if footprintsArrayLost is not None:
                    self.footprintsArray = np.float32(np.vstack((
                        self.footprintsArray, footprintsArrayLost)))
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                pass
        tmpMaxLen = np.max(tmpMax - tmpMin)
        if tmpMaxLen > maxLen:
            maxLen = tmpMaxLen
        self.maxLen = maxLen
        self.newColorAxis = False

    def populateVerticesArray(self, segmentsModelRoot):
        self.populateVerticesOnly(segmentsModelRoot)
        self.populateVScreen()
        if self.vScreenForColors:
            self.populateVerticesOnly(segmentsModelRoot)

    def modelToWorld(self, coords, dimension=None):
        self.maxLen = self.maxLen if self.maxLen != 0 else 1.
        if dimension is None:
            return np.float32(((coords + self.tVec) * self.scaleVec) /
                              self.maxLen)
        else:
            return np.float32(((coords[dimension] + self.tVec[dimension]) *
                              self.scaleVec[dimension]) / self.maxLen)

    def worldToModel(self, coords):
        return np.float32(coords * self.maxLen / self.scaleVec - self.tVec)

    def drawText(self, coord, text, noScalable=False, alignment=None,
                 useCaption=False):
        useScalableFont = False if noScalable else self.useScalableFont
        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
        pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        if not useScalableFont:
            gl.glRasterPos3f(*coord)
            for symbol in text:
                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
        else:
            tLineWidth = gl.glGetDoublev(gl.GL_LINE_WIDTH)
            tLineAA = gl.glIsEnabled(gl.GL_LINE_SMOOTH)
            if self.useFontAA:
                gl.glEnable(gl.GL_LINE_SMOOTH)
            else:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            gl.glLineWidth(self.scalableFontWidth)

            fontScale = self.fontSize / 12500.
            coordShift = np.zeros(3, dtype=np.float32)
            fontSizeLoc = np.float32(np.array([104.76, 119.05, 0])*fontScale)
            if alignment is not None:
                if alignment[0] == 'left':
                    coordShift[0] = -fontSizeLoc[0] * len(text)
                else:
                    coordShift[0] = fontSizeLoc[0]

                if alignment[1] == 'top':
                    vOffset = 0.5
                elif alignment[1] == 'bottom':
                    vOffset = -1.5
                else:
                    vOffset = -0.5
                coordShift[1] = vOffset * fontSizeLoc[1]
            if useCaption:
                textWidth = 0
                for symbol in text.strip(" "):
                    textWidth += gl.glutStrokeWidth(self.scalableFontType,
                                                    ord(symbol))
                gl.glPushMatrix()
                gl.glTranslatef(*coord)
                gl.glRotatef(*self.qText)
                gl.glTranslatef(*coordShift)
                gl.glScalef(fontScale, fontScale, fontScale)
                depthCounter = 1
                spaceFound = False
                while not spaceFound:
                    depthCounter += 1
                    for dy in [-1, 1]:
                        for dx in [1, -1]:
                            textShift = (depthCounter+0.5*dy) * 119.05*1.5
                            gl.glPushMatrix()
                            textPos = [dx*depthCounter * 119.05*1.5 +
                                       (0 if dx > 0 else -1) * textWidth,
                                       dy*textShift, 0]
                            gl.glTranslatef(*textPos)
                            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
                            bottomLeft = np.array(gl.gluProject(
                                *[0, 0, 0], model=pModel, proj=pProjection,
                                view=pView)[:-1])
                            topRight = np.array(gl.gluProject(
                                *[textWidth, 119.05*2.5, 0],
                                model=pModel, proj=pProjection,
                                view=pView)[:-1])
                            gl.glPopMatrix()
                            spaceFound = True
                            for oeLabel in list(self.labelsBounds.values()):
                                if not (bottomLeft[0] > oeLabel[1][0] or
                                        bottomLeft[1] > oeLabel[1][1] or
                                        topRight[0] < oeLabel[0][0] or
                                        topRight[1] < oeLabel[0][1]):
                                    spaceFound = False
                            if spaceFound:
                                self.labelsBounds[text] = [0]*2
                                self.labelsBounds[text][0] = bottomLeft
                                self.labelsBounds[text][1] = topRight
                                break
                        if spaceFound:
                            break

                gl.glPopMatrix()
                gl.glPushMatrix()
                gl.glTranslatef(*coord)
                gl.glRotatef(*self.qText)
                gl.glScalef(fontScale, fontScale, fontScale)
                captionPos = depthCounter * 119.05*1.5
                gl.glBegin(gl.GL_LINE_STRIP)
                gl.glVertex3f(0, 0, 0)
                gl.glVertex3f(captionPos*dx, captionPos*dy, 0)
                gl.glVertex3f(captionPos*dx + textWidth*dx,
                              captionPos*dy, 0)
                gl.glEnd()
                gl.glTranslatef(*textPos)
                for symbol in text.strip(" "):
                    gl.glutStrokeCharacter(self.scalableFontType, ord(symbol))
                gl.glPopMatrix()
            else:
                gl.glPushMatrix()
                gl.glTranslatef(*coord)
                gl.glRotatef(*self.qText)
                gl.glTranslatef(*coordShift)
                gl.glScalef(fontScale, fontScale, fontScale)
                for symbol in text:
                    gl.glutStrokeCharacter(self.scalableFontType, ord(symbol))
                gl.glPopMatrix()

            gl.glLineWidth(tLineWidth)
            if tLineAA:
                gl.glEnable(gl.GL_LINE_SMOOTH)
            else:
                gl.glDisable(gl.GL_LINE_SMOOTH)

    def setMaterial(self, mat):
        if mat == 'Cu':
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT,
                            [0.3, 0.15, 0.15, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE,
                            [0.4, 0.25, 0.15, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR,
                            [1., 0.7, 0.3, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION,
                            [0.1, 0.1, 0.1, 1])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 100)
        elif mat == 'magRed':
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT,
                            [0.6, 0.1, 0.1, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE,
                            [0.8, 0.1, 0.1, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR,
                            [1., 0.1, 0.1, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION,
                            [0.1, 0.1, 0.1, 1])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 100)
        elif mat == 'magBlue':
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT,
                            [0.1, 0.1, 0.6, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE,
                            [0.1, 0.1, 0.8, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR,
                            [0.1, 0.1, 1., 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION,
                            [0.1, 0.1, 0.1, 1])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 100)
        elif mat == 'semiSi':
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT,
                            [0.1, 0.1, 0.1, 0.75])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE,
                            [0.3, 0.3, 0.3, 0.75])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR,
                            [1., 0.9, 0.8, 0.75])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION,
                            [0.1, 0.1, 0.1, 0.75])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 100)
        else:
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT,
                            [0.1, 0.1, 0.1, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE,
                            [0.3, 0.3, 0.3, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR,
                            [1., 0.9, 0.8, 1])
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION,
                            [0.1, 0.1, 0.1, 1])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 100)

    def paintGL(self):
        def makeCenterStr(centerList, prec):
            retStr = '('
            for dim in centerList:
                retStr += '{0:.{1}f}, '.format(dim, prec)
            return retStr[:-2] + ')'

        if self.invertColors:
            gl.glClearColor(1.0, 1.0, 1.0, 1.)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 1.)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        if self.perspectiveEnabled:
            gl.gluPerspective(self.cameraAngle, self.aspect, 0.001, 10000)
        else:
            orthoView = self.cameraPos[0]*0.45
            gl.glOrtho(-orthoView*self.aspect, orthoView*self.aspect,
                       -orthoView, orthoView, -100, 100)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.gluLookAt(self.cameraPos[0], self.cameraPos[1], self.cameraPos[2],
                     self.cameraTarget[0], self.cameraTarget[1],
                     self.cameraTarget[2],
                     0.0, 0.0, 1.0)

        if self.enableBlending:
            gl.glEnable(gl.GL_MULTISAMPLE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#            gl.glBlendFunc(gl.GL_SRC_ALPHA, GL_ONE)
            gl.glEnable(gl.GL_POINT_SMOOTH)
            gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        self.rotateZYX()

        pModel = np.array(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX))[:-1, :-1]
        self.visibleAxes = np.argmax(np.abs(pModel), axis=0)
        self.signs = np.sign(pModel)
        self.axPosModifier = np.ones(3)

        if self.enableAA:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
#            gl.glHint(GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)

        for dim in range(3):
            for iAx in range(3):
                self.axPosModifier[iAx] = (self.signs[iAx][2] if
                                           self.signs[iAx][2] != 0 else 1)
            if self.projectionsVisibility[dim] > 0:
                if self.lineProjectionWidth > 0 and\
                        self.lineProjectionOpacity > 0 and\
                        self.verticesArray is not None:
                    projectionRays = self.modelToWorld(
                        np.copy(self.verticesArray))
                    projectionRays[:, dim] =\
                        -self.aPos[dim] * self.axPosModifier[dim]
                    self.drawArrays(
                        0, gl.GL_LINES, projectionRays, self.raysColor,
                        self.lineProjectionOpacity, self.lineProjectionWidth)

                if self.pointProjectionSize > 0 and\
                        self.pointProjectionOpacity > 0:
                    if self.footprintsArray is not None:
                        projectionDots = self.modelToWorld(
                            np.copy(self.footprintsArray))
                        projectionDots[:, dim] =\
                            -self.aPos[dim] * self.axPosModifier[dim]
                        self.drawArrays(
                            0, gl.GL_POINTS, projectionDots, self.dotsColor,
                            self.pointProjectionOpacity,
                            self.pointProjectionSize)

                    if self.virtDotsArray is not None:
                        projectionDots = self.modelToWorld(
                            np.copy(self.virtDotsArray))
                        projectionDots[:, dim] =\
                            -self.aPos[dim] * self.axPosModifier[dim]
                        self.drawArrays(
                            0, gl.GL_POINTS, projectionDots,
                            self.virtDotsColor,
                            self.pointProjectionOpacity,
                            self.pointProjectionSize)

        if self.enableAA:
            gl.glDisable(gl.GL_LINE_SMOOTH)

        if self.linesDepthTest:
            gl.glEnable(gl.GL_DEPTH_TEST)

        if self.enableAA:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)

        if self.lineWidth > 0 and self.lineOpacity > 0 and\
                self.verticesArray is not None:
            self.drawArrays(1, gl.GL_LINES, self.verticesArray, self.raysColor,
                            self.lineOpacity, self.lineWidth)
        if self.linesDepthTest:
            gl.glDisable(gl.GL_DEPTH_TEST)

        if self.enableAA:
            gl.glDisable(gl.GL_LINE_SMOOTH)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # Surfaces of optical elements:
        if len(self.oesToPlot) > 0 or self.virtScreen is not None:
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glEnable(gl.GL_NORMALIZE)

            self.addLighting(3.)
            for oeString in self.oesToPlot:
                try:
                    oeToPlot = self.oesList[oeString][0]
                    is2ndXtal = self.oesList[oeString][3]
                    if isinstance(oeToPlot, roes.OE):
                        self.plotOeSurface(oeToPlot, is2ndXtal)
                    elif isinstance(oeToPlot, rscreens.HemisphericScreen):
                        self.setMaterial('semiSi')
                        self.plotHemiScreen(oeToPlot)
                    elif isinstance(oeToPlot, rscreens.Screen):
                        self.setMaterial('semiSi')
                        self.plotScreen(oeToPlot, frameColor=[1, 1, 0, 0.8])
                    if isinstance(oeToPlot, (rapertures.RectangularAperture,
                                             rapertures.RoundAperture)):
                        self.setMaterial('Cu')
                        self.plotAperture(oeToPlot)
                    else:
                        continue
                except:  # analysis:ignore
                    if _DEBUG_:
                        raise
                    else:
                        continue

            if self.virtScreen is not None:
                self.setMaterial('semiSi')
                self.plotScreen(self.virtScreen, [self.vScreenSize]*2,
                                [1, 0, 0, 0.8], plotFWHM=True)

            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_NORMALIZE)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisable(gl.GL_DEPTH_TEST)

        if self.enableAA:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        gl.glEnable(gl.GL_DEPTH_TEST)
        if len(self.oesToPlot) > 0:
            for oeString in self.oesToPlot:
                oeToPlot = self.oesList[oeString][0]
                if isinstance(oeToPlot, (rsources.BendingMagnet,
                                         rsources.Wiggler,
                                         rsources.Undulator)):
                    self.plotSource(oeToPlot)
#                elif isinstance(oeToPlot, rscreens.HemisphericScreen):
#                    self.plotHemiScreen(oeToPlot)
#                elif isinstance(oeToPlot, rscreens.Screen):
#                    self.plotScreen(oeToPlot)
#                elif isinstance(oeToPlot, roes.OE):
#                    self.drawOeContour(oeToPlot)
#                elif isinstance(oeToPlot, rapertures.RectangularAperture):
#                    self.drawSlitEdges(oeToPlot)
                else:
                    continue


#            if not self.enableAA:
#                gl.glDisable(gl.GL_LINE_SMOOTH)

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        if self.pointsDepthTest:
            gl.glEnable(gl.GL_DEPTH_TEST)

        if self.pointSize > 0 and self.pointOpacity > 0:
            if self.footprintsArray is not None:
                self.drawArrays(1, gl.GL_POINTS, self.footprintsArray,
                                self.dotsColor, self.pointOpacity,
                                self.pointSize)

            if self.virtDotsArray is not None:
                self.drawArrays(1, gl.GL_POINTS, self.virtDotsArray,
                                self.virtDotsColor, self.pointOpacity,
                                self.pointSize)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        if self.enableAA:
            gl.glDisable(gl.GL_LINE_SMOOTH)

        if self.pointsDepthTest:
            gl.glDisable(gl.GL_DEPTH_TEST)

#        oeLabels = OrderedDict()
        self.labelsBounds = OrderedDict()
        if len(self.labelsToPlot) > 0:
            if self.invertColors:
                gl.glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                gl.glColor4f(1.0, 1.0, 1.0, 1.)
            gl.glLineWidth(1)
#            for oeKey, oeValue in self.oesList.items():
            for oeKey in self.labelsToPlot:
                oeValue = self.oesList[oeKey]
                oeCenterStr = makeCenterStr(oeValue[2],
                                            self.labelCoordPrec)
                oeCoord = np.array(oeValue[2])
                oeCenterStr = '  {0}: {1}mm'.format(
                    oeKey, oeCenterStr)
                oeLabelPos = self.modelToWorld(oeCoord - self.coordOffset)
                self.drawText(oeLabelPos, oeCenterStr, useCaption=True)

        if self.showOeLabels and self.virtScreen is not None:
            vsCenterStr = '    {0}: {1}mm'.format(
                'Virtual Screen', makeCenterStr(self.virtScreen.center,
                                                self.labelCoordPrec))
            try:
                pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
                pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
                pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
                m1 = self.modelToWorld(
                    self.virtScreen.frame[1] - self.coordOffset)
                m2 = self.modelToWorld(
                    self.virtScreen.frame[2] - self.coordOffset)
                scr1 = gl.gluProject(
                    *m1, model=pModel,
                    proj=pProjection, view=pView)[0]
                scr2 = gl.gluProject(
                    *m2, model=pModel,
                    proj=pProjection, view=pView)[0]
                lblCenter = self.virtScreen.frame[1] if scr1 > scr2 else\
                    self.virtScreen.frame[2]
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    lblCenter = self.virtScreen.center
            vsLabelPos = self.modelToWorld(lblCenter - self.coordOffset)
            if self.invertColors:
                gl.glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                gl.glColor4f(1.0, 1.0, 1.0, 1.)
            gl.glLineWidth(1)
            self.drawText(vsLabelPos, vsCenterStr)

        if len(self.oesToPlot) > 0 and self.showLocalAxes:  # Local axes
            for oeString in self.oesToPlot:
                try:
                    oeToPlot = self.oesList[oeString][0]
                    is2ndXtal = self.oesList[oeString][3]
                    if hasattr(oeToPlot, 'local_to_global'):
                        self.drawLocalAxes(oeToPlot, is2ndXtal)
                except:
                    if _DEBUG_:
                        raise
                    else:
                        continue

        gl.glEnable(gl.GL_DEPTH_TEST)
        if self.drawGrid:  # Coordinate grid box
            self.drawCoordinateGrid()
        gl.glFlush()

        self.drawDirectionAxes()
#        if self.showHelp:
#            self.openHelpDialog()
#            self.drawHelp()
        if self.enableBlending:
            gl.glDisable(gl.GL_MULTISAMPLE)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_POINT_SMOOTH)

        gl.glFlush()

    def quatMult(self, qf, qt):
        return [qf[0]*qt[0]-qf[1]*qt[1]-qf[2]*qt[2]-qf[3]*qt[3],
                qf[0]*qt[1]+qf[1]*qt[0]+qf[2]*qt[3]-qf[3]*qt[2],
                qf[0]*qt[2]-qf[1]*qt[3]+qf[2]*qt[0]+qf[3]*qt[1],
                qf[0]*qt[3]+qf[1]*qt[2]-qf[2]*qt[1]+qf[3]*qt[0]]

    def drawCoordinateGrid(self):
        def populateGrid(grids):
            axisLabelC = []
            axisLabelC.extend([np.vstack(
                (self.modelToWorld(grids, 0),
                 np.ones(len(grids[0]))*self.aPos[1]*self.axPosModifier[1],
                 np.ones(len(grids[0]))*-self.aPos[2]*self.axPosModifier[2]
                 ))])
            axisLabelC.extend([np.vstack(
                (np.ones(len(grids[1]))*self.aPos[0]*self.axPosModifier[0],
                 self.modelToWorld(grids, 1),
                 np.ones(len(grids[1]))*-self.aPos[2]*self.axPosModifier[2]
                 ))])
            zAxis = np.vstack(
                (np.ones(len(grids[2]))*-self.aPos[0]*self.axPosModifier[0],
                 np.ones(len(grids[2]))*self.aPos[1]*self.axPosModifier[1],
                 self.modelToWorld(grids, 2)))

            xAxisB = np.vstack(
                (self.modelToWorld(grids, 0),
                 np.ones(len(grids[0]))*-self.aPos[1]*self.axPosModifier[1],
                 np.ones(len(grids[0]))*-self.aPos[2]*self.axPosModifier[2]))
            yAxisB = np.vstack(
                (np.ones(len(grids[1]))*-self.aPos[0]*self.axPosModifier[0],
                 self.modelToWorld(grids, 1),
                 np.ones(len(grids[1]))*-self.aPos[2]*self.axPosModifier[2]))
            zAxisB = np.vstack(
                (np.ones(len(grids[2]))*-self.aPos[0]*self.axPosModifier[0],
                 np.ones(len(grids[2]))*-self.aPos[1]*self.axPosModifier[1],
                 self.modelToWorld(grids, 2)))

            xAxisC = np.vstack(
                (self.modelToWorld(grids, 0),
                 np.ones(len(grids[0]))*-self.aPos[1]*self.axPosModifier[1],
                 np.ones(len(grids[0]))*self.aPos[2]*self.axPosModifier[2]))
            yAxisC = np.vstack(
                (np.ones(len(grids[1]))*-self.aPos[0]*self.axPosModifier[0],
                 self.modelToWorld(grids, 1),
                 np.ones(len(grids[1]))*self.aPos[2]*self.axPosModifier[2]))
            axisLabelC.extend([np.vstack(
                (np.ones(len(grids[2]))*self.aPos[0]*self.axPosModifier[0],
                 np.ones(len(grids[2]))*-self.aPos[1]*self.axPosModifier[1],
                 self.modelToWorld(grids, 2)))])

            xLines = np.vstack(
                (axisLabelC[0], xAxisB, xAxisB, xAxisC)).T.flatten().reshape(
                4*xAxisB.shape[1], 3)
            yLines = np.vstack(
                (axisLabelC[1], yAxisB, yAxisB, yAxisC)).T.flatten().reshape(
                4*yAxisB.shape[1], 3)
            zLines = np.vstack(
                (zAxis, zAxisB, zAxisB, axisLabelC[2])).T.flatten().reshape(
                4*zAxisB.shape[1], 3)

            return axisLabelC, np.vstack((xLines, yLines, zLines))

        def drawGridLines(gridArray, lineWidth, lineOpacity, figType):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gridColor = np.ones((len(gridArray), 4)) * lineOpacity
            gridArrayVBO = gl.vbo.VBO(np.float32(gridArray))
            gridArrayVBO.bind()
            gl.glVertexPointerf(gridArrayVBO)
            gridColorArray = gl.vbo.VBO(np.float32(gridColor))
            gridColorArray.bind()
            gl.glColorPointerf(gridColorArray)
            gl.glLineWidth(lineWidth)
            gl.glDrawArrays(figType, 0, len(gridArrayVBO))
            gridArrayVBO.unbind()
            gridColorArray.unbind()
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        def getAlignment(point, hDim, vDim=None):
            pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
            sp0 = np.array(gl.gluProject(
                *point, model=pModel, proj=pProjection, view=pView))
            pointH = np.copy(point)
            pointH[hDim] *= 1.1
            spH = np.array(gl.gluProject(*pointH, model=pModel,
                                         proj=pProjection, view=pView))
            pointV = np.copy(point)
            if vDim is None:
                vAlign = 'middle'
            else:
                pointV[vDim] *= 1.1
                spV = np.array(gl.gluProject(*pointV, model=pModel,
                                             proj=pProjection, view=pView))
                vAlign = 'top' if spV[1] - sp0[1] > 0 else 'bottom'
            hAlign = 'left' if spH[0] - sp0[0] < 0 else 'right'
            return (hAlign, vAlign)

        back = np.array([[-self.aPos[0], self.aPos[1], -self.aPos[2]],
                         [-self.aPos[0], self.aPos[1], self.aPos[2]],
                         [-self.aPos[0], -self.aPos[1], self.aPos[2]],
                         [-self.aPos[0], -self.aPos[1], -self.aPos[2]]])

        side = np.array([[self.aPos[0], -self.aPos[1], -self.aPos[2]],
                         [-self.aPos[0], -self.aPos[1], -self.aPos[2]],
                         [-self.aPos[0], -self.aPos[1], self.aPos[2]],
                         [self.aPos[0], -self.aPos[1], self.aPos[2]]])

        bottom = np.array([[self.aPos[0], -self.aPos[1], -self.aPos[2]],
                           [self.aPos[0], self.aPos[1], -self.aPos[2]],
                           [-self.aPos[0], self.aPos[1], -self.aPos[2]],
                           [-self.aPos[0], -self.aPos[1], -self.aPos[2]]])

        back[:, 0] *= self.axPosModifier[0]
        side[:, 1] *= self.axPosModifier[1]
        bottom[:, 2] *= self.axPosModifier[2]

#  Calculating regular grids in world coordinates
        limits = np.array([-1, 1])[:, np.newaxis] * np.array(self.aPos)
        allLimits = limits * self.maxLen / self.scaleVec - self.tVec\
            + self.coordOffset
        axisGridArray = []
        gridLabels = []
        precisionLabels = []
        if self.fineGridEnabled:
            fineGridArray = []

        for iAx in range(3):
            m2 = self.aPos[iAx] / 0.9
            dx1 = np.abs(allLimits[:, iAx][0] - allLimits[:, iAx][1]) / m2
            order = np.floor(np.log10(dx1))
            m1 = dx1 * 10**-order

            if (m1 >= 1) and (m1 < 2):
                step = 0.2 * 10**order
            elif (m1 >= 2) and (m1 < 4):
                step = 0.5 * 10**order
            else:
                step = 10**order
            if step < 1:
                decimalX = int(np.abs(order)) + 1 if m1 < 4 else\
                    int(np.abs(order))
            else:
                decimalX = 0

            gridX = np.arange(np.int32(allLimits[:, iAx][0]/step)*step,
                              allLimits[:, iAx][1], step)
            gridX = gridX if gridX[0] >= allLimits[:, iAx][0] else\
                gridX[1:]
            gridLabels.extend([gridX])
            precisionLabels.extend([np.ones_like(gridX)*decimalX])
            axisGridArray.extend([gridX - self.coordOffset[iAx]])
            if self.fineGridEnabled:
                fineStep = step * 0.2
                fineGrid = np.arange(
                    np.int32(allLimits[:, iAx][0]/fineStep)*fineStep,
                    allLimits[:, iAx][1], fineStep)
                fineGrid = fineGrid if\
                    fineGrid[0] >= allLimits[:, iAx][0] else fineGrid[1:]
                fineGridArray.extend([fineGrid - self.coordOffset[iAx]])

        axisL, axGrid = populateGrid(axisGridArray)
        if self.fineGridEnabled:
            tmp, fineAxGrid = populateGrid(fineGridArray)

        if self.invertColors:
            gl.glColor4f(0.0, 0.0, 0.0, 1.)
        else:
            gl.glColor4f(1.0, 1.0, 1.0, 1.)

        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    iAx == self.visibleAxes[2]):
                tAlign = None
                midp = int(len(axisL[iAx][0, :])/2)
                if iAx == self.visibleAxes[1]:  # Side plane,
                    if self.useScalableFont:
                        tAlign = getAlignment(axisL[iAx][:, midp],
                                              self.visibleAxes[0])
                    else:
                        axisL[iAx][self.visibleAxes[2], :] *= 1.05  # depth
                        axisL[iAx][self.visibleAxes[0], :] *= 1.05  # side
                if iAx == self.visibleAxes[0]:  # Bottom plane, left-right
                    if self.useScalableFont:
                        tAlign = getAlignment(axisL[iAx][:, midp],
                                              self.visibleAxes[2],
                                              self.visibleAxes[1])
                    else:
                        axisL[iAx][self.visibleAxes[1], :] *= 1.05  # height
                        axisL[iAx][self.visibleAxes[2], :] *= 1.05  # side
                if iAx == self.visibleAxes[2]:  # Bottom plane, left-right
                    if self.useScalableFont:
                        tAlign = getAlignment(axisL[iAx][:, midp],
                                              self.visibleAxes[0],
                                              self.visibleAxes[1])
                    else:
                        axisL[iAx][self.visibleAxes[1], :] *= 1.05  # height
                        axisL[iAx][self.visibleAxes[0], :] *= 1.05  # side

                for tick, tText, pcs in list(zip(axisL[iAx].T, gridLabels[iAx],
                                                 precisionLabels[iAx])):
                    valueStr = "{0:.{1}f}".format(tText, int(pcs))
                    self.drawText(tick, valueStr, alignment=tAlign)
#            if not self.enableAA:
#                gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        tLineWidth = gl.glGetDoublev(gl.GL_LINE_WIDTH)
        drawGridLines(np.vstack((back, side, bottom)),
                      self.coordinateGridLineWidth * 2, 0.75, gl.GL_QUADS)
        drawGridLines(axGrid, self.coordinateGridLineWidth, 0.5, gl.GL_LINES)
        if self.fineGridEnabled:
            drawGridLines(fineAxGrid, self.coordinateGridLineWidth, 0.25,
                          gl.GL_LINES)
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(tLineWidth)

    def drawArrays(self, tr, geom, vertices, colors, lineOpacity, lineWidth):
        if vertices is None or colors is None:
            return

        if bool(tr):
            vertexArray = gl.vbo.VBO(self.modelToWorld(vertices))
        else:
            vertexArray = gl.vbo.VBO(vertices)
        vertexArray.bind()
        gl.glVertexPointerf(vertexArray)
        pureOpacity = np.copy(colors[:, 3])
        colors[:, 3] = np.float32(pureOpacity * lineOpacity)
        colorArray = gl.vbo.VBO(colors)
        colorArray.bind()
        gl.glColorPointerf(colorArray)
        if geom == gl.GL_LINES:
            gl.glLineWidth(lineWidth)
        else:
            gl.glPointSize(lineWidth)
        gl.glDrawArrays(geom, 0, len(vertices))
        colors[:, 3] = pureOpacity
        colorArray.unbind()
        vertexArray.unbind()

    def plotSource(self, oe):
        # gl.glEnable(gl.GL_MAP2_VERTEX_3)
        # gl.glEnable(gl.GL_MAP2_NORMAL)

        nPeriods = int(oe.Np) if hasattr(oe, 'Np') else 0.5
        if hasattr(oe, 'L0'):
            lPeriod = oe.L0
            maghL = 0.25 * lPeriod * 0.5
        else:
            try:
                lPeriod = (oe.Theta_max - oe.Theta_min) * oe.ro * 1000
            except AttributeError:
                if _DEBUG_:
                    raise
                else:
                    lPeriod = 500.
            maghL = lPeriod

        maghH = 10 * 0.5
        maghW = 10 * 0.5

        surfRot = [[0, 0, 0, 1], [180, 0, 1, 0],
                   [-90, 0, 1, 0], [90, 0, 1, 0],
                   [-90, 1, 0, 0], [90, 1, 0, 0]]
        surfTrans = np.array([[0, 0, maghH], [0, 0, -maghH],
                              [-maghW, 0, 0], [maghW, 0, 0],
                              [0, maghL, 0], [0, -maghL, 0]])
        surfScales = np.array([[maghW*2, maghL*2, 0], [maghW*2, maghL*2, 0],
                               [0, maghL*2, maghH*2], [0, maghL*2, maghH*2],
                               [maghW*2, 0, maghH*2], [maghW*2, 0, maghH*2]])
#        deltaX = 1. / 2.  # float(self.tiles[0])
#        deltaY = 1. / 2.  # float(self.tiles[1])
        magToggle = True
        gl.glLineWidth(1)
        gl.glPushMatrix()
        gl.glTranslatef(*(self.modelToWorld(np.array(oe.center) -
                                            self.coordOffset)))
        gl.glRotatef(np.degrees(oe.pitch * self.scaleVec[2] /
                                self.scaleVec[1]), 1, 0, 0)
        yaw = oe.yaw
        try:
            az = oe.bl.azimuth
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                az = 0
        gl.glRotatef(np.degrees((yaw-az) * self.scaleVec[0] /
                                self.scaleVec[1]), 0, 0, 1)
        gl.glTranslatef(*(-1. * self.modelToWorld(np.array(oe.center) -
                                                  self.coordOffset)))
        for period in range(int(nPeriods) if nPeriods > 0.5 else 1):
            for hp in ([0, 0.5] if nPeriods > 0.5 else [0.25]):
                pY = list(oe.center)[1] - lPeriod * (0.5 * nPeriods -
                                                     period - hp)
                magToggle = not magToggle
                for gap in [maghH*1.25, -maghH*1.25]:
                    cubeCenter = np.array([oe.center[0], pY, oe.center[2]+gap])
#                    self.setMaterial('magRed' if magToggle else 'magBlue')
                    magColor = [0.7, 0.1, 0.1, 1.] if magToggle \
                        else [0.1, 0.1, 0.7, 1.]
                    magToggle = not magToggle
                    for surf in range(6):
                        gl.glPushMatrix()
                        gl.glTranslatef(*(self.modelToWorld(
                            cubeCenter + surfTrans[surf] - self.coordOffset)))
                        gl.glScalef(*(self.modelToWorld(surfScales[surf] -
                                                        self.tVec)))
                        gl.glRotatef(*surfRot[surf])
                        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                        gl.glBegin(gl.GL_QUADS)
                        gl.glColor4f(*magColor)
                        gl.glVertex3f(-0.5, -0.5, 0)
                        gl.glVertex3f(-0.5, 0.5, 0)
                        gl.glVertex3f(0.5, 0.5, 0)
                        gl.glVertex3f(0.5, -0.5, 0)
                        gl.glEnd()
                        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                        gl.glBegin(gl.GL_QUADS)
                        gl.glColor4f(0, 0, 0, 1.)
                        gl.glVertex3f(-0.5, -0.5, 0)
                        gl.glVertex3f(-0.5, 0.5, 0)
                        gl.glVertex3f(0.5, 0.5, 0)
                        gl.glVertex3f(0.5, -0.5, 0)
                        gl.glEnd()
#                        for i in range(2):
#                            xGridOe = np.linspace(-0.5 + i*deltaX,
#                                                  -0.5 + (i+1)*deltaX,
#                                                  self.surfCPOrder)
#                            for k in range(2):
#                                yGridOe = np.linspace(-0.5 + k*deltaY,
#                                                      -0.5 + (k+1)*deltaY,
#                                                      self.surfCPOrder)
#                                xv, yv = np.meshgrid(xGridOe, yGridOe)
#                                xv = xv.flatten()
#                                yv = yv.flatten()
#                                zv = np.zeros_like(xv)
#
#                                surfCP = np.vstack((xv, yv, zv)).T
#                                surfNorm = np.vstack((np.zeros_like(xv),
#                                                      np.zeros_like(xv),
#                                                      np.ones_like(zv),
#                                                      np.ones_like(zv))).T
#
#                                gl.glMap2f(gl.GL_MAP2_VERTEX_3, 0, 1, 0, 1,
#                                           surfCP.reshape(
#                                               self.surfCPOrder,
#                                               self.surfCPOrder, 3))
#
#                                gl.glMap2f(gl.GL_MAP2_NORMAL, 0, 1, 0, 1,
#                                           surfNorm.reshape(
#                                               self.surfCPOrder,
#                                               self.surfCPOrder, 4))
#
#                                gl.glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
#                                               self.surfCPOrder, 0.0, 1.0)
#
#                                gl.glEvalMesh2(gl.GL_FILL, 0,
#                                               self.surfCPOrder,
#                                               0, self.surfCPOrder)
                        gl.glPopMatrix()
        gl.glPopMatrix()
#        gl.glDisable(gl.GL_MAP2_VERTEX_3)
#        gl.glDisable(gl.GL_MAP2_NORMAL)

    def plotCurvedMesh(self, x, y, z, a, b, c, shift):
        surfCP = np.vstack((x - self.coordOffset[0] - shift[0],
                            y - self.coordOffset[1] - shift[1],
                            z - self.coordOffset[2] - shift[2])).T
        gl.glMap2f(gl.GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                   self.modelToWorld(surfCP.reshape(
                       self.surfCPOrder,
                       self.surfCPOrder, 3)))

        surfNorm = np.vstack((a, b, c,
                              np.ones_like(a))).T

        gl.glMap2f(gl.GL_MAP2_NORMAL, 0, 1, 0, 1,
                   surfNorm.reshape(
                       self.surfCPOrder,
                       self.surfCPOrder, 4))

        gl.glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
                       self.surfCPOrder, 0.0, 1.0)

        gl.glEvalMesh2(gl.GL_FILL, 0, self.surfCPOrder,
                       0, self.surfCPOrder)

    def plotOeSurface(self, oe, is2ndXtal):
        def getThickness(element):
            if self.oeThicknessForce is not None:
                return self.oeThicknessForce
            thickness = 0
            if isinstance(oe, roes.Plate):
                if oe.t is not None:
                    return oe.t
            if hasattr(oe, "material"):
                if oe.material is not None:
                    thickness = self.oeThickness
                    if hasattr(oe.material, "t"):
                        thickness = oe.material.t if oe.material.t is not None\
                            else thickness
                    elif isinstance(oe.material, rmats.Multilayer):
                        if oe.material.substrate is not None:
                            if hasattr(oe.material.substrate, 't'):
                                if oe.material.substrate.t is not None:
                                    thickness = oe.material.substrate.t
            return thickness

        thickness = getThickness(oe)

        self.setMaterial('Si')
        gl.glEnable(gl.GL_MAP2_VERTEX_3)
        gl.glEnable(gl.GL_MAP2_NORMAL)

        isWedge = False
        if isinstance(oe, roes.Plate):
            if abs(oe.wedgeAngle) > 0:
                isWedge = True

        # Top and Bottom Surfaces
        nsIndex = int(is2ndXtal)
        if is2ndXtal:
            xLimits = list(oe.limPhysX2)
            yLimits = list(oe.limPhysY2)
        else:
            xLimits = list(oe.limPhysX)
            yLimits = list(oe.limPhysY)

        isClosedSurface = False
        if np.any(np.abs(xLimits) == raycing.maxHalfSizeOfOE):
            isClosedSurface = isinstance(oe, roes.SurfaceOfRevolution)
            if oe.footprint is not None:
                xLimits = oe.footprint[nsIndex][:, 0]
        if np.any(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
            if oe.footprint is not None:
                yLimits = oe.footprint[nsIndex][:, 1]
        localTiles = np.array(self.tiles)

        if oe.shape == 'round':
            rX = np.abs((xLimits[1] - xLimits[0]))*0.5
            rY = np.abs((yLimits[1] - yLimits[0]))*0.5
            cX = (xLimits[1] + xLimits[0])*0.5
            cY = (yLimits[1] + yLimits[0])*0.5
            xLimits = [0, 1.]
            yLimits = [0, 2*np.pi]
            localTiles[1] *= 3
        if isClosedSurface:
            # the limits are in parametric coordinates
            xLimits = yLimits  # s
            yLimits = [0, 2*np.pi]  # phi
            localTiles[1] *= 3

        if is2ndXtal:
            zExt = '2'
        else:
            zExt = '1' if hasattr(oe, 'local_z1') else ''
        local_z = getattr(oe, 'local_r{}'.format(zExt)) if\
            oe.isParametric else getattr(oe, 'local_z{}'.format(zExt))
        local_n = getattr(oe, 'local_n{}'.format(zExt))

        deltaX = (xLimits[1] - xLimits[0]) / float(localTiles[0])
        for i in range(localTiles[0]):
            xGridOe = np.linspace(xLimits[0] + i*deltaX,
                                  xLimits[0] + (i+1)*deltaX,
                                  self.surfCPOrder) + oe.dx

            deltaY = (yLimits[1] - yLimits[0]) / float(localTiles[1])
            for k in range(localTiles[1]):
                yGridOe = np.linspace(yLimits[0] + k*deltaY,
                                      yLimits[0] + (k+1)*deltaY,
                                      self.surfCPOrder)

                xv, yv = np.meshgrid(xGridOe, yGridOe)
                if oe.shape == 'round':
                    xv, yv = rX*xv*np.cos(yv)+cX, rY*xv*np.sin(yv)+cY

                xv = xv.flatten()
                yv = yv.flatten()

                xv = np.copy(xv)  # ?
                yv = np.copy(yv)  # ?
                zv = np.zeros_like(xv)
                if isinstance(oe, roes.SurfaceOfRevolution):
                    # at z=0 (axis of rotation) phi is undefined, therefore:
                    zv -= 100.

                if oe.isParametric and not isClosedSurface:
                    xv, yv, zv = oe.xyz_to_param(xv, yv, zv)

                zv = local_z(xv, yv)
                nv = local_n(xv, yv)

                gbT = rsources.Beam(nrays=len(xv))
                if oe.isParametric:
                    xv, yv, zv = oe.param_to_xyz(xv, yv, zv)

                gbT.x = xv
                gbT.y = yv
                gbT.z = zv
                gbT.a = nv[0] * np.ones_like(zv)
                gbT.b = nv[1] * np.ones_like(zv)
                gbT.c = nv[2] * np.ones_like(zv)

                if thickness > 0 and not isClosedSurface:
                    gbB = rsources.Beam(copyFrom=gbT)
                    if isinstance(oe, (roes.LauePlate, roes.BentLaue2D)):
                        gbB.z[:] = gbT.z - thickness
                        gbB.a = -gbT.a
                        gbB.b = -gbT.b
                        gbB.c = -gbT.c
                    else:
                        gbB.z[:] = -thickness
                        gbB.a[:] = 0
                        gbB.b[:] = 0
                        gbB.c[:] = -1.
                    oe.local_to_global(gbB, is2ndXtal=is2ndXtal)

                oe.local_to_global(gbT, is2ndXtal=is2ndXtal)

                if hasattr(oe, 'centerShift'):
                    cShift = oe.centerShift
                    nSurf = oe._nCRL
                else:
                    cShift = np.zeros(3)
                    nSurf = 1

                for iSurf in range(nSurf):
                    dC = cShift * iSurf
                    self.plotCurvedMesh(gbT.x, gbT.y, gbT.z,
                                        gbT.a, gbT.b, gbT.c, dC)
                    if thickness > 0 \
                            and not isinstance(oe, roes.DoubleParaboloidLens)\
                            and not isClosedSurface and not isWedge:
                        self.plotCurvedMesh(gbB.x, gbB.y, gbB.z,
                                            gbB.a, gbB.b, gbB.c, dC)

    # Side faces
        if isinstance(oe, roes.Plate):
            self.setMaterial('semiSi')
        if thickness > 0 and not isClosedSurface and not isWedge:
            deltaX = (xLimits[1] - xLimits[0]) / float(localTiles[0])
            for ie, yPos in enumerate(yLimits):
                for i in range(localTiles[0]):
                    if oe.shape == 'round':
                        continue
                    xGridOe = np.linspace(xLimits[0] + i*deltaX,
                                          xLimits[0] + (i+1)*deltaX,
                                          self.surfCPOrder) + oe.dx

                    edgeX = xGridOe
                    edgeY = np.ones_like(xGridOe)*yPos
                    edgeZ = np.zeros_like(xGridOe)
                    if oe.isParametric:
                        edgeX, edgeY, edgeZ = oe.xyz_to_param(
                            edgeX, edgeY, edgeZ)

                    edgeZ = local_z(edgeX, edgeY)
                    if oe.isParametric:
                        edgeX, edgeY, edgeZ = oe.param_to_xyz(
                            edgeX, edgeY, edgeZ)

                    gridZ = None
                    for zTop in edgeZ:
                        if isinstance(oe, (roes.LauePlate, roes.BentLaue2D)):
                            gridZ = np.linspace(zTop-thickness, zTop,
                                                self.surfCPOrder) if\
                                gridZ is None else np.concatenate((
                                    gridZ, np.linspace(zTop-thickness, zTop,
                                                       self.surfCPOrder)))    
                        else:
                            gridZ = np.linspace(-thickness, zTop,
                                                self.surfCPOrder) if\
                                gridZ is None else np.concatenate((
                                    gridZ, np.linspace(-thickness, zTop,
                                                       self.surfCPOrder)))

                    gridX = np.repeat(edgeX, len(edgeZ))
                    gridY = np.ones_like(gridX) * yPos

                    xN = np.zeros_like(gridX)
                    yN = (1 if ie == 1 else -1)*np.ones_like(gridX)
                    zN = np.zeros_like(gridX)

                    faceBeam = rsources.Beam(nrays=len(gridX))
                    faceBeam.x = gridX
                    faceBeam.y = gridY
                    faceBeam.z = gridZ
                    faceBeam.a = xN
                    faceBeam.b = yN
                    faceBeam.c = zN
                    oe.local_to_global(faceBeam, is2ndXtal=is2ndXtal)
                    self.plotCurvedMesh(faceBeam.x, faceBeam.y, faceBeam.z,
                                        faceBeam.a, faceBeam.b, faceBeam.c,
                                        [0]*3)

            for ie, xPos in enumerate(xLimits):
                if ie == 0 and oe.shape == 'round':
                    continue
                deltaY = (yLimits[1] - yLimits[0]) / float(localTiles[1])
                for i in range(localTiles[1]):
                    yGridOe = np.linspace(yLimits[0] + i*deltaY,
                                          yLimits[0] + (i+1)*deltaY,
                                          self.surfCPOrder)

                    edgeY = yGridOe
                    edgeX = np.ones_like(yGridOe)*xPos
                    edgeZ = np.zeros_like(xGridOe)

                    if oe.shape == 'round':
                        edgeX, edgeY = rX*edgeX*np.cos(edgeY)+cX,\
                            rY*edgeX*np.sin(edgeY)+cY

                    if oe.isParametric:
                        edgeX, edgeY, edgeZ = oe.xyz_to_param(
                            edgeX, edgeY, edgeZ)
                    edgeZ = local_z(edgeX, edgeY)
                    if oe.isParametric:
                        edgeX, edgeY, edgeZ = oe.param_to_xyz(
                            edgeX, edgeY, edgeZ)

                    zN = 0
                    gridZ = None
                    for zTop in edgeZ:
                        if isinstance(oe, (roes.LauePlate, roes.BentLaue2D)):
                            gridZ = np.linspace(zTop-thickness, zTop,
                                                self.surfCPOrder) if\
                                gridZ is None else np.concatenate((
                                    gridZ, np.linspace(zTop-thickness, zTop,
                                                       self.surfCPOrder)))
                        else:
                            gridZ = np.linspace(-thickness, zTop,
                                                self.surfCPOrder) if\
                                gridZ is None else np.concatenate((
                                    gridZ, np.linspace(-thickness, zTop,
                                                       self.surfCPOrder)))

                    gridY = np.repeat(edgeY, len(edgeZ))
                    if oe.shape == 'round':
                        yN = (gridY-cY) / rY
                        gridX = np.repeat(edgeX, len(edgeZ))
                        xN = (gridX-cX) / rX
                    else:
                        gridX = np.repeat(edgeX, len(edgeZ))
                        yN = np.zeros_like(gridX)
                        xN = (1 if ie == 1 else -1) * np.ones_like(gridX)
                    zN = np.zeros_like(gridX)

                    faceBeam = rsources.Beam(nrays=len(gridX))
                    faceBeam.x = gridX
                    faceBeam.y = gridY
                    faceBeam.z = gridZ

                    faceBeam.a = xN
                    faceBeam.b = yN
                    faceBeam.c = zN

                    oe.local_to_global(faceBeam, is2ndXtal=is2ndXtal)
                    self.plotCurvedMesh(faceBeam.x, faceBeam.y, faceBeam.z,
                                        faceBeam.a, faceBeam.b, faceBeam.c,
                                        [0]*3)
        gl.glDisable(gl.GL_MAP2_VERTEX_3)
        gl.glDisable(gl.GL_MAP2_NORMAL)

        # Contour
#        xBound = np.linspace(xLimits[0], xLimits[1],
#                             self.surfCPOrder*(localTiles[0]+1))
#        yBound = np.linspace(yLimits[0], yLimits[1],
#                             self.surfCPOrder*(localTiles[1]+1))
#        if oe.shape == 'round':
#            oeContour = [0]
#            oneEdge = [0]
#        else:
#            oeContour = [0]*4
#            oneEdge = [0]*4
#            oeContour[0] = np.array([xBound,
#                                     yBound[0]*np.ones_like(xBound)])  # bottom
#            oeContour[1] = np.array([xBound[-1]*np.ones_like(yBound),
#                                     yBound])  # left
#            oeContour[2] = np.array([np.flip(xBound, 0),
#                                     yBound[-1]*np.ones_like(xBound)])  # top
#            oeContour[3] = np.array([xBound[0]*np.ones_like(yBound),
#                                     np.flip(yBound, 0)])  # right
#
#        for ie, edge in enumerate(oeContour):
#            if oe.shape == 'round':
#                edgeX, edgeY = rX*np.cos(yBound)+cX, rY*np.sin(yBound)+cY
#            else:
#                edgeX = edge[0, :]
#                edgeY = edge[1, :]
#            edgeZ = np.zeros_like(edgeX)
#
#            if oe.isParametric:
#                edgeX, edgeY, edgeZ = oe.xyz_to_param(edgeX, edgeY,
#                                                      edgeZ)
#
#            edgeZ = local_z(edgeX, edgeY)
#            if oe.isParametric:
#                edgeX, edgeY, edgeZ = oe.param_to_xyz(
#                        edgeX, edgeY, edgeZ)
#            edgeBeam = rsources.Beam(nrays=len(edgeX))
#            edgeBeam.x = edgeX
#            edgeBeam.y = edgeY
#            edgeBeam.z = edgeZ
#
#            oe.local_to_global(edgeBeam, is2ndXtal=is2ndXtal)
#            oneEdge[ie] = np.vstack((edgeBeam.x - self.coordOffset[0],
#                                     edgeBeam.y - self.coordOffset[1],
#                                     edgeBeam.z - self.coordOffset[2])).T
#
#        self.oeContour[oe.name] = oneEdge

#    def drawOeContour(self, oe):
#        gl.glEnable(gl.GL_MAP1_VERTEX_3)
#        gl.glLineWidth(self.contourWidth)
#        gl.glColor4f(0.0, 0.0, 0.0, 1.0)
#        cpo = self.surfCPOrder
#        for ie in range(len(self.oeContour[oe.name])):
#            edge = self.oeContour[oe.name][ie]
#            nTiles = self.tiles[0] if ie in [0, 2] else self.tiles[1]
#            nTiles = self.tiles[1]*3 if oe.shape == 'round' else nTiles
#            for tile in range(nTiles+1):
#                gl.glMap1f(gl.GL_MAP1_VERTEX_3,  0, 1,
#                           self.modelToWorld(edge[tile*cpo:(tile+1)*cpo+1, :]))
#                gl.glMapGrid1f(cpo, 0.0, 1.0)
#                gl.glEvalMesh1(gl.GL_LINE, 0, cpo)
#
#        gl.glDisable(gl.GL_MAP1_VERTEX_3)

#    def drawSlitEdges(self, oe):
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
#        gl.glLineWidth(self.contourWidth)
#        gl.glColor4f(0.0, 0.0, 0.0, 1.0)
#        gl.glBegin(gl.GL_QUADS)
#        for edge in self.modelToWorld(np.array(self.slitEdges[oe.name]) -
#                                      np.array(self.coordOffset)):
#            gl.glVertex3f(*edge[0, :])
#            gl.glVertex3f(*edge[1, :])
#            gl.glVertex3f(*edge[3, :])
#            gl.glVertex3f(*edge[2, :])
#
#            gl.glVertex3f(*edge[0, :])
#            gl.glVertex3f(*edge[1, :])
#            gl.glVertex3f(*edge[5, :])
#            gl.glVertex3f(*edge[4, :])
#
#            gl.glVertex3f(*edge[5, :])
#            gl.glVertex3f(*edge[1, :])
#            gl.glVertex3f(*edge[3, :])
#            gl.glVertex3f(*edge[7, :])
#
#            gl.glVertex3f(*edge[4, :])
#            gl.glVertex3f(*edge[5, :])
#            gl.glVertex3f(*edge[7, :])
#            gl.glVertex3f(*edge[6, :])
#
#            gl.glVertex3f(*edge[0, :])
#            gl.glVertex3f(*edge[4, :])
#            gl.glVertex3f(*edge[6, :])
#            gl.glVertex3f(*edge[2, :])
#
#            gl.glVertex3f(*edge[2, :])
#            gl.glVertex3f(*edge[3, :])
#            gl.glVertex3f(*edge[7, :])
#            gl.glVertex3f(*edge[6, :])
#        gl.glEnd()

    def plotAperture(self, oe):
        surfCPOrder = self.surfCPOrder
        gl.glEnable(gl.GL_MAP2_VERTEX_3)
        gl.glEnable(gl.GL_MAP2_NORMAL)
        plotVolume = False

        isBeamStop = oe.isBeamStop

        if oe.shape == 'round':
            r = oe.r
            if isBeamStop:
                limits = [[0, r, 0, 2*np.pi]]
            else:
                # wf = max(r*0.25, 2.5)
                wf = 2*r * self.slitThicknessFraction*0.01
                limits = [[r, r+wf, 0, 2*np.pi]]
            tiles = self.tiles[1] * 5
        else:
            try:
                left, right, bottom, top = oe.spotLimits
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    left, right, bottom, top = 0, 0, 0, 0
            for akind, d in zip(oe.kind, oe.opening):
                if akind.startswith('l'):
                    left = d
                elif akind.startswith('r'):
                    right = d
                elif akind.startswith('b'):
                    bottom = d
                elif akind.startswith('t'):
                    top = d

            if isBeamStop:
                limits = [[left, right, bottom, top]]
            else:
                limits = []
                w = right - left
                h = top - bottom
                # wf = max(min(w, h)*0.5, 2.5)
                wf = min(w, h) * self.slitThicknessFraction*0.01
                for akind, d in zip(oe.kind, oe.opening):
                    if akind.startswith('l'):
                        limits.append([left-wf, left, bottom-wf, top+wf])
                    elif akind.startswith('r'):
                        limits.append([right, right+wf, bottom-wf, top+wf])
                    elif akind.startswith('b'):
                        limits.append([left-wf, right+wf, bottom-wf, bottom])
                    elif akind.startswith('t'):
                        limits.append([left-wf, right+wf, top, top+wf])

            tiles = self.tiles[1]

        if not plotVolume:
            for xMin, xMax, yMin, yMax in limits:
                xGridOe = np.linspace(xMin, xMax, surfCPOrder)
                deltaY = (yMax - yMin) / float(tiles)
                for k in range(tiles):
                    yMinT = yMin + k*deltaY
                    yMaxT = yMinT + deltaY
                    yGridOe = np.linspace(yMinT, yMaxT, surfCPOrder)
                    xv, yv = np.meshgrid(xGridOe, yGridOe)
                    if oe.shape == 'round':
                        xv, yv = xv*np.cos(yv), xv*np.sin(yv)
                    xv = xv.flatten()
                    yv = yv.flatten()

                    gbT = rsources.Beam(nrays=len(xv))
                    gbT.x = xv
                    gbT.y = np.zeros_like(xv)
                    gbT.z = yv

                    gbT.a = np.zeros_like(xv)
                    gbT.b = np.ones_like(xv)
                    gbT.c = np.zeros_like(xv)

                    oe.local_to_global(gbT)

                    for surf in [1, -1]:
                        self.plotCurvedMesh(gbT.x, gbT.y, gbT.z,
                                            gbT.a, gbT.b[:]*surf, gbT.c,
                                            [0, 0, 0])
#        else:
#            self.slitEdges[oe.name] = []
#            for iface, face in enumerate(limits):
#                dT = slitT if iface < 2 else -slitT  # Slit thickness
#                # front
#                xGridOe = np.linspace(face[0], face[1], surfCPOrder)
#                zGridOe = np.linspace(face[2], face[3], surfCPOrder)
#                yGridOe = np.linspace(0, -dT, surfCPOrder)
#                xVert, yVert, zVert = np.meshgrid([face[0], face[1]],
#                                                  [0, -dT],
#                                                  [face[2], face[3]])
#                bladeVertices = np.vstack((xVert.flatten(),
#                                           yVert.flatten(),
#                                           zVert.flatten())).T
#                gbt = rsources.Beam(nrays=8)
#                gbt.x = bladeVertices[:, 0]
#                gbt.y = bladeVertices[:, 1]
#                gbt.z = bladeVertices[:, 2]
#                oe.local_to_global(gbt)
#
#                self.slitEdges[oe.name].append(np.vstack((gbt.x, gbt.y,
#                                                          gbt.z)).T)
#
#                xv, zv = np.meshgrid(xGridOe, zGridOe)
#                xv = xv.flatten()
#                zv = zv.flatten()
#
#                gbT = rsources.Beam(nrays=len(xv))
#                gbT.x = xv
#                gbT.y = np.zeros_like(xv)
#                gbT.z = zv
#
#                gbT.a = np.zeros_like(xv)
#                gbT.b = np.ones_like(xv)
#                gbT.c = np.zeros_like(xv)
#
#                oe.local_to_global(gbT)
#
#                for ysurf in [0, dT]:
#                    nsurf = 1. if (dT > 0 and ysurf != 0) or\
#                        (ysurf == 0 and dT < 0) else -1.
#                    self.plotCurvedMesh(gbT.x, gbT.y, gbT.z,
#                                        gbT.a, gbT.b[:]*nsurf, gbT.c,
#                                        [0, ysurf, 0])
#
#                # side
#                zv, yv = np.meshgrid(zGridOe, yGridOe)
#                zv = zv.flatten()
#                yv = yv.flatten()
#
#                gbT = rsources.Beam(nrays=len(yv))
#                gbT.y = yv
#                gbT.x = np.zeros_like(yv)
#                gbT.z = zv
#
#                gbT.a = np.ones_like(yv)
#                gbT.b = np.zeros_like(yv)
#                gbT.c = np.zeros_like(yv)
#
#                oe.local_to_global(gbT)
#
#                for isurf, xsurf in enumerate([face[0], face[1]]):
#                    nsurf = 1. if isurf == 0 else -1
#                    self.plotCurvedMesh(gbT.x, gbT.y, gbT.z,
#                                        gbT.a[:]*nsurf, gbT.b, gbT.c,
#                                        [xsurf, 0, 0])
#
#                # top
#                xv, yv = np.meshgrid(xGridOe, yGridOe)
#                xv = xv.flatten()
#                yv = yv.flatten()
#
#                gbT = rsources.Beam(nrays=len(yv))
#                gbT.x = xv
#                gbT.y = yv
#                gbT.z = np.zeros_like(xv)
#
#                gbT.a = np.zeros_like(yv)
#                gbT.b = np.zeros_like(yv)
#                gbT.c = np.ones_like(yv)
#
#                oe.local_to_global(gbT)
#
#                for isurf, zsurf in enumerate([face[2], face[3]]):
#                    nsurf = 1. if isurf == 0 else -1
#                    self.plotCurvedMesh(gbT.x, gbT.y, gbT.z,
#                                        gbT.a, gbT.b, gbT.c[:]*nsurf,
#                                        [0, 0, zsurf])

        gl.glDisable(gl.GL_MAP2_VERTEX_3)
        gl.glDisable(gl.GL_MAP2_NORMAL)

    def plotScreen(self, oe, dimensions=None, frameColor=None, plotFWHM=False):
        scAbsZ = np.linalg.norm(oe.z * self.scaleVec)
        scAbsX = np.linalg.norm(oe.x * self.scaleVec)
        if dimensions is not None:
            vScrHW = dimensions[0]
            vScrHH = dimensions[1]
        else:
            vScrHW = self.vScreenSize
            vScrHH = self.vScreenSize

        dX = vScrHW * np.array(oe.x) * self.maxLen / scAbsX
        dZ = vScrHH * np.array(oe.z) * self.maxLen / scAbsZ

        vScreenBody = np.zeros((4, 3))
        vScreenBody[0, :] = vScreenBody[1, :] = oe.center - dX
        vScreenBody[2, :] = vScreenBody[3, :] = oe.center + dX
        vScreenBody[0, :] -= dZ
        vScreenBody[1, :] += dZ
        vScreenBody[2, :] += dZ
        vScreenBody[3, :] -= dZ

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBegin(gl.GL_QUADS)

#        if self.invertColors:
#            gl.glColor4f(0.0, 0.0, 0.0, 0.2)
#        else:
#            gl.glColor4f(1.0, 1.0, 1.0, 0.2)

        for i in range(4):
            gl.glVertex3f(*self.modelToWorld(vScreenBody[i, :] -
                                             self.coordOffset))
        gl.glEnd()

        if frameColor is not None:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
            if self.virtScreen is not None:
                self.virtScreen.frame = vScreenBody
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_NORMALIZE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glLineWidth(2)
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(*frameColor)
            for i in range(4):
                gl.glVertex3f(*self.modelToWorld(vScreenBody[i, :] -
                                                 self.coordOffset))
            gl.glEnd()
            if not self.enableAA:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_NORMALIZE)

        if plotFWHM:
            gl.glLineWidth(1)
            gl.glDisable(gl.GL_LINE_SMOOTH)

            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_NORMALIZE)

            if self.invertColors:
                gl.glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                gl.glColor4f(1.0, 1.0, 1.0, 1.)
            startVec = np.array([0, 1, 0])
            destVec = np.array(oe.y / self.scaleVec)
            rotVec = np.cross(startVec, destVec)
            rotAngle = np.degrees(np.arccos(
                np.dot(startVec, destVec) /
                np.linalg.norm(startVec) / np.linalg.norm(destVec)))
            rotVecGL = np.float32(np.hstack((rotAngle, rotVec)))

            pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
            scr = np.zeros((3, 3))
            for iAx in range(3):
                scr[iAx] = np.array(gl.gluProject(
                    *(self.modelToWorld(vScreenBody[iAx] - self.coordOffset)),
                    model=pModel, proj=pProjection, view=pView))

            vFlip = 2. if scr[0, 1] > scr[1, 1] else 0.
            hFlip = 2. if scr[1, 0] > scr[2, 0] else 0.

            for iAx, text in enumerate(oe.FWHMstr):
                fontScale = self.fontSize / 12500.
                coord = self.modelToWorld(
                    (vScreenBody[iAx + 1] + vScreenBody[iAx + 2]) * 0.5 -
                    self.coordOffset)
                coordShift = np.zeros(3, dtype=np.float32)
                if iAx == 0:  # Horizontal Label
                    coordShift[0] = (hFlip - 1.) * fontScale *\
                        len(text) * 104.76 * 0.5
                    coordShift[2] = fontScale * 200.
                else:  # Vertical Label
                    coordShift[0] = fontScale * 200.
                    coordShift[2] = (vFlip - 1.) * fontScale *\
                        len(text) * 104.76 * 0.5

                gl.glPushMatrix()
                gl.glTranslatef(*coord)
                gl.glRotatef(*rotVecGL)
                gl.glTranslatef(*coordShift)
                gl.glRotatef(180.*(vFlip*0.5), 1, 0, 0)
                gl.glRotatef(180.*(hFlip*0.5), 0, 0, 1)
                if iAx > 0:
                    gl.glRotatef(-90, 0, 1, 0)
                if iAx == 0:  # Horizontal Label to half height
                    gl.glTranslatef(0, 0, -50. * fontScale)
                else:  # Vertical Label to half height
                    gl.glTranslatef(-50. * fontScale, 0, 0)
                gl.glRotatef(90, 1, 0, 0)
                gl.glScalef(fontScale, fontScale, fontScale)
                for symbol in text:
                    gl.glutStrokeCharacter(
                        gl.GLUT_STROKE_MONO_ROMAN, ord(symbol))
                gl.glPopMatrix()
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_NORMALIZE)
            if self.enableAA:
                gl.glEnable(gl.GL_LINE_SMOOTH)

    def plotHemiScreen(self, oe, dimensions=None):
        try:
            rMajor = oe.R
        except:  # analysis:ignore
            rMajor = 1000.

        if dimensions is not None:
            rMinor = dimensions
        else:
            rMinor = self.vScreenSize

        if rMinor > rMajor:
            rMinor = rMajor

        yVec = np.array(oe.x)

        sphereCenter = np.array(oe.center)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if self.invertColors:
            gl.glColor4f(0.0, 0.0, 0.0, 0.2)
        else:
            gl.glColor4f(1.0, 1.0, 1.0, 0.2)

        gl.glEnable(gl.GL_MAP2_VERTEX_3)

        dAngle = np.arctan2(rMinor, rMajor)

        xLimits = [-dAngle + yVec[0], dAngle + yVec[0]]
        yLimits = [-dAngle + yVec[2], dAngle + yVec[2]]

        for i in range(self.tiles[0]):
            deltaX = (xLimits[1] - xLimits[0]) /\
                float(self.tiles[0])
            xGridOe = np.linspace(xLimits[0] + i*deltaX,
                                  xLimits[0] + (i+1)*deltaX,
                                  self.surfCPOrder)
            for k in range(self.tiles[1]):
                deltaY = (yLimits[1] - yLimits[0]) /\
                    float(self.tiles[1])
                yGridOe = np.linspace(yLimits[0] + k*deltaY,
                                      yLimits[0] + (k+1)*deltaY,
                                      self.surfCPOrder)
                xv, yv = np.meshgrid(xGridOe, yGridOe)
                xv = xv.flatten()
                yv = yv.flatten()

                ibp = rsources.Beam(nrays=len(xv))
                ibp.x[:] = sphereCenter[0]
                ibp.y[:] = sphereCenter[1]
                ibp.z[:] = sphereCenter[2]
                ibp.b[:] = yVec[1]
                ibp.a = xv
                ibp.c = yv
                ibp.state[:] = 1

                gbp = oe.expose_global(beam=ibp)

                surfCP = np.vstack((gbp.x - self.coordOffset[0],
                                    gbp.y - self.coordOffset[1],
                                    gbp.z - self.coordOffset[2])).T

                gl.glMap2f(gl.GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                           self.modelToWorld(surfCP.reshape(
                               self.surfCPOrder,
                               self.surfCPOrder, 3)))

                gl.glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
                               self.surfCPOrder, 0.0, 1.0)

                gl.glEvalMesh2(gl.GL_FILL, 0, self.surfCPOrder,
                               0, self.surfCPOrder)

        gl.glDisable(gl.GL_MAP2_VERTEX_3)

    def addLighting(self, pos):
        spot = 60
        exp = 30
        ambient = [0.2, 0.2, 0.2, 1]
        diffuse = [0.5, 0.5, 0.5, 1]
        specular = [1.0, 1.0, 1.0, 1]
        gl.glEnable(gl.GL_LIGHTING)

#        corners = [[-pos, pos, pos, 1], [-pos, -pos, -pos, 1],
#                   [-pos, pos, -pos, 1], [-pos, -pos, pos, 1],
#                   [pos, pos, -pos, 1], [pos, -pos, pos, 1],
#                   [pos, pos, pos, 1], [pos, -pos, -pos, 1]]

        corners = [[0, 0, pos, 1], [0, pos, 0, 1],
                   [pos, 0, 0, 1], [-pos, 0, 0, 1],
                   [0, -pos, 0, 1], [0, 0, -pos, 1]]

        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, 0)
        for iLight in range(len(corners)):
            light = gl.GL_LIGHT0 + iLight
            gl.glEnable(light)
            gl.glLightfv(light, gl.GL_POSITION, corners[iLight])
            gl.glLightfv(light, gl.GL_SPOT_DIRECTION,
                         np.array(corners[len(corners)-iLight-1])/pos)
            gl.glLightfv(light, gl.GL_SPOT_CUTOFF, spot)
            gl.glLightfv(light, gl.GL_SPOT_EXPONENT, exp)
            gl.glLightfv(light, gl.GL_AMBIENT, ambient)
            gl.glLightfv(light, gl.GL_DIFFUSE, diffuse)
            gl.glLightfv(light, gl.GL_SPECULAR, specular)
#            gl.glBegin(gl.GL_LINES)
#            glVertex4f(*corners[iLight])
#            glVertex4f(*corners[len(corners)-iLight-1])
#            gl.glEnd()

#    def toggleHelp(self):
#        self.showHelp = not self.showHelp
#        self.glDraw()

#    def drawHelp(self):
#        hHeight = 300
#        hWidth = 500
#        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
#        gl.glViewport(0, self.viewPortGL[3]-hHeight, hWidth, hHeight)
#        gl.glMatrixMode(gl.GL_PROJECTION)
#        gl.glLoadIdentity()
#        gl.glOrtho(-1, 1, -1, 1, -1, 1)
#        gl.glMatrixMode(gl.GL_MODELVIEW)
#        gl.glLoadIdentity()
#
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#        gl.glBegin(gl.GL_QUADS)
#
#        if self.invertColors:
#            gl.glColor4f(1.0, 1.0, 1.0, 0.9)
#        else:
#            gl.glColor4f(0.0, 0.0, 0.0, 0.9)
#        backScreen = [[1, 1], [1, -1],
#                      [-1, -1], [-1, 1]]
#        for corner in backScreen:
#                gl.glVertex3f(corner[0], corner[1], 0)
#
#        gl.glEnd()
#
#        if self.invertColors:
#            gl.glColor4f(0.0, 0.0, 0.0, 1.0)
#        else:
#            gl.glColor4f(1.0, 1.0, 1.0, 1.0)
#        gl.glLineWidth(3)
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
#        gl.glBegin(gl.GL_QUADS)
#        backScreen = [[1, 1], [1, -1],
#                      [-1, -1], [-1, 1]]
#        for corner in backScreen:
#                gl.glVertex3f(corner[0], corner[1], 0)
#        gl.glEnd()
#
#        helpList = [
#            'F1: Open/Close this help window',
#            'F3: Add/Remove Virtual Screen',
#            'F4: Dock/Undock xrtGlow if launched from xrtQook',
#            'F5/F6: Quick Save/Load Scene']
#        if hasattr(self, 'generator'):
#            helpList += ['F7: Start recording movie']
#        helpList += [
#            'LeftMouse: Rotate the Scene',
#            'SHIFT+LeftMouse: Translate in perpendicular to the shortest view axis',  # analysis:ignore
#            'ALT+LeftMouse: Translate in parallel to the shortest view axis',  # analysis:ignore
#            'CTRL+LeftMouse: Drag Virtual Screen',
#            'ALT+WheelMouse: Scale Virtual Screen',
#            'CTRL+SHIFT+LeftMouse: Translate the Beamline around Virtual Screen',  # analysis:ignore
#            '                      (with Beamline along the longest view axis)',  # analysis:ignore
#            'CTRL+ALT+LeftMouse: Translate the Beamline around Virtual Screen',  # analysis:ignore
#            '                      (with Beamline along the shortest view axis)',  # analysis:ignore
#            'CTRL+T: Toggle Virtual Screen orientation (vertical/normal to the beam)',  # analysis:ignore
#            'WheelMouse: Zoom the Beamline',
#            'CTRL+WheelMouse: Zoom the Scene']
#        for iLine, text in enumerate(helpList):
#            self.drawText([-1. + 0.05,
#                           1. - 2. * (iLine + 1) / float(len(helpList)+1), 0],
#                          text, True)
#
#        gl.glFlush()
#        gl.glViewport(*pView)

    def drawCone(self, z, r, nFacets, color):
        phi = np.linspace(0, 2*np.pi, nFacets)
        xp = r * np.cos(phi)
        yp = r * np.sin(phi)
        base = np.vstack((xp, yp, np.zeros_like(xp)))
        coneVertices = np.hstack((np.array([0, 0, z]).reshape(3, 1),
                                  base)).T
        gridColor = np.zeros((len(coneVertices), 4))
        gridColor[:, color] = 1
        gridColor[:, 3] = 0.75
        gridArray = gl.vbo.VBO(np.float32(coneVertices))
        gridArray.bind()
        gl.glVertexPointerf(gridArray)
        gridColorArray = gl.vbo.VBO(np.float32(gridColor))
        gridColorArray.bind()
        gl.glColorPointerf(gridColorArray)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(gridArray))
        gridArray.unbind()
        gridColorArray.unbind()

    def drawLocalAxes(self, oe, is2ndXtal):
        def drawArrow(color, arrowArray, yText='hkl'):
            gridColor = np.zeros((len(arrowArray) - 1, 4))
            gridColor[:, 3] = 0.75
            if color == 4:
                gridColor[:, 0] = 1
                gridColor[:, 1] = 1
            elif color == 5:
                gridColor[:, 0] = 1
                gridColor[:, 1] = 0.5
            else:
                gridColor[:, color] = 1
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gridArray = gl.vbo.VBO(np.float32(arrowArray[1:, :]))
            gridArray.bind()
            gl.glVertexPointerf(gridArray)
            gridColorArray = gl.vbo.VBO(np.float32(gridColor))
            gridColorArray.bind()
            gl.glColorPointerf(gridColorArray)
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(gridArray))
            gridArray.unbind()
            gridColorArray.unbind()
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glBegin(gl.GL_LINES)
            colorVec = [0, 0, 0, 0.75]
            if color == 4:
                colorVec[0] = 1
                colorVec[1] = 1
            elif color == 5:
                colorVec[0] = 1
                colorVec[1] = 0.5
            else:
                colorVec[color] = 1
            gl.glColor4f(*colorVec)
            gl.glVertex3f(*arrowArray[0, :])
            gl.glVertex3f(*arrowArray[1, :])
            gl.glEnd()
            gl.glColor4f(*colorVec)
            gl.glRasterPos3f(*arrowArray[1, :])
            if color == 0:
                axSymb = 'Z'
            elif color == 1:
                axSymb = 'Y'
            elif color == 2:
                axSymb = 'X'
            elif color == 4:
                axSymb = yText
            else:
                axSymb = ''

            for symbol in "  {}".format(axSymb):
                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
            gl.glDisable(gl.GL_LINE_SMOOTH)

        z, r, nFacets = 0.25, 0.02, 20
        phi = np.linspace(0, 2*np.pi, nFacets)
        xp = np.insert(r * np.cos(phi), 0, [0., 0.])
        yp = np.insert(r * np.sin(phi), 0, [0., 0.])
        zp = np.insert(z*0.8*np.ones_like(phi), 0, [0., z])

        crPlaneZ = None
        yText = None
        if hasattr(oe, 'local_n'):
            material = None
            if hasattr(oe, 'material'):
                material = oe.material
            if is2ndXtal:
                zExt = '2'
                if hasattr(oe, 'material2'):
                    material = oe.material2
            else:
                zExt = '1' if hasattr(oe, 'local_n1') else ''
            if raycing.is_sequence(material):
                material = material[oe.curSurface]

            local_n = getattr(oe, 'local_n{}'.format(zExt))
            normals = local_n(0, 0)
            if len(normals) > 3:
                crPlaneZ = np.array(normals[:3], dtype=float)
                crPlaneZ /= np.linalg.norm(crPlaneZ)
                if material not in [None, 'None']:
                    if hasattr(material, 'hkl'):
                        hklSeparator = ',' if np.any(np.array(
                            material.hkl) >= 10) else ''
                        yText = '[{0[0]}{1}{0[1]}{1}{0[2]}]'.format(
                            list(material.hkl), hklSeparator)
#                        yText = '{}'.format(list(material.hkl))

        cb = rsources.Beam(nrays=nFacets+2)
        cb.a[:] = cb.b[:] = cb.c[:] = 0.
        cb.a[0] = cb.b[1] = cb.c[2] = 1.

        if crPlaneZ is not None:  # Adding asymmetric crystal orientation
            asAlpha = np.arccos(crPlaneZ[2])
            acpX = np.array([0., 0., 1.], dtype=float) if asAlpha == 0 else\
                np.cross(np.array([0., 0., 1.], dtype=float), crPlaneZ)
            acpX /= np.linalg.norm(acpX)

            cb.a[3] = acpX[0]
            cb.b[3] = acpX[1]
            cb.c[3] = acpX[2]

        cb.state[:] = 1

        if isinstance(oe, (rscreens.HemisphericScreen, rscreens.Screen)):
            cb.x[:] += oe.center[0]
            cb.y[:] += oe.center[1]
            cb.z[:] += oe.center[2]
            oeNormX = oe.x
            oeNormY = oe.y
        else:
            if is2ndXtal:
                oe.local_to_global(cb, is2ndXtal=is2ndXtal)
            else:
                oe.local_to_global(cb)
            oeNormX = np.array([cb.a[0], cb.b[0], cb.c[0]])
            oeNormY = np.array([cb.a[1], cb.b[1], cb.c[1]])

        scNormX = oeNormX * self.scaleVec
        scNormY = oeNormY * self.scaleVec

        scNormX /= np.linalg.norm(scNormX)
        scNormY /= np.linalg.norm(scNormY)
        scNormZ = np.cross(scNormX, scNormY)
        scNormZ /= np.linalg.norm(scNormZ)

        for iAx in range(3):
            if iAx == 0:
                xVec = scNormX
                yVec = scNormY
                zVec = scNormZ
            elif iAx == 2:
                xVec = scNormY
                yVec = scNormZ
                zVec = scNormX
            else:
                xVec = scNormZ
                yVec = scNormX
                zVec = scNormY

            dX = xp[:, np.newaxis] * xVec
            dY = yp[:, np.newaxis] * yVec
            dZ = zp[:, np.newaxis] * zVec
            coneCP = self.modelToWorld(np.vstack((
                cb.x - self.coordOffset[0], cb.y - self.coordOffset[1],
                cb.z - self.coordOffset[2])).T) + dX + dY + dZ
            drawArrow(iAx, coneCP)

        if crPlaneZ is not None:  # drawAsymmetricPlane:
            crPlaneX = np.array([cb.a[3], cb.b[3], cb.c[3]])
            crPlaneNormX = crPlaneX * self.scaleVec
            crPlaneNormX /= np.linalg.norm(crPlaneNormX)
            crPlaneNormZ = self.rotateVecQ(
                scNormZ, self.vecToQ(crPlaneNormX, asAlpha))
            crPlaneNormZ /= np.linalg.norm(crPlaneNormZ)
            crPlaneNormY = np.cross(crPlaneNormX, crPlaneNormZ)
            crPlaneNormY /= np.linalg.norm(crPlaneNormY)

            color = 4

            dX = xp[:, np.newaxis] * crPlaneNormX
            dY = yp[:, np.newaxis] * crPlaneNormY
            dZ = zp[:, np.newaxis] * crPlaneNormZ
            coneCP = self.modelToWorld(np.vstack((
                cb.x - self.coordOffset[0], cb.y - self.coordOffset[1],
                cb.z - self.coordOffset[2])).T) + dX + dY + dZ
            drawArrow(color, coneCP, yText)

    def drawDirectionAxes(self):
        arrowSize = 0.05
        axisLen = 0.1
        tLen = (arrowSize + axisLen) * 2
        gl.glLineWidth(1.)

        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(0, 0, int(150*self.aspect), 150)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        if self.perspectiveEnabled:
            gl.gluPerspective(60, self.aspect, 0.001, 10)
        else:
            gl.glOrtho(-tLen*self.aspect, tLen*self.aspect, -tLen, tLen, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.gluLookAt(.5, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        self.rotateZYX()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    2-iAx == self.visibleAxes[2]):
                gl.glPushMatrix()
                trVec = np.zeros(3, dtype=np.float32)
                trVec[2-iAx] = axisLen
                gl.glTranslatef(*trVec)
                if iAx == 1:
                    gl.glRotatef(-90, 1.0, 0.0, 0.0)
                elif iAx == 2:
                    gl.glRotatef(90, 0.0, 1.0, 0.0)
                self.drawCone(arrowSize, 0.02, 20, iAx)
                gl.glPopMatrix()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        gl.glBegin(gl.GL_LINES)
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    2-iAx == self.visibleAxes[2]):
                colorVec = [0, 0, 0, 0.75]
                colorVec[iAx] = 1
                gl.glColor4f(*colorVec)
                gl.glVertex3f(0, 0, 0)
                trVec = np.zeros(3, dtype=np.float32)
                trVec[2-iAx] = axisLen
                gl.glVertex3f(*trVec)
                gl.glColor4f(*colorVec)
        gl.glEnd()

        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 2):
            gl.glColor4f(1, 0, 0, 1)
            gl.glRasterPos3f(0, 0, axisLen*1.5)
            for symbol in "  {} (mm)".format('Z'):
                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 1):
            gl.glColor4f(0, 0.75, 0, 1)
            gl.glRasterPos3f(0, axisLen*1.5, 0)
            for symbol in "  {} (mm)".format('Y'):
                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 0):
            gl.glColor4f(0, 0.5, 1, 1)
            gl.glRasterPos3f(axisLen*1.5, 0, 0)
            for symbol in "  {} (mm)".format('X'):
                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
#        gl.glFlush()
        gl.glViewport(*pView)
        gl.glColor4f(1, 1, 1, 1)
        gl.glDisable(gl.GL_LINE_SMOOTH)

    def initializeGL(self):
        gl.glutInit()
        gl.glutInitDisplayMode(gl.GLUT_RGBA | gl.GLUT_DOUBLE | gl.GLUT_DEPTH)
        gl.glViewport(*self.viewPortGL)

    def resizeGL(self, widthInPixels, heightInPixels):
        self.viewPortGL = [0, 0, widthInPixels, heightInPixels]
        gl.glViewport(*self.viewPortGL)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

    def populateVScreen(self):
        if any([prop is None for prop in [self.virtBeam,
                                          self.selColorMax,
                                          self.selColorMin]]):
            return
        startBeam = self.virtBeam
        try:
            vColorArray = self.getColor(startBeam)
        except AttributeError:
            if _DEBUG_:
                raise
            else:
                return

        good = (startBeam.state == 1) | (startBeam.state == 2)
        intensity = startBeam.Jss + startBeam.Jpp
        intensityAll = intensity / np.max(intensity[good])

        good = np.logical_and(good, intensityAll >= self.cutoffI)
        goodC = np.logical_and(
            vColorArray <= self.selColorMax,
            vColorArray >= self.selColorMin)

        good = np.logical_and(good, goodC)
        if len(vColorArray[good]) == 0:
            return
        self.globalColorIndex = good if self.vScreenForColors else None

        if self.globalNorm:
            alphaMax = 1.
        else:
            if len(intensity[good]) > 0:
                alphaMax = np.max(intensity[good])
            else:
                alphaMax = 1.
        alphaMax = alphaMax if alphaMax != 0 else 1.
#        alphaDots = intensity[good].T / alphaMax
#        colorsDots = np.array(vColorArray[good]).T
        alphaDots = intensity.T / alphaMax
        colorsDots = np.array(vColorArray).T
        if self.colorMin == self.colorMax:
            if self.colorMax == 0:  # and self.colorMin == 0 too
                self.colorMin, self.colorMax = -0.1, 0.1
            else:
                self.colorMin = self.colorMax * 0.99
                self.colorMax *= 1.01
        colorsDots = colorFactor * (colorsDots-self.colorMin) /\
            (self.colorMax-self.colorMin)
        depthDots = copy.deepcopy(colorsDots[good]) * self.depthScaler

        colorsDots = np.dstack((colorsDots,
                                np.ones_like(alphaDots)*colorSaturation,
                                alphaDots if self.iHSV else
                                np.ones_like(alphaDots)))

        deltaY = self.virtScreen.y * depthDots[:, np.newaxis]

        vertices = np.array(
            startBeam.x[good] - deltaY[:, 0] - self.coordOffset[0])
        vertices = np.vstack((vertices, np.array(
            startBeam.y[good] - deltaY[:, 1] - self.coordOffset[1])))
        vertices = np.vstack((vertices, np.array(
            startBeam.z[good] - deltaY[:, 2] - self.coordOffset[2])))
        self.virtDotsArray = vertices.T

        colorsRGBDots = np.squeeze(mpl.colors.hsv_to_rgb(colorsDots))
        if self.globalNorm and len(alphaDots[good]) > 0:
            alphaMax = np.max(alphaDots[good])
        else:
            alphaMax = 1.
        alphaColorDots = np.array([alphaDots / alphaMax]).T
        if self.vScreenForColors:
            self.globalColorArray = np.float32(np.hstack([colorsRGBDots,
                                                          alphaColorDots]))
        self.virtDotsColor = np.float32(np.hstack([colorsRGBDots[good],
                                                   alphaColorDots[good]]))

        histogram = np.histogram(np.array(
            vColorArray[good]),
            range=(self.colorMin, self.colorMax),
            weights=intensity[good],
            bins=100)
        self.histogramUpdated.emit(histogram)
        locBeam = self.virtScreen.expose(self.virtScreen.beamToExpose)
        lbi = intensity[good]
        self.virtScreen.FWHMstr = []
        for axis in ['x', 'z']:
            goodlb = getattr(locBeam, axis)[good]
            histAxis = np.histogram(goodlb, weights=lbi, bins=100)
            hMax = np.max(histAxis[0])
            hNorm = histAxis[0] / hMax
            topEl = np.where(hNorm >= 0.5)[0]
            fwhm = np.abs(histAxis[1][topEl[0]] - histAxis[1][topEl[-1]])

            order = np.floor(np.log10(fwhm)) if fwhm > 0 else -10
            if order >= 2:
                units = "m"
                mplier = 1e-3
            elif order >= -1:
                units = "mm"
                mplier = 1.
            elif order >= -4:
                units = "um"
                mplier = 1e3
            else:  # order >= -7:
                units = "nm"
                mplier = 1e6

            self.virtScreen.FWHMstr.append(
                "FWHM({0}) = {1:.3f}{2}".format(
                    str(axis).upper(), fwhm*mplier, units))

    def createVScreen(self):
        try:
            self.virtScreen = rscreens.Screen(
                bl=list(self.oesList.values())[0][0].bl)
            self.virtScreen.center = self.worldToModel(np.array([0, 0, 0])) +\
                self.coordOffset
            self.positionVScreen()
            if self.vScreenForColors:
                self.populateVerticesOnly(self.segmentModel)
            self.glDraw()
        except:  # analysis:ignore
            if _DEBUG_:
                raise
            else:
                self.clearVScreen()

    def positionVScreen(self):
        if self.virtScreen is None:
            return
        cntr = self.virtScreen.center
        tmpDist = 1e12
        totalDist = 1e12
        cProj = None

        for segment in self.arrayOfRays[0]:
            if segment[3] is None:
                continue
            try:
                beamStartTmp = self.beamsDict[segment[1]]
                beamEndTmp = self.beamsDict[segment[3]]

                bStart0 = beamStartTmp.wCenter
                bEnd0 = beamEndTmp.wCenter

                beam0 = bEnd0 - bStart0
                # Finding the projection of the VScreen.center on segments
                cProjTmp = bStart0 + np.dot(cntr-bStart0, beam0) /\
                    np.dot(beam0, beam0) * beam0
                s = 0
                for iDim in range(3):
                    s += np.floor(np.abs(np.sign(cProjTmp[iDim] -
                                                 bStart0[iDim]) +
                                         np.sign(cProjTmp[iDim] -
                                                 bEnd0[iDim]))*0.6)

                dist = np.linalg.norm(cProjTmp-cntr)
                if dist < tmpDist:
                    if s == 0:
                        tmpDist = dist
                        beamStart0 = beamStartTmp
                        bStartC = bStart0
                        bEndC = bEnd0
                        cProj = cProjTmp
                    else:
                        if np.linalg.norm(bStart0-cntr) < totalDist:
                            totalDist = np.linalg.norm(bStart0-cntr)
                            self.virtScreen.center = cProjTmp
                            self.virtScreen.beamStart = bStart0
                            self.virtScreen.beamEnd = bEnd0
                            self.virtScreen.beamToExpose = beamStartTmp
            except:  # analysis:ignore
                if _DEBUG_:
                    raise
                else:
                    continue

        if cProj is not None:
            self.virtScreen.center = cProj
            self.virtScreen.beamStart = bStartC
            self.virtScreen.beamEnd = bEndC
            self.virtScreen.beamToExpose = beamStart0

        if self.isVirtScreenNormal:
            vsX = [self.virtScreen.beamToExpose.b[0],
                   -self.virtScreen.beamToExpose.a[0], 0]
            vsY = [self.virtScreen.beamToExpose.a[0],
                   self.virtScreen.beamToExpose.b[0],
                   self.virtScreen.beamToExpose.c[0]]
            vsZ = np.cross(vsX/np.linalg.norm(vsX),
                           vsY/np.linalg.norm(vsY))
        else:
            vsX = 'auto'
            vsZ = 'auto'
        self.virtScreen.set_orientation(vsX, vsZ)
        try:
            self.virtBeam = self.virtScreen.expose_global(
                self.virtScreen.beamToExpose)
            self.populateVScreen()
        except:  # analysis:ignore
            self.clearVScreen()

    def toggleVScreen(self):
        if self.virtScreen is None:
            self.createVScreen()
        else:
            self.clearVScreen()

    def clearVScreen(self):
        self.virtScreen = None
        self.virtBeam = None
        self.virtDotsArray = None
        self.virtDotsColor = None
        if self.globalColorIndex is not None:
            self.globalColorIndex = None
            self.populateVerticesOnly(self.segmentModel)
        self.histogramUpdated.emit((None, None))
        self.glDraw()

    def switchVScreenTilt(self):
        self.isVirtScreenNormal = not self.isVirtScreenNormal
        self.positionVScreen()
        self.glDraw()

    def mouseMoveEvent(self, mEvent):
        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
        mouseX = mEvent.x()
        mouseY = pView[3] - mEvent.y()
        ctrlOn = bool(int(mEvent.modifiers()) & int(qt.ControlModifier))
        altOn = bool(int(mEvent.modifiers()) & int(qt.AltModifier))
        shiftOn = bool(int(mEvent.modifiers()) & int(qt.ShiftModifier))

        if mEvent.buttons() == qt.LeftButton:
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.gluLookAt(self.cameraPos[0], self.cameraPos[1],
                         self.cameraPos[2],
                         self.cameraTarget[0], self.cameraTarget[1],
                         self.cameraTarget[2],
                         0.0, 0.0, 1.0)
            self.rotateZYX()
            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()

            if self.perspectiveEnabled:
                gl.gluPerspective(self.cameraAngle, self.aspect, 0.01, 100)
            else:
                orthoView = self.cameraPos[0]*0.45
                gl.glOrtho(-orthoView*self.aspect, orthoView*self.aspect,
                           -orthoView, orthoView, -100, 100)
            pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)

            if mEvent.modifiers() == qt.NoModifier:
                self.rotations[2][0] += np.float32(
                    self.signs[2][1] *
                    (mouseX - self.prevMPos[0]) * 36. / 90.)
                self.rotations[1][0] -= np.float32(
                    (mouseY - self.prevMPos[1]) * 36. / 90.)
                for ax in range(2):
                    if self.rotations[self.visibleAxes[ax+1]][0] > 180:
                        self.rotations[self.visibleAxes[ax+1]][0] -= 360
                    if self.rotations[self.visibleAxes[ax+1]][0] < -180:
                        self.rotations[self.visibleAxes[ax+1]][0] += 360
                self.updateQuats()
                self.rotationUpdated.emit(self.rotations)

            elif shiftOn:
                for iDim in range(2):
                    mStart = np.zeros(3)
                    mEnd = np.zeros(3)
                    mEnd[self.visibleAxes[iDim]] = 1.
#                    mEnd = -1 * mStart
                    pStart = np.array(gl.gluProject(
                        *mStart, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pEnd = np.array(gl.gluProject(
                        *mEnd, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pScr = np.array([mouseX, mouseY])
                    prevPScr = np.array(self.prevMPos)
                    bDir = pEnd - pStart
                    pProj = pStart + np.dot(pScr - pStart, bDir) /\
                        np.dot(bDir, bDir) * bDir
                    pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
                        np.dot(bDir, bDir) * bDir
                    self.tVec[self.visibleAxes[iDim]] += np.dot(
                        pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
                        self.maxLen / self.scaleVec[self.visibleAxes[iDim]]
                    if ctrlOn and self.virtScreen is not None:
                        self.virtScreen.center[self.visibleAxes[iDim]] -=\
                            np.dot(
                            pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
                            self.maxLen / self.scaleVec[self.visibleAxes[iDim]]
                if ctrlOn and self.virtScreen is not None:
                    v0 = self.virtScreen.center
                    self.positionVScreen()
                    self.tVec -= self.virtScreen.center - v0

            elif altOn:
                mStart = np.zeros(3)
                mEnd = np.zeros(3)
                mEnd[self.visibleAxes[2]] = 1.
#                    mEnd = -1 * mStart
                pStart = np.array(gl.gluProject(
                    *mStart, model=pModel, proj=pProjection,
                    view=pView)[:-1])
                pEnd = np.array(gl.gluProject(
                    *mEnd, model=pModel, proj=pProjection,
                    view=pView)[:-1])
                pScr = np.array([mouseX, mouseY])
                prevPScr = np.array(self.prevMPos)
                bDir = pEnd - pStart
                pProj = pStart + np.dot(pScr - pStart, bDir) /\
                    np.dot(bDir, bDir) * bDir
                pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
                    np.dot(bDir, bDir) * bDir
                self.tVec[self.visibleAxes[2]] += np.dot(
                    pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
                    self.maxLen / self.scaleVec[self.visibleAxes[2]]
                if ctrlOn and self.virtScreen is not None:
                    self.virtScreen.center[self.visibleAxes[2]] -=\
                        np.dot(pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
                        self.maxLen / self.scaleVec[self.visibleAxes[2]]
                    v0 = self.virtScreen.center
                    self.positionVScreen()
                    self.tVec -= self.virtScreen.center - v0

            elif ctrlOn:
                if self.virtScreen is not None:

                    worldPStart = self.modelToWorld(
                        self.virtScreen.beamStart - self.coordOffset)
                    worldPEnd = self.modelToWorld(
                        self.virtScreen.beamEnd - self.coordOffset)

                    worldBDir = worldPEnd - worldPStart

                    normPEnd = worldPStart + np.dot(
                        np.ones(3) - worldPStart, worldBDir) /\
                        np.dot(worldBDir, worldBDir) * worldBDir

                    normPStart = worldPStart + np.dot(
                        -1. * np.ones(3) - worldPStart, worldBDir) /\
                        np.dot(worldBDir, worldBDir) * worldBDir

                    normBDir = normPEnd - normPStart
                    normScale = np.sqrt(np.dot(normBDir, normBDir) /
                                        np.dot(worldBDir, worldBDir))

                    if np.dot(normBDir, worldBDir) < 0:
                        normPStart, normPEnd = normPEnd, normPStart

                    pStart = np.array(gl.gluProject(
                        *normPStart, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pEnd = np.array(gl.gluProject(
                        *normPEnd, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pScr = np.array([mouseX, mouseY])
                    prevPScr = np.array(self.prevMPos)
                    bDir = pEnd - pStart
                    pProj = pStart + np.dot(pScr - pStart, bDir) /\
                        np.dot(bDir, bDir) * bDir
                    pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
                        np.dot(bDir, bDir) * bDir
                    self.virtScreen.center += normScale * np.dot(
                        pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
                        (self.virtScreen.beamEnd - self.virtScreen.beamStart)
                    self.positionVScreen()

            self.glDraw()
        self.prevMPos[0] = mouseX
        self.prevMPos[1] = mouseY

    def wheelEvent(self, wEvent):
        ctrlOn = bool(int(wEvent.modifiers()) & int(qt.ControlModifier))
        altOn = bool(int(wEvent.modifiers()) & int(qt.AltModifier))
        if qt.QtName == "PyQt4":
            deltaA = wEvent.delta()
        else:
            deltaA = wEvent.angleDelta().y() + wEvent.angleDelta().x()

        if deltaA > 0:
            if altOn:
                self.vScreenSize *= 1.1
            elif ctrlOn:
                self.cameraPos *= 0.9
            else:
                self.scaleVec *= 1.1
        else:
            if altOn:
                self.vScreenSize *= 0.9
            elif ctrlOn:
                self.cameraPos *= 1.1
            else:
                self.scaleVec *= 0.9
        if not ctrlOn:
            self.scaleUpdated.emit(self.scaleVec)
        self.glDraw()

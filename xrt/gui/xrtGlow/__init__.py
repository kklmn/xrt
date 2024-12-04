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
   :align: right
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

from matplotlib import pyplot as plt
# import inspect
import re
import copy
# import time
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as scprot
from collections import OrderedDict
import freetype as ft
from matplotlib import font_manager

from ...backends import raycing
from ...backends.raycing import sources as rsources
from ...backends.raycing import screens as rscreens
from ...backends.raycing import oes as roes
from ...backends.raycing import apertures as rapertures
from ...backends.raycing import materials as rmats
from ..commons import qt
from ..commons import gl
from ...plotter import colorFactor, colorSaturation
_DEBUG_ = True  # If False, exceptions inside the module are ignored
MAXRAYS = 500000

def setVertexBuffer(data_array, dim_vertex, program, shader_str, size=None,
                    oldVBO=None, usage_hint=qt.QOpenGLBuffer.DynamicDraw,
                    is_int=False):
    # slightly modified code from
    # https://github.com/Upcios/PyQtSamples/blob/master/PyQt5/opengl/triangle_simple/main.py
    if oldVBO is not None:
        vbo = oldVBO
    else:
        vbo = qt.QOpenGLBuffer(qt.QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.setUsagePattern(usage_hint)

    dType = np.int32 if is_int else np.float32
    glType = gl.GL_UNSIGNED_INT if is_int else gl.GL_FLOAT
    vertices = np.array(data_array, dType)

    vbo.bind()

    if oldVBO is not None:
        vbo.write(0, vertices, vertices.nbytes)
    else:
        bSize = size if size is not None else vertices.nbytes
        vbo.allocate(vertices, bSize)
        attr_loc = program.attributeLocation(shader_str)
        program.enableAttributeArray(attr_loc)
        program.setAttributeBuffer(attr_loc, glType, 0, dim_vertex)

    vbo.release()
    return vbo

def create_qt_buffer(data, isIndex=False,
                     usage_hint=qt.QOpenGLBuffer.DynamicDraw):
    """Create and populate a QOpenGLBuffer."""
    bufferType = qt.QOpenGLBuffer.IndexBuffer if isIndex else\
        qt.QOpenGLBuffer.VertexBuffer
    buffer = qt.QOpenGLBuffer(bufferType)
    buffer.create()
    buffer.setUsagePattern(usage_hint)
    buffer.bind()
    data = np.array(data, np.uint32 if isIndex else np.float32)
    buffer.allocate(data.tobytes(), data.nbytes)
    buffer.release()
    return buffer


#def setIndexBuffer(data_array, oldIBO=None):
#    # slightly modified code from
#    # https://github.com/Upcios/PyQtSamples/blob/master/PyQt5/opengl/triangle_simple/main.py
#    if oldIBO is None:
#        ibo = qt.QOpenGLBuffer(qt.QOpenGLBuffer.IndexBuffer)
#        ibo.create()
#    else:
#        ibo = oldIBO
#    ibo.bind()
#    indices = np.array(data_array, dtype=np.ushort)
#    ibo.allocate(indices, indices.nbytes)
#    ibo.release()
#    return ibo


def generate_hsv_texture(width, s, v):
    h = np.linspace(0., 1., width, endpoint=False)
    hsv_data = mpl.colors.hsv_to_rgb(np.vstack(
            (h, s*np.ones_like(h), v*np.ones_like(h))).T)
    return (hsv_data * 255).astype(np.uint8)


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
                (0.2, 2., 0.75, 3.))):
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
            aaCheckBox.setChecked(iCB in [1])
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
            image = self.customGlWidget.grabFramebuffer()
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
            self.customGlWidget.updateCutOffI(np.float32(value))
#            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
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


class xrtGlWidget(qt.QOpenGLWidget):
    rotationUpdated = qt.Signal(np.ndarray)
    scaleUpdated = qt.Signal(np.ndarray)
    histogramUpdated = qt.Signal(tuple)

    def __init__(self, parent, arrayOfRays, modelRoot, oesList, b2els, signal):
        super().__init__(parent=parent)
        self.hsvTex = generate_hsv_texture(512, s=1.0, v=1.0)
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
        self.linesDepthTest = False
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
#        self.populateVerticesArray(modelRoot)

        self.drawGrid = True
        self.fineGridEnabled = False
        self.showOeLabels = False
        self.aPos = [0.9, 0.9, 0.9]
        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])
        self.coordinateGridLineWidth = 1
        self.cBoxLineWidth = 1
#        self.fixedFontType = 'GLUT_BITMAP_TIMES_ROMAN'
        self.fixedFontType = 'GLUT_BITMAP_HELVETICA'
        self.fixedFontSize = '12'  # 10, 12, 18 for Helvetica; 10, 24 for Roman
        self.fixedFont = None #getattr(gl, "{0}_{1}".format(self.fixedFontType,
#                                                      self.fixedFontSize))
        self.useScalableFont = False
        self.fontSize = 5
        self.scalableFontType = None #gl.GLUT_STROKE_ROMAN
#        self.scalableFontType = gl.GLUT_STROKE_MONO_ROMAN
        self.scalableFontWidth = 1
        self.useFontAA = False
        self.tVec = np.array([0., 0., 0.])
#        self.cameraTarget = [0., 0., 0.]
#        self.cameraPos = np.float32([3.5, 0., 0.])
#
#        self.upVec = qt.QVector3D(0., 0., 1.)

        self.cameraTarget = qt.QVector3D(0., 0., 0.)
        self.cameraPos = qt.QVector3D(3.5, 0, 0)
        self.upVec = qt.QVector3D(0., 0., 1.)

        self.mView = qt.QMatrix4x4()
        self.mView.lookAt(self.cameraPos,
                          self.cameraTarget,
                          self.upVec)

        self.mProj = qt.QMatrix4x4()
        self.mProj.perspective(self.cameraAngle, self.aspect, 0.01, 1000)

        self.mModScale = qt.QMatrix4x4()
        self.mModTrans = qt.QMatrix4x4()
        self.mModScale.scale(*(self.scaleVec/self.maxLen))
        self.mModTrans.translate(*(self.tVec-self.coordOffset))
        self.mMod = self.mModScale*self.mModTrans

        self.mModAx = qt.QMatrix4x4()
        self.mModAx.setToIdentity()

#        self.isEulerian = False

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
        self.beamVAO = dict()
        self.selectableOEs = {}
        self.selectedOE = 0
        self.isColorAxReady = False
        self.makeCurrent()

#        self.isEulerian = False
#        self.rotations = np.float32([[0., 1., 0., 0.],
#                                     [0., 0., 1., 0.],
#                                     [0., 0., 0., 1.]])
#        self.textOrientation = [0.5, 0.5, 0.5, 0.5]
#        self.updateQuats()
#        pModelT = np.identity(4)
#        self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
#        self.signs = np.ones_like(pModelT)
#        self.invertColors = False
#        self.showHelp = False
#        self.selectableOEs = {}
#        self.selectedOE = 0
#        self.glDraw()

    def init_shaders(self):
        shaderBeam = qt.QOpenGLShaderProgram()
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, Beam3D.vertex_source)
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Geometry, Beam3D.geometry_source)
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, Beam3D.fragment_source)
        if not shaderBeam.link():
            print("Linking Error", str(shaderBeam.log()))
            print('shaderBeam: Failed to link dummy renderer shader!')
        self.shaderBeam = shaderBeam

        shaderFootprint = qt.QOpenGLShaderProgram()
        shaderFootprint.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, Beam3D.vertex_source_point)
        shaderFootprint.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, Beam3D.fragment_source_point)
        if not shaderFootprint.link():
            print("Linking Error", str(shaderFootprint.log()))
            print('shaderFootprint: Failed to link dummy renderer shader!')
        self.shaderFootprint = shaderFootprint

#        shaderHist = qt.QOpenGLShaderProgram()
#        shaderHist.addShaderFromSourceCode(
#                qt.QOpenGLShader.Compute, Beam3D.compute_source)
#        if not shaderHist.link():
#            print("Linking Error", str(shaderHist.log()))
#            print('shaderHist: Failed to link dummy renderer shader!')
#        self.shaderHist = shaderHist
#        print("shaderHist")

        shaderMesh = qt.QOpenGLShaderProgram()
        shaderMesh.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, OEMesh3D.vertex_source)
        shaderMesh.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, OEMesh3D.fragment_source)
        if not shaderMesh.link():
            print("Linking Error", str(shaderMesh.log()))
            print('shaderMesh: Failed to link dummy renderer shader!')
        self.shaderMesh = shaderMesh

    def init_coord_grid(self):
        self.cBox = CoordinateBox(self)
        shaderCoord = qt.QOpenGLShaderProgram()
        shaderCoord.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.vertex_source)
        shaderCoord.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.fragment_source)

        if not shaderCoord.link():
            print("Linking Error", str(shaderCoord.log()))
            print('Failed to link dummy renderer shader!')
        self.cBox.shader = shaderCoord

        shaderText = qt.QOpenGLShaderProgram()
        shaderText.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.text_vertex_code)
        shaderText.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.text_fragment_code)

        if not shaderText.link():
            print("Linking Error", str(shaderText.log()))
            print('Failed to link dummy renderer shader!')
        self.cBox.textShader = shaderText

        origShader = qt.QOpenGLShaderProgram()
        origShader.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.orig_vertex_source)
        origShader.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.orig_fragment_source)

        if not origShader.link():
            print("Linking Error", str(origShader.log()))
            print('Failed to link dummy renderer shader!')
        self.cBox.origShader = origShader
        self.cBox.prepare_grid()

    def generate_beam_texture(self, width):
        hsv_texture_data = generate_hsv_texture(width, s=1.0, v=1.0)

        self.beamTexture = qt.QOpenGLTexture(qt.QOpenGLTexture.Target1D)
        self.beamTexture.create()
        self.beamTexture.setSize(width)  # Width of the texture
        self.beamTexture.setFormat(qt.QOpenGLTexture.RGB8_UNorm)
        self.beamTexture.allocateStorage()

        # Upload data (convert NumPy array to raw bytes)
        self.beamTexture.setData(
            qt.QOpenGLTexture.RGB,                 # Pixel format
            qt.QOpenGLTexture.UInt8,               # Pixel type
            hsv_texture_data.tobytes()          # Raw data as bytes
        )

    def build_histRGB(self, lb, gb, limits=None, isScreen=False, bins=[256, 256]):
        good = (lb.state == 1) | (lb.state == 2)
        if isScreen:
            x, y, z = lb.x[good], lb.z[good], lb.y[good]
        else:
            x, y, z = lb.x[good], lb.y[good], lb.z[good]

        if limits is None:
            limits = np.array([[np.min(x), np.max(x)],
                               [np.min(y), np.max(y)],
                               [np.min(z), np.max(z)]])
            beamLimits = [limits[0, :], limits[1, :]]
        else:
            beamLimits = [limits[:, 1], limits[:, 0]]
        print(f"{beamLimits=}")

        flux = gb.Jss[good]+gb.Jpp[good]

        cData = self.getColor(gb)[good]
        cData01 = ((cData - self.colorMin) * 0.85 /
                   (self.colorMax - self.colorMin)).reshape(-1, 1)

        cDataHSV = np.dstack(
            (cData01, np.ones_like(cData01) * 0.85,
             flux.reshape(-1, 1)))
        cDataRGB = (mpl.colors.hsv_to_rgb(cDataHSV)).reshape(-1, 3)

        hist2dRGB = np.zeros((bins[0], bins[1], 3), dtype=np.float64)
        hist2d = None
        if len(lb.x[good]) > 0:
            for i in range(3):  # over RGB components
                hist2dRGB[:, :, i], yedges, xedges = np.histogram2d(
                    y, x, bins=bins, range=beamLimits,
                    weights=cDataRGB[:, i])

        hist2dRGB /= np.max(hist2dRGB)
        hist2dRGB = np.uint8(hist2dRGB*255)
        print("numpy beamlimits:", beamLimits)
        return hist2d, hist2dRGB, limits

    def generate_hist_texture(self, oe, beam, is2ndXtal=False):
        nsIndex = int(is2ndXtal)
        if not hasattr(oe, 'mesh3D'):
#            print("No 3d mesh, no need to generate historgram")
            return
        meshObj = oe.mesh3D
        lb = rsources.Beam(copyFrom=beam)

        beamLimits = oe.footprint if hasattr(oe, 'footprint') else None

        histAlpha, hist2dRGB, beamLimits = self.build_histRGB(
                lb, beam, beamLimits[nsIndex])

        texture = qt.QImage(hist2dRGB, 256, 256, qt.QImage.Format_RGB888)
#        texture.save(str(oe.name)+"_beam_hist.png")
#        if hasattr(meshObj, 'beamTexture'):
#            oe.beamTexture.setData(texture)
#        meshObj.beamTexture[nsIndex] = qg.QOpenGLTexture(texture)
#        meshObj.beamLimits[nsIndex] = beamLimits
#        self.glDraw()

    def calculate_fp_hist(self, beam, width, height):

        good = (beam.state == 1) # | (beam.state == 2)
        xmin, xmax = np.min(beam.x[good]), np.max(beam.x[good])
        ymin, ymax = np.min(beam.y[good]), np.max(beam.y[good])

#        histTexture = qt.QOpenGLTexture(qt.QOpenGLTexture.Target2D)
#        histTexture.create()
#        histTexture.setSize(width, height)  # Width of the texture
#        histTexture.setFormat(qt.QOpenGLTexture.RGB32F)
#        histTexture.allocateStorage()
#        histTexture.setMinMagFilters(qt.QOpenGLTexture.Nearest,
#                                     qt.QOpenGLTexture.Nearest)

        data = np.zeros((height, width, 3), dtype=np.float32)

        red_data = np.zeros(width * height, dtype=np.uint32)
        green_data = np.zeros_like(red_data)
        blue_data = np.zeros_like(red_data)
        ind_data = np.zeros(width * height, dtype=np.uint32)

        shader = self.shaderHist
        shader.bind()

        beam.vbo['position'].bind()
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0,
                            beam.vbo['position'].bufferId())  # 0 is location for position

        beam.vbo['color'].bind()
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1,
                            beam.vbo['color'].bufferId())  # 1 is location for color

        beam.vbo['state'].bind()
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2,
                            beam.vbo['state'].bufferId())  # 0 is location for state

        beam.vbo['intensity'].bind()
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 3,
                            beam.vbo['intensity'].bufferId())  # 0 is location for intensity

        red_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, red_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, red_data.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, red_buffer)  # Binding = 4

        green_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, green_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, green_data.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, green_buffer)  # Binding = 5

        blue_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, blue_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, blue_data.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 6, blue_buffer)  # Binding = 6

        ind_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ind_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, ind_data.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 7, ind_buffer)  # Binding = 6

        shader.setUniformValue("numBins", qt.QVector2D(width, height))
        shader.setUniformValue(
                    "bounds", qt.QVector4D(xmin, ymin, xmax, ymax))  # [[xmin, ymin], [xmax, ymax]]
        shader.setUniformValue(
                    "colorMinMax", qt.QVector2D(self.colorMin, self.colorMax))
        shader.setUniformValue(
                "iMax",
                float(np.max(beam.Jss[good]+beam.Jpp[good])))

        if self.beamTexture is not None:
            self.beamTexture.bind(0)
            shader.setUniformValue("hsvTexture", 0)
        print("xmin, ymin, xmax, ymax", xmin, ymin, xmax, ymax)

#        gl.glBindImageTexture(1, histTexture.textureId(), 0, gl.GL_FALSE, 0,
#                              gl.GL_WRITE_ONLY, gl.GL_RGB32F)  # we need this to calculate colors

        num_vertices = len(beam.x)
        workgroup_size = 32
        num_workgroups = (num_vertices + workgroup_size - 1) // workgroup_size
#        print("3")
        gl.glDispatchCompute(num_workgroups, 1, 1)
#        print("4")
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)  # Ensure completion
#        print("5")



#        gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_FLOAT, data)
#        hist2dRGB /= np.max(hist2dRGB)
#        hist2dRGB = np.uint8(hist2dRGB*255)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, red_buffer)
#        print("06")
        red_result = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, red_data.nbytes)
#        print("6")
        red_data = np.frombuffer(red_result, dtype=np.uint32).reshape((width, height))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, 0)
        gl.glDeleteBuffers(1, [red_buffer])

        # Read back GreenBuffer
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, green_buffer)
        green_result = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, green_data.nbytes)
#        print("7")
        green_data = np.frombuffer(green_result, dtype=np.uint32).reshape((width, height))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, 0)
        gl.glDeleteBuffers(1, [green_buffer])
#        # Read back BlueBuffer
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, blue_buffer)
        blue_result = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, blue_data.nbytes)
#        print("8")
        blue_data = np.frombuffer(blue_result, dtype=np.uint32).reshape((width, height))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 6, 0)
        gl.glDeleteBuffers(1, [blue_buffer])

        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ind_buffer)
        ind_result = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, ind_data.nbytes)
        print("8")
        ind_data = np.squeeze(np.frombuffer(ind_result, dtype=np.uint32)) #.reshape((width, height))
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 7, 0)
        gl.glDeleteBuffers(1, [ind_buffer])

        beam.vbo['position'].release()
#        beam.vbo['color'].release()
#        beam.vbo['state'].release()
#        beam.vbo['intensity'].release()
        shader.release()
#        data = np.float64(np.dstack((red_data, green_data, blue_data)))
        data[:, :, 0] = np.float32(red_data)
        data[:, :, 1] = np.float32(green_data)
        data[:, :, 2] = np.float32(blue_data)
        data = np.sum(data, axis=2)

#        print(np.max(data))
        data /= np.max(data)
#        print(np.max(data))
        data = np.uint8(data*255)
#        print("SUM", np.sum(data))
#        nmax = 6
#        print(good[0:nmax])
#        print(np.dstack((beam.x[0:nmax], beam.y[0:nmax], beam.z[0:nmax])))
#        print(xmin, ymin, xmax, ymax)
#        print(np.dstack(((beam.x[0:nmax]-xmin)/(xmax-xmin),
#                         (beam.y[0:nmax]-ymin)/(ymax-ymin),
#                          beam.z[0:nmax])))
#        data = np.dstack((red_data[0:nmax], green_data[0:nmax], blue_data[0:nmax]))
#        data = np.dstack((red_data, green_data, blue_data))
#        print(f'{data=}', data.shape)


#        _, cpudata, _ = self.build_histRGB(beam, beam)
#        print(f'{cpudata=}')
        return data, ind_data

    def init_beam_footprint(self, beam, oe=None, is2ndXtal=None):

#        data = np.float32(np.vstack((beam.x, beam.y, beam.z)).T).copy()
        data = np.dstack((beam.x, beam.y, beam.z)).copy()
        dataColor = self.getColor(beam).copy()
        state = np.where((
                (beam.state == 1) | (beam.state == 2)), 1, 0).copy()
        intensity = np.float32(beam.Jss+beam.Jpp).copy()

        vbo = {}
        vbo['position'] = create_qt_buffer(data)
        vbo['color'] = create_qt_buffer(dataColor)
        vbo['state'] = create_qt_buffer(state)
        vbo['intensity'] = create_qt_buffer(intensity)
        goodRays = np.where(((state > 0) & (intensity/self.iMax > self.cutoffI)))[0]
#        if oe is None:
#            oe = self.beamLine.oesDict[beam.parentId][0]

        vbo['indices'] = create_qt_buffer(goodRays.copy(), isIndex=True)
        vbo['goodLen'] = len(goodRays)
#        print(len(goodRays))

        vao = qt.QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()

        vbo['position'].bind()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Attribute 0: position
        gl.glEnableVertexAttribArray(0)
        vbo['position'].release()

        vbo['color'].bind()
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Attribute 1: colorAxis
        gl.glEnableVertexAttribArray(1)
        vbo['color'].release()

        vbo['state'].bind()
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Attribute 2: state
        gl.glEnableVertexAttribArray(2)
        vbo['state'].release()

        vbo['intensity'].bind()
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Attribute 3: intensity
        gl.glEnableVertexAttribArray(3)
        vbo['intensity'].release()

        vao.release()

        beam.beamLen = min(MAXRAYS, len(beam.x))

        beam.vbo = vbo
        beam.vao = vao

    def change_beam_colorax(self):
#        print(self.newColorAxis)
        if self.newColorAxis:
            newColorMax = -1e20
            newColorMin = 1e20
        else:
            newColorMax = self.colorMax
            newColorMin = self.colorMin

        for beamName, beam in self.beamsDict.items():
            if not hasattr(beam, 'vbo'):
                # 3D beam object not initialized
                continue
#            print(self.getColor)
            colorax = np.float32(self.getColor(beam))
            good = (beam.state == 1) | (beam.state == 2)

            beam.vbo['color'].bind()
            beam.vbo['color'].write(0, colorax, colorax.nbytes)
            beam.vbo['color'].release()

            newColorMax = max(np.max(
                colorax[good]),
                newColorMax)
            newColorMin = min(np.min(
                colorax[good]),
                newColorMin)

        if self.newColorAxis:
            if newColorMin != self.colorMin:
                self.colorMin = newColorMin
                self.selColorMin = self.colorMin
            if newColorMax != self.colorMax:
                self.colorMax = newColorMax
                self.selColorMax = self.colorMax

        if self.colorMin == self.colorMax:
            if self.colorMax == 0:  # and self.colorMin == 0 too
                self.colorMin, self.colorMax = -0.1, 0.1
            else:
                self.colorMin = self.colorMax * 0.99
                self.colorMax *= 1.01
#        print(self.colorMin, self.colorMax)

#        hue = (self.colorMin - self.colorMin)/(self.colorMax - self.colorMin)
#        nrange = np.array([n*64 for n in range(8)])
#        print([(n, self.hsvTex[n, :]/255.) for n in nrange])
#        print([(n, mpl.colors.hsv_to_rgb((n/511, 1, 1))) for n in nrange])

#        print()


        if False:  # Updating textures with histograms
            for oeuuid, oeLine in self.beamLine.oesDict.items():
                oeToPlot = oeLine[0]
    #            oeToPlot = self.oesDict[oeuuid]
#                if hasattr(oe, 'beamsOut'):
#                    if 'beamGlobal' in oeToPlot.beamsOut:
#                        beam = oeToPlot.beamsOut['beamGlobal']
#                    else:
#                        beam = oeToPlot.beamsOut['beamLocal']
#                else:
#                    continue
                if hasattr(oeToPlot, 'material'):
                    is2ndXtal = False  # TODO: DCM, Plate
    #                t01 = time.time()
                    self.generate_hist_texture(oeToPlot, beam, is2ndXtal)
#                t02 = time.time()
#            t01 = time.time()
#            print("total", t01-t00, "s")

#        t2 = time.time()
#        print("ch_color for hists took", t2-t0, "s")
#        for ioe in range(self.segmentModel.rowCount() - 1):
#            ioeItem = self.segmentModel.child(ioe + 1, 0)
#            beam = self.beamsDict[self.oesList[str(ioeItem.text())][1]]
#            beam.colorMinMax = qg.QVector2D(self.colorMin, self.colorMax)
        self.isColorAxReady = True

    def updateCutOffI(self, cutOff):
        for beamName, beam in self.beamsDict.items():
            if not hasattr(beam, 'vbo'):
                # 3D beam object not initialized
                continue
            intensity = np.float32(beam.Jss+beam.Jpp)
            goodRays = np.uint32(np.where((((beam.state == 1) | (beam.state == 2)) & (intensity/self.iMax > self.cutoffI)))[0])
            beam.vbo['goodLen'] = len(goodRays)

            beam.vbo['indices'].bind()
            beam.vbo['indices'].write(0, goodRays, goodRays.nbytes)
#            beam.vao.bind()
#            beam.vbo['color'].write(0, colorax, colorax.nbytes)
            beam.vbo['indices'].release()


    def render_beam(self, beam, model, view, projection, target=None):
        shader = self.shaderBeam if target else self.shaderFootprint
        if not hasattr(beam, 'vbo'):
            print("No VBO")
            return

        if target is not None and not hasattr(target, 'vbo'):
            return
        shader.bind()
        beam.vao.bind()

#        oeIn = self.beamLine.oesDict[beam.parentId][0]

        if target is not None:
#            oeTarget = self.beamLine.oesDict[target.parentId][0]
            target.vbo['position'].bind()
            gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(4)
#            target.vbo['color'].bind()
#            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
#            gl.glEnableVertexAttribArray(1)

#            target.vbo['indices'].bind()
#            arrLen = target.vbo['goodLen']
#            print("Rendering beam from {} to {}".format(oeIn.name, oeTarget.name))
#            print(arrLen)
#        else:

        beam.vbo['indices'].bind()
        arrLen = beam.vbo['goodLen']

#        oeuuid = beam.parentId
#        oe = self.beamLine.oesDict[oeuuid][0]
#        if hasattr(oe, 'mesh3D'):
#            oeIndex = int(hasattr(beam, 'is2ndXtal'))
#            oeOrientation = oe.mesh3D.transMatrix[oeIndex]
#            beam.shader.setUniformValue("model", model*oeOrientation)
#        else:
        if self.beamTexture is not None:
            self.beamTexture.bind(0)
            shader.setUniformValue("hsvTexture", 0)

        mPV = projection*view

        shader.setUniformValue("model", model)
#        shader.setUniformValue("view", view)
#        shader.setUniformValue("projection", projection)
        shader.setUniformValue("mPV", mPV)

        shader.setUniformValue(
                    "colorMinMax", qt.QVector2D(self.colorMin, self.colorMax))
#        print(self.colorMin, self.colorMax)
        shader.setUniformValue("gridMask", qt.QVector4D(1, 1, 1, 1))
        shader.setUniformValue("gridProjection", qt.QVector4D(0, 0, 0, 0))

        shader.setUniformValue(
                "pointSize",
                float(self.pointSize if target is None else self.lineWidth))
        shader.setUniformValue(
                "opacity",
                float(self.pointOpacity if target is None else self.lineOpacity))
        shader.setUniformValue(
                "iMax",
                float(self.iMax if self.globalNorm else beam.iMax))

        if target and self.lineWidth > 0:
            gl.glLineWidth(self.lineWidth)

        gl.glDrawElements(gl.GL_POINTS,  arrLen,
                          gl.GL_UNSIGNED_INT, None)

        if target and self.lineProjectionWidth > 0:
            gl.glLineWidth(self.lineProjectionWidth)

        for dim in range(3):
            if self.projectionsVisibility[dim] > 0:
                gridMask = [1.]*4
                gridMask[dim] = 0.
                gridProjection = [0.]*4
                gridProjection[dim] = -self.aPos[dim]
                shader.setUniformValue(
                        "gridMask",
                        qt.QVector4D(*gridMask))
                shader.setUniformValue(
                        "gridProjection",
                        qt.QVector4D(*gridProjection))
                shader.setUniformValue(
                        "pointSize",
                        float(self.pointProjectionSize if target is None else self.lineProjectionWidth))
                shader.setUniformValue(
                        "opacity",
                        float(self.pointProjectionOpacity if target is None else self.lineProjectionOpacity))

                gl.glDrawElements(gl.GL_POINTS,  arrLen,
                                  gl.GL_UNSIGNED_INT, None)


        if target:
            target.vbo['position'].release()
#            target.vbo['indices'].release()

        beam.vao.release()
        beam.vbo['indices'].release()
        shader.release()
        if self.beamTexture is not None:
            self.beamTexture.release()


    def glDraw(self):
        self.update()

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
#        return
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

        return
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
        pass
#        useScalableFont = False if noScalable else self.useScalableFont
#        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
#        pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
#        if not useScalableFont:
#            gl.glRasterPos3f(*coord)
#            for symbol in text:
#                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
#        else:
#            tLineWidth = gl.glGetDoublev(gl.GL_LINE_WIDTH)
#            tLineAA = gl.glIsEnabled(gl.GL_LINE_SMOOTH)
#            if self.useFontAA:
#                gl.glEnable(gl.GL_LINE_SMOOTH)
#            else:
#                gl.glDisable(gl.GL_LINE_SMOOTH)
#            gl.glLineWidth(self.scalableFontWidth)
#
#            fontScale = self.fontSize / 12500.
#            coordShift = np.zeros(3, dtype=np.float32)
#            fontSizeLoc = np.float32(np.array([104.76, 119.05, 0])*fontScale)
#            if alignment is not None:
#                if alignment[0] == 'left':
#                    coordShift[0] = -fontSizeLoc[0] * len(text)
#                else:
#                    coordShift[0] = fontSizeLoc[0]
#
#                if alignment[1] == 'top':
#                    vOffset = 0.5
#                elif alignment[1] == 'bottom':
#                    vOffset = -1.5
#                else:
#                    vOffset = -0.5
#                coordShift[1] = vOffset * fontSizeLoc[1]
#            if useCaption:
#                textWidth = 0
#                for symbol in text.strip(" "):
#                    textWidth += gl.glutStrokeWidth(self.scalableFontType,
#                                                    ord(symbol))
#                gl.glPushMatrix()
#                gl.glTranslatef(*coord)
#                gl.glRotatef(*self.qText)
#                gl.glTranslatef(*coordShift)
#                gl.glScalef(fontScale, fontScale, fontScale)
#                depthCounter = 1
#                spaceFound = False
#                while not spaceFound:
#                    depthCounter += 1
#                    for dy in [-1, 1]:
#                        for dx in [1, -1]:
#                            textShift = (depthCounter+0.5*dy) * 119.05*1.5
#                            gl.glPushMatrix()
#                            textPos = [dx*depthCounter * 119.05*1.5 +
#                                       (0 if dx > 0 else -1) * textWidth,
#                                       dy*textShift, 0]
#                            gl.glTranslatef(*textPos)
#                            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
#                            bottomLeft = np.array(gl.gluProject(
#                                *[0, 0, 0], model=pModel, proj=pProjection,
#                                view=pView)[:-1])
#                            topRight = np.array(gl.gluProject(
#                                *[textWidth, 119.05*2.5, 0],
#                                model=pModel, proj=pProjection,
#                                view=pView)[:-1])
#                            gl.glPopMatrix()
#                            spaceFound = True
#                            for oeLabel in list(self.labelsBounds.values()):
#                                if not (bottomLeft[0] > oeLabel[1][0] or
#                                        bottomLeft[1] > oeLabel[1][1] or
#                                        topRight[0] < oeLabel[0][0] or
#                                        topRight[1] < oeLabel[0][1]):
#                                    spaceFound = False
#                            if spaceFound:
#                                self.labelsBounds[text] = [0]*2
#                                self.labelsBounds[text][0] = bottomLeft
#                                self.labelsBounds[text][1] = topRight
#                                break
#                        if spaceFound:
#                            break
#
#                gl.glPopMatrix()
#                gl.glPushMatrix()
#                gl.glTranslatef(*coord)
#                gl.glRotatef(*self.qText)
#                gl.glScalef(fontScale, fontScale, fontScale)
#                captionPos = depthCounter * 119.05*1.5
#                gl.glBegin(gl.GL_LINE_STRIP)
#                gl.glVertex3f(0, 0, 0)
#                gl.glVertex3f(captionPos*dx, captionPos*dy, 0)
#                gl.glVertex3f(captionPos*dx + textWidth*dx,
#                              captionPos*dy, 0)
#                gl.glEnd()
#                gl.glTranslatef(*textPos)
#                for symbol in text.strip(" "):
#                    gl.glutStrokeCharacter(self.scalableFontType, ord(symbol))
#                gl.glPopMatrix()
#            else:
#                gl.glPushMatrix()
#                gl.glTranslatef(*coord)
#                gl.glRotatef(*self.qText)
#                gl.glTranslatef(*coordShift)
#                gl.glScalef(fontScale, fontScale, fontScale)
#                for symbol in text:
#                    gl.glutStrokeCharacter(self.scalableFontType, ord(symbol))
#                gl.glPopMatrix()
#
#            gl.glLineWidth(tLineWidth)
#            if tLineAA:
#                gl.glEnable(gl.GL_LINE_SMOOTH)
#            else:
#                gl.glDisable(gl.GL_LINE_SMOOTH)

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
            retStr = ''  # disabling parentheses temporarily
            for dim in centerList:
                retStr += '{0:.{1}f}, '.format(dim, prec)
            return retStr[:-2] + ''

        try:
            gl.glClearColor(0.0, 0.0, 0.0, 1.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT |
                       gl.GL_DEPTH_BUFFER_BIT |
                       gl.GL_STENCIL_BUFFER_BIT)

            self.mModScale.setToIdentity()
            self.mModTrans.setToIdentity()
            self.mModScale.scale(*(self.scaleVec/self.maxLen))
            self.mModTrans.translate(*(self.tVec-self.coordOffset))
            self.mMod = self.mModScale*self.mModTrans

            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if self.enableAA:
                gl.glEnable(gl.GL_LINE_SMOOTH)
                gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
                gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)

#            for oeuuid, oeLine in self.beamLine.oesDict.items():
#                oeToPlot = oeLine[0]
#            for oeString in self.oesToPlot:

            if self.enableBlending:
                gl.glEnable(gl.GL_MULTISAMPLE)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    #            gl.glBlendFunc(gl.GL_SRC_ALPHA, GL_ONE)
                gl.glEnable(gl.GL_POINT_SMOOTH)
                gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

            if not self.linesDepthTest:
                gl.glDepthMask(gl.GL_TRUE)

            gl.glEnable(gl.GL_DEPTH_TEST)
            for ioe in range(self.segmentModel.rowCount() - 1):
                if self.segmentModel.child(ioe + 1, 2).checkState() != 2:
                    continue
                ioeItem = self.segmentModel.child(ioe + 1, 0)
                oeString = str(ioeItem.text())
                oeToPlot = self.oesList[oeString][0]
                oeuuid = oeToPlot.uuid

#                print(oeToPlot.name, oeToPlot.uuid)
                if isinstance(oeToPlot, roes.OE):
                    is2ndXtalOpts = [False]
                    if isinstance(oeToPlot, roes.DCM):
                        is2ndXtalOpts.append(True)

                    for is2ndXtal in is2ndXtalOpts:
                        if hasattr(oeToPlot, 'mesh3D') and oeToPlot.mesh3D.isEnabled:
                            isSelected = False
                            if oeuuid in self.selectableOEs.values():
                                oeNum = oeToPlot.mesh3D.stencilNum
                                isSelected = oeNum == self.selectedOE
                                gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum), 0xff)
                            oeToPlot.mesh3D.render_surface(self.mMod, self.mView, self.mProj,
                                                         is2ndXtal, isSelected=isSelected,
                                                         shader=self.shaderMesh)
            gl.glStencilFunc(gl.GL_ALWAYS, 0, 0xff)

            if not self.linesDepthTest:
                gl.glDepthMask(gl.GL_FALSE)

            if self.pointsDepthTest:
                gl.glEnable(gl.GL_DEPTH_TEST)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)

            for ioe in range(self.segmentModel.rowCount() - 1):
                ioeItem = self.segmentModel.child(ioe + 1, 0)
                beam = self.beamsDict[self.oesList[str(ioeItem.text())][1]]
                if self.segmentModel.child(ioe + 1, 1).checkState() == 2:
#                    print(beam, hasattr(beam, 'vbo'))
                    self.render_beam(beam, self.mMod, self.mView, self.mProj, target=None)

            gl.glEnable(gl.GL_DEPTH_TEST)

#            if self.linesDepthTest:
#                gl.glEnable(gl.GL_DEPTH_TEST)
#            else:
#                gl.glDisable(gl.GL_DEPTH_TEST)

            for ioe in range(self.segmentModel.rowCount() - 1):
                ioeItem = self.segmentModel.child(ioe + 1, 0)
                beam = self.beamsDict[self.oesList[str(ioeItem.text())][1]]
                if ioeItem.hasChildren():
                    for isegment in range(ioeItem.rowCount()):
                        segmentItem0 = ioeItem.child(isegment, 0)
                        if segmentItem0.checkState() == 2:
                            endBeam = self.beamsDict[
                                self.oesList[str(segmentItem0.text())[3:]][1]]
                            self.render_beam(beam, self.mMod, self.mView, self.mProj, target=endBeam)

            gl.glDepthMask(gl.GL_TRUE)
            gl.glEnable(gl.GL_DEPTH_TEST)
#            gl.glDisable(gl.GL_MULTISAMPLE)
#            gl.glDisable(gl.GL_BLEND)

            self.cBox.textShader.bind()
            self.cBox.vaoText.bind()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glEnable(gl.GL_POLYGON_SMOOTH)
            gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
            
            sclY = self.cBox.characters[124][1][1]*0.04*self.cBox.fontScale/float(self.viewPortGL[3])
            
            labelBounds = []

            for ioe in range(self.segmentModel.rowCount() - 1):
                if self.segmentModel.child(ioe + 1, 3).checkState() == 2:
                    ioeItem = self.segmentModel.child(ioe + 1, 0)
                    oeString = str(ioeItem.text())
                    oeToPlot = self.oesList[oeString][0]
                    oeCenter = self.modelToWorld(
                            np.array(oeToPlot.center) - self.coordOffset)
                    vpMat = self.mProj * self.mView
                    alignment = "middle"
                    dx = 0.075
                    oeCenterStr = makeCenterStr(oeToPlot.center,
                                                self.labelCoordPrec)
                    oeLabel = '  {0}: {1}mm'.format(
                        oeString, oeCenterStr)
#                    print(oeLabel)
                    labelPos = (vpMat*qt.QVector4D(*oeCenter, 1)).toVector3DAffine() + qt.QVector3D(dx, 0, 0)
                    
#                    mMod = qt.QMatrix4x4()
#                    mMod.setToIdentity()
#                    mMod.translate(labelPos)
#                    mMod.scale(1, sclY, 1)
#                    
#                    labelYmin = mMod*qt.QVector4D(0, 0, 0, 1)
#                    labelYmax = mMod*qt.QVector4D(1, 1, 1, 1)
                    
#                    print(labelPos, labelYmin, labelYmax)
                    
                    intersecting = True
                    while intersecting:
                        labelYmin = labelPos.y()
                        labelYmax = labelYmin + sclY                       
                        for bmin, bmax in labelBounds:
                            if labelYmax > bmin and labelYmin < bmax:
                                labelPos += qt.QVector3D(0, 1.5*sclY, 0)
                                break
                            elif labelYmin > bmax and labelYmax < bmin:
                                labelPos -= qt.QVector3D(0, 1.5*sclY, 0)
                                break
                        else:
                            intersecting = False
                            labelBounds.append((labelPos.y(), labelPos.y() + sclY))

                    
                    self.cBox.render_text(labelPos, oeLabel, alignment=alignment,
                                     scale=0.04*self.cBox.fontScale)
            self.cBox.textShader.release()
            self.cBox.vaoText.release()
            self.cBox.draw(self.mModAx, self.mView, self.mProj)

            if self.enableAA:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            self.eCounter = 0
        except Exception as e:
            raise
            self.eCounter += 1
            if self.eCounter < 10:
                self.update()
            else:
                self.eCounter = 0
                pass

    def paintGL_old(self):

        def makeCenterStr(centerList, prec):
            retStr = '('
            for dim in centerList:
                retStr += '{0:.{1}f}, '.format(dim, prec)
            return retStr[:-2] + ')'

        gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)

        if self.invertColors:
            gl.glClearColor(1.0, 1.0, 1.0, 1.)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 1.)
            gl.glClearColor(0.0, 0.0, 0.25, 0.5)

#        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT |
                   gl.GL_DEPTH_BUFFER_BIT |
                   gl.GL_STENCIL_BUFFER_BIT)

        self.mModScale.setToIdentity()
        self.mModTrans.setToIdentity()
        self.mModScale.scale(*(self.scaleVec/self.maxLen))
        self.mModTrans.translate(*(self.tVec-self.coordOffset))
        self.mMod = self.mModScale*self.mModTrans

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
                    oeuuid = oeToPlot.uuid
#                    is2ndXtal = self.oesList[oeString][3]
                    if isinstance(oeToPlot, roes.OE):
#                        print(oeuuid)
                        is2ndXtalOpts = [False]
                        if isinstance(oeToPlot, roes.DCM):
                            is2ndXtalOpts.append(True)

                        for is2ndXtal in is2ndXtalOpts:
                            if hasattr(oeToPlot, 'mesh3D') and oeToPlot.mesh3D.isEnabled:
                                print(1)
                                isSelected = False
                                if oeuuid in self.selectableOEs.values():
                                    oeNum = oeToPlot.mesh3D.stencilNum
                                    isSelected = oeNum == self.selectedOE
                                    gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum), 0xff)
                                oeToPlot.mesh3D.render_surface(self.mMod, self.mView, self.mProj,
                                                             is2ndXtal, isSelected=isSelected,
                                                             shader=self.shaderMesh)

#                        self.plotOeSurface(oeToPlot, is2ndXtal)
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
            pass
#            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
#            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
#            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
#            gridColor = np.ones((len(gridArray), 4)) * lineOpacity
#            gridArrayVBO = gl.vbo.VBO(np.float32(gridArray))
#            gridArrayVBO.bind()
#            gl.glVertexPointerf(gridArrayVBO)
#            gridColorArray = gl.vbo.VBO(np.float32(gridColor))
#            gridColorArray.bind()
#            gl.glColorPointerf(gridColorArray)
#            gl.glLineWidth(lineWidth)
#            gl.glDrawArrays(figType, 0, len(gridArrayVBO))
#            gridArrayVBO.unbind()
#            gridColorArray.unbind()
#            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
#            gl.glDisableClientState(gl.GL_COLOR_ARRAY)

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
        pass
#        if vertices is None or colors is None:
#            return
#
#        if bool(tr):
#            vertexArray = gl.vbo.VBO(self.modelToWorld(vertices))
#        else:
#            vertexArray = gl.vbo.VBO(vertices)
#        vertexArray.bind()
#        gl.glVertexPointerf(vertexArray)
#        pureOpacity = np.copy(colors[:, 3])
#        colors[:, 3] = np.float32(pureOpacity * lineOpacity)
#        colorArray = gl.vbo.VBO(colors)
#        colorArray.bind()
#        gl.glColorPointerf(colorArray)
#        if geom == gl.GL_LINES:
#            gl.glLineWidth(lineWidth)
#        else:
#            gl.glPointSize(lineWidth)
#        gl.glDrawArrays(geom, 0, len(vertices))
#        colors[:, 3] = pureOpacity
#        colorArray.unbind()
#        vertexArray.unbind()

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
            isClosedSurface = isinstance(oe, roes.SurfaceOfRevolution) or\
                (hasattr(oe, 'isClosed') and oe.isClosed)
            if oe.footprint is not None:
                xLimits = oe.footprint[nsIndex][:, 0]
        if np.any(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
            if oe.footprint is not None:
                yLimits = oe.footprint[nsIndex][:, 1]
        localTiles = np.array(self.tiles)

        if isClosedSurface:
            # the limits are in parametric coordinates
            yLimitsS0, _, _ = oe.xyz_to_param(0, yLimits[0], 0)
            yLimitsS1, _, _ = oe.xyz_to_param(0, yLimits[1], 0)
            xLimits = [yLimitsS0, yLimitsS1]  # s
            yLimits = [0, 2*np.pi]  # phi
            localTiles[1] *= 3
        else:
            if oe.shape == 'round':
                rX = np.abs((xLimits[1] - xLimits[0]))*0.5
                rY = np.abs((yLimits[1] - yLimits[0]))*0.5
                cX = (xLimits[1] + xLimits[0])*0.5
                cY = (yLimits[1] + yLimits[0])*0.5
                xLimits = [0, 1.]
                yLimits = [0, 2*np.pi]
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
                if oe.shape == 'round' and not isClosedSurface:
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
                    if oe.shape == 'round' and not isClosedSurface:
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
                if ie == 0 and oe.shape == 'round' and not isClosedSurface:
                    continue
                deltaY = (yLimits[1] - yLimits[0]) / float(localTiles[1])
                for i in range(localTiles[1]):
                    yGridOe = np.linspace(yLimits[0] + i*deltaY,
                                          yLimits[0] + (i+1)*deltaY,
                                          self.surfCPOrder)

                    edgeY = yGridOe
                    edgeX = np.ones_like(yGridOe)*xPos
                    edgeZ = np.zeros_like(xGridOe)

                    if oe.shape == 'round' and not isClosedSurface:
                        edgeX, edgeY = rX*edgeX*np.cos(edgeY)+cX, \
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
                    if oe.shape == 'round' and not isClosedSurface:
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
#        if oe.shape == 'round' and not isClosedSurface:
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
#            if oe.shape == 'round' and not isClosedSurface:
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

        if oe.shape == 'round':
            r = oe.r
            isBeamStop = len(re.findall('Stop', str(type(oe)))) > 0
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

            w = right - left
            h = top - bottom
            # wf = max(min(w, h)*0.5, 2.5)
            wf = min(w, h) * self.slitThicknessFraction*0.01
            limits = []
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
#                for symbol in text:
#                    gl.glutStrokeCharacter(
#                        gl.GLUT_STROKE_MONO_ROMAN, ord(symbol))
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
        pass
#        phi = np.linspace(0, 2*np.pi, nFacets)
#        xp = r * np.cos(phi)
#        yp = r * np.sin(phi)
#        base = np.vstack((xp, yp, np.zeros_like(xp)))
#        coneVertices = np.hstack((np.array([0, 0, z]).reshape(3, 1),
#                                  base)).T
#        gridColor = np.zeros((len(coneVertices), 4))
#        gridColor[:, color] = 1
#        gridColor[:, 3] = 0.75
#        gridArray = gl.vbo.VBO(np.float32(coneVertices))
#        gridArray.bind()
#        gl.glVertexPointerf(gridArray)
#        gridColorArray = gl.vbo.VBO(np.float32(gridColor))
#        gridColorArray.bind()
#        gl.glColorPointerf(gridColorArray)
#        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(gridArray))
#        gridArray.unbind()
#        gridColorArray.unbind()

    def drawLocalAxes(self, oe, is2ndXtal):
        pass
#        def drawArrow(color, arrowArray, yText='hkl'):
#            gridColor = np.zeros((len(arrowArray) - 1, 4))
#            gridColor[:, 3] = 0.75
#            if color == 4:
#                gridColor[:, 0] = 1
#                gridColor[:, 1] = 1
#            elif color == 5:
#                gridColor[:, 0] = 1
#                gridColor[:, 1] = 0.5
#            else:
#                gridColor[:, color] = 1
#            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
#            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
#            gridArray = gl.vbo.VBO(np.float32(arrowArray[1:, :]))
#            gridArray.bind()
#            gl.glVertexPointerf(gridArray)
#            gridColorArray = gl.vbo.VBO(np.float32(gridColor))
#            gridColorArray.bind()
#            gl.glColorPointerf(gridColorArray)
#            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(gridArray))
#            gridArray.unbind()
#            gridColorArray.unbind()
#            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
#            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
#            gl.glEnable(gl.GL_LINE_SMOOTH)
#            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
#            gl.glBegin(gl.GL_LINES)
#            colorVec = [0, 0, 0, 0.75]
#            if color == 4:
#                colorVec[0] = 1
#                colorVec[1] = 1
#            elif color == 5:
#                colorVec[0] = 1
#                colorVec[1] = 0.5
#            else:
#                colorVec[color] = 1
#            gl.glColor4f(*colorVec)
#            gl.glVertex3f(*arrowArray[0, :])
#            gl.glVertex3f(*arrowArray[1, :])
#            gl.glEnd()
#            gl.glColor4f(*colorVec)
#            gl.glRasterPos3f(*arrowArray[1, :])
#            if color == 0:
#                axSymb = 'Z'
#            elif color == 1:
#                axSymb = 'Y'
#            elif color == 2:
#                axSymb = 'X'
#            elif color == 4:
#                axSymb = yText
#            else:
#                axSymb = ''
#
#            for symbol in "  {}".format(axSymb):
#                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
#            gl.glDisable(gl.GL_LINE_SMOOTH)
#
#        z, r, nFacets = 0.25, 0.02, 20
#        phi = np.linspace(0, 2*np.pi, nFacets)
#        xp = np.insert(r * np.cos(phi), 0, [0., 0.])
#        yp = np.insert(r * np.sin(phi), 0, [0., 0.])
#        zp = np.insert(z*0.8*np.ones_like(phi), 0, [0., z])
#
#        crPlaneZ = None
#        yText = None
#        if hasattr(oe, 'local_n'):
#            material = None
#            if hasattr(oe, 'material'):
#                material = oe.material
#            if is2ndXtal:
#                zExt = '2'
#                if hasattr(oe, 'material2'):
#                    material = oe.material2
#            else:
#                zExt = '1' if hasattr(oe, 'local_n1') else ''
#            if raycing.is_sequence(material):
#                material = material[oe.curSurface]
#
#            local_n = getattr(oe, 'local_n{}'.format(zExt))
#            normals = local_n(0, 0)
#            if len(normals) > 3:
#                crPlaneZ = np.array(normals[:3], dtype=float)
#                crPlaneZ /= np.linalg.norm(crPlaneZ)
#                if material not in [None, 'None']:
#                    if hasattr(material, 'hkl'):
#                        hklSeparator = ',' if np.any(np.array(
#                            material.hkl) >= 10) else ''
#                        yText = '[{0[0]}{1}{0[1]}{1}{0[2]}]'.format(
#                            list(material.hkl), hklSeparator)
##                        yText = '{}'.format(list(material.hkl))
#
#        cb = rsources.Beam(nrays=nFacets+2)
#        cb.a[:] = cb.b[:] = cb.c[:] = 0.
#        cb.a[0] = cb.b[1] = cb.c[2] = 1.
#
#        if crPlaneZ is not None:  # Adding asymmetric crystal orientation
#            asAlpha = np.arccos(crPlaneZ[2])
#            acpX = np.array([0., 0., 1.], dtype=float) if asAlpha == 0 else\
#                np.cross(np.array([0., 0., 1.], dtype=float), crPlaneZ)
#            acpX /= np.linalg.norm(acpX)
#
#            cb.a[3] = acpX[0]
#            cb.b[3] = acpX[1]
#            cb.c[3] = acpX[2]
#
#        cb.state[:] = 1
#
#        if isinstance(oe, (rscreens.HemisphericScreen, rscreens.Screen)):
#            cb.x[:] += oe.center[0]
#            cb.y[:] += oe.center[1]
#            cb.z[:] += oe.center[2]
#            oeNormX = oe.x
#            oeNormY = oe.y
#        else:
#            if is2ndXtal:
#                oe.local_to_global(cb, is2ndXtal=is2ndXtal)
#            else:
#                oe.local_to_global(cb)
#            oeNormX = np.array([cb.a[0], cb.b[0], cb.c[0]])
#            oeNormY = np.array([cb.a[1], cb.b[1], cb.c[1]])
#
#        scNormX = oeNormX * self.scaleVec
#        scNormY = oeNormY * self.scaleVec
#
#        scNormX /= np.linalg.norm(scNormX)
#        scNormY /= np.linalg.norm(scNormY)
#        scNormZ = np.cross(scNormX, scNormY)
#        scNormZ /= np.linalg.norm(scNormZ)
#
#        for iAx in range(3):
#            if iAx == 0:
#                xVec = scNormX
#                yVec = scNormY
#                zVec = scNormZ
#            elif iAx == 2:
#                xVec = scNormY
#                yVec = scNormZ
#                zVec = scNormX
#            else:
#                xVec = scNormZ
#                yVec = scNormX
#                zVec = scNormY
#
#            dX = xp[:, np.newaxis] * xVec
#            dY = yp[:, np.newaxis] * yVec
#            dZ = zp[:, np.newaxis] * zVec
#            coneCP = self.modelToWorld(np.vstack((
#                cb.x - self.coordOffset[0], cb.y - self.coordOffset[1],
#                cb.z - self.coordOffset[2])).T) + dX + dY + dZ
#            drawArrow(iAx, coneCP)
#
#        if crPlaneZ is not None:  # drawAsymmetricPlane:
#            crPlaneX = np.array([cb.a[3], cb.b[3], cb.c[3]])
#            crPlaneNormX = crPlaneX * self.scaleVec
#            crPlaneNormX /= np.linalg.norm(crPlaneNormX)
#            crPlaneNormZ = self.rotateVecQ(
#                scNormZ, self.vecToQ(crPlaneNormX, asAlpha))
#            crPlaneNormZ /= np.linalg.norm(crPlaneNormZ)
#            crPlaneNormY = np.cross(crPlaneNormX, crPlaneNormZ)
#            crPlaneNormY /= np.linalg.norm(crPlaneNormY)
#
#            color = 4
#
#            dX = xp[:, np.newaxis] * crPlaneNormX
#            dY = yp[:, np.newaxis] * crPlaneNormY
#            dZ = zp[:, np.newaxis] * crPlaneNormZ
#            coneCP = self.modelToWorld(np.vstack((
#                cb.x - self.coordOffset[0], cb.y - self.coordOffset[1],
#                cb.z - self.coordOffset[2])).T) + dX + dY + dZ
#            drawArrow(color, coneCP, yText)

    def drawDirectionAxes(self):
        pass
#        arrowSize = 0.05
#        axisLen = 0.1
#        tLen = (arrowSize + axisLen) * 2
#        gl.glLineWidth(1.)
#
#        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
#        gl.glViewport(0, 0, int(150*self.aspect), 150)
#
#        gl.glMatrixMode(gl.GL_PROJECTION)
#        gl.glLoadIdentity()
#        if self.perspectiveEnabled:
#            gl.gluPerspective(60, self.aspect, 0.001, 10)
#        else:
#            gl.glOrtho(-tLen*self.aspect, tLen*self.aspect, -tLen, tLen, -1, 1)
#
#        gl.glMatrixMode(gl.GL_MODELVIEW)
#        gl.glLoadIdentity()
#        gl.gluLookAt(.5, 0.0, 0.0,
#                     0.0, 0.0, 0.0,
#                     0.0, 0.0, 1.0)
#
#        gl.glEnable(gl.GL_LINE_SMOOTH)
#        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
#        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
#        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
#
#        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
#        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
#
#        self.rotateZYX()
#
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#        for iAx in range(3):
#            if not (not self.perspectiveEnabled and
#                    2-iAx == self.visibleAxes[2]):
#                gl.glPushMatrix()
#                trVec = np.zeros(3, dtype=np.float32)
#                trVec[2-iAx] = axisLen
#                gl.glTranslatef(*trVec)
#                if iAx == 1:
#                    gl.glRotatef(-90, 1.0, 0.0, 0.0)
#                elif iAx == 2:
#                    gl.glRotatef(90, 0.0, 1.0, 0.0)
#                self.drawCone(arrowSize, 0.02, 20, iAx)
#                gl.glPopMatrix()
#        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
#        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
#
#        gl.glBegin(gl.GL_LINES)
#        for iAx in range(3):
#            if not (not self.perspectiveEnabled and
#                    2-iAx == self.visibleAxes[2]):
#                colorVec = [0, 0, 0, 0.75]
#                colorVec[iAx] = 1
#                gl.glColor4f(*colorVec)
#                gl.glVertex3f(0, 0, 0)
#                trVec = np.zeros(3, dtype=np.float32)
#                trVec[2-iAx] = axisLen
#                gl.glVertex3f(*trVec)
#                gl.glColor4f(*colorVec)
#        gl.glEnd()
#
#        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 2):
#            gl.glColor4f(1, 0, 0, 1)
#            gl.glRasterPos3f(0, 0, axisLen*1.5)
#            for symbol in "  {} (mm)".format('Z'):
#                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
#        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 1):
#            gl.glColor4f(0, 0.75, 0, 1)
#            gl.glRasterPos3f(0, axisLen*1.5, 0)
#            for symbol in "  {} (mm)".format('Y'):
#                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
#        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 0):
#            gl.glColor4f(0, 0.5, 1, 1)
#            gl.glRasterPos3f(axisLen*1.5, 0, 0)
#            for symbol in "  {} (mm)".format('X'):
#                gl.glutBitmapCharacter(self.fixedFont, ord(symbol))
##        gl.glFlush()
#        gl.glViewport(*pView)
#        gl.glColor4f(1, 1, 1, 1)
#        gl.glDisable(gl.GL_LINE_SMOOTH)

    def initializeGL(self):
#        gl.glutInit()
#        gl.glutInitDisplayMode(gl.GLUT_RGBA | gl.GLUT_DOUBLE | gl.GLUT_DEPTH)

#        gl.glEnable(gl.GL_DEPTH_TEST)
#        gl.glDepthFunc(gl.GL_LESS);
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#        gl.glEnable(gl.GL_POINT_SMOOTH)
#        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

        self.init_coord_grid()  # We let coordBox have it's own shaders
        self.init_shaders()
        self.generate_beam_texture(512)

        self.iMax = -1e20

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

        for beamName, startBeam in self.beamsDict.items():
            good = (startBeam.state == 1) | (startBeam.state == 2)
            if len(startBeam.state[good]) > 0:
                for tmpCoord, tAxis in enumerate(['x', 'y', 'z']):
                    axMin = np.min(getattr(startBeam, tAxis)[good])
                    axMax = np.max(getattr(startBeam, tAxis)[good])
                    if axMin < tmpMin[tmpCoord]:
                        tmpMin[tmpCoord] = axMin
                    if axMax > tmpMax[tmpCoord]:
                        tmpMax[tmpCoord] = axMax

                startBeam.iMax = np.max(startBeam.Jss[good] + startBeam.Jpp[good])
                self.iMax = max(self.iMax, startBeam.iMax)
                newColorMax = max(np.max(
                    self.getColor(startBeam)[good]),
                    newColorMax)
                newColorMin = min(np.min(
                    self.getColor(startBeam)[good]),
                    newColorMin)

                self.init_beam_footprint(startBeam)

        if self.newColorAxis:
            if newColorMin != self.colorMin:
                self.colorMin = newColorMin
                self.selColorMin = self.colorMin
            if newColorMax != self.colorMax:
                self.colorMax = newColorMax
                self.selColorMax = self.colorMax


        tmpMaxLen = np.max(tmpMax - tmpMin)
        if tmpMaxLen > maxLen:
            maxLen = tmpMaxLen
        self.maxLen = maxLen
        self.newColorAxis = False

        for oeString in self.oesList: #self.oesToPlot:
            oeToPlot = self.oesList[oeString][0]
#            print(oeToPlot.name)
            oeuuid = oeToPlot.uuid
#                is2ndXtal = self.oesList[oeString][3]

            if hasattr(oeToPlot, 'material'):  # real oes, no screens or slits (yet)
#                print(oeuuid)
                if not hasattr(oeToPlot, 'mesh3D'):
                    oeToPlot.mesh3D = OEMesh3D(oeToPlot, self)


                is2ndXtalOpts = [False]
                if isinstance(oeToPlot, roes.DCM):
                    is2ndXtalOpts.append(True)

                for is2ndXtal in is2ndXtalOpts:
                    oeToPlot.mesh3D.prepareSurfaceMesh(is2ndXtal)
                    oeToPlot.mesh3D.isEnabled = True
                    if oeuuid not in self.selectableOEs.values():
                        if len(self.selectableOEs):
                            stencilNum = np.max(list(self.selectableOEs.keys()))+1
                        else:
                            stencilNum = 1
                        self.selectableOEs[int(stencilNum)] = oeuuid
                        oeToPlot.mesh3D.stencilNum = stencilNum
#        counter = 0
#        for beamName, startBeam in self.beamsDict.items():
#            if counter > 1:
#                break
##            print(startBeam, hasattr(startBeam, 'vbo'))
#            beamHist, ind_array = self.calculate_fp_hist(startBeam, 256, 256)
#            _, nphist2d, _ = self.build_histRGB(startBeam, startBeam)
#
#            plt.figure(f"{counter}_2d")
#            plt.imshow(beamHist)
#
#            plt.figure(f"{counter}_2d_numpy")
#            plt.imshow(nphist2d)
#
##            plt.figure(f"{counter}_hist")
##            plt.plot(ind_array)
#            counter += 1
#
#
#        plt.show()

        gl.glViewport(*self.viewPortGL)

    def resizeGL(self, widthInPixels, heightInPixels):
        self.viewPortGL = [0, 0, widthInPixels, heightInPixels]
        gl.glViewport(*self.viewPortGL)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

        self.mProj.setToIdentity()
        self.mProj.perspective(self.cameraAngle, self.aspect, 0.01, 1000)


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

#    def mouseMoveEvent(self, mEvent):
#        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
#        mouseX = mEvent.x()
#        mouseY = pView[3] - mEvent.y()
#        ctrlOn = bool(int(mEvent.modifiers()) & int(qt.ControlModifier))
#        altOn = bool(int(mEvent.modifiers()) & int(qt.AltModifier))
#        shiftOn = bool(int(mEvent.modifiers()) & int(qt.ShiftModifier))
#
#        if mEvent.buttons() == qt.LeftButton:
#            gl.glMatrixMode(gl.GL_MODELVIEW)
#            gl.glLoadIdentity()
#            gl.gluLookAt(self.cameraPos[0], self.cameraPos[1],
#                         self.cameraPos[2],
#                         self.cameraTarget[0], self.cameraTarget[1],
#                         self.cameraTarget[2],
#                         0.0, 0.0, 1.0)
#            self.rotateZYX()
#            pModel = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
#            gl.glMatrixMode(gl.GL_PROJECTION)
#            gl.glLoadIdentity()
#
#            if self.perspectiveEnabled:
#                gl.gluPerspective(self.cameraAngle, self.aspect, 0.01, 100)
#            else:
#                orthoView = self.cameraPos[0]*0.45
#                gl.glOrtho(-orthoView*self.aspect, orthoView*self.aspect,
#                           -orthoView, orthoView, -100, 100)
#            pProjection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
#            pm_list = pModel.flatten().tolist()
#            pp_list = pProjection.flatten().tolist()
##            pv_list = pView.flatten().tolist()
##            print(len(pm_list), len(pp_list), len(pv_list))
#
#            self.mMod = qt.QMatrix4x4(*pm_list)  # MODVIEW
#            self.mProj = qt.QMatrix4x4(*pp_list)
##            self.mView = qt.QMatrix4x4(*pv_list)
#
##            self.mView.setToIdentity()
##            self.mView.lookAt(qt.QVector3D(*self.cameraPos),
##                              qt.QVector3D(*self.cameraTarget),
##                              self.upVec)
#
#            if mEvent.modifiers() == qt.NoModifier:
#                self.rotations[2][0] += np.float32(
#                    self.signs[2][1] *
#                    (mouseX - self.prevMPos[0]) * 36. / 90.)
#                self.rotations[1][0] -= np.float32(
#                    (mouseY - self.prevMPos[1]) * 36. / 90.)
#                for ax in range(2):
#                    if self.rotations[self.visibleAxes[ax+1]][0] > 180:
#                        self.rotations[self.visibleAxes[ax+1]][0] -= 360
#                    if self.rotations[self.visibleAxes[ax+1]][0] < -180:
#                        self.rotations[self.visibleAxes[ax+1]][0] += 360
#                self.updateQuats()
#                self.rotationUpdated.emit(self.rotations)
#
#            elif shiftOn:
#                for iDim in range(2):
#                    mStart = np.zeros(3)
#                    mEnd = np.zeros(3)
#                    mEnd[self.visibleAxes[iDim]] = 1.
##                    mEnd = -1 * mStart
#                    pStart = np.array(gl.gluProject(
#                        *mStart, model=pModel, proj=pProjection,
#                        view=pView)[:-1])
#                    pEnd = np.array(gl.gluProject(
#                        *mEnd, model=pModel, proj=pProjection,
#                        view=pView)[:-1])
#                    pScr = np.array([mouseX, mouseY])
#                    prevPScr = np.array(self.prevMPos)
#                    bDir = pEnd - pStart
#                    pProj = pStart + np.dot(pScr - pStart, bDir) /\
#                        np.dot(bDir, bDir) * bDir
#                    pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
#                        np.dot(bDir, bDir) * bDir
#                    self.tVec[self.visibleAxes[iDim]] += np.dot(
#                        pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
#                        self.maxLen / self.scaleVec[self.visibleAxes[iDim]]
#                    if ctrlOn and self.virtScreen is not None:
#                        self.virtScreen.center[self.visibleAxes[iDim]] -=\
#                            np.dot(
#                            pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
#                            self.maxLen / self.scaleVec[self.visibleAxes[iDim]]
#                if ctrlOn and self.virtScreen is not None:
#                    v0 = self.virtScreen.center
#                    self.positionVScreen()
#                    self.tVec -= self.virtScreen.center - v0
#
#            elif altOn:
#                mStart = np.zeros(3)
#                mEnd = np.zeros(3)
#                mEnd[self.visibleAxes[2]] = 1.
##                    mEnd = -1 * mStart
#                pStart = np.array(gl.gluProject(
#                    *mStart, model=pModel, proj=pProjection,
#                    view=pView)[:-1])
#                pEnd = np.array(gl.gluProject(
#                    *mEnd, model=pModel, proj=pProjection,
#                    view=pView)[:-1])
#                pScr = np.array([mouseX, mouseY])
#                prevPScr = np.array(self.prevMPos)
#                bDir = pEnd - pStart
#                pProj = pStart + np.dot(pScr - pStart, bDir) /\
#                    np.dot(bDir, bDir) * bDir
#                pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
#                    np.dot(bDir, bDir) * bDir
#                self.tVec[self.visibleAxes[2]] += np.dot(
#                    pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
#                    self.maxLen / self.scaleVec[self.visibleAxes[2]]
#                if ctrlOn and self.virtScreen is not None:
#                    self.virtScreen.center[self.visibleAxes[2]] -=\
#                        np.dot(pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
#                        self.maxLen / self.scaleVec[self.visibleAxes[2]]
#                    v0 = self.virtScreen.center
#                    self.positionVScreen()
#                    self.tVec -= self.virtScreen.center - v0
#
#            elif ctrlOn:
#                if self.virtScreen is not None:
#
#                    worldPStart = self.modelToWorld(
#                        self.virtScreen.beamStart - self.coordOffset)
#                    worldPEnd = self.modelToWorld(
#                        self.virtScreen.beamEnd - self.coordOffset)
#
#                    worldBDir = worldPEnd - worldPStart
#
#                    normPEnd = worldPStart + np.dot(
#                        np.ones(3) - worldPStart, worldBDir) /\
#                        np.dot(worldBDir, worldBDir) * worldBDir
#
#                    normPStart = worldPStart + np.dot(
#                        -1. * np.ones(3) - worldPStart, worldBDir) /\
#                        np.dot(worldBDir, worldBDir) * worldBDir
#
#                    normBDir = normPEnd - normPStart
#                    normScale = np.sqrt(np.dot(normBDir, normBDir) /
#                                        np.dot(worldBDir, worldBDir))
#
#                    if np.dot(normBDir, worldBDir) < 0:
#                        normPStart, normPEnd = normPEnd, normPStart
#
#                    pStart = np.array(gl.gluProject(
#                        *normPStart, model=pModel, proj=pProjection,
#                        view=pView)[:-1])
#                    pEnd = np.array(gl.gluProject(
#                        *normPEnd, model=pModel, proj=pProjection,
#                        view=pView)[:-1])
#                    pScr = np.array([mouseX, mouseY])
#                    prevPScr = np.array(self.prevMPos)
#                    bDir = pEnd - pStart
#                    pProj = pStart + np.dot(pScr - pStart, bDir) /\
#                        np.dot(bDir, bDir) * bDir
#                    pPrevProj = pStart + np.dot(prevPScr - pStart, bDir) /\
#                        np.dot(bDir, bDir) * bDir
#                    self.virtScreen.center += normScale * np.dot(
#                        pProj - pPrevProj, bDir) / np.dot(bDir, bDir) *\
#                        (self.virtScreen.beamEnd - self.virtScreen.beamStart)
#                    self.positionVScreen()
#
#            self.glDraw()
#        self.prevMPos[0] = mouseX
#        self.prevMPos[1] = mouseY
#
#    def wheelEvent(self, wEvent):
#        ctrlOn = bool(int(wEvent.modifiers()) & int(qt.ControlModifier))
#        altOn = bool(int(wEvent.modifiers()) & int(qt.AltModifier))
#        if qt.QtName == "PyQt4":
#            deltaA = wEvent.delta()
#        else:
#            deltaA = wEvent.angleDelta().y() + wEvent.angleDelta().x()
#
#        if deltaA > 0:
#            if altOn:
#                self.vScreenSize *= 1.1
#            elif ctrlOn:
#                self.cameraPos *= 0.9
#            else:
#                self.scaleVec *= 1.1
#        else:
#            if altOn:
#                self.vScreenSize *= 0.9
#            elif ctrlOn:
#                self.cameraPos *= 1.1
#            else:
#                self.scaleVec *= 0.9
#        if not ctrlOn:
#            self.scaleUpdated.emit(self.scaleVec)
#        self.glDraw()

    def mouseMoveEvent(self, mEvent):

        xView = self.viewPortGL[2]
        yView = self.viewPortGL[3]
        mouseX = mEvent.x()
        mouseY = yView - mEvent.y()
        self.makeCurrent()
        outStencil = gl.glReadPixels(
                mouseX, mouseY-1, 1, 1, gl.GL_STENCIL_INDEX, gl.GL_UNSIGNED_INT)
        overOE = np.squeeze(np.array(outStencil))

        ctrlOn = bool(int(mEvent.modifiers()) & int(qt.ControlModifier))
        altOn = bool(int(mEvent.modifiers()) & int(qt.AltModifier))
        shiftOn = bool(int(mEvent.modifiers()) & int(qt.ShiftModifier))
        polarAx = qt.QVector3D(0, 0, 1)

        dx = mouseX - self.prevMPos[0]
        dy = mouseY - self.prevMPos[1]

        xs = 2*dx/xView
        ys = 2*dy/yView
        xsn = xs*np.tan(np.radians(60))  # divide by near clipping plane
        ysn = ys*np.tan(np.radians(60))
        ym = xsn*3.5  # dist to cam, multiply by near clipping plane
        zm = ysn*3.5


        if mEvent.buttons() == qt.LeftButton:
            if mEvent.modifiers() == qt.NoModifier:
                cR = self.cameraPos.length()
                axR = self.aPos[0]*0.707
                scale = np.tan(np.radians(self.cameraAngle))*(cR-axR)

                yaw = self.aspect*dx/xView*scale/np.pi/axR #dx / xView / cR
                roll = dy/yView*scale/np.pi/axR #dy / yView /cR
                QXY = qt.QQuaternion().fromAxisAndAngle(polarAx, -np.degrees(yaw))
                self.cameraPos = QXY.rotatedVector(self.cameraPos)
                rotAx = qt.QVector3D().crossProduct(polarAx, self.cameraPos.normalized())
                QXR = qt.QQuaternion().fromAxisAndAngle(rotAx, np.degrees(roll))
                self.cameraPos = QXR.rotatedVector(self.cameraPos)

                self.mView.setToIdentity()
                self.mView.lookAt(self.cameraPos, self.cameraTarget, self.upVec)

                pModel = np.array(self.mView.data()).reshape(4, 4)[:-1, :-1]
                newVisAx = np.argmax(pModel, axis=0)
                if len(np.unique(newVisAx)) == 3:
                    self.visibleAxes = newVisAx
                self.cBox.update_grid()

            elif shiftOn:
                yShift = ym*self.maxLen/self.scaleVec[1]
                zShift = zm*self.maxLen/self.scaleVec[2]
                self.tVec[1] += yShift
                self.tVec[2] += zShift
                self.cBox.update_grid()

            self.doneCurrent()
            self.glDraw()
        else:
            if overOE != self.selectedOE:
                self.selectedOE = int(overOE)
                self.doneCurrent()
                self.glDraw()
        self.prevMPos[0] = mouseX
        self.prevMPos[1] = mouseY

    def mouseDoubleClickEvent(self, mdcevent):
        pass
#        if self.selectedOE > 0:
#            oeSel = self.parent.beamLine.oesDict[self.selectableOEs[int(
#                    self.selectedOE)]][0]
#            self.openElViewer.emit([oeSel])

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
                self.cameraAngle *= 0.9
#                self.cameraPos *= 0.9
            else:
                self.scaleVec *= 1.1
        else:
            if altOn:
                self.vScreenSize *= 0.9
            elif ctrlOn:
                self.cameraAngle *= 1.1
            else:
                self.scaleVec *= 0.9

        if ctrlOn:
            self.mProj.setToIdentity()
            self.mProj.perspective(self.cameraAngle, self.aspect, 0.01, 1000)
        else:
            self.scaleUpdated.emit(self.scaleVec)
        self.cBox.update_grid()
        self.glDraw()



class Beam3D():

    vertex_source = '''
    #version 430 core

    layout(location = 0) in vec3 position_start;
    layout(location = 4) in vec3 position_end;

    layout(location = 1) in float colorAxis;
    layout(location = 2) in float state;
    layout(location = 3) in float intensity;

    uniform sampler1D hsvTexture;

    uniform float opacity;
    uniform float iMax;
    uniform vec2 colorMinMax;

    uniform mat4 mPV;
    uniform mat4 model;
    
    uniform vec4 gridMask;
    uniform vec4 gridProjection;

    out vec4 vs_out_start;
    out vec4 vs_out_end;
    out vec4 vs_out_color;

    float hue;
    float intensity_v;
    vec4 hrgb;

    void main()
    {

     vs_out_start = mPV * (gridMask * (model * vec4(position_start, 1.0)) + gridProjection);
     vs_out_end = mPV * (gridMask * (model * vec4(position_end, 1.0)) + gridProjection);

     hue = (colorAxis - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
     intensity_v = opacity*intensity/iMax;
     hrgb = vec4(texture(hsvTexture, hue*0.85).rgb, intensity_v);
     vs_out_color = hrgb;

    }
    '''

    geometry_source = '''
    #version 430 core

    layout(points) in;
    layout(line_strip, max_vertices = 2) out;

    in vec4 vs_out_start[];
    in vec4 vs_out_end[];
    in vec4 vs_out_color[];

    uniform float pointSize;

    out vec4 gs_out_color;

    void main() {

        gl_Position = vs_out_start[0];
        gl_PointSize = pointSize;
        gs_out_color = vs_out_color[0];
        EmitVertex();

        gl_Position = vs_out_end[0];
        gs_out_color = vs_out_color[0];
        EmitVertex();

        EndPrimitive();
    }
    '''

    compute_source = '''
    #version 430

    layout(local_size_x = 32) in;

    layout(std430, binding = 0) buffer VertexBuffer {
        vec3 position[];
    };

    layout(std430, binding = 1) buffer ColorBuffer {
        float color[];
    };

    layout(std430, binding = 2) buffer StateBuffer {
        float state[];
    };

    layout(std430, binding = 3) buffer IntensityBuffer {
        float intensity[];
    };

    layout(std430, binding = 4) buffer RedBuffer {
        int red[];
    };

    layout(std430, binding = 5) buffer GreenBuffer {
        int green[];
    };

    layout(std430, binding = 6) buffer BlueBuffer {
        int blue[];
    };

    layout(std430, binding = 7) buffer IndexBuffer {
        uint ind_out[];
    };

    uniform sampler1D hsvTexture;

    uniform vec2 numBins;
    uniform vec4 bounds; // Min and max for X, Y as [[xmin, ymin], [xmax, ymax]]
    uniform vec2 colorMinMax;
    uniform float iMax;

    vec3 rgb_color;

    void main(void) {
        uint idx = gl_GlobalInvocationID.x;

        vec2 normalized = (position[idx].xy - bounds.xy) / (bounds.zw - bounds.xy);
        //ivec2 binIndex = ivec2(clamp(normalized * numBins, vec2(0, 0), numBins - vec2(1., 1.)));
        vec2 binIndex = vec2(normalized.x * numBins.x, normalized.y * numBins.y);
        uint flatIndex = uint(binIndex.x) + uint(binIndex.y * numBins.x);

        float hue = (color[idx] - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
        //float hue = 0.5;
        if (state[idx] > 0) {
                //rgb_color = vec3(1, 1, 1);
                rgb_color =  intensity[idx] / iMax * 10000. * texture(hsvTexture, hue*0.85).rgb;
        } else {
                //rgb_color = vec3(1, 1, 1);
                rgb_color = vec3(0, 0, 0);
        };

        atomicAdd(red[flatIndex], int(rgb_color.x));
        atomicAdd(green[flatIndex], int(rgb_color.y));
        atomicAdd(blue[flatIndex], int(rgb_color.z));
        ind_out[idx] = flatIndex;
    }
    '''

    fragment_source = '''
    #version 430 core

    in vec4 gs_out_color;

    out vec4 fragColor;

    void main()
    {
      fragColor = gs_out_color;
    }
    '''

    vertex_source_point = '''
    #version 430 core

    layout(location = 0) in vec3 position_start;
    layout(location = 1) in float colorAxis;
    layout(location = 2) in float state;
    layout(location = 3) in float intensity;

    uniform sampler1D hsvTexture;

    uniform float opacity;
    uniform float iMax;
    uniform vec2 colorMinMax;

    uniform mat4 mPV;
    uniform mat4 model;
    
    uniform float pointSize;
    uniform vec4 gridMask;
    uniform vec4 gridProjection;

    out vec4 vs_out_color;

    float hue;
    vec4 hrgb;

    void main()
    {

     gl_Position = mPV * (gridMask * (model * vec4(position_start, 1.0)) + gridProjection);
     gl_PointSize = pointSize;

     hue = (colorAxis - colorMinMax.x) / (colorMinMax.y - colorMinMax.x);
     hrgb = vec4(texture(hsvTexture, hue).rgb, opacity*intensity/iMax);
     vs_out_color = hrgb;

    }
    '''

    fragment_source_point = '''
    #version 430 core

    in vec4 vs_out_color;

    out vec4 fragColor;

    void main()
    {

      fragColor = vs_out_color;

    }
    '''


class OEMesh3D():
    """Container for an optical element mesh"""

    vertex_source = '''
    #version 430 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normals;

    out vec4 w_position;  // position of the vertex (and fragment) in world space
    out vec3 varyingNormalDirection;  // surface normal vector in world space
    out vec3 localPos;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    uniform mat3 m_3x3_inv_transp;
    //varying vec2 texUV;

    void main()
    {
      localPos = position;
      w_position = model * vec4(position, 1.);
      varyingNormalDirection = normalize(m_3x3_inv_transp * normals);

      //mat4 mvp = projection*view*model;
      gl_Position = projection*view*model * vec4(position, 1.);

    }
    '''

    fragment_source = '''
    #version 430 core

    in vec4 w_position;  // position of the vertex (and fragment) in world space
    in vec3 varyingNormalDirection;  // surface normal vector in world space
    in vec3 localPos;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    uniform mat4 v_inv;
    uniform vec2 texlimitsx;
    uniform vec2 texlimitsy;
    uniform vec2 texlimitsz;
    uniform sampler2D u_texture;
    uniform float opacity;

    out vec4 fragColor;

    vec2 texUV;
    vec4 histColor;

    struct lightSource
    {
      vec4 position;
      vec4 diffuse;
      vec4 specular;
      float constantAttenuation, linearAttenuation, quadraticAttenuation;
      float spotCutoff, spotExponent;
      vec3 spotDirection;
    };

    lightSource light0 = lightSource(
      vec4(0.0,  0.0,  3.0, 0.0),
      vec4(0.6,  0.6,  0.6, 1.0),
      vec4(1.0,  1.0,  1.0, 1.0),
      0.0, 1.0, 0.0,
      90.0, 0.0,
      vec3(0.0, 0.0, -1.0)
    );
    vec4 scene_ambient = vec4(0.5, 0.5, 0.5, 1.0);

    struct material
    {
      vec4 ambient;
      vec4 diffuse;
      vec4 specular;
      float shininess;
    };

    uniform material frontMaterial;
//        material frontMaterial = material(
//          vec4(0.2, 0.2, 0.2, 1.0),
//          vec4(1.0, 0.8, 0.8, 1.0),
//          vec4(1.0, 1.0, 1.0, 1.0),
//          100.0
//        );

    void main()
    {
      vec3 normalDirection = normalize(varyingNormalDirection);
      vec3 viewDirection = normalize(vec3(v_inv * vec4(0.0, 0.0, 0.0, 1.0) - w_position));
      vec3 lightDirection;
      float attenuation;

      if (0.0 == light0.position.w) // directional light?
        {
          attenuation = 1.0; // no attenuation
          lightDirection = normalize(vec3(light0.position));
        }
      else // point light or spotlight (or other kind of light)
        {
          vec3 positionToLightSource = -viewDirection;
          //vec3 positionToLightSource = vec3(light0.position - w_position);
          float distance = length(positionToLightSource);
          lightDirection = normalize(positionToLightSource);
          attenuation = 1.0 / (light0.constantAttenuation
                               + light0.linearAttenuation * distance
                               + light0.quadraticAttenuation * distance * distance);

          if (light0.spotCutoff <= 90.0) // spotlight?
    	{
    	  float clampedCosine = max(0.0, dot(-lightDirection, light0.spotDirection));
    	  if (clampedCosine < cos(radians(light0.spotCutoff))) // outside of spotlight cone?
    	    {
    	      attenuation = 0.0;
    	    }
    	  else
    	    {
    	      attenuation = attenuation * pow(clampedCosine, light0.spotExponent);
    	    }
    	}
        }

      vec3 ambientLighting = vec3(scene_ambient) * vec3(frontMaterial.ambient);

      vec3 diffuseReflection = attenuation
        * vec3(light0.diffuse) * vec3(frontMaterial.diffuse)
        * max(0.0, dot(normalDirection, lightDirection));

      vec3 specularReflection;
      if (dot(normalDirection, lightDirection) < 0.0) // light source on the wrong side?
        {
          specularReflection = vec3(0.0, 0.0, 0.0); // no specular reflection
        }
      else // light source on the right side
        {
          specularReflection = attenuation * vec3(light0.specular) * vec3(frontMaterial.specular)
    	* pow(max(0.0, dot(reflect(-lightDirection, normalDirection), viewDirection)), frontMaterial.shininess);
        }
     texUV = vec2((localPos.x-texlimitsx.x)/(texlimitsx.y-texlimitsx.x),
                 (localPos.y-texlimitsy.x)/(texlimitsy.y-texlimitsy.x));
     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1 && localPos.z<texlimitsz.y && localPos.z>texlimitsz.x)
         histColor = texture(u_texture, texUV);
     else
         histColor = vec4(0, 0, 0, 0);

      //gl_FragColor = vec4(1, 1, 1, 1.0);
      fragColor = vec4(ambientLighting + diffuseReflection + specularReflection, 1.0) + histColor*opacity;
    }
    '''


    vertex_source_flat = '''
    #version 430 core

    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };

    struct Light {
        vec3 position;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
    };

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normals;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    uniform vec2 texlimitsx;
    uniform vec2 texlimitsy;
    //uniform vec2 texlimitsz;

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform Material material;
    uniform Light light;

    //uniform vec3 lightColor;

    out vec4 color_out;
    out vec2 texUV;


    void main()
    {
        vec4 worldCoord = model * vec4(position, 1.0);
        gl_Position = projection * view * worldCoord;

        vec3 ambient = light.ambient * material.ambient;
        vec3 norm = vec3(model * vec4(normals, 0));
        vec3 lightDir = vec3(0, 0, -1);
        //vec3 lightDir = normalize(lightPos - worldCoord.xyz);

        float diff = max(dot(norm, -lightDir), 0.0);
        vec3 diffuse = light.diffuse * (diff * material.diffuse);

        vec3 viewDir = normalize(viewPos - worldCoord.xyz);
        vec3 reflectDir = reflect(lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        vec3 specular = light.specular * (spec * material.specular);
        //vec3 result = ambient + diffuse + specular;
        vec3 result = ambient + specular;
        color_out = vec4(result, 1.0);

        texUV = vec2((position.x-texlimitsx.x)/(texlimitsx.y-texlimitsx.x),
                     (position.y-texlimitsy.x)/(texlimitsy.y-texlimitsy.x));
    }
    '''

    fragment_source_flat = '''
    #version 430 core

    in vec4 color_out;
    in vec2 texUV;

    uniform sampler2D u_texture;
    uniform float opacity;
    vec4 histColor;

    out vec4 fragColor;

    void main()
    {

     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1)
         histColor = texture(u_texture, texUV);
     else
         histColor = vec4(0, 0, 0, 0);

     //gl_FragColor = vec4(color_out+histColor);
     fragColor = color_out;
    }
    '''

    vertex_contour = '''
    #version 430 core
    layout(location = 0) in vec3 position;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    void main()
    {
        gl_Position = projection*view*model*vec4(position, 1.);
    }
    '''

    fragment_contour = '''
    #version 430 core
    uniform vec4 cColor;
    out vec4 fragColor;

    void main()
    {
        fragColor = cColor;
    }
    '''

    vertex_arrow = '''
    #version 430 core

    layout(location = 0) in vec3 position;

    uniform mat4 model;
    uniform mat4 projection;
    uniform mat4 view;

    void main()
    {
        gl_Position = projection*view*model*vec4(position, 1.);
    }
    '''

    fragment_arrow = '''
    #version 430 core

    uniform vec4 aColor;
    out vec4 fragColor;

    void main()
    {
        fragColor = aColor;
    }
    '''

#    geometry_source = '''
#    #version 430 core
#
#    varying vec4 color_out;
#    varying vec2 texUV;
#
#    uniform sampler2D u_texture;
#    vec4 histColor;
#
#    void main()
#    {
#
#     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1)
#         histColor = texture(u_texture, texUV);
#     else
#         histColor = vec4(0, 0, 0, 0);
#
#     //gl_FragColor = vec4(color_out+histColor);
#     gl_FragColor = color_out;
#    }
#    '''
#
#    fg_source = '''
#    #version 430 core
#
#    in vec4 color_out;
#    in vec2 texUV;
#
#    uniform sampler2D u_texture;
#    vec4 histColor;
#
#    out vec4 fragColor;
#
#    void main()
#    {
#
#     if (texUV.x>0 && texUV.x<1 && texUV.y>0 && texUV.y<1)
#         histColor = texture(u_texture, texUV);
#     else
#         histColor = vec4(0, 0, 0, 0);
#
#     //gl_FragColor = vec4(color_out+histColor);
#     fragColor = color_out;
#    }
#    '''

    def __init__(self, parentOE, parentWidget):
        self.emptyTex = qt.QOpenGLTexture(
                qt.QImage(np.zeros((256, 256, 3)),
                          256, 256, qt.QImage.Format_RGB888))
        self.defaultLimits = np.array([[-1.]*3, [1.]*3])
#        print("def shape", self.defaultLimits.shape)
#        texture.save(str(oe.name)+"_beam_hist.png")
#        if hasattr(meshObj, 'beamTexture'):
#            oe.beamTexture.setData(texture)
#        meshObj.beamTexture[nsIndex] = qg.QOpenGLTexture(texture)
#        meshObj.beamLimits[nsIndex] = beamLimits
        self.oe = parentOE
        self.parent = parentWidget
        self.isStl = False
        self.shader = {}
        self.shader_c = {}
        self.vao = {}
        self.vao_c = {}
        self.ibo = {}
        self.beamTexture = {}
        self.beamLimits = {}
        self.transMatrix = {}
        self.arrLengths = {}

        self.vbo_vertices = {}
        self.vbo_normals = {}
        self.vbo_contour = {}

        self.oeThickness = 5
        self.tiles = [25, 25]
        self.showLocalAxes = False
        self.isEnabled = False
        self.stencilNum = 0
#        self.createArrowArray(0.25, 0.02, 20)

    def updateSurfaceMesh(self, is2ndXtal=False):
        pass

    @staticmethod
    def get_loc2glo_transformation_matrix(oe, is2ndXtal=False):
#        oe = self.oe

        if isinstance(oe, roes.OE):
            dx, dy, dz = 0, 0, 0
            extraAnglesSign = 1.  # only for pitch and yaw
            if isinstance(oe, roes.DCM):
                if is2ndXtal:
                    pitch = -oe.pitch - oe.bragg + oe.cryst2pitch +\
                        oe.cryst2finePitch
                    roll = oe.roll + oe.cryst2roll + oe.positionRoll
                    yaw = -oe.yaw
                    dx = -oe.dx
                    dy = oe.cryst2longTransl
                    dz = -oe.cryst2perpTransl
                    extraAnglesSign = -1.
                else:
                    pitch = oe.pitch + oe.bragg
                    roll = oe.roll + oe.positionRoll + oe.cryst1roll
                    yaw = oe.yaw
                    dx = oe.dx
            else:
                pitch = oe.pitch
                roll = oe.roll + oe.positionRoll
                yaw = oe.yaw

            rotAx = {'x': pitch,
                     'y': roll,
                     'z': yaw}
            extraRotAx = {'x': extraAnglesSign*oe.extraPitch,
                          'y': oe.extraRoll,
                          'z': extraAnglesSign*oe.extraYaw}

            rotSeq = (oe.rotationSequence[slice(1, None, 2)])[::-1]
            extraRotSeq = (oe.extraRotationSequence[slice(1, None, 2)])[::-1]

#            if isinstance(oe, rscreens.Screen):
#                rotation = oe.orientationQuat
#            print("Transmatr", oe.name)

            rotation = (scprot.from_euler(
                    rotSeq, [rotAx[i] for i in rotSeq])).as_quat()
            extraRot = (scprot.from_euler(
                    extraRotSeq,
                    [extraRotAx[i] for i in extraRotSeq])).as_quat()
            rotation = [rotation[-1], rotation[0], rotation[1], rotation[2]]
            extraRot = [extraRot[-1], extraRot[0], extraRot[1], extraRot[2]]
#            if hasattr(oe, 'orientationQuat'):
#                print("oe.quat", oe.orientationQuat)
#                print("tm.rot", raycing.multiply_quats(rotation, extraRot))
            # 1. Only for DCM - translate to 2nd crystal position
            m2ndXtalPos = qt.QMatrix4x4()
            m2ndXtalPos.translate(dx, dy, dz)

            # 2. Apply extra rotation
            mExtraRot = qt.QMatrix4x4()
            mExtraRot.rotate(qt.QQuaternion(*extraRot))

            # 3. Apply rotation
            mRotation = qt.QMatrix4x4()
            mRotation.rotate(qt.QQuaternion(*rotation))

            # 4. Only for DCM - flip 2nd crystal
            m2ndXtalRot = qt.QMatrix4x4()
            if isinstance(oe, roes.DCM):
                if is2ndXtal:
                    m2ndXtalRot.rotate(180, 0, 1, 0)

            # 5. Move to position in global coordinates
            mTranslation = qt.QMatrix4x4()
            mTranslation.translate(*oe.center)

#            if isinstance(oe, roes.DCM):
#                print(oe.name, m2ndXtalRot*mRotation, is2ndXtal)

            return mTranslation*m2ndXtalRot*mRotation*mExtraRot*m2ndXtalPos
        else:
            posMatr = qt.QMatrix4x4()
            posMatr.translate(*oe.center)
            return posMatr

#    def createArrowArray(self, z, r, nSegments):
#        phi = np.linspace(0, 2*np.pi, nSegments)
#        xp = r * np.cos(phi)
#        yp = r * np.sin(phi)
#        base = np.vstack((xp, yp, np.zeros_like(xp)))
#        coneVertices = np.hstack((np.array([0, 0, 0, 0, 0, z]).reshape(3, 2),
#                                  base)).T
#        self.arrowLen = len(coneVertices)
##        print(coneVertices, coneVertices.shape)
#
#        vao = qg.QOpenGLVertexArrayObject()
#        vao.create()
#        shader = qg.QOpenGLShaderProgram()
#        shader.addShaderFromSourceCode(
#                qt.QOpenGLShader.Vertex, self.vertex_arrow)
#        shader.addShaderFromSourceCode(
#                qt.QOpenGLShader.Fragment, self.fragment_arrow)
#
#        if not shader.link():
#            print("Linking Error", str(shader.log()))
#            print('Failed to link dummy renderer shader!')
#
#        shader.bind()
#        vao.bind()
#
#        self.vbo_arrow = setVertexBuffer(coneVertices.copy(), 3,
#                                         shader, "position")
#        vao.release()
#        shader.release()
#        self.shader_arrow = shader
#        self.vao_arrow = vao

    def prepareSurfaceMesh(self, is2ndXtal=False, updateMesh=False, shader=None):
        def getThickness():
#            if self.oeThicknessForce is not None:
#                return self.oeThicknessForce
            thickness = self.oeThickness
            if isinstance(self.oe, roes.Plate):
                if self.oe.t is not None:
                    thickness = self.oe.t
                    if hasattr(self.oe, 'zmax'):
                        if self.oe.zmax is not None:
                            thickness += self.oe.zmax
                            if isinstance(self.oe, roes.DoubleParaboloidLens):
                                thickness += self.oe.zmax
                    return thickness
            if hasattr(self.oe, "material"):
                if self.oe.material is not None:
                    thickness = self.oeThickness
                    if hasattr(self.oe.material, "t"):
                        thickness = self.oe.material.t if\
                            self.oe.material.t is not None else thickness
                    elif isinstance(self.oe.material, rmats.Multilayer):
                        if self.oe.material.substrate is not None:
                            if hasattr(self.oe.material.substrate, 't'):
                                if self.oe.material.substrate.t is not None:
                                    thickness = self.oe.material.substrate.t
            return thickness

        nsIndex = int(is2ndXtal)
        self.transMatrix[nsIndex] = self.get_loc2glo_transformation_matrix(
                self.oe, is2ndXtal=is2ndXtal)

        if nsIndex in self.vao.keys():
            vao = self.vao[nsIndex]
        else:
#            print("Creating VAO")
            vao = qt.QOpenGLVertexArrayObject()
            vao.create()

#        if nsIndex in self.shader.keys():
#            shader = self.shader[nsIndex]
#        else:
#            shader = qg.QOpenGLShaderProgram()
#            shader.addShaderFromSourceCode(
#                    qt.QOpenGLShader.Vertex, self.vertex_source)
#            shader.addShaderFromSourceCode(
#                    qt.QOpenGLShader.Fragment, self.fragment_source)
#
#            if not shader.link():
#                print("Linking Error", str(shader.log()))
#                print('Failed to link dummy renderer shader!')

        if hasattr(self.oe, 'stl_mesh'):
            vao.bind()
            shader.bind()  # Will fail here if shader is none

            self.isStl = True
            self.vbo_vertices[nsIndex] = setVertexBuffer(
                    self.oe.stl_mesh[0].copy(), 3, shader, "position")
            self.vbo_normals[nsIndex] = setVertexBuffer(
                    self.oe.stl_mesh[1].copy(), 3, shader, "normals")
            self.arrLengths[nsIndex] = len(self.oe.stl_mesh[0])

            shader.release()
            vao.release()

            self.vao[nsIndex] = vao
            self.ibo[nsIndex] = None
            return

        isPlate = isinstance(self.oe, roes.Plate)

        thickness = getThickness()

        self.bBox = np.zeros((3, 2))
        self.bBox[:, 0] = 1e10
        self.bBox[:, 1] = -1e10

        # TODO: Consider plates

        if is2ndXtal:
            xLimits = list(self.oe.limPhysX2)
            yLimits = list(self.oe.limPhysY2)
        else:
            xLimits = list(self.oe.limPhysX)
            yLimits = list(self.oe.limPhysY)

        isClosedSurface = False
        if np.any(np.abs(xLimits) == raycing.maxHalfSizeOfOE):
            isClosedSurface = isinstance(self.oe, roes.SurfaceOfRevolution)
            if len(self.oe.footprint) > 0:
                xLimits = self.oe.footprint[nsIndex][:, 0]
        if np.any(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
            if len(self.oe.footprint) > 0:
                yLimits = self.oe.footprint[nsIndex][:, 1]

        localTiles = np.array(self.tiles)

        if self.oe.shape == 'round':
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

        xGridOe = np.linspace(xLimits[0], xLimits[1],
                              localTiles[0]) + self.oe.dx
        yGridOe = np.linspace(yLimits[0], yLimits[1], localTiles[1])

        xv, yv = np.meshgrid(xGridOe, yGridOe)

        sideL = np.vstack((xv[:, 0], yv[:, 0])).T
        sideR = np.vstack((xv[:, -1], yv[:, -1])).T
        sideF = np.vstack((xv[0, :], yv[0, :])).T
        sideB = np.vstack((xv[-1, :], yv[-1, :])).T

        if self.oe.shape == 'round':
            xv, yv = rX*xv*np.cos(yv)+cX, rY*xv*np.sin(yv)+cY

        xv = xv.flatten()
        yv = yv.flatten()

        if is2ndXtal:
            zExt = '2'
        else:
            zExt = '1' if hasattr(self.oe, 'local_z1') else ''
        local_z = getattr(self.oe, 'local_r{}'.format(zExt)) if\
            self.oe.isParametric else getattr(self.oe,
                                              'local_z{}'.format(zExt))
        local_n = getattr(self.oe, 'local_n{}'.format(zExt))

        xv = np.copy(xv)
        yv = np.copy(yv)
        zv = np.zeros_like(xv)
        if isinstance(self.oe, roes.SurfaceOfRevolution):
            # at z=0 (axis of rotation) phi is undefined, therefore:
            zv -= 100.

        if self.oe.isParametric and not isClosedSurface:
            xv, yv, zv = self.oe.xyz_to_param(xv, yv, zv)

        zv = np.array(local_z(xv, yv))
        nv = np.array(local_n(xv, yv)).T

        if len(nv) == 3:  # flat
            nv = np.ones_like(zv)[:, np.newaxis] * np.array(nv)

        if self.oe.isParametric:
            xv, yv, zv = self.oe.param_to_xyz(xv, yv, zv)

#        zmax = np.max(zv)
#        zmin =
#        self.bBox[:, 1] = yLimit

        if self.oe.shape == 'round':
            xC, yC = rX*sideR[:, 0]*np.cos(sideR[:, -1]) +\
                     cX, rY*sideR[:, 0]*np.sin(sideR[:, -1]) + cY
            zC = np.array(local_z(xC, yC))
            if self.oe.isParametric:
                xC, yC, zC = self.oe.param_to_xyz(xC, yC, zC)

        points = np.vstack((xv, yv, zv)).T
        surfmesh = {}

        triS = Delaunay(points[:, :-1])

        if not isPlate:
            bottomPoints = points.copy()
            bottomPoints[:, 2] = -thickness
            bottomNormals = np.zeros((len(points), 3))
            bottomNormals[:, 2] = -1

        zL = np.array(local_z(sideL[:, 0], sideL[:, -1]))
        zR = np.array(local_z(sideR[:, 0], sideR[:, -1]))
        zF = np.array(local_z(sideF[:, 0], sideF[:, -1]))
        zB = np.array(local_z(sideB[:, 0], sideB[:, -1]))

        tL = np.vstack((sideL.T, np.ones_like(zL)*thickness))
        bottomLine = zL - thickness if isPlate else -np.ones_like(zL)*thickness
        tL = np.hstack((tL, np.vstack((np.flip(sideL.T, axis=1),
                                       -np.ones_like(zL)*thickness)))).T
        normsL = np.zeros((len(zL)*2, 3))
        normsL[:, 0] = -1
        triLR = Delaunay(tL[:, [1, -1]])  # Used for round elements also
        tL[:len(zL), 2] = zL
        tL[len(zL):, 2] = bottomLine

        tR = np.vstack((sideR.T, zR))
        bottomLine = zR - thickness if isPlate else -np.ones_like(zR)*thickness
        tR = np.hstack((tR, np.vstack((np.flip(sideR.T, axis=1),
                                       bottomLine)))).T
        normsR = np.zeros((len(zR)*2, 3))
        normsR[:, 0] = 1

        tF = np.vstack((sideF.T, np.ones_like(zF)*thickness))
        bottomLine = zF - thickness if isPlate else -np.ones_like(zF)*thickness
        tF = np.hstack((tF, np.vstack((np.flip(sideF.T, axis=1),
                                       bottomLine)))).T
        normsF = np.zeros((len(zF)*2, 3))
        normsF[:, 1] = -1
        triFB = Delaunay(tF[:, [0, -1]])
        tF[:len(zF), 2] = zF

        if self.oe.shape == 'round':
            tB = np.vstack((xC, yC, zC))
            bottomLine = zC - thickness if isPlate else\
                -np.ones_like(zC)*thickness
            tB = np.hstack((tB, np.vstack((xC, np.flip(yC), bottomLine)))).T
            normsB = np.vstack((tB[:, 0], tB[:, 1], np.zeros_like(tB[:, 0]))).T
            norms = np.linalg.norm(normsB, axis=1, keepdims=True)
            normsB /= norms
        else:
            tB = np.vstack((sideB.T, zB))
            bottomLine = zB - thickness if isPlate else\
                -np.ones_like(zB)*thickness
            tB = np.hstack((tB, np.vstack((np.flip(sideB.T, axis=1),
                                           bottomLine)))).T
            normsB = np.zeros((len(zB)*2, 3))
            normsB[:, 1] = 1

        allSurfaces = points
        allNormals = nv
        allIndices = triS.simplices.flatten()
        indArrOffset = len(points)

        # Bottom Surface, use is2ndXtal for plates
        if not isPlate:
            allSurfaces = np.vstack((allSurfaces, bottomPoints))
            allNormals = np.vstack((nv, bottomNormals))
            allIndices = np.hstack((allIndices,
                                    triS.simplices.flatten() + indArrOffset))
            indArrOffset += len(points)

        # Side Surface, do not plot for 2ndXtal of Plate
        if not (isPlate and is2ndXtal):
            if self.oe.shape == 'round':  # Side surface
                allSurfaces = np.vstack((allSurfaces, tB))
                allNormals = np.vstack((allNormals, normsB))
                allIndices = np.hstack((allIndices,
                                        triLR.simplices.flatten() +
                                        indArrOffset))
            else:
                allSurfaces = np.vstack((allSurfaces, tL, tF, tR, tB))
                allNormals = np.vstack((allNormals, normsL, normsF,
                                        normsR, normsB))
                allIndices = np.hstack((allIndices,
                                        triLR.simplices.flatten() +
                                        indArrOffset,
                                        triFB.simplices.flatten() +
                                        indArrOffset+len(tL),
                                        triLR.simplices.flatten() +
                                        indArrOffset+len(tL)+len(tF),
                                        triFB.simplices.flatten() +
                                        indArrOffset+len(tL)*2+len(tF)))

        surfmesh['points'] = allSurfaces.copy()
        surfmesh['normals'] = allNormals.copy()
        surfmesh['indices'] = allIndices

        if self.oe.shape == 'round':
            surfmesh['contour'] = tB
        else:
            surfmesh['contour'] = np.vstack((tL, tF, np.flip(tR, axis=0), tB))
        surfmesh['lentb'] = len(tB)

        self.bBox[:, 0] = np.min(surfmesh['contour'], axis=0)
        self.bBox[:, 1] = np.max(surfmesh['contour'], axis=0)

        oldVBOpoints = self.vbo_vertices[nsIndex] if\
            nsIndex in self.vbo_vertices.keys() else None
        oldVBOnorms = self.vbo_normals[nsIndex] if\
            nsIndex in self.vbo_normals.keys() else None
        oldIBO = self.ibo[nsIndex] if nsIndex in self.ibo.keys() else None

        if updateMesh:
            if oldVBOpoints is not None:
                oldVBOpoints.destroy()
            if oldVBOnorms is not None:
                oldVBOnorms.destroy()
            if oldIBO is not None:
                oldIBO.destroy()
            oldVBOpoints, oldVBOnorms, oldIBO = None, None, None

        self.vbo_vertices[nsIndex] = create_qt_buffer(surfmesh['points'])
        self.vbo_normals[nsIndex] = create_qt_buffer(surfmesh['normals'])
        self.ibo[nsIndex] = create_qt_buffer(surfmesh['indices'], isIndex=True)

#        self.vbo_vertices[nsIndex] = setVertexBuffer(
#                surfmesh['points'], 3, shader, "position", None, oldVBOpoints)
#        self.vbo_normals[nsIndex] = setVertexBuffer(
#                surfmesh['normals'], 3, shader, "normals", None, oldVBOnorms)
#        self.ibo[nsIndex] = setIndexBuffer(surfmesh['indices'], oldIBO)
        self.arrLengths[nsIndex] = len(surfmesh['indices'])

        vao.bind()

        self.vbo_vertices[nsIndex].bind()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Attribute 0: position
        gl.glEnableVertexAttribArray(0)
        self.vbo_vertices[nsIndex].release()

        self.vbo_normals[nsIndex].bind()
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        self.vbo_normals[nsIndex].release()

        self.ibo[nsIndex].bind()

        vao.release()

        self.vao[nsIndex] = vao
        self.z2y = qt.QMatrix4x4().rotate(90, 1, 0, 0)
        self.z2x = qt.QMatrix4x4().rotate(90, 0, 1, 0)

    def drawLocalAxes(self, mMod, mView, mProj, is2ndXtal):
        oeIndex = int(is2ndXtal)
        oeOrientation = self.transMatrix[oeIndex]
        shader = self.shader_arrow
        shader.bind()
        self.vao_arrow.bind()
        shader.setUniformValue("view", mView)
        shader.setUniformValue("projection", mProj)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        for iAx in range(3):
            color = np.array([0., 0., 0., 1.])
            color[iAx] = 1.
            colorV = qt.QVector4D(*color)
            modMatr = mMod*oeOrientation
            if iAx == 0:
                modMatr *= self.z2x
            elif iAx == 1:
                modMatr *= self.z2y

#            shader.setUniformValue("model", )
            shader.setUniformValue("aColor", colorV)
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 1, self.arrowLen-1)
#            gl.glEnable(gl.GL_LINE_SMOOTH)
#            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
#            gl.glBegin(gl.GL_LINES)
#            gl.glDrawArrays(gl.GL_LINES, 0, 2)
        self.vao_arrow.release()
        shader.release()

    def render_surface(self, mMod, mView, mProj, is2ndXtal=False,
                     isSelected=False, shader=None):

        oeIndex = int(is2ndXtal)

#        shader = self.shader[oeIndex]
        vao = self.vao[oeIndex]
#        ibo = self.ibo[oeIndex]

        beamTexture = self.beamTexture[oeIndex] if len(self.beamTexture) > 0\
            else self.emptyTex  # what if there's no texture?
        beamLimits = self.beamLimits[oeIndex] if len(self.beamLimits) > 0\
            else self.defaultLimits

        oeOrientation = self.transMatrix[oeIndex]
        arrLen = self.arrLengths[oeIndex]

        shader.bind()
        vao.bind()

        shader.setUniformValue("model", mMod*oeOrientation)
        shader.setUniformValue("view", mView)
        shader.setUniformValue("projection", mProj)

        mvp = mMod*oeOrientation*mView
        shader.setUniformValue("m_3x3_inv_transp", mvp.normalMatrix())
        shader.setUniformValue("v_inv", mView.inverted()[0])

        shader.setUniformValue("texlimitsx", qt.QVector2D(*beamLimits[:, 0]))
        shader.setUniformValue("texlimitsy", qt.QVector2D(*beamLimits[:, 1]))
        shader.setUniformValue("texlimitsz", qt.QVector2D(*beamLimits[:, 2]))

        # TODO: configurable colors
        ambient = qt.QVector4D(0.89225, 0.89225, 0.49225, 1.) if\
            isSelected else qt.QVector4D(2*0.29225, 2*0.29225, 2*0.29225, 1.)
        shader.setUniformValue("frontMaterial.ambient", ambient)
        shader.setUniformValue("frontMaterial.diffuse",
                               qt.QVector4D(0.50754, 0.50754,
                                            0.50754, 1.))
        shader.setUniformValue("frontMaterial.specular",
                               qt.QVector4D(1., 0.9, 0.8, 1.))
        shader.setUniformValue("frontMaterial.shininess", 100.)

        shader.setUniformValue("opacity", float(self.parent.pointOpacity*2))

        if beamTexture is not None:
            beamTexture.bind()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if self.isStl:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, arrLen)  #
        else:

#            ibo.bind()
#            gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xff)
            gl.glDrawElements(gl.GL_TRIANGLES, arrLen,
                              gl.GL_UNSIGNED_INT, [])
#            ibo.release()
        if beamTexture is not None:
            beamTexture.release()
        shader.release()
        vao.release()


class CoordinateBox():

    vertex_source = '''
    #version 430 core
    layout(location = 0) in vec3 position;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main()
    {
      //gl_Position = projection * model * vec4(position, 1.0);
      gl_Position = projection * view * model * vec4(position, 1.0);
    }
    '''

    fragment_source = '''
    #version 430 core
    uniform float lineOpacity;
    out vec4 fragColor;
    void main()
    {
      fragColor = vec4(1.0, 1.0, 1.0, lineOpacity);
    }
    '''

    orig_vertex_source = '''
    #version 430 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 linecolor;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 out_color;
    void main()
    {
     out_color = linecolor;
//     gl_Position = projection * model * vec4(position, 1.0);
     gl_Position = projection * view * model * vec4(position, 1.0);
    }
    '''

    orig_fragment_source = '''
    #version 430 core
    uniform float lineOpacity;
    in vec3 out_color;
    out vec4 fragColor;
    void main()
    {
      fragColor = vec4(out_color, lineOpacity);
    }
    '''

    text_vertex_code = """
    #version 430 core

    in vec4 in_pos;

    out vec2 vUV;

    uniform mat4 model;
    //uniform mat4 projection;

    void main()
    {
        vUV         = in_pos.zw;
        gl_Position = model * vec4(in_pos.xy, 0.0, 1.0);
    }
    """

    text_fragment_code = """
    #version 430 core

    in vec2 vUV;

    uniform sampler2D u_texture;

    out vec4 fragColor;

    void main()
    {
        vec2 uv = vUV.xy;
        float text = texture(u_texture, uv).r;
        fragColor = vec4(text, text, text, text);
    }
    """

    def __init__(self, parent):

        self.parent = parent
#        self.aPos = [0.9, 0.9, 0.9]
        self.axPosModifier = np.ones(3)
        self.perspectiveEnabled = True
#        self.cBoxLineWidth = 1
        self.shader = None
        self.origShader = None
        self.textShader = None
        self.vaoFrame = qt.QOpenGLVertexArrayObject()
        self.vaoFrame.create()

        self.vaoGrid = qt.QOpenGLVertexArrayObject()
        self.vaoGrid.create()

        self.vaoFineGrid = qt.QOpenGLVertexArrayObject()
        self.vaoFineGrid.create()

        self.vaoOrigin = qt.QOpenGLVertexArrayObject()
        self.vaoOrigin.create()

        self.vaoOrigin = qt.QOpenGLVertexArrayObject()
        self.vaoOrigin.create()

        self.vaoText = qt.QOpenGLVertexArrayObject()
        self.vaoText.create()

        self.characters = []
        self.fontSize = 32
        self.fontScale = 4.
        self.fontFile = 'FreeSans.ttf'
#        self.vquad = [
#          # x   y  u  v
#            0, 1, 0, 0,
#            0,  0, 0, 1,
#            1,  0, 1, 1,
#            0, 1, 0, 0,
#            1,  0, 1, 1,
#            1, 1, 1, 0
#        ]

#        self.prepare_grid()

    def make_frame(self):
        back = np.array([[-self.parent.aPos[0],
                          self.parent.aPos[1],
                          -self.parent.aPos[2]],
                         [-self.parent.aPos[0],
                          self.parent.aPos[1],
                          self.parent.aPos[2]],
                         [-self.parent.aPos[0],
                          -self.parent.aPos[1],
                          self.parent.aPos[2]],
                         [-self.parent.aPos[0],
                          -self.parent.aPos[1],
                          -self.parent.aPos[2]]])

        side = np.array([[self.parent.aPos[0],
                          -self.parent.aPos[1],
                          -self.parent.aPos[2]],
                         [-self.parent.aPos[0],
                          -self.parent.aPos[1],
                          -self.parent.aPos[2]],
                         [-self.parent.aPos[0],
                          -self.parent.aPos[1],
                          self.parent.aPos[2]],
                         [self.parent.aPos[0],
                          -self.parent.aPos[1],
                          self.parent.aPos[2]]])

        bottom = np.array([[self.parent.aPos[0],
                            -self.parent.aPos[1],
                            -self.parent.aPos[2]],
                           [self.parent.aPos[0],
                            self.parent.aPos[1],
                            -self.parent.aPos[2]],
                           [-self.parent.aPos[0],
                            self.parent.aPos[1],
                            -self.parent.aPos[2]],
                           [-self.parent.aPos[0],
                            -self.parent.aPos[1],
                            -self.parent.aPos[2]]])

        back[:, 0] *= self.axPosModifier[0]
        side[:, 1] *= self.axPosModifier[1]
        bottom[:, 2] *= self.axPosModifier[2]
        self.halfCube = np.float32(np.vstack((back, side, bottom)))

    def make_coarse_grid(self):

        self.gridLabels = []
        self.precisionLabels = []
        #  Calculating regular grids in world coordinates
        limits = np.array([-1, 1])[:, np.newaxis] * np.array(self.parent.aPos)
        allLimits = limits * self.parent.maxLen / self.parent.scaleVec -\
            self.parent.tVec + self.parent.coordOffset
        axisGridArray = []

        for iAx in range(3):
            m2 = self.parent.aPos[iAx] / 0.9
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
            self.gridLabels.extend([gridX])
            self.precisionLabels.extend([np.ones_like(gridX)*decimalX])
            axisGridArray.extend([gridX - self.parent.coordOffset[iAx]])
#            if self.parent.fineGridEnabled:
#                fineStep = step * 0.2
#                fineGrid = np.arange(
#                    np.int32(allLimits[:, iAx][0]/fineStep)*fineStep,
#                    allLimits[:, iAx][1], fineStep)
#                fineGrid = fineGrid if\
#                    fineGrid[0] >= allLimits[:, iAx][0] else fineGrid[1:]
#                fineGridArray.extend([fineGrid - self.parent.coordOffset[iAx]])

        self.axisL, self.axGrid = self.populateGrid(axisGridArray)
        self.gridLen = len(self.axGrid)

#        for iAx in range(3):
#            if not (not self.perspectiveEnabled and
#                    iAx == self.parent.visibleAxes[2]):
#
#                midp = int(len(self.axisL[iAx][0, :])/2)
#                if iAx == self.parent.visibleAxes[1]:  # Side plane,
#                    print(self.axisL[iAx][:, midp], self.parent.visibleAxes[0])
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[0])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[2], :] *= 1.05  # depth
#                    self.axisL[iAx][self.parent.visibleAxes[0], :] *= 1.05  # side
#                if iAx == self.parent.visibleAxes[0]:  # Bottom plane, left-right
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[2],
##                                              self.visibleAxes[1])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[1], :] *= 1.05  # height
#                    self.axisL[iAx][self.parent.visibleAxes[2], :] *= 1.05  # side
#                if iAx == self.parent.visibleAxes[2]:  # Bottom plane, left-right
##                    if self.useScalableFont:
##                        tAlign = getAlignment(axisL[iAx][:, midp],
##                                              self.visibleAxes[0],
##                                              self.visibleAxes[1])
##                    else:
#                    self.axisL[iAx][self.parent.visibleAxes[1], :] *= 1.05  # height
#                    self.axisL[iAx][self.parent.visibleAxes[0], :] *= 1.05  # side

    def update_grid(self):
        if hasattr(self, "vbo_frame"):
            self.make_frame()
            self.vbo_frame.bind()
            self.vbo_frame.write(0, self.halfCube, self.halfCube.nbytes)
            self.vbo_frame.release()
        if hasattr(self, "vbo_grid"):
            self.make_coarse_grid()
            self.vbo_grid.bind()
            self.vbo_grid.write(0, self.axGrid, self.axGrid.nbytes)
            self.vbo_grid.release()

    def prepare_grid(self):

        self.make_font()
        self.make_frame()
        self.make_coarse_grid()
#        if self.parent.fineGridEnabled:
#            fineGridArray = []


#        print(axisL)
#        if self.parent.fineGridEnabled:
#            tmp, fineAxGrid = self.populateGrid(fineGridArray)
#            self.fineGridLen = len(fineAxGrid)
#            self.vaoFineGrid.bind()
#            self.vbo_fineGrid = self.setVertexBuffer(fineAxGrid, 3, self.shader, "position" )
#            self.vaoFineGrid.release()

        cLines = np.array([[-self.parent.aPos[0], 0, 0],
                           [self.parent.aPos[0], 0, 0],
                           [0, -self.parent.aPos[1], 0],
                           [0, self.parent.aPos[1], 0],
                           [0, 0, -self.parent.aPos[2]],
                           [0, 0, self.parent.aPos[2]]])*0.5

        cLineColors = np.array([[0, 0.5, 1],
                                [0, 0.5, 1],
                                [0, 0.9, 0],
                                [0, 0.9, 0],
                                [0.8, 0, 0],
                                [0.8, 0, 0]])

        self.vaoFrame.bind()
        self.vbo_frame = setVertexBuffer(
                self.halfCube, 3, self.shader, "position")
        self.vaoFrame.release()

        self.vaoGrid.bind()
        self.vbo_grid = setVertexBuffer(
                self.axGrid, 3, self.shader, "position",
                size=np.float32(self.axGrid).nbytes*100)  # TODO: calculate
        self.vaoGrid.release()

        self.vaoOrigin.bind()
        self.vbo_origin = setVertexBuffer(
                cLines, 3, self.origShader, "position")
        self.vbo_oc = setVertexBuffer(
                cLineColors, 3, self.origShader, "linecolor")
        self.vaoOrigin.release()

        # x  y  u  v
        vquad = [
            0, 1, 0, 0,
            0,  0, 0, 1,
            1,  0, 1, 1,
            0, 1, 0, 0,
            1,  0, 1, 1,
            1, 1, 1, 0
        ]

        self.vaoText.bind()
        self.vbo_Text = setVertexBuffer(vquad, 4, self.textShader, "in_pos")
        self.vaoText.release()

    def populateGrid(self, grids):
        pModel = np.array(self.parent.mView.data()).reshape(4, 4)[:-1, :-1]
#                print(pModel)
#        self.visibleAxes = np.argmax(np.abs(pModel), axis=0)
        self.signs = np.sign(pModel)
#                self.axPosModifier = np.ones(3)
        for iAx in range(3):
            self.axPosModifier[iAx] = (self.signs[iAx][2] if
                                       self.signs[iAx][2] != 0 else 1)
        axisLabelC = []
        axisLabelC.extend([np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*-self.parent.aPos[2]*self.axPosModifier[2]
             ))])
        axisLabelC.extend([np.vstack(
            (np.ones(len(grids[1]))*self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*-self.parent.aPos[2]*self.axPosModifier[2]
             ))])
        zAxis = np.vstack(
            (np.ones(len(grids[2]))*-self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))

        xAxisB = np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*-self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*-self.parent.aPos[2]*self.axPosModifier[2]))
        yAxisB = np.vstack(
            (np.ones(len(grids[1]))*-self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*-self.parent.aPos[2]*self.axPosModifier[2]))
        zAxisB = np.vstack(
            (np.ones(len(grids[2]))*-self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*-self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))

        xAxisC = np.vstack(
            (self.parent.modelToWorld(grids, 0),
             np.ones(len(grids[0]))*-self.parent.aPos[1]*self.axPosModifier[1],
             np.ones(len(grids[0]))*self.parent.aPos[2]*self.axPosModifier[2]))
        yAxisC = np.vstack(
            (np.ones(len(grids[1]))*-self.parent.aPos[0]*self.axPosModifier[0],
             self.parent.modelToWorld(grids, 1),
             np.ones(len(grids[1]))*self.parent.aPos[2]*self.axPosModifier[2]))
        axisLabelC.extend([np.vstack(
            (np.ones(len(grids[2]))*self.parent.aPos[0]*self.axPosModifier[0],
             np.ones(len(grids[2]))*-self.parent.aPos[1]*self.axPosModifier[1],
             self.parent.modelToWorld(grids, 2)))])

        xLines = np.vstack(
            (axisLabelC[0], xAxisB, xAxisB, xAxisC)).T.flatten().reshape(
            4*xAxisB.shape[1], 3)
        yLines = np.vstack(
            (axisLabelC[1], yAxisB, yAxisB, yAxisC)).T.flatten().reshape(
            4*yAxisB.shape[1], 3)
        zLines = np.vstack(
            (zAxis, zAxisB, zAxisB, axisLabelC[2])).T.flatten().reshape(
            4*zAxisB.shape[1], 3)

        return axisLabelC, np.float32(np.vstack((xLines, yLines, zLines)))

#    def drawGridLines(self, gridArray, lineWidth, lineOpacity, figType):

#        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
#        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
#        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
#        gridColor = np.ones((len(gridArray), 4)) * lineOpacity
#        gridArrayVBO = gl.vbo.VBO(np.float32(gridArray))
#        gridArrayVBO.bind()
#        gl.glVertexPointerf(gridArrayVBO)
#        gridColorArray = gl.vbo.VBO(np.float32(gridColor))
#        gridColorArray.bind()
#        gl.glColorPointerf(gridColorArray)
#        gl.glLineWidth(lineWidth)
#        gl.glDrawArrays(figType, 0, len(gridArrayVBO))
#        gridArrayVBO.unbind()
#        gridColorArray.unbind()
#        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
#        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

#    def modelToWorld(self, coords, dimension=None):
#        if dimension is None:
#            return np.float32(coords)
#        else:
#            return np.float32(coords[dimension])

#    def setVertexBuffer( self, data_array, dim_vertex, program, shader_str, size=None):
#        # https://github.com/Upcios/PyQtSamples/blob/master/PyQt5/opengl/triangle_simple/main.py
#        vbo = qg.QOpenGLBuffer(qg.QOpenGLBuffer.VertexBuffer)
#        vbo.create()
#        vbo.bind()
##        print(vbo.bufferId())
#
#        vertices = np.array(data_array, np.float32)
##        print(type(vertices), vertices.shape[0], vertices.itemsize, vertices.shape)
#        bSize = size if size is not None else vertices.nbytes
#        vbo.allocate( vertices, bSize)
#
#        attr_loc = program.attributeLocation( shader_str )
#        program.enableAttributeArray(attr_loc )
#        program.setAttributeBuffer(attr_loc, gl.GL_FLOAT, 0, dim_vertex)
#        vbo.release()
#
#        return vbo

    def draw(self, model, view, projection):

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        self.shader.bind()
        self.shader.setUniformValue("model", model)
        self.shader.setUniformValue("view", view)
        self.shader.setUniformValue("projection", projection)

        self.vaoFrame.bind()
        self.shader.setUniformValue("lineOpacity", 0.75)
        gl.glLineWidth(self.parent.cBoxLineWidth * 2)
        gl.glDrawArrays(gl.GL_QUADS, 0, 12)
        self.vaoFrame.release()

        self.vaoGrid.bind()
        self.shader.setUniformValue("lineOpacity", 0.5)
        gl.glLineWidth(self.parent.cBoxLineWidth)
        gl.glDrawArrays(gl.GL_LINES, 0, self.gridLen)
        self.vaoGrid.release()

#        if self.parent.fineGridEnabled:
#            self.vaoFineGrid.bind()
#            self.shader.setUniformValue("lineOpacity", 0.25)
#            gl.glLineWidth(self.parent.cBoxLineWidth)
#            gl.glDrawArrays(gl.GL_LINES, 0, self.fineGridLen)
#            self.vaoFineGrid.release()
        self.shader.release()

        self.origShader.bind()
        self.origShader.setUniformValue("model", model)
        self.origShader.setUniformValue("view", view)
        self.origShader.setUniformValue("projection", projection)
        self.vaoOrigin.bind()
        self.origShader.setUniformValue("lineOpacity", 0.85)
        gl.glLineWidth(self.parent.cBoxLineWidth)
        gl.glDrawArrays(gl.GL_LINES, 0, 6)
        self.vaoOrigin.release()
        self.origShader.release()

        self.textShader.bind()
        self.vaoText.bind()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
        vpMat = projection*view
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    iAx == self.parent.visibleAxes[2]):

                midp = int(len(self.axisL[iAx][0, :])/2)
                p0 = self.axisL[iAx][:, midp]
                alignment = None
                if iAx == self.parent.visibleAxes[1]:  # Side plane,
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[0])
                if iAx == self.parent.visibleAxes[0]:  # Bottom plane, L-R
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[2],
                                                  self.parent.visibleAxes[1])
                if iAx == self.parent.visibleAxes[2]:  # Bottom plane, L-R
                    alignment = self.getAlignment(vpMat, p0,
                                                  self.parent.visibleAxes[0],
                                                  self.parent.visibleAxes[1])

                for tick, tText, pcs in list(zip(self.axisL[iAx].T,
                                                 self.gridLabels[iAx],
                                                 self.precisionLabels[iAx])):
                    valueStr = "{0:.{1}f}".format(tText, int(pcs))
                    tickPos = (vpMat*qt.QVector4D(*tick, 1)).toVector3DAffine()
                    self.render_text(tickPos, valueStr, alignment=alignment,
                                     scale=0.04*self.fontScale)
        self.vaoText.release()
        self.textShader.release()

    def render_text(self, pos, text, alignment, scale):

        char_x = 0
        pView = gl.glGetIntegerv(gl.GL_VIEWPORT)
        scaleX = scale/float(pView[2])
        scaleY = scale/float(pView[3])
        coordShift = np.zeros(2, dtype=np.float32)

        aw = []
        ah = []
        axrel = []
        ayrel = []

        for c in text:
            c = ord(c)
            ch = self.characters[c]
            w, h = ch[1][0] * scaleX, ch[1][1] * scaleY
            xrel = char_x + ch[2][0] * scaleX
            yrel = (ch[1][1] - ch[2][1]) * scaleY
            if c == 45:
                yrel = ch[1][0] * scaleY
            char_x += (ch[3] >> 6) * scaleX
            aw.append(w)
            ah.append(h)
            axrel.append(xrel)
            ayrel.append(yrel)

        if alignment is not None:
            if alignment[0] == 'left':
                coordShift[0] = -(axrel[-1]+2*aw[-1])
            else:
                coordShift[0] = 2*aw[-1]

            if alignment[1] == 'top':
                vOffset = 0.5
            elif alignment[1] == 'bottom':
                vOffset = -2
            else:
                vOffset = -1
            coordShift[1] = vOffset*ah[-1]

        for ic, c in enumerate(text):
            c = ord(c)
            ch = self.characters[c]
            if ch[1] == (0, 0):
                continue
            mMod = qt.QMatrix4x4()
            mMod.setToIdentity()

            mMod.translate(pos)
            mMod.translate(axrel[ic]+coordShift[0], ayrel[ic]+coordShift[1], 0)
            mMod.scale(aw[ic], ah[ic], 1)
            ch[0].bind()
            self.textShader.setUniformValue("model", mMod)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            ch[0].release()

    def get_sans_font(self):
        fallback_fonts = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "Sans-serif"]

        available_fonts = [font_manager.FontProperties(fname=path).get_name()
                           for path in font_manager.findSystemFonts()]

        for font in fallback_fonts:
            if font in available_fonts:
                return font

        return "Sans-serif"

    def make_font(self):
#        fontpath = os.path.dirname(__file__)
#        filename = os.path.join(fontpath, self.fontFile)
        fontName = self.get_sans_font()
#        print("Font found:", fontName)
        font_path = font_manager.findfont(fontName)
        face = ft.Face(font_path)
        face.set_pixel_sizes(self.fontSize*8, self.fontSize*8)
#        faceTexture = ft.Face(filename)
#        faceTexture.set_pixel_sizes(self.fontSize, self.fontSize)

        for c in range(128):
            face.load_char(chr(c), ft.FT_LOAD_RENDER)
#            faceTexture.load_char(chr(c), ft.FT_LOAD_RENDER)
            glyph = face.glyph
            bitmap = glyph.bitmap
            size = bitmap.width, bitmap.rows
            bearing = glyph.bitmap_left, 2 * bitmap.rows - glyph.bitmap_top
            advance = glyph.advance.x

            qi = qt.QImage(np.array(bitmap.buffer, dtype=np.uint8),
                           int(bitmap.width), int(bitmap.rows),
                           int(bitmap.width),
                           qt.QImage.Format_Grayscale8)
#            if chr(c) == "0":
#                qi.save("0.jpg")
            texObj = qt.QOpenGLTexture(qi)
            texObj.setMinificationFilter(qt.QOpenGLTexture.LinearMipMapLinear)
            texObj.setMagnificationFilter(qt.QOpenGLTexture.Linear)
            texObj.generateMipMaps()
            self.characters.append((texObj, size, bearing, advance))

    def getAlignment(self, pvMatr, point, hDim, vDim=None):
        pointH = np.copy(point)
        pointV = np.copy(point)

        sp0 = pvMatr * qt.QVector4D(*point, 1)
        pointH[hDim] *= 1.1
        spH = pvMatr * qt.QVector4D(*pointH, 1)

        if vDim is None:
            vAlign = 'middle'
        else:
            pointV[vDim] *= 1.1
            spV = pvMatr * qt.QVector4D(*pointV, 1)
            vAlign = 'top' if spV[1] - sp0[1] > 0 else 'bottom'
        hAlign = 'left' if spH[0] - sp0[0] < 0 else 'right'
        return (hAlign, vAlign)


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:07:53 2017

@author: Roman Chernikov
"""

import os
import numpy as np
from functools import partial
import matplotlib as mpl
import inspect
import re
import copy
#import time

from OpenGL.GL import glRotatef, glMaterialfv, glClearColor, glMatrixMode,\
    glLoadIdentity, glOrtho, glClear, glEnable, glBlendFunc,\
    glEnableClientState, glPolygonMode, glGetDoublev, glDisable,\
    glDisableClientState, glRasterPos3f, glPushMatrix, glTranslatef, glScalef,\
    glPopMatrix, glFlush, glVertexPointerf, glColorPointerf, glLineWidth,\
    glDrawArrays, glMap2f, glMapGrid2f, glEvalMesh2, glLightModeli, glLightfv,\
    glGetIntegerv, glColor4f, glVertex3f, glBegin, glEnd, glViewport,\
    glMaterialf, glHint, glPointSize,\
    GL_FRONT_AND_BACK, GL_AMBIENT, GL_DIFFUSE, GL_SPECULAR, GL_EMISSION,\
    GL_FRONT, GL_SHININESS, GL_PROJECTION, GL_MODELVIEW, GL_COLOR_BUFFER_BIT,\
    GL_DEPTH_BUFFER_BIT, GL_MULTISAMPLE, GL_BLEND, GL_SRC_ALPHA,\
    GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH, GL_COLOR_ARRAY, GL_LINE,\
    GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,\
    GL_NICEST, GL_POLYGON_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_DEPTH_TEST,\
    GL_FILL, GL_NORMAL_ARRAY, GL_NORMALIZE, GL_VERTEX_ARRAY,\
    GL_QUADS, GL_MAP2_VERTEX_3, GL_MAP2_NORMAL, GL_LIGHTING, GL_POINTS,\
    GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHT0, GL_POSITION, GL_SPOT_DIRECTION,\
    GL_SPOT_CUTOFF, GL_SPOT_EXPONENT, GL_TRIANGLE_FAN, GL_VIEWPORT, GL_LINES,\
    GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX

from OpenGL.GLU import gluPerspective, gluLookAt, gluProject

from OpenGL.GLUT import glutBitmapCharacter, glutStrokeCharacter, glutInit,\
    glutInitDisplayMode, GLUT_BITMAP_HELVETICA_12, GLUT_STROKE_ROMAN,\
    GLUT_RGBA, GLUT_DOUBLE, GLUT_DEPTH, GLUT_STROKE_MONO_ROMAN

from OpenGL.arrays import vbo

from collections import OrderedDict
try:
    from matplotlib.backends import qt_compat
except ImportError:
    from matplotlib.backends import qt4_compat
    qt_compat = qt4_compat

if 'pyqt4' in qt_compat.QT_API.lower():  # also 'PyQt4v2'
    QtName = "PyQt4"
    from PyQt4 import QtGui, QtCore
    import PyQt4.QtGui as myQtGUI
    import PyQt4.QtOpenGL as myQtGL
    try:
        import PyQt4.Qwt5 as Qwt
    except:
        pass
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as\
        FigCanvas
elif 'pyqt5' in qt_compat.QT_API.lower():
    QtName = "PyQt5"
    from PyQt5 import QtGui, QtCore
    import PyQt5.QtWidgets as myQtGUI
    import PyQt5.QtOpenGL as myQtGL
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as\
        FigCanvas
else:
    raise ImportError("Cannot import any Python Qt package!")

QWidget, QApplication, QAction, QTabWidget, QToolBar, QStatusBar, QTreeView,\
    QShortcut, QAbstractItemView, QHBoxLayout, QVBoxLayout, QSplitter,\
    QComboBox, QMenu, QListWidget, QTextEdit, QMessageBox, QFileDialog,\
    QListWidgetItem, QGLWidget, QGroupBox, QGridLayout,\
    QLabel, QSizePolicy, QLineEdit, QCheckBox, QSpinBox, QSlider = (
        myQtGUI.QWidget, myQtGUI.QApplication, myQtGUI.QAction,
        myQtGUI.QTabWidget, myQtGUI.QToolBar, myQtGUI.QStatusBar,
        myQtGUI.QTreeView, myQtGUI.QShortcut, myQtGUI.QAbstractItemView,
        myQtGUI.QHBoxLayout, myQtGUI.QVBoxLayout, myQtGUI.QSplitter,
        myQtGUI.QComboBox, myQtGUI.QMenu, myQtGUI.QListWidget,
        myQtGUI.QTextEdit, myQtGUI.QMessageBox, myQtGUI.QFileDialog,
        myQtGUI.QListWidgetItem, myQtGL.QGLWidget, myQtGUI.QGroupBox,
        myQtGUI.QGridLayout, myQtGUI.QLabel, myQtGUI.QSizePolicy,
        myQtGUI.QLineEdit, myQtGUI.QCheckBox, myQtGUI.QSpinBox,
        myQtGUI.QSlider)
QIcon, QFont, QKeySequence, QStandardItemModel, QStandardItem, QPixmap,\
    QDoubleValidator, QIntValidator =\
    (QtGui.QIcon, QtGui.QFont, QtGui.QKeySequence, QtGui.QStandardItemModel,
     QtGui.QStandardItem, QtGui.QPixmap,
     QtGui.QDoubleValidator, QtGui.QIntValidator)

from ..backends import raycing  # analysis:ignore
from ..backends.raycing import sources as rsources  # analysis:ignore
from ..backends.raycing import screens as rscreens  # analysis:ignore


class mySlider(QSlider):
    def __init__(self, parent, scaleDirection, scalePosition):
        super(mySlider, self).__init__(scaleDirection)
        self.setTickPosition(scalePosition)
        self.scale = 1.

    def setRange(self, start, end, step):
        self.scale = 1. / step
        QSlider.setRange(self, start / step, end / step)

    def setValue(self, value):
        QSlider.setValue(self, int(value*self.scale))

try:
    glowSlider = Qwt.QwtSlider
    glowTopScale = Qwt.QwtSlider.TopScale
except:
    glowSlider = mySlider
    glowTopScale = QSlider.TicksAbove


class xrtGlow(QWidget):
    def __init__(self, arrayOfRays, parent=None, progressSignal=None):
        super(xrtGlow, self).__init__()
        self.parentRef = parent
        self.setWindowTitle('xrtGlow')
        iconsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '_icons')
        self.setWindowIcon(QIcon(os.path.join(iconsDir,
                                              'icon-GLow.ico')))
        self.populate_oes_list(arrayOfRays)

        self.segmentsModel = self.init_segments_model()
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()

        self.populate_segments_model(arrayOfRays)

        self.fluxDataModel = QStandardItemModel()

        for rfName, rfObj in inspect.getmembers(raycing):
            if rfName.startswith('get_') and\
                    rfName != "get_output":
                flItem = QStandardItem(rfName.replace("get_", ''))
                self.fluxDataModel.appendRow(flItem)

        self.customGlWidget = xrtGlWidget(self, arrayOfRays,
                                          self.segmentsModelRoot,
                                          self.oesList,
                                          self.beamsToElements,
                                          progressSignal)
        self.customGlWidget.rotationUpdated.connect(self.updateRotationFromGL)
        self.customGlWidget.scaleUpdated.connect(self.updateScaleFromGL)
        self.customGlWidget.histogramUpdated.connect(self.updateColorMap)
        self.customGlWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customGlWidget.customContextMenuRequested.connect(self.glMenu)
#  Zoom panel
        self.zoomPanel = QGroupBox(self)
        self.zoomPanel.setFlat(False)
        self.zoomPanel.setTitle("Scale")
        zoomLayout = QGridLayout()

        scaleValidator = QDoubleValidator()
        scaleValidator.setRange(0, 7, 7)
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = QLabel()
            axLabel.setText(axis+' (log)')
            axLabel.objectName = "scaleLabel_" + axis
            if iaxis == 1:
                axEdit = QLineEdit("1")
            else:
                axEdit = QLineEdit("3")
            axEdit.setValidator(scaleValidator)
            axSlider = glowSlider(
                self, QtCore.Qt.Horizontal, glowTopScale)
            axSlider.setRange(0, 7, 0.01)
            if iaxis == 1:
                axSlider.setValue(1)
            else:
                axSlider.setValue(3)
            axEdit.editingFinished.connect(self.updateScaleFromQLE)
            axSlider.objectName = "scaleSlider_" + axis
            axSlider.valueChanged.connect(self.updateScale)
            zoomLayout.addWidget(axLabel, iaxis*2+1, 0)
            zoomLayout.addWidget(axEdit, iaxis*2+1, 1)
            zoomLayout.addWidget(axSlider, iaxis*2+2, 0, 1, 2)

        self.zoomPanel.setLayout(zoomLayout)

#  Rotation panel
        self.rotationPanel = QGroupBox(self)
        self.rotationPanel.setFlat(False)
        self.rotationPanel.setTitle("Rotation")
        rotationLayout = QGridLayout()

#        rotModeCB = QCheckBox()
#        rotModeCB.setCheckState(2)
#        rotModeCB.stateChanged.connect(self.checkEulerian)
#        rotModeLabel = QLabel()
#        rotModeLabel.setText('Use Eulerian rotation')
#        rotationLayout.addWidget(rotModeCB, 0, 0)
#        rotationLayout.addWidget(rotModeLabel, 0, 1)

        rotValidator = QDoubleValidator()
        rotValidator.setRange(-180, 180, 9)
        for iaxis, axis in enumerate(['x (pitch)', 'y (roll)', 'z (yaw)']):
            axLabel = QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "rotLabel_" + axis[0]
            axEdit = QLineEdit("0.")
            axEdit.setValidator(rotValidator)
            axSlider = glowSlider(
                self, QtCore.Qt.Horizontal, glowTopScale)
            axSlider.setRange(-180, 180, 0.01)
            axSlider.setValue(0)
            axEdit.editingFinished.connect(self.updateRotationFromQLE)
            axSlider.objectName = "rotSlider_" + axis[0]
            axSlider.valueChanged.connect(self.updateRotation)
            rotationLayout.addWidget(axLabel, iaxis*2, 0)
            rotationLayout.addWidget(axEdit, iaxis*2, 1)
            rotationLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)
        self.rotationPanel.setLayout(rotationLayout)

        self.transformationPanel = QWidget(self)
        transformationLayout = QVBoxLayout()
        transformationLayout.addWidget(self.zoomPanel)
        transformationLayout.addWidget(self.rotationPanel)
        self.transformationPanel.setLayout(transformationLayout)
#  Opacity panel
        self.opacityPanel = QGroupBox(self)
        self.opacityPanel.setFlat(False)
        self.opacityPanel.setTitle("Opacity")
        opacityLayout = QGridLayout()
        for iaxis, axis in enumerate(
                ['Line opacity', 'Line width', 'Point opacity', 'Point size']):
            axLabel = QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "opacityLabel_" + str(iaxis)
            opacityValidator = QDoubleValidator()
            axSlider = glowSlider(
                self, QtCore.Qt.Horizontal, glowTopScale)

            if iaxis in [0, 2]:
                axSlider.setRange(0, 1., 0.001)
                axSlider.setValue(0.1)
                axEdit = QLineEdit("0.1")
                opacityValidator.setRange(0, 1., 5)

            else:
                axSlider.setRange(0, 20, 0.01)
                axSlider.setValue(1.)
                axEdit = QLineEdit("1")
                opacityValidator.setRange(0, 20., 5)

            axEdit.setValidator(opacityValidator)
            axEdit.editingFinished.connect(self.updateOpacityFromQLE)
            axSlider.objectName = "opacitySlider_" + str(iaxis)
            axSlider.valueChanged.connect(self.updateOpacity)
            opacityLayout.addWidget(axLabel, iaxis*2, 0)
            opacityLayout.addWidget(axEdit, iaxis*2, 1)
            opacityLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)
        self.opacityPanel.setLayout(opacityLayout)

#  Color panel
        self.colorPanel = QGroupBox(self)
        self.colorPanel.setFlat(False)
        self.colorPanel.setTitle("Color")
        colorLayout = QGridLayout()
        self.mplFig = mpl.figure.Figure(figsize=(3, 3))
        self.mplAx = self.mplFig.add_subplot(111)
        self.mplFig.suptitle("")

        self.drawColorMap('energy')
        self.paletteWidget = FigCanvas(self.mplFig)
        self.paletteWidget.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Expanding)
        self.paletteWidget.span = mpl.widgets.RectangleSelector(
            self.mplAx, self.updateColorSelFromMPL, drawtype='box',
            useblit=True, rectprops=dict(alpha=0.4, facecolor='white'),
            button=1, interactive=True)

        colorLayout.addWidget(self.paletteWidget, 0, 0, 1, 2)

        colorCBLabel = QLabel()
        colorCBLabel.setText('Color Axis:')

        colorCB = QComboBox()
        colorCB.setModel(self.fluxDataModel)
        colorCB.setCurrentIndex(colorCB.findText('energy'))
        colorCB.currentIndexChanged['QString'].connect(self.changeColorAxis)
        colorLayout.addWidget(colorCBLabel, 1, 0)
        colorLayout.addWidget(colorCB, 1, 1)
        for icSel, cSelText in enumerate(['Selection min',
                                          'Selection max']):
            selLabel = QLabel()
            selLabel.setText(cSelText)
            selValidator = QDoubleValidator()
            selValidator.setRange(self.customGlWidget.colorMin,
                                  self.customGlWidget.colorMax, 5)
            selQLE = QLineEdit()
            selQLE.setValidator(selValidator)
            selQLE.setText('{0:.3f}'.format(
                self.customGlWidget.colorMin if icSel == 0 else
                self.customGlWidget.colorMax))
            selQLE.editingFinished.connect(self.updateColorSelFromQLE)
            colorLayout.addWidget(selLabel, 2, icSel)
            colorLayout.addWidget(selQLE, 3, icSel)
        selSlider = glowSlider(
            self, QtCore.Qt.Horizontal, glowTopScale)
        rStep = (self.customGlWidget.colorMax -
                 self.customGlWidget.colorMin) / 100.
        rValue = (self.customGlWidget.colorMax +
                  self.customGlWidget.colorMin) * 0.5
        selSlider.setRange(self.customGlWidget.colorMin,
                           self.customGlWidget.colorMax, rStep)
        selSlider.setValue(rValue)
        selSlider.sliderMoved.connect(self.updateColorSel)
        colorLayout.addWidget(selSlider, 4, 0, 1, 2)

        axLabel = QLabel()
        axLabel.setText("Intensity cut-off")
        axLabel.objectName = "cutLabel_I"
        axEdit = QLineEdit("0.01")
        cutValidator = QDoubleValidator()
        cutValidator.setRange(0, 1, 3)
        axEdit.setValidator(cutValidator)
        axEdit.editingFinished.connect(self.updateCutoffFromQLE)

        explLabel = QLabel()
        explLabel.setText("Color bump height, mm")
        explLabel.objectName = "explodeLabel"
        explEdit = QLineEdit("0.0")
        explValidator = QDoubleValidator()
        explValidator.setRange(-100, 100, 3)
        explEdit.setValidator(explValidator)
        explEdit.editingFinished.connect(self.updateExplosionDepth)

#        axSlider = glowSlider(
#            self, QtCore.Qt.Horizontal, glowTopScale)
#        axSlider.setRange(0, 1, 0.001)
#        axSlider.setValue(0.01)
#        axSlider.objectName = "cutSlider_I"
#        axSlider.valueChanged.connect(self.updateCutoff)

        glNormCB = QCheckBox()
        glNormCB.objectName = "gNormChb"
        glNormCB.setCheckState(2)
        glNormCB.stateChanged.connect(self.checkGNorm)
        glNormLabel = QLabel()
        glNormLabel.setText('Global Normalization')

        iHSVCB = QCheckBox()
        iHSVCB.objectName = "iHSVChb"
        iHSVCB.setCheckState(0)
        iHSVCB.stateChanged.connect(self.check_iHSV)
        iHSVLabel = QLabel()
        iHSVLabel.setText('Intensity as HSV Value')

        colorLayout.addWidget(axLabel, 2+3, 0)
        colorLayout.addWidget(axEdit, 2+3, 1)
        colorLayout.addWidget(explLabel, 3+3, 0)
        colorLayout.addWidget(explEdit, 3+3, 1)
#        colorLayout.addWidget(axSlider, 3+3, 0, 1, 2)
        colorLayout.addWidget(glNormCB, 4+3, 0, 1, 1)
        colorLayout.addWidget(glNormLabel, 4+3, 1, 1, 1)
        colorLayout.addWidget(iHSVCB, 5+3, 0, 1, 1)
        colorLayout.addWidget(iHSVLabel, 5+3, 1, 1, 1)
        self.colorPanel.setLayout(colorLayout)

        self.colorOpacityPanel = QWidget(self)
        colorOpacityLayout = QVBoxLayout()
        colorOpacityLayout.addWidget(self.colorPanel)
        colorOpacityLayout.addWidget(self.opacityPanel)
        self.colorOpacityPanel.setLayout(colorOpacityLayout)

#  Projection panel
        self.projectionPanel = QGroupBox(self)
        self.projectionPanel.setFlat(False)
#        self.projectionPanel.setTitle("Line properties")
        projectionLayout = QGridLayout()
        self.projVisPanel = QGroupBox(self)
        self.projVisPanel.setFlat(False)
        self.projVisPanel.setTitle("Projections visibility")
        projVisLayout = QGridLayout()
        self.projLinePanel = QGroupBox(self)
        self.projLinePanel.setFlat(False)
        self.projLinePanel.setTitle("Projections opacity")
        projLineLayout = QGridLayout()
        self.projectionControls = []
        for iaxis, axis in enumerate(['Show Side (YZ)', 'Show Front (XZ)',
                                      'Show Top (XY)']):
            checkBox = QCheckBox()
            checkBox.objectName = "visChb_" + str(iaxis)
            checkBox.setCheckState(0)
            checkBox.stateChanged.connect(self.projSelection)
            visLabel = QLabel()
            visLabel.setText(axis)
            self.projectionControls.append([visLabel, checkBox])
            projVisLayout.addWidget(checkBox, iaxis*2, 0, 1, 1)
            projVisLayout.addWidget(visLabel, iaxis*2, 1, 1, 1)

        for (iCB, cbCaption), cbFunction in zip(enumerate(
                ['Coordinate grid', 'Fine grid', 'Perspective']),
                [self.checkDrawGrid, self.checkFineGrid, self.checkPerspect]):
            checkBox = QCheckBox()
            checkBox.objectName = "visChb_" + str(3+iCB)
            checkBox.setCheckState(0 if iCB == 1 else 2)
            checkBox.stateChanged.connect(cbFunction)
            visLabel = QLabel()
            visLabel.setText(cbCaption)
            self.projectionControls.append([visLabel, checkBox])
            projVisLayout.addWidget(checkBox, (3+iCB)*2, 0, 1, 1)
            projVisLayout.addWidget(visLabel, (3+iCB)*2, 1, 1, 1)

        self.projVisPanel.setLayout(projVisLayout)

        for iaxis, axis in enumerate(
                ['Line opacity', 'Line width', 'Point opacity', 'Point size']):
            axLabel = QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "projectionLabel_" + str(iaxis)
            projectionValidator = QDoubleValidator()
            axSlider = glowSlider(
                self, QtCore.Qt.Horizontal, glowTopScale)

            if iaxis in [0, 2]:
                axSlider.setRange(0, 1., 0.001)
                axSlider.setValue(0.1)
                axEdit = QLineEdit("0.1")
                projectionValidator.setRange(0, 1., 5)

            else:
                axSlider.setRange(0, 20, 0.01)
                axSlider.setValue(1.)
                axEdit = QLineEdit("1")
                projectionValidator.setRange(0, 20., 5)

            axEdit.setValidator(projectionValidator)
            axEdit.editingFinished.connect(self.updateProjectionOpacityFromQLE)
            axSlider.objectName = "projectionSlider_" + str(iaxis)
            axSlider.valueChanged.connect(self.updateProjectionOpacity)
            projLineLayout.addWidget(axLabel, iaxis*2, 0)
            projLineLayout.addWidget(axEdit, iaxis*2, 1)
            projLineLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)
        self.projLinePanel.setLayout(projLineLayout)
        projectionLayout.addWidget(self.projVisPanel, 0, 0)
        projectionLayout.addWidget(self.projLinePanel, 1, 0)
        self.projectionPanel.setLayout(projectionLayout)

        self.scenePanel = QGroupBox(self)
        self.scenePanel.setFlat(False)
        self.scenePanel.setTitle("Scale coordinate grid")
        sceneLayout = QGridLayout()
        sceneValidator = QDoubleValidator()
        sceneValidator.setRange(0, 10, 3)
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "sceneLabel_" + axis
            axEdit = QLineEdit("0.9")
            axEdit.setValidator(scaleValidator)
            axSlider = glowSlider(
                self, QtCore.Qt.Horizontal, glowTopScale)
            axSlider.setRange(0, 10, 0.01)
            axSlider.setValue(0.9)
            axEdit.editingFinished.connect(self.updateSceneFromQLE)
            axSlider.objectName = "sceneSlider_" + axis
            axSlider.valueChanged.connect(self.updateScene)
            sceneLayout.addWidget(axLabel, iaxis*2, 0)
            sceneLayout.addWidget(axEdit, iaxis*2, 1)
            sceneLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)
        self.sceneControls = []
        for (iCB, cbText), cbFunc in zip(enumerate(['Enable antialiasing',
                                                    'Enable blending',
                                                    'Depth test for Lines',
                                                    'Depth test for Points',
                                                    'Invert scene color',
                                                    'Use scalable font',
                                                    'Show VScreen Label',
                                                    'Show lost rays']),
                                         [self.checkAA,
                                          self.checkBlending,
                                          self.checkLineDepthTest,
                                          self.checkPointDepthTest,
                                          self.invertSceneColor,
                                          self.checkScalableFont,
                                          self.checkShowLabels,
                                          self.checkShowLost]):
            aaCheckBox = QCheckBox()
            aaCheckBox.objectName = "aaChb" + str(iCB)
            aaCheckBox.setCheckState(2) if iCB in [1, 2] else\
                aaCheckBox.setCheckState(0)
            aaCheckBox.stateChanged.connect(cbFunc)
            aaLabel = QLabel()
            aaLabel.setText(cbText)
            self.sceneControls.append([aaLabel, aaCheckBox])
            sceneLayout.addWidget(aaCheckBox, 6+iCB, 0)
            sceneLayout.addWidget(aaLabel, 6+iCB, 1)

        axLabel = QLabel()
        axLabel.setText('Font Size')
        axSlider = glowSlider(
            self, QtCore.Qt.Horizontal, glowTopScale)
        axSlider.setRange(1, 20, 0.5)
        axSlider.setValue(5)
        axSlider.valueChanged.connect(self.updateFontSize)
        sceneLayout.addWidget(axLabel, 8+iCB, 0)
        sceneLayout.addWidget(axSlider, 8+iCB, 1, 1, 2)

        labelPrec = QSpinBox()
        labelPrec.setRange(0, 4)
        labelPrec.setValue(1)
        labelPrec.setSuffix('mm')
        labelPrec.valueChanged.connect(self.setLabelPrec)
        aaLabel = QLabel()
        aaLabel.setText('Label Precision')
        sceneLayout.addWidget(aaLabel, 9+iCB, 0)
        sceneLayout.addWidget(labelPrec, 9+iCB, 1)

        oeTileValidator = QIntValidator()
        oeTileValidator.setRange(1, 20)
        for iaxis, axis in enumerate(['OE tessellation X',
                                      'OE tessellation Y']):
            axLabel = QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "oeTileLabel_" + axis
            axEdit = QLineEdit("2")
            axEdit.setValidator(oeTileValidator)
            axEdit.objectName = "oeTileEditor_" + axis
#            axSlider = glowSlider(
#                self, QtCore.Qt.Horizontal, glowTopScale)
#            axSlider.setRange(1, 20, 1)
#            axSlider.setValue(2)
            axEdit.editingFinished.connect(self.updateTileFromQLE)
#            axSlider.objectName = "oeTileSlider_" + axis
#            axSlider.valueChanged.connect(self.updateTile)
            sceneLayout.addWidget(axLabel, 17+iaxis*2, 0)
            sceneLayout.addWidget(axEdit, 17+iaxis*2, 1)
#            sceneLayout.addWidget(axSlider, 17+iaxis*2+1, 0, 1, 2)

        self.scenePanel.setLayout(sceneLayout)

#  Navigation panel
        self.navigationPanel = QGroupBox(self)
        self.navigationPanel.setFlat(False)
#        self.navigationPanel.setTitle("Navigation")
        self.navigationLayout = QGridLayout()

        centerCBLabel = QLabel()
        centerCBLabel.setText('Center at:')
        self.centerCB = QComboBox()
        for key in self.oesList.keys():
            self.centerCB.addItem(str(key))
#        centerCB.addItem('customXYZ')
        self.centerCB.currentIndexChanged['QString'].connect(self.centerEl)
        self.centerCB.setCurrentIndex(0)

        self.navigationLayout.addWidget(centerCBLabel, 0, 0)
        self.navigationLayout.addWidget(self.centerCB, 0, 1)
        self.oeTree = QTreeView()
        self.oeTree.setModel(self.segmentsModel)
        self.oeTree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.oeTree.customContextMenuRequested.connect(self.oeTreeMenu)
        self.navigationLayout.addWidget(self.oeTree, 1, 0, 1, 2)
        self.navigationPanel.setLayout(self.navigationLayout)

        mainLayout = QHBoxLayout()
        sideLayout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self.navigationPanel, "Navigation")
        tabs.addTab(self.transformationPanel, "Transformation")
        tabs.addTab(self.colorOpacityPanel, "Color")
        tabs.addTab(self.projectionPanel, "Projections")
        tabs.addTab(self.scenePanel, "Scene")
        sideLayout.addWidget(tabs)
        self.canvasSplitter = QSplitter()
        self.canvasSplitter.setChildrenCollapsible(False)
        self.canvasSplitter.setOrientation(QtCore.Qt.Horizontal)
        mainLayout.addWidget(self.canvasSplitter)
        sideWidget = QWidget()
        sideWidget.setLayout(sideLayout)
        self.canvasSplitter.addWidget(self.customGlWidget)
        self.canvasSplitter.addWidget(sideWidget)

        self.setLayout(mainLayout)
        self.customGlWidget.oesList = self.oesList
        toggleHelp = QShortcut(self)
        toggleHelp.setKey(QtCore.Qt.Key_F1)
        toggleHelp.activated.connect(self.customGlWidget.toggleHelp)
        fastSave = QShortcut(self)
        fastSave.setKey(QtCore.Qt.Key_F5)
        fastSave.activated.connect(partial(self.saveScene, '_xrtScnTmp_.npy'))
        fastLoad = QShortcut(self)
        fastLoad.setKey(QtCore.Qt.Key_F6)
        fastLoad.activated.connect(partial(self.loadScene, '_xrtScnTmp_.npy'))
        toggleScreen = QShortcut(self)
        toggleScreen.setKey(QtCore.Qt.Key_F3)
        toggleScreen.activated.connect(self.customGlWidget.toggleVScreen)
        self.dockToQook = QShortcut(self)
        self.dockToQook.setKey(QtCore.Qt.Key_F4)
        self.dockToQook.activated.connect(self.toggleDock)
        tiltScreen = QShortcut(self)
        tiltScreen.setKey(QtCore.Qt.CTRL + QtCore.Qt.Key_T)
        tiltScreen.activated.connect(self.customGlWidget.switchVScreenTilt)

    def toggleDock(self):
        if self.parentRef is not None:
            self.parentRef.catch_viewer()
            self.parentRef = None

    def init_segments_model(self, isNewModel=True):
        newModel = QStandardItemModel()
        newModel.setHorizontalHeaderLabels(['Rays',
                                            'Footprint',
                                            'Surface',
                                            'Label'])
        if isNewModel:
            headerRow = []
            for i in range(4):
                child = QStandardItem("")
                child.setEditable(False)
                child.setCheckable(True)
                child.setCheckState(0 if i > 1 else 2)
                headerRow.append(child)
            newModel.invisibleRootItem().appendRow(headerRow)
        newModel.itemChanged.connect(self.updateRaysList)
        return newModel

    def update_oes_list(self, arrayOfRays):
        self.oesList = None
        self.beamsToElements = None
        self.populate_oes_list(arrayOfRays)
        self.update_segments_model(arrayOfRays)
        self.centerCB.blockSignals(True)
        tmpIndex = self.centerCB.currentIndex()
        for i in range(self.centerCB.count()):
            self.centerCB.removeItem(0)
        for key in self.oesList.keys():
            self.centerCB.addItem(str(key))
#        self.segmentsModel.layoutChanged.emit()
        try:
            self.centerCB.setCurrentIndex(tmpIndex)
        except:
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

    def populate_oes_list(self, arrayOfRays):
        self.oesList = OrderedDict()
        self.beamsToElements = dict()
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
                    center = [arrayOfRays[1][oeRecord[3]].x[0],
                              arrayOfRays[1][oeRecord[3]].y[0],
                              arrayOfRays[1][oeRecord[3]].z[0]]
                    is2ndXtal = True

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
                    self.oesList[elName].append(
                        None)
                self.oesList[elName].append(center)
                self.oesList[elName].append(is2ndXtal)

    def create_row(self, text, segMode):
        newRow = []
        for iCol in range(4):
            newItem = QStandardItem(str(text) if iCol == 0 else "")
            newItem.setCheckable(True if (segMode == 3 and iCol == 0) or
                                 (segMode == 1 and iCol > 0) else False)
            if newItem.isCheckable():
                newItem.setCheckState(2 if iCol < 2 else 0)
            newItem.setEditable(False)
            newRow.append(newItem)
        return newRow

    def update_segments_model(self, arrayOfRays):
        def copy_row(item, row):
            newRow = []
            for iCol in range(4):
                oldItem = item.child(row, iCol)
                newItem = QStandardItem(str(oldItem.text()))
                newItem.setCheckable(oldItem.isCheckable())
                if newItem.isCheckable():
                    newItem.setCheckState(oldItem.checkState())
                newItem.setEditable(oldItem.isEditable())
                newRow.append(newItem)
            return newRow

        newSegmentsModel = self.init_segments_model(isNewModel=False)
        newSegmentsModel.invisibleRootItem().appendRow(
            copy_row(self.segmentsModelRoot, 0))
        for element, elRecord in self.oesList.items():
            for iel in range(self.segmentsModelRoot.rowCount()):
                elItem = self.segmentsModelRoot.child(iel, 0)
                elName = str(elItem.text())
                if str(element) == elName:
                    elRow = copy_row(self.segmentsModelRoot, iel)
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
                                                copy_row(elItem, ich))
                                            break
                                    else:
                                        elRow[0].appendRow(self.create_row(
                                            endBeamText, 3))
                                else:
                                    elRow[0].appendRow(self.create_row(
                                        endBeamText, 3))
                    newSegmentsModel.invisibleRootItem().appendRow(elRow)
                    break
            else:
                elRow = self.create_row(str(element), 1)
                for segment in arrayOfRays[0]:
                    if str(segment[1]) == str(elRecord[1]) and\
                            segment[3] is not None:
                        endBeamText = "to {}".format(
                            self.beamsToElements[segment[3]])
                        elRow[0].appendRow(self.create_row(endBeamText, 3))
                newSegmentsModel.invisibleRootItem().appendRow(elRow)
        self.segmentsModel = newSegmentsModel
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()
        self.oeTree.setModel(self.segmentsModel)

    def populate_segments_model(self, arrayOfRays):
        for element, elRecord in self.oesList.items():
            newRow = self.create_row(element, 1)
            for segment in arrayOfRays[0]:
                if str(segment[1]) == str(elRecord[1]):
                    try:  # if segment[3] is not None:
                        endBeamText = "to {}".format(
                            self.beamsToElements[segment[3]])
                        newRow[0].appendRow(self.create_row(endBeamText, 3))
                    except:
                        continue
            self.segmentsModelRoot.appendRow(newRow)

    def drawColorMap(self, axis):
        xv, yv = np.meshgrid(np.linspace(0, 1, 200),
                             np.linspace(0, 1, 200))
        xv = xv.flatten()
        yv = yv.flatten()
        self.im = self.mplAx.imshow(mpl.colors.hsv_to_rgb(np.vstack((
            xv, np.ones_like(xv)*0.85, yv)).T).reshape((200, 200, 3)),
            aspect='auto', origin='lower',
            extent=(self.customGlWidget.colorMin,
                    self.customGlWidget.colorMax,
                    0, 1))
        self.mplAx.set_xlabel(axis)
        self.mplAx.set_ylabel('Intensity')
        self.mplFig.tight_layout()

    def updateColorMap(self, histArray):
        if histArray[0] is not None:
            size = len(histArray[0])
            histImage = np.zeros((size, size, 3))
            colorMin = self.customGlWidget.colorMin
            colorMax = self.customGlWidget.colorMax
            hMax = np.float(np.max(histArray[0]))
            intensity = np.float64(np.array(histArray[0]) / hMax)
            histVals = np.int32(intensity * (size-1))
            for col in range(size):
                histImage[0:histVals[col], col, :] = mpl.colors.hsv_to_rgb(
                    ((histArray[1][col] - colorMin) / (colorMax - colorMin),
                     0.85, intensity[col]))
            self.im.set_data(histImage)
            try:
                topEl = np.where(intensity >= 0.5)[0]
                hwhm = (np.abs(histArray[1][topEl[0]] -
                               histArray[1][topEl[-1]])) * 0.5
                cntr = (histArray[1][topEl[0]] + histArray[1][topEl[-1]]) * 0.5
                newLabel = u"FWHM: {0:.3f}\u00b1{1:.3f}".format(
                    cntr, hwhm)
                self.mplAx.set_title(newLabel)
            except:
                pass
            self.mplFig.canvas.draw()
            self.mplFig.canvas.blit()
            self.paletteWidget.span.extents = self.paletteWidget.span.extents
        else:
            xv, yv = np.meshgrid(np.linspace(0, 1, 200),
                                 np.linspace(0, 1, 200))
            xv = xv.flatten()
            yv = yv.flatten()
            self.im.set_data(mpl.colors.hsv_to_rgb(np.vstack((
                xv, np.ones_like(xv)*0.85, yv)).T).reshape((200, 200, 3)))
#            colorCB = self.colorPanel.layout().itemAt(2).widget()
#            self.mplAx.set_xlabel(colorCB.currentText())
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

    def check_iHSV(self, state):
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

    def checkShowLost(self, state):
        self.customGlWidget.showLostRays = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def setSceneParam(self, iAction, state):
        self.sceneControls[iAction][1].setCheckState(int(state)*2)

    def setProjectionParam(self, iAction, state):
        self.projectionControls[iAction][1].setCheckState(int(state)*2)

    def setLabelPrec(self, prec):
        self.customGlWidget.labelCoordPrec = prec
        self.customGlWidget.glDraw()

    def changeColorAxis(self, selAxis):
        if selAxis is None:
            selAxis = self.colorPanel.layout().itemAt(2).widget().currentText()
        else:
            self.customGlWidget.getColor = getattr(
                raycing, 'get_{}'.format(selAxis))
        oldColorMin = self.customGlWidget.colorMin
        oldColorMax = self.customGlWidget.colorMax
        self.customGlWidget.newColorAxis = True
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.mplAx.set_xlabel(selAxis)
        if oldColorMin == self.customGlWidget.colorMin and\
                oldColorMax == self.customGlWidget.colorMax:
            return
        self.customGlWidget.selColorMin = self.customGlWidget.colorMin
        self.customGlWidget.selColorMax = self.customGlWidget.colorMax
        extents = (self.customGlWidget.colorMin,
                   self.customGlWidget.colorMax, 0, 1)
        self.im.set_extent(extents)
        extents = list(extents)
        self.colorPanel.layout().itemAt(4).widget().setText(
            str(self.customGlWidget.colorMin))
        self.colorPanel.layout().itemAt(6).widget().validator().setRange(
            self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
        self.colorPanel.layout().itemAt(6).widget().setText(
            str(self.customGlWidget.colorMax))
        self.colorPanel.layout().itemAt(4).widget().validator().setRange(
            self.customGlWidget.colorMin, self.customGlWidget.colorMax, 5)
        slider = self.colorPanel.layout().itemAt(7).widget()
        center = 0.5 * (extents[0] + extents[1])
        newMin = self.customGlWidget.colorMin
        newMax = self.customGlWidget.colorMax
        newRange = (newMax - newMin) * 0.01
        slider.setRange(newMin, newMax, newRange)
        slider.setValue(center)
        self.mplFig.canvas.draw()
        self.paletteWidget.span.active_handle = None
        self.paletteWidget.span.to_draw.set_visible(False)
        self.customGlWidget.glDraw()

    def updateColorSelFromMPL(self, eclick, erelease):
        try:
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.selColorMin = np.min([extents[0], extents[1]])
            self.customGlWidget.selColorMax = np.max([extents[0], extents[1]])
            self.colorPanel.layout().itemAt(4).widget().setText(str(
                self.customGlWidget.selColorMin))
            self.colorPanel.layout().itemAt(6).widget().validator().setBottom(
                self.customGlWidget.selColorMin)
            self.colorPanel.layout().itemAt(6).widget().setText(str(
                self.customGlWidget.selColorMax))
            self.colorPanel.layout().itemAt(4).widget().validator().setTop(
                self.customGlWidget.selColorMax)
            slider = self.colorPanel.layout().itemAt(7).widget()
            center = 0.5 * (extents[0] + extents[1])
            halfWidth = (extents[1] - extents[0]) * 0.5
            newMin = self.customGlWidget.colorMin + halfWidth
            newMax = self.customGlWidget.colorMax - halfWidth
            newRange = (newMax - newMin) * 0.01
            slider.setRange(newMin, newMax, newRange)
            slider.setValue(center)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except:
            pass

    def updateColorSel(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        try:
            extents = list(self.paletteWidget.span.extents)
            width = np.abs(extents[1] - extents[0])
            self.customGlWidget.selColorMin = position - 0.5 * width
            self.customGlWidget.selColorMax = position + 0.5 * width
            self.colorPanel.layout().itemAt(4).widget().setText(
                '{0:.3f}'.format(position - 0.5 * width))
            self.colorPanel.layout().itemAt(6).widget().validator().setBottom(
                position - 0.5 * width)
            self.colorPanel.layout().itemAt(6).widget().setText(
                '{0:.3f}'.format(position + 0.5 * width))
            self.colorPanel.layout().itemAt(4).widget().validator().setTop(
                position + 0.5 * width)
            newExtents = (position - 0.5 * width, position + 0.5 * width,
                          extents[2], extents[3])
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except:
            pass

    def updateColorSelFromQLE(self):
        try:
            cPan = self.sender()
            cIndex = cPan.parent().layout().indexOf(cPan)
            value = float(str(cPan.text()))
            extents = list(self.paletteWidget.span.extents)
            slider = self.colorPanel.layout().itemAt(7).widget()
            if cIndex == 4:
                self.customGlWidget.selColorMin = value
                newExtents = (value, extents[1],
                              extents[2], extents[3])
                self.colorPanel.layout().itemAt(6).widget().validator(
                    ).setBottom(value)
            else:
                self.customGlWidget.selColorMax = value
                newExtents = (extents[0], value,
                              extents[2], extents[3])
                self.colorPanel.layout().itemAt(4).widget().validator().setTop(
                    value)
            center = 0.5 * (newExtents[0] + newExtents[1])
            halfWidth = (newExtents[1] - newExtents[0]) * 0.5
            newMin = self.customGlWidget.colorMin + halfWidth
            newMax = self.customGlWidget.colorMax - halfWidth
            newRange = (newMax - newMin) * 0.01
            slider.setRange(newMin, newMax, newRange)
            slider.setValue(center)
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except:
            pass

    def projSelection(self, state):
        cPan = self.sender()
        projIndex = int(cPan.objectName[-1])
        self.customGlWidget.projectionsVisibility[projIndex] = state
        self.customGlWidget.glDraw()

    def updateRotation(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        if cPan.objectName[-1] == 'x':
            self.customGlWidget.rotations[0][0] = np.float32(position)
        elif cPan.objectName[-1] == 'y':
            self.customGlWidget.rotations[1][0] = np.float32(position)
        elif cPan.objectName[-1] == 'z':
            self.customGlWidget.rotations[2][0] = np.float32(position)
        self.customGlWidget.updateQuats()
        self.customGlWidget.glDraw()

    def updateRotationFromGL(self, actPos):
        self.rotationPanel.layout().itemAt(2).widget().setValue(actPos[0][0])
        self.rotationPanel.layout().itemAt(5).widget().setValue(actPos[1][0])
        self.rotationPanel.layout().itemAt(8).widget().setValue(actPos[2][0])

    def updateRotationFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)

    def updateScale(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        if cPan.objectName[-1] == 'x':
            self.customGlWidget.scaleVec[0] =\
                np.float32(np.power(10, position))
        elif cPan.objectName[-1] == 'y':
            self.customGlWidget.scaleVec[1] =\
                np.float32(np.power(10, position))
        elif cPan.objectName[-1] == 'z':
            self.customGlWidget.scaleVec[2] =\
                np.float32(np.power(10, position))
        self.customGlWidget.glDraw()

    def updateScaleFromGL(self, scale):
        self.zoomPanel.layout().itemAt(2).widget().setValue(np.log10(scale[0]))
        self.zoomPanel.layout().itemAt(5).widget().setValue(np.log10(scale[1]))
        self.zoomPanel.layout().itemAt(8).widget().setValue(np.log10(scale[2]))

    def updateScaleFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)

    def updateFontSize(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
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
            menu = QMenu()
            menu.addAction('Center here',
                           partial(self.centerEl, str(selectedItem.text())))
            menu.exec_(self.oeTree.viewport().mapToGlobal(position))
        else:
            pass

    def updateScene(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        aIndex = int(((cIndex + 1) / 3) - 1)
        if position != 0:
            self.customGlWidget.aPos[aIndex] = np.float32(position)
            self.customGlWidget.glDraw()

    def updateSceneFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)

    def glMenu(self, position):
        menu = QMenu()
        for actText, actFunc in zip(['Export to image', 'Save scene geometry',
                                     'Load scene geometry'],
                                    [self.exportToImage, self.saveSceneDialog,
                                     self.loadSceneDialog]):
            mAction = QAction(self)
            mAction.setText(actText)
            mAction.triggered.connect(actFunc)
            menu.addAction(mAction)
        menu.addSeparator()
        mAction = QAction(self)
        mAction.setText("Show Virtual Screen")
        mAction.setCheckable(True)
        mAction.setChecked(False if self.customGlWidget.virtScreen is None
                           else True)
        mAction.triggered.connect(self.customGlWidget.toggleVScreen)
        menu.addAction(mAction)
        menu.addSeparator()
        for iAction, actCnt in enumerate(self.sceneControls):
            mAction = QAction(self)
            mAction.setText(actCnt[0].text())
            mAction.setCheckable(True)
            mAction.setChecked(bool(actCnt[1].checkState()))
            mAction.triggered.connect(partial(self.setSceneParam, iAction))
            menu.addAction(mAction)
        menu.addSeparator()
        for iAction, actCnt in enumerate(self.projectionControls):
            mAction = QAction(self)
            mAction.setText(actCnt[0].text())
            mAction.setCheckable(True)
            mAction.setChecked(bool(actCnt[1].checkState()))
            mAction.triggered.connect(partial(self.setProjectionParam,
                                              iAction))
            menu.addAction(mAction)
        menu.exec_(self.customGlWidget.mapToGlobal(position))

    def exportToImage(self):
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setNameFilter("BMP files (*.bmp);;JPG files (*.jpg);;JPEG files (*.jpeg);;PNG files (*.png);;TIFF files (*.tif)")  # analysis:ignore
        saveDialog.selectNameFilter("JPG files (*.jpg)")
        if (saveDialog.exec_()):
            image = self.customGlWidget.grabFrameBuffer(withAlpha=True)
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
#            print filename
            image.save(filename)

    def saveSceneDialog(self):
        saveDialog = QFileDialog()
        saveDialog.setFileMode(QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setNameFilter("Numpy files (*.npy)")  # analysis:ignore
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = 'npy'
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            self.saveScene(filename)

    def loadSceneDialog(self):
        loadDialog = QFileDialog()
        loadDialog.setFileMode(QFileDialog.AnyFile)
        loadDialog.setAcceptMode(QFileDialog.AcceptOpen)
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
        print(self.geometry())
        params['sizeGL'] = self.canvasSplitter.sizes()
        params['colorAxis'] = str(self.colorPanel.layout().itemAt(2).widget(
            ).currentText())
        try:
            np.save(filename, params)
        except:
            print('Error saving file')
            return
        print('Saved scene to {}'.format(filename))

    def loadScene(self, filename):
        try:
            params = np.load(filename).item()
        except:
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
        for axis in range(3):
            self.zoomPanel.layout().itemAt((axis+1)*3-1).widget().setValue(
                np.log10(self.customGlWidget.scaleVec[axis]))
        self.blockSignals(True)
        self.rotationPanel.layout().itemAt(2).widget().setValue(
            self.customGlWidget.rotations[0][0])
        self.rotationPanel.layout().itemAt(5).widget().setValue(
            self.customGlWidget.rotations[1][0])
        self.rotationPanel.layout().itemAt(8).widget().setValue(
            self.customGlWidget.rotations[2][0])
        self.customGlWidget.updateQuats()

        self.opacityPanel.layout().itemAt(2).widget().setValue(
            self.customGlWidget.lineOpacity)
        self.opacityPanel.layout().itemAt(5).widget().setValue(
            self.customGlWidget.lineWidth)
        self.opacityPanel.layout().itemAt(8).widget().setValue(
            self.customGlWidget.pointOpacity)
        self.opacityPanel.layout().itemAt(11).widget().setValue(
            self.customGlWidget.pointSize)

        for axis in range(3):
            self.projVisPanel.layout().itemAt(axis*2).widget().setCheckState(
                int(self.customGlWidget.projectionsVisibility[axis]))

        self.projVisPanel.layout().itemAt(6).widget().setCheckState(
                        int(self.customGlWidget.drawGrid)*2)
        self.projVisPanel.layout().itemAt(8).widget().setCheckState(
                        int(self.customGlWidget.fineGridEnabled)*2)
        self.projVisPanel.layout().itemAt(10).widget().setCheckState(
                        int(self.customGlWidget.perspectiveEnabled)*2)

        self.projLinePanel.layout().itemAt(2).widget().setValue(
            self.customGlWidget.lineProjectionOpacity)
        self.projLinePanel.layout().itemAt(5).widget().setValue(
            self.customGlWidget.lineProjectionWidth)
        self.projLinePanel.layout().itemAt(8).widget().setValue(
            self.customGlWidget.pointProjectionOpacity)
        self.projLinePanel.layout().itemAt(11).widget().setValue(
            self.customGlWidget.pointProjectionSize)

        for axis in range(3):
            self.scenePanel.layout().itemAt((axis+1)*3-1).widget(
                ).setValue(self.customGlWidget.aPos[axis])

        self.scenePanel.layout().itemAt(17).widget(
            ).setCheckState(int(self.customGlWidget.invertColors)*2)
        self.scenePanel.layout().itemAt(19).widget(
            ).setCheckState(int(self.customGlWidget.useScalableFont)*2)

        self.colorPanel.layout().itemAt(12).widget(
            ).setCheckState(int(self.customGlWidget.globalNorm)*2)
        self.colorPanel.layout().itemAt(14).widget(
            ).setCheckState(int(self.customGlWidget.iHSV)*2)
        self.blockSignals(False)
        self.mplFig.canvas.draw()
        colorCB = self.colorPanel.layout().itemAt(2).widget()
        colorCB.setCurrentIndex(colorCB.findText(params['colorAxis']))
        newExtents = list(self.paletteWidget.span.extents)
        newExtents[0] = params['selColorMin']
        newExtents[1] = params['selColorMax']
        try:
            self.paletteWidget.span.extents = newExtents
        except:
            pass
        self.updateColorSelFromMPL(0, 0)

        print('Loaded scene from {}'.format(filename))

    def centerEl(self, oeName):
        self.customGlWidget.coordOffset = list(self.oesList[str(oeName)][2])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

#    def updateCutoff(self, position):
#        try:
#            cPan = self.sender()
#            if isinstance(position, int):
#                try:
#                    position /= cPan.scale
#                except:
#                    pass
#            cIndex = cPan.parent().layout().indexOf(cPan)
#            cPan.parent().layout().itemAt(cIndex-1).widget().setText(
#                str(position))
#            extents = list(self.paletteWidget.span.extents)
#            self.customGlWidget.cutoffI = np.float32(position)
#            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
#            newExtents = (extents[0], extents[1],
#                          self.customGlWidget.cutoffI, extents[3])
#            self.paletteWidget.span.extents = newExtents
#            self.customGlWidget.glDraw()
#        except:
#            pass

    def updateCutoffFromQLE(self):
        try:
            cPan = self.sender()
#            cIndex = cPan.parent().layout().indexOf(cPan)
            value = float(str(cPan.text()))
#            cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.cutoffI = np.float32(value)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            newExtents = (extents[0], extents[1],
                          self.customGlWidget.cutoffI, extents[3])
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.glDraw()
        except:
            pass

    def updateExplosionDepth(self):
        try:
            cPan = self.sender()
#            cIndex = cPan.parent().layout().indexOf(cPan)
            value = float(str(cPan.text()))
#            cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
            self.customGlWidget.depthScaler = np.float32(value)
            if self.customGlWidget.virtScreen is not None:
                self.customGlWidget.populateVScreen()
                self.customGlWidget.glDraw()
        except:
            pass

    def updateOpacityFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
        self.customGlWidget.glDraw()

    def updateOpacity(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        objNameType = cPan.objectName[-1]
        if objNameType == '0':
            self.customGlWidget.lineOpacity = np.float32(position)
        elif objNameType == '1':
            self.customGlWidget.lineWidth = np.float32(position)
        elif objNameType == '2':
            self.customGlWidget.pointOpacity = np.float32(position)
        elif objNameType == '3':
            self.customGlWidget.pointSize = np.float32(position)
        self.customGlWidget.glDraw()

    def updateProjectionOpacityFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
        self.customGlWidget.glDraw()

#    def updateTile(self, position):
#        cPan = self.sender()
#        cIndex = cPan.parent().layout().indexOf(cPan)
#        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
#        objNameType = cPan.objectName[-1]
#        if objNameType == 'X':
#            self.customGlWidget.tiles[0] = np.int(position)
#        elif objNameType == 'Y':
#            self.customGlWidget.tiles[1] = np.int(position)
#        self.customGlWidget.glDraw()

    def updateTileFromQLE(self):
        cPan = self.sender()
#        cIndex = cPan.parent().layout().indexOf(cPan)
        value = int(str(cPan.text()))
#        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
        objNameType = cPan.objectName[-1]
        if objNameType == 'X':
            self.customGlWidget.tiles[0] = np.int(value)
        elif objNameType == 'Y':
            self.customGlWidget.tiles[1] = np.int(value)
        self.customGlWidget.glDraw()

    def updateProjectionOpacity(self, position):
        cPan = self.sender()
        if isinstance(position, int):
            try:
                position /= cPan.scale
            except:
                pass
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        objNameType = cPan.objectName[-1]
        if objNameType == '0':
            self.customGlWidget.lineProjectionOpacity = np.float32(position)
        elif objNameType == '1':
            self.customGlWidget.lineProjectionWidth = np.float32(position)
        elif objNameType == '2':
            self.customGlWidget.pointProjectionOpacity = np.float32(position)
        elif objNameType == '3':
            self.customGlWidget.pointProjectionSize = np.float32(position)
        self.customGlWidget.glDraw()


class xrtGlWidget(QGLWidget):
    rotationUpdated = QtCore.pyqtSignal(np.ndarray)
    scaleUpdated = QtCore.pyqtSignal(np.ndarray)
    histogramUpdated = QtCore.pyqtSignal(tuple)

    def __init__(self, parent, arrayOfRays, modelRoot, oesList, b2els, signal):
        QGLWidget.__init__(self, parent)
        self.QookSignal = signal
        self.virtScreen = None
        self.virtBeam = None
        self.virtDotsArray = None
        self.virtDotsColor = None
        self.isVirtScreenNormal = False
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
        self.beamsToElements = b2els

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
        self.colorMax = -1e20
        self.colorMin = 1e20
        self.scaleVec = np.array([1e3, 1e1, 1e3])
        self.maxLen = 1.
        self.showLostRays = False
        self.populateVerticesArray(modelRoot)

        self.drawGrid = True
        self.fineGridEnabled = False
        self.showOeLabels = False
        self.aPos = [0.9, 0.9, 0.9]
        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])
        self.useScalableFont = False
        self.fontSize = 5
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
        self.glDraw()

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
            glRotatef(*self.rotations[0])
            glRotatef(*self.rotations[1])
            glRotatef(*self.rotations[2])
        else:
            glRotatef(*self.rotationVec)

    def updateQuats(self):
        self.qRot = self.eulerToQ(self.rotations)
        self.rotationVec = self.qToVec(self.qRot)
        self.qText = self.qToVec(
            self.quatMult([self.qRot[0], -self.qRot[1],
                           -self.qRot[2], -self.qRot[3]],
                          self.textOrientation))

    def setPointSize(self, pSize):
        self.pointSize = pSize
        self.glDraw()

    def setLineWidth(self, lWidth):
        self.lineWidth = lWidth
        self.glDraw()

    def populateVerticesArray(self, segmentsModelRoot):
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

        verticesArrayLost = None
        colorsRaysLost = None
        footprintsArrayLost = None
        colorsDotsLost = None
        maxLen = 1.
        tmpMax = -1.e12 * np.ones(3)
        tmpMin = -1. * tmpMax

        if self.newColorAxis:
            newColorMax = -1e20
            newColorMin = 1e20
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
                good = startBeam.state > 0
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
            except:
                continue

            if self.newColorAxis:
                if newColorMax != self.colorMax and\
                        newColorMin != self.colorMin:
                    self.colorMax = newColorMax
                    self.colorMin = newColorMin
                    self.selColorMin = self.colorMin
                    self.selColorMax = self.colorMax

            if ioeItem.hasChildren():
                for isegment in range(ioeItem.rowCount()):
                    segmentItem0 = ioeItem.child(isegment, 0)
                    if segmentItem0.checkState() == 2:
                        endBeam = self.beamsDict[
                            self.oesList[str(segmentItem0.text())[3:]][1]]
                        good = startBeam.state > 0

                        intensity = np.sqrt(np.abs(
                            startBeam.Jss**2 + startBeam.Jpp**2))
                        intensityAll = intensity / np.max(intensity[good])

                        good = np.logical_and(good,
                                              intensityAll >= self.cutoffI)
                        goodC = np.logical_and(
                            self.getColor(startBeam) <= self.selColorMax,
                            self.getColor(startBeam) >= self.selColorMin)

                        good = np.logical_and(good, goodC)

                        if self.globalNorm:
                            alphaMax = 1.
                        else:
                            if len(intensity[good]) > 0:
                                alphaMax = np.max(intensity[good])
                            else:
                                alphaMax = 1.
                        alphaMax = alphaMax if alphaMax != 0 else 1.
                        alphaRays = np.repeat(intensity[good] / alphaMax, 2).T\
                            if alphaRays is None else np.concatenate(
                                (alphaRays.T,
                                 np.repeat(intensity[good] / alphaMax, 2).T))

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
                            except:
                                lostNum = 'None'
                            lost = startBeam.state == lostNum
                            try:
                                lostOnes = len(startBeam.x[lost]) * 2
                            except:
                                lostOnes = 0
                            colorsRaysLost = lostOnes if colorsRaysLost is\
                                None else colorsRaysLost + lostOnes
                            verticesLost = np.array(
                                [startBeam.x[lost] - self.coordOffset[0],
                                 endBeam.x[lost] -
                                 self.coordOffset[0]]).flatten('F')
                            verticesLost = np.vstack((verticesLost, np.array(
                                [startBeam.y[lost] - self.coordOffset[1],
                                 endBeam.y[lost] -
                                 self.coordOffset[1]]).flatten('F')))
                            verticesLost = np.vstack((verticesLost, np.array(
                                [startBeam.z[lost] - self.coordOffset[2],
                                 endBeam.z[lost] -
                                 self.coordOffset[2]]).flatten('F')))
                            verticesArrayLost = verticesLost.T if\
                                verticesArrayLost is None else\
                                np.vstack((verticesArrayLost, verticesLost.T))

            if segmentsModelRoot.child(ioe + 1, 1).checkState() == 2:
                good = startBeam.state > 0
                intensity = np.sqrt(np.abs(
                    startBeam.Jss**2 + startBeam.Jpp**2))
                try:
                    intensityAll = intensity / np.max(intensity[good])
                    good = np.logical_and(good, intensityAll >= self.cutoffI)
                    goodC = np.logical_and(
                        self.getColor(startBeam) <= self.selColorMax,
                        self.getColor(startBeam) >= self.selColorMin)

                    good = np.logical_and(good, goodC)
                except:
                    continue

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
                    except:
                        lostNum = 'None'
                    lost = startBeam.state == lostNum
                    try:
                        lostOnes = len(startBeam.x[lost])
                    except:
                        lostOnes = 0
                    colorsDotsLost = lostOnes if\
                        colorsDotsLost is None else\
                        colorsDotsLost + lostOnes
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
                self.colorMin = self.colorMax * 0.99
                self.colorMax *= 1.01
            if colorsRays is not None:
                colorsRays = (colorsRays-self.colorMin) / (self.colorMax -
                                                           self.colorMin)
                colorsRays = np.dstack((colorsRays,
                                        np.ones_like(alphaRays)*0.85,
                                        alphaRays if self.iHSV else
                                        np.ones_like(alphaRays)))
                colorsRGBRays = np.squeeze(mpl.colors.hsv_to_rgb(colorsRays))
                if self.globalNorm:
                    alphaMax = np.max(alphaRays)
                else:
                    alphaMax = 1.
                alphaColorRays = np.array([alphaRays / alphaMax]).T *\
                    self.lineOpacity
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
        except:
            raise
        try:
            if self.colorMin == self.colorMax:
                self.colorMin = self.colorMax * 0.99
                self.colorMax *= 1.01
            if colorsDots is not None:
                colorsDots = (colorsDots-self.colorMin) / (self.colorMax -
                                                           self.colorMin)
                colorsDots = np.dstack((colorsDots,
                                        np.ones_like(alphaDots)*0.85,
                                        alphaDots if self.iHSV else
                                        np.ones_like(alphaDots)))
                colorsRGBDots = np.squeeze(mpl.colors.hsv_to_rgb(colorsDots))
                if self.globalNorm:
                    alphaMax = np.max(alphaDots)
                else:
                    alphaMax = 1.
                alphaColorDots = np.array([alphaDots / alphaMax]).T *\
                    self.pointOpacity
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
        except:
            pass

        tmpMaxLen = np.max(tmpMax - tmpMin)
        if tmpMaxLen > maxLen:
            maxLen = tmpMaxLen
        self.maxLen = maxLen
        self.newColorAxis = False
        self.populateVScreen()

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

    def drawText(self, coord, text, noScalable=False):
        useScalableFont = False if noScalable else self.useScalableFont
        if not useScalableFont:
            glRasterPos3f(*coord)
            for symbol in text:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,
                                    ord(symbol))
        else:
            glPushMatrix()
            glTranslatef(*coord)
            glRotatef(*self.qText)
            fontScale = self.fontSize / 12500.
            glScalef(fontScale, fontScale, fontScale)
            for symbol in text:
                glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(symbol))
            glPopMatrix()

    def paintGL(self):
        def setMaterial(mat):
            if mat == 'Cu':
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,
                             [0.3, 0.15, 0.15, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,
                             [0.4, 0.25, 0.15, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,
                             [1., 0.7, 0.3, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,
                             [0.1, 0.1, 0.1, 1])
                glMaterialf(GL_FRONT, GL_SHININESS, 100)
            else:
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,
                             [0.1, 0.1, 0.1, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,
                             [0.3, 0.3, 0.3, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,
                             [1., 0.9, 0.8, 1])
                glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,
                             [0.1, 0.1, 0.1, 1])
                glMaterialf(GL_FRONT, GL_SHININESS, 100)

        def makeCenterStr(centerList, prec):
            retStr = '('
            for dim in centerList:
                retStr += '{0:.{1}f}, '.format(dim, prec)
            return retStr[:-2] + ')'

        if self.invertColors:
            glClearColor(1.0, 1.0, 1.0, 1.)
        else:
            glClearColor(0.0, 0.0, 0.0, 1.)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspectiveEnabled:
            gluPerspective(self.cameraAngle, self.aspect, 0.001, 10000)
        else:
            orthoView = self.cameraPos[0]*0.45
            glOrtho(-orthoView*self.aspect, orthoView*self.aspect,
                    -orthoView, orthoView, -100, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(self.cameraPos[0], self.cameraPos[1], self.cameraPos[2],
                  self.cameraTarget[0], self.cameraTarget[1],
                  self.cameraTarget[2],
                  0.0, 0.0, 1.0)

        if self.enableBlending:
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glEnableClientState(GL_VERTEX_ARRAY)

        glEnableClientState(GL_COLOR_ARRAY)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        self.rotateZYX()

        pModel = np.array(glGetDoublev(GL_MODELVIEW_MATRIX))[:-1, :-1]
        self.visibleAxes = np.argmax(np.abs(pModel), axis=0)
        self.signs = np.sign(pModel)
        self.axPosModifier = np.ones(3)

        if self.enableAA:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
#            glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

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
                        0, GL_LINES, projectionRays, self.raysColor,
                        self.lineProjectionOpacity, self.lineProjectionWidth)

                if self.pointProjectionSize > 0 and\
                        self.pointProjectionOpacity > 0:
                    if self.footprintsArray is not None:
                        projectionDots = self.modelToWorld(
                            np.copy(self.footprintsArray))
                        projectionDots[:, dim] =\
                            -self.aPos[dim] * self.axPosModifier[dim]
                        self.drawArrays(
                            0, GL_POINTS, projectionDots, self.dotsColor,
                            self.pointProjectionOpacity,
                            self.pointProjectionSize)

                    if self.virtDotsArray is not None:
                        projectionDots = self.modelToWorld(
                            np.copy(self.virtDotsArray))
                        projectionDots[:, dim] =\
                            -self.aPos[dim] * self.axPosModifier[dim]
                        self.drawArrays(
                            0, GL_POINTS, projectionDots, self.virtDotsColor,
                            self.pointProjectionOpacity,
                            self.pointProjectionSize)

        if self.enableAA:
            glDisable(GL_LINE_SMOOTH)

        glEnable(GL_DEPTH_TEST)
        if self.drawGrid:  # Coordinate grid box
            self.drawCoordinateGrid()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        if len(self.oesToPlot) > 0:  # Surfaces of optical elements
            glEnableClientState(GL_NORMAL_ARRAY)
            glEnable(GL_NORMALIZE)

            self.addLighting(3.)
            for oeString in self.oesToPlot:
                try:
                    oeToPlot = self.oesList[oeString][0]
                    is2ndXtal = self.oesList[oeString][3]
                    elType = str(type(oeToPlot))
                    if len(re.findall('raycing.oe', elType.lower())) > 0:  # OE
                        setMaterial('Si')
                        self.plotOeSurface(oeToPlot, is2ndXtal)
                    elif len(re.findall('raycing.apert', elType)) > 0:
                        setMaterial('Cu')
                        self.plotAperture(oeToPlot)
                    else:
                        continue
                except:
                    continue

            glDisable(GL_LIGHTING)
            glDisable(GL_NORMALIZE)
            glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_DEPTH_TEST)

        if self.linesDepthTest:
            glEnable(GL_DEPTH_TEST)

        if self.enableAA:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        if self.lineWidth > 0 and self.lineOpacity > 0 and\
                self.verticesArray is not None:
            self.drawArrays(1, GL_LINES, self.verticesArray, self.raysColor,
                            self.lineOpacity, self.lineWidth)
        if self.linesDepthTest:
            glDisable(GL_DEPTH_TEST)

        if self.enableAA:
            glDisable(GL_LINE_SMOOTH)

        if self.enableAA:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_DEPTH_TEST)
        if len(self.oesToPlot) > 0:
            for oeString in self.oesToPlot:
                oeToPlot = self.oesList[oeString][0]
                elType = str(type(oeToPlot))
                if len(re.findall('raycing.screens.Sc', elType)) > 0:  # screen
                    self.plotScreen(oeToPlot)
                elif len(re.findall('raycing.screens.Hemispher', elType)) > 0:
                    self.plotHemiScreen(oeToPlot)
                else:
                    continue

        if self.virtScreen is not None:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

            self.plotScreen(self.virtScreen, [self.vScreenSize]*2,
                            [1, 0, 0, 1], plotFWHM=True)

#            if not self.enableAA:
#                glDisable(GL_LINE_SMOOTH)

        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.pointsDepthTest:
            glEnable(GL_DEPTH_TEST)

        if self.pointSize > 0 and self.pointOpacity > 0:
            if self.footprintsArray is not None:
                self.drawArrays(1, GL_POINTS, self.footprintsArray,
                                self.dotsColor, self.pointOpacity,
                                self.pointSize)

            if self.virtDotsArray is not None:
                self.drawArrays(1, GL_POINTS, self.virtDotsArray,
                                self.virtDotsColor, self.pointOpacity,
                                self.pointSize)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if self.enableAA:
            glDisable(GL_LINE_SMOOTH)

        if self.pointsDepthTest:
            glDisable(GL_DEPTH_TEST)

        oeLabels = dict()
        if len(self.labelsToPlot) > 0:
            for oeKey, oeValue in self.oesList.items():
                if oeKey in self.labelsToPlot:
                    oeCenterStr = makeCenterStr(oeValue[2],
                                                self.labelCoordPrec)
                    addStr = True
                    for oeLabelKey, oeLabelValue in oeLabels.items():
                        if np.all(np.round(
                                oeLabelValue[0], self.labelCoordPrec) ==
                                np.round(oeValue[2], self.labelCoordPrec)):
                            oeLabelValue.append(oeKey)
                            addStr = False
                    if addStr:
                        oeLabels[oeCenterStr] = [oeValue[2], oeKey]
            if self.invertColors:
                glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                glColor4f(1.0, 1.0, 1.0, 1.)
            glLineWidth(1)
            for oeKey, oeValue in oeLabels.items():
                outCenterStr = ''
                for oeIndex, oeLabel in enumerate(oeValue):
                    if oeIndex > 0:
                        outCenterStr += '{}, '.format(oeLabel)
                    else:
                        oeCoord = np.array(oeLabel)
                oeCenterStr = '    {0}: {1}mm'.format(
                    outCenterStr[:-2], oeKey)
                oeLabelPos = self.modelToWorld(oeCoord - self.coordOffset)
                self.drawText(oeLabelPos, oeCenterStr)

        if self.showOeLabels and self.virtScreen is not None:
            vsCenterStr = '    {0}: {1}mm'.format(
                'Virtual Screen', makeCenterStr(self.virtScreen.center,
                                                self.labelCoordPrec))
            try:
                pModel = glGetDoublev(GL_MODELVIEW_MATRIX)
                pProjection = glGetDoublev(GL_PROJECTION_MATRIX)
                pView = glGetIntegerv(GL_VIEWPORT)
                m1 = self.modelToWorld(
                    self.virtScreen.frame[1] - self.coordOffset)
                m2 = self.modelToWorld(
                    self.virtScreen.frame[2] - self.coordOffset)
                scr1 = gluProject(
                    *m1, model=pModel,
                    proj=pProjection, view=pView)[0]
                scr2 = gluProject(
                    *m2, model=pModel,
                    proj=pProjection, view=pView)[0]
                lblCenter = self.virtScreen.frame[1] if scr1 > scr2 else\
                    self.virtScreen.frame[2]
            except:
                lblCenter = self.virtScreen.center
            vsLabelPos = self.modelToWorld(lblCenter - self.coordOffset)
            if self.invertColors:
                glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                glColor4f(1.0, 1.0, 1.0, 1.)
            glLineWidth(1)
            self.drawText(vsLabelPos, vsCenterStr)
        glFlush()

        self.drawAxes()
        if self.showHelp:
            self.drawHelp()
        if self.enableBlending:
            glDisable(GL_MULTISAMPLE)
            glDisable(GL_BLEND)
            glDisable(GL_POINT_SMOOTH)

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
            gridColor = np.ones((len(gridArray), 4)) * lineOpacity
            gridArrayVBO = vbo.VBO(np.float32(gridArray))
            gridArrayVBO.bind()
            glVertexPointerf(gridArrayVBO)
            gridColorArray = vbo.VBO(np.float32(gridColor))
            gridColorArray.bind()
            glColorPointerf(gridColorArray)
            glLineWidth(lineWidth)
            glDrawArrays(figType, 0, len(gridArrayVBO))
            gridArrayVBO.unbind()
            gridColorArray.unbind()

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

            gridX = np.arange(np.int(allLimits[:, iAx][0]/step)*step,
                              allLimits[:, iAx][1], step)
            gridX = gridX if gridX[0] >= allLimits[:, iAx][0] else\
                gridX[1:]
            gridLabels.extend([gridX])
            precisionLabels.extend([np.ones_like(gridX)*decimalX])
            axisGridArray.extend([gridX - self.coordOffset[iAx]])
            if self.fineGridEnabled:
                fineStep = step * 0.2
                fineGrid = np.arange(
                    np.int(allLimits[:, iAx][0]/fineStep)*fineStep,
                    allLimits[:, iAx][1], fineStep)
                fineGrid = fineGrid if\
                    fineGrid[0] >= allLimits[:, iAx][0] else fineGrid[1:]
                fineGridArray.extend([fineGrid - self.coordOffset[iAx]])

        axisL, axGrid = populateGrid(axisGridArray)
        if self.fineGridEnabled:
            tmp, fineAxGrid = populateGrid(fineGridArray)

        if self.invertColors:
            glColor4f(0.0, 0.0, 0.0, 1.)
        else:
            glColor4f(1.0, 1.0, 1.0, 1.)
        glLineWidth(1)
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    iAx == self.visibleAxes[2]):
                if iAx == self.visibleAxes[1]:
                    axisL[iAx][self.visibleAxes[2], :] *= 1.05
                    axisL[iAx][self.visibleAxes[0], :] *= 1.05
                if iAx == self.visibleAxes[0]:
                    axisL[iAx][self.visibleAxes[1], :] *= 1.05
                    axisL[iAx][self.visibleAxes[2], :] *= 1.05
                if iAx == self.visibleAxes[2]:
                    axisL[iAx][self.visibleAxes[1], :] *= 1.05
                    axisL[iAx][self.visibleAxes[0], :] *= 1.05
                for tick, tText, pcs in list(zip(axisL[iAx].T, gridLabels[iAx],
                                                 precisionLabels[iAx])):
                    valueStr = "{0:.{1}f}".format(tText, int(pcs))
                    self.drawText(tick, valueStr)
#            if not self.enableAA:
#                glDisable(GL_LINE_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        drawGridLines(np.vstack((back, side, bottom)), 2., 0.75, GL_QUADS)
        drawGridLines(axGrid, 1., 0.5, GL_LINES)
        if self.fineGridEnabled:
            drawGridLines(fineAxGrid, 1., 0.25, GL_LINES)
        glDisable(GL_LINE_SMOOTH)

    def drawArrays(self, tr, geom, vertices, colors, lineOpacity, lineWidth):

        if bool(tr):
            vertexArray = vbo.VBO(self.modelToWorld(vertices))
        else:
            vertexArray = vbo.VBO(vertices)
        vertexArray.bind()
        glVertexPointerf(vertexArray)
        colors[:, 3] = np.float32(lineOpacity)
        colorArray = vbo.VBO(colors)
        colorArray.bind()
        glColorPointerf(colorArray)
        if geom == GL_LINES:
            glLineWidth(lineWidth)
        else:
            glPointSize(lineWidth)
        glDrawArrays(geom, 0, len(vertices))
        colorArray.unbind()
        vertexArray.unbind()

    def plotOeSurface(self, oe, is2ndXtal):
        glEnable(GL_MAP2_VERTEX_3)
        glEnable(GL_MAP2_NORMAL)
        nsIndex = int(is2ndXtal)
        if is2ndXtal:
            xLimits = list(oe.limOptX2) if\
                oe.limOptX2 is not None else oe.limPhysX2
            if np.any(np.abs(xLimits) == raycing.maxHalfSizeOfOE):
                if oe.footprint is not None:
                    xLimits = oe.footprint[nsIndex][:, 0]
            yLimits = list(oe.limOptY2) if\
                oe.limOptY2 is not None else oe.limPhysY2
            if np.any(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
                if oe.footprint is not None:
                    yLimits = oe.footprint[nsIndex][:, 1]
        else:
            xLimits = list(oe.limOptX) if\
                oe.limOptX is not None else oe.limPhysX
            if np.any(np.abs(xLimits) == raycing.maxHalfSizeOfOE):
                if oe.footprint is not None:
                    xLimits = oe.footprint[nsIndex][:, 0]
            yLimits = list(oe.limOptY) if\
                oe.limOptY is not None else oe.limPhysY
            if np.any(np.abs(yLimits) == raycing.maxHalfSizeOfOE):
                if oe.footprint is not None:
                    yLimits = oe.footprint[nsIndex][:, 1]
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

                if is2ndXtal:
                    zExt = '2'
                else:
                    zExt = '1' if hasattr(oe, 'local_z1') else ''
                local_z = getattr(oe, 'local_z{}'.format(zExt))
                local_n = getattr(oe, 'local_n{}'.format(zExt))

                zv = local_z(xv, yv)
                nv = local_n(xv, yv)

                gbp = rsources.Beam(nrays=len(xv))
                gbp.x = xv
                gbp.y = yv
                gbp.z = zv

                gbp.a = nv[0] * np.ones_like(zv)
                gbp.b = nv[1] * np.ones_like(zv)
                gbp.c = nv[2] * np.ones_like(zv)

                oe.local_to_global(gbp, is2ndXtal=is2ndXtal)
                surfCP = np.vstack((gbp.x - self.coordOffset[0],
                                    gbp.y - self.coordOffset[1],
                                    gbp.z - self.coordOffset[2])).T

                glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                        self.modelToWorld(surfCP.reshape(
                            self.surfCPOrder,
                            self.surfCPOrder, 3)))

                surfNorm = np.vstack((gbp.a, gbp.b, gbp.c,
                                      np.ones_like(gbp.a))).T

                glMap2f(GL_MAP2_NORMAL, 0, 1, 0, 1,
                        surfNorm.reshape(
                            self.surfCPOrder,
                            self.surfCPOrder, 4))

                glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
                            self.surfCPOrder, 0.0, 1.0)

                glEvalMesh2(GL_FILL, 0, self.surfCPOrder,
                            0, self.surfCPOrder)

        glDisable(GL_MAP2_VERTEX_3)
        glDisable(GL_MAP2_NORMAL)

    def plotAperture(self, oe):
        surfCPOrder = self.surfCPOrder
        glEnable(GL_MAP2_VERTEX_3)
        glEnable(GL_MAP2_NORMAL)

        if oe.shape == 'round':
            r = oe.r
            w = r
            h = r
            cX = 0
            cY = 0
            wf = r
        else:
            opening = oe.opening
            w = np.abs(opening[1]-opening[0]) * 0.5
            h = np.abs(opening[3]-opening[2]) * 0.5
            cX = 0.5 * (opening[1]+opening[0])
            cY = 0.5 * (opening[3]+opening[2])
            wf = min(w, h)
        isBeamStop = len(re.findall('Stop', str(type(oe)))) > 0
        if isBeamStop:  # BeamStop
            limits = list(zip([0], [w], [0], [h]))
        else:
            limits = list(zip([0, w], [w+wf, w+wf], [h, 0], [h+wf, h]))
        for ix in [1, -1]:
            for iy in [1, -1]:
                for xMin, xMax, yMin, yMax in limits:
                    if oe.shape == 'round':
                        xMin = 0
                        tiles = self.tiles[1] * 5
                    else:
                        tiles = self.tiles[1]
                    xGridOe = np.linspace(xMin, xMax, surfCPOrder)

                    for k in range(tiles):
                        deltaY = (yMax - yMin) / float(tiles)
                        yGridOe = np.linspace(yMin + k*deltaY,
                                              yMin + (k+1)*deltaY,
                                              surfCPOrder)
                        xv, yv = np.meshgrid(xGridOe, yGridOe)
                        if oe.shape == 'round' and yMin == 0:
                            phi = np.arcsin(yGridOe/r)
                            if isBeamStop:
                                xv = xv * (r * np.cos(phi) /
                                           (w + wf))[:, np.newaxis]
                            else:
                                xv = xv * (1 - r * np.cos(phi) /
                                           (w + wf))[:, np.newaxis] +\
                                    (r * np.cos(phi))[:, np.newaxis]
                        xv *= ix
                        yv *= iy
                        xv = xv.flatten() + cX
                        yv = yv.flatten() + cY

                        gbp = rsources.Beam(nrays=len(xv))
                        gbp.x = xv
                        gbp.y = np.zeros_like(xv)
                        gbp.z = yv

                        gbp.a = np.zeros_like(xv)
                        gbp.b = np.ones_like(xv)
                        gbp.c = np.zeros_like(xv)

                        oe.local_to_global(gbp)
                        surfCP = np.vstack((gbp.x - self.coordOffset[0],
                                            gbp.y - self.coordOffset[1],
                                            gbp.z - self.coordOffset[2])).T

                        glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                                self.modelToWorld(surfCP.reshape(
                                    surfCPOrder,
                                    surfCPOrder, 3)))

                        surfNorm = np.vstack((gbp.a, gbp.b, gbp.c,
                                              np.ones_like(gbp.a))).T

                        glMap2f(GL_MAP2_NORMAL, 0, 1, 0, 1,
                                surfNorm.reshape(
                                    surfCPOrder,
                                    surfCPOrder, 4))

                        glMapGrid2f(surfCPOrder*4, 0.0, 1.0,
                                    surfCPOrder*4, 0.0, 1.0)

#                        glEnable(GL_POLYGON_SMOOTH)
#                        glLineWidth(3)
#                        glEvalMesh2(GL_LINE, 0, surfCPOrder*4,
#                                    0, surfCPOrder*4)
                        glEvalMesh2(GL_FILL, 0, surfCPOrder*4,
                                    0, surfCPOrder*4)

        glDisable(GL_MAP2_VERTEX_3)
        glDisable(GL_MAP2_NORMAL)

    def plotScreen(self, oe, dimensions=None, frameColor=None, plotFWHM=False):
        scAbsZ = np.linalg.norm(oe.z * self.scaleVec)
        scAbsX = np.linalg.norm(oe.x * self.scaleVec)
        if dimensions is not None:
            vScrHW = dimensions[0]
            vScrHH = dimensions[1]
        else:
            vScrHW = self.vScreenSize
            vScrHH = self.vScreenSize

        vScreenBody = np.zeros((4, 3))
        vScreenBody[0, :] = vScreenBody[1, :] =\
            oe.center - vScrHW * np.array(oe.x) * self.maxLen / scAbsX
        vScreenBody[2, :] = vScreenBody[3, :] =\
            oe.center + vScrHW * np.array(oe.x) * self.maxLen / scAbsX
        vScreenBody[0, :] -=\
            vScrHH * np.array(oe.z) * self.maxLen / scAbsZ
        vScreenBody[3, :] -=\
            vScrHH * np.array(oe.z) * self.maxLen / scAbsZ
        vScreenBody[1, :] +=\
            vScrHH * np.array(oe.z) * self.maxLen / scAbsZ
        vScreenBody[2, :] +=\
            vScrHH * np.array(oe.z) * self.maxLen / scAbsZ

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)

        if self.invertColors:
            glColor4f(0.0, 0.0, 0.0, 0.2)
        else:
            glColor4f(1.0, 1.0, 1.0, 0.2)

        for i in range(4):
            glVertex3f(*self.modelToWorld(vScreenBody[i, :] -
                                          self.coordOffset))
        glEnd()

        if frameColor is not None:
            self.virtScreen.frame = vScreenBody
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2)
            glBegin(GL_QUADS)
            glColor4f(*frameColor)
            for i in range(4):
                glVertex3f(*self.modelToWorld(vScreenBody[i, :] -
                                              self.coordOffset))
            glEnd()

        if plotFWHM:
            glLineWidth(1)
            glDisable(GL_LINE_SMOOTH)
            if self.invertColors:
                glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                glColor4f(1.0, 1.0, 1.0, 1.)
            startVec = np.array([0, 1, 0])
            destVec = np.array(oe.y / self.scaleVec)
            rotVec = np.cross(startVec, destVec)
            rotAngle = np.degrees(np.arccos(
                np.dot(startVec, destVec) /
                np.linalg.norm(startVec) / np.linalg.norm(destVec)))
            rotVecGL = np.float32(np.hstack((rotAngle, rotVec)))

            pView = glGetIntegerv(GL_VIEWPORT)
            pModel = glGetDoublev(GL_MODELVIEW_MATRIX)
            pProjection = glGetDoublev(GL_PROJECTION_MATRIX)
            scr = np.zeros((3, 3))
            for iAx in range(3):
                scr[iAx] = np.array(gluProject(
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

                glPushMatrix()
                glTranslatef(*coord)
                glRotatef(*rotVecGL)
                glTranslatef(*coordShift)
                glRotatef(180.*(vFlip*0.5), 1, 0, 0)
                glRotatef(180.*(hFlip*0.5), 0, 0, 1)
                if iAx > 0:
                    glRotatef(-90, 0, 1, 0)
                if iAx == 0:  # Horizontal Label to half height
                    glTranslatef(0, 0, -50. * fontScale)
                else:  # Vertical Label to half height
                    glTranslatef(-50. * fontScale, 0, 0)
                glRotatef(90, 1, 0, 0)
                glScalef(fontScale, fontScale, fontScale)
                for symbol in text:
                    glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, ord(symbol))
                glPopMatrix()
            glEnable(GL_LINE_SMOOTH)

    def plotHemiScreen(self, oe, dimensions=None):
        try:
            rMajor = oe.R
        except:
            rMajor = 1000.

        if dimensions is not None:
            rMinor = dimensions
        else:
            rMinor = self.vScreenSize

        if rMinor > rMajor:
            rMinor = rMajor

        yVec = np.array(oe.x)

        sphereCenter = np.array(oe.center)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.invertColors:
            glColor4f(0.0, 0.0, 0.0, 0.2)
        else:
            glColor4f(1.0, 1.0, 1.0, 0.2)

        glEnable(GL_MAP2_VERTEX_3)

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

                glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                        self.modelToWorld(surfCP.reshape(
                            self.surfCPOrder,
                            self.surfCPOrder, 3)))

                glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
                            self.surfCPOrder, 0.0, 1.0)

                glEvalMesh2(GL_FILL, 0, self.surfCPOrder,
                            0, self.surfCPOrder)

        glDisable(GL_MAP2_VERTEX_3)

    def addLighting(self, pos):
        spot = 60
        exp = 30
        ambient = [0.2, 0.2, 0.2, 1]
        diffuse = [0.5, 0.5, 0.5, 1]
        specular = [1.0, 1.0, 1.0, 1]
        glEnable(GL_LIGHTING)

#        corners = [[-pos, pos, pos, 1], [-pos, -pos, -pos, 1],
#                   [-pos, pos, -pos, 1], [-pos, -pos, pos, 1],
#                   [pos, pos, -pos, 1], [pos, -pos, pos, 1],
#                   [pos, pos, pos, 1], [pos, -pos, -pos, 1]]

        corners = [[0, 0, pos, 1], [0, pos, 0, 1],
                   [pos, 0, 0, 1], [-pos, 0, 0, 1],
                   [0, -pos, 0, 1], [0, 0, -pos, 1]]

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        for iLight in range(len(corners)):
            light = GL_LIGHT0 + iLight
            glEnable(light)
            glLightfv(light, GL_POSITION, corners[iLight])
            glLightfv(light, GL_SPOT_DIRECTION,
                      np.array(corners[len(corners)-iLight-1])/pos)
            glLightfv(light, GL_SPOT_CUTOFF, spot)
            glLightfv(light, GL_SPOT_EXPONENT, exp)
            glLightfv(light, GL_AMBIENT, ambient)
            glLightfv(light, GL_DIFFUSE, diffuse)
            glLightfv(light, GL_SPECULAR, specular)
#            glBegin(GL_LINES)
#            glVertex4f(*corners[iLight])
#            glVertex4f(*corners[len(corners)-iLight-1])
#            glEnd()

    def toggleHelp(self):
        self.showHelp = not self.showHelp
        self.glDraw()

    def drawHelp(self):
        hHeight = 300
        hWidth = 500
        pView = glGetIntegerv(GL_VIEWPORT)
        glViewport(0, self.viewPortGL[3]-hHeight,
                   hWidth, hHeight)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)

        if self.invertColors:
            glColor4f(1.0, 1.0, 1.0, 0.9)
        else:
            glColor4f(0.0, 0.0, 0.0, 0.9)
        backScreen = [[1, 1], [1, -1],
                      [-1, -1], [-1, 1]]
        for corner in backScreen:
                glVertex3f(corner[0], corner[1], 0)

        glEnd()

        if self.invertColors:
            glColor4f(0.0, 0.0, 0.0, 1.0)
        else:
            glColor4f(1.0, 1.0, 1.0, 1.0)
        glLineWidth(3)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBegin(GL_QUADS)
        backScreen = [[1, 1], [1, -1],
                      [-1, -1], [-1, 1]]
        for corner in backScreen:
                glVertex3f(corner[0], corner[1], 0)
        glEnd()

        helpList = ['F1: Open/Close this help window',
                    'F3: Add/Remove Virtual Screen (Slicer)',
                    'F4: Dock/Undock xrtGlow if launched from xrtQook',
                    'F5/F6: Quick Save/Load Scene',
                    'LeftMouse: Rotate the Scene',
                    'SHIFT+LeftMouse: Translate the Beamline in transverse plane',  # analysis:ignore
                    'ALT+LeftMouse: Translate the Beamline in longitudinal direction',  # analysis:ignore
                    'CTRL+LeftMouse: Drag the Slicer',
                    'ALT+LeftMouse: Scale the Slicer',
                    'CTRL+SHIFT+LeftMouse: Translate the Beamline around Slicer (transverse)',  # analysis:ignore
                    'CTRL+ALT+LeftMouse: Translate the Beamline around Slicer (longitudinal)',  # analysis:ignore
                    'CTRL+T: Toggle the Slicer orientation (vertical/normal to the beam)',  # analysis:ignore
                    'WheelMouse: Zoom the Beamline',
                    'CTRL+WheelMouse: Zoom the Scene'
                    ]
        for iLine, text in enumerate(helpList):
            self.drawText([-1. + 0.05,
                           1. - 2. * (iLine + 1) / float(len(helpList)+1), 0],
                          text, True)

        glFlush()
        glViewport(*pView)

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
        gridArray = vbo.VBO(np.float32(coneVertices))
        gridArray.bind()
        glVertexPointerf(gridArray)
        gridColorArray = vbo.VBO(np.float32(gridColor))
        gridColorArray.bind()
        glColorPointerf(gridColorArray)
        glDrawArrays(GL_TRIANGLE_FAN, 0, len(gridArray))
        gridArray.unbind()
        gridColorArray.unbind()

    def drawAxes(self):
        arrowSize = 0.05
        axisLen = 0.1
        tLen = (arrowSize + axisLen) * 2
        glLineWidth(1.)

        pView = glGetIntegerv(GL_VIEWPORT)
        glViewport(0, 0, int(150*self.aspect), 150)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspectiveEnabled:
            gluPerspective(60, self.aspect, 0.001, 10)
        else:
            glOrtho(-tLen*self.aspect, tLen*self.aspect, -tLen, tLen, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(.5, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        self.rotateZYX()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    2-iAx == self.visibleAxes[2]):
                glPushMatrix()
                trVec = np.zeros(3, dtype=np.float32)
                trVec[2-iAx] = axisLen
                glTranslatef(*trVec)
                if iAx == 1:
                    glRotatef(-90, 1.0, 0.0, 0.0)
                elif iAx == 2:
                    glRotatef(90, 0.0, 1.0, 0.0)
                self.drawCone(arrowSize, 0.02, 20, iAx)
                glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glBegin(GL_LINES)
        for iAx in range(3):
            if not (not self.perspectiveEnabled and
                    2-iAx == self.visibleAxes[2]):
                colorVec = [0, 0, 0, 0.75]
                colorVec[iAx] = 1
                glColor4f(*colorVec)
                glVertex3f(0, 0, 0)
                trVec = np.zeros(3, dtype=np.float32)
                trVec[2-iAx] = axisLen
                glVertex3f(*trVec)
                glColor4f(*colorVec)
        glEnd()

        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 2):
            glColor4f(1, 0, 0, 1)
            glRasterPos3f(0, 0, axisLen*1.5)
            for symbol in "  {}, mm".format('Z'):
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))
        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 1):
            glColor4f(0, 0.75, 0, 1)
            glRasterPos3f(0, axisLen*1.5, 0)
            for symbol in "  {}, mm".format('Y'):
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))
        if not (not self.perspectiveEnabled and self.visibleAxes[2] == 0):
            glColor4f(0, 0.5, 1, 1)
            glRasterPos3f(axisLen*1.5, 0, 0)
            for symbol in "  {}, mm".format('X'):
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))
#        glFlush()
        glViewport(*pView)
        glColor4f(1, 1, 1, 1)
        glDisable(GL_LINE_SMOOTH)

    def initializeGL(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glViewport(*self.viewPortGL)

    def resizeGL(self, widthInPixels, heightInPixels):
        self.viewPortGL = [0, 0, widthInPixels, heightInPixels]
        glViewport(*self.viewPortGL)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

    def populateVScreen(self):
        if self.virtBeam is not None:
            try:
                startBeam = self.virtBeam
                good = startBeam.state > 0
                intensity = np.sqrt(np.abs(
                    startBeam.Jss**2 + startBeam.Jpp**2))
                intensityAll = intensity / np.max(intensity[good])

                good = np.logical_and(good,
                                      intensityAll >= self.cutoffI)
                goodC = np.logical_and(
                    self.getColor(startBeam) <= self.selColorMax,
                    self.getColor(startBeam) >= self.selColorMin)

                good = np.logical_and(good, goodC)

                if self.globalNorm:
                    alphaMax = 1.
                else:
                    if len(intensity[good]) > 0:
                        alphaMax = np.max(intensity[good])
                    else:
                        alphaMax = 1.
                alphaMax = alphaMax if alphaMax != 0 else 1.
                alphaDots = intensity[good].T / alphaMax
                colorsDots = np.array(self.getColor(startBeam)[good]).T

                if self.colorMin == self.colorMax:
                    self.colorMin = self.colorMax * 0.99
                    self.colorMax *= 1.01
                colorsDots = (colorsDots-self.colorMin) / (self.colorMax -
                                                           self.colorMin)
                depthDots = copy.deepcopy(colorsDots) * self.depthScaler

                colorsDots = np.dstack((colorsDots,
                                        np.ones_like(alphaDots)*0.85,
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
                if self.globalNorm:
                    alphaMax = np.max(alphaDots)
                else:
                    alphaMax = 1.
                alphaColorDots = np.array([alphaDots / alphaMax]).T *\
                    self.pointOpacity
                self.virtDotsColor = np.float32(np.hstack([colorsRGBDots,
                                                           alphaColorDots]))
                histogram = np.histogram(np.array(
                    self.getColor(startBeam)[good]),
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
                    hwhm = np.abs(histAxis[1][topEl[0]] -
                                  histAxis[1][topEl[-1]]) * 0.5
#                    cntr = (histAxis[1][topEl[0]] +
#                            histAxis[1][topEl[-1]]) * 0.5
                    order = np.floor(np.log10(hwhm))

                    if order >= 2:
                        units = "m"
                        mplier = 1e-3
                    if order >= -1:
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
                            str(axis).upper(), hwhm*mplier*2, units))
            except:
                self.virtDotsArray = None

    def createVScreen(self):
        try:
            self.virtScreen = rscreens.Screen(
                bl=list(self.oesList.values())[0][0].bl)
            self.virtScreen.center = self.worldToModel(np.array([0, 0, 0])) +\
                self.coordOffset
            self.positionVScreen()
            self.glDraw()
        except:
            raise
            self.clearVScreen()

    def positionVScreen(self):
        if self.virtScreen is not None:
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
                except:
                    continue

            if cProj is not None:
                self.virtScreen.center = cProj
                self.virtScreen.beamStart = bStartC
                self.virtScreen.beamEnd = bEndC
                self.virtScreen.beamToExpose = beamStart0

            try:
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
            except:
                pass
            try:
                self.virtBeam = self.virtScreen.expose_global(
                    self.virtScreen.beamToExpose)
                self.populateVScreen()
            except:
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
        self.histogramUpdated.emit((None, None))
        self.glDraw()

    def switchVScreenTilt(self):
        self.isVirtScreenNormal = not self.isVirtScreenNormal
        self.positionVScreen()
        self.glDraw()

    def mouseMoveEvent(self, mEvent):
        pView = glGetIntegerv(GL_VIEWPORT)
        mouseX = mEvent.x()
        mouseY = pView[3] - mEvent.y()
        ctrlOn = bool(int(mEvent.modifiers()) & int(QtCore.Qt.ControlModifier))
        altOn = bool(int(mEvent.modifiers()) & int(QtCore.Qt.AltModifier))
        shiftOn = bool(int(mEvent.modifiers()) & int(QtCore.Qt.ShiftModifier))

        if mEvent.buttons() == QtCore.Qt.LeftButton:
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(self.cameraPos[0], self.cameraPos[1],
                      self.cameraPos[2],
                      self.cameraTarget[0], self.cameraTarget[1],
                      self.cameraTarget[2],
                      0.0, 0.0, 1.0)
            self.rotateZYX()
            pModel = glGetDoublev(GL_MODELVIEW_MATRIX)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()

            if self.perspectiveEnabled:
                gluPerspective(self.cameraAngle, self.aspect, 0.01, 100)
            else:
                orthoView = self.cameraPos[0]*0.45
                glOrtho(-orthoView*self.aspect, orthoView*self.aspect,
                        -orthoView, orthoView, -100, 100)
            pProjection = glGetDoublev(GL_PROJECTION_MATRIX)

            if mEvent.modifiers() == QtCore.Qt.NoModifier:
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
                    pStart = np.array(gluProject(
                        *mStart, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pEnd = np.array(gluProject(
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
                pStart = np.array(gluProject(
                    *mStart, model=pModel, proj=pProjection,
                    view=pView)[:-1])
                pEnd = np.array(gluProject(
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

                    pStart = np.array(gluProject(
                        *normPStart, model=pModel, proj=pProjection,
                        view=pView)[:-1])
                    pEnd = np.array(gluProject(
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
        ctrlOn = bool(int(wEvent.modifiers()) & int(QtCore.Qt.ControlModifier))
        altOn = bool(int(wEvent.modifiers()) & int(QtCore.Qt.AltModifier))
        if QtName == "PyQt4":
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

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:07:53 2017

@author: Roman Chernikov
"""

import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from collections import OrderedDict
try:
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    import PyQt4.Qwt5 as Qwt
    from PyQt4.QtOpenGL import *
except:
    try:
        from PyQt5 import QtGui
        from PyQt5 import QtCore
        import PyQt5.Qwt5 as Qwt
        from PyQt5.QtOpenGL import *
    except:
        sys.exit()
import numpy as np
from functools import partial
import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
import inspect
import backends.raycing as raycing
import backends.raycing.sources as rsources
import re


class xrtGlow(QtGui.QWidget):
    def __init__(self, arrayOfRays):
        super(xrtGlow, self).__init__()

        self.oesList = OrderedDict()
        self.segmentsModel = QtGui.QStandardItemModel()
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()
        self.segmentsModel.setHorizontalHeaderLabels(['Rays',
                                                      'Footprint',
                                                      'Surface'])
        self.beamsToElements = dict()
        oesList = arrayOfRays[2]
        for segment in arrayOfRays[0]:
            if segment[0] == segment[2]:
                oesList[segment[0]].append(segment[1])
                oesList[segment[0]].append(segment[3])

        for segOE, oeRecord in oesList.iteritems():
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
                self.oesList[elName].append(center)
                print elName, center
                self.oesList[elName].append(is2ndXtal)

        headerRow = []
        for i in range(3):
            child = QtGui.QStandardItem("")
            child.setEditable(False)
            child.setCheckable(True)
            child.setCheckState(0)
            headerRow.append(child)
        self.segmentsModelRoot.appendRow(headerRow)

        for element, elRecord in self.oesList.iteritems():
            child0 = QtGui.QStandardItem(str(element))
            child0.setEditable(False)
            child0.setCheckable(False)
            child1 = QtGui.QStandardItem("")
            child1.setEditable(False)
            child1.setCheckable(True)
            child1.setCheckState(2)
            child2 = QtGui.QStandardItem("")
            child2.setEditable(False)
            child2.setCheckable(True)
            child2.setCheckState(0)
            self.segmentsModelRoot.appendRow([child0, child1, child2])
            for segment in arrayOfRays[0]:
                if str(segment[1]) == str(elRecord[1]):
                    child3 = QtGui.QStandardItem(
                        "to {}".format(self.beamsToElements[segment[3]]))
                    child3.setCheckable(True)
                    child3.setCheckState(2)
                    child3.setEditable(False)
                    child0.appendRow([child3, None, None])

        self.fluxDataModel = QtGui.QStandardItemModel()

        for rfName, rfObj in inspect.getmembers(raycing):
            if rfName.startswith('get_') and\
                    rfName != "get_output":
                flItem = QtGui.QStandardItem(rfName.replace("get_", ''))
                self.fluxDataModel.appendRow(flItem)

        self.customGlWidget = xrtGlWidget(self, arrayOfRays,
                                          self.segmentsModelRoot,
                                          self.oesList,
                                          self.beamsToElements)
        self.customGlWidget.rotationUpdated.connect(self.updateRotationFromGL)
        self.customGlWidget.scaleUpdated.connect(self.updateScaleFromGL)
        self.customGlWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customGlWidget.customContextMenuRequested.connect(self.glMenu)
        self.segmentsModel.itemChanged.connect(self.updateRaysList)
#  Zoom panel
        self.zoomPanel = QtGui.QGroupBox(self)
        self.zoomPanel.setFlat(False)
#        self.zoomPanel.setTitle("Scale")
        zoomLayout = QtGui.QGridLayout()
        scaleValidator = QtGui.QDoubleValidator()
        scaleValidator.setRange(-7, 7, 7)
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis+' (log)')
            axLabel.objectName = "scaleLabel_" + axis
            if iaxis == 1:
                axEdit = QtGui.QLineEdit("1")
            else:
                axEdit = QtGui.QLineEdit("3")
            axEdit.setValidator(scaleValidator)
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
            axSlider.setRange(-7, 7, 0.01)
            if iaxis == 1:
                axSlider.setValue(1)
            else:
                axSlider.setValue(3)
            axEdit.editingFinished.connect(self.updateScaleFromQLE)
            axSlider.objectName = "scaleSlider_" + axis
            axSlider.valueChanged.connect(self.updateScale)
            zoomLayout.addWidget(axLabel, iaxis*2, 0)
            zoomLayout.addWidget(axEdit, iaxis*2, 1)
            zoomLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)

        self.zoomPanel.setLayout(zoomLayout)

#  Rotation panel
        self.rotationPanel = QtGui.QGroupBox(self)
        self.rotationPanel.setFlat(False)
#        self.rotationPanel.setTitle("Rotation")
        rotationLayout = QtGui.QGridLayout()
        rotValidator = QtGui.QDoubleValidator()
        rotValidator.setRange(-180, 180, 9)
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "rotLabel_" + axis
            axEdit = QtGui.QLineEdit("0.")
            axEdit.setValidator(rotValidator)
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
            axSlider.setRange(-180, 180, 0.01)
            axSlider.setValue(0)
            axEdit.editingFinished.connect(self.updateRotationFromQLE)
            axSlider.objectName = "rotSlider_" + axis
            axSlider.valueChanged.connect(self.updateRotation)
            rotationLayout.addWidget(axLabel, iaxis*2, 0)
            rotationLayout.addWidget(axEdit, iaxis*2, 1)
            rotationLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)
        self.rotationPanel.setLayout(rotationLayout)

#  Opacity panel
        self.opacityPanel = QtGui.QGroupBox(self)
        self.opacityPanel.setFlat(False)
#        self.opacityPanel.setTitle("Opacity")
        opacityLayout = QtGui.QGridLayout()
        for iaxis, axis in enumerate(
                ['Line opacity', 'Line width', 'Point opacity', 'Point size']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "opacityLabel_" + str(iaxis)
            opacityValidator = QtGui.QDoubleValidator()
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)

            if iaxis in [0, 2]:
                axSlider.setRange(0, 1., 0.001)
                axSlider.setValue(0.1)
                axEdit = QtGui.QLineEdit("0.1")
                opacityValidator.setRange(0, 1., 5)

            else:
                axSlider.setRange(0, 20, 0.01)
                axSlider.setValue(1.)
                axEdit = QtGui.QLineEdit("1")
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
        self.colorPanel = QtGui.QGroupBox(self)
        self.colorPanel.setFlat(False)
#        self.colorPanel.setTitle("Color")
        colorLayout = QtGui.QGridLayout()
        self.mplFig = mpl.figure.Figure(figsize=(4, 4))
        self.mplAx = self.mplFig.add_subplot(111)

        self.drawColorMap('energy')
        self.paletteWidget = FigCanvas(self.mplFig)
        self.paletteWidget.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                         QtGui.QSizePolicy.Expanding)
        self.paletteWidget.span = mpl.widgets.RectangleSelector(
            self.mplAx, self.updateColorSelFromMPL, drawtype='box',
            useblit=True, rectprops=dict(alpha=0.4, facecolor='white'),
            button=1, interactive=True)

        colorLayout.addWidget(self.paletteWidget, 0, 0, 1, 2)

        colorCBLabel = QtGui.QLabel()
        colorCBLabel.setText('Color Axis:')

        colorCB = QtGui.QComboBox()
        colorCB.setModel(self.fluxDataModel)
        colorCB.setCurrentIndex(colorCB.findText('energy'))
        colorCB.currentIndexChanged['QString'].connect(self.changeColorAxis)
        colorLayout.addWidget(colorCBLabel, 1, 0)
        colorLayout.addWidget(colorCB, 1, 1)
        for icSel, cSelText in enumerate(['Selection<sub>min</sub>',
                                          'Selection<sub>max</sub>']):
            selLabel = QtGui.QLabel()
            selLabel.setText(cSelText)
            selValidator = QtGui.QDoubleValidator()
            selValidator.setRange(self.customGlWidget.colorMin,
                                  self.customGlWidget.colorMax, 3)
            selQLE = QtGui.QLineEdit()
            selQLE.setValidator(selValidator)
            selQLE.setText('{0:.3f}'.format(
                self.customGlWidget.colorMin if icSel == 0 else
                self.customGlWidget.colorMax))
            selQLE.editingFinished.connect(self.updateColorSelFromQLE)
            colorLayout.addWidget(selLabel, 2, icSel)
            colorLayout.addWidget(selQLE, 3, icSel)
        selSlider = Qwt.QwtSlider(
            self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
        rStep = (self.customGlWidget.colorMax -
                 self.customGlWidget.colorMin) / 100.
        rValue = (self.customGlWidget.colorMax +
                  self.customGlWidget.colorMin) * 0.5
        selSlider.setRange(self.customGlWidget.colorMin,
                           self.customGlWidget.colorMax, rStep)
        selSlider.setValue(rValue)
        selSlider.sliderMoved.connect(self.updateColorSel)
        colorLayout.addWidget(selSlider, 4, 0, 1, 2)

        axLabel = QtGui.QLabel()
        axLabel.setText("Intensity cut-off")
        axLabel.objectName = "cutLabel_I"
        axEdit = QtGui.QLineEdit("0.01")
        cutValidator = QtGui.QDoubleValidator()
        cutValidator.setRange(0, 1, 9)
        axEdit.setValidator(cutValidator)
        axSlider = Qwt.QwtSlider(
            self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
        axSlider.setRange(0, 1, 0.001)
        axSlider.setValue(0.01)
        axEdit.editingFinished.connect(self.updateCutoffFromQLE)
        axSlider.objectName = "cutSlider_I"
        axSlider.valueChanged.connect(self.updateCutoff)

        glNormCB = QtGui.QCheckBox()
        glNormCB.objectName = "gNormChb_" + str(iaxis)
        glNormCB.setCheckState(2)
        glNormCB.stateChanged.connect(self.checkGNorm)
        glNormLabel = QtGui.QLabel()
        glNormLabel.setText('Global Normalization')

        colorLayout.addWidget(axLabel, 2+3, 0)
        colorLayout.addWidget(axEdit, 2+3, 1)
        colorLayout.addWidget(axSlider, 3+3, 0, 1, 2)
        colorLayout.addWidget(glNormCB, 4+3, 0, 1, 1)
        colorLayout.addWidget(glNormLabel, 4+3, 1, 1, 1)
        self.colorPanel.setLayout(colorLayout)

#  Projection panel
        self.projectionPanel = QtGui.QGroupBox(self)
        self.projectionPanel.setFlat(False)
#        self.projectionPanel.setTitle("Line properties")
        projectionLayout = QtGui.QGridLayout()
        self.projVisPanel = QtGui.QGroupBox(self)
        self.projVisPanel.setFlat(False)
        self.projVisPanel.setTitle("Projection visibility")
        projVisLayout = QtGui.QGridLayout()
        self.projLinePanel = QtGui.QGroupBox(self)
        self.projLinePanel.setFlat(False)
        self.projLinePanel.setTitle("Line properties")
        projLineLayout = QtGui.QGridLayout()

        for iaxis, axis in enumerate(['Show Side (YZ)', 'Show Front (XZ)',
                                      'Show Top (XY)']):
            checkBox = QtGui.QCheckBox()
            checkBox.objectName = "visChb_" + str(iaxis)
            checkBox.setCheckState(0)
            checkBox.stateChanged.connect(self.projSelection)
            visLabel = QtGui.QLabel()
            visLabel.setText(axis)
            projVisLayout.addWidget(checkBox, iaxis*2, 0, 1, 1)
            projVisLayout.addWidget(visLabel, iaxis*2, 1, 1, 1)

        checkBox = QtGui.QCheckBox()
        checkBox.objectName = "visChb_3"
        checkBox.setCheckState(2)
        checkBox.stateChanged.connect(self.checkDrawGrid)
        visLabel = QtGui.QLabel()
        visLabel.setText('Coordinate grid')
        projVisLayout.addWidget(checkBox, 3*2, 0, 1, 1)
        projVisLayout.addWidget(visLabel, 3*2, 1, 1, 1)

        self.projVisPanel.setLayout(projVisLayout)

        for iaxis, axis in enumerate(
                ['Line opacity', 'Line width', 'Point opacity', 'Point size']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "projectionLabel_" + str(iaxis)
            projectionValidator = QtGui.QDoubleValidator()
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)

            if iaxis in [0, 2]:
                axSlider.setRange(0, 1., 0.001)
                axSlider.setValue(0.1)
                axEdit = QtGui.QLineEdit("0.1")
                projectionValidator.setRange(0, 1., 5)

            else:
                axSlider.setRange(0, 20, 0.01)
                axSlider.setValue(1.)
                axEdit = QtGui.QLineEdit("1")
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

        self.scenePanel = QtGui.QGroupBox(self)
        self.scenePanel.setFlat(False)
        self.scenePanel.setTitle("Scale coordinate grid")
        sceneLayout = QtGui.QGridLayout()
        sceneValidator = QtGui.QDoubleValidator()
        sceneValidator.setRange(0, 10, 3)
        for iaxis, axis in enumerate(['x', 'y', 'z']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "sceneLabel_" + axis
            axEdit = QtGui.QLineEdit("0.9")
            axEdit.setValidator(scaleValidator)
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
            axSlider.setRange(0, 10, 0.01)
            axSlider.setValue(0.9)
            axEdit.editingFinished.connect(self.updateSceneFromQLE)
            axSlider.objectName = "sceneSlider_" + axis
            axSlider.valueChanged.connect(self.updateScene)
            sceneLayout.addWidget(axLabel, iaxis*2, 0)
            sceneLayout.addWidget(axEdit, iaxis*2, 1)
            sceneLayout.addWidget(axSlider, iaxis*2+1, 0, 1, 2)

        for (iCB, cbText), cbFunc in zip(enumerate(['Enable antialiasing',
                                                    'Enable blending',
                                                    'Depth test for Lines',
                                                    'Depth test for Points',
                                                    'Invert scene color']),
                                         [self.checkAA,
                                          self.checkBlending,
                                          self.checkLineDepthTest,
                                          self.checkPointDepthTest,
                                          self.invertSceneColor]):
            aaCheckBox = QtGui.QCheckBox()
            aaCheckBox.objectName = "aaChb" + str(iCB)
            aaCheckBox.setCheckState(2) if iCB in [1, 2] else\
                aaCheckBox.setCheckState(0)
            aaCheckBox.stateChanged.connect(cbFunc)
            aaLabel = QtGui.QLabel()
            aaLabel.setText(cbText)
            sceneLayout.addWidget(aaCheckBox, 6+iCB, 0, 1, 1)
            sceneLayout.addWidget(aaLabel, 6+iCB, 1, 1, 1)

        oeTileValidator = QtGui.QIntValidator()
        sceneValidator.setRange(1, 20)
        for iaxis, axis in enumerate(['local x', 'local y']):
            axLabel = QtGui.QLabel()
            axLabel.setText(axis)
            axLabel.objectName = "oeTileLabel_" + axis
            axEdit = QtGui.QLineEdit("2")
            axEdit.setValidator(oeTileValidator)
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
            axSlider.setRange(1, 20, 1)
            axSlider.setValue(2)
            axEdit.editingFinished.connect(self.updateTileFromQLE)
            axSlider.objectName = "oeTileSlider_" + axis
            axSlider.valueChanged.connect(self.updateTile)
            sceneLayout.addWidget(axLabel, 11+iaxis*2, 0)
            sceneLayout.addWidget(axEdit, 11+iaxis*2, 1)
            sceneLayout.addWidget(axSlider, 11+iaxis*2+1, 0, 1, 2)

        self.scenePanel.setLayout(sceneLayout)

#  Navigation panel
        self.navigationPanel = QtGui.QGroupBox(self)
        self.navigationPanel.setFlat(False)
        self.navigationPanel.setTitle("Navigation")
        self.navigationLayout = QtGui.QVBoxLayout()
        self.oeTree = QtGui.QTreeView()
        self.oeTree.setModel(self.segmentsModel)
        self.oeTree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.oeTree.customContextMenuRequested.connect(self.oeTreeMenu)
        self.navigationLayout.addWidget(self.oeTree)
        self.navigationPanel.setLayout(self.navigationLayout)

        mainLayout = QtGui.QHBoxLayout()
        sideLayout = QtGui.QVBoxLayout()

        tabs = QtGui.QTabWidget()
#        tabs = myTabWidget()
        tabs.addTab(self.zoomPanel, "Scaling")
        tabs.addTab(self.rotationPanel, "Rotation")
        tabs.addTab(self.opacityPanel, "Opacity")
        tabs.addTab(self.colorPanel, "Color")
        tabs.addTab(self.projectionPanel, "Projections")
        tabs.addTab(self.scenePanel, "Scene")
        tabs.addTab(self.navigationPanel, "Navigation")
        sideLayout.addWidget(tabs)
        canvasSplitter = QtGui.QSplitter()
        canvasSplitter.setChildrenCollapsible(False)
        canvasSplitter.setOrientation(QtCore.Qt.Horizontal)
        mainLayout.addWidget(canvasSplitter)
        sideWidget = QtGui.QWidget()
        sideWidget.setLayout(sideLayout)
        canvasSplitter.addWidget(self.customGlWidget)
        canvasSplitter.addWidget(sideWidget)

        self.setLayout(mainLayout)
        self.customGlWidget.oesList = self.oesList
        fastSave = QtGui.QShortcut(self)
        fastSave.setKey(QtCore.Qt.Key_F5)
        fastSave.activated.connect(partial(self.saveScene, '_xrtScnTmp_.npy'))
        fastLoad = QtGui.QShortcut(self)
        fastLoad.setKey(QtCore.Qt.Key_F6)
        fastLoad.activated.connect(partial(self.loadScene, '_xrtScnTmp_.npy'))

    def drawColorMap(self, axis):
        xv, yv = np.meshgrid(np.linspace(0, 1, 200),
                             np.linspace(0, 1, 200))
        xv = xv.flatten()
        yv = yv.flatten()
        self.im = self.mplAx.imshow(mpl.colors.hsv_to_rgb(np.vstack((
            xv, np.ones_like(xv)*0.85, yv)).T).reshape((200, 200, 3)),
            aspect='auto', origin='lower',
            extent=(self.customGlWidget.colorMin, self.customGlWidget.colorMax,
                    0, 1))
        self.mplAx.set_xlabel(axis)
        self.mplAx.set_ylabel('Intensity')

    def checkGNorm(self, state):
        self.customGlWidget.globalNorm = True if state > 0 else False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def checkDrawGrid(self, state):
        self.customGlWidget.drawGrid = True if state > 0 else False
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

    def changeColorAxis(self, selAxis):
        self.customGlWidget.getColor = getattr(
            raycing, 'get_{}'.format(selAxis))
        self.customGlWidget.newColorAxis = True
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.selColorMin = self.customGlWidget.colorMin
        self.customGlWidget.selColorMin = self.customGlWidget.colorMax
        self.mplAx.set_xlabel(selAxis)
        extents = (self.customGlWidget.colorMin,
                   self.customGlWidget.colorMax, 0, 1)
        self.im.set_extent(extents)
        extents = list(extents)
        self.colorPanel.layout().itemAt(4).widget().setText(str(extents[0]))
        self.colorPanel.layout().itemAt(6).widget().validator().setBottom(
            extents[0])
        self.colorPanel.layout().itemAt(6).widget().setText(str(extents[1]))
        self.colorPanel.layout().itemAt(4).widget().validator().setTop(
            extents[1])
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

    def updateColorSelFromMPL(self, eclick, erelease):
        try:
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.selColorMin = np.min([extents[0], extents[1]])
            self.customGlWidget.selColorMax = np.max([extents[0], extents[1]])
            self.colorPanel.layout().itemAt(4).widget().setText(str(
                extents[0]))
            self.colorPanel.layout().itemAt(6).widget().validator().setBottom(
                extents[0])
            self.colorPanel.layout().itemAt(6).widget().setText(str(
                extents[1]))
            self.colorPanel.layout().itemAt(4).widget().validator().setTop(
                extents[1])
            slider = self.colorPanel.layout().itemAt(7).widget()
            center = 0.5 * (extents[0] + extents[1])
            halfWidth = (extents[1] - extents[0]) * 0.5
            slider.setValue(center)
            newMin = self.customGlWidget.colorMin + halfWidth
            newMax = self.customGlWidget.colorMax - halfWidth
            newRange = (newMax - newMin) * 0.01
            slider.setRange(newMin, newMax, newRange)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            self.customGlWidget.glDraw()
        except:
            pass

    def updateColorSel(self, position):
        try:
            extents = list(self.paletteWidget.span.extents)
            width = extents[1] - extents[0]
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
                center = (value + float(str(self.colorPanel.layout().itemAt(
                    6).widget().text()))) * 0.5
                newMin = -center + value + self.customGlWidget.colorMin
                newRange = (slider.maxValue() - newMin) * 0.01
                slider.setRange(newMin, slider.maxValue(), newRange)
            else:
                self.customGlWidget.selColorMax = value
                newExtents = (extents[0], value,
                              extents[2], extents[3])
                self.colorPanel.layout().itemAt(4).widget().validator().setTop(
                    value)
                center = (value + float(str(self.colorPanel.layout().itemAt(
                    4).widget().text()))) * 0.5
                newMax = center - value + self.customGlWidget.colorMax
                newRange = (newMax - slider.minValue()) * 0.01
                slider.setRange(slider.minValue(), newMax, newRange)
            self.colorPanel.layout().itemAt(7).widget().setValue(center)
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
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        if cPan.objectName[-1] == 'x':
            self.customGlWidget.rotations[0][0] = np.float32(position)
        elif cPan.objectName[-1] == 'y':
            self.customGlWidget.rotations[1][0] = np.float32(position)
        elif cPan.objectName[-1] == 'z':
            self.customGlWidget.rotations[2][0] = np.float32(position)
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

    def updateRaysList(self, item):
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
            menu = QtGui.QMenu()
            menu.addAction('Center here',
                           partial(self.centerEl, str(selectedItem.text())))
        else:
            pass

        menu.exec_(self.oeTree.viewport().mapToGlobal(position))

    def updateScene(self, position):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        aIndex = int(((cIndex + 1) / 3) - 1)
        self.customGlWidget.aPos[aIndex] = np.float32(position)
        self.customGlWidget.glDraw()

    def updateSceneFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)

    def glMenu(self, position):
        menu = QtGui.QMenu()
        for actText, actFunc in zip(['Export to image', 'Save scene geometry',
                                     'Load scene geometry'],
                                    [self.exportToImage, self.saveSceneDialog,
                                     self.loadSceneDialog]):
            mAction = QtGui.QAction(self)
            mAction.setText(actText)
            mAction.triggered.connect(actFunc)
            menu.addAction(mAction)
        menu.exec_(self.customGlWidget.mapToGlobal(position))

    def exportToImage(self):
        saveDialog = QtGui.QFileDialog()
        saveDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("BMP files (*.bmp);;JPG files (*.jpg);;JPG files (*.jpg);;JPEG files (*.jpeg);;PNG files (*.png);;TIFF files (*.tif)")  # analysis:ignore
        saveDialog.selectNameFilter("PNG files (*.png)")
        if (saveDialog.exec_()):
            image = self.customGlWidget.grabFrameBuffer(withAlpha=True)
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
#            print filename
            image.save(filename)

    def saveSceneDialog(self):
        saveDialog = QtGui.QFileDialog()
        saveDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        saveDialog.setNameFilter("Numpy files (*.npy)")  # analysis:ignore
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = 'npy'
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            self.saveScene(filename)

    def loadSceneDialog(self):
        loadDialog = QtGui.QFileDialog()
        loadDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        loadDialog.setAcceptMode(QtGui.QFileDialog.AcceptOpen)
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
                      'colorMin', 'colorMax']:
            params[param] = getattr(self.customGlWidget, param)
        params['size'] = self.geometry()
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
                      'colorMin', 'colorMax']:
            setattr(self.customGlWidget, param, params[param])
        self.setGeometry(params['size'])

        for axis in range(3):
            self.zoomPanel.layout().itemAt((axis+1)*3-1).widget().setValue(
                np.log10(self.customGlWidget.scaleVec[axis]))

        self.rotationPanel.layout().itemAt(2).widget().setValue(
            self.customGlWidget.rotations[0][0])
        self.rotationPanel.layout().itemAt(5).widget().setValue(
            self.customGlWidget.rotations[1][0])
        self.rotationPanel.layout().itemAt(8).widget().setValue(
            self.customGlWidget.rotations[2][0])

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

        self.customGlWidget.newColorAxis = False
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()
        print('Loaded scene from {}'.format(filename))

    def centerEl(self, oeName):
        self.customGlWidget.coordOffset = list(self.oesList[oeName][2])
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def updateCutoff(self, position):
        try:
            cPan = self.sender()
            cIndex = cPan.parent().layout().indexOf(cPan)
            cPan.parent().layout().itemAt(cIndex-1).widget().setText(
                str(position))
            extents = list(self.paletteWidget.span.extents)
            self.customGlWidget.cutoffI = np.float32(position)
            self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
            newExtents = (extents[0], extents[1],
                          self.customGlWidget.cutoffI, extents[3])
            self.paletteWidget.span.extents = newExtents
            self.customGlWidget.glDraw()
        except:
            pass

    def updateCutoffFromQLE(self):
        try:
            cPan = self.sender()
            cIndex = cPan.parent().layout().indexOf(cPan)
            value = float(str(cPan.text()))
            cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
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

    def updateTile(self, position):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        objNameType = cPan.objectName[-1]
        if objNameType == 'x':
            self.customGlWidget.tiles[0] = np.int(position)
        elif objNameType == 'y':
            self.customGlWidget.tiles[1] = np.int(position)
        self.customGlWidget.glDraw()

    def updateTileFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = int(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
        self.customGlWidget.glDraw()

    def updateProjectionOpacity(self, position):
        cPan = self.sender()
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

    def __init__(self, parent, arrayOfRays, modelRoot, oesList, b2els):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)
        self.aspect = 1.
        self.cameraAngle = 60
        self.setMouseTracking(True)
        self.surfCPOrder = 4
        self.oesToPlot = []
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
        self.newColorAxis = True
        self.scaleVec = np.array([1e3, 1e1, 1e3])
        self.populateVerticesArray(modelRoot)

        maxC = np.max(self.verticesArray, axis=0)
        minC = np.min(self.verticesArray, axis=0)
        self.maxLen = np.max(maxC - minC)

        self.drawGrid = True
        self.aPos = [0.9, 0.9, 0.9]
        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])

        self.tVec = np.array([0., 0., 0.])
        self.cameraTarget = [0., 0., 0.]
        self.cameraPos = np.float32([3.5, 0., 0.])
        self.rotations = np.float32([[0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
        pModelT = np.identity(4)
        self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
        self.signs = np.ones_like(pModelT)
        self.invertColors = False
        self.glDraw()

    def rotateZYX(self):
#        hRoll = np.radians(self.rotations[0][0]) * 0.5
#        hPitch = np.radians(self.rotations[1][0]) * 0.5
#        hYaw = np.radians(self.rotations[2][0]) * 0.5
#        t0 = np.cos(hYaw)
#        t1 = np.sin(hYaw)
#        t2 = np.cos(hRoll)
#        t3 = np.sin(hRoll)
#        t4 = np.cos(hPitch)
#        t5 = np.sin(hPitch)
#
#        qForward = [t0 * t2 * t4 + t1 * t3 * t5,
#                    (t0 * t3 * t4 - t1 * t2 * t5),
#                    (t0 * t2 * t5 + t1 * t3 * t4),
#                    (t1 * t2 * t4 - t0 * t3 * t5)]
#
#        angle = 2 * np.arccos(qForward[0])
#        q2v = np.sin(angle * 0.5)
#        qbt1 = qForward[1] / q2v if q2v != 0\
#            else 0
#        qbt2 = qForward[2] / q2v if q2v != 0\
#            else 0
#        qbt3 = qForward[3] / q2v if q2v != 0\
#            else 0
#        glRotatef(np.degrees(angle), qbt1, qbt2, qbt3)
        glRotate(*self.rotations[0])
        glRotate(*self.rotations[1])
        glRotate(*self.rotations[2])

    def setPointSize(self, pSize):
        self.pointSize = pSize
        self.glDraw()

    def setLineWidth(self, lWidth):
        self.lineWidth = lWidth
        self.glDraw()

    def populateVerticesArray(self, segmentsModelRoot):
        self.verticesArray = None
        self.footprintsArray = None
        self.oesToPlot = []
        self.footprints = dict()
        colorsRays = None
        alphaRays = None
        colorsDots = None
        alphaDots = None
        if self.newColorAxis:
            self.colorMax = -1e20
            self.colorMin = 1e20
        for ioe in range(segmentsModelRoot.rowCount() - 1):
            ioeItem = segmentsModelRoot.child(ioe + 1, 0)
            if segmentsModelRoot.child(ioe + 1, 2).checkState() == 2:
                self.oesToPlot.append(str(ioeItem.text()))
                self.footprints[str(ioeItem.text())] = None
            try:
                startBeam = self.beamsDict[
                    self.oesList[str(ioeItem.text())][1]]
                good = startBeam.state > 0

                self.colorMax = max(np.max(
                    self.getColor(startBeam)[good]),
                    self.colorMax)
                self.colorMin = min(np.min(
                    self.getColor(startBeam)[good]),
                    self.colorMin)
                if self.newColorAxis:
                    self.selColorMin = self.colorMin
                    self.selColorMax = self.colorMax
            except:
                continue

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

        try:
            colorsRays = (colorsRays-self.colorMin) / (self.colorMax -
                                                       self.colorMin)
            colorsRays = np.dstack((colorsRays,
                                    np.ones_like(alphaRays)*0.85,
                                    alphaRays))
            colorsRGBRays = np.squeeze(mpl.colors.hsv_to_rgb(colorsRays))
            if self.globalNorm:
                alphaMax = np.max(alphaRays)
            else:
                alphaMax = 1.
            alphaColorRays = np.array([alphaRays / alphaMax]).T *\
                self.lineOpacity
            self.raysColor = np.float32(np.hstack([colorsRGBRays,
                                                   alphaColorRays]))

            colorsDots = (colorsDots-self.colorMin) / (self.colorMax -
                                                       self.colorMin)
            colorsDots = np.dstack((colorsDots,
                                    np.ones_like(alphaDots)*0.85,
                                    alphaDots))

            colorsRGBDots = np.squeeze(mpl.colors.hsv_to_rgb(colorsDots))
            if self.globalNorm:
                alphaMax = np.max(alphaDots)
            else:
                alphaMax = 1.
            alphaColorDots = np.array([alphaDots / alphaMax]).T *\
                self.pointOpacity
            self.dotsColor = np.float32(np.hstack([colorsRGBDots,
                                                   alphaColorDots]))
            self.newColorAxis = False
        except:
            pass

    def modelToWorld(self, coords, dimension=None):
        if dimension is None:
            return np.float32(((coords + self.tVec) * self.scaleVec) /
                              self.maxLen)
        else:
            return np.float32(((coords[dimension] + self.tVec[dimension]) *
                              self.scaleVec[dimension]) / self.maxLen)

    def paintGL(self):
        def quatMult(qf, qt):
            return [qf[0]*qt[0]-qf[1]*qt[1]-qf[2]*qt[2]-qf[3]*qt[3],
                    qf[0]*qt[1]+qf[1]*qt[0]+qf[2]*qt[3]-qf[3]*qt[2],
                    qf[0]*qt[2]-qf[1]*qt[3]+qf[2]*qt[0]+qf[3]*qt[1],
                    qf[0]*qt[3]+qf[1]*qt[2]-qf[2]*qt[1]+qf[3]*qt[0]]

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

        if self.invertColors:
            glClearColor(1.0, 1.0, 1.0, 1.)
        else:
            glClearColor(0.0, 0.0, 0.0, 1.)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.cameraAngle, self.aspect, 0.001, 1000)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gluLookAt(self.cameraPos[0], self.cameraPos[1], self.cameraPos[2],
                  self.cameraTarget[0], self.cameraTarget[1],
                  self.cameraTarget[2],
                  0.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)

        if self.enableAA:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        if self.enableBlending:
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)

        glEnableClientState(GL_VERTEX_ARRAY)

        glEnableClientState(GL_COLOR_ARRAY)
# Coordinate box
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glLoadIdentity()
        self.rotateZYX()
        axPosModifier = np.ones(3)

        for dim in range(3):
            for iAx in range(3):
                axPosModifier[iAx] = (self.signs[0][iAx] if
                                      self.signs[0][iAx] != 0 else 1)
            if self.projectionsVisibility[dim] > 0:
                if self.lineProjectionWidth > 0 and\
                        self.lineProjectionOpacity > 0 and\
                        self.verticesArray is not None:
                    projectionRays = self.modelToWorld(
                        np.copy(self.verticesArray))
                    projectionRays[:, dim] =\
                        -self.aPos[dim] * axPosModifier[dim]
                    self.drawArrays(
                        0, GL_LINES, projectionRays, self.raysColor,
                        self.lineProjectionOpacity, self.lineProjectionWidth)

                if self.pointProjectionSize > 0 and\
                        self.pointProjectionOpacity > 0 and\
                        self.footprintsArray is not None:
                    projectionDots = self.modelToWorld(
                        np.copy(self.footprintsArray))
                    projectionDots[:, dim] =\
                        -self.aPos[dim] * axPosModifier[dim]
                    self.drawArrays(
                        0, GL_POINTS, projectionDots, self.dotsColor,
                        self.pointProjectionOpacity, self.pointProjectionSize)

        glEnable(GL_DEPTH_TEST)

        if self.drawGrid:

            glLoadIdentity()
            self.rotateZYX()

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

#  Calculating regular grids in world coordinates
            limits = np.array([-1, 1])[:, np.newaxis] * np.array(self.aPos)
            allLimits = limits * self.maxLen / self.scaleVec - self.tVec\
                + self.coordOffset
            axGrids = []
            gridLabels = []
            precisionLabels = []

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
                axGrids.extend([gridX - self.coordOffset[iAx]])

            back[:, 0] *= axPosModifier[0]
            side[:, 1] *= axPosModifier[1]
            bottom[:, 2] *= axPosModifier[2]

            axisL = []

            axisL.extend([np.vstack(
                (self.modelToWorld(axGrids, 0),
                 np.ones(len(axGrids[0]))*self.aPos[1]*axPosModifier[1],
                 np.ones(len(axGrids[0]))*-self.aPos[2]*axPosModifier[2]))])
            axisL.extend([np.vstack(
                (np.ones(len(axGrids[1]))*self.aPos[0]*axPosModifier[0],
                 self.modelToWorld(axGrids, 1),
                 np.ones(len(axGrids[1]))*-self.aPos[2]*axPosModifier[2]))])
            zAxis = np.vstack(
                (np.ones(len(axGrids[2]))*-self.aPos[0]*axPosModifier[0],
                 np.ones(len(axGrids[2]))*self.aPos[1]*axPosModifier[1],
                 self.modelToWorld(axGrids, 2)))

            xAxisB = np.vstack(
                (self.modelToWorld(axGrids, 0),
                 np.ones(len(axGrids[0]))*-self.aPos[1]*axPosModifier[1],
                 np.ones(len(axGrids[0]))*-self.aPos[2]*axPosModifier[2]))
            yAxisB = np.vstack(
                (np.ones(len(axGrids[1]))*-self.aPos[0]*axPosModifier[0],
                 self.modelToWorld(axGrids, 1),
                 np.ones(len(axGrids[1]))*-self.aPos[2]*axPosModifier[2]))
            zAxisB = np.vstack(
                (np.ones(len(axGrids[2]))*-self.aPos[0]*axPosModifier[0],
                 np.ones(len(axGrids[2]))*-self.aPos[1]*axPosModifier[1],
                 self.modelToWorld(axGrids, 2)))

            xAxisC = np.vstack(
                (self.modelToWorld(axGrids, 0),
                 np.ones(len(axGrids[0]))*-self.aPos[1]*axPosModifier[1],
                 np.ones(len(axGrids[0]))*self.aPos[2]*axPosModifier[2]))
            yAxisC = np.vstack(
                (np.ones(len(axGrids[1]))*-self.aPos[0]*axPosModifier[0],
                 self.modelToWorld(axGrids, 1),
                 np.ones(len(axGrids[1]))*self.aPos[2]*axPosModifier[2]))
            axisL.extend([np.vstack(
                (np.ones(len(axGrids[2]))*self.aPos[0]*axPosModifier[0],
                 np.ones(len(axGrids[2]))*-self.aPos[1]*axPosModifier[1],
                 self.modelToWorld(axGrids, 2)))])

            xLines = np.vstack(
                (axisL[0], xAxisB, xAxisB, xAxisC)).T.flatten().reshape(
                4*xAxisB.shape[1], 3)
            yLines = np.vstack(
                (axisL[1], yAxisB, yAxisB, yAxisC)).T.flatten().reshape(
                4*yAxisB.shape[1], 3)
            zLines = np.vstack(
                (zAxis, zAxisB, zAxisB, axisL[2])).T.flatten().reshape(
                4*zAxisB.shape[1], 3)

#            axTicks = np.vstack((xAxis.T, yAxis.T, zAxisC.T))
            axGrid = np.vstack((xLines, yLines, zLines))

            if self.invertColors:
                glColor4f(0.0, 0.0, 0.0, 1.)
            else:
                glColor4f(1.0, 1.0, 1.0, 1.)
#            for tick, tText, pcs in zip(axTicks, gridLabels, precisionLabels):
#                glRasterPos3f(*tick)
#                for symbol in "   {0:.{1}f}".format(tText, int(pcs)):
#                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))
#            if not self.enableAA:
#                glEnable(GL_LINE_SMOOTH)
#                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
#                glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
            for iAx in range(3):
                for tick, tText, pcs in zip(axisL[iAx].T, gridLabels[iAx],
                                            precisionLabels[iAx]):
                    glRasterPos3f(*tick)
                    for symbol in "   {0:.{1}f}".format(tText, int(pcs)):
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 
                                            ord(symbol))

#                    glPushMatrix()
#                    glTranslatef(*tick)
#                    hRoll = np.radians(self.rotations[0][0]) * 0.5
#                    hPitch = np.radians(self.rotations[1][0]) * 0.5
#                    hYaw = np.radians(self.rotations[2][0]) * 0.5
#                    t0 = np.cos(hYaw)
#                    t1 = np.sin(hYaw)
#                    t2 = np.cos(hRoll)
#                    t3 = np.sin(hRoll)
#                    t4 = np.cos(hPitch)
#                    t5 = np.sin(hPitch)
#                    qBack = [t0 * t2 * t4 + t1 * t3 * t5,
#                             -(t0 * t3 * t4 - t1 * t2 * t5),
#                             -(t0 * t2 * t5 + t1 * t3 * t4),
#                             -(t1 * t2 * t4 - t0 * t3 * t5)]
#
#                    qText = [0.5, 0.5, 0.5, 0.5]
#
#                    qb = quatMult(qBack, qText)
#                    angle = 2 * np.arccos(qb[0])
#                    q2v = np.sin(angle * 0.5)
#                    qbt1 = qb[1] / q2v if q2v != 0\
#                        else 0
#                    qbt2 = qb[2] / q2v if q2v != 0\
#                        else 0
#                    qbt3 = qb[3] / q2v if q2v != 0\
#                        else 0
#                    glRotatef(np.degrees(angle), qbt1, qbt2, qbt3)
#
#                    glScalef(1./2500., 1./2500., 1./2500.)
#
#                    for symbol in " {0:.{1}f}".format(tText, int(pcs)):
#                        glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(symbol))
#                    glPopMatrix()
#            if not self.enableAA:
#                glDisable(GL_LINE_SMOOTH)

            gridColor = np.ones((len(axGrid), 4)) * 0.25
            gridArray = vbo.VBO(np.float32(axGrid))
            gridArray.bind()
            glVertexPointerf(gridArray)
            gridColorArray = vbo.VBO(np.float32(gridColor))
            gridColorArray.bind()
            glColorPointerf(gridColorArray)
            glLineWidth(1.)
            glDrawArrays(GL_LINES, 0, len(gridArray))
            gridArray.unbind()
            gridColorArray.unbind()

            grid = np.vstack((back, side, bottom))

            gridColor = np.ones((len(grid), 4)) * 0.5
            gridArray = vbo.VBO(np.float32(grid))
            gridArray.bind()
            glVertexPointerf(gridArray)
            gridColorArray = vbo.VBO(np.float32(gridColor))
            gridColorArray.bind()
            glColorPointerf(gridColorArray)
            glLineWidth(2.)
            glDrawArrays(GL_QUADS, 0, len(gridArray))
            gridArray.unbind()
            gridColorArray.unbind()

        glLoadIdentity()
        self.rotateZYX()
        if len(self.oesToPlot) > 0:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnableClientState(GL_NORMAL_ARRAY)
            glEnable(GL_NORMALIZE)
            glShadeModel(GL_SMOOTH)

            self.addLighting(3.)
            for oeString in self.oesToPlot:
                oeToPlot = self.oesList[oeString][0]
                is2ndXtal = self.oesList[oeString][3]
                elType = str(type(oeToPlot))
                if len(re.findall('raycing.oe', elType.lower())) > 0:  # OE
                    setMaterial('Si')

                    self.plotOeSurface(oeToPlot, is2ndXtal)
                elif len(re.findall('raycing.apert', elType)) > 0:  # aperture
                    setMaterial('Cu')
                    self.plotAperture(oeToPlot)
                elif len(re.findall('raycing.screen', elType)) > 0:  # screen
                    continue
                else:
                    continue

            glDisable(GL_LIGHTING)
            glDisable(GL_NORMALIZE)
            glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.linesDepthTest:
            glEnable(GL_DEPTH_TEST)

        if self.lineWidth > 0 and self.lineOpacity > 0 and\
                self.verticesArray is not None:
            self.drawArrays(1, GL_LINES, self.verticesArray, self.raysColor,
                            self.lineOpacity, self.lineWidth)
        if self.linesDepthTest:
            glDisable(GL_DEPTH_TEST)

        if self.pointsDepthTest:
            glEnable(GL_DEPTH_TEST)

        if self.pointSize > 0 and self.pointOpacity > 0 and\
                self.footprintsArray is not None:
            self.drawArrays(1, GL_POINTS, self.footprintsArray, self.dotsColor,
                            self.pointOpacity, self.pointSize)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        if self.pointsDepthTest:
            glDisable(GL_DEPTH_TEST)

        glFlush()

        self.drawAxes()

        if self.enableAA:
            glDisable(GL_LINE_SMOOTH)

        if self.enableBlending:
            glDisable(GL_MULTISAMPLE)
            glDisable(GL_BLEND)
            glDisable(GL_POINT_SMOOTH)

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
            limits = zip([0], [w], [0], [h])
        else:
            limits = zip([0, w], [w+wf, w+wf], [h, 0], [h+wf, h])
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

                        glEvalMesh2(GL_FILL, 0, surfCPOrder*4,
                                    0, surfCPOrder*4)
        glDisable(GL_MAP2_VERTEX_3)
        glDisable(GL_MAP2_NORMAL)

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

    def drawAxes(self):
        arrowSize = 0.05
        axisLen = 0.1
        glLineWidth(1.)

        def drawCone(z, r, nFacets, color):
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
        pView = glGetIntegerv(GL_VIEWPORT)
        glViewport(0, 0, int(150*self.aspect), 150)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.aspect, 0.001, 10)

        gluLookAt(.5, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glLoadIdentity()
        self.rotateZYX()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPushMatrix()
        glTranslatef(0, 0, axisLen)
        drawCone(arrowSize, 0.02, 20, 0)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0, axisLen, 0)
        glRotatef(-90, 1.0, 0.0, 0.0)
        drawCone(arrowSize, 0.02, 20, 1)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(axisLen, 0, 0)
        glRotatef(90, 0.0, 1.0, 0.0)
        drawCone(arrowSize, 0.02, 20, 2)
        glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBegin(GL_LINES)
        glColor4f(1, 0, 0, 0.75)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axisLen)
        glColor4f(0, 1, 0, 0.75)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axisLen, 0)
        glColor4f(0, 0, 1, 0.75)
        glVertex3f(0, 0, 0)
        glVertex3f(axisLen, 0, 0)
        glEnd()

        glColor4f(1, 0, 0, 1)
        glRasterPos3f(0, 0, axisLen*1.5)
        for symbol in "  {}, mm".format('Z'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))

        glColor4f(0, 0.75, 0, 1)
        glRasterPos3f(0, axisLen*1.5, 0)
        for symbol in "  {}, mm".format('Y'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))

        glColor4f(0, 0.5, 1, 1)
        glRasterPos3f(axisLen*1.5, 0, 0)
        for symbol in "  {}, mm".format('X'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))
        glFlush()
        glViewport(*pView)
        glColor4f(1, 1, 1, 1)

    def initializeGL(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glViewport(0, 0, 900, 900)

    def resizeGL(self, widthInPixels, heightInPixels):
        glViewport(0, 0, widthInPixels, heightInPixels)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

    def mouseMoveEvent(self, mouseEvent):
        if mouseEvent.buttons() == QtCore.Qt.LeftButton:
            glLoadIdentity()
            self.rotateZYX()
            pModelT = np.array(glGetDoublev(GL_TRANSPOSE_MODELVIEW_MATRIX))
            self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
            self.signs = np.sign(pModelT)

            if mouseEvent.modifiers() == QtCore.Qt.NoModifier:
                self.rotations[1][0] += np.float32(
                    (mouseEvent.y() - self.prevMPos[1]) * 36. / 90.)
                self.rotations[2][0] += np.float32(
                    (mouseEvent.x() - self.prevMPos[0]) * 36. / 90.)
                for ax in range(2):
                    if self.rotations[self.visibleAxes[ax+1]][0] > 180:
                        self.rotations[self.visibleAxes[ax+1]][0] -= 360
                    if self.rotations[self.visibleAxes[ax+1]][0] < -180:
                        self.rotations[self.visibleAxes[ax+1]][0] += 360
                self.rotationUpdated.emit(self.rotations)
            elif mouseEvent.modifiers() == QtCore.Qt.ShiftModifier:
                pProjectionT = glGetDoublev(GL_TRANSPOSE_PROJECTION_MATRIX)
                pView = glGetIntegerv(GL_VIEWPORT)
                pScale = np.float32(pProjectionT[2][3]*1.25)
                self.tVec[self.visibleAxes[1]] +=\
                    self.signs[1][self.visibleAxes[1]] * pScale *\
                    (mouseEvent.x() - self.prevMPos[0]) / pView[2] /\
                    self.scaleVec[self.visibleAxes[1]] * self.maxLen
                self.tVec[self.visibleAxes[2]] -=\
                    self.signs[2][self.visibleAxes[2]] * pScale *\
                    (mouseEvent.y() - self.prevMPos[1]) / pView[3] /\
                    self.scaleVec[self.visibleAxes[2]] * self.maxLen
#                self.verticesArray[:, self.visibleAxes[1]] += \
#                    self.signs[1][self.visibleAxes[1]] * pScale *\
#                    (mouseEvent.x() - self.prevMPos[0]) / pView[2] /\
#                    self.scaleVec[self.visibleAxes[1]] * self.maxLen
#                self.verticesArray[:, self.visibleAxes[2]] -= \
#                    self.signs[2][self.visibleAxes[2]] * pScale *\
#                    (mouseEvent.y() - self.prevMPos[1]) / pView[3] /\
#                    self.scaleVec[self.visibleAxes[2]] * self.maxLen

            elif mouseEvent.modifiers() == QtCore.Qt.AltModifier:
                pProjectionT = glGetDoublev(GL_TRANSPOSE_PROJECTION_MATRIX)
                pView = glGetIntegerv(GL_VIEWPORT)
                pScale = np.float32(pProjectionT[2][3]*1.25)
#                self.verticesArray[:, self.visibleAxes[0]] -= \
#                    self.signs[2][self.visibleAxes[0]] * pScale *\
#                    (mouseEvent.y() - self.prevMPos[1]) / pView[3] /\
#                    self.scaleVec[self.visibleAxes[0]] * self.maxLen
                self.tVec[self.visibleAxes[0]] +=\
                    self.signs[0][self.visibleAxes[0]] * pScale *\
                    (mouseEvent.y() - self.prevMPos[1]) / pView[3] /\
                    self.scaleVec[self.visibleAxes[0]] * self.maxLen

        self.glDraw()
        self.prevMPos[0] = mouseEvent.x()
        self.prevMPos[1] = mouseEvent.y()

    def wheelEvent(self, wEvent):
        ctrlOn = (wEvent.modifiers() == QtCore.Qt.ControlModifier)
        if wEvent.delta() > 0:
            if ctrlOn:
                self.cameraPos *= 0.9
            else:
                self.scaleVec *= 1.1
        else:
            if ctrlOn:
                self.cameraPos *= 1.1
            else:
                self.scaleVec *= 0.9
        if not ctrlOn:
            self.scaleUpdated.emit(self.scaleVec)
        self.glDraw()

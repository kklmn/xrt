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


class myTabWidget(QtGui.QWidget):
    def __init__(self):
        super(myTabWidget, self).__init__()
        layout = QtGui.QVBoxLayout()
        self.tabBars = [QtGui.QTabBar(), QtGui.QTabBar()]
        self.tabPanel = QtGui.QStackedWidget()
        for i in range(2):
            self.tabBars[i].objectName = 'myTabBar_{}'.format(i)
            self.tabBars[i].currentChanged.connect(self._showTab)
        layout.addWidget(self.tabBars[0])
        layout.addWidget(self.tabBars[1])
        layout.addWidget(self.tabPanel)
        self.setLayout(layout)
        self.tabList = [[], []]

    def addTab(self, widget, name, level):
        self.tabBars[level].addTab(name)
        tabIndex = self.tabPanel.addWidget(widget)
        self.tabList[level].append(tabIndex)
#        print name, level, tabIndex, self.tabList

    def _showTab(self, position):
        cTab = self.sender()
        tabIndex = int(cTab.objectName[-1])
        if position < len(self.tabList[tabIndex]):
            self.tabPanel.setCurrentIndex(self.tabList[tabIndex][position])


class xrtGlow(QtGui.QWidget):
    def __init__(self, arrayOfRays):
        super(xrtGlow, self).__init__()
        self.segmentsModel = QtGui.QStandardItemModel()
        self.segmentsModelRoot = self.segmentsModel.invisibleRootItem()
        self.segmentsModel.setHorizontalHeaderLabels(['Beam start',
                                                      'Beam end'])
        self.oesList = arrayOfRays[2]
        for segOE in self.oesList.keys():
            child = QtGui.QStandardItem(str(segOE))
            child.setEditable(False)
            child.setCheckable(True)
            child.setCheckState(0)
            self.segmentsModelRoot.appendRow([child])
            for segment in arrayOfRays[0]:
                if str(segment[0]) == str(segOE):
                    child1 = QtGui.QStandardItem(str(segment[1]))
                    child2 = QtGui.QStandardItem(str(segment[3]))
                    child1.setCheckable(True)
                    child1.setCheckState(2)
                    child1.setEditable(False)
                    child2.setCheckable(True)
                    child2.setCheckState(2)
                    child2.setEditable(False)
                    child.appendRow([child1, child2])

        self.fluxDataModel = QtGui.QStandardItemModel()
#        self.fluxDataModel.appendRow(QtGui.QStandardItem("auto"))
        for rfName, rfObj in inspect.getmembers(raycing):
            if rfName.startswith('get_') and\
                    rfName != "get_output":
                flItem = QtGui.QStandardItem(rfName.replace("get_", ''))
                self.fluxDataModel.appendRow(flItem)

        self.customGlWidget = xrtGlWidget(self, arrayOfRays,
                                          self.segmentsModelRoot)
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
            axEdit = QtGui.QLineEdit("0.")
            axEdit.setValidator(scaleValidator)
            axSlider = Qwt.QwtSlider(
                self, QtCore.Qt.Horizontal, Qwt.QwtSlider.TopScale)
            axSlider.setRange(-7, 7, 0.01)
            axSlider.setValue(0)
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
            self.mplAx, self.onselect, drawtype='box', useblit=True,
            rectprops=dict(alpha=0.4, facecolor='white'), button=1,
            interactive=True)

        colorLayout.addWidget(self.paletteWidget, 0, 0, 1, 2)

        colorCBLabel = QtGui.QLabel()
        colorCBLabel.setText('Color Axis:')

        colorCB = QtGui.QComboBox()
        colorCB.setModel(self.fluxDataModel)
        colorCB.setCurrentIndex(colorCB.findText('energy'))
        colorCB.currentIndexChanged['QString'].connect(self.changeColorAxis)
        colorLayout.addWidget(colorCBLabel, 1, 0)
        colorLayout.addWidget(colorCB, 1, 1)

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
#        globalNormCB = QtGui.QCheckBox()
        colorLayout.addWidget(axLabel, 2, 0)
        colorLayout.addWidget(axEdit, 2, 1)
        colorLayout.addWidget(axSlider, 3, 0, 1, 2)
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
                                                    'Enable blending']),
                                         [self.checkAA,
                                          self.checkBlending]):
            aaCheckBox = QtGui.QCheckBox()
            aaCheckBox.objectName = "aaChb" + str(iCB)
            aaCheckBox.setCheckState(2) if iCB > 0 else\
                aaCheckBox.setCheckState(0)
            aaCheckBox.stateChanged.connect(cbFunc)
            aaLabel = QtGui.QLabel()
            aaLabel.setText(cbText)
            sceneLayout.addWidget(aaCheckBox, 6+iCB, 0, 1, 1)
            sceneLayout.addWidget(aaLabel, 6+iCB, 1, 1, 1)

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
#        sideSplitter = QtGui.QSplitter()
#        sideSplitter.setChildrenCollapsible(False)
#        sideSplitter.setOrientation(QtCore.Qt.Vertical)
        sideWidget = QtGui.QWidget()
#        self.setMinimumSize(750, 500)
        sideWidget.setLayout(sideLayout)
#        sideSplitter.addWidget(tabs)
#        sideSplitter.addWidget(self.navigationPanel)
#        sideLayout.addWidget(sideSplitter)
        canvasSplitter.addWidget(self.customGlWidget)
        canvasSplitter.addWidget(sideWidget)

#        sideLayout.addWidget(self.navigationPanel)
        self.setLayout(mainLayout)
#        self.resize(1200, 900)
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

    def onselect(self, eclick, erelease):
        extents = list(self.paletteWidget.span.extents)
        self.customGlWidget.selColorMin = np.min([extents[0], extents[1]])
        self.customGlWidget.selColorMax = np.max([extents[0], extents[1]])
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

    def changeColorAxis(self, selAxis):
        self.customGlWidget.getColor = getattr(
            raycing, 'get_{}'.format(selAxis))
        self.customGlWidget.newColorAxis = True
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.selColorMin = self.customGlWidget.colorMin
        self.customGlWidget.selColorMin = self.customGlWidget.colorMax
        self.mplAx.set_xlabel(selAxis)
        self.im.set_extent((self.customGlWidget.colorMin,
                            self.customGlWidget.colorMax,
                            0, 1))
        self.mplFig.canvas.draw()
        self.paletteWidget.span.active_handle = None
        self.paletteWidget.span.to_draw.set_visible(False)

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
            self.customGlWidget.rotVecX[0] = np.float32(position)
        elif cPan.objectName[-1] == 'y':
            self.customGlWidget.rotVecY[0] = np.float32(position)
        elif cPan.objectName[-1] == 'z':
            self.customGlWidget.rotVecZ[0] = np.float32(position)
        self.customGlWidget.glDraw()

    def updateRotationFromGL(self, rotY, rotZ):
        self.rotationPanel.layout().itemAt(5).widget().setValue(rotY)
        self.rotationPanel.layout().itemAt(8).widget().setValue(rotZ)

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
                      'tVec', 'cameraPos', 'rotVecX', 'rotVecY', 'rotVecZ',
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
                      'tVec', 'cameraPos', 'rotVecX', 'rotVecY', 'rotVecZ',
                      'visibleAxes', 'signs', 'selColorMin', 'selColorMax',
                      'colorMin', 'colorMax']:
            setattr(self.customGlWidget, param, params[param])
        self.setGeometry(params['size'])

        for axis in range(3):
            self.zoomPanel.layout().itemAt((axis+1)*3-1).widget().setValue(
                np.log10(self.customGlWidget.scaleVec[axis]))

        self.rotationPanel.layout().itemAt(2).widget().setValue(
            self.customGlWidget.rotVecX[0])
        self.rotationPanel.layout().itemAt(5).widget().setValue(
            self.customGlWidget.rotVecY[0])
        self.rotationPanel.layout().itemAt(8).widget().setValue(
            self.customGlWidget.rotVecZ[0])

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
        self.customGlWidget.coordOffset = list(self.oesList[oeName].center)
        self.customGlWidget.tVec = np.float32([0, 0, 0])
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        self.customGlWidget.glDraw()

    def updateCutoff(self, position):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        cPan.parent().layout().itemAt(cIndex-1).widget().setText(str(position))
        extents = list(self.paletteWidget.span.extents)
        self.customGlWidget.cutoffI = np.float32(position)
        self.customGlWidget.populateVerticesArray(self.segmentsModelRoot)
        newExtents = (extents[0], extents[1],
                      self.customGlWidget.cutoffI, extents[3])
        self.paletteWidget.span.extents = newExtents
        self.customGlWidget.glDraw()

    def updateCutoffFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        value = float(str(cPan.text()))
        cPan.parent().layout().itemAt(cIndex+1).widget().setValue(value)
        self.customGlWidget.glDraw()

    def updateOpacityFromQLE(self):
        cPan = self.sender()
        cIndex = cPan.parent().layout().indexOf(cPan)
        if cPan.objectName[-1] == '0':
            value = float(str(cPan.text()))
        else:
            value = int(str(cPan.text()))
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
        if cPan.objectName[-1] == '0':
            value = float(str(cPan.text()))
        else:
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
    rotationUpdated = QtCore.pyqtSignal(np.float32, np.float32)
    scaleUpdated = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent, arrayOfRays, modelRoot):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)
        self.aspect = 1.
        self.cameraAngle = 60
        self.setMouseTracking(True)
        self.surfCPOrder = 4
        self.oesToPlot = []
        self.tiles = [5, 5]
#        self.eMin = arrayOfRays[2][0].eMin
#        self.eMax = arrayOfRays[2][0].eMax
        self.arrayOfRays = arrayOfRays
        self.beamsDict = arrayOfRays[1]

        self.projectionsVisibility = [0, 0, 0]
        self.lineOpacity = 0.1
        self.lineWidth = 1
        self.pointOpacity = 0.1
        self.pointSize = 1

        self.lineProjectionOpacity = 0.1
        self.lineProjectionWidth = 1
        self.pointProjectionOpacity = 0.1
        self.pointProjectionSize = 1

        self.coordOffset = [0., 0., 0.]
        self.enableAA = False
        self.enableBlending = True
        self.cutoffI = 0.01
        self.getColor = raycing.get_energy
#        self.selColorMax = 1e20
#        self.selColorMin = -1e20
        self.newColorAxis = True
        self.populateVerticesArray(modelRoot)

        maxC = np.max(self.verticesArray, axis=0)
        minC = np.min(self.verticesArray, axis=0)
        self.maxLen = np.max(maxC - minC)

        self.drawGrid = True
        self.aPos = [0.9, 0.9, 0.9]
        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])
        self.scaleVec = np.array([1., 1., 1.])
        self.tVec = np.array([0., 0., 0.])
        self.cameraTarget = [0., 0., 0.]
        self.cameraPos = np.float32([3.5, 0., 0.])
        self.rotVecX = np.float32([0., 1., 0., 0.])
        self.rotVecY = np.float32([0., 0., 1., 0.])
        self.rotVecZ = np.float32([0., 0., 0., 1.])
        pModelT = np.identity(4)
        self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
        self.signs = np.ones_like(pModelT)
        self.oesList = None
        self.glDraw()

    def setPointSize(self, pSize):
        self.pointSize = pSize
        self.glDraw()

    def setLineWidth(self, lWidth):
        self.lineWidth = lWidth
        self.glDraw()

    def populateVerticesArray(self, segmentsModelRoot):
        self.verticesArray = None
        self.oesToPlot = []
        self.footprints = dict()
        colors = None
        alpha = None
        if self.newColorAxis:
            self.colorMax = -1e20
            self.colorMin = 1e20
        for ioe in range(segmentsModelRoot.rowCount()):
            ioeItem = segmentsModelRoot.child(ioe, 0)
            if ioeItem.checkState() == 2:
                self.oesToPlot.append(str(ioeItem.text()))
                self.footprints[str(ioeItem.text())] = None
            if ioeItem.hasChildren():
                for isegment in range(ioeItem.rowCount()):
                    segmentItem0 = ioeItem.child(isegment, 0)
                    segmentItem1 = ioeItem.child(isegment, 1)
#                    beams = str(segmentItem.text())
                    startBeam = self.beamsDict[str(segmentItem0.text())]
                    good = startBeam.state > 0

                    self.colorMax = max(np.max(self.getColor(startBeam)[good]),
                                        self.colorMax)
                    self.colorMin = min(np.min(self.getColor(startBeam)[good]),
                                        self.colorMin)
                    if self.newColorAxis:
                        self.selColorMin = self.colorMin
                        self.selColorMax = self.colorMax

                    if segmentItem0.checkState() == 2 and\
                            segmentItem1.checkState() == 2:

                        endBeam = self.beamsDict[str(segmentItem1.text())]
                        intensity = np.abs(startBeam.Jss**2+startBeam.Jpp**2)
                        intensity /= np.max(intensity)

                        good = np.logical_and(startBeam.state > 0,
                                              intensity >= self.cutoffI)
                        goodC = np.logical_and(startBeam.E <= self.selColorMax,
                                               startBeam.E >= self.selColorMin)

                        good = np.logical_and(good, goodC)

                        alpha = np.repeat(intensity[good], 2).T if\
                            alpha is None else np.concatenate(
                                (alpha.T, np.repeat(intensity[good], 2).T))

                        colors = np.repeat(np.array(self.getColor(
                            startBeam)[good]), 2).T if\
                            colors is None else np.concatenate(
                                (colors.T,
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

        colors = (colors - self.colorMin) / (self.colorMax - self.colorMin)
        colors = np.dstack((colors,
                            np.ones_like(alpha)*0.85,
                            alpha))

        colorsRGB = np.squeeze(mpl.colors.hsv_to_rgb(colors))
        alphaColor = np.array([alpha]).T * self.lineOpacity
        self.allColor = np.float32(np.hstack([colorsRGB, alphaColor]))
        self.newColorAxis = False

    def modelToWorld(self, coords, dimension=None):
        if dimension is None:
            return np.float32(((coords + self.tVec) * self.scaleVec) /
                              self.maxLen)
        else:
            return np.float32(((coords[dimension] + self.tVec[dimension]) *
                              self.scaleVec[dimension]) / self.maxLen)

    def paintGL(self):

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
        glRotatef(*self.rotVecX)
        glRotatef(*self.rotVecY)
        glRotatef(*self.rotVecZ)
        axPosModifier = np.ones(3)

#        glPushMatrix();
#        glTranslatef(0.0,-1.2,-6);
#        glutWireCone(1,2, 16, 16);
#        glPopMatrix();

        for dim in range(3):
            for iAx in range(3):
                axPosModifier[iAx] = (self.signs[0][iAx] if
                                      self.signs[0][iAx] != 0 else 1)
            if self.projectionsVisibility[dim] > 0:
                projection = self.modelToWorld(np.copy(self.verticesArray))
                projection[:, dim] = -self.aPos[dim] * axPosModifier[dim]
                vertexArray = vbo.VBO(projection)
                vertexArray.bind()
                glVertexPointerf(vertexArray)

                if self.lineProjectionWidth > 0:
                    self.allColor[:, 3] = np.float32(
                        self.lineProjectionOpacity)
                    colorArray = vbo.VBO(self.allColor)
                    colorArray.bind()
                    glColorPointerf(colorArray)
                    glLineWidth(self.lineProjectionWidth)
                    glDrawArrays(GL_LINES, 0, len(self.verticesArray))
                    colorArray.unbind()

                if self.pointProjectionSize > 0:
                    self.allColor[:, 3] = np.float32(
                        self.pointProjectionOpacity)
                    colorArray = vbo.VBO(self.allColor)
                    colorArray.bind()
                    glColorPointerf(colorArray)
                    glPointSize(self.pointProjectionSize)
                    glDrawArrays(GL_POINTS, 0, len(self.verticesArray))

                vertexArray.unbind()

        if self.drawGrid:
            glLoadIdentity()
            glRotatef(*self.rotVecX)
            glRotatef(*self.rotVecY)
            glRotatef(*self.rotVecZ)
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
            allLimits = limits * self.maxLen / self.scaleVec - self.tVec
            axGrids = []

            gridLabels = None
            precisionLabels = None

            for iAx in range(3):
                dx = np.abs(allLimits[:, iAx][0] - allLimits[:, iAx][1])*0.15
                decimalX = int(np.abs(np.modf(np.log10(dx))[1])) + 1 if\
                    dx < 10 else 0
                dx = np.round(dx, decimalX)
                gridX = np.arange(np.round(allLimits[:, iAx][0], decimalX),
                                  allLimits[:, iAx][1], dx)
                gridX = gridX if gridX[0] >= allLimits[:, iAx][0] else\
                    gridX[1:]
                gridLabels = np.concatenate((
                    gridLabels, gridX + self.coordOffset[iAx])) if gridLabels\
                    is not None else gridX + self.coordOffset[iAx]
                precisionLabels = np.concatenate((
                    precisionLabels,
                    np.ones_like(gridX)*decimalX)) if precisionLabels\
                    is not None else np.ones_like(gridX)*decimalX
                axGrids.extend([gridX])

            back[:, 0] *= axPosModifier[0]
            side[:, 1] *= axPosModifier[1]
            bottom[:, 2] *= axPosModifier[2]

            xAxis = np.vstack(
                (self.modelToWorld(axGrids, 0),
                 np.ones(len(axGrids[0]))*self.aPos[1]*axPosModifier[1],
                 np.ones(len(axGrids[0]))*-self.aPos[2]*axPosModifier[2]))
            yAxis = np.vstack(
                (np.ones(len(axGrids[1]))*self.aPos[0]*axPosModifier[0],
                 self.modelToWorld(axGrids, 1),
                 np.ones(len(axGrids[1]))*-self.aPos[2]*axPosModifier[2]))
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
            zAxisC = np.vstack(
                (np.ones(len(axGrids[2]))*self.aPos[0]*axPosModifier[0],
                 np.ones(len(axGrids[2]))*-self.aPos[1]*axPosModifier[1],
                 self.modelToWorld(axGrids, 2)))

            xLines = np.vstack(
                (xAxis, xAxisB, xAxisB, xAxisC)).T.flatten().reshape(
                4*xAxis.shape[1], 3)
            yLines = np.vstack(
                (yAxis, yAxisB, yAxisB, yAxisC)).T.flatten().reshape(
                4*yAxis.shape[1], 3)
            zLines = np.vstack(
                (zAxis, zAxisB, zAxisB, zAxisC)).T.flatten().reshape(
                4*zAxis.shape[1], 3)

            axTicks = np.vstack((xAxis.T, yAxis.T, zAxisC.T))
            axGrid = np.vstack((xLines, yLines, zLines))

            for tick, tText, pcs in zip(axTicks, gridLabels, precisionLabels):
#                glPushMatrix()
#                glTranslatef(*tick)
#                glScalef(1./2000., 1./2000., 1./2000.)
#                for symbol in "   {0:.{1}f}".format(tText, int(pcs)):
#                    glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(symbol))
#                glPopMatrix()
                glRasterPos3f(*tick)
                for symbol in "   {0:.{1}f}".format(tText, int(pcs)):
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(symbol))

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
        glRotatef(*self.rotVecX)
        glRotatef(*self.rotVecY)
        glRotatef(*self.rotVecZ)

        if self.oesList is not None:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_DEPTH_TEST)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_LIGHTING)
            self.addLighting(3.)

            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.3, 0.3, 0.3, 1])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1., 0.9, 0.8, 1])
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.1, 0.1, 0.1, 1])
            glMaterialf(GL_FRONT, GL_SHININESS, 100)
            glEnable(GL_MAP2_VERTEX_3)
            glEnable(GL_MAP2_NORMAL)
#            glEnable(GL_AUTO_NORMAL)

            for oeString in self.oesToPlot:
                oeToPlot = self.oesList[oeString]
                elType = str(type(oeToPlot))
                if len(re.findall('raycing.oe', elType.lower())) > 0:  # OE
                    if hasattr(oeToPlot, 'local_z2'):  # DCM
                        for surf in [1, 2]:
                            self.plotSurface(oeToPlot, surf)
                    else:
                        self.plotSurface(oeToPlot)

                elif len(re.findall('raycing.apert', elType)) > 0:  # aperture
                    continue
                elif len(re.findall('raycing.screen', elType)) > 0:  # screen
                    continue
                else:
                    continue

            glDisable(GL_MAP2_VERTEX_3)
            glDisable(GL_MAP2_NORMAL)
#            glDisable(GL_AUTO_NORMAL)
            glDisable(GL_DEPTH_TEST)
#            glShadeModel( GL_SMOOTH )
            glDisable(GL_LIGHTING)
#            glDisable(GL_LIGHT0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        vertexArray = vbo.VBO(self.modelToWorld(self.verticesArray))
        vertexArray.bind()
        glVertexPointerf(vertexArray)

        if self.lineWidth > 0:
            self.allColor[:, 3] = np.float32(self.lineOpacity)
            colorArray = vbo.VBO(self.allColor)
            colorArray.bind()
            glColorPointerf(colorArray)
            glLineWidth(self.lineWidth)
            glDrawArrays(GL_LINES, 0, len(self.verticesArray))
            colorArray.unbind()

        if self.pointSize > 0:
            self.allColor[:, 3] = np.float32(self.pointOpacity)
            colorArray = vbo.VBO(self.allColor)
            colorArray.bind()
            glColorPointerf(colorArray)
            glPointSize(self.pointSize)
            glDrawArrays(GL_POINTS, 0, len(self.verticesArray))
            colorArray.unbind()

        vertexArray.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glFlush()
        self.drawAxes()

        if self.enableAA:
            glDisable(GL_LINE_SMOOTH)

        if self.enableBlending:
            glDisable(GL_MULTISAMPLE)
            glDisable(GL_BLEND)
            glDisable(GL_POINT_SMOOTH)

    def plotSurface(self, oe, nSurf=0):
        nsIndex = nSurf - 1 if nSurf > 0 else nSurf
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

                local_z = getattr(oe, 'local_z') if nSurf == 0 else\
                    getattr(oe, 'local_z{}'.format(nSurf))
                local_n = getattr(oe, 'local_n') if nSurf == 0 else\
                    getattr(oe, 'local_n{}'.format(nSurf))
                zv = local_z(xv, yv)
                nv = local_n(xv, yv)

                gbp = rsources.Beam(nrays=len(xv))
                gbp.x = xv
                gbp.y = yv
                gbp.z = zv
                gbp.a = nv[0] * np.ones_like(zv)
                gbp.b = nv[1] * np.ones_like(zv)
                gbp.c = nv[2] * np.ones_like(zv)

                if nSurf == 2:
                    oe.local_to_global(gbp, is2ndXtal=True)
                else:
                    oe.local_to_global(gbp)
                surfCP = np.vstack((gbp.x, gbp.y, gbp.z)).T -\
                    self.coordOffset

                glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1,
                        self.modelToWorld(surfCP.reshape(
                            self.surfCPOrder,
                            self.surfCPOrder, 3)))

                surfNorm = np.vstack((gbp.a, gbp.b, gbp.c)).T

                glMap2f(GL_MAP2_NORMAL, 0, 1, 0, 1,
                        surfNorm.reshape(
                            self.surfCPOrder,
                            self.surfCPOrder, 3))

                glMapGrid2f(self.surfCPOrder, 0.0, 1.0,
                            self.surfCPOrder, 0.0, 1.0)

                glEvalMesh2(GL_FILL, 0, self.surfCPOrder,
                            0, self.surfCPOrder)


    def addLighting(self, pos):
        spot = 60
        exp = 10
        ambient = [0.2, 0.2, 0.2, 1]
        diffuse = [0.3, 0.3, 0.3, 1]
        specular = [1.0, 1.0, 1.0, 1]
        glEnable(GL_LIGHT0)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, pos, 1])
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [0, 0, -pos, 1])
        glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT0, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular)

        glEnable(GL_LIGHT1)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT1, GL_POSITION, [0, 0, -pos, 1])
        glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, [0, 0, pos, 1])
        glLightfv(GL_LIGHT1, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT1, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT1, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT1, GL_SPECULAR, specular)

        glEnable(GL_LIGHT2)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT2, GL_POSITION, [0, pos, 0, 1])
        glLightfv(GL_LIGHT2, GL_SPOT_DIRECTION, [0, -pos, 0, 1])
        glLightfv(GL_LIGHT2, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT2, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT2, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT2, GL_SPECULAR, specular)

        glEnable(GL_LIGHT3)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT3, GL_POSITION, [0, -pos, 0, 1])
        glLightfv(GL_LIGHT3, GL_SPOT_DIRECTION, [0, pos, 0, 1])
        glLightfv(GL_LIGHT3, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT3, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT3, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT3, GL_SPECULAR, specular)

        glEnable(GL_LIGHT4)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT4, GL_POSITION, [pos, 0, 0, 1])
        glLightfv(GL_LIGHT4, GL_SPOT_DIRECTION, [-pos, 0, 0, 1])
        glLightfv(GL_LIGHT4, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT4, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT4, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT4, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT4, GL_SPECULAR, specular)

        glEnable(GL_LIGHT5)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
        glLightfv(GL_LIGHT5, GL_POSITION, [-pos, 0, 0, 1])
        glLightfv(GL_LIGHT5, GL_SPOT_DIRECTION, [pos, 0, 0, 1])
        glLightfv(GL_LIGHT5, GL_SPOT_CUTOFF, spot)
        glLightfv(GL_LIGHT5, GL_SPOT_EXPONENT, exp)
        glLightfv(GL_LIGHT5, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT5, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT5, GL_SPECULAR, specular)


    def drawAxes(self):
        arrowSize = 0.05
        axisLen = 0.1

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
        glRotatef(*self.rotVecX)
        glRotatef(*self.rotVecY)
        glRotatef(*self.rotVecZ)

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
        for symbol in "  {}".format('Z'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(symbol))

        glColor4f(0, 1, 0, 1)
        glRasterPos3f(0, axisLen*1.5, 0)
        for symbol in "  {}".format('Y'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(symbol))

        glColor4f(0, 0, 1, 1)
        glRasterPos3f(axisLen*1.5, 0, 0)
        for symbol in "  {}".format('X'):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(symbol))
        glFlush()
        glViewport(*pView)
        glColor4f(1, 1, 1, 1)

    def initializeGL(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glClearColor(0.0, 0.0, 0.0, 1.)
        glViewport(0, 0, 900, 900)

    def resizeGL(self, widthInPixels, heightInPixels):
        glViewport(0, 0, widthInPixels, heightInPixels)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

    def mouseMoveEvent(self, mouseEvent):
        if mouseEvent.buttons() == QtCore.Qt.LeftButton:
            glLoadIdentity()
            glRotatef(*self.rotVecX)
            glRotatef(*self.rotVecY)
            glRotatef(*self.rotVecZ)
            pModelT = np.array(glGetDoublev(GL_TRANSPOSE_MODELVIEW_MATRIX))
            self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
            self.signs = np.sign(pModelT)

            if mouseEvent.modifiers() == QtCore.Qt.NoModifier:
                self.rotVecY[0] += np.float32(
                    (mouseEvent.y() - self.prevMPos[1])*36./90.)
                self.rotVecZ[0] += np.float32(
                    (mouseEvent.x() - self.prevMPos[0])*36./90.)
                self.rotationUpdated.emit(self.rotVecY[0], self.rotVecZ[0])
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

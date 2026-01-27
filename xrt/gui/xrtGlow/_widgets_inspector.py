# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:28:37 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import re
import copy
import numpy as np
from datetime import datetime
from functools import partial
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS

from ..commons import qt

from ...backends import raycing
from ...backends.raycing import materials as rmats
from ...multipro import GenericProcessOrThread as GP
from ...runner import RunCardVals
from ...plotter import deserialize_plots


class InstanceInspector(qt.QDialog):
    """
    This is a basic version of element editor.
    """
    propertiesChanged = qt.Signal(dict)

    def __init__(self, parent=None, dataDict={}, initDict={},
                 epicsDict={}, viewOnly=False, beamLine=None,
                 categoriesDict=None):
        super().__init__(parent)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.windowTitleStr = "{} Live Object Properties".format(dataDict.get(
                'name'))
        self.setWindowTitle(self.windowTitleStr)

        self.model = qt.QStandardItemModel()
        self.modelRoot = self.model.invisibleRootItem()
        self.original_data = raycing.OrderedDict()
        headerLine = ["Property", "Value"]
        self.viewOnly = viewOnly
        self.liveUpdateEnabled = True  # TODO: configurable
        self.widgetType = 'oe'
        self.objectFlag = qt.Qt.ItemFlags(0)
        self.paramFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.valueFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsEditable |
            qt.Qt.ItemIsSelectable)
        self.checkFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable |
            qt.Qt.ItemIsSelectable)

        elementId = dataDict.get('uuid')
        self.elementId = elementId
        epicsTree = None
        if epicsDict:
            epicsTree = epicsDict.pv_map.get(elementId)
            if epicsTree:
                headerLine.append('EPICS PV')

        self.model.setHorizontalHeaderLabels(headerLine)
        self.itemGroups = {}
        self.categoriesDict = categoriesDict
        # We can pass different categories at init for oes/sources/mats
        if categoriesDict is not None:
            for groupName in categoriesDict.keys():
                self.itemGroups[groupName] = self.add_prop(
                        self.modelRoot, groupName)
            self.itemGroups['Other'] = self.add_prop(self.modelRoot, 'Other')

        for key, value in dataDict.items():
            if categoriesDict is not None:
                for igName, igSet in categoriesDict.items():
                    if key in igSet:
                        parentItem = self.itemGroups.get(igName)
                        break
                else:
                    parentItem = self.itemGroups.get('Other')
            else:
                parentItem = self.modelRoot

            if key in ['center']:
                if initDict.get(key) is not None:
                    spVal = raycing.parametrize(initDict.get(key))
                else:
                    spVal = value.strip('([])').split(",")

                for field, val in zip(['x', 'y', 'z'], spVal):
                    nkey = f"{key}.{field}"
                    nvalue = str(val).strip()
                    if epicsTree is not None:
                        epv = epicsTree.get(nkey)
                    else:
                        epv = None
                    self.add_param(parentItem, nkey, nvalue, epv=epv)
                    self.original_data[nkey] = nvalue
                self.add_param(parentItem, f"{key} rbk", value)

            elif key in ['limPhysX', 'limPhysY', 'limPhysX2', 'limPhysY2']:
                spVal = value.strip('([])')
                for field, val in zip(['lmin', 'lmax'], spVal.split(",")):
                    nkey = f"{key}.{field}"
                    nvalue = val.strip()
                    if epicsTree is not None:
                        epv = epicsTree.get(nkey)
                    else:
                        epv = None
                    self.add_param(parentItem, nkey, nvalue, epv=epv)
                    self.original_data[nkey] = nvalue
#                    self.add_row(nkey, nvalue)
#            if hasattr(value, "_fields"):
#                for subfield in value._fields:
#                    subkey = f"{key}.{subfield}"
#                    subval = getattr(value, subfield)
#                    self.add_row(subkey, subval)
#                    self.original_data[subkey] = subval
            else:
                if epicsTree is not None:
                    epv = epicsTree.get(key)
                else:
                    epv = None
                self.original_data[key] = str(value)
                if key in raycing.derivedArgSet:
                    spVal = raycing.parametrize(initDict.get(key))
                    if spVal is None:
                        spVal = value
                    self.add_param(parentItem, key, spVal, epv=epv)
                    self.add_param(parentItem, f"{key} rbk", value)
                else:
#                    if key in raycing.diagnosticArgs:
#                        print(key, value)
                    self.add_param(parentItem, key, value, epv=epv)

#        for item in self.itemGroups.values():
        self.changed_data = {}
        self.model.itemChanged.connect(self.on_item_changed)
        self.highlight_color = qt.QColor("#fffacd")

#        self.table = qt.QTableView()
        self.table = qt.QTreeView()
        self.table.setModel(self.model)
#        self.table.horizontalHeader().setStretchLastSection(True)
#        self.table.resizeRowsToContents()
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.expandAll()
        self.table.setUniformRowHeights(False)
#        self.table.setStyleSheet("""
#            QHeaderView::section {
#                background-color: #002080;
#                color: white;
#            }
#            """)
        comboDelegate = qt.DynamicArgumentDelegate(bl=beamLine)
        self.table.setItemDelegateForColumn(1, comboDelegate)

        # Buttons
        self.button_box = qt.QDialogButtonBox()
        self.ok_button = self.button_box.addButton(
                "OK", qt.QDialogButtonBox.AcceptRole)
        if not viewOnly:
            self.apply_button = self.button_box.addButton(
                    "Apply", qt.QDialogButtonBox.ApplyRole)
            self.cancel_button = self.button_box.addButton(
                    "Cancel", qt.QDialogButtonBox.RejectRole)
            self.apply_button.clicked.connect(self.apply_changes)
            self.cancel_button.clicked.connect(self.reject)

        self.button_box.accepted.connect(self.on_ok_clicked)
        self.ok_button.clicked.connect(self.accept)

        layout = qt.QHBoxLayout(self)

        widgetL = qt.QWidget()
        layoutL = qt.QVBoxLayout(widgetL)
        layoutL.addWidget(self.table)
        layoutL.addWidget(self.button_box)
        self.beamLine = beamLine

        if self.beamLine is None:
            layout.addWidget(widgetL)
            self.liveUpdateEnabled = False
        elif self.beamLine.materialsDict.get(elementId) is not None:
            self.widgetType = 'mat'
            canvasSplitter = qt.QSplitter()
            canvasSplitter.setChildrenCollapsible(False)
            self.dynamicPlotWidget = Curve1dWidget(
                    beamLine=beamLine, elementId=elementId)
            widgetR = qt.QWidget()
            layoutR = qt.QVBoxLayout(widgetR)
            layoutR.addWidget(self.dynamicPlotWidget)
            layout.addWidget(canvasSplitter)
            canvasSplitter.addWidget(widgetL)
            canvasSplitter.addWidget(widgetR)
        elif self.beamLine.fesDict.get(elementId) is not None:
            self.widgetType = 'fe'
            canvasSplitter = qt.QSplitter()
            canvasSplitter.setChildrenCollapsible(False)
            self.dynamicPlotWidget = SurfacePlotWidget(
                    beamLine=beamLine, elementId=elementId)
            widgetR = qt.QWidget()
            layoutR = qt.QVBoxLayout(widgetR)
            layoutR.addWidget(self.dynamicPlotWidget)
            layout.addWidget(canvasSplitter)
            canvasSplitter.addWidget(widgetL)
            canvasSplitter.addWidget(widgetR)
        elif self.beamLine.beamsDictU.get(elementId) is None:  # nothing to show
            layout.addWidget(widgetL)
            self.liveUpdateEnabled = False
        else:  # create dynamicPlotWidget
            plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
            axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))
            plotProps = plotDefArgs
            axHints = {'xaxis': {'label': 'x', 'unit': 'mm'},
                       'yaxis': {'label': 'y', 'unit': 'mm'},
                       'caxis': {'label': 'energy', 'unit': 'eV'}}

            beamDict = self.beamLine.beamsDictU.get(elementId)
            defBeam = list(beamDict.keys())[-1]
            plotProps['beam'] = (elementId, defBeam)

            if defBeam.endswith('lobal'):
                axHints['yaxis']['label'] = 'z'
            else:
                if len(beamDict) > 1:
                    axHints['yaxis']['label'] = r"y"
                    plotProps['aspect'] = 'auto'
                else:   # screen or aperture
                    axHints['yaxis']['label'] = r"z"

            for pname in ['xaxis', 'yaxis', 'caxis']:
                plotProps[pname] = copy.deepcopy(axDefArgs)
                plotProps[pname].update(axHints[pname])

            hiddenProps = {'xPos', 'yPos', 'ePos', 'contourLevels',
                           'contourColors', 'contourFmt', 'contourFactor',
                           'saveName', 'persistentName', 'oe', 'raycingParam',
                           'beamState', 'beamC', 'useQtWidget', 'title',
                           'rayFlag', 'density', 'outline', 'fluxUnit'}

            self.dynamicPlotWidget = ConfigurablePlotWidget(
                    plotProps, parent=self, viewOnly=False,
                    beamLine=self.beamLine,
                    plotId=self.elementId,
                    hiddenProps=hiddenProps)

            canvasSplitter = qt.QSplitter()
            canvasSplitter.setChildrenCollapsible(False)

            widgetR = qt.QWidget()
            layoutR = qt.QVBoxLayout(widgetR)
            layoutR.addWidget(self.dynamicPlotWidget)
            layout.addWidget(canvasSplitter)
            canvasSplitter.addWidget(widgetL)
            canvasSplitter.addWidget(widgetR)

        self.edited_data = {}

    def add_prop(self, parent, propName):
        """Add non-editable Item"""
        child0 = qt.QStandardItem(str(propName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem()
        child1.setFlags(self.paramFlag)
        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        child0.setBackground(qt.QColor("#001a66"))   # dark blue
        child0.setForeground(qt.QColor("white"))     # font color
        child1.setBackground(qt.QColor("#001a66"))   # dark blue
        parent.appendRow([child0, child1])
        return child0

    def add_param(self, parent, paramName, value, epv=None, source=None,
                  unit=None):
        """Add a pair of Parameter-Value Items"""
        toolTip = None
        child0 = qt.QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem(str(value))

        if str(paramName) == 'name' or paramName.endswith('rbk') or\
                parent is self.itemGroups.get('Diagnostic'):
            ch1flag = self.paramFlag
        elif str(paramName) in ['uuid']:
            ch1flag = self.objectFlag
        else:
            ch1flag = self.valueFlag

        child1.setFlags(ch1flag)

        if paramName.endswith('rbk') or\
                parent is self.itemGroups.get('Diagnostic'):
            child1.setBackground(qt.QColor('#E0F7FA'))
            child0.setBackground(qt.QColor('#E0F7FA'))

        if unit is not None:
            child1u = qt.QStandardItem(str(unit))
            child1u.setFlags(self.valueFlag)

        if epv is not None:
            child1e = qt.QStandardItem(str(epv))
            child1e.setFlags(self.valueFlag)

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
        if epv is not None:
            row.append(child1e)

        if not isinstance(source, qt.QStandardItem):
            parent.appendRow(row)
        else:
            parent.insertRow(source.row() + 1, row)

        return child0, child1

    def add_row(self, key, value):
        key_item = qt.QStandardItem(key)
        key_item.setEditable(False)
        val_item = qt.QStandardItem(str(value))
        val_item.setEditable(not self.viewOnly)
        self.model.appendRow([key_item, val_item])

    def on_item_changed(self, item):
        if item.column() != 1:
            return

        row = item.row()
        parent = item.parent() or item.model().invisibleRootItem()
        key = str(parent.child(row, 0).text())
        if key.endswith('rbk') or key in raycing.diagnosticArgs:
            return
        value_str = str(item.text())

        try:
            value = raycing.parametrize(value_str)
        except Exception:
            value = value_str

        original_value = self.original_data.get(key)
        value_changed = value_str != original_value

        # Update the changed_data dictionary
        if value_changed:
            self.changed_data[key] = value
            self.set_row_highlight(item, True)
        else:
            self.changed_data.pop(key, None)
            self.set_row_highlight(item, False)

    def set_row_highlight(self, item, highlight=True):
        row = item.row()
        parent = item.parent() or item.model().invisibleRootItem()
        item.model().blockSignals(True)
        for col in range(self.model.columnCount()):
            itemH = parent.child(row, col)
            if itemH is None:
                continue
            if highlight:
                itemH.setBackground(self.highlight_color)
            else:
                itemH.setBackground(qt.QBrush(qt.Qt.NoBrush))
        item.model().blockSignals(False)

    def show_context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid():
            return

        menu = qt.QMenu(self.table)
        copy_action = qt.QAction("Copy value", self)
        menu.addAction(copy_action)

        def copy_value():
            value = self.model.itemFromIndex(index).text()
            qt.QApplication.clipboard().setText(value)

        copy_action.triggered.connect(copy_value)
        menu.exec_(self.table.viewport().mapToGlobal(position))

    def on_ok_clicked(self):
        self.apply_changes()  # apply changes before accepting
        self.accept()

    def apply_changes(self):  # test vs. materials viewer
        if not self.changed_data:
            return  # nothing to do

        self.propertiesChanged.emit(copy.deepcopy(self.changed_data))

        if self.widgetType == 'oe':
            for row in range(self.modelRoot.rowCount()):  # for categorized trees
                catItem = self.modelRoot.child(row, 0)
                for j in range(catItem.rowCount()):
                    key = str(catItem.child(j, 0).text())
                    if key in self.changed_data:
                        new_value = self.changed_data[key]
                        self.original_data[key] = new_value
                        catItem.child(j, 1).setText(str(new_value))
                        self.set_row_highlight(catItem.child(j, 0), False)
        elif self.widgetType in ['mat', 'fe']:
            rootItem = self.modelRoot
            for j in range(rootItem.rowCount()):
                key = str(rootItem.child(j, 0).text())
                if key in self.changed_data:
                    new_value = self.changed_data[key]
                    self.original_data[key] = str(new_value)
                    rootItem.child(j, 1).setText(str(new_value))
                    self.set_row_highlight(rootItem.child(j, 0), False)
        self.changed_data.clear()

    def update_param(self, pTuple):
        parentItem = None
        if pTuple[0] == self.elementId:
            if self.categoriesDict is not None:
                for igName, igSet in self.categoriesDict.items():
                    if pTuple[1] in igSet:
                        parentItem = self.itemGroups.get(igName)
                        break
                else:
                    parentItem = self.itemGroups.get('Other')
            if parentItem is not None:
                for i in range(parentItem.rowCount()):
                    child0 = parentItem.child(i, 0)
                    if str(child0.text()) == f'{pTuple[1]} rbk':
                        child1 = parentItem.child(i, 1)
                        child1.setText(str(pTuple[2]))
                    elif parentItem is self.itemGroups.get('Diagnostic') and\
                            str(child0.text()) == f'{pTuple[1]}':
                        child1 = parentItem.child(i, 1)
                        child1.setText(str(pTuple[2]))
#                    else:  # all other params? need more conditions?
#                        child1 = parentItem.child(i, 1)
#                        child1.setText(str(pTuple[2]))

    def update_beam(self, beamTag):
        self.dynamicPlotWidget.update_beam(beamTag)


class ConfigurablePlotWidget(qt.QWidget):
    def __init__(self, plotProps, parent=None, viewOnly=False, beamLine=None,
                 plotId=None, hiddenProps={}):
        super().__init__(parent)
#        self.setAttribute(qt.Qt.WA_DeleteOnClose)
#        self.setWindowTitle("Live Plot Builder")
        self.plotId = plotId
        self.hiddenProps = hiddenProps
        plotProps['useQtWidget'] = True
        plotInit = {'Project': {'plots': {'plot': plotProps}}}
        plotObj = deserialize_plots(plotInit)

        self.objectFlag = qt.Qt.ItemFlags(0)
        self.paramFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        self.valueFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsEditable |
            qt.Qt.ItemIsSelectable)
        self.checkFlag = qt.Qt.ItemFlags(
            qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable |
            qt.Qt.ItemIsSelectable)

        self.plotProps = plotProps
        self.liveUpdateEnabled = True  # TODO: configurable
        self.beamLine = beamLine
        self.dynamicPlot = plotObj[0]

        self.set_beam(plotProps.get('beam'))

        layout = qt.QHBoxLayout(self)
        self.dynamicPlot.canvas.setSizePolicy(
            qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        layout.addWidget(self.dynamicPlot.canvas, 1)

        self.fluxLabelList = raycing.allBeamFields
        self.fluxDataList = ['auto'] + list(self.fluxLabelList)
        self.lengthUnitList = list(raycing.allUnitsLenStr.values())
        self.angleUnitList = list(raycing.allUnitsAngStr.values())
        self.energyUnitList = list(raycing.allUnitsEnergyStr.values())

        if not viewOnly:
            self.init_tabs()
            tabs = qt.QTabWidget()
            tabs.addTab(self.trees['top'], "Plot")
            tabs.addTab(self.trees['xaxis'], "X-Axis")
            tabs.addTab(self.trees['yaxis'], "Y-Axis")
            tabs.addTab(self.trees['caxis'], "Color Axis")
            tabs.setTabPosition(qt.QTabWidget.West)
            tabs.tabBar().setStyleSheet("""
                QTabBar::tab {
                    background: #001a66;
                    color: white;
                    padding: 6px 12px;
                    margin-top: 2px;
                    border: 1px solid #009999;
                }

                QTabBar::tab:selected {
                    background: #0033cc;
                    color: white;
                }

                QTabBar::tab:hover {
                    background: #0033cc;
                    color: white;
                }
            """)

            layoutCtrl = qt.QVBoxLayout()
            layoutCtrl.addWidget(tabs)
            layoutCtrl.addWidget(self.exportsPanel)
            layout.addLayout(layoutCtrl, 0)

        self.plot_beam()

    def init_tabs(self):
        headerLine = ["Property", "Value"]

        comboDelegate = qt.DynamicArgumentDelegate(bl=self.beamLine,
                                                   mainWidget=self)
        self.models = {}
        self.trees = {}
        for mLabel in ['top', 'xaxis', 'yaxis', 'caxis']:
            model = qt.QStandardItemModel()
            model.setHorizontalHeaderLabels(headerLine)
            model.itemChanged.connect(self.on_item_changed)
            model.invisibleRootItem().setData(mLabel, qt.Qt.UserRole)
            self.models[mLabel] = model

            table = qt.QTreeView()
            table.setModel(model)
            table.setAlternatingRowColors(True)
            table.setItemDelegateForColumn(1, comboDelegate)
            table.expandAll()
            table.setUniformRowHeights(False)
#            table.setStyleSheet("""
#                QHeaderView::section {
#                    background-color: #002080;
#                    color: white;
#                }
#                """)
            self.trees[mLabel] = table

        for key, value in self.plotProps.items():
            if key.endswith('axis'):
                model = self.models[key]
                parentItem = model.invisibleRootItem()
                for axkey, axval in value.items():
                    if axkey not in self.hiddenProps:
                        self.add_param(parentItem, axkey, axval)
            else:
                model = self.models['top']
                parentItem = model.invisibleRootItem()
                if key in ['beam'] and raycing.is_valid_uuid(self.elementId):
                    value = value[-1]

                if key not in self.hiddenProps:
                    self.add_param(parentItem, key, value)

        self.exportsPanel = qt.QGroupBox(self)
        self.exportsPanel.setSizePolicy(qt.QSizePolicy.Minimum,
                                        qt.QSizePolicy.Minimum)
        self.exportsPanel.setFlat(False)
        self.exportsPanel.setTitle("File Export")
        exportLayout = qt.QHBoxLayout(self.exportsPanel)
        exportLayout.setSpacing(0)
        exportLayout.setContentsMargins(0, 0, 0, 0)

        for label in ['Save plot', 'Pickle plot', 'Export beam']:
            button = qt.QPushButton(label)
            func = getattr(self, label.lower().replace(' ', '_'))
            button.clicked.connect(func)
            exportLayout.addWidget(button)

    def add_param(self, parent, paramName, value):
        """Add a pair of Parameter-Value Items"""
        toolTip = None
        child0 = qt.QStandardItem(str(paramName))
        child0.setFlags(self.paramFlag)
        child1 = qt.QStandardItem(str(value))

        ch1flag = self.valueFlag
        child1.setFlags(ch1flag)

        child0.setDropEnabled(False)
        child0.setDragEnabled(False)
        if toolTip is not None:
            child1.setToolTip(str(toolTip))
        row = [child0, child1]
        parent.appendRow(row)

        return child0, child1

    def on_item_changed(self, item):
        parent = item.model().invisibleRootItem()
        if item.column() == 0:
            return

        elif item.column() == 1 and item.isEnabled():
            paramValue = raycing.parametrize(item.text())
            objChng = str(parent.data(qt.Qt.UserRole))

            row = item.row()
            paramName = str(parent.child(row, 0).text())
            if paramName == 'beam':
                paramValue = self.get_beam_tag(paramValue)
            plotParamTuple = self.plotId, objChng, paramName, paramValue
            try:
                self.update_plot_param(plotParamTuple)
            except Exception as e:
                print(e)
            if paramName == 'aspect':
                self.resizeEvent()
                self.dynamicPlot.plot_plots()

    def get_beam_tag(self, value):
        if raycing.is_valid_uuid(self.plotId):  # oe id
            return (self.plotId, value)
        else:  # plot name, run from Qook. value contains beam name
            pass
#            beams = self.beamModel.findItems(beamName, column=0)
#            beamTag = []
#            for bItem in beams:
#                row = bItem.row()
#                btype = self.beamModel.item(row, 1).text()
#                oeid = self.beamModel.item(row, 2).text()
#                beamTag = (oeid, btype)
#                break
#            return beamTag

    def update_plot_param(self, paramTuple):
        """(PlotUUID, obj: XYCPlot or XYCAxis, pName, pValue)"""
        if paramTuple[0] != self.plotId:
            return

        if paramTuple[2] == 'beam':
            self.set_beam(paramTuple[3])
        elif paramTuple[1].endswith('axis'):  # we only check if it's 'axis'
            axis = getattr(self.dynamicPlot, paramTuple[1])
            setattr(axis, paramTuple[2], paramTuple[3])
        elif paramTuple[2] in ['negative']:
            self.dynamicPlot.set_negative()
        elif paramTuple[2] in ['invertColorMap']:
            self.dynamicPlot.set_invert_colors()
        else:
            setattr(self.dynamicPlot, paramTuple[2], paramTuple[3])

        if paramTuple[2] in ['bins', 'ppb', 'ePos', 'xPos', 'yPos']:
            self.dynamicPlot.reset_bins2D()
            self.dynamicPlot.reset_fig_layout()

        if paramTuple[2] not in ['negative', 'invertColorMap']:
            self.dynamicPlot.clean_plots()
            self.plot_beam()

    def set_beam(self, beamTag):
        elementId, beamKey = beamTag
        self.elementId = elementId
        bdu = self.beamLine.beamsDictU
        self.beamDict = bdu.get(elementId)
        self.dynamicPlot.beam = str(beamKey)

    def update_beam(self, beamTag):
        currentTag = (getattr(self, 'elementId', None), self.dynamicPlot.beam)
        if self.liveUpdateEnabled and beamTag == currentTag:
            self.dynamicPlot.clean_plots()
            self.plot_beam()

    def update_plot(self, outList, iteration=0):
        self.dynamicPlot.nRaysAll += outList[13]
        nRaysVarious = outList[14]
        self.dynamicPlot.nRaysAlive += nRaysVarious[0]
        self.dynamicPlot.nRaysGood += nRaysVarious[1]
        self.dynamicPlot.nRaysOut += nRaysVarious[2]
        self.dynamicPlot.nRaysOver += nRaysVarious[3]
        self.dynamicPlot.nRaysDead += nRaysVarious[4]
        self.dynamicPlot.nRaysAccepted += nRaysVarious[5]
        self.dynamicPlot.nRaysAcceptedE += nRaysVarious[6]
        self.dynamicPlot.nRaysSeeded += nRaysVarious[7]
        self.dynamicPlot.nRaysSeededI += nRaysVarious[8]
        self.dynamicPlot.displayAsAbsorbedPower = outList[15]

        for iaxis, axis in enumerate(
                [self.dynamicPlot.xaxis,
                 self.dynamicPlot.yaxis,
                 self.dynamicPlot.caxis]):
            if (iaxis == 2) and (not self.dynamicPlot.ePos):
                continue
            axis.total1D += outList[0+iaxis*3]
            axis.total1D_RGB += outList[1+iaxis*3]
            if iteration == 0:
                axis.binEdges = outList[2+iaxis*3]

        self.dynamicPlot.total2D += outList[9]
        self.dynamicPlot.total2D_RGB += outList[10]
        if self.dynamicPlot.fluxKind.lower().endswith('4d'):
            self.dynamicPlot.total4D += outList[11]
        elif self.dynamicPlot.fluxKind.lower().endswith('pca'):
            self.dynamicPlot.total4D.append(outList[11])
        self.dynamicPlot.intensity += outList[12]

        if self.dynamicPlot.fluxKind.startswith('E') and \
                self.dynamicPlot.fluxKind.lower().endswith('pca'):
            xbin, zbin =\
                self.dynamicPlot.xaxis.bins, self.dynamicPlot.yaxis.bins
            self.dynamicPlot.total4D = np.concatenate(
                    self.dynamicPlot.total4D).reshape(-1, xbin, zbin)
            self.dynamicPlot.field3D = self.dynamicPlot.total4D
        self.dynamicPlot.textStatus.set_text('')
        self.dynamicPlot.plot_plots()
        self.resizeEvent()
        self.dynamicPlot.plot_plots()

    def plot_beam(self, key=None):
        locCard = RunCardVals(threads=0,
                              processes=1,
                              repeats=1,
                              updateEvery=1,
                              pickleEvery=0,
                              backend='raycing',
                              globalNorm=False,
                              runfile=None)

        locCard.beamLine = self.beamLine

        self.dynamicPlot.runCardVals = locCard
        sproc = GP(locCard=locCard,
                   plots=[self.dynamicPlot.card_copy()],
                   outPlotQueues=[None],
                   alarmQueue=[None],
                   idLoc=0,
                   beamDict=self.beamDict)
        try:
            outList = sproc.run()
            self.update_plot(outList)
        except Exception as e:
            print(e)

    def save_plot(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
#        saveDialog.selectFile("plotname.jpg")
        saveDialog.setNameFilter(
            "JPG files (*.jpg);;PDF files (*.pdf);;SVG files (*.svg);;"
            "PNG files (*.png);;TIFF files (*.tif)")
        saveDialog.selectNameFilter("JPG files (*.jpg)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            try:
                self.dynamicPlot.saveName = filename
                self.dynamicPlot.save()
                self.dynamicPlot.saveName = None
            except Exception as e:
                print(e)

    def pickle_plot(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
#        saveDialog.selectFile("plotname.pickle")
        saveDialog.setNameFilter(
            "Matlab files (*.mat);;"
            "Pickle files (*.pickle)")
        saveDialog.selectNameFilter("Pickle files (*.pickle)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            try:
                self.dynamicPlot.persistentName = filename
                self.dynamicPlot.store_plots()
                self.dynamicPlot.persistentName = None
            except Exception as e:
                print(e)

    def export_beam(self):
        saveDialog = qt.QFileDialog()
        saveDialog.setFileMode(qt.QFileDialog.AnyFile)
        saveDialog.setAcceptMode(qt.QFileDialog.AcceptSave)
#        saveDialog.selectFile("beamname.npy")
        saveDialog.setNameFilter(
            "Matlab files (*.mat);;NPY files (*.npy);;"
            "Pickle files (*.pickle)")
        saveDialog.selectNameFilter("NPY files (*.npy)")
        if (saveDialog.exec_()):
            filename = saveDialog.selectedFiles()[0]
            extension = str(saveDialog.selectedNameFilter())[-5:-1].strip('.')
            if not filename.endswith(extension):
                filename = "{0}.{1}".format(filename, extension)
            beam = self.beamDict.get(self.dynamicPlot.beam)
            if beam is not None:
                try:
                    beam.export_beam(filename, fformat=extension)
                except Exception as e:
                    print(e)

    def resizeEvent(self, event=None):
        b2 = self.dynamicPlot.ax2dHist.get_position().bounds
        x1 = self.dynamicPlot.ax1dHistX.get_position().bounds
        y1 = self.dynamicPlot.ax1dHistY.get_position().bounds
        xp = self.dynamicPlot.xaxis.pixels
        yp = self.dynamicPlot.yaxis.pixels
        if self.dynamicPlot.ePos != 0:
            e1 = self.dynamicPlot.ax1dHistE.get_position().bounds
            b1 = self.dynamicPlot.ax1dHistEbar.get_position().bounds
            ep = self.dynamicPlot.caxis.pixels

        self.dynamicPlot.ax1dHistX.set_position([b2[0], x1[1], b2[2], x1[3]])
        self.dynamicPlot.ax1dHistY.set_position([y1[0], b2[1], y1[2], b2[3]])

        if self.dynamicPlot.ePos == 1:
            self.dynamicPlot.ax1dHistE.set_position(
                [e1[0], b2[1], e1[2], b2[3]*ep/yp])
            self.dynamicPlot.ax1dHistEbar.set_position(
                [b1[0], b2[1], b1[2], b2[3]*ep/yp])
        elif self.dynamicPlot.ePos == 2:
            self.dynamicPlot.ax1dHistE.set_position(
                [b2[0], e1[1], b2[2]*ep/xp, e1[3]])
            self.dynamicPlot.ax1dHistEbar.set_position(
                [b2[0], b1[1], b2[2]*ep/xp, b1[3]])


class Curve1dWidget(qt.QWidget):

    allCurves = {'σ': '-', 'π': '--'}  # can also be marker+linestyle, e.g 'd-'

    allColors = []
    for color in TABLEAU_COLORS.keys():
        colorName = color.split(":")[-1]
        allColors.append(colorName)

    initParams = dict(
        common1=[("E from Source", None), ("Emin (eV)", 5000.),
                 ("Emax (eV)", 15000.), ("N points", 1000)],
        plate=[],
        mirror=[(u"θ from OE", None), (u"Grazing angle θ (mrad)", 5.),
                ("Curves", ['σ', ])],
        crystal=[(u"θ from OE", None), (u"Grazing angle θ (°)", 15.),
                ("Asymmetry angle", 0), ("Curves", ['σ', ])],
        common2=[("Curve Color", "blue")],
        )

    def __init__(self, beamLine=None, elementId=None):
        super().__init__()

        self.beamLine = beamLine
        self.elementId = elementId
        self.layout = qt.QHBoxLayout()
        self.mainSplitter = qt.QSplitter(qt.Qt.Horizontal, self)

        # Create a QVBoxLayout for the plot and the toolbar
        plot_widget = qt.QWidget(self)
        self.plot_layout = qt.QVBoxLayout()

        self.allIcons = {}
        for colorName, colorCode in zip(
                self.allColors, TABLEAU_COLORS.values()):
            self.allIcons[colorName] = self.create_colored_icon(colorCode)

        self.figure = Figure()
        self.canvas = qt.FigCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.angleUnitList = list(raycing.allUnitsAng.keys())

        self.plot_lines = {}

        # Add the Matplotlib toolbar to the QVBoxLayout
        self.toolbar = qt.NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)

        # Add the canvas to the QVBoxLayout
        self.plot_layout.addWidget(self.canvas)

        plot_widget.setLayout(self.plot_layout)
        self.mainSplitter.addWidget(plot_widget)

        tree_widget = qt.QWidget(self)
        self.tree_layout = qt.QVBoxLayout()
        self.model = qt.QStandardItemModel()
        self.tree_view = qt.QTreeView(self)
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(True)

        self.tree_view.setAlternatingRowColors(True)
#        self.tree_view.setContextMenuPolicy(qt.Qt.CustomContextMenu)
#        self.tree_view.customContextMenuRequested.connect(
#                self.show_context_menu)
        comboDelegate = qt.DynamicArgumentDelegate(bl=beamLine,
                                                   mainWidget=self)
        self.tree_view.setItemDelegateForColumn(1, comboDelegate)
#        self.add_plot_button = qt.QPushButton("Add curve")
#        self.add_plot_button.clicked.connect(self.add_plot)

        self.export_button = qt.QPushButton("Export curve")

        self.buttons_layout = qt.QHBoxLayout()
#        self.buttons_layout.addWidget(self.add_plot_button)
        self.buttons_layout.addWidget(self.export_button)
        self.tree_layout.addWidget(self.tree_view)
        self.tree_layout.addLayout(self.buttons_layout)

        tree_widget.setLayout(self.tree_layout)
        self.mainSplitter.addWidget(tree_widget)
        self.layout.addWidget(self.mainSplitter)
        self.setLayout(self.layout)

#       keep it here for crystals
#        self.xlabel_base_e = r'$E - E_B$'
        self.xlabel_base_e = r'$E$'
        self.axes.set_xlabel('{0} (eV)'.format(self.xlabel_base_e))

        self.default_plot = self.add_plot()
        self.setCurveColor(self.allColors[0])
        self.calculate()
#        self.resize(1100, 700)
#        self.mainSplitter.setSizes([700, 400])
        self.tree_view.resizeColumnToContents(0)
        self.tree_view.clicked.connect(self.on_item_clicked)
        self.model.itemChanged.connect(self.on_item_changed)
        self.export_button.clicked.connect(self.export_curve)

    def add_plot(self):
        eMin = eMax = None
        angle_rad = None
        srcName = "None"
        fromOeTxt = "None"
        self.kindParams = None
        if self.beamLine is not None and self.elementId is not None:
            mat = self.beamLine.materialsDict.get(self.elementId)
            matName = mat.name if mat is not None else ''
            plot_name = f"{matName}"
            # if hasattr(mat, 'efficiency') and mat.efficiency is not None:
            #     plot_name = f"{matName} efficiency"
            if mat.kind in ('plate', 'lens'):
                plot_name = f"{matName} abs coeff"
                self.axes.set_ylabel('Absorption coefficient (cm⁻¹)')
                self.kindParams = 'plate'
            elif mat.kind in ('mirror', 'multilayer'):
                if hasattr(mat, 'geom'):
                    if mat.geom.endswith('transmitted'):
                        plot_name = f"{matName} transmittivity"
                    else:
                        plot_name = f"{matName} reflectivity"
                self.axes.set_ylabel(plot_name)
                self.kindParams = 'mirror'
            elif 'crystal' in mat.kind:
                if hasattr(mat, 'geom'):
                    if hasattr(mat, 'geom') and mat.geom.endswith('mitted'):
                        plot_name = f"{matName} transmittivity"
                    else:
                        plot_name = f"{matName} reflectivity"
                self.axes.set_ylabel(plot_name)
                self.kindParams = 'crystal'

            for oeLine in self.beamLine.oesDict.values():
                oeObj = oeLine[0]
                if hasattr(oeObj, 'nrays') and eMin is None:
                    srcName = oeObj.name
                    eMin, eMax = self.get_source_range(srcName)

                if hasattr(oeObj, 'material') and angle_rad in [0, None]:
                    if raycing.is_valid_uuid(oeObj.material):
                        oeMatId = oeObj.material
                    else:
                        oeMatId = getattr(oeObj.material, 'uuid', None)

                    if oeMatId == self.elementId:
                        oeName = oeObj.name
                        angle_rad = self.get_oe_angle(oeName, 'beam')
                        fromOeTxt = oeName + ': local beam'
                        if angle_rad is None:
                            angle_rad = self.get_oe_angle(oeName, 'pitch')
                            fromOeTxt = oeName + ': pitch'

        plot_uuid = raycing.uuid.uuid4()
        plots = []
        for k, v in self.allCurves.items():
            if v[0] in Line2D.markers:
                kw = dict(linestyle=v[1:], marker=v[0])
            else:
                kw = dict(linestyle=v)
            line = Line2D([], [], label=k, **kw)
            plots.append(line)
            self.axes.add_line(line)
            if self.kindParams == 'plate':
                self.axes.set_yscale('log')
        self.plot_lines[plot_uuid] = plots

        plot_item = qt.QStandardItem()
        plot_item.setFlags(plot_item.flags() | qt.Qt.ItemIsEditable)
        plot_item.plot_index = plot_uuid
        plot_item.skipRecalculation = False
        plot_item.fwhms = [None for label in self.allCurves]

        cbk_item = qt.QStandardItem()
        cbk_item.setFlags(qt.Qt.NoItemFlags)
        self.model.appendRow([plot_item, cbk_item])

        self.params = self.initParams['common1'] + \
            self.initParams[self.kindParams] + self.initParams['common2']
        for iname, ival in self.params:
            item_name = qt.QStandardItem(iname)
            item_name.setFlags(item_name.flags() & ~qt.Qt.ItemIsEditable)
            if iname.startswith("Curves"):
                item_value = qt.QStandardItem()
                item_value.setFlags(item_value.flags())
                w = qt.StateButtons(
                    self.tree_view, list(self.allCurves.keys()), ival)
                w.statesActive.connect(partial(
                    self.on_item_changed, item_value))
                plot_item.appendRow([item_name, item_value])
                self.tree_view.setIndexWidget(item_value.index(), w)
            else:
                item_value = qt.QStandardItem(str(ival))
                item_value.setFlags(item_value.flags() | qt.Qt.ItemIsEditable)
                plot_item.appendRow([item_name, item_value])

            if "emin" in iname.lower() and eMin is not None:
                item_value.setText(str(eMin))
            elif "emax" in iname.lower() and eMin is not None:
                item_value.setText(str(eMax))
            elif "grazing" in iname.lower():
                if angle_rad in [0, None]:
                    angle_val = ival
                    if '(°)' in iname:
                        angle_rad = np.radians(angle_val)
                    elif '(mrad)' in iname:
                        angle_rad = angle_val*1e-3
                    else:
                        angle_rad = angle_val
                else:
                    if '(°)' in iname:
                        angle_val = np.degrees(angle_rad)
                    elif '(mrad)' in iname:
                        angle_val = angle_rad*1e3
                    else:
                        angle_val = angle_rad
                item_value.setText(str(angle_val))
                item_value.setData(float(angle_rad), role=qt.Qt.UserRole)
            elif "from source" in iname.lower():
                item_value.setText(srcName)
            elif "from oe" in iname.lower():
                item_value.setText(fromOeTxt)
            elif "asymmetry" in iname.lower():
                # if not hasattr(mat, 'get_Bragg_angle'):
                #     item_name.setEnabled(False)
                #     item_value.setEnabled(False)
                item_value.setData(0, role=qt.Qt.UserRole)
            elif iname == "Curve Color":
                cb = qt.QComboBox()
                cb.setMaxVisibleItems(25)
                model = qt.QStandardItemModel()
                cb.setModel(model)
                for color in self.allColors:
                    item = qt.QStandardItem(color)
                    item.setIcon(self.allIcons[color])
                    model.appendRow(item)
                cb.setCurrentText(ival)
                self.tree_view.setIndexWidget(item_value.index(), cb)
                cb.currentTextChanged.connect(self.setCurveColor)

        plot_item.setText(plot_name)
        if self.kindParams != 'plate':
            self.add_legend()

        plot_index = self.model.indexFromItem(plot_item)
        self.tree_view.expand(plot_index)
        return plot_item

    def add_legend(self):
        if self.axes.get_legend() is not None:
            self.axes.get_legend().remove()
        lines, labels = [], []
        for plots in self.plot_lines.values():
            for line in plots:
                if line.get_visible():
                    lines.append(line)
                    labels.append(line.get_label())
        self.axes.legend(lines, labels)

    def create_colored_icon(self, color):
        pixmap = qt.QPixmap(16, 16)
        pixmap.fill(qt.QColor(color))
        return qt.QIcon(pixmap)

    def on_item_changed(self, item):
        if item.index().column() == 1:
            pname = self.default_plot.child(item.index().row(), 0).text()
            pvalue = item.text()
            dest = self.default_plot.child(item.index().row(), 1)

            if "grazing" in pname.lower():
                try:
                    pvalue = float(pvalue)
                except ValueError as e:
                    print(e)
                    return
                if '(°)' in pname:
                    angle = np.radians(pvalue)
                elif '(mrad)' in pname:
                    angle = pvalue*1e-3
                else:
                    angle = pvalue
                self.model.blockSignals(True)
                dest.setData(angle, role=qt.Qt.UserRole)
                self.model.blockSignals(False)
            elif "asymmetry" in pname.lower():
                try:
                    pvalue = float(pvalue)
                except ValueError as e:
                    print(e)
                    return
                angle = np.radians(pvalue)
                self.model.blockSignals(True)
                dest.setData(angle, role=qt.Qt.UserRole)
                self.model.blockSignals(False)
            elif "from source" in pname.lower():
                eMin, eMax = self.get_source_range(pvalue)
                self.model.blockSignals(True)
                if eMin is not None:
                    ind = self.findIndexFromText("Emin")
                    self.default_plot.child(ind, 1).setText(str(eMin))
                if eMax is not None:
                    ind = self.findIndexFromText("Emax")
                    self.default_plot.child(ind, 1).setText(str(eMax))
                self.model.blockSignals(False)
            elif "from oe" in pname.lower():
                try:
                    oeName, fromWhat = pvalue.split(': ')
                except ValueError:
                    return
                angle_rad = self.get_oe_angle(oeName, fromWhat)
                if angle_rad is None:
                    return
                self.model.blockSignals(True)
                if self.kindParams == 'mirror':  # mrad
                    angle_val = angle_rad*1e3
                elif self.kindParams == 'crystal':  # deg
                    angle_val = np.degrees(angle_rad)
                else:
                    angle_val = angle_rad
                ind = self.findIndexFromText("Grazing")
                grdest = self.default_plot.child(ind, 1)
                grdest.setText(str(angle_val))
                grdest.setData(angle_rad, role=qt.Qt.UserRole)
                self.model.blockSignals(False)

            if pname not in ["Curves", "Curve Color"]:
                self.calculate()
            else:
                xaxis, curS, curP = copy.copy(self.default_plot.curves)
                self.on_calculation_result((xaxis, curS, curP, 0))

    def get_oe_angle(self, oeName, fromWhat='pitch'):
        angle_rad = None
        if self.beamLine is not None and oeName != 'None':
            oeid = self.beamLine.oenamesToUUIDs.get(oeName)
            oeLn = self.beamLine.oesDict.get(oeid)
            if oeLn is not None:
                oeObj = oeLn[0]
                if fromWhat == 'pitch':
                    angle_rad = getattr(oeObj, '_braggVal', None)
                    if angle_rad in [0, None]:
                        angle_rad = getattr(oeObj, '_pitchVal', None)
                else:  # from 'beam.theta'
                    try:
                        beamDict = self.beamLine.beamsDictU[oeid]
                        if 'beamLocal1' in beamDict:
                            beam = beamDict['beamLocal1']
                        elif 'beamLocal' in beamDict:
                            beam = beamDict['beamLocal']
                        else:
                            beam = beamDict['beamGlobal']
                        good = (beam.state == 1)
                        angle_rad = beam.theta[good].mean()
                    except Exception:
                        angle_rad = None
        if angle_rad is not None:
            angle_rad = abs(angle_rad)
        return angle_rad

    def get_source_range(self, srcName):
        eMin = eMax = None
        if self.beamLine is not None and srcName != 'None':
            oeid = self.beamLine.oenamesToUUIDs.get(srcName)
            srcLn = self.beamLine.oesDict.get(oeid)
            if srcLn is not None:
                srcObj = srcLn[0]
            if hasattr(srcObj, 'eMin'):
                eMin = getattr(srcObj, 'eMin', None)
                eMax = getattr(srcObj, 'eMax', None)
            elif hasattr(srcObj, 'energies'):
                energies = getattr(srcObj, 'energies', None)
                distE = getattr(srcObj, 'distE', None)
                if distE == 'flat':
                    eMin = np.min(energies)
                    eMax = np.max(energies)
                elif distE == 'normal':
                    try:
                        eMin = energies[0] - energies[-1]*5
                        eMax = energies[0] + energies[-1]*5
                    except:
                        commons = self.initParams['common1']
                        eMin = commons[0][1]
                        eMax = commons[1][1]
                elif distE == 'lines':
                    eMin = np.min(energies)
                    eMax = np.max(energies)
                    if eMin == eMax:
                        eMin = eMin * 0.9
                        eMax = eMax * 1.1
        return eMin, eMax

    def findIndexFromText(self, text):
        for i, param in enumerate(self.params):
            if param[0].startswith(text):
                return i
        print(f'Could not find index of "{text}"!')

    def get_e_min(self, item):
        ind = self.findIndexFromText("Emin")
        try:
            return float(item.child(ind, 1).text())
        except ValueError:
            return float(self.params[ind][1])

    def get_e_max(self, item):
        ind = self.findIndexFromText("Emax")
        try:
            return float(item.child(ind, 1).text())
        except ValueError:
            return float(self.params[ind][1])

    def get_npoints(self, item):
        ind = self.findIndexFromText("N ")
        try:
            return int(item.child(ind, 1).text())
        except ValueError:
            return int(self.params[ind][1])

    def get_theta0(self, item):
        ind = self.findIndexFromText("Grazing")
        try:
            theta = float(item.child(ind, 1).data(qt.Qt.UserRole))
            return theta
        except ValueError:
            return 1e-2

    def get_alpha(self, item):
        ind = self.findIndexFromText("Asymmetry")
        try:
            alpha = float(item.child(ind, 1).data(qt.Qt.UserRole))
            return alpha
        except ValueError:
            return 0

    def get_curve_types(self, item):
        ind = self.findIndexFromText("Curves")
        modelIndex = self.model.indexFromItem(item.child(ind, 1))
        w = self.tree_view.indexWidget(modelIndex)
        return w.getActive()

    def get_color(self, item):
        ind = self.findIndexFromText("Curve Color")
        return item.child(ind, 1).text()

    def setCurveColor(self, txt):
        plot_item = self.default_plot
        plots = self.plot_lines[plot_item.plot_index]
        color = "tab:" + txt
        for line in plots:
            line.set_color(color)
        if self.kindParams != 'plate':
            self.add_legend()
        self.canvas.draw()

    def calculate(self):
        plot_item = self.default_plot
        eMin = self.get_e_min(plot_item)
        eMax = self.get_e_max(plot_item)
        nPoints = self.get_npoints(plot_item)
        xenergy = np.linspace(eMin, eMax, nPoints)
        if self.kindParams in ('mirror', 'crystal'):
            theta0 = self.get_theta0(plot_item)
        mat = self.beamLine.materialsDict.get(self.elementId)
        try:
            if isinstance(mat, rmats.Crystal):
                geometry = getattr(mat, 'geom', None)
                alpha = self.get_alpha(plot_item)
                if geometry is not None:
                    if geometry.startswith("B"):
                        gamma0 = -np.sin(theta0+alpha)
                        gammah = np.sin(theta0-alpha)
                    else:
                        gamma0 = -np.cos(theta0+alpha)
                        gammah = -np.cos(theta0-alpha)
                    hns0 = np.sin(alpha)*np.cos(theta0+alpha) -\
                        np.cos(alpha)*np.sin(theta0+alpha)
                    ampS, ampP = mat.get_amplitude(
                        xenergy, gamma0, gammah, hns0)
            elif isinstance(mat, rmats.Multilayer):
                ampS, ampP = mat.get_amplitude(xenergy, np.sin(theta0))
            elif self.kindParams == 'mirror':
                ampS, ampP = mat.get_amplitude(xenergy, np.sin(theta0))[0:2]
            elif self.kindParams == 'plate':
                ampS = mat.get_amplitude(xenergy, 0.5)[2]
                ampP = np.zeros_like(xenergy)
        except ValueError as e:
            print(e)
            ampS = np.zeros_like(xenergy)
            ampP = ampS
        self.on_calculation_result((xenergy, ampS, ampP, 0))

    def on_calculation_result(self, res_tuple):
        energy, curS, curP, plot_nr = res_tuple
        plot_item = self.model.item(plot_nr)
        plot_item.curves = energy, curS, curP
        lines = self.plot_lines[plot_item.plot_index]
        if self.kindParams == 'plate':
            lines[0].set_xdata(energy)
            lines[0].set_ydata(curS)
        else:
            selected = self.get_curve_types(plot_item)
            for line, label in zip(lines, self.allCurves):
                if label in selected:
                    line.set_xdata(energy)
                    if label == 'σ':
                        ydata = abs(curS)**2
                    elif label == 'π':
                        ydata = abs(curP)**2
                    line.set_ydata(ydata)
                line.set_visible(label in selected)
            self.add_legend()

        self.axes.set_xlim(energy[0], energy[-1])
        self.rescale_axes()
        self.canvas.draw()

    def rescale_axes(self):
        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(True)
        self.axes.relim()
        self.axes.autoscale_view()

    def export_curve(self):
        plot_item = self.default_plot
        fileName = re.sub(r'[^a-zA-Z0-9_\-.]+', '_', plot_item.text())
        options = qt.QFileDialog.Options()
        options |= qt.QFileDialog.ReadOnly
        file_name, _ = qt.QFileDialog.getSaveFileName(
            self, "Save File", fileName,
            "Text Files (*.txt);;All Files (*)", options=options)
        if not file_name:
            return

        lines = self.plot_lines[plot_item.plot_index]
        names = self.allCurves.keys()
        outLines, outNames = [], []
        for line, name in zip(lines, names):
            if line.get_visible():
                if len(outLines) == 0:
                    outLines.append(line.get_xdata())
                    outNames.append('energy')
                outLines.append(line.get_ydata())
                outNames.append(name)
        what = self.axes.get_ylabel()
        now = datetime.now()
        nowStr = now.strftime("%d/%m/%Y %H:%M:%S")
        header = f"{what} calculated by xrt on {nowStr}\n"
        header += "\t".join(outNames)
        np.savetxt(file_name, np.array(outLines).T, fmt='%#.7g',
                   delimiter='\t', header=header, encoding='utf-8')

    def on_item_clicked(self, index):
        while index.parent().isValid():
            index = index.parent()  # of plot_item
        if index.column() == 1:  # the empty cell near the plot title
            return
#        plot_item = self.model.itemFromIndex(index)
#        self.update_thetaB(plot_item)
        self.canvas.draw()

#     def update_legend(self, item):
#         plot_index = item.plot_index
#         for line, label, fwhm in zip(
#                 self.plot_lines[plot_index], self.allCurves.keys(),
#                 item.fwhms):
#             tt = ''
# #            convFactor = self.allUnits[self.get_units(item)]
# #            unitsStr = self.get_units(item)
# #            unit = self.allUnitsStr[unitsStr]
# #            sp = '' if unit == '°' else ' '
# #            tt = '' if fwhm is None else\
# #                ": {0:#.3g}{1}{2}".format(fwhm/convFactor, sp, unit)
#             line.set_label("{0} {1}{2}".format(item.text(), label, tt))
#         self.add_legend()


class SurfacePlotWidget(qt.QWidget):
    def __init__(self, parent=None, beamLine=None, elementId=None):
        super().__init__(parent)

        self.beamLine = beamLine
        self.elementId = elementId

        self.figure = Figure()
        self.canvas = qt.FigCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.toolbar = qt.NavigationToolbar(self.canvas, self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.surface = None
        self.cbar = None

        self.dims = [500, 500]
        self.update_surface(data=None, newPlot=True)

    def update_surface(self, data, newPlot=False):
        if self.beamLine is not None:
            surfObj = self.beamLine.fesDict.get(self.elementId)

        if surfObj is None:
            return

        xLim, yLim = surfObj.limPhysX, surfObj.limPhysY

        x = np.linspace(min(xLim), max(xLim), self.dims[0])
        y = np.linspace(min(yLim), max(yLim), self.dims[-1])

        xm, ym = np.meshgrid(x, y)

        z = surfObj.local_z_distorted(xm.flatten(), ym.flatten())

        if not newPlot:
            self.cbar.remove()
            self.cbar = None
            self.surface.remove()
            self.surface = None

        self.surface = self.ax.plot_surface(
            xm, ym, z.reshape(self.dims[-1], self.dims[0])*1e6,
            cmap="jet",
            linewidth=0,
            antialiased=True
        )

        self.cbar = self.figure.colorbar(
            self.surface,
            ax=self.ax,
            shrink=0.7,
            pad=0.1
        )
        self.cbar.set_label("Height [nm]")
        self.ax.set_title(f"{surfObj.name} height profile")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Z [nm]")

        self.canvas.draw_idle()
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:27:36 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import copy  # analysis:ignore
from collections import OrderedDict  # analysis:ignore
from ._widgets_qookbase import XrtQookBase  # analysis:ignore
from ._constants import _DEBUG_  # analysis:ignore
from ..commons import qt  # analysis:ignore
from ..xrtGlow import is_screen, is_aperture  # analysis:ignore
from ...backends import raycing  # analysis:ignore


class XrtQookElements(XrtQookBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

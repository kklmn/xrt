# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:05:03 2026

"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"

import sys  # analysis:ignore
import re  # analysis:ignore
import inspect  # analysis:ignore
from collections import OrderedDict  # analysis:ignore
from datetime import date  # analysis:ignore
from .qookelements import XrtQookElements  # analysis:ignore
from .._constants import path_to_xrt, myTab  # analysis:ignore

from ...commons import qt  # analysis:ignore
from ...commons import ext  # analysis:ignore
import xrt  #analysis:ignore
from ....backends import raycing  # analysis:ignore
from ....backends.raycing import sources as rsources  # analysis:ignore
from ....backends.raycing import screens as rscreens  # analysis:ignore
from ....backends.raycing import materials as rmats  # analysis:ignore
from ....backends.raycing import figure_error as rfe  # analysis:ignore
from ....backends.raycing import oes as roes  # analysis:ignore
from ....backends.raycing import apertures as rapts  # analysis:ignore
from ....backends.raycing import oes as roes  # analysis:ignore
from ....backends.raycing import run as rrun  # analysis:ignore
from ....version import __version__ as xrtversion  # analysis:ignore
from .... import plotter as xrtplot  # analysis:ignore
from .... import runner as xrtrun  # analysis:ignore
try:
    from ....backends.raycing.materials import elemental as rmatsel  # analysis:ignore
    from ....backends.raycing.materials import compounds as rmatsco  # analysis:ignore
    from ....backends.raycing.materials import crystals as rmatscr  # analysis:ignore
    pdfMats = True
except ImportError:
    pdfMats = False
    raise ImportError("no predef mats")

if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec


class XrtQook(XrtQookElements):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def generateCode(self):
        self.progressBar.setValue(0)
        self.progressBar.setFormat("Flattening structure.")
        for tree, item in zip([self.tree, self.matTree, self.feTree,
                               self.plotTree, self.runTree],
                              [self.rootBLItem, self.rootMatItem,
                               self.rootFEItem,
                               self.rootPlotItem, self.rootRunItem]):
            item.model().blockSignals(True)
#            self.flattenElement(tree, item)
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

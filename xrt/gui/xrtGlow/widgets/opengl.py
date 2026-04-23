# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:21:08 2026

"""

import os
import copy
import numpy as np
from functools import partial
from multiprocessing import Process, Queue
from collections import OrderedDict, deque
from matplotlib.colors import hsv_to_rgb

from .._constants import (msg_start, msg_stop, msg_exit, MAXRAYS, itemTypes,
                          scr_m, DEFAULT_SCENE_SETTINGS)
from .._utils import (generate_hsv_texture, create_qt_buffer, update_qt_buffer,
                      is_source, is_oe, is_aperture, is_screen, is_dcm, snsc)
from ..ogl import CoordinateBox, Beam3D, OEMesh3D

from ...commons import qt
from ...commons import gl

from ....backends import raycing
from ....backends.raycing import (propagationProcess, renderOnlyArgSet,
                                  orientationArgSet, shapeArgSet, EpicsDevice)
from ....backends.raycing import sources as rsources
from ....backends.raycing import screens as rscreens

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "27 Jan 2026"


class xrtGlWidget(qt.QOpenGLWidget):
    rotationUpdated = qt.Signal(np.ndarray)
    scaleUpdated = qt.Signal(np.ndarray)
    beamUpdated = qt.Signal(tuple)
    colorsUpdated = qt.Signal()
    oePropsUpdated = qt.Signal(tuple)
    updateQookTree = qt.Signal(tuple)
    histogramUpdated = qt.Signal(tuple)
    openElViewer = qt.Signal(str)

    def __init__(self,
                 parent=None,
                 arrayOfRays=None,
                 modelRoot=None,
                 oesList=None,
                 b2els=None,
                 signal=None,
                 beamLayout=None,
                 epicsPrefix=None,
                 epicsMap={}):
        super().__init__(parent=parent)
        self.parent = parent
        self.hsvTex = generate_hsv_texture(512, s=1.0, v=1.0)
        self.QookSignal = signal

        self.beamline = raycing.BeamLine()
        self.loopRunning = False
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.needMeshUpdate = deque()
        self.needBeamUpdate = deque()
        self.deletionQueue = deque()

        for pName, pValue in DEFAULT_SCENE_SETTINGS.items():
            if pName in ['scaleVec', 'rotations', 'tmpOffset', 'tVec',
                         'coordOffset']:
                pValue = np.array(pValue)
            setattr(self, pName, pValue)

        self.cameraTarget = qt.QVector3D(0., 0., 0.)

        self.showOeLabels = False
        self.enableBlending = True
        self.newColorAxis = True
        self.showVirtualScreen = False
        self.virtBeam = None
        self.virtDotsArray = None
        self.virtDotsColor = None
        self.vScreenForColors = False
        self.globalColorIndex = None
        self.isVirtScreenNormal = False
        self.vScreenManualSize = False
        self.vScreenSize = 0.5
        self.setMinimumSize(240, 400)

        self.setMouseTracking(True)
        self.surfCPOrder = 4
        self.oesToPlot = []
        self.labelsToPlot = []

        self.meshDict = OrderedDict()
        self.beamBufferDict = OrderedDict()

        self.arrayOfRays = arrayOfRays
        self.segmentModel = modelRoot
        self.oesList = oesList
        self.beamsToElements = b2els

        if arrayOfRays is not None:
            self.renderingMode = 'static'
            self.epicsPrefix = None
            self.epicsMap = {}
            self.beamline.flow = arrayOfRays[0]
            for eluuid, elline in arrayOfRays[2].items():
                self.beamline.oesDict[elline[0].uuid] = elline
                for flowLine in arrayOfRays[0]:
                    beamDict = self.beamline.beamsDictU[eluuid] if \
                        eluuid in self.beamline.beamsDictU else {}
                    if flowLine[0] == eluuid:
                        if flowLine[0] != flowLine[2]:
                            if 'beamGlobal' not in beamDict:
                                beamDict['beamGlobal'] =\
                                    arrayOfRays[1][flowLine[1]]
                            if is_dcm(elline[0]):
                                if 'beamLocal2' not in beamDict:
                                    beamDict['beamLocal2'] =\
                                        arrayOfRays[1][flowLine[1]]
                            else:
                                if 'beamLocal' not in beamDict:
                                    beamDict['beamLocal'] =\
                                        arrayOfRays[1][flowLine[1]]
                        else:
                            if 'beamLocal1' not in beamDict:
                                beamDict['beamLocal1'] =\
                                    arrayOfRays[1][flowLine[1]]
                    elif flowLine[2] == eluuid:
                        if is_screen(elline[0]) or is_aperture(elline[0]):
                            beamDict['beamLocal'] = arrayOfRays[1][flowLine[3]]
                    self.beamline.beamsDictU[eluuid] = beamDict
        elif beamLayout is not None:
            self.renderingMode = 'dynamic'
            self.epicsPrefix = epicsPrefix
            self.epicsMap = epicsMap

            self.beamline.deserialize(beamLayout)
            self.beamline.flowSource = 'Qook'

            self.calc_process = Process(
                    target=propagationProcess,
                    args=(self.input_queue, self.output_queue))
            self.calc_process.start()

            msg_init_bl = {
                    "command": "create",
                    "object_type": "beamline",
                    "kwargs": beamLayout
                    }
            self.input_queue.put(msg_init_bl)

            self._progressTimerSlot = partial(self.check_progress,
                                              self.output_queue)
            self.timer = qt.QTimer(self)
            self.timer.timeout.connect(self._progressTimerSlot)
            self.timer.start(10)  # Adjust the interval as needed

            for oeid, meth in self.beamline.flowU.items():
                oe = self.beamline.oesDict[oeid][0]
                for func, fkwargs in meth.items():
                    getattr(oe, func)(**fkwargs)

        self.virtScreen = {'uuid': rscreens.Screen(
                            bl=self.beamline,
                            limPhysX=[-10, 10],
                            limPhysY=[-10, 10],
                            name="VirtualScreen").uuid,
                           'beamStart': None,
                           'beamEnd': None,
                           'beamPlane': None,
                           'offsetOn': False,
                           'offset': np.zeros(3),
                           'center': np.zeros(3)}

        self.oeContour = dict()
        self.slitEdges = dict()

        self.contourWidth = 2
        self.arrowSize = [0.4, 0.05, 0.025, 13]  # length, tip length, tip R
        self.labelLines = None

        self.prevMPos = [0, 0]
        self.prevWC = np.float32([0, 0, 0])
        self.coordinateGridLineWidth = 1
        self.cBoxLineWidth = 1
#        self.useScalableFont = False
        self.scalableFontType = "Sans-serif"
        self.scalableFontWidth = 1
        self.useFontAA = False

        self.cameraPos = qt.QVector3D(self.cameraDistance, 0, 0)
        self.upVec = qt.QVector3D(0., 0., 1.)

        self.mView = qt.QMatrix4x4()
        self.mView.lookAt(self.cameraPos,
                          self.cameraTarget,
                          self.upVec)

        self.orthoScale = 2.0 * self.cameraDistance * np.tan(
            np.radians(self.cameraAngle*0.5))

        self.mProj = qt.QMatrix4x4()
        self.mProj.setToIdentity()
        if self.perspectiveEnabled:
            self.mProj.perspective(self.cameraAngle, self.aspect, 0.01, 1000)
        else:
            halfHeight = self.orthoScale * 0.5
            halfWidth = halfHeight * self.aspect
            self.mProj.ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                             0.01, 1000)

        self.mModScale = qt.QMatrix4x4()
        self.mModTrans = qt.QMatrix4x4()
        self.mModScale.scale(*(self.scaleVec/self.maxLen))
        self.mModTrans.translate(*(self.tVec-self.coordOffset))
        self.mMod = self.mModScale*self.mModTrans

        self.mModAx = qt.QMatrix4x4()
        self.mModAx.setToIdentity()

        self.fixProj = qt.QMatrix4x4()
        self.fixProj.perspective(self.cameraAngle*1.5, 1, 0.001, 1000)

        pModelT = np.identity(4)
        self.visibleAxes = np.argmax(np.abs(pModelT), axis=1)
#        self.signs = np.ones_like(pModelT)

        self.showHelp = False
        self.beamVAO = dict()
        self.uuidDict = dict()
        self.selectableOEs = {}
        self.selectedOE = 0
        self.isColorAxReady = False
        self.makeCurrent()
        if self.epicsPrefix is not None and self.renderingMode == 'dynamic':
            try:
                os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
                os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
                self.epicsInterface = EpicsDevice(
                        bl=self.beamline,
                        epicsPrefix=epicsPrefix,
                        epicsMap=epicsMap,
                        callback=self.update_beamline_async)
#                self.build_epics_device(epicsPrefix, softioc, builder,
#                                        asyncio_dispatcher)
            except ImportError:
                print("pythonSoftIOC not installed")
                self.epicsPrefix = None

        self.colorsUpdated.connect(self.getColorLimits)
        self.oePropsUpdated.connect(self.update_oe_transform)

#        self.getColorLimits()

    @property
    def colorAxis(self):
        return self._colorAxis

    @colorAxis.setter
    def colorAxis(self, colorAxis):
        funcStr = 'get_{}'.format(colorAxis)
        if hasattr(raycing, funcStr):
            self._colorAxis = colorAxis
            self.getColor = getattr(raycing, funcStr)

    @property
    def globalColors(self):
        return self._globalColors

    @globalColors.setter
    def globalColors(self, globalColors):
        self._globalColors = globalColors
        self.newColorAxis = True
        self.change_beam_colorax()

    @property
    def oeThickness(self):
        return self._oeThickness

    @oeThickness.setter
    def oeThickness(self, oeThickness):
        self._oeThickness = oeThickness
        self.queue_mesh_update()

    @property
    def oeThicknessForce(self):
        return self._oeThicknessForce

    @oeThicknessForce.setter
    def oeThicknessForce(self, oeThicknessForce):
        self._oeThicknessForce = oeThicknessForce
        self.queue_mesh_update()

    @property
    def slitThicknessFraction(self):
        return self._slitThicknessFraction

    @slitThicknessFraction.setter
    def slitThicknessFraction(self, slitThicknessFraction):
        self._slitThicknessFraction = slitThicknessFraction
        self.queue_mesh_update()

    @property
    def autoSizeOe(self):
        return self._autoSizeOe

    @autoSizeOe.setter
    def autoSizeOe(self, autoSizeOe):
        self._autoSizeOe = autoSizeOe
        self.needMeshUpdate.extend(
            list(self.beamline.oesDict.keys()))

    def queue_mesh_update(self, oeuuid=None):
        if raycing.is_valid_uuid(oeuuid):
            self.needMeshUpdate.append(oeuuid)
        else:
            for oeid, oeLine in self.beamline.oesDict.items():
                if is_oe(oeLine[0]) and\
                        oeid not in self.needMeshUpdate:
                    self.needMeshUpdate.append(oeid)

    @property
    def invertColors(self):
        return self._invertColors

    @invertColors.setter
    def invertColors(self, invertColors):
        self._invertColors = invertColors
        self.set_scene_colors()

    async def update_beamline_async(self, oeid, argName, argValue):
        """Update from EPICS interface """
        self.update_beamline(oeid, {argName: argValue}, sender="epics")

    def update_epics_record(self, oeid, kwargs):
        """Update EPICS record if the value was changed from Qook or Explorer
        """
        if self.epicsPrefix is not None:
            elementBase = self.epicsInterface.pv_map.get(oeid)
            if elementBase is not None:
                for key, val in kwargs.items():
                    if key == 'center':
                        for argVal, field in zip(val, ['x', 'y', 'z']):
                            record = elementBase.get(f'center.{field}')
                            if record is not None:
                                record.set(argVal)
                    elif key in ['limPhysX', 'limPhysY', 'limPhysX2',
                                 'limPhysY2']:
                        for argVal, field in zip(val, ['lmin', 'lmax']):
                            record = elementBase.get(f'{key}.{field}')
                            if record is not None:
                                record.set(argVal)
#                    elif key == 'opening':
#                        oeObj = self.beamline.oesDict[oeid][0]
#                        for argVal, field in zip(val, oeObj.kind):
#                            record = elementBase.get(f'opening.{field}')
#                            if record is not None:
#                                record.set(argVal)
                    elif key == 'blades':
                        for field, argVal in val.items():
                            record = elementBase.get(f'blades.{field}')
                            if record is not None:
                                record.set(argVal)
                    else:
                        record = elementBase.get(key)
                        if record is not None:
                            record.set(val)

    def update_beamline(self, oeid, kwargs, sender="gui"):  # one OE at a time
        if '_object' in kwargs:  # only Qook can create new elements for now
            # new element
            if 'material' in kwargs['_object']:
                self.beamline.init_material_from_json(oeid, kwargs)
                message = {"command": "create",
                           "object_type": "mat",
                           "uuid": oeid,
                           "kwargs": kwargs.copy()
                           }
                if hasattr(self, 'input_queue'):
                    self.input_queue.put(message)
            elif 'figure' in kwargs['_object']:
                self.beamline.init_fe_from_json(oeid, kwargs)
                message = {"command": "create",
                           "object_type": "fe",
                           "uuid": oeid,
                           "kwargs": kwargs.copy()
                           }
                if hasattr(self, 'input_queue'):
                    self.input_queue.put(message)
            elif 'properties' in kwargs:
                if sender != 'Qook':  # Already called in Qook.update_beamline
                    self.beamline.init_oe_from_json(kwargs)
                if oeid not in self.needMeshUpdate:
                    self.needMeshUpdate.append(oeid)
                self.glDraw()
                if self.parent is not None:
                    self.parent.addElementToModel(oeid)
                try:
                    self.getMinMax()
                except:
                    pass

                self.maxLen = np.max(np.abs(
                        self.minmax[0, :] - self.minmax[1, :]))
                self.parent.updateMaxLenFromGL(self.maxLen)  # TODO: replace with signal

                message = {"command": "create",
                           "object_type": "oe",
                           "uuid": oeid,
                           "kwargs": kwargs.copy()
                           }

                if hasattr(self, 'input_queue'):
                    self.input_queue.put(message)

            elif 'parameters' in kwargs:  # update flow
                methStr = kwargs['_object'].split('.')[-1]
                if sender != 'Qook':  # Already called in Qook.update_beamline
                    self.beamline.update_flow_from_json(
                            oeid, {methStr: kwargs})
                    self.beamline.sort_flow()
                if self.parent is not None:
                    self.parent.updateTargets()
                message = {"command": "flow",
                           "uuid": oeid,
                           "kwargs": {methStr: kwargs.copy()}
                           }
                if hasattr(self, 'input_queue'):
                    self.input_queue.put(message)

            return

        for argName, argValue in kwargs.items():
            if isinstance(argValue, str):
                argValue = raycing.parametrize(argValue)
                kwargs[argName] = argValue

            if oeid is None:
                if self.epicsPrefix is not None:
                    if argName == 'Acquire':
                        self.epicsInterface.pv_records['AcquireStatus'].set(1)
                        if str(argValue) == '1':
                            if hasattr(self, 'input_queue'):
                                self.input_queue.put({
                                            "command": "run_once",
                                            "object_type": "beamline"
                                            })
                    elif argName == 'AutoUpdate':
                        if hasattr(self, 'input_queue'):
                            self.input_queue.put({
                                        "command": "auto_update",
                                        "object_type": "beamline",
                                        "kwargs": {"value": int(argValue)}
                                        })
                return

            if oeid in self.beamline.oesDict:
                obj_type = "oe"
                elLine = self.beamline.oesDict.get(oeid)
                updObj = elLine[0]
            elif oeid in self.beamline.materialsDict:
                obj_type = "mat"
                updObj = self.beamline.materialsDict.get(oeid)
            elif oeid in self.beamline.fesDict:
                obj_type = "fe"
                updObj = self.beamline.fesDict.get(oeid)
            else:
                obj_type = None

            if obj_type is None:
                return

            argComps = argName.split('.')
            arg0 = argComps[0]
            if len(argComps) > 1 and arg0 == 'blades':
                field = argComps[-1]
                bladesValue = dict(getattr(updObj, 'blades', {}) or {})
                bladesValue[field] = argValue
                argValue = bladesValue
            elif len(argComps) > 1:  # compound args: center, limits, opening
                field = argComps[-1]
                if field == 'energy':
                    argValue = raycing.format_energy_input(argValue)
                else:
                    argIn = getattr(updObj, f'_{arg0}', None)
                    arrayValue = getattr(updObj,
                                         arg0) if argIn is None else argIn

                    # avoid writing string to numpy array
                    if hasattr(arrayValue, 'tolist'):
                        arrayValue = arrayValue.tolist()
                    elif isinstance(arrayValue, tuple):
                        arrayValue = list(arrayValue)

                    for fList in raycing.compoundArgs.values():
                        if field in fList:
                            idx = fList.index(field)
                            break
                    arrayValue[idx] = argValue
                    argValue = arrayValue

            elif any(arg0.lower().startswith(v) for v in
                     ['mater', 'tlay', 'blay', 'coat', 'substrate']):
                if not raycing.is_valid_uuid(argValue):
                    # objects need material uuid rather than name
                    argValue = self.beamline.matnamesToUUIDs.get(argValue)
                    kwargs[arg0] = argValue
            elif any(arg0.lower().startswith(v) for v in
                     ['figureerr', 'basefe']):
                if not raycing.is_valid_uuid(argValue):
                    # objects need fe uuid rather than name
                    argValue = self.beamline.fenamesToUUIDs.get(argValue)
                    kwargs[arg0] = argValue

            # updating local beamline tree here
            setattr(updObj, arg0, argValue)

            if sender == 'OEE':
                self.updateQookTree.emit((oeid, {arg0: argValue}))
            if obj_type == "oe":
                if arg0.lower().startswith('center'):
                    flow = copy.deepcopy(self.beamline.flowU)
                    self.beamline.sort_flow()
                    if self.parent is not None and flow != self.beamline.flowU:
                        print("Flow updated. Updating targets")
                        self.parent.updateTargets()

                skipUpdate = False
                if arg0 == 'center':
                    skipUpdate = 'auto' in str(getattr(updObj, '_center',
                                                      updObj.center))
                elif arg0 in ['pitch', 'bragg']:
                    skipUpdate = raycing.is_auto_align_value(
                        getattr(updObj, f'_{arg0}', None))

                if arg0 in orientationArgSet and not skipUpdate:
                    self.meshDict[oeid].update_transformation_matrix()
                    self.getMinMax()
                    self.maxLen = np.max(np.abs(
                            self.minmax[0, :] - self.minmax[1, :]))
                    self.parent.updateMaxLenFromGL(self.maxLen)
                    if arg0 in ['pitch'] and hasattr(updObj, 'reset_pq'):
                        if oeid not in self.needMeshUpdate:
                            self.needMeshUpdate.append(oeid)
                elif arg0 in shapeArgSet:
                    if oeid not in self.needMeshUpdate:
                        self.needMeshUpdate.append(oeid)
                elif arg0 in {'name'}:
                    if self.parent is not None:
                        self.parent.updateNames()

                if arg0 in renderOnlyArgSet:
                    self.glDraw()

                # updating the beamline model in the runner
            if self.epicsPrefix is not None:
                self.epicsInterface.pv_records['AcquireStatus'].set(1)

        message = {"command": "modify",
                   "object_type": obj_type,
                   "uuid": oeid,
                   "kwargs": kwargs.copy()
                   }

        if hasattr(self, 'input_queue'):
            self.input_queue.put(message)

        if sender == "gui":
            self.update_epics_record(oeid, kwargs)

    def init_shaders(self):
        shaderBeam = qt.QOpenGLShaderProgram()
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, Beam3D.vertex_source)
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Geometry, Beam3D.geometry_source)
        shaderBeam.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, Beam3D.fragment_source)
        gl.glGetError()
        if not shaderBeam.link():
            print("Linking Error", str(shaderBeam.log()))
            print('shaderBeam: Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderBeam: Done!')
        self.shaderBeam = shaderBeam
        gl.glGetError()
        shaderFootprint = qt.QOpenGLShaderProgram()
        shaderFootprint.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, Beam3D.vertex_source_point)
        shaderFootprint.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, Beam3D.fragment_source_point)
        gl.glGetError()
        if not shaderFootprint.link():
            print("Linking Error", str(shaderFootprint.log()))
            print('shaderFootprint: Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderFootprint: Done!')
        self.shaderFootprint = shaderFootprint
        gl.glGetError()
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
        gl.glGetError()
        if not shaderMesh.link():
            print("Linking Error", str(shaderMesh.log()))
            print('shaderMesh: Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderMesh: Done!')
        self.shaderMesh = shaderMesh
        gl.glGetError()
        shaderMag = qt.QOpenGLShaderProgram()
        shaderMag.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, OEMesh3D.vertex_magnet)
        shaderMag.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, OEMesh3D.fragment_magnet)
        gl.glGetError()
        if not shaderMag.link():
            print("Linking Error", str(shaderMag.log()))
            print('shaderMag: Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderMag: Done!')
        self.shaderMag = shaderMag
        gl.glGetError()
        shaderGeo = qt.QOpenGLShaderProgram()
        shaderGeo.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, OEMesh3D.vertex_contour)
        shaderGeo.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, OEMesh3D.fragment_contour)
        gl.glGetError()
        if not shaderGeo.link():
            print("Linking Error", str(shaderGeo.log()))
            print('shaderMag: Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderMag: Done!')
        self.shaderGeo = shaderGeo
        gl.glGetError()


    def init_coord_grid(self):
        self.cBox = CoordinateBox(self)
        shaderCoord = qt.QOpenGLShaderProgram()
        shaderCoord.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.vertex_source)
        shaderCoord.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.fragment_source)
        gl.glGetError()
        if not shaderCoord.link():
            print("Linking Error", str(shaderCoord.log()))
            print('Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderCoord: Done!')
        self.cBox.shader = shaderCoord
        gl.glGetError()
        shaderText = qt.QOpenGLShaderProgram()
        shaderText.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.text_vertex_code)
        shaderText.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.text_fragment_code)
        gl.glGetError()
        if not shaderText.link():
            print("Linking Error", str(shaderText.log()))
            print('Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderText: Done!')
        self.cBox.textShader = shaderText
        gl.glGetError()
        origShader = qt.QOpenGLShaderProgram()
        origShader.addShaderFromSourceCode(
                qt.QOpenGLShader.Vertex, self.cBox.orig_vertex_source)
        origShader.addShaderFromSourceCode(
                qt.QOpenGLShader.Fragment, self.cBox.orig_fragment_source)
        gl.glGetError()
        if not origShader.link():
            print("Linking Error", str(origShader.log()))
            print('Failed to link dummy renderer shader!')
        else:
            pass
#            print('\nshaderArrow: Done!')
        self.cBox.origShader = origShader
        self.cBox.prepare_grid()
        self.cBox.prepare_arrows(*self.arrowSize)  # in model space
        gl.glGetError()

    def generate_beam_texture(self, width):
        hsv_texture_data = generate_hsv_texture(width, s=1.0, v=1.0)

        self.beamTexture = qt.QOpenGLTexture(qt.QOpenGLTexture.Target1D)
        self.beamTexture.create()
        self.beamTexture.setSize(width)  # Width of the texture
        self.beamTexture.setFormat(qt.QOpenGLTexture.RGB8_UNorm)
        self.beamTexture.allocateStorage()

        # Upload data (convert numpy array to raw bytes)
        self.beamTexture.setData(
            qt.QOpenGLTexture.RGB,                 # Pixel format
            qt.QOpenGLTexture.UInt8,               # Pixel type
            hsv_texture_data.tobytes()          # Raw data as bytes
        )

    def build_histRGB(self, beam, limits=None, isScreen=False,
                      bins=[256, 256]):
        good = (beam.state == 1) | (beam.state == 2)
        if isScreen:
            x, y, z = beam.x[good], beam.z[good], beam.y[good]
        else:
            x, y, z = beam.x[good], beam.y[good], beam.z[good]

        if limits is None:
            limits = np.array([[np.min(x), np.max(x)],
                               [np.min(y), np.max(y)],
                               [np.min(z), np.max(z)]])
            beamLimits = [limits[0, :], limits[1, :]]
        else:
            beamLimits = [limits[:, 1], limits[:, 0]]

        flux = beam.Jss[good] + beam.Jpp[good]

#        cData = self.getColorData(beam, beamTag)[good]
        cData = self.getColor(beam)[good]
#        colorMin =
        cData01 = ((cData - self.colorMin) * 0.85 /
                   (self.colorMax - self.colorMin)).reshape(-1, 1)

        cDataHSV = np.dstack(
            (cData01, np.ones_like(cData01) * 0.85,
             flux.reshape(-1, 1)))
        cDataRGB = (hsv_to_rgb(cDataHSV)).reshape(-1, 3)

        hist2dRGB = np.zeros((bins[0], bins[1], 3), dtype=np.float64)
        hist2d = None
        if len(beam.x[good]) > 0:
            for i in range(3):  # over RGB components
                hist2dRGB[:, :, i], yedges, xedges = np.histogram2d(
                    y, x, bins=bins, range=beamLimits,
                    weights=cDataRGB[:, i])

        hist2dRGB /= np.max(hist2dRGB)
        hist2dRGB = np.uint8(hist2dRGB*255)

        return hist2d, hist2dRGB, limits

    def toggleLoop(self):
        if self.loopRunning:
            self.input_queue.put(msg_stop)
            self.loopRunning = False
        else:
            self.input_queue.put(msg_start)
            self.loopRunning = True

    def check_progress(self, progress_queue):

        while not progress_queue.empty():
            msg = progress_queue.get()
            if 'beam' in msg:
                for beamKey, beam in msg['beam'].items():
                    self.needBeamUpdate.append((msg['sender_id'], beamKey))
                    self.beamline.beamsDictU[msg['sender_id']][beamKey] = beam
                    self.beamUpdated.emit((msg['sender_id'], beamKey))
            elif 'histogram' in msg and self.epicsPrefix is not None:
                record = self.epicsInterface.pv_map.get(msg['sender_id'],
                                                        {}).get('image')
                if record is not None:
                    imgHist = np.flipud(msg['histogram'])
                    record.set(imgHist.flatten())

            elif 'repeat' in msg:
                print("Total repeats:", msg['repeat'])
                if self.epicsPrefix is not None:
                    self.epicsInterface.pv_records['AcquireStatus'].set(0)
                self.colorsUpdated.emit()
                if self.QookSignal is not None:
                    self.QookSignal.emit((1., "Propagation complete"))
            elif 'pos_attr' in msg:  # TODO: Update epics rbv
                oeLine = self.beamline.oesDict.get(msg['sender_id'])
                if oeLine is not None:
                    setattr(oeLine[0],
                            f"{msg['pos_attr']}" if msg['pos_attr'] in
                            ['footprint'] else f"_{msg['pos_attr']}Val",
                            msg['pos_value'])
                    if hasattr(oeLine[0], 'get_orientation'):
                        oeLine[0].get_orientation()
                if msg['pos_attr'] in ['footprint']:
                    if self.autoSizeOe:
                        self.needMeshUpdate.append(msg['sender_id'])
                else:
                    self.oePropsUpdated.emit((msg['sender_id'],
                                              msg['pos_attr'],
                                              msg['pos_value']))
            elif 'diag_attr' in msg:
                self.oePropsUpdated.emit((msg['sender_id'],
                                          msg['diag_attr'],
                                          msg['diag_value']))
            elif 'progress' in msg and self.QookSignal is not None:
                    self.QookSignal.emit((msg['progress'],
                                          "Running propagation"))
#            elif 'depend_attr' in msg:
#                self.oePropsUpdated.emit((msg['sender_id'],
#                                          msg['depend_attr'],
#                                          msg['depend_value']))
#                    self.meshDict[msg['sender_id']].update_transformation_matrix()
#                    try:
#                        self.getMinMax()
#                        self.maxLen = np.max(np.abs(
#                                self.minmax[0, :] - self.minmax[1, :]))
#                        self.parent.updateMaxLenFromGL(self.maxLen)
#                    except TypeError:
#                        print("Cannot find limits")

#                if self.epicsPrefix is not None:
#                    self.epicsInterface.pv_records['AcquireStatus'].set(0)
#                self.glDraw()

    def close_calc_process(self):
        timer = getattr(self, 'timer', None)
        if timer is not None:
            timer.stop()
            try:
                timer.timeout.disconnect(getattr(self, '_progressTimerSlot',
                                                 None))
            except Exception:
                pass
            timer.deleteLater()
            self.timer = None

        if hasattr(self, 'calc_process') and\
                self.calc_process is not None:
            try:
                self.input_queue.put(msg_exit)
            except Exception:
                pass
            self.calc_process.join(timeout=1)
            if self.calc_process.is_alive():
                self.calc_process.terminate()
                self.calc_process.join()
            self.calc_process = None

        for queue_name in ['input_queue', 'output_queue']:
            queue_obj = getattr(self, queue_name, None)
            if queue_obj is None:
                continue
            try:
                queue_obj.close()
            except Exception:
                pass
            try:
                queue_obj.cancel_join_thread()
            except Exception:
                pass
            setattr(self, queue_name, None)

    def cleanup_gl_resources(self):
        if getattr(self, '_glResourcesCleaned', False):
            return
        if self.context() is None:
            return

        try:
            self.makeCurrent()
        except Exception:
            return

        try:
            for beamTag in list(self.beamBufferDict):
                self.delete_beam_footprint(beamTag)
                del self.beamBufferDict[beamTag]

            for meshId, mesh in list(self.meshDict.items()):
                if mesh is not None:
                    mesh.delete_mesh()
                del self.meshDict[meshId]

            if hasattr(self, 'cBox') and self.cBox is not None:
                self.cBox.cleanup_gl_resources()
                self.cBox = None

            if hasattr(self, 'llVBO') and self.llVBO is not None:
                self.llVBO.destroy()
                self.llVBO = None

            if hasattr(self, 'labelvao') and self.labelvao is not None:
                self.labelvao.destroy()
                self.labelvao = None

            if hasattr(self, 'beamTexture') and self.beamTexture is not None:
                self.beamTexture.destroy()
                self.beamTexture = None

            self._glResourcesCleaned = True
        finally:
            self.doneCurrent()

#    def generate_hist_texture(self, oe, beam, is2ndXtal=False):
#        nsIndex = int(is2ndXtal)
#        if not hasattr(oe, 'mesh3D'):
#            return
#        meshObj = oe.mesh3D
#        lb = rsources.Beam(copyFrom=beam)
#
#        beamLimits = oe.footprint if hasattr(oe, 'footprint') else None
#
#        histAlpha, hist2dRGB, beamLimits = self.build_histRGB(
#                lb, beam, beamLimits[nsIndex])
#
#        texture = qt.QImage(hist2dRGB, 256, 256, qt.QImage.Format_RGB888)

#        texture.save(str(oe.name)+"_beam_hist.png")
#        if hasattr(meshObj, 'beamTexture'):
#            oe.beamTexture.setData(texture)
#        meshObj.beamTexture[nsIndex] = qg.QOpenGLTexture(texture)
#        meshObj.beamLimits[nsIndex] = beamLimits
#        self.glDraw()

    def init_beam_footprint(self, beam=None, beamTag=None):
        """beamTag: ('oeuuid', 'beamKey') """
        if beam is None:
            beam = self.beamline.beamsDictU[beamTag[0]][beamTag[1]]
            if beam is None:
                return

        data = np.column_stack((beam.x, beam.y, beam.z)).copy()
        dataColor = self.getColorData(beam, beamTag).copy()
#        dataColor = self.getColor(beam).copy()
        state = np.where((
                (beam.state == 1) | (beam.state == 2)), 1, 0).copy()
        intensity = np.float32(beam.Jss+beam.Jpp).copy()

        vbo = {}

        vbo['position'] = create_qt_buffer(data)
        vbo['color'] = create_qt_buffer(dataColor)
        vbo['state'] = create_qt_buffer(state)
        vbo['intensity'] = create_qt_buffer(intensity)
#        goodRays = np.where(((state > 0) & (intensity/self.iMax >
#                             self.cutoffI)))[0]

        goodRays = np.where((state > 0))[0]
#        lowIntRays = np.where((intensity/self.iMax > self.cutoffI))[0]
        oeObj = self.beamline.oesDict.get(beamTag[0])
        if oeObj is not None and hasattr(oeObj[0], 'lostNum'):
            lostRays = np.where((beam.state == oeObj[0].lostNum))[0]
        else:
            lostRays = np.array([])
#        if beamTag[0] == self.virtScreen:
#            print(data, dataColor, state, intensity, goodRays, len(goodRays))

        vbo['indices_lost'] = create_qt_buffer(beam.state.copy(), isIndex=True)
        update_qt_buffer(vbo['indices_lost'], lostRays.copy(), isIndex=True)
        vbo['lostLen'] = len(lostRays)

        vbo['indices'] = create_qt_buffer(beam.state.copy(), isIndex=True)
        update_qt_buffer(vbo['indices'], goodRays.copy(), isIndex=True)
        vbo['goodLen'] = len(goodRays)

        gl.glGetError()
        vao = qt.QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()

        vbo['position'].bind()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)  # Attribute 0: position
        vbo['position'].release()

        vbo['color'].bind()
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)  # Attribute 1: colorAxis
        vbo['color'].release()

        vbo['state'].bind()
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)  # Attribute 2: state
        vbo['state'].release()

        vbo['intensity'].bind()
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(3)  # Attribute 3: intensity
        vbo['intensity'].release()

        vao.release()

        vboStore = {}
        vboStore['beamLen'] = min(MAXRAYS, len(beam.x))

        vboStore['vbo'] = vbo
        vboStore['vao'] = vao

        if not hasattr(beam, 'iMax'):
            beam.iMax = np.max(beam.Jss[goodRays] + beam.Jpp[goodRays]) if\
                len(goodRays) > 0 else 0

        vboStore['iMax'] = beam.iMax

        self.beamBufferDict[beamTag] = vboStore

    def delete_beam_footprint(self, beamTag):
        beamBuffer = self.beamBufferDict.get(beamTag)
        if beamBuffer is None:
            return

        vbo = beamBuffer.get('vbo')
        if vbo is not None:
            for buffKey in ['position', 'color', 'state',
                            'intensity', 'indices', 'indices_lost']:
                buff = vbo.get(buffKey)
                if buff is not None:
                    buff.destroy()
                    gl.glGetError()
                buff = None
            vbo.clear()

        vao = beamBuffer.get('vao')
        if vao is not None:
            vao.destroy()
            gl.glGetError()
#            vao.clear()

        beamBuffer.clear()

    def rebuild_selectable_oes(self):
        self.selectableOEs.clear()
        self.selectedOE = 0
        for oeuuid in self.beamline.oesDict:
            mesh = self.meshDict.get(oeuuid)
            if mesh is None:
                continue
            stencilNum = len(self.selectableOEs) + 1
            self.selectableOEs[stencilNum] = oeuuid
            mesh.stencilNum = stencilNum

    def delete_all_oe_buffers(self, oeid):
        for beamTag in list(self.beamBufferDict):
            if beamTag[0] == oeid:
                self.delete_beam_footprint(beamTag)
                del self.beamBufferDict[beamTag]

#        mesh = self.meshDict.get(oeid)
#        if mesh is not None:
#            mesh.delete_mesh()
#            del self.meshDict[oeid]  # what about mesh.oe?

    def delete_object(self, objuuid):  # TODO: to be triggered by a signal after deleting the buffers
        try:
            objType = None
            if objuuid in self.beamline.oesDict:
                mesh = self.meshDict.get(objuuid)
                if mesh is not None:
                    mesh.delete_mesh()
                del self.meshDict[objuuid]
                self.beamline.delete_oe_by_id(objuuid)
                if self.parent is not None:
                    self.parent.updateTargets()
                objType = "oe"
            elif objuuid in self.beamline.materialsDict:
                self.beamline.delete_mat_by_id(objuuid)
                objType = "mat"
            elif objuuid in self.beamline.fesDict:
                self.beamline.delete_fe_by_id(objuuid)
                objType = "fe"

            if objType is not None:
                message = {"command": "delete",
                           "object_type": objType,
                           "uuid": objuuid
                           }

                if hasattr(self, 'input_queue'):
                    self.input_queue.put(message)
        except:
            raise

    def update_beam_footprint(self, beam=None, beamTag=None):
        if beam is None:
            beam = self.beamline.beamsDictU[beamTag[0]][beamTag[1]]

        if self.beamBufferDict.get(beamTag) is None:
            print("Buffers require init")
            self.init_beam_footprint(beam, beamTag)
            return

        beamvbo = self.beamBufferDict[beamTag]['vbo']
        data = np.dstack((beam.x, beam.y, beam.z)).copy()
        dataColor = self.getColorData(beam, beamTag)
        state = np.where((
                (beam.state == 1) | (beam.state == 2)), 1, 0).copy()
        intensity = np.float32(beam.Jss+beam.Jpp).copy()
        oeObj = self.beamline.oesDict.get(beamTag[0])
        if oeObj is not None and hasattr(oeObj[0], 'lostNum'):
            lostRays = np.where((beam.state == oeObj[0].lostNum))[0]
        else:
            lostRays = np.array([])
        update_qt_buffer(beamvbo['indices_lost'],
                         lostRays.copy(), isIndex=True)
        goodRays = np.where((state > 0))[0]
        update_qt_buffer(beamvbo['position'], data)
        update_qt_buffer(beamvbo['color'], dataColor)
        update_qt_buffer(beamvbo['state'], state)
        update_qt_buffer(beamvbo['intensity'], intensity)
        update_qt_buffer(beamvbo['indices'], goodRays.copy(), isIndex=True)
        beamvbo['goodLen'] = len(goodRays)
        beamvbo['lostLen'] = len(lostRays)

        beam.iMax = np.max(beam.Jss[goodRays] + beam.Jpp[goodRays]) if\
            len(goodRays) > 0 else 0

        beamvbo['iMax'] = beam.iMax

    def render_beam(self, beamTag, model, view, projection, target=None):
        """beam: ('oeuuid', 'beamKey') """
        shader =\
            self.shaderBeam if target is not None else self.shaderFootprint
        beamBuffers = self.beamBufferDict.get(beamTag)
        targetBuffers = self.beamBufferDict.get(target)
        targetvbo = None

        if beamBuffers is None:
            return

        beamvbo = beamBuffers.get('vbo')
        beamvao = beamBuffers.get('vao')

        if targetBuffers is not None:
            targetvbo = self.beamBufferDict[target].get('vbo')

        if beamvbo is None:
            print("No VBO")
            return

        if target is not None and targetvbo is None:
            return

        shader.bind()

        beamvao.bind()

        if target is not None:
            targetvbo['position'].bind()
            gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(4)
            targetvbo['indices'].bind()
            arrLen = targetvbo['goodLen']

        else:
            beamvbo['indices'].bind()
            arrLen = beamvbo['goodLen']

        modelStart = copy.deepcopy(model)
        modelEnd = copy.deepcopy(model)

        if self.renderingMode == 'dynamic':
            oeuuid = beamTag[0]
            oeIndex = int(beamTag[1] == 'beamLocal2')
            oeOrientation = self.meshDict[oeuuid].transMatrix[oeIndex]
            if 'Global' not in beamTag[1]:
                modelStart *= oeOrientation

            if target is not None:
                enduuid = target[0]
                modelEnd = copy.deepcopy(model)
                oeIndex = int(target[1] == 'beamLocal2')
                oeOrientation = self.meshDict[enduuid].transMatrix[oeIndex]
                modelEnd *= oeOrientation

        shader.setUniformValue("modelStart", modelStart)
        if target is not None:
            shader.setUniformValue("modelEnd", modelEnd)

        if self.beamTexture is not None:
            self.beamTexture.bind(0)
            shader.setUniformValue("hsvTexture", 0)

        mPV = projection*view

        shader.setUniformValue("mPV", mPV)

        if self.globalColors:
            shader.setUniformValue(
                        "colorMinMax", qt.QVector2D(self.colorMin,
                                                    self.colorMax))
        else:
            shader.setUniformValue(
                        "colorMinMax",
                        qt.QVector2D(beamBuffers.get('colorMin', 0),
                                     beamBuffers.get('colorMax', 0)))
        shader.setUniformValue("gridMask", qt.QVector4D(1, 1, 1, 1))
        shader.setUniformValue("gridProjection", qt.QVector4D(0, 0, 0, 0))

        shader.setUniformValue(
                "pointSize",
                float(self.pointSize if target is None else self.lineWidth))
        shader.setUniformValue(
                "opacity",
                float(self.pointOpacity if target is None else
                      self.lineOpacity))
        shader.setUniformValue(
                "iMax",
                float(self.iMax if self.globalNorm else beamBuffers['iMax']))
        shader.setUniformValue("isLost", int(0))

        if target is not None and self.lineWidth > 0:
            gl.glLineWidth(min(self.lineWidth, 1.))

        gl.glDrawElements(gl.GL_POINTS,  arrLen,
                          gl.GL_UNSIGNED_INT, None)

        if target is not None and self.lineProjectionWidth > 0:
            gl.glLineWidth(min(self.lineProjectionWidth, 1.))

        for dim in range(3):
            if self.projectionsVisibility[dim] > 0:
                gridMask = [1.]*4
                gridMask[dim] = 0.
                gridProjection = [0.]*4
                gridProjection[dim] = -self.aPos[dim] *\
                    self.cBox.axPosModifier[dim]
                shader.setUniformValue(
                        "gridMask",
                        qt.QVector4D(*gridMask))
                shader.setUniformValue(
                        "gridProjection",
                        qt.QVector4D(*gridProjection))
                shader.setUniformValue(
                        "pointSize",
                        float(self.pointProjectionSize if target is None
                              else self.lineProjectionWidth))
                shader.setUniformValue(
                        "opacity",
                        float(self.pointProjectionOpacity if target is None
                              else self.lineProjectionOpacity))

                gl.glDrawElements(gl.GL_POINTS,  arrLen,
                                  gl.GL_UNSIGNED_INT, None)

        if target is not None:
            targetvbo['indices'].release()
        else:
            beamvbo['indices'].release()

        if self.showLostRays:
            if target is not None:
                targetvbo['indices_lost'].bind()
                arrLen = targetvbo['lostLen']
            else:
                beamvbo['indices_lost'].bind()
                arrLen = beamvbo['lostLen']
            shader.setUniformValue("isLost", int(0))  # TODO: controllable

            gl.glDrawElements(gl.GL_POINTS,  arrLen,
                              gl.GL_UNSIGNED_INT, None)

            if target is not None:
                targetvbo['indices_lost'].release()
            else:
                beamvbo['indices_lost'].release()

        if target is not None:
            targetvbo['position'].release()

        beamvao.release()

        shader.release()
        if self.beamTexture is not None:
            self.beamTexture.release()

    def getColorData(self, beam, beamTag):
        """beamTag: ('oeuuid', 'beamKey') """
        if beamTag[1].startswith('beamLoc')\
                and self.renderingMode == 'dynamic'\
                and self.globalColors:
            oe = self.beamline.oesDict[beamTag[0]][0]
            beamGlo = rsources.Beam(copyFrom=beam)
            is2ndXtal = beamTag[1] == 'beamLocal2'
            if self.globalColors:
                if is_screen(oe):
                    raycing.virgin_local_to_global(
                            self.beamline, beamGlo, oe.center)
                elif hasattr(oe, 'local_to_global'):
                    oe.local_to_global(
                            beamGlo, is2ndXtal=is2ndXtal)
            colorData = self.getColor(beamGlo)
        else:
            colorData = self.getColor(beam)

        return colorData.copy()

    def change_beam_colorax(self):
        if self.newColorAxis:
            newColorMax = -1e20
            newColorMin = 1e20
        else:
            newColorMax = self.colorMax
            newColorMin = self.colorMin

        for oeuuid, beamDict in self.beamline.beamsDictU.items():
            for beamKey, beam in beamDict.items():

                vboStore = self.beamBufferDict.get((oeuuid, beamKey))
                if vboStore is None:
                    continue

                good = (beam.state == 1) | (beam.state == 2)

                colorax = self.getColorData(beam, (oeuuid, beamKey))

                update_qt_buffer(vboStore['vbo']['color'], colorax.copy())

                if self.globalColors:
                    newColorMax = max(np.max(
                        colorax[good]),
                        newColorMax) if len(colorax[good]) else newColorMax
                    newColorMin = min(np.min(
                        colorax[good]),
                        newColorMin) if len(colorax[good]) else newColorMin
                else:
                    newColorMax = np.max(
                            colorax[good]) if len(
                                    colorax[good]) else newColorMax
                    newColorMin = np.min(
                            colorax[good]) if len(
                                    colorax[good]) else newColorMin

                if newColorMin == newColorMax:
                    if newColorMax == 0:
                        colorMinLoc, colorMaxLoc = -0.1, 0.1
                    else:
                        colorMinLoc = newColorMin * 0.99
                        colorMaxLoc = newColorMax * 1.01
                else:
                    colorMinLoc = newColorMin
                    colorMaxLoc = newColorMax

                vboStore['colorMin'] = colorMinLoc
                vboStore['colorMax'] = colorMaxLoc

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

#        if False:  # Updating textures with histograms
#            for oeuuid, beamDict in self.beamline.beamsDictU.items():
#                for beamKey, beam in beamDict:
#                    self.generate_hist_texture(beam, (oeuuid, beamKey))

        self.isColorAxReady = True

    def updateCutOffI(self, cutOff):
        for beamName, beam in self.beamsDict.items():
            if not hasattr(beam, 'vbo'):
                # 3D beam object not initialized
                continue
            intensity = np.float32(beam.Jss+beam.Jpp)
            goodRays = np.uint32(np.where(((
                    (beam.state == 1) | (beam.state == 2)) &
                    (intensity/self.iMax > self.cutoffI)))[0])
            beam.vbo['goodLen'] = len(goodRays)

            beam.vbo['indices'].bind()
            beam.vbo['indices'].write(0, goodRays, goodRays.nbytes)
            beam.vbo['indices'].release()

    def init_oe_surface(self, oeuuid):
        def assign_stencil_num(mesh):
            if oeuuid not in self.selectableOEs.values():
                if len(self.selectableOEs):
                    stencilNum = np.max(
                            list(self.selectableOEs.keys())) + 1
                else:
                    stencilNum = 1
                self.selectableOEs[int(stencilNum)] = oeuuid
                mesh.stencilNum = stencilNum

        oeLine = self.beamline.oesDict.get(oeuuid)
        mesh3D = None

        if oeLine is not None:
            oeToPlot = oeLine[0]
        else:
            return

        useAutoSize = self.autoSizeOe and not (
            oeuuid == self.virtScreen['uuid'] and self.vScreenManualSize)

        if is_oe(oeToPlot) or is_screen(oeToPlot):
            if oeuuid not in self.meshDict:
                mesh3D = OEMesh3D(oeToPlot, self)  # need to pass context

            is2ndXtalOpts = [False]
            if is_dcm(oeToPlot):
                is2ndXtalOpts.append(True)

            for is2ndXtal in is2ndXtalOpts:
                try:
                    mesh3D.prepare_surface_mesh(nsIndex=int(is2ndXtal),
                                                autoSize=useAutoSize)
                    mesh3D.isEnabled = True
                except Exception:
                    mesh3D.isEnabled = False
                assign_stencil_num(mesh3D)

        elif is_aperture(oeToPlot):
            if oeuuid not in self.meshDict:
                mesh3D = OEMesh3D(oeToPlot, self)  # need to pass context
            for blade in oeToPlot.blades:
                try:
                    mesh3D.prepare_surface_mesh(blade)
                    mesh3D.isEnabled = True
                except Exception:
#                    raise
                    mesh3D.isEnabled = False
                assign_stencil_num(mesh3D)

        else:  # must be the source
            try:
                mesh3D = self.meshDict.get(oeuuid, OEMesh3D(oeToPlot, self))
                if isinstance(oeToPlot, raycing.sources.GeometricSource):
                    mesh3D.prepare_geometric_source(shape=self.geomSrcParam)
                else:
                    mesh3D.prepare_magnets(shape=self.magnetShape)
                mesh3D.isEnabled = True
            except Exception:
                mesh3D.isEnabled = False
            assign_stencil_num(mesh3D)

        self.meshDict[oeuuid] = mesh3D

    def update_oe_transform(self, posData):
        oeid, pName, pValue = posData
        if pName in ['center']:
            try:
                self.getMinMax()
                self.maxLen = np.max(np.abs(
                        self.minmax[0, :] - self.minmax[1, :]))
                self.parent.updateMaxLenFromGL(self.maxLen)
            except TypeError:
                print("Cannot find limits")
        if pName in raycing.orientationArgSet:
            mesh = self.meshDict.get(oeid)
            if mesh is not None:  # TODO: may miss initial positioning
                mesh.update_transformation_matrix()

        self.glDraw()

    def update_oe_surface(self, oeuuid):
        oeToPlot = self.meshDict[oeuuid].oe
        useAutoSize = self.autoSizeOe and not (
            oeuuid == self.virtScreen['uuid'] and self.vScreenManualSize)
        if is_source(oeToPlot):
            try:
                if isinstance(oeToPlot, raycing.sources.GeometricSource):
                    self.meshDict[oeuuid].prepare_geometric_source(
                            updateMesh=True)
                else:
                    self.meshDict[oeuuid].prepare_magnets(updateMesh=True)
            except Exception as e:
                print(e)
                print("Update failed, disabling mesh for", oeuuid)
                self.meshDict[oeuuid].isEnabled = False
        else:
            for surfIndex in self.meshDict[oeuuid].vao.keys():
                try:
                    self.meshDict[oeuuid].prepare_surface_mesh(
                            nsIndex=surfIndex, updateMesh=True,
                            autoSize=useAutoSize)
                    self.meshDict[oeuuid].isEnabled = True
                except Exception as e:
                    print(e)
                    print("Update failed, disabling mesh for", oeuuid)
                    self.meshDict[oeuuid].isEnabled = False


#    def eulerToQ(self, rotMatrXYZ):
#        hPitch = np.radians(rotMatrXYZ[0][0]) * 0.5
#        hRoll = np.radians(rotMatrXYZ[1][0]) * 0.5
#        hYaw = np.radians(rotMatrXYZ[2][0]) * 0.5
#
#        cosPitch = np.cos(hPitch)
#        sinPitch = np.sin(hPitch)
#        cosRoll = np.cos(hRoll)
#        sinRoll = np.sin(hRoll)
#        cosYaw = np.cos(hYaw)
#        sinYaw = np.sin(hYaw)
#
#        return [cosPitch*cosRoll*cosYaw - sinPitch*sinRoll*sinYaw,
#                sinRoll*sinYaw*cosPitch + sinPitch*cosRoll*cosYaw,
#                sinRoll*cosPitch*cosYaw - sinPitch*sinYaw*cosRoll,
#                sinYaw*cosPitch*cosRoll + sinPitch*sinRoll*cosYaw]

#    def qToVec(self, quat):
#        angle = 2 * np.arccos(quat[0])
#        q2v = np.sin(angle * 0.5)
#        qbt1 = quat[1] / q2v if q2v != 0 else 0
#        qbt2 = quat[2] / q2v if q2v != 0 else 0
#        qbt3 = quat[3] / q2v if q2v != 0 else 0
#        return [np.degrees(angle), qbt1, qbt2, qbt3]

#    def rotateZYX(self):
#        if self.isEulerian:
#            gl.glRotatef(*self.rotations[0])
#            gl.glRotatef(*self.rotations[1])
#            gl.glRotatef(*self.rotations[2])
#        else:
#            gl.glRotatef(*self.rotationVec)

    def updateQuats(self):
        cR = self.cameraDistance
        azimuth = np.radians(self.rotations[0])
        elevation = np.radians(self.rotations[1])

        cosel = np.cos(elevation)

        self.cameraPos = qt.QVector3D(
                cR * cosel * np.cos(azimuth),
                cR * cosel * np.sin(azimuth),
                cR * np.sin(elevation))
        self.mView.setToIdentity()
        self.mView.lookAt(self.cameraPos, self.cameraTarget,
                          self.upVec)

        pModel = np.array(self.mView.data()).reshape(4, 4)[:-1, :-1]
        newVisAx = np.argmax(pModel, axis=0)
        if len(np.unique(newVisAx)) == 3:
            self.visibleAxes = newVisAx
        self.cBox.update_grid()

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

    def populateVerticesArray(self, segmentsModelRoot):
        pass

    def getMinMax(self):
        mins = np.array([1e20, 1e20, 1e20])
        maxs = -1 * mins

        for oeid, elline in self.beamline.oesDict.items():
            elCenter = elline[0].center
            for ic, coord in enumerate(elCenter):
                if isinstance(coord, str):
                    elCenter[ic] = 0
            mins = np.vstack((mins, elCenter))
            maxs = np.vstack((maxs, elCenter))
            beamDict = self.beamline.beamsDictU.get(oeid)
            if beamDict is not None:
                for beamkey, beam in self.beamline.beamsDictU[oeid].items():
                    if beamkey.startswith('beamGlo') and beam is not None:
                        good = (beam.state == 1) | (beam.state == 2)
                        bx, by, bz = beam.x[good], beam.y[good], beam.z[good]
                        if len(bx) == 0:
                            continue
                        mins = np.vstack((mins, np.array([np.min(bx),
                                                          np.min(by),
                                                          np.min(bz)])))
                        maxs = np.vstack((maxs, np.array([np.max(bx),
                                                          np.max(by),
                                                          np.max(bz)])))
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        self.minmax = np.vstack((mins, maxs))
        for dim in range(3):
            if self.minmax[0, dim] == self.minmax[1, dim]:
                self.minmax[0, dim] -= 100.  # TODO: configurable
                self.minmax[1, dim] += 100.
        return self.minmax

    def updateGlobalIntensity(self, newIMax):
        # TODO: consider decreasing globals
        if newIMax > self.iMax:
            self.iMax = newIMax

    def updateColorLimits(self, colorMin, colorMax):
        # TODO: consider decreasing globals
        if colorMin < self.colorMin:
            self.colorMin = colorMin
        if colorMax > self.colorMax:
            self.colorMax = colorMax
        if self.parent is not None:
            self.parent.changeColorAxis(None, newLimits=True)

    def getColorLimits(self):
        self.iMax = -1e20
        rebuildColorLimits = self.newColorAxis or not all(np.isfinite(
            [self.colorMin, self.colorMax])) or self.colorMin >= self.colorMax

        if rebuildColorLimits:
            newColorMax = -1e20
            newColorMin = 1e20
            if self.selColorMin is None:
                self.selColorMin = newColorMin
            if self.selColorMax is None:
                self.selColorMax = newColorMax
        else:
            newColorMax = self.colorMax
            newColorMin = self.colorMin

        for oeuuid, beamDict in self.beamline.beamsDictU.items():
            for beamKey, startBeam in beamDict.items():
                if startBeam is None:
                    continue
                good = (startBeam.state == 1) | (startBeam.state == 2)

                if len(startBeam.state[good]) > 0:
                    startBeam.iMax = np.max(startBeam.Jss[good] +
                                            startBeam.Jpp[good])
                    self.iMax = max(self.iMax, startBeam.iMax)
                    if rebuildColorLimits:
                        colorax = self.getColorData(startBeam,
                                                    (oeuuid, beamKey))
                        newColorMax = max(np.max(colorax[good]), newColorMax)
                        newColorMin = min(np.min(colorax[good]), newColorMin)

        if rebuildColorLimits:
            if newColorMin != self.colorMin:
                self.colorMin = newColorMin
                self.selColorMin = self.colorMin
            if newColorMax != self.colorMax:
                self.colorMax = newColorMax
                self.selColorMax = self.colorMax

            if self.colorMin == self.colorMax:
                if self.colorMax == 0:
                    self.colorMin, self.colorMax = -0.1, 0.1
                else:
                    self.colorMin = self.colorMax * 0.99
                    self.colorMax *= 1.01

        self.newColorAxis = False
        if self.parent is not None:
            self.parent.changeColorAxis(None, newLimits=rebuildColorLimits)
        if not rebuildColorLimits:
            self.glDraw()

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

    def set_scene_colors(self):
        colorTable = 'invertedColors' if self.invertColors else 'normalColors'
        if hasattr(self, 'sceneColors'):
            for colorKey, colorValue in self.sceneColors[colorTable].items():
                setattr(self, colorKey, colorValue)

    def quatMult(self, qf, qt):
        return [qf[0]*qt[0]-qf[1]*qt[1]-qf[2]*qt[2]-qf[3]*qt[3],
                qf[0]*qt[1]+qf[1]*qt[0]+qf[2]*qt[3]-qf[3]*qt[2],
                qf[0]*qt[2]-qf[1]*qt[3]+qf[2]*qt[0]+qf[3]*qt[1],
                qf[0]*qt[3]+qf[1]*qt[2]-qf[2]*qt[1]+qf[3]*qt[0]]

    def drawLocalAxes(self, oe, is2ndXtal):
        pass  # keep it here
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

    def initializeGL(self):
        gl.glGetError()
#        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glGetError()
#        gl.glEnable(gl.GL_POINT_SMOOTH)
#        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        print("Compiling shaders...", end='')
        self.init_shaders()
        gl.glGetError()
        self.init_coord_grid()  # We let coordBox have it's own shaders
        gl.glGetError()
        self.generate_beam_texture(512)
        gl.glGetError()
        print(" Done!")

        try:
            for oeuuid, beamDict in self.beamline.beamsDictU.items():
                for beamKey, startBeam in beamDict.items():
                    good = (startBeam.state == 1) | (startBeam.state == 2)
                    if len(startBeam.state[good]) > 0:
                        self.init_beam_footprint(startBeam, (oeuuid, beamKey))
        except AttributeError:
            pass

        self.getMinMax()
        self.maxLen = np.max(np.abs(self.minmax[0, :] - self.minmax[1, :]))
        self.parent.updateMaxLenFromGL(self.maxLen)
        self.newColorAxis = False
        self.labelLines = np.zeros((len(self.beamline.oesDict)*4, 3))
        self.llVBO = create_qt_buffer(self.labelLines)  # TODO
        self.labelvao = qt.QOpenGLVertexArrayObject()
        self.labelvao.create()
        self.labelvao.bind()
        self.llVBO.bind()
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        self.llVBO.release()
        self.labelvao.release()

        for oeuuid in self.beamline.oesDict:
            self.init_oe_surface(oeuuid)

        self.getColorLimits()

        gl.glViewport(*self.viewPortGL)
        pModel = np.array(self.mView.data()).reshape(4, 4)[:-1, :-1]
        newVisAx = np.argmax(pModel, axis=0)
        if len(np.unique(newVisAx)) == 3:
            self.visibleAxes = newVisAx
        self.cBox.update_grid()
        self.toggleLoop()

    def paintGL(self):
        # TODO: might be better to use dedicated dicts for sources, oes etc.
        def makeCenterStr(centerList, prec):
            retStr = '('
            for dim in centerList:
                retStr += '{0:.{1}f}, '.format(dim, prec)
            return retStr[:-2] + ')'

        def getItem(iId, itemType='beam', targetId=None):
            item = None
            start_index = model.index(0, 0)
            flags = qt.Qt.MatchExactly
            matches = model.match(start_index, qt.Qt.UserRole, iId, hits=1,
                                  flags=flags)
            if matches:
                item = model.item(matches[0].row(), itemTypes[itemType])
                if itemType == 'beam' and item.rowCount() > 0:
                    parent_index = model.indexFromItem(item)
                    fcIndex = item.child(0, 0).index()
                    tgt_matches = model.match(fcIndex, qt.Qt.UserRole,
                                              targetId, hits=-1, flags=flags)
                    for line in tgt_matches:
                        if line.parent() == parent_index:
                            item = model.itemFromIndex(line)
                            break
            return item

        if True:
            gl.glClearColor(*self.bgColor, 1.0)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT |
                       gl.GL_DEPTH_BUFFER_BIT |
                       gl.GL_STENCIL_BUFFER_BIT)

            self.mProj.setToIdentity()
            if self.perspectiveEnabled:
                self.mProj.perspective(
                        self.cameraAngle, self.aspect, 0.01, 1000)
            else:
                halfHeight = self.orthoScale * 0.5
                halfWidth = halfHeight * self.aspect
                self.mProj.ortho(-halfWidth, halfWidth,
                                 -halfHeight, halfHeight,
                                 0.01, 1000)

            self.mModScale.setToIdentity()
            self.mModTrans.setToIdentity()
            self.mModScale.scale(*(self.scaleVec/self.maxLen))
            self.mModTrans.translate(*(self.tVec-self.coordOffset))
            self.mMod = self.mModScale*self.mModTrans

            mMMLoc = self.mMod * self.mModLocal

            vpMat = self.mProj * self.mView * mMMLoc

            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if self.enableAA:
                gl.glEnable(gl.GL_LINE_SMOOTH)
                gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
#                gl.glEnable(gl.GL_POLYGON_SMOOTH)
#                gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)

            if self.enableBlending:
                gl.glEnable(gl.GL_MULTISAMPLE)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#                gl.glEnable(gl.GL_POINT_SMOOTH)
#                gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

#            if self.linesDepthTest:
#                gl.glDepthMask(gl.GL_FALSE)
            if not self.linesDepthTest:
                gl.glDepthMask(gl.GL_TRUE)

            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
#            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            model = self.segmentModel.model()

            while self.needMeshUpdate:  # TODO: process one element per call?
                elementId = self.needMeshUpdate.popleft()

                gl.glGetError()
                newMesh = self.meshDict.get(elementId) is None

                if newMesh:
                    self.init_oe_surface(elementId)
                else:
                    self.update_oe_surface(elementId)

            while self.needBeamUpdate:  # TODO: process one element per call?
                beamTag = self.needBeamUpdate.popleft()
                self.update_beam_footprint(beamTag=beamTag)

            while self.deletionQueue:
                oeid = self.deletionQueue.popleft()
                self.delete_all_oe_buffers(oeid)
#            self.needBeamUpdate = []

#            if self.needMeshUpdate is not None:
#                gl.glGetError()
#                if is_source(self.meshDict[self.needMeshUpdate].oe):
#                    self.meshDict[self.needMeshUpdate].prepare_magnets(
#                            updateMesh=True)
#                else:
#                    for surfIndex in self.meshDict[self.needMeshUpdate].vao.keys():
#                        self.meshDict[self.needMeshUpdate].prepare_surface_mesh(
#                                nsIndex=surfIndex, updateMesh=True)
#                self.needMeshUpdate = None

            gl.glEnable(gl.GL_STENCIL_TEST)

            for oeuuid, mesh3D in self.meshDict.items():
                item = getItem(oeuuid, 'surface')
                if item is None or item.checkState() != 2:
                    continue

                oeToPlot = mesh3D.oe

                if is_oe(oeToPlot):
                    is2ndXtalOpts = [False]
                    if is_dcm(oeToPlot):
                        is2ndXtalOpts.append(True)

                    for is2ndXtal in is2ndXtalOpts:
                        if mesh3D.isEnabled:
                            isSelected = False
                            if oeuuid in self.selectableOEs.values():
                                oeNum = mesh3D.stencilNum
                                isSelected = oeNum == self.selectedOE
                                gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum),
                                                 0xff)
                            try:
                                mesh3D.render_surface(
                                    mMMLoc, self.mView,
                                    self.mProj, int(is2ndXtal),
                                    isSelected=isSelected,
                                    shader=self.shaderMesh)
                            except Exception as e:
                                print(e)
                elif is_aperture(oeToPlot):
                    for blade in oeToPlot.blades:
                        if mesh3D.isEnabled:
                            isSelected = False
                            if oeuuid in self.selectableOEs.values():
                                oeNum = mesh3D.stencilNum
                                isSelected = oeNum == self.selectedOE
                                gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum),
                                                 0xff)
                            try:
                                mesh3D.render_surface(
                                    mMMLoc, self.mView,
                                    self.mProj, blade, isSelected=isSelected,
                                    shader=self.shaderMesh)
                            except Exception as e:
                                print(e)
                elif is_source(oeToPlot):
                    if mesh3D.isEnabled:
                        isSelected = False
                        if oeuuid in self.selectableOEs.values():
                            oeNum = mesh3D.stencilNum
                            isSelected = oeNum == self.selectedOE
                            gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum),
                                             0xff)
                        try:
                            if isinstance(oeToPlot, raycing.sources.GeometricSource):
                                mesh3D.render_geometric_source(
                                    mMMLoc, self.mView, self.mProj,
                                    scale = self.scaleVec,
                                    shape=self.geomSrcParam,
                                    isSelected=isSelected,
                                    shader=self.shaderGeo)
                            else:
                                mesh3D.render_magnets(
                                    mMMLoc, self.mView, self.mProj,
                                    shape=self.magnetShape,
                                    isSelected=isSelected,
                                    shader=self.shaderMag)
                        except Exception as e:
                            print(e)

            if not self.linesDepthTest:
                gl.glDepthMask(gl.GL_FALSE)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)
            # Screens are semi-transparent, DepthMask must be OFF
#            for ioe in range(self.segmentModel.rowCount() - 1):

#            if self.showVirtualScreen:
#                self.positionVScreen()

            for oeuuid, mesh3D in self.meshDict.items():
                item = getItem(oeuuid, 'surface')
                if item is None or item.checkState() != 2:
                    if not (self.showVirtualScreen and
                            oeuuid == self.virtScreen['uuid']):
                        continue

                if is_screen(mesh3D.oe):
                    is2ndXtal = False
                    if mesh3D.isEnabled:  # TODO: Looks like double check
                        isSelected = False
                        if oeuuid in self.selectableOEs.values():
                            oeNum = mesh3D.stencilNum
                            isSelected = oeNum == self.selectedOE
                            gl.glStencilFunc(gl.GL_ALWAYS, np.uint8(oeNum),
                                             0xff)
                        try:
                            mesh3D.render_surface(
                                    mMMLoc, self.mView,
                                    self.mProj, int(is2ndXtal),
                                    isSelected=isSelected,
                                    shader=self.shaderMesh)
                        except Exception as e:
                            print(e)

            gl.glStencilFunc(gl.GL_ALWAYS, 0, 0xff)
            gl.glDisable(gl.GL_STENCIL_TEST)

            if self.pointsDepthTest:
                gl.glEnable(gl.GL_DEPTH_TEST)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)

            # RENDER FOOTPRINTS

            for oeuuid, beamDict in self.beamline.beamsDictU.items():
                item = getItem(oeuuid, 'footprint')
                if item is None or item.checkState() != 2:
                    if not (self.showVirtualScreen and
                            oeuuid == self.virtScreen['uuid']):
                        continue

                for bField, bObj in beamDict.items():
                    if bField == 'beamGlobal' and {
                            'beamLocal', 'beamLocal1'} & beamDict.keys():
                        continue

                    beamTag = (oeuuid, bField)
#                    if oeuuid == self.virtScreen:
#                        print("Plotting virtual screen", bField)
                    try:
                        self.render_beam(beamTag, mMMLoc,
                                         self.mView, self.mProj, target=None)
                    except Exception as e:
                        print(e)

            gl.glEnable(gl.GL_DEPTH_TEST)

            # RENDER BEAMS

            if self.renderingMode == 'dynamic':
                for eluuid, operations in self.beamline.flowU.items():
                    for kwargset in operations.values():
                        if 'beam' in kwargset:
                            sourceuuid = kwargset['beam']
                            item = getItem(sourceuuid, 'beam', eluuid)
                            if item is None or item.checkState() != 2:
                                continue

                            beamStartDict = self.beamline.beamsDictU.get(
                                    sourceuuid)
                            beamEndDict = self.beamline.beamsDictU.get(eluuid)
                            startField = None
                            endField = None

                            if beamStartDict is None or beamEndDict is None:
                                continue

                            if 'beamLocal' in beamStartDict:
                                startField = 'beamLocal'
                            elif 'beamLocal2' in beamStartDict:
                                startField = 'beamLocal2'
                            elif 'beamGlobal' in beamStartDict:
                                startField = 'beamGlobal'

                            if 'beamLocal' in beamEndDict:
                                endField = 'beamLocal'
                            elif 'beamLocal1' in beamEndDict:
                                endField = 'beamLocal1'
                            elif 'beamGlobal' in beamEndDict:
                                endField = 'beamGlobal'

                            beamStart = (sourceuuid, startField)
                            beamEnd = (eluuid, endField)
                            try:
                                self.render_beam(beamStart, mMMLoc,
                                                 self.mView, self.mProj,
                                                 target=beamEnd)
                            except Exception as e:
                                print(e)

                            if self.showInternalBeam and\
                                    'beamLocal2' in beamStartDict:
                                try:
                                    self.render_beam(
                                            (sourceuuid, 'beamLocal1'),
                                            mMMLoc, self.mView, self.mProj,
                                            target=(sourceuuid, 'beamLocal2'))
                                except Exception as e:
                                    print(e)

            else:
                for flowLine in self.beamline.flow:
                    sourceuuid = flowLine[0]
                    eluuid = flowLine[2]
                    item = getItem(sourceuuid, 'beam', eluuid)
                    if item is None or item.checkState() != 2:
                        continue
                    if eluuid not in [None, sourceuuid]:
                        elObj = self.beamline.oesDict[eluuid][0]
                        startField = 'beamGlobal'
                        if is_dcm(elObj):
                            endField = 'beamLocal1'
                        elif is_screen(elObj) or is_aperture(elObj):
                            endField = 'beamLocal'
                        else:
                            endField = 'beamGlobal'

                        beamStart = (sourceuuid, startField)
                        beamEnd = (eluuid, endField)
                        try:
                            self.render_beam(beamStart, mMMLoc,
                                             self.mView, self.mProj,
                                             target=beamEnd)
                        except Exception as e:
                            print(e)

            self.cBox.textShader.bind()
            self.cBox.vaoText.bind()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            # RENDER LABELS

            sclY = self.cBox.characters[124][1][1] * 0.04 *\
                self.cBox.fontScale / float(self.viewPortGL[3])
            labelBounds = []
            lineCounter = 0
            labelLines = None
            gl.glDisable(gl.GL_DEPTH_TEST)

            for oeuuid, mesh3D in self.meshDict.items():
                item = getItem(oeuuid, 'label')
                if item is None or item.checkState() != 2:
                    continue

                oeToPlot = self.beamline.oesDict[oeuuid][0]
                oeCenter = oeToPlot.center
                if any([isinstance(val, str) for val in oeCenter]):
                    continue

                oeString = oeToPlot.name
                alignment = "middle"
                dx = 0.1
                oeCenterStr = makeCenterStr(oeToPlot.center,
                                            self.labelCoordPrec)
                oeLabel = '  {0}: {1}mm'.format(
                    oeString, oeCenterStr)

                oePos = (vpMat*qt.QVector4D(*oeCenter,
                                            1)).toVector3DAffine()
                lineHint = [oePos.x(), oePos.y(), oePos.z()]
                labelPos = qt.QVector3D(*lineHint) + qt.QVector3D(dx, 0, 0)

                intersecting = True
                fbCounter = 0
                while intersecting and fbCounter < 3*(len(labelBounds)+1):
                    labelYmin = labelPos.y()
                    labelYmax = labelYmin + sclY
                    for bmin, bmax in labelBounds:
                        if labelYmax > bmin and labelYmin < bmax:
                            labelPos += qt.QVector3D(0, 2*sclY, 0)
                            break
                        elif labelYmin > bmax and labelYmax < bmin:
                            labelPos -= qt.QVector3D(0, 2*sclY, 0)
                            break
                    else:
                        intersecting = False
                        labelBounds.append((labelPos.y(),
                                            labelPos.y() + sclY))
                    fbCounter += 1

                try:
                    endPos = self.cBox.render_text(
                        labelPos, oeLabel, alignment=alignment,
                        scale=0.04*self.cBox.fontScale,
                        textColor=qt.QVector3D(*self.textColor))
                    labelLinesN = np.vstack(
                        (np.array(lineHint),
                         np.array([labelPos.x(), labelPos.y()-sclY, 0.0]),
                         np.array([labelPos.x(), labelPos.y()-sclY, 0.0]),
                         np.array([endPos.x(), labelPos.y()-sclY, 0.0])))
                    labelLines = labelLinesN if labelLines is None else\
                        np.vstack((labelLines, labelLinesN))
                    lineCounter += 1
                except Exception as e:
                    print(e)
            self.cBox.textShader.release()
            self.cBox.vaoText.release()

            if False:  # labelLines is not None:  # Must be dynamic
                update_qt_buffer(self.llVBO, labelLines)

                self.cBox.shader.bind()
                self.cBox.shader.setUniformValue("lineOpacity", 0.85)
                self.cBox.shader.setUniformValue("lineColor",
                                                 qt.QVector3D(*self.textColor))
                self.labelvao.bind()
                self.cBox.shader.setUniformValue(
                        "pvm", qt.QMatrix4x4())
                gl.glLineWidth(min(self.cBoxLineWidth, 1.))
                gl.glDrawArrays(gl.GL_LINES, 0, lineCounter*4)
                self.labelvao.release()
                self.cBox.shader.release()

            # RENDER LOCAL AXES

            if self.showLocalAxes:
                self.cBox.origShader.bind()
                self.cBox.vao_arrow.bind()
                self.cBox.origShader.setUniformValue("lineOpacity", 0.85)
                gl.glLineWidth(min(self.cBoxLineWidth, 1.))

                for oeuuid, mesh3D in self.meshDict.items():
                    item = getItem(oeuuid, 'surface')
                    if item is None or item.checkState() != 2:
                        continue

                    oeToPlot = mesh3D.oe

                    is2ndXtalOpts = [False]
                    if is_dcm(oeToPlot):
                        is2ndXtalOpts.append(True)

                    for is2ndXtal in is2ndXtalOpts:
                        try:
                            if is2ndXtal:
                                trMat =\
                                    mesh3D.transMatrix[int(is2ndXtal)].data()
                                oeCenter = [trMat[12], trMat[13], trMat[14]]
                            else:
                                oeCenter = list(oeToPlot.center)

                            if any([isinstance(val, str) for val in oeCenter]):
                                continue

                            oePos = (mMMLoc*qt.QVector4D(*oeCenter,
                                                         1)).toVector3DAffine()
                            oeNorm = mesh3D.transMatrix[int(is2ndXtal)]
                            self.cBox.render_local_axes(
                                    mMMLoc*oeNorm, oePos, self.mView,
                                    self.mProj, self.cBox.origShader,
                                    is_screen(oeToPlot) or is_aperture(
                                            oeToPlot))
                        except Exception as e:
                            print(e)
                self.cBox.vao_arrow.release()
                self.cBox.origShader.release()

            # RENDER GRID ON SCREENS

#            self.cBox.shader.bind()
#            self.cBox.shader.setUniformValue("lineColor",
#                                             qt.QVector3D(0.0, 1.0, 1.0))
#
#            for oeuuid, mesh3D in self.meshDict.items():
#                continue
#                item = getItem(oeuuid, 'surface')
#                if item is None or item.checkState() != 2:
#                    continue
#
#                if is_screen(mesh3D.oe):
#                    mesh3D.grid_vbo['vertices'].bind()
#                    gl.glVertexAttribPointer(
#                            0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
#                    gl.glEnableVertexAttribArray(0)
#                    oeOrientation = mesh3D.transMatrix[0]
#                    grMod = mMMLoc*oeOrientation
#                    pvm = self.mProj*self.mView*grMod
#                    self.cBox.shader.setUniformValue("pvm", pvm)
#                    self.cBox.shader.setUniformValue("lineOpacity", 0.3)
#                    gl.glLineWidth(1.)
#                    gl.glDrawArrays(gl.GL_LINES, 0,
#                                    mesh3D.grid_vbo['gridLen'])
#                    mesh3D.grid_vbo['vertices'].release()
#
#            self.cBox.shader.release()

            # RENDER COORDINATE BOX

            if not self.linesDepthTest:
                gl.glDepthMask(gl.GL_TRUE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            if self.drawGrid:
                self.cBox.render_grid(self.mModAx, self.mView, self.mProj)

            if self.enableAA:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            self.eCounter = 0

            # RENDER DIRECTIONAL AXES WIDGET

            gl.glViewport(0, 0, int(0.2*self.viewPortGL[-1]),
                          int(0.2*self.viewPortGL[-1]))
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

            self.cBox.origShader.bind()
            self.cBox.vao_arrow.bind()
            self.cBox.origShader.setUniformValue("lineOpacity", 0.85)
            gl.glLineWidth(min(self.cBoxLineWidth, 1.))

            fixView = qt.QMatrix4x4()
            fixView.lookAt(self.cameraPos/self.cameraDistance,
                           self.cameraTarget,
                           self.upVec)

            self.cBox.render_local_axes(
                    None, None, fixView, self.fixProj,
                    self.cBox.origShader, False)

            self.cBox.vao_arrow.release()
            self.cBox.origShader.release()

            self.cBox.textShader.bind()
            self.cBox.vaoText.bind()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_DEPTH_TEST)
#            labelPos = qt.QVector3D(0.05, 0, 0) + qt.QVector3D(0, 0, 0.4)
            for label, labelPos, labelColor in zip(
                    ["x", "y", "z"],
                    [[0.5, 0, 0], [0, 0.5, 0], [0, 0, -0.6]],
                    [[0.5, 0.5, 1], [0.3, 1, 0.3], [1, 0.3, 0.3]]):

                endPos = self.cBox.render_text(
                    (fixView * self.fixProj *
                     qt.QVector4D(*labelPos, 0)).toVector3D(),
                    label, alignment=None,
                    scale=0.04*self.cBox.fontScale,
                    textColor=qt.QVector3D(*labelColor))

            self.cBox.textShader.release()
            self.cBox.vaoText.release()

#        except Exception as e:  # TODO: properly handle exceptions
#            raise
#            pass

    def resizeGL(self, widthInPixels, heightInPixels):
        self.viewPortGL = [0, 0, widthInPixels, heightInPixels]
        gl.glViewport(*self.viewPortGL)
        self.aspect = np.float32(widthInPixels)/np.float32(heightInPixels)

        self.mProj.setToIdentity()
        if self.perspectiveEnabled:
            self.mProj.perspective(self.cameraAngle, self.aspect, 0.01, 1000)
        else:
            halfHeight = self.orthoScale * 0.5
            halfWidth = halfHeight * self.aspect
            self.mProj.ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                             0.01, 1000)

    def populateVScreen(self):
        pass

    def createVScreen(self):
        try:
            if self.virtScreen is None:
                self.virtScreen['uuid'] = rscreens.Screen(
                    bl=self.beamline).uuid
            self.positionVScreen()
        except:  # analysis:ignore
            raise
#            if _DEBUG_:
#                raise
#            else:
#                self.clearVScreen()

    def positionVScreen(self, cntr=None):

        if self.virtScreen is None:
            return
        if cntr is None:
            mLoc = self.mMod * self.mModLocal
            orgLoc = mLoc.inverted()[0] * qt.QVector3D(0.0, 0.0, 0.0)
            cntr = [orgLoc.x(), orgLoc.y(), orgLoc.z()]
        dist = 1e12
        cProj = None

        if self.renderingMode == 'dynamic':
            for eluuid, operations in self.beamline.flowU.items():
                for kwargset in operations.values():
                    if 'beam' in kwargset:
                        tmpSource = kwargset['beam']

                        bStart0 = self.beamline.oesDict[tmpSource][0].center
                        bEnd0 = self.beamline.oesDict[eluuid][0].center

                        beam0 = bEnd0 - bStart0
                        # Finding the projection of the VScreen.center on
                        # segments
                        t = np.dot(cntr-bStart0, beam0) / np.dot(beam0, beam0)
                        t = np.clip(t, 0.0, 1.0)
                        cProjTmp = bStart0 + t * beam0

                        tmpDist = np.linalg.norm(cProjTmp-cntr)
                        if tmpDist < dist:
                            dist = tmpDist
                            sourceuuid = tmpSource
                            bStartCenter = bStart0
                            bEndCenter = bEnd0
                            cProj = cProjTmp

            if cProj is not None:
                scrId = self.virtScreen['uuid']
                screenObj = self.beamline.oesDict[scrId][0]
                screenObj.center = cProj
                self.virtScreen['center'] = cProj
                self.virtScreen['beamStart'] = bStartCenter
                self.virtScreen['beamEnd'] = bEndCenter
                self.virtScreen['beamPlane'] =\
                    np.cross(bEndCenter-bStartCenter,
                             np.array([0, 0, 1]))  # TODO: dynamic
                self.meshDict[scrId].update_transformation_matrix()
                beamToExpose =\
                    self.beamline.beamsDictU[sourceuuid]['beamGlobal']  # TODO: DCMs

                exBeam = raycing.inspect.unwrap(screenObj.expose)(
                        screenObj,
                        beam=beamToExpose)
                if hasattr(beamToExpose, 'iMax'):  # TODO: check
                    exBeam.iMax = beamToExpose.iMax
                else:
                    exBeam.iMax = self.iMax
                self.beamline.beamsDictU[scrId] =\
                    {'beamLocal': exBeam}
                virtBeamTag = (scrId, 'beamLocal')
#                self.update_beam_footprint(beam=exBeam, beamTag=virtBeamTag)
                self.beamUpdated.emit(virtBeamTag)
                self.needBeamUpdate.append(virtBeamTag)
#                self.virtScreen.beamToExpose = beamStart0
#
#        if self.isVirtScreenNormal:
#            vsX = [self.virtScreen.beamToExpose.b[0],
#                   -self.virtScreen.beamToExpose.a[0], 0]
#            vsY = [self.virtScreen.beamToExpose.a[0],
#                   self.virtScreen.beamToExpose.b[0],
#                   self.virtScreen.beamToExpose.c[0]]
#            vsZ = np.cross(vsX/np.linalg.norm(vsX),
#                           vsY/np.linalg.norm(vsY))
#        else:
#            vsX = 'auto'
#            vsZ = 'auto'
#        self.virtScreen.set_orientation(vsX, vsZ)
#        try:
#            self.virtBeam = self.virtScreen.expose_global(
#                self.virtScreen.beamToExpose)
#        except:  # analysis:ignore
#            self.clearVScreen()

    def toggleVScreen(self):
        self.showVirtualScreen = not self.showVirtualScreen
        if self.showVirtualScreen:
            self.positionVScreen()
        self.glDraw()

    def clearVScreen(self):
        self.showVirtualScreen = False

    def switchVScreenTilt(self):
        self.isVirtScreenNormal = not self.isVirtScreenNormal
        self.positionVScreen()
        self.glDraw()

    def getPlanePoint(self, mX, mY, plane_pt, plane_n):
        def getProjRay():
            x_ndc = (2 * mX) / xView - 1
            y_ndc = 1 - (2 * mY) / yView
            nearN = qt.QVector4D(x_ndc, y_ndc, -1.0, 1.0)
            farN = qt.QVector4D(x_ndc, y_ndc, 1.0, 1.0)

            mLoc = self.mMod * self.mModLocal
            vp = self.mProj * self.mView * mLoc
            inv, ok = vp.inverted()
            if not ok:
                return qt.QVector3D(), qt.QVector3D(0, 0, -1)

            nW4 = inv * nearN
            fW4 = inv * farN
            nW = qt.QVector3D(nW4.x()/nW4.w(), nW4.y()/nW4.w(),
                              nW4.z()/nW4.w())
            fW = qt.QVector3D(fW4.x()/fW4.w(), fW4.y()/fW4.w(),
                              fW4.z()/fW4.w())

            rd = (fW - nW).normalized()
            return (np.array([nW.x(), nW.y(), nW.z()]),
                    np.array([rd.x(), rd.y(), rd.z()]))

        xView = self.viewPortGL[2]
        yView = self.viewPortGL[3]

        r0, rd = getProjRay()
        denom = np.dot(rd, plane_n)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(plane_pt - r0, plane_n) / denom
        return r0 + rd * t

    def mouseMoveEvent(self, mEvent):
        xView = self.viewPortGL[2]
        yView = self.viewPortGL[3]
        mouseX = mEvent.x()
        mouseY = yView - mEvent.y()
        self.makeCurrent()
        try:
            try:
                outStencil = gl.glReadPixels(
                        mouseX, mouseY-1, 1, 1, gl.GL_STENCIL_INDEX,
                        gl.GL_UNSIGNED_INT)
            except OSError:
                return
            overOE = np.squeeze(np.array(outStencil))

            ctrlOn = bool(int(mEvent.modifiers()) & int(qt.Qt.ControlModifier))
#        altOn = bool(int(mEvent.modifiers()) & int(qt.Qt.AltModifier))
            shiftOn = bool(int(mEvent.modifiers()) & int(qt.Qt.ShiftModifier))
#        polarAx = qt.QVector3D(0, 0, 1)

            dx = mouseX - self.prevMPos[0]
            dy = mouseY - self.prevMPos[1]

            xs = 2 * dx / xView
            ys = 2 * dy / yView
            xsn = xs * np.tan(np.radians(60))
            ysn = ys * np.tan(np.radians(60))
            xm = xsn * self.cameraDistance / 3.5
            ym = ysn * self.cameraDistance / 3.5

            if mEvent.buttons() == qt.Qt.LeftButton:
                if mEvent.modifiers() == qt.Qt.NoModifier:
                    sensitivity = 120
                    self.rotations[0] -= sensitivity*self.aspect*xs
                    self.rotations[1] -= sensitivity*ys

                    if self.rotations[0] < -180:
                        self.rotations[0] += 360

                    if self.rotations[0] > 180:
                        self.rotations[0] -= 360

                    if self.rotations[1] >= 90:
                        self.rotations[1] = 89.99

                    if self.rotations[1] <= -90:
                        self.rotations[1] = -89.99

                    self.rotationUpdated.emit(self.rotations)

                elif shiftOn:
                    az, el = self.rotations
                    mouse_h = np.array([-snsc(az, 45), snsc(az, -45), 0])
                    psgn = -snsc(el, 45)
                    mouse_v = np.array([psgn*snsc(az, -45), psgn*snsc(az, 45),
                                        snsc(el, -45)])
                    shifts = xm * mouse_h + ym * mouse_v

                    self.tVec += shifts*self.maxLen/self.scaleVec
                    self.cBox.update_grid()

                elif ctrlOn and self.showVirtualScreen:
                    tPlane = self.virtScreen['beamStart']
                    nPlane = self.virtScreen['beamPlane']
                    pPlane = self.getPlanePoint(mouseX, mouseY, tPlane, nPlane)
                    if self.virtScreen['offsetOn']:
                        self.virtScreen['offset'] =\
                            pPlane - self.virtScreen['center']
                        self.virtScreen['offsetOn'] = False
                    self.positionVScreen(pPlane - self.virtScreen['offset'])

                self.glDraw()
            else:
                if int(overOE) in self.selectableOEs:
                    oe = self.beamline.oesDict[
                        self.selectableOEs[int(overOE)]][0]
                    try:
                        oePitchStr = np.degrees(
                            oe.pitch + (oe.bragg if hasattr(oe, 'bragg')
                                        else 0)) if hasattr(oe, 'pitch') else 0
                    except TypeError:
                        oePitchStr = 0

                    try:
                        tooltipStr = "{0}\n[x, y, z]: [{1:.3f}, {2:.3f}, {3:.3f}]mm\n[p, r, y]: ({4:.3f}, {5:.3f}, {6:.3f})\u00B0".format(
                                oe.name, *oe.center,
                                oePitchStr,
                                np.degrees(oe.roll+oe.positionRoll)
                                if is_oe(oe) else 0,
                                np.degrees(oe.yaw)
                                if hasattr(oe, 'yaw') else 0)
                    except ValueError:
                        tooltipStr = str(oe.name)
                    qt.QToolTip.showText(mEvent.globalPos(), tooltipStr, self)
                else:
                    qt.QToolTip.hideText()
                if overOE != self.selectedOE:
                    self.selectedOE = int(overOE)
                    self.glDraw()
            self.prevMPos[0] = mouseX
            self.prevMPos[1] = mouseY
        finally:
            self.doneCurrent()

    def mouseDoubleClickEvent(self, mdcevent):
        if self.selectedOE > 0:
            self.openElViewer.emit(self.selectableOEs.get(int(self.selectedOE),
                                                          'None'))

    def mousePressEvent(self, mpevent):
        ctrlOn = bool(int(mpevent.modifiers()) & int(qt.Qt.ControlModifier))
        self.virtScreen['offsetOn'] = ctrlOn
        super().mousePressEvent(mpevent)

    def wheelEvent(self, wEvent):
        ctrlOn = bool(int(wEvent.modifiers()) & int(qt.Qt.ControlModifier))
        altOn = bool(int(wEvent.modifiers()) & int(qt.Qt.AltModifier))
        tpad = False

        if qt.QtName == "PyQt4":
            deltaA = wEvent.delta()
        else:
            deltaA = wEvent.angleDelta().y() + wEvent.angleDelta().x()
            tpad = wEvent.pixelDelta().y() > 0

        if deltaA > 0:
            if altOn:
                scrId = self.virtScreen['uuid']
                scrLine = self.beamline.oesDict.get(scrId)
                if scrLine is not None:
                    scrObj = scrLine[0]
                    self.vScreenManualSize = True
                    scrObj.limPhysX *= 1.1
                    scrObj.limPhysY *= 1.1
                if scrId not in self.needMeshUpdate:
                    self.needMeshUpdate.append(scrId)
            elif ctrlOn and not tpad:
                self.cameraDistance *= 0.9
            else:
                self.scaleVec *= 1.1
        else:
            if altOn:
                scrId = self.virtScreen['uuid']
                scrLine = self.beamline.oesDict.get(scrId)
                if scrLine is not None:
                    scrObj = scrLine[0]
                    self.vScreenManualSize = True
                    scrObj.limPhysX *= 0.9
                    scrObj.limPhysY *= 0.9
                self.needMeshUpdate.append(scrId)  # TODO: check for duplicates
            elif ctrlOn and not tpad:
                self.cameraDistance *= 1.1
            else:
                self.scaleVec *= 0.9

        if ctrlOn:
            self.orthoScale = 2.0 * self.cameraDistance * np.tan(
                np.radians(self.cameraAngle*0.5))
            self.updateQuats()
        else:
            self.scaleUpdated.emit(self.scaleVec)

        self.cBox.update_grid()
        self.glDraw()

    def glDraw(self):
        self.update()

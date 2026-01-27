# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
# import os
import numpy as np
# from itertools import compress
from itertools import islice
from collections import OrderedDict
import re
import copy
import inspect
import uuid
import importlib
import json
import xml.etree.ElementTree as ET
import time

from .singletons import colorPrint, basestring, is_sequence, _VERBOSITY_
from .physconsts import SIE0, CH  # analysis:ignore

from ._flow_utils import (
    get_params, create_paramdict_oe, is_valid_uuid, parametrize,
    create_paramdict_mat, get_init_val, get_init_kwargs, get_obj_str,
    create_paramdict_fe)
from ._rotate import rotate_z, rotate_beam
from ._named_arrays import Center

_DEBUG_ = True  # If False, exceptions inside the module are ignored


def distance_xy(p1, p2):
    """Calculates 2D distance between p1 and p2. p1 and p2 are vectors of
    length >= 2."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def distance_xyz(p1, p2):
    """Calculates 2D distance between p1 and p2. p1 and p2 are vectors of
    length >= 3."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5


def global_to_virgin_local(bl, beam, lo, center=None, part=None):
    """Transforms *beam* from the global to the virgin (i.e. with pitch, roll
    and yaw all zeros) local system. The resulting local beam is *lo*. If
    *center* is provided, the rotation Rz is about it, otherwise is about the
    origin of *beam*. The beam arrays can be sliced by *part* indexing array.
    *bl* is an instance of :class:`BeamLine`"""
    if part is None:
        part = np.ones(beam.x.shape, dtype=bool)
    if center is None:
        center = [0, 0, 0]
    lo.x[part] = beam.x[part] - center[0]
    lo.y[part] = beam.y[part] - center[1]
    lo.z[part] = beam.z[part] - center[2]
    if isinstance(bl, BeamLine):
        a0, b0 = bl.sinAzimuth, bl.cosAzimuth
        if a0 == 0:
            lo.a[part] = beam.a[part]
            lo.b[part] = beam.b[part]
        else:
            lo.x[part], lo.y[part] = rotate_z(lo.x[part], lo.y[part], b0, a0)
            lo.a[part], lo.b[part] = \
                rotate_z(beam.a[part], beam.b[part], b0, a0)
        lo.c[part] = beam.c[part]  # unchanged
    elif isinstance(bl, (list, tuple)):
        lx, ly, lz = bl
        xyz = lo.x[part], lo.y[part], lo.z[part]
        lo.x[part], lo.y[part], lo.z[part] = (
            sum(c*b for c, b in zip(lx, xyz)),
            sum(c*b for c, b in zip(ly, xyz)),
            sum(c*b for c, b in zip(lz, xyz)))
        abc = beam.a[part], beam.b[part], beam.c[part]
        lo.a[part], lo.b[part], lo.c[part] = (
            sum(c*b for c, b in zip(lx, abc)),
            sum(c*b for c, b in zip(ly, abc)),
            sum(c*b for c, b in zip(lz, abc)))


def virgin_local_to_global(bl, vlb, center=None, part=None,
                           skip_xyz=False, skip_abc=False, is2ndXtal=False):
    """Transforms *vlb* from the virgin (i.e. with pitch, roll and yaw all
    zeros) local to the global system and overwrites the result to *vlb*. If
    *center* is provided, the rotation Rz is about it, otherwise is about the
    origin of *beam*. The beam arrays can be sliced by *part* indexing array.
    *bl* is an instance of :class:`BeamLine`"""
    if part is None:
        part = np.ones(vlb.x.shape, dtype=bool)
    a0, b0 = bl.sinAzimuth, bl.cosAzimuth
    if a0 != 0:
        if not skip_abc:
            vlb.a[part], vlb.b[part] = rotate_z(
                vlb.a[part], vlb.b[part], b0, -a0)
        if not skip_xyz:
            vlb.x[part], vlb.y[part] = rotate_z(
                vlb.x[part], vlb.y[part], b0, -a0)
    if (center is not None) and (not skip_xyz):
        vlb.x[part] += center[0]
        vlb.y[part] += center[1]
        vlb.z[part] += center[2]


def xyz_from_xz(obj, x=None, z=None):
    if isinstance(x, basestring) and isinstance(z, basestring):
        return 'auto'
    bl = obj.bl
    if isinstance(x, (list, tuple, np.ndarray)):
        norm = sum([xc**2 for xc in x])**0.5
        retx = [xc/norm for xc in x]
    else:
        retx = bl.cosAzimuth, -bl.sinAzimuth, 0.

    if isinstance(z, (list, tuple, np.ndarray)):
        norm = sum([zc**2 for zc in z])**0.5
        retz = [zc/norm for zc in z]
    else:
        retz = 0., 0., 1.

    xdotz = np.dot(retx, retz)
    if abs(xdotz) > 1e-8:
        try:
            objName = obj.name
        except AttributeError:
            objName = obj.__class__.__name__
        colorPrint('x and z must be orthogonal, got xz={0:.4e} for {1}'
                   .format(xdotz, objName), 'RED')
    rety = np.cross(retz, retx)
    return [retx, rety, retz]


def is_auto_align_required(oe):
    needAutoAlign = False
    for autoParam in ["_center", "_pitch", "_bragg"]:
        naParam = autoParam.strip("_")
        if hasattr(oe, autoParam) and hasattr(oe, naParam):
            if str(getattr(oe, autoParam)) == str(getattr(oe, naParam)):
                if _VERBOSITY_ > 20:
                    print(autoParam, str(getattr(oe, autoParam)),
                          naParam, str(getattr(oe, naParam)))
                needAutoAlign = True
                if _VERBOSITY_ > 10:
                    try:
                        objName = oe.name
                    except AttributeError:
                        objName = oe.__class__.__name__
                    print("{0}.{1} requires auto-calculation".format(
                        objName, naParam))
    return needAutoAlign


class AlignmentBeam(object):
    def __init__(self):
        for prop in ['a', 'b', 'c', 'x', 'y', 'z', 'E']:
            setattr(self, prop, np.zeros(2))


class BeamLine(object):
    u"""
    Container class for beamline components. It also defines the beam line
    direction and height."""

    def __init__(self, azimuth=0., height=0., alignE='auto', fileName=None,
                 name='beamLine', description=''):
        u"""
        *azimuth*: float
            Is counted in cw direction from the global Y axis. At
            *azimuth* = 0 the local Y coincides with the global Y.

        *height*: float
            Beamline height in the global system.

        *alignE*: float or 'auto'
            Energy for automatic alignment in [eV]. If 'auto', alignment energy
            is defined as the middle of the Source energy range.
            Plays a role if the *pitch* or *bragg* parameters of the energy
            dispersive optical elements were set to 'auto'.


        """

        self.azimuth = azimuth
#        self.sinAzimuth = np.sin(azimuth)  # a0
#        self.cosAzimuth = np.cos(azimuth)  # b0
        self.height = height
        self.alignE = alignE
        self.sources = []
        self.oes = []
        self.slits = []
        self.screens = []
        self.alarms = []
        self.name = name
        self.description = description
        self.oesDict = OrderedDict()
        self.oenamesToUUIDs = {}  # Reverse lookup for oe names
        self.matnamesToUUIDs = {}  # Reverse lookup for mat names
        self.fenamesToUUIDs = {}
        self.flow = []
        self.materialsDict = OrderedDict()
        self.beamsDict = OrderedDict()
        self.fesDict = OrderedDict()
        self.flowSource = 'legacy'
        self.forceAlign = False
        self.beamsDictU = OrderedDict()
        self.flowU = OrderedDict()
        self.beamNamesDict = {}  # Used in run_process_from_file
        self.beamsRevDict = OrderedDict()
        self.beamsRevDictUsed = {}
        self.blViewer = None
        self.blExplorer = None
        self.statusSignal = None
        self.layoutStr = None
        if fileName:
            if str(fileName).lower().endswith("xml"):
                self.load_from_xml(fileName)
            elif str(fileName).lower().endswith("json"):
                self.load_from_json(fileName)

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value
        self.sinAzimuth = float(np.sin(value))
        self.cosAzimuth = float(np.cos(value))

    def orient_along_global_Y(self, center='auto'):
        if center == 'auto':
            center0 = self.sources[0].center
        a0, b0 = self.sinAzimuth, self.cosAzimuth
        for oe in self.sources + self.oes + self.slits + self.screens:
            newC = [c-c0 for c, c0 in zip(oe.center, center0)]
            newC[0], newC[1] = rotate_z(newC[0], newC[1], b0, a0)
            oe.center = newC
            if hasattr(oe, 'jack1'):
                oe.jack1 = [c-c0 for c, c0 in zip(oe.jack1, center0)]
                oe.jack1[0], oe.jack1[1] = \
                    rotate_z(oe.jack1[0], oe.jack1[1], b0, a0)
            if hasattr(oe, 'jack2'):
                oe.jack2 = [c-c0 for c, c0 in zip(oe.jack2, center0)]
                oe.jack2[0], oe.jack2[1] = \
                    rotate_z(oe.jack2[0], oe.jack2[1], b0, a0)
            if hasattr(oe, 'jack3'):
                oe.jack3 = [c-c0 for c, c0 in zip(oe.jack3, center0)]
                oe.jack3[0], oe.jack3[1] = \
                    rotate_z(oe.jack3[0], oe.jack3[1], b0, a0)

        self.azimuth = 0

    def prepare_flow(self):
        def _warning(v1=None, v2=None):
            if v1 is None or v2 is None:
                addw = ""
            else:
                addw = "\nThis beam has been used for {0} and is attempted"\
                    " for {1}.".format(v1, v2)
            print("Warning: the flow seems corrupt. Make sure each propagation"
                  " method assigns returned beams to local variables." + addw)
        if self.flowSource != 'legacy':
            return
        frame = inspect.currentframe()
        localsDict = frame.f_back.f_locals
        globalsDict = frame.f_back.f_globals
        for objectName, memObject in globalsDict.items():
            if len(re.findall('raycing.materials', str(type(memObject)))) > 0:
                self.materialsDict[objectName] = memObject

        for objectName, memObject in localsDict.items():
            if len(re.findall('sources_beams.Beam', str(type(memObject)))) > 0:
                self.beamsDict[objectName] = memObject
                self.beamsRevDict[id(memObject)] = objectName
            if objectName == 'outDict':
                for odObjectName, odMemObject in memObject.items():
                    self.beamsDict[odObjectName] = odMemObject
                    self.beamsRevDict[id(odMemObject)] = odObjectName
        if self.flow is not None and len(self.beamsRevDict) > 0:
            for segment in self.flow:
                for iseg in [2, 3]:
                    for argName, argVal in segment[iseg].items():
                        if len(re.findall('beam', str(argName))) > 0:
                            if iseg == 3:
                                if argVal in self.beamsRevDictUsed:
                                    _warning(self.beamsRevDictUsed[argVal],
                                             segment[0])
                                self.beamsRevDictUsed[argVal] = segment[0]
                            try:
                                segment[iseg][argName] =\
                                    self.beamsRevDict[argVal]
                            except KeyError:
                                segment[iseg][argName] = 'beamTmp'
                                _warning()
        self.flowSource = 'prepared_to_run'

    def auto_align(self, oe, beam):
        if self.flowSource == 'Qook':
            self.forceAlign = True
        if not (self.forceAlign or is_auto_align_required(oe)):
            return

        autoCenter = [False] * 3
        autoPitch = autoBragg = False
        alignE = self._alignE if hasattr(self, '_alignE') else self.alignE

        if hasattr(oe, '_center') and isinstance(oe._center, list):
            autoCenter = [isinstance(x, str) for x in oe._center]
#            autoCenter = ['auto' in str(x) for x in oe._center]

        if hasattr(oe, '_pitch'):
            try:
                if isinstance(oe._pitch, (list, tuple)):
                    alignE = float(oe._pitch[-1])
                autoPitch = oe._pitch is not None
            except Exception:
                print("Automatic Bragg angle calculation failed.")
                raise

        if hasattr(oe, '_bragg'):
            try:
                if isinstance(oe._bragg, (list, tuple)):
                    alignE = float(oe._bragg[-1])
                autoBragg = True
            except Exception:
                print("Automatic Bragg angle calculation failed.")
                raise

        if any(autoCenter) or autoPitch or autoBragg:
            good = (beam.state == 1) | (beam.state == 2)
            if self.flowSource == 'Qook':
                beam.state[0] = 1
#                beam.E[0] = alignE
            intensity = beam.Jss[good] + beam.Jpp[good]
            totalI = np.sum(intensity)
            inBeam = AlignmentBeam()
            for fieldName in ['x', 'y', 'z', 'a', 'b', 'c']:
                field = getattr(beam, fieldName)
                if totalI == 0:
                    fNorm = 1.
                else:
                    fNorm = np.sum(field[good] * intensity) / totalI
                try:
                    setattr(inBeam, fieldName,
                            np.ones(2) * fNorm)
                    if self.flowSource == 'Qook':
                        field[0] = fNorm
                        setattr(inBeam, fieldName, field)
                except Exception:
                    print("Cannot find direction for automatic alignment.")
                    raise

            dirNorm = np.sqrt(inBeam.a[0]**2 + inBeam.b[0]**2 + inBeam.c[0]**2)
            inBeam.a[0] /= dirNorm
            inBeam.b[0] /= dirNorm
            inBeam.c[0] /= dirNorm

            if self.flowSource == 'Qook':
                beam.a[0] /= dirNorm
                beam.b[0] /= dirNorm
                beam.c[0] /= dirNorm

        if any(autoCenter):
            centerList = copy.deepcopy(oe.center)
            bStartC = np.array([inBeam.x[0], inBeam.y[0], inBeam.z[0]])
            bStartDir = np.array([inBeam.a[0], inBeam.b[0], inBeam.c[0]])
            fixedCoord = np.where(np.invert(np.array(autoCenter)))[0]
            autoCoord = np.where(autoCenter)[0]
            for dim in fixedCoord:
                if np.abs(bStartDir[dim]) > 1e-3:
                    plNorm = np.squeeze(np.identity(3)[dim, :])
                    newCenter = bStartC - (np.dot(
                        bStartC, plNorm) - oe.center[dim]) /\
                        np.dot(bStartDir, plNorm) * bStartDir
                    if np.linalg.norm(newCenter - bStartC) > 0:
                        break
            for dim in autoCoord:
                centerList[dim] = newCenter[dim]
            oe._centerVal = Center(centerList)
            if _VERBOSITY_ > 0:
                print(oe.name, "center:", oe.center)

        if autoBragg or autoPitch:
            if self.flowSource == 'Qook':
                inBeam.E[0] = alignE
            try:
                if is_sequence(oe.material):
                    mat = oe.material[oe.curSurface]
                else:
                    mat = oe.material
                if not hasattr(mat, 'get_Bragg_angle'):
                    if autoPitch:
                        oe._pitchVal = 0
                    elif autoBragg:
                        oe._braggVal = 0
                    return
                braggT = mat.get_Bragg_angle(alignE)
                alphaT = 0.
                lauePitch = 0.
                if mat.kind == 'multilayer':
                    braggT += -mat.get_dtheta(alignE)
                else:
                    alphaT = 0 if oe.alpha is None else oe.alpha
                    if mat.geom.startswith('Laue'):
                        lauePitch = 0.5 * np.pi
                    else:
                        braggT += -mat.get_dtheta(alignE, alphaT)

                loBeam = copy.deepcopy(inBeam)  # Beam(copyFrom=inBeam)
                global_to_virgin_local(self, inBeam, loBeam, center=oe.center)
                rotate_beam(loBeam, roll=-(oe.positionRoll + oe.roll),
                            yaw=-oe.yaw, pitch=0)
                theta0 = np.arctan2(-loBeam.c[0], loBeam.b[0])
                th2pitch = np.sqrt(1. - loBeam.a[0]**2)
                targetPitch = np.arcsin(np.sin(braggT) / th2pitch) - theta0
                targetPitch += alphaT + lauePitch
                if autoBragg:
                    if autoPitch:
                        oe.pitch = 0
                    oe._braggVal = targetPitch - oe.pitch
                    if _VERBOSITY_ > 0:
                        print("{0}: Bragg={1} at E={2}".format(
                                oe.name, oe.bragg, alignE))
                else:  # autoPitch
                    oe._pitchVal = targetPitch
                    if _VERBOSITY_ > 0:
                        print(oe.name, "pitch:", oe.pitch)
            except Exception as e:
                if _DEBUG_:
                    raise e
                else:
                    pass

    def propagate_flow(self, startFrom=0, signal=None):
        if self.oesDict is None or self.flow is None:
            return
        totalStages = len(self.flow[startFrom:])
        for iseg, segment in enumerate(self.flow[startFrom:]):
            segOE = self.oesDict[segment[0]][0]
            fArgs = OrderedDict()
            for inArg in segment[2].items():
                if inArg[0].startswith('beam'):
                    if inArg[1] is None:
                        inBeam = None
                        break
                    fArgs[inArg[0]] = self.beamsDict[inArg[1]]
                    inBeam = fArgs['beam']
                else:
                    fArgs[inArg[0]] = inArg[1]
            try:
                if inBeam is None:
                    continue
            except NameError:
                pass

            try:  # protection againt incorrect propagation parameters
                if signal is not None:
                    signalStr = "Propagation: {0} {1}(), %p% done.".format(
                        str(segment[0]),
                        str(segment[1]).split(".")[-1].strip(">").split(
                                " ")[0])
                    signal.emit((float(iseg+1)/float(totalStages), signalStr))
                    self.statusSignal =\
                        [signal, iseg+1, totalStages, signalStr]
            except Exception:
                pass

            try:
                outBeams = segment[1](segOE, **fArgs)
            except Exception:
                if _DEBUG_:
                    raise
                else:
                    continue

            if isinstance(outBeams, tuple):
                for outBeam, beamName in zip(list(outBeams),
                                             list(segment[3].values())):
                    self.beamsDict[beamName] = outBeam
            else:
                self.beamsDict[str(list(segment[3].values())[0])] = outBeams

    def sort_flow(self):
        visited = set()
        result = []
        newFlow = OrderedDict()

        def get_beam_id(oeid):
            methDict = self.flowU.get(oeid)
            if methDict is None:
                return None
            for kwargs in methDict.values():
                sourceid = kwargs.get('beam')
                return sourceid

        def dfs(oeid):
            visited.add(oeid)
            for rcv in receivers.get(oeid, []):
                if rcv not in visited:
                    dfs(rcv)
            result.append(oeid)

        def distance(id1, id2):
            line1 = self.oesDict.get(id1)
            line2 = self.oesDict.get(id2)

            if line1 and line2:
                obj1 = line1[0]
                obj2 = line2[0]
            else:
                return 0

            eid1c = np.array(getattr(obj1, 'center', [0, 0, 0]))
            eid2c = np.array(getattr(obj2, 'center', [0, 0, 0]))

            try:
                dist = np.linalg.norm(eid1c - eid2c)
            except Exception:
                dist = 0

            return dist

        receivers = {}
        for oeid in self.flowU:
            sourceid = get_beam_id(oeid)
            if sourceid is not None:
                receivers.setdefault(sourceid, []).append(oeid)

        for sourceid, rcv in receivers.items():
            if sourceid in self.flowU:
                rcv.sort(key=lambda oid: distance(oid, sourceid), reverse=True)

        for oe in self.flowU:
            if oe not in visited:
                dfs(oe)

        for oeuuid in reversed(result):
            tmpRec = self.flowU.get(oeuuid)
            if tmpRec is not None:
                newFlow[oeuuid] = tmpRec

        self.flowU = newFlow

    def sort_materials(self, matDict=None, fromJSON=False):
        visited = set()
        visiting = set()
        sortedMatList = []
        matDeps = ['tLayer', 'bLayer', 'coating', 'substrate']

        def get_dep_obj(matObj):
            deps = []
            for attr in matDeps:
                if hasattr(matObj, attr):
                    v = getattr(matObj, attr)
                    if is_valid_uuid(v):
                        deps.append(v)
                    elif v is not None and hasattr(v, 'uuid'):
                        deps.append(getattr(v, 'uuid'))
            return deps

        def get_dep_json(pDict):
            deps = []
            props = pDict.get('properties')
            if props is not None:
                for attr in matDeps:
                    v = props.get(attr)
                    if v is not None:
                        deps.append(v)
            return deps

        def dfs(mId, mProps):
            if mId in visited:
                return
            if mId in visiting:
                raise ValueError(
                    f"Circular dependency detected involving {mId}")
            visiting.add(mId)
            if mProps is not None:
                for dep in get_dependencies(mProps):
                    dfs(dep, None)
            visiting.remove(mId)
            visited.add(mId)
            sortedMatList.append(mId)

        if matDict is None:
            matDict = self.materialsDict

        get_dependencies = get_dep_json if fromJSON else get_dep_obj

        for mId, mProps in matDict.items():
            dfs(mId, mProps)

        return sortedMatList

    def sort_figerrors(self, feDict=None, fromJSON=False):
        visited = set()
        visiting = set()
        sortedFEList = []
        feDeps = ['baseFE']

        def get_dep_obj(feObj):
            deps = []
            for attr in feDeps:
                if hasattr(feObj, attr):
                    v = getattr(feObj, attr)
                    if is_valid_uuid(v):
                        deps.append(v)
                    elif v is not None and hasattr(v, 'uuid'):
                        deps.append(getattr(v, 'uuid'))
            return deps

        def get_dep_json(pDict):
            deps = []
            props = pDict.get('properties')
            if props is not None:
                for attr in feDeps:
                    v = props.get(attr)
                    if v is not None:
                        deps.append(v)
            return deps

        def dfs(mId, mProps):
            if mId in visited:
                return
            if mId in visiting:
                raise ValueError(
                    f"Circular dependency detected involving {mId}")
            visiting.add(mId)
            if mProps is not None:
                for dep in get_dependencies(mProps):
                    dfs(dep, None)
            visiting.remove(mId)
            visited.add(mId)
            sortedFEList.append(mId)

        if feDict is None:
            feDict = self.fesDict

        get_dependencies = get_dep_json if fromJSON else get_dep_obj

        for mId, mProps in feDict.items():
            dfs(mId, mProps)

        return sortedFEList

    def index_materials(self):  # materials and figure error objects
        def register_material(mat):
            if mat is None:
                return None

            for subAttr in ['tLayer', 'bLayer', 'coating', 'substrate']:
                if hasattr(mat, subAttr):
                    subMat = getattr(mat, subAttr)
        
                    if subMat is not None and hasattr(subMat, 'uuid'):
                        if subMat.uuid not in materialsDict:
                            materialsDict[subMat.uuid] = subMat
                            matNamesDict[subMat.name] = subMat.uuid
                            materialsDict.move_to_end(subMat.uuid, last=False)
                        setattr(mat, subAttr, subMat.uuid)

            if mat.uuid not in materialsDict:
                materialsDict[mat.uuid] = mat
                matNamesDict[mat.name] = mat.uuid

            return mat.uuid

        materialsDict = OrderedDict()
        matNamesDict = OrderedDict()
        feDict = OrderedDict()
        feNamesDict = OrderedDict()

        rmats = importlib.import_module(
            '.materials', package='xrt.backends.raycing')
        rfe = importlib.import_module(
            '.figure_error', package='xrt.backends.raycing')
        for i in range(len(inspect.stack())):
            scr_globals = inspect.stack()[i][0].f_globals

            for objName, objInstance in scr_globals.items():
                if isinstance(objInstance, (rmats.Element, rmats.Material,
                                            rmats.Multilayer)):
                    objId = getattr(objInstance, 'uuid', None)
                    if is_valid_uuid(objId) and objId not in materialsDict:
                        materialsDict[objId] = objInstance
                elif isinstance(objInstance, (rfe.FigureErrorBase)):
                    objId = getattr(objInstance, 'uuid', None)
                    if is_valid_uuid(objId) and objId not in feDict:
                        feDict[objId] = objInstance

        for ename, eLine in self.oesDict.items():
            oe = eLine[0]
            for attr in ['material', 'material2']:
                if not hasattr(oe, attr):
                    continue
            
                attrMat = getattr(oe, attr)
            
                if is_sequence(attrMat):
                    new_val = [register_material(m) for m in attrMat]
                else:
                    new_val = register_material(attrMat)
            
                setattr(oe, attr, new_val)

            for attr in ['figureError']:
                if hasattr(oe, attr):
                    newFE = getattr(oe, attr, None)
                    if hasattr(newFE, 'uuid') and newFE.uuid not in feDict:
                        feDict[newFE.uuid] = newFE
                        feNamesDict[newFE.name] = newFE.uuid
                        setattr(oe, 'figureError', newFE.uuid)

                    for subAttr in ['baseFE']:
                        if hasattr(newFE, subAttr):
                            subFE = getattr(newFE, subAttr, None)
                            if hasattr(subFE, 'uuid') and\
                                    subFE.uuid not in feDict:
                                feDict[subFE.uuid] = subFE
                                feNamesDict[subFE.name] = subFE.uuid
                                feDict.move_to_end(
                                        subFE.uuid,
                                        last=False)
                                setattr(subFE, 'baseFE', subFE.uuid)

        self.materialsDict.update(materialsDict)
        # self.matnamesToUUIDs.update(matNamesDict)
        self.fesDict.update(feDict)
        # self.fenamesToUUIDs.update(feNamesDict)

    def glow(self, scale=[], centerAt='', startFrom=0, colorAxis=None,
             colorAxisLimits=None, generator=None, generatorArgs=[], v2=False,
             **kwargs):
        if generator is not None:
            gen = generator(*generatorArgs)
            try:
                if sys.version_info < (3, 1):
                    gen.next()
                else:
                    next(gen)
            except StopIteration:
                return

        try:
            from ...gui import xrtGlow as xrtglow
        except ImportError as e:
            print("Cannot import xrtGlow. "
                  "If you run your script from an IDE, don't.")
            print(e)
            return

        from .run import run_process
        run_process(self)

        if self.blViewer is None:
            app = xrtglow.qt.QApplication.instance()
            if app is None:
                app = xrtglow.qt.QApplication(sys.argv)
            if v2:
                self.index_materials()
#                materialsDict = OrderedDict()
#                for ename, eLine in self.oesDict.items():
#                    oe = eLine[0]
#                    for attr in ['material', 'material2']:
#                        if hasattr(oe, attr):
#                            attrMat = getattr(oe, attr)
#                            if not is_sequence(attrMat):
#                                seqMat = (attrMat,)
#                            for newMat in seqMat:
#                                if newMat is not None and newMat.uuid not in\
#                                        materialsDict:
#                                    materialsDict[newMat.uuid] = newMat
#                                for subAttr in ['tlayer', 'blayer',
#                                                'coating', 'substrate']:
#                                    if hasattr(newMat, subAttr):
#                                        subMat = getattr(newMat, subAttr)
#                                        if subMat.uuid not in materialsDict:
#                                            materialsDict[subMat.uuid] = subMat
#                                            materialsDict.move_to_end(
#                                                    subMat.uuid,
#                                                    last=False)
#                self.materialsDict.update(materialsDict)
                _ = self.export_to_json()  # layoutStr is populated inside

                self.blViewer = xrtglow.xrtGlow(layout=self.layoutStr,
                                                **kwargs)
            else:
                rayPath = self.export_to_glow()
                self.blViewer = xrtglow.xrtGlow(rayPath)
            self.blViewer.generator = generator
            self.blViewer.generatorArgs = generatorArgs
            self.blViewer.customGlWidget.generator = generator
            self.blViewer.setWindowTitle("xrtGlow")
            self.blViewer.startFrom = startFrom
            self.blViewer.bl = self
            if scale:
                try:
                    self.blViewer.updateScaleFromGL(scale)
                except Exception:
                    pass
            if centerAt:
                try:
                    self.blViewer.centerEl(centerAt)
                except Exception:
                    pass
            if colorAxis:
                try:
                    colorCB = self.blViewer.colorControls[0]
                    colorCB.setCurrentIndex(colorCB.findText(colorAxis))
                except Exception:
                    pass
            if colorAxisLimits:
                try:
                    self.blViewer.customGlWidget.colorMin, \
                        self.blViewer.customGlWidget.colorMax = \
                        colorAxisLimits
                    self.blViewer.changeColorAxis(None, newLimits=True)
                except Exception:
                    pass

            self.blViewer.show()
            sys.exit(app.exec_())
        else:
            self.blViewer.show()

    def explore(self, plots=None):
        try:
            from ...gui import xrtQook as xrtqook
        except ImportError as e:
            print("Cannot import xrtGlow. "
                  "If you run your script from an IDE, don't.")
            print(e)
            return

        from .run import run_process
        run_process(self)

        if self.blExplorer is None:
            app = xrtqook.qt.QApplication.instance()
            if app is None:
                app = xrtqook.qt.QApplication(sys.argv)
            self.index_materials()
            layout = self.export_to_json()

            if plots is not None and not layout['plots']:
                layout['plots'].update(plots)

            self.blExplorer = xrtqook.XrtQook(loadLayout=layout)
            self.blExplorer.setWindowTitle("xrtQook")
            self.blExplorer.show()

    def export_to_glow(self, signal=None):
        def calc_weighted_center(beam):
            good = (beam.state == 1) | (beam.state == 2)
            intensity = beam.Jss[good] + beam.Jpp[good]
            totalI = np.sum(intensity)
            if totalI == 0:
                beam.wCenter = np.array([0., 0., 0.])
            else:
                beam.wCenter = np.array(
                    [np.sum(beam.x[good] * intensity),
                     np.sum(beam.y[good] * intensity),
                     np.sum(beam.z[good] * intensity)]) /\
                    totalI

        if self.flow is not None:
            beamDict = OrderedDict()
            rayPath = []
            outputBeamMatch = OrderedDict()
            oesDict = OrderedDict()
            totalStages = len(self.flow)
            for iseg, segment in enumerate(self.flow):
                try:
                    if signal is not None:
                        signalStr = "Processing {0} beams, %p% done.".format(
                            str(segment[0]))
                        signal.emit((float(iseg+1) / float(totalStages),
                                     signalStr))
                except Exception:
                    if _DEBUG_:
                        raise
                    else:
                        pass

                try:
                    methStr = str(segment[1])

                    oeStr = segment[0]
                    segOE = self.oesDict[oeStr][0]
                    if segOE is None:  # Protection from non-initialized OEs
                        continue
                    oesDict[oeStr] = self.oesDict[oeStr]
                    if 'beam' in segment[2].keys():
                        if str(segment[2]['beam']) == 'None':
                            continue
                        tmpBeamName = segment[2]['beam']
                        beamDict[tmpBeamName] = copy.deepcopy(
                            self.beamsDict[tmpBeamName])

                    if 'beamGlobal' in segment[3].keys():
                        outputBeamMatch[segment[3]['beamGlobal']] = oeStr

                    if len(re.findall('raycing.sou',
                                      str(type(segOE)).lower())):
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([oeStr, gBeamName, None, None])
                    elif len(re.findall(('expose'), methStr)) > 0 and\
                            len(re.findall(('expose_global'), methStr)) == 0:
                        gBeam = self.oesDict[oeStr][0].expose_global(
                            self.beamsDict[tmpBeamName])
                        gBeamName = '{}toGlobal'.format(
                            segment[3]['beamLocal'])
                        beamDict[gBeamName] = gBeam
                        if tmpBeamName in outputBeamMatch:
                            # if no good rays, the condition is False
                            rayPath.append([outputBeamMatch[tmpBeamName],
                                            tmpBeamName, oeStr, gBeamName])
                    elif len(re.findall(('double'), methStr)) +\
                            len(re.findall(('multiple'), methStr)) > 0:
                        lBeam1Name = segment[3]['beamLocal1']
                        gBeam = copy.deepcopy(self.beamsDict[lBeam1Name])
                        segOE.local_to_global(gBeam)
                        g1BeamName = '{}toGlobal'.format(lBeam1Name)
                        beamDict[g1BeamName] = gBeam
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, g1BeamName])
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([oeStr, g1BeamName,
                                       oeStr, gBeamName])
                    elif len(re.findall(('propagate'), methStr)) > 0:
                        if 'beamGlobal' in segment[3].keys():
                            lBeam1Name = segment[3]['beamGlobal']
                            gBeamName = lBeam1Name
                        else:
                            lBeam1Name = segment[3]['beamLocal']
                            gBeamName = '{}toGlobal'.format(lBeam1Name)
                        gBeam = copy.deepcopy(self.beamsDict[lBeam1Name])
                        segOE.local_to_global(gBeam)
                        beamDict[gBeamName] = gBeam
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, gBeamName])
                    else:
                        gBeamName = segment[3]['beamGlobal']
                        beamDict[gBeamName] = self.beamsDict[gBeamName]
                        rayPath.append([outputBeamMatch[tmpBeamName],
                                        tmpBeamName, oeStr, gBeamName])
                except Exception as e:
                    if _DEBUG_:
                        raise e
                    else:
                        continue

        totalBeams = len(beamDict)
        for itBeam, tBeam in enumerate(beamDict.values()):
            if signal is not None:
                try:
                    signalStr = "Calculating trajectory, %p% done."
                    signal.emit((float(itBeam+1)/float(totalBeams), signalStr))
                except Exception:
                    if _DEBUG_:
                        raise
                    else:
                        pass
            if tBeam is not None:
                calc_weighted_center(tBeam)
        return [rayPath, beamDict, oesDict]

    def init_oe_from_json(self, elProps, isString=True):
        oeParams = elProps.get('properties')

        if elProps.get('_object') is None:
            return
        else:
            oeModule, oeClass = elProps['_object'].rsplit('.', 1)
            oeModule = importlib.import_module(oeModule)
            defArgs = dict(get_params(elProps['_object']))

            if isString:
                initKWArgs = create_paramdict_oe(oeParams, defArgs, self)
            else:
                initKWArgs = oeParams

            try:
                _ = getattr(oeModule, oeClass)(**initKWArgs)
                initStatus = 0
            except Exception as e:  # TODO: Needs testing
                print(oeClass, "Init problem:", e)
                initStatus = 1
                # raise

        self.oenamesToUUIDs[oeParams['name']] = oeParams['uuid']
        self.update_flow_from_json(oeParams['uuid'], elProps)

        return initStatus

    def delete_oe_by_id(self, elid):
        for oename, oeid in self.oenamesToUUIDs.items():
            if oeid == elid:
                del self.oenamesToUUIDs[oename]
                break

        if elid in self.flowU:
            del self.flowU[elid]
            for eluuid, props in list(self.flowU.items()):
                for methName, methArgs in list(props.items()):
                    if methArgs.get('beam') == elid:
                        methArgs['beam'] = None

        if elid in self.beamsDictU:
            del self.beamsDictU[elid]

        if elid in self.oesDict:
            del self.oesDict[elid]

    def delete_mat_by_id(self, matid):
        for matname, tmpid in self.matnamesToUUIDs.items():
            if tmpid == matid:
                del self.matnamesToUUIDs[matname]
                break

        for elid, elLine in self.oesDict.items():
            elObj = elLine[0]
            for prop in ['material', 'material2']:
                if hasattr(elObj, prop) and \
                        getattr(elObj, prop) == self.materialsDict[matid]:
                    setattr(elObj, prop, None)

        for tmpid, matobj in self.materialsDict.items():
            for prop in ['tLayer', 'bLayer', 'coating', 'substrate']:
                if hasattr(matobj, prop) and \
                        getattr(matobj, prop) == self.materialsDict[matid]:
                    setattr(matobj, prop, None)

        if matid in self.materialsDict:
            del self.materialsDict[matid]

    def delete_fe_by_id(self, feid):
        for fename, tmpid in self.fenamesToUUIDs.items():
            if tmpid == feid:
                del self.fenamesToUUIDs[fename]
                break

        for elid, elLine in self.oesDict.items():
            elObj = elLine[0]
            for prop in ['figureError']:
                if hasattr(elObj, prop) and \
                        getattr(elObj, prop) == self.fesDict[feid]:
                    setattr(elObj, prop, None)

        for tmpid, feobj in self.fesDict.items():
            for prop in ['baseFE']:
                if hasattr(feobj, prop) and \
                        getattr(feobj, prop) == self.fesDict[feid]:
                    setattr(feobj, prop, None)

        if feid in self.fesDict:
            del self.fesDict[feid]

    def update_flow_from_json(self, oeid, methDict):
        for methStr, methArgs in methDict.items():
            if methStr in ['properties', '_object']:
                continue
            else:
                fArgs = {}
                isEmpty = False
                for argName, argVal in methArgs['parameters'].items():
                    if argName == "beam":
                        if is_valid_uuid(argVal):
                            fArgs[argName] = argVal
                        elif argVal == 'None' or argVal is None:
                            isEmpty = True
                        else:
                            beamTag = self.beamNamesDict.get(str(argVal))
                            if beamTag is not None:
                                fArgs[argName] = beamTag[0]
                            else:
                                print(argVal, "missing in beamNamesDict")
                                return
                    else:
                        fArgs[argName] = parametrize(argVal)

                if isEmpty:
                    self.flowU.pop(oeid, None)
                    continue

                self.flowU[oeid] = {methStr: fArgs}
                if 'output' in methArgs:
                    self.beamsDictU[oeid] = {}
                    for beamType, beamName in methArgs['output'].items():
                        self.beamNamesDict[str(beamName)] = (oeid, beamType)
                        self.beamsDictU[oeid][beamType] = None
                    break  # Normally just one method per element.

    def populate_oes_dict_from_json(self, dictIn):
        if not isinstance(dictIn, dict):
            return
        for elName, elProps in dictIn.items():
            if elName in ['properties', '_object']:
                continue
            if is_valid_uuid(elName):
                elKey = elName
                if 'name' not in elProps['properties']:
                    tmpName = "{}_{}".format(
                            str(dictIn['_object']).split('.')[-1],
                            np.random.randint(1000, 9999))
                    elProps['properties']['name'] = tmpName
                # TODO: check if the name is unique
#                self.oenamesToUUIDs[elProps['properties']['name']] = elKey
            else:
                elKey = str(uuid.uuid4())
                elProps['properties']['name'] = elName
            elProps['properties']['uuid'] = elKey
            _ = self.init_oe_from_json(elProps)  # oesDict populated in oe init

        for elName, eluuid in self.oenamesToUUIDs.items():
            if elName in dictIn:
                dictIn[eluuid] = dictIn.pop(elName)

    def init_material_from_json(self, matName, dictIn):
        matModule, matClass = dictIn['_object'].rsplit('.', 1)
        matModule = importlib.import_module(matModule)
        matParams = dictIn['properties']

        defArgs = dict(get_params(dictIn['_object']))
        initKWArgs = create_paramdict_mat(matParams, defArgs, self)

        if is_valid_uuid(matName):
            initKWArgs['uuid'] = matName
        else:
            initKWArgs['name'] = matName

        matObject = None
        initKWArgs['bl'] = self
        max_retries = 5
        delay = 0.001
        for retry in range(max_retries):
            try:
                matClass = getattr(matModule, matClass, None)
                if matClass is None:
                    return 1
                matObject = matClass(**initKWArgs)
                initStatus = 0
                break
    #            print("Initalized", matObject, initKWArgs)
            except FileNotFoundError:
                delay *= 5
                print("File read retry", retry, "delay", delay, "s")
                time.sleep(delay)
            except Exception as e:
                matObject = getattr(matModule, "EmptyMaterial")()
                matObject.uuid = initKWArgs.get('uuid')
                matObject.name = initKWArgs.get('name')
                initStatus = 1
                print(matClass, "Init problem. Falling back to EmptyMaterial")
                print(e)
                break
    #            raise

        self.matnamesToUUIDs[matObject.name] = matObject.uuid
        self.materialsDict[matObject.uuid] = matObject
        return initStatus

    def populate_materials_dict_from_json(self, dictIn):
        if not isinstance(dictIn, dict):
            return

        matSorted = self.sort_materials(matDict=dictIn, fromJSON=True)

        for matName in matSorted:
            matProps = dictIn.get(matName)
            if matProps is not None:
                _ = self.init_material_from_json(matName, matProps)

    def init_fe_from_json(self, feName, dictIn):
        feModule, feClass = dictIn['_object'].rsplit('.', 1)
        feModule = importlib.import_module(feModule)
        feParams = dictIn['properties']

        defArgs = dict(get_params(dictIn['_object']))
        initKWArgs = create_paramdict_fe(feParams, defArgs, self)

        if is_valid_uuid(feName):
            initKWArgs['uuid'] = feName
        else:
            initKWArgs['name'] = feName

        feObject = None
        initKWArgs['bl'] = self
        feClass = getattr(feModule, feClass, None)

        if feClass is None:
            return 1

        feObject = feClass(**initKWArgs)
        initStatus = 0

        self.fenamesToUUIDs[feObject.name] = feObject.uuid
        self.fesDict[feObject.uuid] = feObject
        return initStatus

    def populate_figerrors_dict_from_json(self, dictIn):
        if not isinstance(dictIn, dict):
            return

        feSorted = self.sort_figerrors(feDict=dictIn, fromJSON=True)

        for feName in feSorted:
            feProps = dictIn.get(feName)
            if feProps is not None:
                _ = self.init_fe_from_json(feName, feProps)

    def load_from_xml(self, openFileName):
        def xml_to_dict(element):
            # Recursively convert XML elements into a dictionary
            if len(element) == 0:  # Base case: if element has no children
                return element.text

            result = OrderedDict()
            for child in element:
                result[child.tag] = xml_to_dict(child)

            return result

        with open(openFileName, "r", encoding="utf-8") as f:
            treeImport = ET.parse(f)

#        treeImport = ET.parse(openFileName)
        root = treeImport.getroot()
        xml_dict = OrderedDict()
        xml_dict[root.tag] = xml_to_dict(root)
        self.deserialize(xml_dict)

    def load_from_json(self, openFileName):
        with open(openFileName, 'r', encoding="utf-8") as file:
            data = json.load(file)
        self.deserialize(data)

    def deserialize(self, data):
        if 'Project' not in data:
            data = {'Project': data}

        self.layoutStr = data
        beamlineName = next(islice(data['Project'].keys(), 2, 3))
        self.name = beamlineName
        beamlineInitKWargs = data['Project'][beamlineName].get('properties')
        if beamlineInitKWargs is None:
            beamlineInitKWargs = {}

        for key, value in beamlineInitKWargs.items():
            setattr(self, key, get_init_val(value))

        matDict = data['Project'].get('Materials')
        if matDict is not None:
            self.populate_materials_dict_from_json(matDict)

        feDict = data['Project'].get('FigureErrors')
        if feDict is not None:
            self.populate_figerrors_dict_from_json(feDict)

        self.populate_oes_dict_from_json(data['Project'][beamlineName])
        if 'flow' in data['Project'].keys():
            self.flowU = data['Project']['flow']

    def export_to_json(self):

        matDict = OrderedDict()
        feDict = OrderedDict()
        beamlineDict = OrderedDict()
        plotsDict = OrderedDict()
        runDict = None
        descriptionStr = None

        if self.layoutStr is not None:
            plotsDict = self.layoutStr['Project'].get('plots')
            runDict = self.layoutStr['Project'].get('run_ray_tracing')
            descriptionStr = self.layoutStr['Project'].get('description')

            if not isinstance(plotsDict, dict):
                plotsDict = {}

            if runDict is not None:
                runDict['beamLine'] = self.name

        for objName, objInstance in self.materialsDict.items():
            matRecord = OrderedDict()
            matRecord['properties'] = get_init_kwargs(objInstance,
                                                      compact=False)
            matRecord['_object'] = get_obj_str(objInstance)

            if not matRecord['properties']['name']:
                matRecord['properties']['name'] = objName
            field = objInstance.uuid if hasattr(
                    objInstance, 'uuid') else objName
            matDict[field] = matRecord

        for objName, objInstance in self.fesDict.items():
            feRecord = OrderedDict()
            feRecord['properties'] = get_init_kwargs(objInstance,
                                                     compact=False)
            feRecord['_object'] = get_obj_str(objInstance)

            if not feRecord['properties']['name']:
                feRecord['properties']['name'] = objName
            field = objInstance.uuid if hasattr(
                    objInstance, 'uuid') else objName
            feDict[field] = feRecord

        blArgs = get_init_kwargs(self, compact=False)

        beamlineDict['properties'] = blArgs
        beamlineDict['_object'] = get_obj_str(self)

        for oeid, oeline in self.oesDict.items():
            oeObj = oeline[0]
            oeRecord = OrderedDict()
            oeRecord['properties'] = get_init_kwargs(oeObj, compact=True,
                                                     blname=self.name,
                                                     resolveAuto=False)
            oeRecord['_object'] = get_obj_str(oeObj)
            if 'name' not in oeRecord['properties']:
                tmpName = "{}_{}".format(
                        str(oeRecord['_object']).split('.')[-1],
                        np.random.randint(1000, 9999))
                oeRecord['properties']['name'] = tmpName
            beamlineDict[oeid] = oeRecord

#        beamsDict = {}
#
#        # TODO: replace with flowU?
#        for segment in self.flow:
#            method = segment[1]
#            module = method.__module__
#            class_name = method.__class__.__name__
#            method_name = method.__name__
#            methDict = OrderedDict()
#            methDict['_object'] = "{0}.{1}.{2}".format(module, class_name,
#                                                       method_name)
#            methDict['parameters'] = {k: str(v) for k, v
#                                      in segment[2].items()}
#            methDict['output'] = segment[3]
#            beamlineDict[segment[0]][method_name] = methDict
#            for bname in segment[3].values():
#                beamsDict[bname] = None

        projectDict = OrderedDict()
        projectDict['Beams'] = self.beamNamesDict
        projectDict['Materials'] = matDict
        projectDict[self.name] = beamlineDict
        projectDict['FigureErrors'] = feDict
        projectDict['flow'] = self.flowU

        if plotsDict is not None:
            for plot, props in plotsDict.items():
                for argName, argVal in props.items():
                    if argName == 'beam' and argVal in self.beamNamesDict:
                        props[argName] = self.beamNamesDict.get(argVal)
                        break
        projectDict['plots'] = plotsDict
        projectDict['run_ray_tracing'] = runDict
        projectDict['description'] = descriptionStr

        self.layoutStr = {'Project': projectDict}
#        print("EXPORT:", self.layoutStr)
        return projectDict

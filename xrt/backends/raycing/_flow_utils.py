# -*- coding: utf-8 -*-

import sys
# import os
import numpy as np
# from itertools import compress
from collections import OrderedDict
from functools import wraps
import re
import inspect
import uuid
import importlib

from matplotlib.colors import hsv_to_rgb
if sys.version_info < (3, 1):
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec

from .singletons import is_sequence
from ._sets_units import (
    allBeamFields, orientationArgSet, shapeArgSet, derivedArgSet,
    renderOnlyArgSet, compoundArgs, dependentArgs, diagnosticArgs, allUnitsAng,
    allUnitsAngStr, allUnitsLen, allUnitsLenStr, allUnitsEnergy,
    allUnitsEnergyStr, allUnitsEmittance, allUnitsEmittanceStr,
    allUnitsCurrent, allUnitsCurrentStr, lengthUnitParams)

basestring = (str, bytes)

safe_globals = {
    'np': np,
    '__builtins__': {}  # Disable built-in functions
}


def auto_units_angle(angle, defaultFactor=1.):
    if isinstance(angle, basestring):
        if len(re.findall("auto", angle)) > 0:
            return angle
        elif len(re.findall("mrad", angle)) > 0:
            return float(angle.split("m")[0].strip())*1e-3
        elif len(re.findall("urad", angle)) > 0:
            return float(angle.split("u")[0].strip())*1e-6
        elif len(re.findall("nrad", angle)) > 0:
            return float(angle.split("n")[0].strip())*1e-9
        elif len(re.findall("rad", angle)) > 0:
            return float(angle.split("r")[0].strip())
        elif len(re.findall("deg", angle)) > 0:
            return np.radians(float(angle.split("d")[0].strip()))
        else:
            print("Could not identify the units")
            return angle
    elif angle is None or isinstance(angle, (list, tuple)):
        return angle
    else:
        return angle * defaultFactor


def append_to_flow(meth, bOut, frame):
    oe = meth.__self__
    if oe.bl is None:
        return
    if oe.bl.flowSource != 'legacy':
        return
    argValues = inspect.getargvalues(frame)
    fdoc = re.findall(r"Returned values:.*", meth.__doc__)
    if fdoc:
        fdoc = fdoc[0].replace("Returned values: ", '').split(',')
        if 'needNewGlobal' in argValues.args[1:]:
            if argValues.locals['needNewGlobal']:
                fdoc.insert(0, 'beamGlobal')

    kwArgsIn = OrderedDict()
    kwArgsOut = OrderedDict()
    for arg in argValues.args[1:]:
        if str(arg) == 'beam':
            kwArgsIn[arg] = id(argValues.locals[arg])
        else:
            kwArgsIn[arg] = argValues.locals[arg]

    for outstr, outbm in zip(list(fdoc), bOut):
        kwArgsOut[outstr.strip()] = id(outbm)

    oe.bl.flow.append([oe.uuid, meth.__func__, kwArgsIn, kwArgsOut])


def append_to_flow_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargsIn):
        methStr = func.__name__

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargsIn)
        bound_args.apply_defaults()

        kwargs = {k: v for k, v in bound_args.arguments.items() if k != 'self'}

        beamIn = None
        if 'beam' in kwargs:
            beamIn = 'beam'
        elif 'accuBeam' in kwargs:
            beamIn = 'accuBeam'

        toGlobal = kwargs.get('toGlobal', True)

        if beamIn and kwargs[beamIn] is not None:
            if hasattr(self, 'bl') and self.bl is not None and\
                    not self.bl.flowSource.endswith('refract'):
                if is_valid_uuid(kwargs[beamIn]):
                    beamId = kwargs[beamIn]
                    kwargs[beamIn] = self.bl.beamsDictU[beamId][
                            'beamGlobal' if toGlobal else 'beamLocal']
                else:
                    beamId = kwargs[beamIn].parentId

                if methStr != 'shine':
                    self.bl.auto_align(self, kwargs[beamIn])
        if hasattr(self, 'get_orientation'):
            self.get_orientation()

        result = func(self, **kwargs)

        if hasattr(self, 'bl') and self.bl is not None:
            if isinstance(result, tuple):
                for a in result:
                    a.parentId = self.uuid
                if len(result) > 2:
                    result[0].parentId = self.uuid
                    if self.bl.flowSource.endswith('refract'):
                        ret_dict = {}
                    else:
                        ret_dict = {'beamGlobal': result[0],
                                    'beamLocal1': result[1],
                                    'beamLocal2': result[2]}
                else:
                    ret_dict = {'beamGlobal': result[0],
                                'beamLocal': result[1]}
            else:
                result.parentId = self.uuid
                if methStr in ['propagate', 'expose']:
                    ret_dict = {'beamLocal': result}
                else:
                    ret_dict = {'beamGlobal' if toGlobal else
                                'beamLocal': result}
            if ret_dict:
                if 'beam' in kwargs:
                    kwargs['beam'] = beamId
                self.bl.flowU[self.uuid] = {methStr: kwargs}
                self.bl.beamsDictU[self.uuid] = ret_dict

        return result
    return wrapper


def set_name(elementClass, name):
    if name not in [None, 'None', '']:
        elementClass.name = name
    elif not hasattr(elementClass, 'name'):
        elementClass.name = '{0}{1}'.format(
            elementClass.__class__.__name__,
            elementClass.ordinalNum if hasattr(elementClass, 'ordinalNum')
            else '')


def vec_to_quat(vec, alpha):
    """ Quaternion from vector and angle"""

    return np.insert(vec*np.sin(alpha*0.5), 0, np.cos(alpha*0.5))


def multiply_quats(qf, qt):
    """Multiplication of quaternions"""

    return [qf[0]*qt[0]-qf[1]*qt[1]-qf[2]*qt[2]-qf[3]*qt[3],
            qf[0]*qt[1]+qf[1]*qt[0]+qf[2]*qt[3]-qf[3]*qt[2],
            qf[0]*qt[2]-qf[1]*qt[3]+qf[2]*qt[0]+qf[3]*qt[1],
            qf[0]*qt[3]+qf[1]*qt[2]-qf[2]*qt[1]+qf[3]*qt[0]]


def quat_vec_rotate(vec, q):
    """Rotate vector by a quaternion"""

    qn = np.copy(q)
    qn[1:] *= -1
    return multiply_quats(multiply_quats(
        q, vec_to_quat(vec, np.pi*0.25)), qn)[1:]


def get_init_val(value):
    if str(value) == 'round':
        return str(value)

    if "," in str(value):  # mixed list.
        while 'np.float64(' in value:
            pos1 = value.find('np.float64(')
            pos2 = value.find(')', pos1+1)
            value = value[:pos1] + value[pos1+11:pos2] + value[pos2+1:]
        s = str(value).replace(" ", "").replace("(", "[").replace(")", "]")
        if s.startswith('[['):  # nested list
            try:
                v = eval(s, safe_globals)
            except (NameError, SyntaxError):
                v = s
            return v

        paravalue = str(value).strip('[]() ')
        listvalue = []
        for c in paravalue.split(','):
            c_strip = str(c).strip()
            try:
                if not c_strip:
                    continue
                v = eval(c_strip, safe_globals)
            except (NameError, SyntaxError):
                v = c_strip
            listvalue.append(v)
        return listvalue

    try:
        return eval(str(value), safe_globals)
    except (NameError, SyntaxError):  # Intentionally string
        return str(value)


def get_params(objStr):  # Returns a collection of default parameters
    uArgs = OrderedDict()
    args = []
    argVals = []
#    objStr = "{0}.{1}".format(oeObj.__module__, type(oeObj).__name__)
    components = objStr.split('.')
    module_path = '.'.join(components[:-1])
    class_name = components[-1]
    moduleObj = importlib.import_module(module_path)
#    print("get_params for", class_name)
    try:
        objRef = getattr(moduleObj, class_name)
    except:  # TODO: remove if works correctly
        return {}

    isMethod = False
    if hasattr(objRef, 'hiddenParams'):
        hpList = objRef.hiddenParams
    else:
        hpList = []

    if inspect.isclass(objRef):
        for parent in (inspect.getmro(objRef))[:-1]:
            for namef, objf in inspect.getmembers(parent):
                if inspect.ismethod(objf) or inspect.isfunction(objf):
                    argSpec = getargspec(objf)
                    if namef == "__init__":
                        argnames = []
                        argdefaults = []
                        if argSpec[3] is not None:
                            argnames = argSpec[0][1:]
                            argdefaults = argSpec[3]
                        elif hasattr(argSpec, 'kwonlydefaults') and\
                                argSpec.kwonlydefaults:
                            argnames = argSpec.kwonlydefaults.keys()
                            argdefaults = argSpec.kwonlydefaults.values()
                        for arg, argVal in zip(argnames, argdefaults):
                            if arg == 'bl':
                                argVal = None
                            if arg not in args and arg not in hpList:
                                uArgs[arg] = argVal
                    if namef == "__init__" or namef.endswith("pop_kwargs"):
                        kwa = re.findall(r"(?<=kwargs\.pop).*?\)",
                                         inspect.getsource(objf),
                                         re.S)
                        if len(kwa) > 0:
                            kwa = [re.split(
                                ",", kwline.strip("\n ()"),
                                maxsplit=1) for kwline in kwa]
                            for kwline in kwa:
                                arg = kwline[0].strip("\' ")
                                if len(kwline) > 1:
                                    argVal = kwline[1].strip("\' ")
                                else:
                                    argVal = "None"
                                if arg not in args and arg not in hpList:
                                    uArgs[arg] = get_init_val(argVal)
                    attr = 'varkw' if hasattr(argSpec, 'varkw') else \
                        'keywords'
                    if namef == "__init__" and\
                            str(argSpec.varargs) == 'None' and\
                            str(getattr(argSpec, attr)) == 'None':
                        break  # To prevent the parent class __init__
            else:
                continue
            break
    elif inspect.ismethod(objRef) or inspect.isfunction(objRef):
        argList = getargspec(objRef)
        if argList[3] is not None:
            if objRef.__name__ == 'run_ray_tracing':
                uArgs = OrderedDict(zip(argList[0], argList[3]))
            else:
                isMethod = True
                uArgs = OrderedDict(zip(argList[0][1:], argList[3]))

    if hasattr(moduleObj, 'allArguments') and not isMethod:
        for argName in moduleObj.allArguments:
            if str(argName) in uArgs.keys():
                args.append(argName)
                argVals.append(uArgs[argName])
    else:
        args = list(uArgs.keys())
        argVals = list(uArgs.values())

    return zip(args, argVals)


def parametrize(value):
    value = get_init_val(value)
    if isinstance(value, tuple):
        value = list(value)
    return value


def create_paramdict_oe(paramDictStr, defArgs, beamLine=None):
    kwargs = OrderedDict()

    for paraname, paravalue in paramDictStr.items():
        if not isinstance(paravalue, str):
            paravalue = str(paravalue)  # TODO: temporary workaround
        if (paraname in defArgs and paravalue != str(defArgs[paraname])) or\
                paraname in ['bl', 'uuid']:

            if paraname == 'center':
                paravalue = paravalue.strip('[]() ')
                paravalue =\
                    [get_init_val(c.strip())
                     for c in str.split(
                     paravalue, ',')]
            elif paraname.startswith('limPhys'):
                paravalue = paravalue.strip('[]() ')
                paravalue =\
                    [get_init_val(c.strip())
                     for c in str.split(
                     paravalue, ',')]
            elif paraname.startswith('material'):
                if str(paravalue) in beamLine.matnamesToUUIDs:
                    paravalue = beamLine.matnamesToUUIDs[paravalue]
                else:
                    paravalue = paravalue.strip('[]() ')
                    paravalue =\
                        [get_init_val(c.strip())
                         for c in str.split(
                         paravalue, ',')]
            elif paraname.startswith('figure'):
                if str(paravalue) in beamLine.fenamesToUUIDs:
                    paravalue = beamLine.fenamesToUUIDs[paravalue]
#                if is_valid_uuid(paravalue):
#                    paravalue = beamLine.materialsDict[paravalue]
#                elif paravalue in beamLine.matnamesToUUIDs:
#                    paravalue = beamLine.materialsDict.get(
#                            beamLine.matnamesToUUIDs[paravalue])
            elif paraname == 'bl':
                paravalue = beamLine
            else:
                paravalue = parametrize(paravalue)
            kwargs[paraname] = paravalue

    return kwargs


def create_paramdict_mat(paramDictStr, defArgs, bl=None):
    kwargs = OrderedDict()

    for paraname, paravalue in paramDictStr.items():
        if (paraname in defArgs and paravalue != str(defArgs[paraname])) or\
                paravalue == 'bl':
            if paraname.lower() in ['tlayer', 'blayer', 'coating',
                                    'substrate']:
                if str(paravalue) in bl.matnamesToUUIDs:
                    paravalue = bl.matnamesToUUIDs[paravalue]
#                if is_valid_uuid(paravalue):
#                    paravalue = bl.materialsDict[paravalue]
#                elif paravalue in bl.matnamesToUUIDs:
#                    paravalue = bl.materialsDict.get(
#                            bl.matnamesToUUIDs[paravalue])
            else:
                paravalue = parametrize(paravalue)
            kwargs[paraname] = paravalue
    return kwargs

def create_paramdict_fe(paramDictStr, defArgs, bl=None):
    kwargs = OrderedDict()

    for paraname, paravalue in paramDictStr.items():
        if (paraname in defArgs and paravalue != str(defArgs[paraname])) or\
                paravalue == 'bl':
            if paraname.lower() in ['basefe']:
                if str(paravalue) in bl.fenamesToUUIDs:
                    paravalue = bl.fenamesToUUIDs[paravalue]
            else:
                paravalue = parametrize(paravalue)
            kwargs[paraname] = paravalue
    return kwargs

def get_obj_str(obj):
    return "{0}.{1}".format(obj.__module__, type(obj).__name__)


def get_init_kwargs(oeObj, compact=True, needRevG=False, blname=None,
                    resolveAuto=True):

    if needRevG:
        globRev = {str(v): k for k, v in globals().items()}

    defArgs = dict(get_params(get_obj_str(oeObj)))

    initArgs = {}
    for arg, val in defArgs.items():
        try:
            if hasattr(oeObj, arg):
                if arg == 'data':
                    continue
                if hasattr(oeObj, f'_{arg}Init') and not resolveAuto:
                    realval = getattr(oeObj, f'_{arg}Init')
#                    print(oeObj.name, f'_{arg}Init', realval)
                else:
                    realval = getattr(oeObj, arg)

                if arg == 'bl':
                    realval = blname
                if arg == 'elements':
                    realval = [x.name for x in realval]

                if str(arg).lower().startswith(
                        ('material', 'coating', 'substrate', 'tlay', 'blay')):
                    if is_sequence(realval):
                        outv = []
                        for trval in realval:
                            if hasattr(trval, 'uuid'):
                                trval = trval.uuid
                            elif needRevG:
                                trval = globRev[str(trval)]
                            else:  # already uuid or something is wrong
                                pass
                            outv.append(trval)
                        realval = outv
                    else:
                        if hasattr(realval, 'uuid'):
                            realval = realval.uuid
                        elif needRevG:
                            realval = globRev[str(realval)]
                        else:  # already uuid or something is wrong
                            pass

                if str(arg).lower().startswith(
                        ('figureerr', 'basefe')):
                    if hasattr(realval, 'uuid'):
                        realval = realval.uuid
                    elif needRevG:
                        realval = globRev[str(realval)]
                    else:
                        pass
#                        print("Cannot resolve material")
                if realval != val:
                    if isinstance(realval, tuple):
                        realval = list(realval)
                    defArgs[arg] = str(realval)
                    if compact:
                        initArgs[arg] = str(realval)
                else:
                    defArgs[arg] = str(val)
            else:
                defArgs[arg] = str(val)

        except RuntimeError:  # Unclear error on plot init
            pass

    return initArgs if compact else defArgs


def is_valid_uuid(uuid_string):
    try:
        _ = uuid.UUID(str(uuid_string))
        return True
    except ValueError:
        return False


def run_process_from_file(beamLine):
    outDict = {}

    for oeid, meth in beamLine.flowU.items():
        oe = beamLine.oesDict[oeid][0]
        for func, fkwargs in meth.items():
            getattr(oe, func)(**fkwargs)
    for beamName, beamTag in beamLine.beamNamesDict.items():
        outDict[beamName] = beamLine.beamsDictU[beamTag[0]][beamTag[1]]

    return outDict


def build_hist(beam, limits=None, isScreen=False, shape=[256, 256],
               cDataFunc=None, cLimits=None):
    """This is a simplified standalone implementation of
    multipro.do_hist2d()
    cData is one of get_NNN methods or None. In the latter case the function
    returns only intensity histogram
    """
    shape = [int(val) for val in shape]  # sometimes arrives as float
    good = (beam.state == 1) | (beam.state == 2)
    if isScreen:
        x, y, z = beam.x[good], beam.z[good], beam.y[good]
    else:
        x, y, z = beam.x[good], beam.y[good], beam.z[good]
    goodlen = len(beam.x[good])
    hist2dRGB = None
    hist2d = np.zeros((shape[1], shape[0]), dtype=np.float64)

    if limits is None and goodlen > 0:
        limits = np.array([[np.min(x), np.max(x)],
                           [np.min(y), np.max(y)],
                           [np.min(z), np.max(z)]])

    if goodlen > 0:
        beamLimits = [limits[1], limits[0]] or None
        flux = beam.Jss[good]+beam.Jpp[good]
        hist2d, yedges, xedges = np.histogram2d(
            y, x, bins=[shape[1], shape[0]], range=beamLimits, weights=flux)

    if cDataFunc is not None:
        hist2dRGB = np.zeros((shape[1], shape[0], 3), dtype=np.float64)
        cData = cDataFunc(beam)[good]
        if cLimits is None:
            colorMin, colorMax = np.min(cData), np.max(cData)
        else:
            colorMin, colorMax = cLimits[0], cLimits[-1]
        cData01 = ((cData - colorMin) * 0.85 /
                   (colorMax - colorMin)).reshape(-1, 1)

        cDataHSV = np.dstack(
            (cData01, np.ones_like(cData01) * 0.85,
             flux.reshape(-1, 1)))
        cDataRGB = (hsv_to_rgb(cDataHSV)).reshape(-1, 3)

        hist2dRGB = np.zeros((shape[0], shape[1], 3), dtype=np.float64)
        hist2d = None
        if len(beam.x[good]) > 0:
            for i in range(3):  # over RGB components
                hist2dRGB[:, :, i], yedges, xedges = np.histogram2d(
                    y, x, bins=shape, range=beamLimits,
                    weights=cDataRGB[:, i])

        hist2dRGB /= np.max(hist2dRGB)
        hist2dRGB = np.uint8(hist2dRGB*255)

    return hist2d, hist2dRGB, limits

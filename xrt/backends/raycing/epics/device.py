# -*- coding: utf-8 -*-
import copy
import numpy as np
import os
import queue
import threading
# from itertools import compress
from functools import partial
from multiprocessing import Process, Queue
import re

from ..physconsts import SIE0, CH  # analysis:ignore
from .._sets_units import (
    derivedArgSet, diagnosticArgs, orientationArgSet, shapeArgSet)
from .._named_arrays import NamedArrayFactory, Center, Limits, Opening, Image2D

DEFAULT_IMAGE_WAVEFORM_LENGTH = 1024 * 1024

HEADLESS_OUTPUT_POLICY = {
    'beams': False,
    'histograms': True,
    'auto_properties': True,
    'footprints': False,
    'diagnostics': True,
    'progress': False,
}

COMPOUND_RECORD_FIELDS = {
    'center': ('x', 'y', 'z'),
    'limPhysX': ('lmin', 'lmax'),
    'limPhysY': ('lmin', 'lmax'),
    'limPhysX2': ('lmin', 'lmax'),
    'limPhysY2': ('lmin', 'lmax'),
    'histShape': ('width', 'height'),
}


def _initial_energy_from_angle(material, angle):
    try:
        angle = np.asarray(angle, dtype=float)
        angle = float(angle.flat[0] if angle.shape else angle)
        denominator = 2 * material.d * np.sin(angle)
        if denominator == 0:
            return 0.
        return float(np.abs(CH / denominator))
    except (TypeError, ValueError, AttributeError):
        return 0.


def _initial_numeric_field(value, index, field, default=0.):
    if isinstance(value, dict):
        field_value = value.get(field, default)
    elif hasattr(value, field):
        field_value = getattr(value, field)
    else:
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) > index:
            field_value = value[index]
        else:
            field_value = default

    try:
        return float(field_value)
    except (TypeError, ValueError):
        return default


def _initial_numeric_value(value, default=0.):
    try:
        value = np.asarray(value, dtype=float)
        return float(value.flat[0] if value.shape else value)
    except (TypeError, ValueError):
        return default


def _compound_field_value(value, index, field):
    if isinstance(value, dict):
        return value.get(field)
    if hasattr(value, field):
        return getattr(value, field)
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return None


def _record_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def update_epics_readback(epics, oeid, key, value):
    if epics is None or oeid is None or key is None:
        return False

    rbv_map = getattr(epics, 'rbv_map', {})
    elementReadbacks = rbv_map.get(oeid, {})
    updated = False

    if key in COMPOUND_RECORD_FIELDS:
        for index, field in enumerate(COMPOUND_RECORD_FIELDS[key]):
            record = elementReadbacks.get(f'{key}.{field}')
            if record is None:
                continue
            fieldValue = _compound_field_value(value, index, field)
            if fieldValue is None:
                continue
            record.set(_record_value(fieldValue))
            updated = True
        return updated

    if key == 'blades' and isinstance(value, dict):
        for field, fieldValue in value.items():
            record = elementReadbacks.get(f'blades.{field}')
            if record is None:
                continue
            record.set(_record_value(fieldValue))
            updated = True
        return updated

    record = elementReadbacks.get(key)
    if record is not None:
        record.set(_record_value(value))
        updated = True
    return updated


def resolve_epics_record(default_record, epics_map):
    if not epics_map:
        return default_record
    if default_record not in epics_map:
        return None
    mapped = epics_map.get(default_record)
    return default_record if mapped is None else mapped


def resolve_epics_readback(default_record, epics_map):
    base_record = resolve_epics_record(default_record, epics_map)
    default_rbv = f'{default_record}_RBV'

    if epics_map and default_rbv in epics_map:
        mapped_rbv = resolve_epics_record(default_rbv, epics_map)
        return mapped_rbv, base_record

    if base_record is None:
        return None, None

    return f'{base_record}_RBV', base_record


def to_valid_var_name(name, default='unnamed'):
    # Replace invalid characters with underscores
    var_name = re.sub(r'\W|^(?=\d)', '_', name.strip())

    # Ensure the name is not empty or a Python keyword
    if not var_name or not re.match(r'[A-Za-z_]', var_name[0]):
        var_name = f"{default}_{var_name}"

    # Avoid Python reserved keywords
    import keyword
    if keyword.iskeyword(var_name):
        var_name += '_var'

    return var_name


#class DynamicBeamline:
#    """Placeholder for a headless dynamic beamline"""
#    def __init__(self, bl, epicsPrefix=None):
#
#        self.epicsPrefix = epicsPrefix
#
#        self.beamline.deserialize(beamLayout)
#        self.input_queue = Queue()
#        self.output_queue = Queue()
#
#        self.calc_process = Process(
#                target=propagationProcess,
#                args=(self.input_queue, self.output_queue))
#        self.calc_process.start()
#        msg_init_bl = {
#                "command": "create",
#                "object_type": "beamline",
#                "kwargs": beamLayout
#                }
#        self.input_queue.put(msg_init_bl)
#        self.loopRunning = True
#
#        self.timer = qt.QTimer()
#        self.timer.timeout.connect(
#            partial(self.check_progress, self.output_queue))
#        self.timer.start(10)  # Adjust the interval as needed
#        if self.epicsPrefix is not None and self.renderingMode == 'dynamic':
#            try:
#                os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
#                os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
#                self.epicsInterface = EpicsBeamline(
#                        bl=self.beamline,
#                        prefix=epicsPrefix,
#                        callback=self.update_beamline_async)
##                self.build_epics_device(epicsPrefix, softioc, builder,
##                                        asyncio_dispatcher)
#            except ImportError:
#                print("pythonSoftIOC not installed")
#                self.epicsPrefix = None
#
#    async def update_beamline_async(self, oeid, argName, argValue):
#        self.update_beamline(oeid, {argName: argValue})
#
#    def update_beamline(self, oeid, kwargs):
#        for argName, argValue in kwargs.items():
#            if oeid is None:
#                if self.epicsPrefix is not None:
#                    if argName == 'Acquire':
#                        self.epicsInterface.pv_records['AcquireStatus'].set(1)
#                        if str(argValue) == '1':
#                            if hasattr(self, 'input_queue'):
#                                self.input_queue.put({
#                                            "command": "run_once",
#                                            "object_type": "beamline"
#                                            })
#                    elif argName == 'AutoUpdate':
#                        if hasattr(self, 'input_queue'):
#                            self.input_queue.put({
#                                        "command": "auto_update",
#                                        "object_type": "beamline",
#                                        "kwargs": {"value": int(argValue)}
#                                        })
#                return
#
#            oe = self.beamline.oesDict[oeid][0]
#
#            args = argName.split('.')
#            arg = args[0]
#            if len(args) > 1:
#                field = args[-1]
#                if field == 'energy':
#                    if arg == 'bragg':
#                        argValue = [float(argValue)]
#                    else:
#                        argValue = oe.material.get_Bragg_angle(float(argValue))
#                else:
#                    arrayValue = getattr(oe, arg)
#                    setattr(arrayValue, field, argValue)
#                    argValue = arrayValue
#
#            # updating local beamline tree
#            setattr(oe, arg, argValue)
#            if arg in orientationArgSet:
#                self.meshDict[oeid].update_transformation_matrix()
#            elif arg in shapeArgSet:
#                self.needMeshUpdate = oeid
#
#            # updating the beamline model in the runner
#        if self.epicsPrefix is not None:
#            self.epicsInterface.pv_records['AcquireStatus'].set(1)
#        message = {"command": "modify",
#                   "object_type": "beamline",
#                   "uuid": oeid,
#                   "kwargs": kwargs.copy()
##                        "kwargs": {arg: argValue.tolist() if isinstance(
##                                argValue, np.ndarray) else argValue}
#                        }
#        if hasattr(self, 'input_queue'):
#            self.input_queue.put(message)
#
#    def check_progress(self, progress_queue):
##        progress = None
#        while not progress_queue.empty():
#            msg = progress_queue.get()
#            if 'beam' in msg:
##                print(msg['sender_name'], msg['sender_id'], msg['beam'])
#                for beamKey, beam in msg['beam'].items():
#                    self.update_beam_footprint(beam, (msg['sender_id'],
#                                                      beamKey))
#                    self.beamline.beamsDictU[msg['sender_id']][beamKey] = beam
#            elif 'histogram' in msg and self.epicsPrefix is not None:
#                histPvName = f'{to_valid_var_name(msg["sender_name"])}:image'
#                if histPvName in self.epicsInterface.pv_records:
#                    imgHist = np.flipud(msg['histogram'])  # Appears flipped
#                    self.epicsInterface.pv_records[histPvName].set(
#                            imgHist.flatten())
#            elif 'repeat' in msg:
#                print("Total repeats:", msg['repeat'])
#                if self.epicsPrefix is not None:
#                    self.epicsInterface.pv_records['AcquireStatus'].set(0)
#                self.glDraw()
#
#    def close_calc_process(self):
#        if hasattr(self, 'calc_process') and\
#                self.calc_process is not None:
#            self.input_queue.put(msg_exit)
#            self.calc_process.join(timeout=1)
#            if self.calc_process.is_alive():
#                self.calc_process.terminate()
#                self.calc_process.join()


class EpicsDevice:
    def __init__(self, bl, epicsPrefix, epicsMap, callback,
                 imageMaxLength=None):
        u"""
        Create SoftIOC records for a beamline and connect writable PVs to xrt
        property updates.


        *bl*: raycing.BeamLine
            Beamline instance whose elements are exposed through EPICS records.

        *epicsPrefix*: str
            Prefix passed to ``softioc.builder.SetDeviceName()``.
            The final PV names are formed as::

                <epicsPrefix><record_name>        if epicsPrefix is empty
                <epicsPrefix>:<record_name>       if epicsPrefix is not empty

            Workflow control records ``Acquire``, ``AcquireStatus`` and
            ``AutoUpdate`` are always created when this class is instantiated.

        *epicsMap*: dict or None
            Optional mapping from default record names to custom record names.
            Keys must use the default per-element record names generated by
            xrt, for example::

                {
                    "Mono:ENERGY": "MONO:E",
                    "Slit1:blades:left": "SLIT1:L",
                    "Screen1:image": "CAM1:ArrayData"
                }

            Note that ``epicsMap`` values replace only the per-record suffix.
            The final PV name is still formed as
            ``f"{epicsPrefix}:{mapped_name}"`` when ``epicsPrefix`` is not
            empty.

            If ``epicsMap`` is empty or ``None``, all supported records are
            created with their default names.

            Readback records follow the same map automatically. If a writable
            record is included, its readback is created as
            ``"<mapped_record>_RBV"``. An explicit ``"<default_record>_RBV"``
            key in ``epicsMap`` overrides that generated readback name.

            The mapping changes only the exposed EPICS record names. Internal
            xrt property paths used by callbacks and ``pv_map`` remain
            unchanged, for example ``"pitch"``, ``"center.x"``,
            ``"histShape.width"`` and ``"blades.left"``.

        *callback*: callable
            Callback invoked on record updates as::

                callback(oeid, argName, value)

            where ``argName`` is an internal xrt property path.

        *imageMaxLength*: int or None
            Maximum number of pixels allocated for screen image waveform
            records. The EPICS waveform length is fixed at IOC startup, while
            screen ``histShape`` can change at runtime. If omitted, image
            waveform records reserve space for at least a 1024 x 1024 image,
            or the initial screen ``histShape`` if it is larger.

        Notes
        -----
        Records are created once during IOC initialization and cannot be added,
        removed or renamed after ``builder.LoadDatabase()`` and
        ``softioc.iocInit()`` have been called.


        """

        self.bl = bl
        self.epicsPrefix = epicsPrefix + ":" if epicsPrefix else ""
        self.epicsMap = epicsMap or {}
        self.imageMaxLength = imageMaxLength
        self.image_lengths = {}
        self.pv_map = {}
        self.rbv_map = {}
        self.dbl = set()

        try:
            from softioc import softioc, builder, asyncio_dispatcher
        except ImportError:
            print("Missing softioc dependencies")
            return
        # Create an asyncio dispatcher, the event loop is now running
        self.dispatcher = asyncio_dispatcher.AsyncioDispatcher()

        # Set the record prefix
        builder.SetDeviceName(epicsPrefix)
        pv_records = {}
        pvFields = {'name'} | orientationArgSet | shapeArgSet

        def add_numeric_rbv(oeid, key, default_pvname, initial_value=0.):
            if key in self.rbv_map[oeid]:
                return
            rbvname, _ = resolve_epics_readback(
                default_pvname, self.epicsMap)
            if rbvname is None:
                return
            pv_records[rbvname] = builder.aIn(
                rbvname,
                initial_value=_initial_numeric_value(initial_value))
            self.rbv_map[oeid][key] = pv_records[rbvname]

        pv_records['Acquire'] = builder.boolOut(
            'Acquire', ZNAM=0, ONAM=1,
            initial_value=0, always_update=True,
            on_update=partial(callback, None, 'Acquire'))

        pv_records['AcquireStatus'] = builder.boolIn(
            'AcquireStatus', ZNAM=0, ONAM=1,
            initial_value=0)

        pv_records['AutoUpdate'] = builder.boolOut(
            'AutoUpdate', ZNAM=0, ONAM=1,
            initial_value=1, always_update=True,
            on_update=partial(callback, None, 'AutoUpdate'))

        for oeid, oeline in bl.oesDict.items():
            oeObj = oeline[0]
            oename = to_valid_var_name(oeObj.name)
            oePvFields = set(pvFields)
            if hasattr(oeObj, 'shine') and hasattr(oeObj, 'nrays'):
                oePvFields.add('nrays')

            self.pv_map[oeid] = {}
            self.rbv_map[oeid] = {}

            if hasattr(oeObj, 'material') and oeObj.material is not None:
                if hasattr(oeObj.material, 'get_Bragg_angle'):
                    if hasattr(oeObj, 'bragg'):
                        e_field = 'bragg.energy'
                        try:
                            angle = oeObj.bragg - oeObj.braggOffset
                        except (TypeError, ValueError):
                            angle = oeObj.bragg
                    else:
                        e_field = 'pitch.energy'
                        angle = oeObj.pitch
                    initial_e = _initial_energy_from_angle(
                        oeObj.material, angle)
                    default_pvname = f'{oename}:ENERGY'
                    pvname = resolve_epics_record(
                        default_pvname, self.epicsMap)
                    if pvname is not None:
                        pv_records[pvname] = builder.aOut(
                                pvname,
                                initial_value=initial_e,
                                always_update=True,
                                on_update=partial(callback, oeid, e_field))
                        self.pv_map[oeid]['bragg'] = pv_records[pvname]

            if hasattr(oeObj, 'expose') and oeObj.limPhysX is not None:
                default_pvname = f'{oename}:image'
                histShape = getattr(oeObj, 'histShape')
                imageLength = int(histShape[0]*histShape[1])
                if self.imageMaxLength is not None:
                    imageLength = max(imageLength, int(self.imageMaxLength))
                else:
                    imageLength = max(imageLength,
                                      DEFAULT_IMAGE_WAVEFORM_LENGTH)
                pvname = resolve_epics_record(default_pvname, self.epicsMap)
                if pvname is not None:
                    pv_records[pvname] = builder.WaveformIn(
                        pvname,
                        length=imageLength
                        )
                    self.pv_map[oeid]['image'] = pv_records[pvname]
                    self.image_lengths[oeid] = imageLength

                for fIndex, field in enumerate(['width', 'height']):
                    default_pvname = f'{oename}:histShape:{field}'
                    pvname = resolve_epics_record(
                        default_pvname, self.epicsMap)
                    if pvname is not None:
                        dimObj = getattr(oeObj, 'histShape')
                        if dimObj is not None:
                            pv_records[pvname] = builder.aOut(
                                pvname,
                                initial_value=dimObj[fIndex],
                                always_update=True,
                                on_update=partial(callback,
                                                  oeid, f'histShape.{field}'))
                            self.pv_map[oeid][f'histShape.{field}'] =\
                                pv_records[pvname]

            for argName in oePvFields:
                if argName in ['shape', 'renderStyle']:
                    continue
                if hasattr(oeObj, argName):
                    if argName in ['name', 'rotationSequence']:
                        default_pvname = f'{oename}:{argName}'
                        pvname = resolve_epics_record(
                            default_pvname, self.epicsMap)
                        if pvname is not None:
                            pv_records[pvname] = builder.stringOut(
                                pvname,
                                initial_value=str(getattr(oeObj, argName)),
                                always_update=True,
                                on_update=partial(callback, oeid, argName))
                            self.pv_map[oeid][argName] = pv_records[pvname]
                    elif argName in ['center']:
                        cntrObj = getattr(oeObj, argName)
                        for fIndex, field in enumerate(['x', 'y', 'z']):
                            default_pvname = f'{oename}:{argName}:{field}'
                            pvname = resolve_epics_record(
                                default_pvname, self.epicsMap)
                            if pvname is not None:
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=_initial_numeric_field(
                                        cntrObj, fIndex, field),
                                    always_update=True,
                                    on_update=partial(callback, oeid,
                                                      f'{argName}.{field}'))
                                self.pv_map[oeid][f'{argName}.{field}'] =\
                                    pv_records[pvname]
                                add_numeric_rbv(
                                    oeid, f'{argName}.{field}',
                                    default_pvname, _initial_numeric_field(
                                        cntrObj, fIndex, field))
                    elif argName in ['limPhysX', 'limPhysY', 'limPhysX2',
                                     'limPhysY2']:  # TODO: startswith?
                        for fIndex, field in enumerate(['lmin', 'lmax']):
                            default_pvname = f'{oename}:{argName}:{field}'
                            pvname = resolve_epics_record(
                                default_pvname, self.epicsMap)
                            if pvname is not None:
                                limObj = getattr(oeObj, argName)
                                if isinstance(limObj, Limits):
                                    pv_records[pvname] = builder.aOut(
                                        pvname,
                                        initial_value=limObj[fIndex],
                                        always_update=True,
                                        on_update=partial(
                                                callback, oeid,
                                                f'{argName}.{field}'))
                                    self.pv_map[oeid][f'{argName}.{field}'] =\
                                        pv_records[pvname]
                    elif argName == 'blades':
                        bladesObj = getattr(oeObj, 'blades')
                        if isinstance(bladesObj, dict):
                            for field, value in bladesObj.items():
                                default_pvname = f'{oename}:blades:{field}'
                                pvname = resolve_epics_record(
                                    default_pvname, self.epicsMap)
                                if pvname is not None:
                                    pv_records[pvname] = builder.aOut(
                                        pvname,
                                        initial_value=value,
                                        always_update=True,
                                        on_update=partial(
                                                callback, oeid,
                                                f'blades.{field}'))
                                    self.pv_map[oeid][f'blades.{field}'] =\
                                        pv_records[pvname]
                                    add_numeric_rbv(
                                        oeid, f'blades.{field}',
                                        default_pvname, value)
                    else:
                        default_pvname = f'{oename}:{argName}'
                        pvname = resolve_epics_record(
                            default_pvname, self.epicsMap)
                        if pvname is not None:
                            initial_value = getattr(oeObj, argName)
                            if isinstance(initial_value, (int, float,
                                                         np.number)):  # TODO: process sequence args
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=initial_value,
                                    always_update=True,
                                    on_update=partial(callback, oeid, argName))
                                self.pv_map[oeid][argName] = pv_records[pvname]

            for argName in derivedArgSet:
                if argName == 'center' or not hasattr(oeObj, argName):
                    continue
                default_pvname = f'{oename}:{argName}'
                add_numeric_rbv(
                    oeid, argName, default_pvname,
                    initial_value=getattr(oeObj, argName))

            for argName in diagnosticArgs:
                if not hasattr(oeObj, argName):
                    continue
                default_pvname = f'{oename}:{argName}'
                add_numeric_rbv(
                    oeid, argName, default_pvname,
                    initial_value=getattr(oeObj, argName))

        [print(f'{self.epicsPrefix}{recName}') for recName in pv_records]
        builder.LoadDatabase()
        softioc.iocInit(self.dispatcher)
        self.pv_records = pv_records
        for key in self.pv_records.keys():
            self.dbl.add(f'{epicsPrefix}{key}')


class DynamicBeamline:
    """Headless EPICS controller for dynamic raycing propagation.

    The controller is deliberately not a live beamline model. It uses a
    temporary beamline loaded from a Qook XML/JSON layout only to build the
    EPICS record schema. Runtime PV writes are forwarded to the calculation
    process, whose own BeamLine instance remains authoritative.
    """

    def __init__(self, layout=None, fileName=None, epicsPrefix="",
                 epicsMap=None, imageMaxLength=None, localEpics=True,
                 queueMaxSize=0, outputPollInterval=0.1,
                 with_epics_histograms=True, output_policy=None,
                 startOutputThread=True, start=True):
        if layout is not None and fileName is not None:
            raise ValueError("Use either layout or fileName, not both.")

        self.layout, schema_beamline = self._load_layout(layout, fileName)
        self._schema_beamline = schema_beamline
        self.object_types = self._make_object_type_map(schema_beamline)

        self.epicsPrefix = epicsPrefix
        self.epicsMap = epicsMap or {}
        self.imageMaxLength = imageMaxLength
        self.localEpics = localEpics
        self.queueMaxSize = queueMaxSize
        self.outputPollInterval = outputPollInterval
        self.with_epics_histograms = with_epics_histograms
        self.startOutputThread = startOutputThread
        self.output_policy = HEADLESS_OUTPUT_POLICY.copy()
        if output_policy is not None:
            self.output_policy.update(output_policy)

        self.input_queue = None
        self.output_queue = None
        self.calc_process = None
        self.epicsInterface = None
        self._output_thread = None
        self._running = False
        self._closed = False
        self.autoUpdate = True

        if start:
            self.start()

    @staticmethod
    def _load_layout(layout, fileName):
        from ..beamline import BeamLine

        source = fileName if fileName is not None else layout
        if source is None:
            raise ValueError("layout or fileName must be specified.")

        if hasattr(source, 'layoutStr'):
            layout_data = getattr(source, 'layoutStr')
            if layout_data is None:
                layout_data = source.export_to_json()
            schema_beamline = BeamLine()
            schema_beamline.deserialize(copy.deepcopy(layout_data))
            return copy.deepcopy(layout_data), schema_beamline

        if isinstance(source, dict):
            layout_data = copy.deepcopy(source)
            schema_beamline = BeamLine()
            schema_beamline.deserialize(copy.deepcopy(layout_data))
            return layout_data, schema_beamline

        if isinstance(source, (str, bytes, os.PathLike)):
            schema_beamline = BeamLine(fileName=os.fspath(source))
            if schema_beamline.layoutStr is None:
                raise ValueError(
                    "The supplied layout file did not produce layoutStr.")
            return copy.deepcopy(schema_beamline.layoutStr), schema_beamline

        raise TypeError(
            "layout must be a dict, a BeamLine-like object, or a path.")

    @staticmethod
    def _make_object_type_map(beamline):
        object_types = {}
        for objuuid in getattr(beamline, 'oesDict', {}):
            object_types[objuuid] = 'oe'
        for objuuid in getattr(beamline, 'materialsDict', {}):
            object_types[objuuid] = 'mat'
        for objuuid in getattr(beamline, 'fesDict', {}):
            object_types[objuuid] = 'fe'
        return object_types

    @staticmethod
    def _configure_local_epics():
        os.environ["EPICS_CAS_INTF_ADDR_LIST"] = "127.0.0.1"
        os.environ["EPICS_CAS_BEACON_ADDR_LIST"] = "127.0.0.1"
        os.environ["EPICS_CAS_AUTO_BEACON_ADDR_LIST"] = "NO"
        os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
        os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

    def start(self):
        if self.calc_process is not None and self.calc_process.is_alive():
            return

        from .._flow import propagationProcess

        self.input_queue = Queue(maxsize=self.queueMaxSize)
        self.output_queue = Queue(maxsize=self.queueMaxSize)
        self.calc_process = Process(
            target=propagationProcess,
            args=(self.input_queue, self.output_queue,
                  self.with_epics_histograms, self.output_policy))
        self.calc_process.start()
        self.input_queue.put({
            "command": "create",
            "object_type": "beamline",
            "kwargs": copy.deepcopy(self.layout)
        })
        self.input_queue.put({"command": "start", "run": False})

        if self.localEpics:
            self._configure_local_epics()

        if self.epicsPrefix is not None:
            try:
                self.epicsInterface = EpicsDevice(
                    bl=self._schema_beamline,
                    epicsPrefix=self.epicsPrefix,
                    epicsMap=self.epicsMap,
                    callback=self.update_beamline_async,
                    imageMaxLength=self.imageMaxLength)
                if not hasattr(self.epicsInterface, 'pv_records'):
                    raise RuntimeError("EPICS softIOC initialization failed.")
            except Exception:
                self.close()
                raise

        self._schema_beamline = None
        self._running = True
        self._closed = False
        if self.startOutputThread:
            self._output_thread = threading.Thread(
                target=self._output_loop,
                name="xrt-dynamic-beamline-output",
                daemon=True)
            self._output_thread.start()

    async def update_beamline_async(self, oeid, argName, argValue):
        self.update_beamline(oeid, {argName: argValue})

    def update_beamline(self, oeid, kwargs):
        for argName, argValue in kwargs.items():
            if oeid is None:
                if argName == 'Acquire' and self._bool_value(argValue):
                    self.request_propagation_once()
                elif argName == 'AutoUpdate':
                    self.set_auto_update(argValue)
                continue

            object_type = self.object_types.get(oeid, 'oe')
            if self.autoUpdate:
                self.set_acquire_status(1)
            self.input_queue.put({
                "command": "modify",
                "object_type": object_type,
                "uuid": oeid,
                "kwargs": {argName: argValue}
            })

    def request_propagation_once(self):
        self.set_acquire_status(1)
        self.input_queue.put({
            "command": "run_once",
            "object_type": "beamline"
        })

    def set_auto_update(self, value, clear_beams=False):
        self.autoUpdate = self._bool_value(value)
        kwargs = {"value": int(self.autoUpdate)}
        if clear_beams and self.autoUpdate:
            kwargs["clear_beams"] = 1
        if self.autoUpdate:
            self.set_acquire_status(1)
        self.input_queue.put({
            "command": "auto_update",
            "object_type": "beamline",
            "kwargs": kwargs
        })

    def _output_loop(self):
        while self._running:
            try:
                self.poll_once(timeout=self.outputPollInterval)
            except queue.Empty:
                pass
            except (EOFError, OSError, ValueError):
                break

    def poll_once(self, timeout=0):
        if self.output_queue is None:
            raise queue.Empty
        msg = self.output_queue.get(timeout=timeout)
        self.handle_output_message(msg)
        return msg

    def handle_output_message(self, msg):
        if 'histogram' in msg:
            self.update_epics_image(msg)
        elif 'pos_attr' in msg:
            self.update_epics_record(
                msg.get('sender_id'), msg.get('pos_attr'),
                msg.get('pos_value'))
        elif 'diag_attr' in msg:
            self.update_epics_record(
                msg.get('sender_id'), msg.get('diag_attr'),
                msg.get('diag_value'))
        elif 'repeat' in msg:
            print("Total repeats:", msg['repeat'])
            self.set_acquire_status(0)

    def _record_for(self, oeid, key):
        epics = self.epicsInterface
        if epics is None:
            return None
        return getattr(epics, 'pv_map', {}).get(oeid, {}).get(key)

    def set_acquire_status(self, value):
        epics = self.epicsInterface
        if epics is None:
            return
        record = getattr(epics, 'pv_records', {}).get('AcquireStatus')
        if record is not None:
            record.set(int(self._bool_value(value)))

    def update_epics_image(self, msg):
        epics = self.epicsInterface
        if epics is None:
            return

        oeid = msg.get('sender_id')
        record = self._record_for(oeid, 'image')
        if record is None:
            return

        histogram = msg.get('histogram')
        if histogram is None:
            return

        flatHist = np.flipud(histogram).flatten()
        recordLength = getattr(epics, 'image_lengths', {}).get(
            oeid, getattr(record, '_nelm', None))
        if recordLength is not None and len(flatHist) > recordLength:
            print(
                f"Skipping EPICS image update for "
                f"{msg.get('sender_name', oeid)}: histogram has "
                f"{len(flatHist)} pixels, waveform length is "
                f"{recordLength}. Increase imageMaxLength when creating "
                f"DynamicBeamline.")
            return
        record.set(flatHist)

    def update_epics_record(self, oeid, key, value):
        update_epics_readback(self.epicsInterface, oeid, key, value)

    @staticmethod
    def _bool_value(value):
        if isinstance(value, str):
            return value.strip().lower() not in ['', '0', 'false', 'off',
                                                 'no', 'none']
        return bool(value)

    def close(self, timeout=1):
        self._running = False
        if self.input_queue is not None:
            try:
                self.input_queue.put({"command": "exit"})
            except Exception:
                pass

        if self.calc_process is not None:
            self.calc_process.join(timeout=timeout)
            if self.calc_process.is_alive():
                self.calc_process.terminate()
                self.calc_process.join()
            self.calc_process = None

        if self._output_thread is not None and\
                threading.current_thread() is not self._output_thread:
            self._output_thread.join(timeout=timeout)
        self._output_thread = None

        for queue_obj in [self.input_queue, self.output_queue]:
            if queue_obj is None:
                continue
            try:
                queue_obj.close()
            except Exception:
                pass
            try:
                queue_obj.join_thread()
            except Exception:
                pass
        self.input_queue = None
        self.output_queue = None
        self._closed = True

    def run_forever(self, sleepTime=0.5):
        import time

        try:
            while self.calc_process is not None and\
                    self.calc_process.is_alive():
                time.sleep(sleepTime)
        finally:
            self.close()

    def __enter__(self):
        if self.calc_process is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

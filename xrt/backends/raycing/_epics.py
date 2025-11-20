# -*- coding: utf-8 -*-
import numpy as np
# from itertools import compress
from functools import partial
import re

from .physconsts import SIE0, CH  # analysis:ignore
from ._sets_units import orientationArgSet, shapeArgSet


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
    def __init__(self, bl, prefix, callback):

        self.bl = bl
        self.epicsPrefix = prefix
        self.pv_map = {}
        self.dbl = set()

        try:
            from softioc import softioc, builder, asyncio_dispatcher
        except ImportError:
            print("Missing softioc dependencies")
            return
        # Create an asyncio dispatcher, the event loop is now running
        self.dispatcher = asyncio_dispatcher.AsyncioDispatcher()

        # Set the record prefix
        builder.SetDeviceName(prefix)
        pv_records = {}
        pvFields = {'name'} | orientationArgSet | shapeArgSet

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
            self.pv_map[oeid] = {}

            if hasattr(oeObj, 'material') and oeObj.material is not None:
                if hasattr(oeObj.material, 'get_Bragg_angle'):
                    if hasattr(oeObj, 'bragg'):
                        e_field = 'bragg.energy'
                        initial_e = np.abs(CH / (
                                2*oeObj.material.d*np.sin(
                                        oeObj.bragg-oeObj.braggOffset)))
                    else:
                        e_field = 'pitch.energy'
                        initial_e = np.abs(CH / (
                                2*oeObj.material.d*np.sin(oeObj.pitch)))
                    pvname = f'{oename}:ENERGY'
                    pv_records[pvname] = builder.aOut(
                            pvname,
                            initial_value=initial_e,
                            always_update=True,
                            on_update=partial(callback, oeid, e_field))
                    self.pv_map[oeid]['bragg'] = pv_records[pvname]

            if hasattr(oeObj, 'expose') and oeObj.limPhysX is not None:
                pvname = f'{oename}:image'
                histShape = getattr(oeObj, 'histShape')
                imageLength = int(histShape[0]*histShape[1])
                pv_records[pvname] = builder.WaveformIn(
                    pvname,
                    length=imageLength
                    )

                for fIndex, field in enumerate(['width', 'height']):
                    pvname = f'{oename}:histShape:{field}'
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

            for argName in pvFields:
                if argName in ['shape', 'renderStyle']:
                    continue
                if hasattr(oeObj, argName):
                    if argName in ['name', 'rotationSequence']:
                        pvname = f'{oename}:{argName}'
                        pv_records[pvname] = builder.stringOut(
                            pvname,
                            initial_value=str(getattr(oeObj, argName)),
                            always_update=True,
                            on_update=partial(callback, oeid, argName))
                        self.pv_map[oeid][argName] = pv_records[pvname]
                    elif argName in ['center']:
                        for field in ['x', 'y', 'z']:
                            pvname = f'{oename}:{argName}:{field}'
                            pv_records[pvname] = builder.aOut(
                                pvname,
                                initial_value=getattr(oeObj.center, field),
                                always_update=True,
                                on_update=partial(callback, oeid,
                                                  f'{argName}.{field}'))
                            self.pv_map[oeid][f'{argName}.{field}'] =\
                                pv_records[pvname]
                    elif argName in ['limPhysX', 'limPhysY', 'limPhysX2',
                                     'limPhysY2']:  # TODO: startswith?
                        for fIndex, field in enumerate(['lmin', 'lmax']):
                            pvname = f'{oename}:{argName}:{field}'
                            limObj = getattr(oeObj, argName)
                            if limObj is not None:
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=limObj[fIndex],
                                    always_update=True,
                                    on_update=partial(callback, oeid,
                                                      f'{argName}.{field}'))
                                self.pv_map[oeid][f'{argName}.{field}'] =\
                                    pv_records[pvname]
                    elif argName in ['opening']:
                        for field in oeObj.kind:
                            pvname = f'{oename}:{argName}:{field}'
                            limObj = getattr(oeObj, argName)
                            if limObj is not None:
                                pv_records[pvname] = builder.aOut(
                                    pvname,
                                    initial_value=getattr(limObj, field),
                                    always_update=True,
                                    on_update=partial(callback, oeid,
                                                      f'{argName}.{field}'))
                                self.pv_map[oeid][f'{argName}.{field}'] =\
                                    pv_records[pvname]
                    else:
                        pvname = f'{oename}:{argName}'
                        pv_records[pvname] = builder.aOut(
                            pvname,
                            initial_value=getattr(oeObj, argName),
                            always_update=True,
                            on_update=partial(callback, oeid, argName))
                        self.pv_map[oeid][argName] = pv_records[pvname]

        [print(f'{self.epicsPrefix}:{recName}') for recName in pv_records]
        builder.LoadDatabase()
        softioc.iocInit(self.dispatcher)
        self.pv_records = pv_records
        for key in self.pv_records.keys():
            self.dbl.add(f'{prefix}:{key}')

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:40:52 2024

@author: roman
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
#sys.path.append(r"C:/github/xrt")  # analysis:ignore
import xrt
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
from functools import partial
from softioc import softioc, builder
import copy
import numbers

os.environ["EPICS_CA_ADDR_LIST"] = "192.168.152.101"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

def submit_param(beamline, key, value):
    setattr(beamline, key, raycing.get_init_val(value))

def run_ray_tracing(beamline, plots):
    raycing.run_process_from_file(beamline, plots)

def get_init_kwargs(obj):
    defArgs = dict(raycing.get_params(obj))
    initArgs = {}

    for arg, val in defArgs.items():
        if hasattr(oeObj, arg) and arg not in ['bl', 'material', 'material2']:
            realval = getattr(oeObj, arg)
            initArgs[arg] = copy.deepcopy(realval)
        else:
            continue
    return initArgs


# fileName = r"C:/github/xrt/examples/withRaycing/"
fileName = "1crystal.xml"

bl = raycing.BeamLine(fileName=fileName)
raycing.run_process_from_file(bl)

builder.SetDeviceName("BL")

pv_records = {}
print(bl.oesDict.items())
for oeid, oeline in bl.oesDict.items():
    if not raycing.is_valid_uuid(oeid):
        continue
    print(oeid, oeline)
    oeObj = oeline
    oeRecord = get_init_kwargs(raycing.get_obj_str(oeObj))  # this code will be a BeamLine class method
    for argName, argVal in oeRecord.items():
        pvname = f'{oeObj.name}:{argName}'

        if isinstance(argVal, numbers.Number):
            recordType = builder.aOut
        elif isinstance(argVal, str):
            recordType = builder.stringOut
        elif isinstance(argVal, list):
            recordType = builder.WaveformOut
        else:
            continue
        print(f'{pvname=}, {argVal=}', recordType)
        pv_records[pvname] = recordType(
                pvname, 
                initial_value=argVal, 
                always_update=True,
                on_update=partial(submit_param, bl, argName))

builder.LoadDatabase()
softioc.iocInit()

# Start processes required to be run after iocInit
#def update():
#    while True:
#        ai.set(ai.get() + 1)
#        cothread.Sleep(1)

#cothread.Spawn(update)

# Finally leave the IOC running with an interactive shell.
softioc.interactive_ioc(pv_records)
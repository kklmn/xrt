# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:40:52 2024

@author: roman
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
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
import time

os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

def submit_param(component, key, value):
    print(component, key, value, raycing.get_init_val(value))
    setattr(component, key, int(raycing.get_init_val(value)))
    
# def read_image()

def run_ray_tracing(beamline, db, value): #, plots):
    if int(value):
        run_process(beamline) #, plots)
        # raycing.run_process_from_file(beamline) #, plots)
        for obj in beamline.oesDict.values():
            # print(obj)
            if isinstance(obj[0], rscreens.Screen):
                name = obj[0].name
                if obj[0].image is not None:
                    db[f'{name}:Array'].set(obj[0].image.flatten())
                    db[f'{name}:Acquire'].set(0)
                    db[f'{name}:Status'].set(0)

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

def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        center=[0, 0, 0],
        nrays=250000,
        energies=(9000, 100),
        distE='normal',
        dx=0.2,
        dz=0.1,
        dxprime=0.00015,
        name="GS01")

# beamLine.toroidMirror01.R=38245.71081889952
# beamLine.toroidMirror02.R=21035.140950394736


    beamLine.toroidMirror01 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 10000, 0],
        pitch=r"5deg",
        limPhysX=[-20.0, 20.0],
        limPhysY=[-150.0, 150.0],
        # R=55000,
        R=38245,
        # R=[10000, 2000],
        r=100000000.0,
        name="TM_VERT")
    print(f"{beamLine.toroidMirror01.R=}")

    beamLine.toroidMirror02 = roes.ToroidMirror(
        bl=beamLine,
        center=[0, 11000, r"auto"],
        pitch=r"5deg",
        yaw=r"10deg",
        positionRoll=r"90deg",
        rotationSequence=r"RyRxRz",
        limPhysX=[-20, 20],
        limPhysY=[-150, 150],
        # R=25000,
        R=21035,
        # R=[11000, 1000],
        r=100000000.0,
        name="TM_HOR")
    print(f"{beamLine.toroidMirror02.R=}")


    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        center=[164.87347936545572, 11935, 343.73164815693235],
        name="Screen1",
        limPhysX=[-0.6, 0.6],
        limPhysY=[-0.45, 0.45],
        histShape=[400, 300]
        # histShape=[512, 512]
        )
        # center=[r"auto", 11935, r"auto"])

    return beamLine

def run_process(beamLine):
    time0 = time.time()
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()

    toroidMirror01beamGlobal01, toroidMirror01beamLocal01 = beamLine.toroidMirror01.reflect(
        beam=geometricSource01beamGlobal01)

    toroidMirror02beamGlobal01, toroidMirror02beamLocal01 = beamLine.toroidMirror02.reflect(
        beam=toroidMirror01beamGlobal01)

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=toroidMirror02beamGlobal01, withHistogram=True)

    outDict = {
        'geometricSource01beamGlobal01': geometricSource01beamGlobal01,
        'toroidMirror01beamGlobal01': toroidMirror01beamGlobal01,
        'toroidMirror01beamLocal01': toroidMirror01beamLocal01,
        'toroidMirror02beamGlobal01': toroidMirror02beamGlobal01,
        'toroidMirror02beamLocal01': toroidMirror02beamLocal01,
        'screen01beamLocal01': screen01beamLocal01}
    print("Tracing takes {:.3f}ms".format(1000*(time.time()-time0)))
    # beamLine.prepare_flow()
    return outDict


# fileName = r"C:/github/xrt/examples/withRaycing/"
fileName = "1crystal.xml"

# bl = raycing.BeamLine(fileName=fileName)
# raycing.run_process_from_file(bl)

bl = build_beamline()
run_process(bl)
# bl.

builder.SetDeviceName("BL")

pv_records = {}
# print(bl.oesDict.items())
for oeid, oeline in bl.oesDict.items():
    # if not raycing.is_valid_uuid(oeid):
    #     continue
    # print(oeid, oeline)
    oeObj = oeline[0]
    oeRecord = get_init_kwargs(raycing.get_obj_str(oeObj))  # this code will be a BeamLine class method
    # print(oeRecord)
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
                on_update=partial(submit_param, oeObj, argName))
        
    
    if isinstance(oeObj, rscreens.Screen):
        histShape = oeObj.histShape
        if histShape is not None:
            length = np.prod(histShape)
        pv_records[f'{oeObj.name}:Array'] = builder.WaveformIn(
            f'{oeObj.name}:Array', length=length)  # TODO: check API for 
        pv_records[f'{oeObj.name}:Status'] = builder.mbbIn(
            f'{oeObj.name}:Status', "OK", ("FAILING", "MINOR"),
            ("FAILED", "MAJOR"), ("NOT CONNECTED", "INVALID"))
        pv_records[f'{oeObj.name}:Acquire'] = builder.boolOut(
            f'{oeObj.name}:Acquire', ZNAM=0, ONAM=1,
            initial_value=0, always_update=True,
            on_update=partial(run_ray_tracing, bl, pv_records))


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
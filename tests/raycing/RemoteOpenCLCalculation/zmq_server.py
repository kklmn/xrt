# -*- coding: utf-8 -*-
"""
OpenCL Server
---------

Server for remote OpenCL calculations.
"""

__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "6 Jul 2023"

import zmq
import sys
import os; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import xrt.backends.raycing as raycing
import xrt.backends.raycing.myopencl as mcl
from xrt.backends.raycing.oes import OE
import pickle
from datetime import datetime

raycing._VERBOSITY_ = 80
PYVERSION = int(sys.version[0])


def send_zipped_pickle(socket, obj, flags=0, protocol=2):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
#    z = zlib.compress(p, 8)
    return socket.send(p, flags=flags)


def recv_zipped_pickle(socket, flags=0, protocol=2):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
#        p = zlib.decompress(z)
    return pickle.loads(z)

diffractKernels = ['integrate_kirchhoff']
materialsKernels = ['get_amplitude', 'get_amplitude_multilayer',
                    'get_amplitude_graded_multilayer',
                    'get_amplitude_graded_multilayer_tran',
                    'estimate_bent_width', 'get_amplitudes_pytte']
undulatorKernels = ['undulator', 'undulator_taper', 'undulator_nf',
                    'undulator_full', 'undulator_nf_full',
                    'get_trajectory_filament', 'custom_field_filament',
                    'get_trajectory', 'custom_field']
oesKernels = ['reflect_crystal']


targetOpenCL = 'GPU'

matCL32 = mcl.XRT_CL(r'materials.cl', precisionOpenCL='float32',
                     targetOpenCL=targetOpenCL)
matCL64 = mcl.XRT_CL(r'materials.cl', precisionOpenCL='float64',
                     targetOpenCL=targetOpenCL)
sourceCL32 = mcl.XRT_CL(r'undulator.cl', precisionOpenCL='float32',
                        targetOpenCL=targetOpenCL)
sourceCL64 = mcl.XRT_CL(r'undulator.cl', precisionOpenCL='float64',
                        targetOpenCL=targetOpenCL)
waveCL = mcl.XRT_CL(r'diffract.cl', precisionOpenCL='float64',
                    targetOpenCL=targetOpenCL)

OE32 = OE(precisionOpenCL='float32', targetOpenCL=targetOpenCL)
OE64 = OE(precisionOpenCL='float64', targetOpenCL=targetOpenCL)
oeCL32 = OE32.ucl
oeCL64 = OE64.ucl


port = "15560"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:%s" % port)

while True:
    # message = socket.recv_pyobj()  # Python 3 only
    message = recv_zipped_pickle(socket)
    precision = 64
    reply = None
    dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    if 'scalarArgs' in message.keys():
        for arg in message['scalarArgs']:
            if isinstance(arg, np.int32):
                continue
            elif isinstance(arg, np.float32):
                precision = 32
                break
            elif isinstance(arg, np.float64):
                break
            else:
                continue
    else:
        print(dtstr, "ERROR: not a kernel")
        reply = ("ERROR", "not a kernel")

    if 'kernelName' in message.keys():
        kName = message['kernelName']
        if kName in diffractKernels:
            xrtClContext = waveCL
        elif kName in undulatorKernels:
            xrtClContext = sourceCL32 if precision == 32 else sourceCL64
        elif kName in materialsKernels:
            xrtClContext = matCL32 if precision == 32 else matCL64
        elif kName in oesKernels:
            xrtClContext = oeCL32 if precision == 32 else oeCL64
        else:
            print(dtstr, "ERROR: unknown kernel:", kName)
            reply = ("ERROR", "unknown kernel: "+kName)
    else:
        print(dtstr, "ERROR: not a kernel")
        reply = ("ERROR", "not a kernel")

    if reply is None:
        try:
            dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            print("{2} Calculating '{0}' in {1}-bit".format(kName, precision,
                                                            dtstr))
            reply = xrtClContext.run_parallel(**message)
            dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            print(dtstr, "Calculations complete. Sending back results")
        except:
            dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
            print(dtstr, "ERROR: error while calculating a kernel")
            reply = ("ERROR", "error while calculating a kernel")

    send_zipped_pickle(socket, reply, protocol=2)

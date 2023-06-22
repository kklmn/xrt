# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 00:01:09 2021

@author: roman
"""

import zmq
import sys
import os; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import xrt.backends.raycing as raycing
import xrt.backends.raycing.myopencl as mcl
import pickle

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

# oeCL32 = mcl.XRT_CL(r'OE.cl', precisionOpenCL='float32',
#                        targetOpenCL=targetOpenCL)
# oeCL64 = mcl.XRT_CL(r'OE.cl', precisionOpenCL='float64',
#                        targetOpenCL=targetOpenCL)


port = "15560"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:%s" % port)

while True:
    # message = socket.recv_pyobj()  # Python 3 only
    message = recv_zipped_pickle(socket)
    precision = 64
    reply = None

    if 'scalarArgs' in message.keys():
        for arg in message['scalarArgs']:
            if isinstance(arg, np.int32):
                continue
            elif isinstance(arg, np.float32):
                precision = 32
                break
            else:  # np.float64
                break
    else:
        print("ERROR: not a kernel")
        reply = ("ERROR", "not a kernel")

    if 'kernelName' in message.keys():
        kName = message['kernelName']
        if kName in diffractKernels:
            xrtClContext = waveCL
        elif kName in undulatorKernels:
            xrtClContext = sourceCL32 if precision == 32 else sourceCL64
        elif kName in materialsKernels:
            xrtClContext = matCL32 if precision == 32 else matCL64
        else:
            print("ERROR: unknown kernel:", kName)
            reply = ("ERROR", "unknown kernel: "+kName)
    else:
        print("ERROR: not a kernel")
        reply = ("ERROR", "not a kernel")

    if reply is None:
        try:
            print("Calculating {0} in {1}-bit precision".format(
                    kName, precision))
            reply = xrtClContext.run_parallel(**message)
            print("calculations complete. sending back results")
        except:
            print("ERROR: error while calculating a kernel")
            reply = ("ERROR", "error while calculating a kernel")

    send_zipped_pickle(socket, reply, protocol=2)

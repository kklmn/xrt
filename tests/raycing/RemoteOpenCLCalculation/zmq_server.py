# -*- coding: utf-8 -*-
"""
.. _oclserver:

OpenCL Server
-------------

This example contains a Python script designed to run on a GPU server,
leveraging ZeroMQ (ZMQ) for efficient data transfer. The script acts as a
remote accelerator device, receiving data from a client Python script,
performing calculations on the GPU, and returning the results for plotting on
a local computer.

Users can seamlessly execute their scripts in their favorite IDE, offloading
resource-intensive calculations to a remote server over the network. The only
trade-off is the potential delay due to data transfer, which is outweighed by
the benefits when local computations take longer than data transfer time.
Furthermore, the local graphical user interface (GUI) remains responsive
without freezes or issues caused by high GPU/CPU loads. This script now
supports all acceleration scenarios:

* synchrotron sources,
* wave propagation,
* bent crystals,
* multilayer reflectivity.

Script Components
~~~~~~~~~~~~~~~~~

The GPU accelerator script is comprised of two files:

1. ``zmq_server.py``: The server script is the main component, responsible for
   receiving data and getting kernel names from the client. It listens on a
   predefined port, processes the received package, executes the specified
   kernel on the GPU and sends the computed data back to the client. This
   server script can be executed independently or in conjunction with the
   queue manager.

2. ``queue_device.py``: The queue manager script facilitates the handling of
   multiple user requests and the distribution of computational tasks across
   multiple servers. It provides scalability and load balancing capabilities.
   The queue manager can be executed on the same machine as the server or on a
   dedicated node. However, when running the queue manager on a separate node,
   data transfer times may increase.

Running the Script
~~~~~~~~~~~~~~~~~~

To execute the GPU accelerator script, follow these steps:

Set up the GPU server environment with the necessary dependencies, including
pyzmq, xrt and underlying dependencies (numpy, scipy, matplotlib, pyopencl).
Start the server script, either as a standalone process or in conjunction with
the queue manager, based on your specific requirements.

Ensure that the *client* Python script is configured to connect to the correct
server address and port:

for standalone server:
``targetOpenCL="GPU_SERVER_ADDRESS:15560"``

for queue manager:
``targetOpenCL="QUEUE_MANAGER_ADDRESS:15559"``


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
    """pickle an object and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    # z = zlib.compress(p, 8)
    return socket.send(p, flags=flags)


def recv_zipped_pickle(socket, flags=0, protocol=2):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    # p = zlib.decompress(z)
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

# Please specify here the OpenCL device installed on the server
targetOpenCL = 'GPU'


def main():
    matCL32 = mcl.XRT_CL(
        r'materials.cl', precisionOpenCL='float32', targetOpenCL=targetOpenCL)
    matCL64 = mcl.XRT_CL(
        r'materials.cl', precisionOpenCL='float64', targetOpenCL=targetOpenCL)
    sourceCL32 = mcl.XRT_CL(
        r'undulator.cl', precisionOpenCL='float32', targetOpenCL=targetOpenCL)
    sourceCL64 = mcl.XRT_CL(
        r'undulator.cl', precisionOpenCL='float64', targetOpenCL=targetOpenCL)
    waveCL = mcl.XRT_CL(
        r'diffract.cl', precisionOpenCL='float64', targetOpenCL=targetOpenCL)

    OE32 = OE(precisionOpenCL='float32', targetOpenCL=targetOpenCL)
    OE64 = OE(precisionOpenCL='float64', targetOpenCL=targetOpenCL)
    oeCL32 = OE32.ucl
    oeCL64 = OE64.ucl

    port = "15560"
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    # Replace 'localhost' with queue manager address if not on the same machine
    # Replace 'localhost' with '*' for standalone mode
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
                reply = ("ERROR", "unknown kernel: " + kName)
        else:
            print(dtstr, "ERROR: not a kernel")
            reply = ("ERROR", "not a kernel")

        if reply is None:
            try:
                dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
                print("{2} Calculating '{0}' in {1}-bit".format(
                    kName, precision, dtstr))
                reply = xrtClContext.run_parallel(**message)
                dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
                print(dtstr, "Calculations complete. Sending back results")
            except Exception as e:
                dtstr = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
                print(dtstr, " ERROR while calculating a kernel:")
                print(e)
                reply = ("ERROR", "error while calculating a kernel")

        send_zipped_pickle(socket, reply, protocol=2)


if __name__ == "__main__":
    main()

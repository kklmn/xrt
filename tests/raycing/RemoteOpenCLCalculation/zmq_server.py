# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 00:01:09 2021

@author: roman
"""

import zmq
import time
import sys
#import random
import os; 
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
#sys.path.append(r'C:\Github\xrt')  # analysis:ignore
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import xrt.backends.raycing as raycing
from xrt.backends.raycing.physconsts import E2W, CH, PI2
raycing._VERBOSITY_ = 80
import xrt.backends.raycing.sources as rs
import pickle
#import zlib

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

CustomSource = rs.SourceFromField(targetOpenCL='CPU')

port = "15560"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:%s" % port)
while True:
#    message = socket.recv_pyobj()  # Python 3 only
    message = recv_zipped_pickle(socket)
    reply = CustomSource.ucl.run_parallel(**message)
    
    print("calculations complete. sending back results")
    send_zipped_pickle(socket, reply, protocol=2)

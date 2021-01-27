# -*- coding: utf-8 -*-
r"""
Initialization code of pyopencl.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "19 Mar 2017"
import numpy as np
import os
#import time
from .. import raycing
import pickle
#import zlib
import sys
try:
    import pyopencl as cl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False

try:
    import zmq
    isZMQ = True
except ImportError:
    isZMQ = False

PYVERSION = int(sys.version[0])

__fdir__ = os.path.dirname(__file__)
_DEBUG = 20


class XRT_CL(object):
    def __init__(self, filename=None, targetOpenCL=raycing.targetOpenCL,
                 precisionOpenCL=raycing.precisionOpenCL, kernelsource=None):
        self.kernelsource = kernelsource
        self.cl_filename = filename
        self.lastTargetOpenCL = None
        self.lastPrecisionOpenCL = None
        self.set_cl(targetOpenCL, precisionOpenCL)
        self.cl_is_blocking = True

    def send_zipped_pickle(self, socket, obj, flags=0, protocol=2):
        """Pickle an object, and zip the pickle before sending it.
           From pyzmq docs"""
        p = pickle.dumps(obj, protocol)
#        z = zlib.compress(p, 0)
        return socket.send(p, flags=flags)
    
    def recv_zipped_pickle(self, socket, flags=0, protocol=2):
        """inverse of send_zipped_pickle. From pyzmq docs"""
        z = socket.recv(flags)
        kw = {}
        if PYVERSION > 2:  # Python 2 compatibility. Do we need it?
            kw['encoding'] = 'bytes'
#        p = zlib.decompress(z)
        return pickle.loads(z, **kw)

    def set_cl(self, targetOpenCL=raycing.targetOpenCL,
               precisionOpenCL=raycing.precisionOpenCL):
        if (targetOpenCL == self.lastTargetOpenCL) and\
                (precisionOpenCL == self.lastPrecisionOpenCL):
            return
        self.lastTargetOpenCL = targetOpenCL
        self.lastPrecisionOpenCL = precisionOpenCL
        try:
            if ':' in targetOpenCL and isZMQ:
                self.address, self.port = targetOpenCL.split(':')
                if not self.address.startswith('tcp://'):
                    self.address = 'tcp://' + self.address
                print(self.address, self.port)
                self.ZMQcontext = zmq.Context()
                self.ZMQsocket = self.ZMQcontext.socket(zmq.REQ)
                self.ZMQsocket.connect("{0}:{1}".format(self.address,
                                                        self.port))
                self.useZMQ = True
                self.lastTargetOpenCL = targetOpenCL
            else:
                self.useZMQ = False
        except:
            self.useZMQ = False
        print("useZMQ:", self.useZMQ)

        if isOpenCL and not self.useZMQ:
            try:
                cl_platforms = cl.get_platforms()
            except:
                targetOpenCL = None
                raise EnvironmentError("Unknown error. OpenCL disabled")
            if isinstance(targetOpenCL, (tuple, list)):
                iDevice = []
                targetOpenCL = list(targetOpenCL)
                if isinstance(targetOpenCL[0], int):
                    nPlatform, nDevice = targetOpenCL
                    platform = cl_platforms[nPlatform]
                    try:
                        dev = platform.get_devices()[nDevice]
                        iDevice.extend([dev])
                    except:
                        pass
                else:
                    for target in targetOpenCL:
                        if isinstance(target, (tuple, list)):
                            target = list(target)
                            if len(target) > 1:
                                nPlatform, nDevice = target
                                platform = cl_platforms[nPlatform]
                                try:
                                    iDevice.extend(
                                        [platform.get_devices()[nDevice]])
                                except:
                                    pass
                            else:
                                nPlatform = target[0]
                                platform = cl_platforms[nPlatform]
                                try:
                                    iDevice.extend(platform.get_devices())
                                except:
                                    pass
            elif isinstance(targetOpenCL, int):
                nPlatform = targetOpenCL
                platform = cl_platforms[nPlatform]
                try:
                    iDevice = platform.get_devices()
                except:
                    pass
            elif isinstance(targetOpenCL, str):
                iDeviceCPU = []
                iDeviceGPU = []
                iDeviceAcc = []
                iDevice = []
                for platform in cl_platforms:
                    if 'mesa' in platform.vendor.lower():
                        continue  # for new Linuxes Mesa provides OpenCL 1.1
                    CPUdevices = []
                    GPUdevices = []
                    AccDevices = []
                    try:  # at old pyopencl versions:
                        CPUdevices =\
                            platform.get_devices(
                                device_type=cl.device_type.CPU)
                        GPUdevices =\
                            platform.get_devices(
                                device_type=cl.device_type.GPU)
                        AccDevices =\
                            platform.get_devices(
                                device_type=cl.device_type.ACCELERATOR)
                    except cl.RuntimeError:
                        pass

                    if len(CPUdevices) > 0:
                        if len(iDeviceCPU) > 0:
                            if CPUdevices[0].vendor == \
                                    CPUdevices[0].platform.vendor:
                                try:
                                    tmpctx = cl.Context(devices=CPUdevices)
                                    iDeviceCPU = CPUdevices
                                except:
                                    pass
                        else:
                            try:
                                tmpctx = cl.Context(devices=CPUdevices)
                                iDeviceCPU.extend(CPUdevices)
                            except:
                                pass
                    for GPUDevice in GPUdevices:
                        try:
                            if ('graphics' in GPUDevice.name.lower() and
                                    'intel' in GPUDevice.vendor.lower()):
                                continue
                            tmpctx = cl.Context(devices=[GPUDevice])
                            if GPUDevice.double_fp_config > 0:
                                iDeviceGPU.extend([GPUDevice])
                        except:
                            pass
                    iDeviceAcc.extend(AccDevices)
                if _DEBUG > 10:
                    print("OpenCL: bulding {0} ...".format(self.cl_filename))
                    print("OpenCL: found {0} CPU{1}".format(
                          len(iDeviceCPU) if len(iDeviceCPU) > 0 else 'none',
                          's' if len(iDeviceCPU) > 1 else ''))
                    print("OpenCL: found {0} GPU{1}".format(
                          len(iDeviceGPU) if len(iDeviceGPU) > 0 else 'none',
                          's' if len(iDeviceGPU) > 1 else ''))
                    print("OpenCL: found {0} other accelerator{1}".format(
                          len(iDeviceAcc) if len(iDeviceAcc) > 0 else 'none',
                          's' if len(iDeviceAcc) > 1 else ''))

                if targetOpenCL.upper().startswith('GPU'):
                    iDevice.extend(iDeviceGPU)
                elif targetOpenCL.upper().startswith('CPU'):
                    iDevice.extend(iDeviceCPU)
                elif targetOpenCL.upper().startswith('ALL'):
                    iDevice.extend(iDeviceGPU)
                    iDevice.extend(iDeviceCPU)
                    iDevice.extend(iDeviceAcc)
                else:  # auto
                    if len(iDeviceGPU) > 0:
                        iDevice = iDeviceGPU
                    elif len(iDeviceAcc) > 0:
                        iDevice = iDeviceAcc
                    else:
                        iDevice = iDeviceCPU
                if len(iDevice) == 0:
                    targetOpenCL = None
                    self.lastTargetOpenCL = targetOpenCL
            else:  # None
                targetOpenCL = None
                self.lastTargetOpenCL = targetOpenCL
        elif not self.useZMQ:
            raise EnvironmentError("Neither pyopencl nor ZMQ are available!")

        if targetOpenCL is not None:
            if _DEBUG > 10 and not self.useZMQ:
                autoStr = "Autos" if isinstance(targetOpenCL, str) else "S"
                for idn, idv in enumerate(iDevice):
                    print("OpenCL for {0}: {1}elected device {2}: {3}".format(
                        self.cl_filename, autoStr, idn, idv.name))

            if self.kernelsource is None:
                cl_file = os.path.join(os.path.dirname(__file__),
                                       self.cl_filename)
                with open(cl_file, 'r') as f:
                    kernelsource = f.read()
            else:
                kernelsource = self.kernelsource
            if precisionOpenCL == 'auto':
                if self.useZMQ:
                    precisionOpenCL = 'float64'
                else:
                    try:
                        for device in iDevice:
                            if device.double_fp_config == 63:
                                precisionOpenCL = 'float64'
                            else:
                                raise AttributeError
                    except AttributeError:
                        precisionOpenCL = 'float32'
            if _DEBUG > 10:
                print('precisionOpenCL = {0}'.format(precisionOpenCL))
            if precisionOpenCL == 'float64':
                self.cl_precisionF = np.float64
                self.cl_precisionC = np.complex128
                kernelsource = kernelsource.replace('float', 'double')
            elif precisionOpenCL == 'float32':
                self.cl_precisionF = np.float32
                self.cl_precisionC = np.complex64
            else:
                raise ValueError("Unknown precision")
            self.cl_queue = []
            self.cl_ctx = []
            self.cl_program = []
            self.cl_mf = None
            if not self.useZMQ:
                for device in iDevice:
                    cl_ctx = cl.Context(devices=[device])
                    self.cl_queue.extend([cl.CommandQueue(cl_ctx, device)])
                    self.cl_program.extend(
                        [cl.Program(cl_ctx, kernelsource).build(
                            options=['-I', __fdir__])])
                    self.cl_ctx.extend([cl_ctx])
                self.cl_mf = cl.mem_flags

    def run_parallel(self, kernelName='', scalarArgs=None,
                     slicedROArgs=None, nonSlicedROArgs=None,
                     slicedRWArgs=None, nonSlicedRWArgs=None, dimension=0):

        if self.useZMQ:
            outgoing_dict = {'kernelName': kernelName, 
                             'scalarArgs': scalarArgs,
                             'slicedROArgs': slicedROArgs,
                             'nonSlicedROArgs': nonSlicedROArgs,
                             'slicedRWArgs': slicedRWArgs,
                             'nonSlicedRWArgs': nonSlicedRWArgs,
                             'dimension': dimension}
#            self.ZMQsocket.send_pyobj(outgoing_dict, protocol=0)
            self.send_zipped_pickle(self.ZMQsocket, outgoing_dict)
            #  Get the reply.
            ret = self.recv_zipped_pickle(self.ZMQsocket)
#            message = self.ZMQsocket.recv_multipart()
#            dataType = self.cl_precisionC if len(message) < 3 else self.cl_precisionF
#            ret = [np.frombuffer(arr, dtype=dataType) for arr in message]
#            print(kernelName, [len(arr) for arr in ret])
        else:
            ka_offset = len(scalarArgs) if scalarArgs is not None else 0
            ro_offset = len(slicedROArgs) if slicedROArgs is not None else 0
            ns_offset = len(nonSlicedROArgs) if nonSlicedROArgs is not None else 0
            rw_offset = len(slicedRWArgs) if slicedRWArgs is not None else 0
            rw_pos = ka_offset + ro_offset + ns_offset
            nsrw_pos = ka_offset + ro_offset + ns_offset + rw_offset
    
            kernel_bufs = []
            global_size = []
            ev_h2d = []
            ev_run = []
            nCU = []
            ndstart = []
            ndslice = []
            ndsize = []
            minWGS = 1e20
    
            for ictx, ctx in enumerate(self.cl_ctx):
                nCUw = 1
                nCU.extend([nCUw])
                tmpWGS = ctx.devices[0].max_work_group_size
                if tmpWGS < minWGS:
                    minWGS = tmpWGS
                # nCU.extend([ctx.devices[0].max_compute_units*nCUw])
    
            totalCUs = np.sum(nCU)
            minWGS = 256
            divider = minWGS * totalCUs
            n2f = np.remainder(dimension, divider)
            needResize = False
            # odd dimension performance fix
            if n2f != 0 and dimension > divider:
                oldSize = dimension
                dimension = (np.trunc(dimension/divider) + 1) * divider
                nDiff = int(dimension - oldSize)
                needResize = True
    
            work_cl_ctx = self.cl_ctx if dimension > totalCUs else [self.cl_ctx[0]]
            nctx = len(work_cl_ctx)
    
            for ictx, ctx in enumerate(work_cl_ctx):
                ev_h2d.extend([[]])
                kernel_bufs.extend([[]])
                if scalarArgs is not None:
                    kernel_bufs[ictx].extend(scalarArgs)
                ndstart.extend([sum(ndsize)])
    
                if dimension > 1:
                    if ictx < nctx - 1:
                        ndsize.extend([np.floor(dimension*nCU[ictx]/totalCUs)])
                    else:
                        ndsize.extend([dimension-ndstart[ictx]])
                    ndslice.extend([slice(ndstart[ictx],
                                          ndstart[ictx]+ndsize[ictx])])
                else:
                    ndslice.extend([0])
    # In case each photon has an array of input/output data we define a second
    # dimension
                if slicedROArgs is not None and dimension > 1:
                    for iarg, arg in enumerate(slicedROArgs):
                        newArg = np.concatenate([arg, arg[:nDiff]]) if needResize\
                            else arg
                        secondDim = np.int(len(newArg) / dimension)
                        iSlice = slice(int(ndstart[ictx]*secondDim),
                                       int((ndstart[ictx]+ndsize[ictx])*secondDim))
                        kernel_bufs[ictx].extend([cl.Buffer(
                            self.cl_ctx[ictx], self.cl_mf.READ_ONLY |
                            self.cl_mf.COPY_HOST_PTR, hostbuf=newArg[iSlice])])
    
                if nonSlicedROArgs is not None:
                    for iarg, arg in enumerate(nonSlicedROArgs):
                        kernel_bufs[ictx].extend([cl.Buffer(
                            self.cl_ctx[ictx], self.cl_mf.READ_ONLY |
                            self.cl_mf.COPY_HOST_PTR, hostbuf=arg)])
    
                if slicedRWArgs is not None:
                    for iarg, arg in enumerate(slicedRWArgs):
                        newArg = np.concatenate([arg, arg[:nDiff]]) if needResize\
                            else arg
                        secondDim = np.int(len(newArg) / dimension)
                        iSlice = slice(int(ndstart[ictx]*secondDim),
                                       int((ndstart[ictx]+ndsize[ictx])*secondDim))
                        kernel_bufs[ictx].extend([cl.Buffer(
                            self.cl_ctx[ictx], self.cl_mf.READ_WRITE |
                            self.cl_mf.COPY_HOST_PTR, hostbuf=newArg[iSlice])])
                    global_size.extend([(np.int(ndsize[ictx]),)])
                if nonSlicedRWArgs is not None:
                    for iarg, arg in enumerate(nonSlicedRWArgs):
                        kernel_bufs[ictx].extend([cl.Buffer(
                            self.cl_ctx[ictx], self.cl_mf.READ_WRITE |
                            self.cl_mf.COPY_HOST_PTR, hostbuf=arg)])
                    global_size.extend([np.array([1]).shape])
    
            local_size = None
            for ictx, ctx in enumerate(work_cl_ctx):
                kernel = getattr(self.cl_program[ictx], kernelName)
                ev_run.extend([kernel(
                    self.cl_queue[ictx],
                    global_size[ictx],
                    local_size,
                    *kernel_bufs[ictx])])
    
            for iev, ev in enumerate(ev_run):
                status = cl.command_execution_status.to_string(
                    ev.command_execution_status)
                if _DEBUG > 20:
                    print("ctx status {0} {1}".format(iev, status))
    
            ret = ()
    
            if slicedRWArgs is not None:
                for ictx, ctx in enumerate(work_cl_ctx):
                    for iarg, arg in enumerate(slicedRWArgs):
                        newArg = np.concatenate([arg, arg[:nDiff]]) if needResize\
                            else arg
                        secondDim = np.int(len(newArg) / dimension)
                        iSlice = slice(int(ndstart[ictx]*secondDim),
                                       int((ndstart[ictx]+ndsize[ictx])*secondDim))
                        cl.enqueue_copy(self.cl_queue[ictx],
                                        slicedRWArgs[iarg][iSlice],
                                        kernel_bufs[ictx][iarg + rw_pos],
                                        is_blocking=self.cl_is_blocking)
                if needResize:
                    for arg in slicedRWArgs:
                        arg = arg[:oldSize]
                ret += tuple(slicedRWArgs)
    
            if nonSlicedRWArgs is not None:
                for ictx, ctx in enumerate(work_cl_ctx):
                    for iarg, arg in enumerate(nonSlicedRWArgs):
                        cl.enqueue_copy(self.cl_queue[ictx],
                                        nonSlicedRWArgs[iarg],
                                        kernel_bufs[ictx][iarg + nsrw_pos],
                                        is_blocking=self.cl_is_blocking)
                if needResize:
                    for arg in nonSlicedRWArgs:
                        arg = arg[:oldSize]
                ret += tuple(nonSlicedRWArgs)
#        print("Total CL execution time:", time.time() - t0, "s")
        return ret

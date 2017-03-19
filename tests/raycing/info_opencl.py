import pyopencl as cl  # Import the OpenCL GPU computing API

print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
for platform in cl.get_platforms():  # Print each platform on this computer
    print('=' * 60)
    print('Platform - Name:  ' + platform.name)
    print('Platform - Vendor:  ' + platform.vendor)
    print('Platform - Version:  ' + platform.version)
    print('Platform - Extensions:  ' + platform.extensions)
    print('Platform - Profile:  ' + platform.profile)
    for device in platform.get_devices():  # Print each device per-platform
        print('    ' + '-' * 56)
        print('    Device - Name:  ' + device.name)
        print('    Device - Vendor:  ' + device.vendor)
        print('    Device - Type:  ' +
              cl.device_type.to_string(device.type))
        print('    Device - Max Clock Speed:  {0} Mhz'.format(
            device.max_clock_frequency))
        print('    Device - Compute Units:  {0}'.format(
            device.max_compute_units))
        print('    Device - Local Memory:  {0:.0f} KB'.format(
            device.local_mem_size/1024))
        print('    Device - Constant Memory:  {0:.0f} KB'.format(
            device.max_constant_buffer_size/1024))
        print('    Device - Global Memory: {0:.0f} GB'.format(
            device.global_mem_size/1073741824.0))
        print('    Device - FP:  ' + str(device.double_fp_config))
print('\n')

#ctx = cl.create_some_context()

#test your iPlatform, iDevice here. Read the output. Is it your GPU?
iPlatform, iDevice = 0, 0
platform = cl.get_platforms()[iPlatform]
device = platform.get_devices()[iDevice]
ctx = cl.Context(devices=[device])

print(ctx)

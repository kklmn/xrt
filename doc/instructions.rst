.. _instructions:

Detailed installation instructions
----------------------------------

Get Python
~~~~~~~~~~

`WinPython <https://sourceforge.net/projects/winpython/files>`_ is the easiest
way to get Python on Windows. It is portable (movable) and one can have many
WinPython installations without mutual interference.

`Anaconda <https://www.anaconda.com/download>`_ is another popular Python
distribution. It works on Linux, MacOS and Windows.

Automatic installation of xrt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    pip install xrt

or::

    conda install conda-forge::xrt

Running xrt without installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because xrt does not build anything at the installation time, it can be used
*without installation*, only its source code is required. One advantage of no
installation is a single location of xrt served by all, possibly many, Python
installations; that location can even span various OS's if it is on a network
drive. Get xrt as a zip from GitHub and unzip it to a suitable location.

For running xrt without installation, all required dependencies must be
installed beforehand. Look into xrt's `setup.py` and find those dependencies in
the lists `install_requires` and `extras_require`. They are pip installable.

::

    pip install numpy scipy matplotlib sphinx sphinxcontrib-jquery sphinx-tabs
    pip install colorama pyopencl pyopengl siphash24 pyqt5 PyQtWebEngine

After having dependencies installed, you may run any script from `examples` or
`tests`. The scripts refer to xrt located a few levels higher.

A typical pitfall in this scenario is having xrt at several locations. To
discover which xrt package is actually in use, start a Python session, import
xrt and examine its `xrt.__file__`.

.. rubric:: Spyder

Spyder (>=3.0.0) is a cross-platform IDE for python; xrtQook GUI uses some of
its libraries to provide the editor and the console interface (highly
recommended for a nicer look). Be aware that starting from version 3.2.0 spyder
switched to IPython and has no classic python console. The IPython console of
spyder integrates matplotlib images (can be switched off) and prohibits the use
of multiprocessing. Therefore, if you run an xrt script from spyder, select to
run it in an external console.

.. rubric:: GLUT

If the xrtQook GUI reports "GLUT is not found", here are a few possible
solutions.

If on Windows, examine these solutions:
`one <https://github.com/kklmn/xrt/issues/196>`_ and
`two <https://github.com/kklmn/xrt/issues/180>`_.

If using Anaconda, this installation may help::

    conda install -c conda-forge freeglut  # reboot

If on Linux, GLUT can be installed like this::

    sudo apt install freeglut3-dev  # or  sudo dnf install freeglut-devel

.. rubric:: PyOpenCL

Before installing PyOpenCL you need at least one existing OpenCL implementation
(driver). OpenCL can come with a graphics card driver and/or with an OpenCL CPU
runtime (search for ``Intel CPU only OpenCL runtime``). High profile graphics
cards (those with a high FP64/FP32 ratio) are advantageous. If you try xrt in
a VM, `pocl` may be useful::

    ./conda install -c conda-forge pocl  # from anaconda/bin

In Linux Anaconda you may encounter the situation when pyopencl finds no OpenCL
driver, which is reported by xrtQook on its welcome screen. The solution is
presented `here <https://documen.tician.de/pyopencl/misc.html#using-vendor-supplied-opencl-drivers-linux>`_. 
It consists of copying \*.icd files from /etc/OpenCL/vendors to
<your-anaconda>/etc/OpenCL/vendors or to your environment within anaconda if
you use it.

If you use a system-wide Python on Linux, do similar to this (works on Ubuntu
18.04 with the recommended Nvidia proprietary driver or
`OpenCL runtime for Intel processors <https://software.intel.com/en-us/articles/opencl-drivers>`_)::

    sudo apt-get install opencl-headers ocl-icd-opencl-dev
    pip install pyopencl

Instead of installing ocl-icd-opencl-dev, one can locate libOpenCL.so and
create a symbolic link in /usr/lib or any other lib folder in the path search.


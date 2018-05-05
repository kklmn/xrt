.. _instructions:

Detailed instructions for installing dependencies
-------------------------------------------------

Python
~~~~~~

.. rubric:: Windows

`WinPython <https://sourceforge.net/projects/winpython/files>`_ or 
`Anaconda <https://www.anaconda.com/download>`_ are advised.
xrt can run in both Python 2 and Python 3 branches. Python 3 is recommended as
Python 2 will `retire soon <https://pythonclock.org>`_.

In WinPython, there are "Qt" and "Zero" installation versions for,
correspondingly, many and no site-packages included. In the latter case use
``pip`` to install the required dependencies.

.. rubric:: Linux

Python is usually pre-installed in all popular distributions. You may prefer
Anaconda, as it already has all the required packages, except pyopencl and
glut (see below).

.. rubric:: macOS

Anaconda is the only option to run xrt. 

Dependencies
~~~~~~~~~~~~

xrt relies heavily on numpy (>=1.8.0), scipy (>=0.17.0) and matplotlib
(>=2.0.0) packages, these three are essential. Tkinter is used by matplotlib
for plotting but often requires manual installation.

Some of xrt examples require xlrd, xlwt and pandas for working with Excel files
(i.e. for custom magnetic field data).

Spyder (>=3.0.0) is a cross-platform IDE for python; xrtQook GUI uses some of
its libraries to render the live help and provide the console interface (highly
recommended for nicer look). Be aware that starting from version 3.2.0 spyder
switched to IPython and has no classic python console (consider version 3.1.4
in case you want the classic console). The IPython console of spyder does
unwanted integration of matplotlib images (can be switched off) and prohibits
the use of multiprocessing. Therefore, if you run an xrt script from spyder,
select to run it in an external console.

pyopencl (>=2015.1) is highly recommended if you calculate undulator sources
(it’s still possible in pure numpy, but significantly slower), and is required
for custom magnetic field sources and wave propagation. Some materials (Powder,
CrystalHarmonics) will not work without pyopencl.

PyQt4 (Qt>=4.8) or PyQt5 (Qt>=5.2) are needed for xrtQook interface.

PyOpenGL (>=3.1.1) and PyOpenGL_Accelerate (>=3.1.1) (not required but
recommended) are used by xrtGlow for the 3D scene rendering.

Automatic installation of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from version 1.3.1 xrt installer script automatically analyses the
list of dependencies. Just run the pip install command::

    pip install xrt
    #or, if you've downloaded the archive from github:
    pip install <xrt-zip-file>

In addition to the automatic installation: be aware that in python2 you have to
install PyQt4 libraries manually. 
Linux users should install tkinter backend (python-tk or python3-tk) using a
system package manager. 
Binary packages of pyopengl are highly recommended for Windows users (see
below).

Manual installation of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Windows

If you prefer to work with WinPython "Zero", you should manually install all
required packages first. The simplest way for this is to get the pre-compiled
whl distributions from
`Unofficial Windows Binaries for Python Extension Packages by C. Gohlke
<https://www.lfd.uci.edu/~gohlke/pythonlibs>`_.
To install a whl package, start a terminal: as "WinPython Command Prompt.exe"
and run::

    pip install <path-to-whl>
    
If you use Anaconda, type the command above in a system terminal launched from
"Anaconda/Scripts" or "Anaconda/bin" folder.

.. rubric:: Linux

If you use Anaconda, the required packages are already there, except it
probably lacks the GLUT library used by PyOpenGL. Then you can first install
freeglut3-dev by apt and then pyopengl by pip or directly the packaged
python-opengl or python3-opengl by apt.

If you use a system-wide Python, all required packages can be installed with a
system package manager (aptitude, yum) or with pip. Names of the packages can
differ, here are the corresponding commands for both cases.

For python2::

    #base xrt dependencies
    sudo apt-get install python-numpy python-scipy python-matplotlib python-tk spyder
    #file import
    sudo apt-get install python-pandas python-xlrd python-xlwt
    #GUI (xrtQook and xrtGlow)
    sudo apt-get install python-qt4 python-qt4-gl python-qwt5-qt4 freeglut3-dev python-opengl

For python3::

    sudo apt-get install python3-numpy python3-scipy python3-matplotlib python3-tk spyder3
    sudo apt-get install python3-pandas python3-xlrd
    sudo apt-get install python3-pyqt5 python3-pyqt5.qtopengl freeglut3-dev python3-opengl

If using pip, some packages still have to be installed from a system package
manager. In case of python 2, these are all GUI packages (see above) and
python-tk. pip will take care of the rest. Python2::

    pip install numpy scipy matplotlib spyder
    pip install pandas xlrd xlwt
    pip install pyopengl pyopengl_accelerate

In Python3 replace pip with pip3. PyQt5 is available from pip. python3-tk
should be installed with a system package manager. Python3::

    pip3 install pyqt5

.. rubric:: macOS

Use conda package manager to install all required packages::

    conda install numpy scipy matplotlib pytools spyder pyqt pyopengl pyopencl

PyOpenCL
~~~~~~~~

Before installing PyOpenCL you need at least one existing OpenCL implementation
(driver). OpenCL can come with a graphics card driver and/or with an OpenCL CPU
runtime. High profile graphics cards (those with a high FP64/FP32 ratio) are
advantageous.

On Windows, the binary package of pyopencl by C. Gohlke usually works out of
the box.

For installing on macOS and Linux, see the
`pyopencl site <https://documen.tician.de/pyopencl/misc.html>`_.
The following works on Ubuntu (used on Ubuntu 18.04 with the recommended Nvidia
proprietary driver or
`OpenCL runtime for Intel processors <https://software.intel.com/en-us/articles/opencl-drivers>`_)::

    apt-get install opencl-headers ocl-icd-opencl-dev
    pip install pyopencl

Instead of installing ocl-icd-opencl-dev, one can locate libOpenCL.so and
create a symbolic link in /usr/lib or any other lib folder in the path search.

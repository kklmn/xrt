Installation
------------

Unzip the .zip file into a suitable directory and use
sys.path.append(path-to-xrt)
in your script. You can also install xrt to the standard location by running 
python setup.py install from the directory where you have unzipped the archive,
which is less convenient if you try different versions of xrt and/or different
versions of python.

To run xrtQook, simply start xrtQookStart.pyw from xrt/gui or, if you have
installed xrt by running setup.py, type `xrtQookStart.pyw` from any location.

Note that python-64-bit is by ~20% faster than the 32-bit version (tested with
WinPython).

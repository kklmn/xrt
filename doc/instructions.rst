.. _instructions:

Detailed installation instructions
----------------------------------

Choose the installation method that fits how you want to use xrt. The options
below start with a ready-to-use environment and progress to managing your own
Python environment.

Pixi
~~~~

Pixi sets up Python, xrt from GitHub, and all its dependencies on Windows,
Linux and macOS. Start with a single ``pixi.toml`` file and launch xrtQook
with ``pixi run -e gui xrtQook``.

See :ref:`instructions-pixi` for installing Pixi on your platform, running xrt,
and setting up a developer checkout.

.. toctree::
   :hidden:

   instructions_pixi

WinPython on Windows
~~~~~~~~~~~~~~~~~~~~

`WinPython <https://winpython.github.io/>`_ is a portable Python distribution
for Windows. Choose a distribution that includes the scientific packages and
Spyder IDE. It provides most of the dependencies and lets you keep several
Python environments side by side.

Download the `xrt source archive
<https://github.com/kklmn/xrt/archive/refs/heads/master.zip>`_ and unzip it to
a suitable location. Open the command prompt supplied with WinPython and
install the packages listed under :ref:`instructions-source`; pip keeps
already installed packages that satisfy the requirements.

You can then open an example script or ``xrt/gui/xrtQookStart.py`` from the
source tree in Spyder. Run scripts from their own directory and use Spyder's
external terminal for scripts that use multiprocessing (see
:ref:`Spyder settings <instructions-spyder>`).

.. _instructions-source:

Running from source without installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xrt itself does not require compilation and can be used directly from its
source tree. This is useful for development or for sharing one copy of xrt
between several Python environments. Each environment needs its own
dependencies. For an environment managed by Pixi, use the developer workflow
in :ref:`instructions-pixi`.

Clone `the GitHub repository <https://github.com/kklmn/xrt>`_ or download and
unzip its source archive. The current source requires Python 3.9 or newer.
The requirements are recorded in `pyproject.toml
<https://github.com/kklmn/xrt/blob/master/pyproject.toml>`_. Install the
scientific and GUI dependencies in the Python environment you intend to use::

    python -m pip install numpy scipy matplotlib sphinx sphinxcontrib-jquery
    python -m pip install distro colorama pyopencl pyopengl siphash24 freetype-py
    python -m pip install PyQt5 PyQtWebEngine qtpy qtpynodeeditor numpy-stl

For the EPICS integration, also install its optional dependencies::

    python -m pip install softioc pyepics phoebusgen

Building the documentation additionally requires ``sphinx-tabs``.

Run scripts from their directory within ``examples`` or ``tests``. Many of
these scripts add a relative path to the source tree, so their working
directory matters. Edits to the Python source take effect when the script or
application is restarted.

If you have several copies of xrt, check which one Python imports::

    python -c "import xrt; print(xrt.__file__)"

Using xrt as a library: pip and Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These options suit users who already have a Python environment and an IDE
and want to import xrt in their own scripts.

.. rubric:: pip

Install the published release from PyPI into your chosen Python environment::

    python -m pip install xrt

To include the Qt GUI dependencies, use ``python -m pip install "xrt[pyqt5]"``.
For a PyPI prerelease, add ``--pre`` to the install command.

.. rubric:: Conda

Install the conda-forge package in your chosen Conda environment::

    conda install conda-forge::xrt

PyPI and conda-forge releases may differ from each other and from the current
GitHub source. Check the available versions on `PyPI
<https://pypi.org/project/xrt/>`_ and `conda-forge
<https://anaconda.org/conda-forge/xrt>`_.

.. rubric:: Installing from GitHub source

To install the downloaded or cloned source as a library, navigate to the
directory containing ``pyproject.toml`` and run::

    python -m pip install ".[all]"

This includes all optional dependencies. Add ``-e`` for an editable installation
if you want Python to use subsequent changes in that source tree.

.. _instructions-spyder:

.. rubric:: Spyder

For xrt scripts that use multiprocessing, select **Run in external terminal**
in Spyder's **Run** menu (Spyder 6), or configure the script to use an external
terminal in earlier versions. See the `Spyder FAQ
<https://docs.spyder-ide.org/current/faq.html>`_ for run configuration options.

.. _instructions-opencl:

PyOpenCL and OpenCL runtimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenCL calculations require an OpenCL implementation (runtime) in addition
to PyOpenCL. This can come with a graphics card driver or a CPU runtime.
Installing PyOpenCL alone does not install a GPU driver. Graphics cards with
a high FP64/FP32 ratio are advantageous for double-precision calculations.

PyOpenCL provides binary wheels for common platforms, so installing it with
pip usually does not require development headers or compiling from source.
See the `PyOpenCL installation instructions
<https://documen.tician.de/pyopencl/misc.html>`_ for current platform-specific
runtime choices, including CPU runtimes.

On Linux, if a conda-forge environment cannot find system OpenCL drivers,
install the package that exposes the system ICD files to that environment::

    conda install -c conda-forge ocl-icd-system

For a Pixi environment, run
``pixi add --platform linux-64 ocl-icd-system`` in its workspace instead.
For a CPU runtime on Linux or macOS, PoCL is another option: use
``conda install -c conda-forge pocl``. With Pixi, add it for the relevant
platform, for example ``pixi add --platform linux-64 pocl`` on Linux.

Linux GUI system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xrtQook and xrtGlow need system OpenGL libraries. In a Wayland session,
the GLX backend also requires XWayland. The following packages were tested
on RHEL 9 with KDE/Wayland::

    sudo dnf install \
        mesa-libGL mesa-dri-drivers \
        libglvnd-glx \
        xorg-x11-server-Xwayland

On Ubuntu Desktop, these dependencies are normally installed by default.
If the viewer fails to start, check that the following packages are installed::

    dpkg -l libgl1 libglx0 libglx-mesa0 libgl1-mesa-dri xwayland

Installed packages have ``ii`` at the beginning of their output line.
To install any missing packages, run::

    sudo apt update
    sudo apt install \
        libgl1 libglx0 libglx-mesa0 libgl1-mesa-dri xwayland

These are operating-system dependencies and are needed even when Python
packages are managed by Pixi or Conda.


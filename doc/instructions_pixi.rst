.. _instructions-pixi:

Installing and running xrt with Pixi
------------------------------------

Pixi manages a separate Python environment for xrt, including its scientific
packages, GUI and EPICS dependencies. It works on Windows, Linux and macOS.

Install Pixi
~~~~~~~~~~~~

Follow the `Pixi installation instructions
<https://pixi.prefix.dev/latest/installation/>`_ for your platform, then
restart your terminal so that the ``pixi`` command is available. A separate
Python installation is not needed.

Use xrt
~~~~~~~

Download `pixi.toml
<https://raw.githubusercontent.com/kklmn/xrt/master/pixi.toml>`_ and save it
in a folder of your choice. Open a terminal in that folder and
run::

    pixi run -e gui xrtQook

On the first run, Pixi downloads Python, xrt from GitHub and all dependencies,
then starts xrtQook. Subsequent runs reuse the environment. You can create a
beamline in xrtQook, run its calculations and view it in xrtGlow.

Develop xrt
~~~~~~~~~~~

Clone the repository and enter its directory::

    git clone https://github.com/kklmn/xrt.git
    cd xrt

Open ``pixi.toml`` in a text editor. Under ``[pypi-dependencies]``, uncomment
the local editable dependency and comment out the GitHub dependency.

Start xrtQook from the repository directory::

    pixi run -e gui xrtQook

Pixi now uses your local source tree. Edits to the Python code take effect
when you restart the application or script. You can also run your own scripts
in this environment::

    pixi run -e gui python my_script.py

For examples that use relative paths, change to the example's directory first
and run it with the same ``pixi run -e gui python`` command. Pixi finds the
manifest in the parent repository directory.

To check which source tree is imported, run::

    pixi run -e gui python -c "import xrt; print(xrt.__file__)"

Update xrt and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the single-file installation, update the GitHub dependency with::

    pixi update xrt
    pixi run -e gui xrtQook

For a developer checkout, update the local source through Git. For example,
on a branch tracking its upstream, ``git pull --ff-only`` downloads available
changes. Keep the local editable dependency selected in ``pixi.toml``.

In either workflow, update all dependencies within the requirements in
``pixi.toml`` with::

    pixi update
    pixi run -e gui xrtQook

Pixi records the resolved versions in ``pixi.lock``. Normal runs reuse those
versions. See the `pixi update reference
<https://pixi.prefix.dev/latest/reference/cli/pixi/update/>`_ for updating
selected packages or previewing changes with ``--dry-run``.

Recreate an environment
~~~~~~~~~~~~~~~~~~~~~~~

If you need to recreate the GUI environment, close applications using it and
run::

    pixi clean -e gui
    pixi run -e gui xrtQook

``pixi clean -e gui`` removes the installed GUI environment. Plain
``pixi clean`` removes all workspace environments and local caches. Your
source files, ``pixi.toml`` and ``pixi.lock`` are kept, and the next run
recreates the environment from the lockfile. Cleaning does not update the
locked versions. See the `pixi clean reference
<https://pixi.prefix.dev/latest/reference/cli/pixi/clean/>`_.


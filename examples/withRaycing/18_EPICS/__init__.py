# -*- coding: utf-8 -*-
r"""
Remote control with EPICS
-------------------------

Files in ``examples/withRaycing/18_EPICS``.

This example demonstrates how to control a virtual xrt beamline remotely
with EPICS_ control tools. All three scripts use the same xrtQook template,
``1crystal.xml``, containing a source, a crystal (``oe01``) and a screen
(``screen01``).

Install pythonSoftIOC_ and PyEpics_ before running the example. The
pythonSoftIOC distribution is named ``softioc``:

.. code-block:: console

    pip install softioc pyepics

Run the following scripts from this example's directory. Their EPICS process
variables (PVs) expose supported shape, position and orientation parameters;
not every field in the xrtQook template is controllable.

0. Generate Phoebus widgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``1cr_0_generate_bob.py`` generates display files for
`CS-Studio/Phoebus`_ from the template, with widgets for the supported PVs
of its beamline elements:

.. code-block:: console

    python 1cr_0_generate_bob.py

The files are written to ``bob/`` and use the PV prefix ``BL``. Using the
Phoebus displays is outside the scope of this example.

Screens publish a histogram to a waveform PV, here ``BL:screen01:image``.
This adds histogramming to the EPICS propagation workflow: normally a screen
returns a beam and the plots perform the histogramming. The waveform contains
the flattened two-dimensional histogram; ``screen01:histShape:width`` and
``screen01:histShape:height``, with the same prefix.

1. Run the headless beamline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``1cr_1_run_headless.py`` loads the same template and exposes PV controls
without opening a GUI:

.. code-block:: console

    python 1cr_1_run_headless.py

The IOC uses prefix ``BL`` and serves Channel Access locally. In a separate
Python console on the same computer, configure the local connection before
importing PyEpics' ``caget`` and ``caput`` functions:

.. code-block:: python

    import os
    os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

    from epics import caget, caput

Trigger a propagation to resolve the template's automatic crystal pitch and
screen position:

.. code-block:: python

    caput("BL:Acquire", 1, wait=True)

Once ``caget("BL:AcquireStatus")`` returns to zero after that propagation,
read the crystal pitch and the screen's vertical position:

.. code-block:: python

    caget("BL:oe01:pitch_RBV")          # radians
    caget("BL:screen01:center:z_RBV")   # mm

The ``_RBV`` suffix selects the current numerical readback, including values
resolved from ``auto``. Stop the headless script with Ctrl+C.

2. Control the beamline in xrtGlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``1cr_2_run_glow.py`` opens the same template in xrtGlow and starts its
EPICS controls with the distinct prefix ``VBL``. Its two main calls are:

.. code-block:: python

    beamLine = raycing.BeamLine(fileName=fileName)
    beamLine.glow(epicsPrefix="VBL")

Run it with:

.. code-block:: console

    python 1cr_2_run_glow.py

After propagation, use the same client console to read the same properties
with the new prefix:

.. code-block:: python

    caget("VBL:oe01:pitch_RBV")         # radians
    screen_z = caget("VBL:screen01:center:z_RBV")  # mm

Now change the crystal energy to 10005 eV, within the source's energy range,
and move the screen up by 1 mm:

.. code-block:: python

    caput("VBL:oe01:ENERGY", 10005.0, wait=True)
    caput("VBL:screen01:center:z", screen_z + 1.0, wait=True)

The ``ENERGY`` control is available only for crystal optics and double crystal
monochromators (DCMs); it adjusts their Bragg angle. With automatic updates
enabled, these writes trigger propagation and the result appears in xrtGlow.
``wait=True`` waits for the PV write to complete; propagation and updated
readbacks arrive afterward.

.. _EPICS: https://epics-controls.org/
.. _pythonSoftIOC: https://github.com/DiamondLightSource/pythonSoftIOC
.. _PyEpics: https://pyepics.github.io/pyepics/
.. _CS-Studio/Phoebus: https://github.com/ControlSystemStudio/phoebus

"""

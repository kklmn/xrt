# -*- coding: utf-8 -*-
r"""
Single crystal and powder diffraction
-------------------------------------

Files in ``examples/withRaycing/15_XRD``.

Single crystal diffraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~xrt.backends.raycing.materials.MonoCrystal` in `xrd_mono.py` is used
to model Laue diffraction pattern from a silicon monocrystal illuminated by a
polychromatic beam. The source samples photon energies uniformly from 1 to 100
keV. Different reciprocal-lattice planes satisfy the diffraction condition at
different energies, producing discrete spots on the detector.

The Lauegram shows the accumulated intensity on the detector, with darker spots
representing higher intensity. For each ray, the material selects one
reflection with a probability weighted by its intensity.

+-------------+--------------+
|   |xrd3d|   |  |lauegram|  |
+-------------+--------------+

.. |xrd3d| imagezoom:: _images/XRD_3D.png
   :scale: 80%
   :alt: Single crystal diffraction geometry and rays in xrtGlow.

.. |lauegram| imagezoom:: _images/Lauegram_Intensity.png
   :scale: 80%
   :alt: Simulated silicon Laue diffraction intensity on the detector.

.. warning::
   These diffraction calculations can use NumPy or OpenCL. Select an available
   OpenCL device with the sample's ``targetOpenCL`` setting.
   Set ``targetOpenCL = None`` to use NumPy diffraction.
   Larger reflection ranges and ray counts increase the computational load.


.. include:: ../examples/withRaycing/15_XRD/P02_PD.xml
   :start-after: <description>
   :end-before: </description>

"""

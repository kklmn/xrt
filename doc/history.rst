.. _history:

Version history
---------------

1.1.0 (26 Mar 2016):
    - :ref:`xrtQook <qook>` -- a GUI for creating scripts. Tested with Python 2 and 3,
      PyQt4, PyQt5 and PySide, Windows and Linux.

    - The examples have been restructured such that the creation of plots and
      scan generators has moved into module-level functions. With this
      structure, the examples are better readable.

    - Re-written startup routines for running OpenCL codes.

    - Improved alignment of :ref:`highly asymmetric crystals <get_dtheta>`.

    - Several minor bug fixes and updates.

    - The documentation has switched to MathJax from pngmath (nicer view of
      mathematics formulas).

1.0.2 (21 Jan 2016):
    - :ref:`A new analysis method <coh_signs_PCA>` for the quantification of
      degree of coherence based on PCA. It is equivalent to the modal analysis
      but is much cheaper.

    - :ref:`Examples of usage of xrt as a library for x-ray calculations
      <calc>`.

    - :class:`~xrt.backends.raycing.materials.Multilayer` can now be not only
      laterally graded but also depth graded, see a
      :ref:`reflectivity curve <multilayer_reflectivity>`.

1.0.1 (07 Jan 2016):
    - Bug fixes.

1.0.0 (05 Jan 2016):
    - xrt can now calculate sequential wave propagation. Added example for a
      :ref:`complete beamline<SoftiMAX>` comparing pure ray tracing,
      rays+wave combination and pure wave propagation.

    - Added :ref:`analysis of correlation functions<coh_signs>` as means of
      quantifying coherence properties.

    - Added example for using :ref:`mirrors with a figure error<warping>`
      defined as a tabulation or a function.

    - xrt can now run in both Python branches: 2 and 3, without translation.

    - The usage of pyopencl is extended to include multiple *simultaneous*
      platforms/devices.

    - Physical constants are unified in a single module ``physconsts``.

0.9.99 (12 Apr 2015):
    - xrt can now calculate :ref:`wave diffraction <waves>` via Kirchhoff
      integral. The present usage scenarios include diffraction at the last
      optical element.

    - Added examples for diffraction from :ref:`mirror <mirrorDiffraction>`,
      :ref:`slit <slitDiffraction>`, :ref:`double slit <YoungDiffraction>`,
      :ref:`grating <gratingDiffraction>` and :ref:`FZP <fzpDiffraction>`.

    - Diffraction efficiency of gratings and FZPs can now be calculated via
      wave diffraction. See the comparison with :ref:`REFLEC curves
      <gratingDiffraction>`.

    - Gratings and FZPs in ray tracing regime can now accept externally
      calculated efficiency weights per diffraction order, see
      :class:`~xrt.backends.raycing.materials.Material`.

    - :class:`~xrt.backends.raycing.oes.BlazedGrating` has been added.

    - Multilayers are now possible. See the mathematical description in
      :class:`~xrt.backends.raycing.materials.Multilayer`, a
      :ref:`reflectivity curve <multilayer_reflectivity>` and a
      :ref:`ray-tracing example of a scanning double multilayer monochromator
      <dmm>`.

    - A new :ref:`example of von Hamos spectrometer in circular and elliptical
      shapes <elliptical_VonHamos>`.

    - :ref:`The example of Montel mirror <montel>` has been revised: we have
      added a gap, user-selectable mirror shape and the local footprints
      colored by the number of reflections.

    - Export of plot attributes to Matlab has been added, see the parameter
      :ref:`persistentName <persistentName>`.

0.9.5 (Dec 2014):
    - :class:`~xrt.backends.raycing.screens.HemisphericScreen` has been added.

    - Extra angles for OE misalignments.

    - The constructor of apertures has changed! It now has `center` field, as
      many other objects. Before, it had `x` and `y`. This change requires
      small modifications in old application scripts.

    - :ref:`Example of von Hamos spectrometer <VonHamos>` and comparison with
      Rowland circle based spectrometers.

    - Minor bug fixes.

0.9.4 (13 Jun 2014):
    - :ref:`Near field <near_field_comparison>` calculations of undulators.

    - Search for intersections of rays with surface done with OpenCL.

    - Rotations of optical elements have been revised. Now, the sequence of
      pitch, roll and yaw can be re-defined by the user, which can be
      convenient when rotations are more than one.

    - Minor bug fixes.

0.9.3 (23 Apr 2014):
    - :class:`~xrt.backends.raycing.materials.CrystalFromCell` is added.
      Now, crystals of "any" structure can be ray-traced, not only of fcc and
      diamond-like structures, as was before.

    - Minor bug fixes.

0.9.2 (03 Apr 2014):
    - The code is prepared for fully automatic 2to3 conversion.

    - Undulator can now  have a :ref:`tapered gap <tapering_comparison>`.

    - Undulator can now be :ref:`calculated on GPU <calculations_on_GPU>`.

    - Natural source size of undulator radiation is now
      :math:`\sigma_R = \sqrt{2\lambda L}/(2\pi)`, as by Walker and by Ellaume;
      SPECTRA's value is two times smaller; the value by Kim (the orange
      booklet) is yet :math:`\sqrt{2}` times smaller.

0.9.1 (08 Jan 2014):
    - Minor bug fixes in OEs and examples.

0.9.0 (03 Jan 2014):
    - Internal implementation of synchrotron sources. Roman Chernikov as
      co-author of xrt.

    - Minor new features and minor bug fixes.

    - Example of bent tapered polycapillary.

0.8.1 (12 Sep 2013):
    - Bug fixes.

0.8.0:
    - Synchrotron sources (external),
    - Absolute flux units,
    - Coloring by power,
    - Power density isolines,
    - Gratings,
    - FZPs,
    - Bragg-Fresnel optics,
    - Multiple reflections,
    - Non-sequential optics.

0.7:
    - Several examples are generated on refractive and crystal optics:
      single- and double- crystal monochromators in Bragg and Laue geometries,
      bent and ground-bent crystal analyzers in Bragg and Laue geometries with
      optionally diced crystals, compound refractive lenses, plane and bent
      quarter-wave plates in Bragg and Laue geometries.

0.6:
    :mod:`~xrt.backends.raycing.materials` created for getting properties of
    elements, materials and crystal optics

0.5:
    :mod:`~xrt.backends.raycing` implemented with material- and polarization-
    dependent reflectivity

0.4:
    Own ray-tracing backend (:mod:`~xrt.backends.raycing`) implemented
    without intensity (reflectivity) and without synchrotron sources

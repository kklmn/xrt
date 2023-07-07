.. _history:

Version history
---------------

Available on GitHub (7 Jul 2023):
    - Enable calculations of elastically deformed crystals on GPUs. Based on
      PyTTE code [PyTTE1]_ [PyTTE2]_. Used both in ray tracing and
      :ref:`xrtBentXtal GUI <guis>` -- a GUI for comparative GPU-based bent
      crystal calculations.

    - Add :ref:`predefined material classes <predefmats>` in three categorees:
      crystals, compounds and elemental. Several crystal classes also include
      elastic constants needed to calculate bent crystal reflectivity.

    - All OpenCL tasks can be run on a :ref:`remote GPU server <oclserver>`.

    - Use `python properties
      <https://docs.python.org/3/library/functions.html#property>`_
      in OEs, screens, apertures etc. to transparently set object attributes.

    - Extend the :ref:`examples of 1D- and 2D-bent crystal analyzers
      <JohanssonTT>` with elastically deformed crystal reflectivity.

    - Add :ref:`docs <sampling-strategies>` on sampling strategies of
      syncrotron sources. Add an example
      :ref:`Undulator radiation through rectangular aperture <through-aperture>`
      that illustrates various sampling methods.

    - Add user classes of optical elements to xrtQook.

    - Add an example :ref:`Orbital Angular Momentum of helical undulator
      ratiation <OAM-HelicalU>`.

    - Bug fixes.

1.5.0 (8 Sep 2022):
    - Propagation of individual source modes, as waves, hybrid waves
      (partially as rays and then as waves) and only rays.
      See :ref:`Coherent mode decomposition and propagation <modes>`.

    - :class:`~xrt.backends.raycing.materials.Multilayer` can now be used in
      transmission. See :ref:`the mathematical description <descr_ml_tran>`,
      :ref:`a few comparative test curves <tests_ml_tran>` and the test script
      ``tests/test_multilayer_transmission.py``.

    - Add elliptical Gaussian beam, see
      `here <https://github.com/kklmn/xrt/issues/96>`_.

    - Ray-tracing of mosaic crystals in reflected and transmitted geometry.
      See the test script ``tests/test_mosaic_xtal_thin.py``.

    - Minor bug fixes.

1.4.0 (22 Sep 2021):
    - Major update for the :ref:`undulator sources <undulator-grid>` module:

    - Custom synchrotron sources calculation extended for non-periodic cases,
      including bending magnets.

    - Multiple performance optimizations, Gauss-Legendre grid replaced with
      Clenshaw-Curtis.

    - Extended functionality to :ref:`estimate and visualize convergence
      <test_undulator>`.

    - Added pure NumPy implementation for near field model and custom
      magnetic structures.

    - Added setters and getters, doing reset() is no longer required after
      post-init update of parameters.

    - Added asymmetric angular limits.

    - Angular limits get automatically extended to account for
      divergence/emittance, important if used with the slits/apertures
      matching angular acceptance.

    - Enable closed surfaces in xrtGlow.

    - Bug fixes.

1.3.5 (19 Nov 2020):
    - Bug fixes.

    - Variable d-spacing in crystals given by a user method; thanks to
      H. Gretarsson (DESY) for testing.

1.3.4 (21 May 2020):
    - Bug fixes and minor updates.

    - Several user stories made us insert warnings in the code and explanations
      in the :ref:`docs <mesh-methods>` about the proper usage of mesh-based
      methods of xrt Undulator.

    - Added custom orientation to apertures.

    - Added undulator source size from FT of the back propagated angular
      distribution (following Coïsson [Coïsson]_). The description to come in
      a paper about coherence properties.

1.3.3 (11 Mar 2019):
    - Added mosaic crystals. (thank you to B. Kozioziemski (LLNL) for deep
      testing)

    - Added Polygonal Apertures.

    - Bug fixes and minor updates.

1.3.2 (7 Jun 2018):
    - Bug fixes and minor updates.

1.3.1 (24 May 2018):
    - Added :ref:`detailed instructions for installing dependencies <instructions>`.

    - Added :ref:`Hermite-gaussian beam <test_waves>` to the tests of wave propagation.

    - Added :ref:`degree of transverse coherence <coh_signs_DoTC>` to analysis
      methods of coherence signatures.

    - Minor bug fixes and updates.

1.3.0 (25 Mar 2018):
    - Addition of :ref:`xrtGlow <guis>` -- a 3D beamline viewer.

    - Almost all old examples can now be viewed in xrtGlow as well, just select
      a proper value for the switch `showIn3D`. Those example scripts having a
      generator for making scans can also save a movie -- a series of grabbed
      3D views, as e.g. in :ref:`here <balder_pitch>`.

    - The documentation has moved to
      `Read the Docs <http://xrt.readthedocs.io>`_.
      It loads much faster and builds automatically from GitHub xrt sources.

    - Added 'Chantler total' (see
      :class:`~xrt.backends.raycing.materials.Material`) to the list of
      absorption tables. This table also adds inelastic scattering channels to
      the photoelectric absorption cross-section (thanks to B. Kozioziemski
      (LLNL) for discovering the need).

    - Added modelling of interdiffusion/roughness interface to
      :class:`~xrt.backends.raycing.materials.Multilayer`. Added
      :class:`~xrt.backends.raycing.materials.Coated` material -- a derivative
      class from :class:`~xrt.backends.raycing.materials.Multilayer` with a
      single reflective layer on a substrate.

    - A new module :mod:`~xrt.backends.raycing.coherence` that has functions
      for 1D and 2D analysis of coherence and functions for 1D plotting of
      degree of coherence and 2D plotting of eigen modes. Reworked analysis of
      coherence in :ref:`SoftiMAX` example.

    - Added electron energy spread dependence to the linear and angular sizes
      of undulator source. See the :ref:`formulation <undulator-source-size>`
      and an :ref:`application example<example-undulator-sizes>`.

    - Added :ref:`tests of optical elements <test_oes>` which currently have
      a test for asymmetric crystal optics (phase space volume conservation)
      and a test with backscattering at highly asymmetric crystals (comparison
      with experiment).

    - :ref:`Speed tests <tests>` include wave propagation on CPU and GPU nodes.

    - Numerous updates.

1.2.4 (3 May 2017):
    - Reworked and extended :ref:`Speed tests <tests>`.
    - Minor bug fixes and updates.

1.2.3 (19 Mar 2017):
    - Minor bug fixes.

1.2.2 (17 Mar 2017):
    - Numerous minor bug fixes and updates.

1.2.1 (19 Sep 2016):
    - Added SRW to some comparisons of synchrotron sources. See :ref:`here
      <undulator_highE>`.

    - As the major browsers stop supporting flash animations, we have rebuilt
      all our animations on the documentation pages. The images are now
      animated in JavaScript and feature on spot zooming by mouse click (not in
      IE though). The total size of the images has become smaller at better
      image quality and bidirectional animations (doubled number of frames).

    - xrtQook works now with Qt versions up to 5.7.

    - Minor bug fixes and updates.

1.2.0 (09 Jul 2016):
    - We've created a repository on `GitHub <https://github.com/kklmn/xrt>`_.

    - Added custom field undulator, see :ref:`an example <undulator_custom>`.

    - Improved and optimized formulas of undulator source, resulting in correct
      behaviour at high energies, see :ref:`here <undulator_highE>`.

    - Prepared for singular optics (vortex beams): added GaussianBeam and
      LaguerreGaussianBeam as geometric sources. We have used them for testing
      our Kirchhoff integration. The tests demonstrate identical images in
      analytical and numerically diffracted fields, see
      :ref:`here <test-Laguerre-Gaussian>`.

    - Modified CRLs to get loops over the lenses internally by
      ``multiple_refract`` method. The number of lenses can also be calculated
      internally given energy, material and focal distance.

    - Several minor bug fixes and updates.

1.1.0 (26 Mar 2016):
    - :ref:`xrtQook <guis>` -- a GUI for creating scripts. Tested with Python 2
      and 3, PyQt4, PyQt5 and PySide, Windows and Linux.

    - The examples have been restructured such that the creation of plots and
      scan generators has moved into module-level functions. With this
      structure, the examples are better readable.

    - Re-written startup routines for running OpenCL codes.

    - Improved alignment of :ref:`highly asymmetric crystals <get_dtheta>`.

    - Several minor bug fixes and updates.

    - The documentation has switched to MathJax from pngmath (nicer view of
      mathematical formulas).

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
      :math:`\sigma_R = \sqrt{2\lambda L}/(2\pi)`, as by Walker, by Ellaume
      and by Tanaka and Kitamura; the value by Kim (the orange booklet) is
      :math:`2\sqrt{2}` times smaller.

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

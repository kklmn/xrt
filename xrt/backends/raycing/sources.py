# -*- coding: utf-8 -*-
r"""
Sources
-------

Geometric sources
^^^^^^^^^^^^^^^^^

Module :mod:`~xrt.backends.raycing.sources` defines the container class
:class:`Beam` that holds beam properties as numpy arrays and defines the beam
sources. The sources are geometric sources in continuous and mesh forms and
:ref:`synchrotron sources <own-synchrotron-sources>`.

The intensity and polarization of each ray are determined via coherency matrix.
This way is appropriate for incoherent addition of rays, in which the
components of the coherency matrix of different rays are simply added.

.. autoclass:: Beam
.. autoclass:: GeometricSource()
   :members: __init__
.. autoclass:: MeshSource()
   :members: __init__
.. autoclass:: CollimatedMeshSource
.. autoclass:: GaussianBeam
   :members: __init__
.. autoclass:: LaguerreGaussianBeam
   :members: __init__
.. autoclass:: HermiteGaussianBeam
   :members: __init__

.. autofunction:: make_energy
.. autofunction:: make_polarization
.. autofunction:: rotate_coherency_matrix
.. autofunction:: shrink_source

.. _own-synchrotron-sources:

Synchrotron sources
^^^^^^^^^^^^^^^^^^^

.. note::

    In this section we consider z-axis to be directed along the beamline in
    order to be compatible with the cited works. Elsewhere in xrt z-axis looks
    upwards.

The synchrotron sources have two implementations: based on own fully pythonic
or OpenCL aided calculations and based on external codes [Urgent]_ and [WS]_.
The latter codes have some drawbacks, as demonstrated in the section
:ref:`comparison-synchrotron-sources`, but nonetheless can be used for
comparison purposes. If you are going to use them, the codes are freely
available as parts of [XOP]_ package.

.. [Urgent] R. P. Walker, B. Diviacco, URGENT, A computer program for
   calculating undulator radiation spectral, angular, polarization and power
   density properties, ST-M-91-12B, July 1991, Presented at 4th Int. Conf. on
   Synchrotron Radiation Instrumentation, Chester, England, Jul 15-19, 1991

.. [WS] R. J. Dejus, Computer simulations of the wiggler spectrum, Nucl.
   Instrum. Methods Phys. Res., Sect. A, **347** (1994) 56-60

.. [XOP] M. Sanchez del Rio, R. J. Dejus, XOP: A Multiplatform Graphical User
   Interface for Synchrotron Radiation Spectral and Optics Calculations,
   SPIE Proc. **3152** (1997) 148; web page:
   http://www.esrf.eu/Instrumentation/software/data-analysis/xop2.3.

The internal synchrotron sources are based on the following works:
[Kim]_ and [Walker]_.
We use the general formulation for the flux distribution in 3-dimensional
phase space (solid angle and energy) [Kim]_:

    .. math::
        \mathcal{F}(\theta,\psi,E) &= \alpha\frac{\Delta \omega}{\omega}
        \frac{I_e}{e^{-}}(A_{\sigma}^2 + A_{\pi}^2)\\

For the bending magnets the amplitudes can be calculated analytically using the
modified Bessel functions :math:`K_v(y)`:

   .. math::
       \begin{bmatrix}
       A_{\sigma}\\
       A_{\pi}
       \end{bmatrix} &= \frac{\sqrt{3}}{2\pi}\gamma\frac{\omega}{\omega_c}
       (1+\gamma^2\psi^2)\begin{bmatrix}-i K_{2/3}(\eta)\\
       \frac{\gamma\psi}{\sqrt{1+\gamma^2\psi^2}}
       K_{1/3}(\eta)\end{bmatrix}

where
   .. math::
       \gamma &= \frac{E_e}{m_{e}c^2} = 1957E_e[{\rm GeV}]\\
       \eta &= \frac{1}{2}\frac{\omega}{\omega_c}(1+\gamma^2\psi^2)^{3/2}\\
       \omega_c &= \frac{3\gamma^{3}c}{2\rho}\\
       \rho &= \frac{m_{e}c\gamma}{e^{-}B}

with :math:`I_e` - the current in the synchrotron ring, :math:`B` - magnetic
field strength, :math:`e^{-}, m_e`- the electron charge and mass,
:math:`c` - the speed of light.

Wiggler radiation relies on the same equation considering each pole as a
bending magnet with variable magnetic field/curvature radius:
:math:`\rho(\theta) = \sin(\arccos(\theta\gamma/K))`, where :math:`K` is
deflection parameter. Total flux is multiplied then by :math:`2N`, where
:math:`N` is the number of wiggler periods.

For the undulator sources the amplitude integrals must be calculated
numerically, starting from the magnetic field.

    .. math::
        \begin{bmatrix}
        A_{\sigma}\\
        A_{\pi}
        \end{bmatrix} &= \frac{1}{2\pi}\int\limits_{-\infty}^{+\infty}dt'
        \begin{bmatrix}\frac{[\textbf{n}\times[(\textbf{n}-
        \boldsymbol{\beta})\times\dot{\boldsymbol{\beta}}]]}
        {(1 - \textbf{n}\cdot\boldsymbol{\beta})^2}\end{bmatrix}_{x, y}
        e^{i\omega (t' + R(t')/c)}

    .. math::
        B_x &= B_{x0}\sin(2\pi z /\lambda_u + \phi),\\
        B_y &= B_{y0}\sin(2\pi z /\lambda_u)

the corresponding velosity components are

    .. math::
        \beta_x &= \frac{K_y}{\gamma}\cos(\omega_u t),\\
        \beta_y &= -\frac{K_x}{\gamma}\cos(\omega_u t + \phi)\\
        \beta_z &= \sqrt{1-\frac{1}{\gamma^2}-\beta_{x}^{2}-\beta_{y}^{2}},

where :math:`\omega_u = 2\pi c /\lambda_u` - undulator frequency,
:math:`\phi` - phase shift between the magnetic field components. In this
simple case one can consider only one period in the integral phase term
replacing the exponential series by a factor
:math:`\frac{\sin(N\pi\omega/\omega_1)}{\sin(\pi\omega/\omega_1)}`, where
:math:`\omega_1 = \frac{2\gamma^2}{1+K_x^2/2+K_y^2/2+\gamma^2(\theta^2+\psi^2)}
\omega_u`.

In the case of tapered undulator, the vertical magnetic field is multiplied by
an additional factor :math:`1 - \alpha z`, that in turn results in modification
of horizontal velocity and coordinate.

In the far-field approximation we consider the undulator as a point source and
replace the distance :math:`R` by a projection
:math:`-\mathbf{n}\cdot\mathbf{r}`, where :math:`\mathbf{n} =
[\theta,\psi,1-\frac{1}{2}(\theta^2+\psi^2)]` - direction to the observation
point, :math:`\mathbf{r} = [\frac{K_y}{\gamma}\sin(\omega_u t'),
-\frac{K_x}{\gamma}\sin(\omega_u t' + \phi)`,
:math:`\beta_m\omega t'-\frac{1}{8\gamma^2}
(K_y^2\sin(2\omega_u t')+K_x^2\sin(2\omega_u t'+2\phi))]` - electron
trajectory, :math:`\beta_m = 1-\frac{1}{2\gamma^2}-\frac{K_x^2}{4\gamma^2}-
\frac{K_y^2}{4\gamma^2}`.

Configurations with non-equivalent undulator periods i.e. tapered undulator
require integration over full undulator length, similar approach is
used for the near field calculations, where the undulator extension is taken
into account and the phase component in the integral is taken in its initial
form :math:`i\omega (t' + R(t')/c)`.

For the custom field configuratiuons, where the magnetic field components
are tabulated as functions of longitudinal coordinate
:math:`\textbf{B}=(B_{x}(z), B_{y}(z), B_{z}(z))`, preliminary numerical
calculation of the velocity and coordinate is nesessary. For that we solve the
system of differential equations

    .. math::
        \begin{bmatrix}{\ddot{x}\\ \ddot{y}\\ \ddot{z}}\end{bmatrix} &=
        \frac{e^{-}}{\gamma m_{e} c}
        \begin{bmatrix}{\beta_{y}B_{z}-B_{y}\\
        -\beta_{x}B_{z}+B_{x}\\
        -\beta_{y}B_{x}+\beta_{x}B_{y}}\end{bmatrix}

using the classical Runge-Kutta method. Integration step is varied in order to
provide the values of :math:`\beta` and :math:`\textbf{r}` in the knots of
the Gauss-Legendre grid.

For the Undulator and custom field models we directly calculate the integral
using the `Clenshaw-Curtis quadrature
<https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature>`, it proves
to converge as quickly as previously used Gauss-Legendre method, but the nodes
and weights calcuation is performed significantly faster. The size
of the integration grid is evaluated at the points of slowest convergence 
(highest energy, maximum angular deviation i.e. the corner of the plot) before
the start of intensity map calculation and then applied to all points.
This approach creates certain computational overhead for the on-axis/low energy
parts of the distibution, but enables efficient parallelization and gives
significant overall gain in performance. Initial evaluation typically takes
just a few seconds, but might get much longer for custom magnetic fields and
near edge calculations. If such heavy task is repeated many times for the given
angular and energy limits it might make sense to note the evaluated size of
the grid during the first run or call the :meth:`test_convergence` method once,
and then use the fixed grid by defining the *gNodes* at the init. 
Note also that the grid size will be automatically re-evaluated if any of the
limits/electron energy/undulator deflection parameter or period length are
redefined dynamically in the script.
Typical convergence threshold is defined by machine precizion multiplied by the
size of the integration grid. Default integration parameters proved to work
very well in most cases, but may fail if the angular and/or energy limits are
unreasonably wide. If in doubt, check the convergence
with :meth:`test_convergence`.


.. _undulator-source-size:

For the purpose of ray tracing (and this is not necessary for wave propagation)
the undulator source size is calculated following [TanakaKitamura]_. Their
formulation includes dependence on electron beam energy spread. The effective
linear and angular source sizes are given by

    .. math::
        \Sigma &= \left(\sigma_e^2 + \sigma_r^2\right)^{1/2}
        =\left(\varepsilon\beta + \frac{\lambda L}{2\pi^2}
        [Q(\sigma_\epsilon/4)]^{4/3}\right)^{1/2}\\
        \Sigma' &= \left({\sigma'_e}^2 + {\sigma'_r}^2\right)^{1/2}
        =\left(\varepsilon/\beta + \frac{\lambda}{2L}
        [Q(\sigma_\epsilon)]^2\right)^{1/2},

where :math:`\varepsilon` and :math:`\beta` are the electron beam emittance and
betatron function, the scaling function :math:`Q` is defined as

    .. math::
        Q(x) = \left(\frac{2x^2}{\exp(-2x^2)+(2\pi)^{1/2}x\ {\rm erf}
        (2^{1/2}x)-1}\right)^{1/2}

(notice :math:`Q(0)=1`) and :math:`\sigma_\epsilon` is the normalized energy
spread

    .. math::
        \sigma_\epsilon = 2\pi nN\sigma_E

i.e. the energy spread :math:`\sigma_E` divided by the undulator bandwidth
:math:`1/nN` of the n-th harmonic, with an extra factor :math:`2\pi`. See an
application example :ref:`here <example-undulator-sizes>`.

.. note::

   If you want to compare the source size with that by [SPECTRA]_, note that
   their radiation source size :math:`\sigma_r` is by a factor of 2 smaller in
   order to be compatible with the traditional formula by [Kim]_. In this
   aspect SPECTRA contradicts to their own paper [TanakaKitamura]_, see the
   paragraph after Eq(23).

.. [Kim] K.-J. Kim, Characteristics of Synchrotron Radiation, AIP Conference
   Proceedings, **184** (AIP, 1989) 565.

.. [Walker] R. Walker, Insertion devices: undulators and wigglers, CAS - CERN
   Accelerator School: Synchrotron Radiation and Free Electron Lasers,
   Grenoble, France, 22-27 Apr 1996: proceedings (CERN. Geneva, 1998) 129-190.

.. [TanakaKitamura] T. Tanaka and H. Kitamura, Universal function for the
   brilliance of undulator radiation considering the energy spread effect,
   J. Synchrotron Rad. **16** (2009) 380–6.

.. autoclass:: UndulatorUrgent()
   :members: __init__
.. autoclass:: WigglerWS()
   :members: __init__
.. autoclass:: BendingMagnetWS()
   :members: __init__

.. autoclass:: Undulator()
   :members: __init__, get_SIGMA, get_SIGMAP,
             real_photon_source_sizes, multi_electron_stack,
             intensities_on_mesh, power_vs_K, tuning_curves
.. autoclass:: Wiggler()
   :members: __init__
.. autoclass:: BendingMagnet()
   :members: __init__

.. _comparison-synchrotron-sources:

Comparison of synchrotron source codes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _mesh-methods:

Using xrt synchrotron sources on a mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main modus operandi of xrt synchrotron sources is to provide Monte Carlo
rays or wave samples. For comparing our sources with other codes – all of them
are fully deterministic, being defined on certain meshes – we also supply mesh
methods such as `intensities_on_mesh`, `power_vs_K` and `tuning_curves`. Note
that we do not provide any internal mesh optimization, as these mesh functions
are not our core objectives. Instead, the user themself should care about the
proper mesh limits and step sizes. In particular, the angular meshes must be
wider than the electron beam divergences in order to convolve the field
distribution with the electron distribution of non-zero emittance. The above
mentioned mesh methods will print a warning (new in version 1.3.4) if the
requested meshes are too narrow.

If you want to calculate flux through a narrow aperture, you first calculate
`intensities_on_mesh` on wide enough angular meshes and then slice the
intensity down to the needed aperture size. An example of such calculations is
given in `tests/raycing/test_undulator_on_mesh.py` which produces the following
plot (for a BESSY undulator, zero energy spread, as Urgent cannot take account
of it):

.. imagezoom:: _images/flux_case3_xrt_UrgentICALC1.png

Undulator spectrum across a harmonic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above six classes result in ray distributions with the density of rays
proportional to intensity. This requires an algorithm of Monte Carlo ray
sampling that needs a 3D (two directions + energy) intensity distribution. The
classes using the external mesh-based codes interpolate the source intensity
pre-calculated over the user specified mesh. The latter three classes (internal
implementation of synchrotron sources) do not require interpolation, which
eliminates two problems: artefacts of interpolation and the need for the mesh
optimization. However, this requires the calculations of intensity *for each*
ray.

For bending magnet and wiggler sources these calculations are not heavy and
are actually faster than 3D interpolation. See the absence of interpolation
artefacts in :ref:`Synchrotron sources <synchrotron-sources>` in the gallery.

For an undulator the calculations are much more demanding and for a wide
angular acceptance the Monte Carlo ray sampling can be extremely inefficient.
To improve the efficiency, a reasonably small acceptance should be considered.

There are several codes that can calculate undulator spectra: [Urgent]_,
[US]_, [SPECTRA]_. There is a common problem about them that the
energy spectrum may get *strong distortions* if calculated with a sparse
spatial and energy mesh. SPECTRA code seems to provide the best reference for
undulator spectra, which was used to optimize the meshes of the other codes.
The final comparison of the resulted undulator spectra around a single
harmonic is shown below.

.. [US] R. J. Dejus,
   *US: Program to calculate undulator spectra.* Part of XOP.

.. [SPECTRA] T. Tanaka and H. Kitamura, *SPECTRA - a synchrotron radiation
   calculation code*, J. Synchrotron Radiation **8** (2001) 1221-8.

.. note::

    If you are going to use UndulatorUrgent, you should optimize the spatial
    and energy meshes! The resulted ray distribution is strongly dependent on
    them, especially on the energy mesh. Try different numbers of points and
    various energy ranges.

.. imagezoom:: _images/compareUndulators.png

.. _undulator_highE:

Undulator spectrum at very high energies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The codes [Urgent]_ and [SPECTRA]_ result in saturation at high
energies (see the image below) thus leading to a divergent total power
integral. The false radiation has a circular off-axis shape. To the contrary,
xrt and [SRW]_ flux at high energies vanishes and follows the wiggler
approximation. More discussion will follow in a future journal article about
xrt.

.. imagezoom:: _images/flux_BioNanoMAX.png

Single-electron and multi-electron undulator radiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we compare single-electron and multi-electron (i.e. with a finite electron
beam size and energy spread) undulator radiation, as calculated by xrt and
[SRW]_. The calculations are done on a 3D mesh of energy (the long axis on the
images below) and two transverse angles. Notice also the duration of execution
given below each image. The 3D mesh was the following: theta = 321 point, -0.3
to +0.3 mrad, psi = 161 point, -0.15 to +0.15 mrad, energy: 301 point 1.5 to
4.5 keV.

.. [SRW] O. Chubar, P. Elleaume, *Accurate And Efficient Computation Of
   Synchrotron Radiation In The Near Field Region*, proc. of the EPAC98
   Conference, 22-26 June 1998, p.1177-1179.

+-------------+------------------------+------------------------+
|             |         SRW            |           xrt          |
+=============+========================+========================+
|  single     |    |srw_single|        |    |xrt_single|        |
|  electron   +------------------------+------------------------+
|             | execution time 974 s   | execution time 17.4 s  |
+-------------+------------------------+------------------------+
|  non-zero   |    |srw_non0em|        |    |xrt_non0em|        |
|  emittance  +------------------------+------------------------+
|             | execution time 65501 s | execution time 18.6 s  |
|             | (*sic*)                |                        |
+-------------+------------------------+------------------------+
|  non-zero   |   |srw_non0emsp|       |   |xrt_non0emsp|       |
|  emittance, +------------------------+------------------------+
|  non-zero   | execution time 66180 s | execution time 216 s   |
|  energy     | (*sic*)                |                        |
|  spread     |                        |                        |
+-------------+------------------------+------------------------+

.. |srw_single| imagezoom:: _images/mayavi_0em_2srw.png
.. |srw_non0em| imagezoom:: _images/mayavi_non0em_2srw.png
.. |srw_non0emsp| imagezoom:: _images/mayavi_non0em_non0spread_2srw.png
.. |xrt_single| imagezoom:: _images/mayavi_0em_3xrt.png
   :loc: upper-right-corner
.. |xrt_non0em| imagezoom:: _images/mayavi_non0em_3xrt.png
   :loc: upper-right-corner
.. |xrt_non0emsp| imagezoom:: _images/mayavi_non0em_non0spread_3xrt.png
   :loc: upper-right-corner

.. _tapering_comparison:

Undulator spectrum with tapering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectrum can be broadened by tapering the magnetic gap. The figure below
shows a comparison of xrt with [SPECTRA]_, [YAUP]_ and
experimental measurements [exp_taper]_. The gap values and the taper were
slightly varied in all three codes to reach the best match with the
experimental curve. We had to do so because in the other codes taper is not
clearly defined (where is the gap invariant -- at the center or at one of the
ends?) and also because the nominal experimental gap and taper are not fully
trustful.

.. [YAUP] B. I. Boyanov, G. Bunker, J. M. Lee, and T. I. Morrison, *Numerical
   Modeling of Tapered Undulators*, Nucl. Instr. Meth. **A339** (1994) 596-603.

.. [exp_taper] Measured on 27 Nov 2013 on P06 beamline at Petra 3,
   R. Chernikov and O. Müller, unpublished.

The source code is in ``examples\withRaycing\01_SynchrotronSources``

.. imagezoom:: _images/compareTaper.png

Notice that not only the band width is affected by tapering. Also the
transverse distribution attains inhomogeneity which varies with energy, see the
animation below. Notice also that such a picture is sensitive to emittance; the
one below was calculated for the emittance of Petra III ring.

.. imagezoom:: _images/taperingEnergyScan

Undulator spectrum in transverse plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The codes calculating undulator spectra -- [Urgent]_, [US]_, [SPECTRA]_ --
calculate either the spectrum of flux through a given aperture *or* the
transversal distribution at a fixed energy. It is not possible to
simultaneously have two dependencies: on energy *and* on transversal
coordinates.

Whereas xrt gives equal results to other codes in such univariate
distributions as flux through an aperture:

.. imagezoom:: _images/compareFlux.png

... and transversal distribution at a fixed energy:

+----------------+--------------------+--------------------+
|                |       SPECTRA      |        xrt         |
+================+====================+====================+
| *E* = 4850 eV  |                    |                    |
| (3rd harmonic) |   |spectra_lowE|   |     |xrt_lowE|     |
+----------------+--------------------+--------------------+
| *E* = 11350 eV |                    |                    |
| (7th harmonic) |   |spectra_highE|  |    |xrt_highE|     |
+----------------+--------------------+--------------------+

.. |spectra_lowE| imagezoom:: _images/undulator-E=04850eV-spectra.*
.. |spectra_highE| imagezoom:: _images/undulator-E=11350eV-spectra.*
.. |xrt_lowE| imagezoom:: _images/undulator-E=04850eV-xrt.*
   :loc: upper-right-corner
.. |xrt_highE| imagezoom:: _images/undulator-E=11350eV-xrt.*
   :loc: upper-right-corner

..., xrt can combine the two distributions in one image and thus be more
informative:

+----------------+-------------------+--------------------+
|                |      zoom in      |      zoom out      |
+================+===================+====================+
| *E* ≈ 4850 eV  |                   |                    |
| (3rd harmonic) |      |xrtLo|      |      |xrtLo5|      |
+----------------+-------------------+--------------------+
| *E* ≈ 11350 eV |                   |                    |
| (7th harmonic) |      |xrtHi|      |      |xrtHi5|      |
+----------------+-------------------+--------------------+

.. |xrtLo| imagezoom:: _images/oneHarmonic-E=04850eV-xrt.*
.. |xrtHi| imagezoom:: _images/oneHarmonic-E=11350eV-xrt.*
.. |xrtLo5| imagezoom:: _images/oneHarmonic-E=04850eV-xrt_x5.*
   :loc: upper-right-corner
.. |xrtHi5| imagezoom:: _images/oneHarmonic-E=11350eV-xrt_x5.*
   :loc: upper-right-corner

In particular, it is seen that divergence strongly depends on energy, even for
such a narrow energy band within one harmonic. It is also seen that the maximum
flux corresponds to slightly off-axis radiation (greenish color) but not to the
on-axis radiation (bluish color).

.. _near_field_comparison:

Undulator radiation in near field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice that on the following pictures the p-polarized flux is only ~6% of the
total flux.

+------------------------+--------------------+--------------------+
|  at 5 m                |      SPECTRA       |         xrt        |
+========================+====================+====================+
|   far field at 5 m,    |                    |                    |
|   full flux            |  |spectra_f05m0|   |    |xrt_f05m0|     |
+------------------------+--------------------+--------------------+
|   far field at 5 m,    |                    |                    |
|   p-polarized          |  |spectra_f05mP|   |    |xrt_f05mP|     |
+------------------------+--------------------+--------------------+
|   near field at 5 m,   |                    |                    |
|   full flux            |  |spectra_n05m0|   |    |xrt_n05m0|     |
+------------------------+--------------------+--------------------+
|   near field at 5 m,   |                    |                    |
|   p-polarized          |  |spectra_n05mP|   |    |xrt_n05mP|     |
+------------------------+--------------------+--------------------+

.. |spectra_f05m0| imagezoom:: _images/spectra-05m-far.png
.. |spectra_f05mP| imagezoom:: _images/spectra-05m-far_p.png
.. |spectra_n05m0| imagezoom:: _images/spectra-05m-near.png
.. |spectra_n05mP| imagezoom:: _images/spectra-05m-near_p.png

.. |xrt_f05m0| imagezoom:: _images/xrt-far05m1TotalFlux-rays.png
   :loc: upper-right-corner
.. |xrt_f05mP| imagezoom:: _images/xrt-far05m3vertFlux-rays.png
   :loc: upper-right-corner
.. |xrt_n05m0| imagezoom:: _images/xrt-near05m1TotalFlux-rays.png
   :loc: upper-right-corner
.. |xrt_n05mP| imagezoom:: _images/xrt-near05m3vertFlux-rays.png
   :loc: upper-right-corner

+------------------------+--------------------+--------------------+
|  at 25 m               |      SPECTRA       |         xrt        |
+========================+====================+====================+
|   far field at 25 m,   |                    |                    |
|   full flux            |  |spectra_f25m0|   |    |xrt_f25m0|     |
+------------------------+--------------------+--------------------+
|   far field at 25 m,   |                    |                    |
|   p-polarized          |  |spectra_f25mP|   |    |xrt_f25mP|     |
+------------------------+--------------------+--------------------+
|   near field at 25 m   |                    |                    |
|   full flux            |  |spectra_n25m0|   |    |xrt_n25m0|     |
+------------------------+--------------------+--------------------+
|   near field at 25 m,  |                    |                    |
|   p-polarized          |  |spectra_n25mP|   |    |xrt_n25mP|     |
+------------------------+--------------------+--------------------+

.. |spectra_f25m0| imagezoom:: _images/spectra-25m-far.png
.. |spectra_f25mP| imagezoom:: _images/spectra-25m-far_p.png
.. |spectra_n25m0| imagezoom:: _images/spectra-25m-near.png
.. |spectra_n25mP| imagezoom:: _images/spectra-25m-near_p.png

.. |xrt_f25m0| imagezoom:: _images/xrt-far25m1TotalFlux-rays.png
   :loc: upper-right-corner
.. |xrt_f25mP| imagezoom:: _images/xrt-far25m3vertFlux-rays.png
   :loc: upper-right-corner
.. |xrt_n25m0| imagezoom:: _images/xrt-near25m1TotalFlux-rays.png
   :loc: upper-right-corner
.. |xrt_n25mP| imagezoom:: _images/xrt-near25m3vertFlux-rays.png
   :loc: upper-right-corner

Field phase in near field
~~~~~~~~~~~~~~~~~~~~~~~~~

The phase of the radiation field on a flat screen as calculated by the three
codes is compared below. Notice that the visualization (brightness=intensity,
color=phase) is not by SRW and SPECTRA but was done by us.

+--------------------+--------------------+--------------------+
|     SRW [SRW]_     |      SPECTRA       |         xrt        |
+====================+====================+====================+
|      |srw_ps|      |    |spectra_ps|    |      |xrt_ps|      |
+--------------------+--------------------+--------------------+
|      |srw_pp|      |    |spectra_pp|    |      |xrt_pp|      |
+--------------------+--------------------+--------------------+

.. |srw_ps| imagezoom:: _images/phase_SRWres-0em-05m_s.png
.. |srw_pp| imagezoom:: _images/phase_SRWres-0em-05m_p.png
.. |spectra_ps| imagezoom:: _images/phase_spectra-near5-0em-field_s.png
.. |spectra_pp| imagezoom:: _images/phase_spectra-near5-0em-field_p.png
.. |xrt_ps| imagezoom:: _images/phase_xrt-near05m1horFlux-wave-filament.png
   :loc: upper-right-corner
.. |xrt_pp| imagezoom:: _images/phase_xrt-near05m3verFlux-wave-filament.png
   :loc: upper-right-corner

.. _example-undulator-sizes:

Undulator source size dependent on energy spread and detuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear and angular source sizes, as calculated with equations from
[TanakaKitamura]_ (summarized :ref:`here <undulator-source-size>`) for a
U19 undulator in MAX IV 3 GeV ring (:math:`E_1` = 1429 eV) with
:math:`\varepsilon_x` = 263 pmrad, :math:`\varepsilon_y` = 8 pmrad,
:math:`\beta_x` = 9 m and :math:`\beta_y` = 2 m, are shown below. Energy
spread mainly affects the angular sizes and not the linear ones.

The calculated sizes were further compared with those of the sampled field
(circles) at variable energies around the nominal harmonic positions, i.e. at
so called undulator detuning. To get the photon source size distribution, the
angular distributions of Es and Ep field amplitudes were Fourier transformed,
as described in [Coïsson]_. The sampled field sizes strongly vary due to
undulator detuning, as is better seen on the magnified insets. The size
variation by detuning is the underlying reason for the size dependence on
energy spread: with a non-zero energy spread the undulator becomes effectively
detuned for some electrons depending on their velocity.

The size of the circles is proportional to the total flux normalized to the
maximum at each respective harmonic. It sharply decreases at the higher energy
end of a harmonic and has a long tail at the lower energy end, in accordance
with the above examples.

The effect of energy detuning from the nominal undulator harmonic energy on the
photon source size is compared to the results by Coïsson [Coïsson]_ (crosses in
the figures below). He calculated the sizes for a single electron field, and
thus without emittance and energy spread. For comparison, we also sampled the
undulator field at zero energy spread and emittance.

.. imagezoom:: _images/undulatorLinearSize.png
.. imagezoom:: _images/undulatorAngularSize.png

.. [Coïsson] R. Coïsson, Effective phase space widths of undulator radiation,
   Opt. Eng. **27** (1988) 250–2.

"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "03 Jul 2016"
__all__ = ('GeometricSource', 'MeshSource', 'BendingMagnet', 'Wiggler',
           'Undulator')

from .sources_beams import Beam, copy_beam, rotate_coherency_matrix,\
    defaultEnergy
from .sources_geoms import GeometricSource, MeshSource, NESWSource,\
    CollimatedMeshSource, shrink_source, make_energy, make_polarization,\
    GaussianBeam, LaguerreGaussianBeam, HermiteGaussianBeam
from .sources_legacy import UndulatorUrgent, WigglerWS, BendingMagnetWS,\
    UndulatorSRW, SourceFromFieldSRW
from .sources_synchr import BendingMagnet, Wiggler, Undulator, SourceFromField

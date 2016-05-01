# -*- coding: utf-8 -*-
r"""
Sources
-------

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
.. autofunction:: make_energy
.. autofunction:: make_polarization
.. autofunction:: rotate_coherency_matrix
.. autofunction:: shrink_source

.. _own-synchrotron-sources:

Synchrotron sources
^^^^^^^^^^^^^^^^^^^

The synchrotron sources have two implementations: based on own fully pythonic
or OpenCL aided calculations and based on external codes Urgent [Urgent]_ and
WS [WS]_. The latter codes have some drawbacks, as demonstrated in the section
:ref:`comparison-synchrotron-sources`, but nonetheless can be used for
comparison purposes. If you are going to use them, the codes are freely
available as parts of XOP package [XOP]_.

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
        \end{bmatrix} &= \frac{\omega}{2\pi}\int\limits_{-\infty}^{+\infty}dt'
        \begin{bmatrix}\theta-\beta_x\\
        \psi-\beta_y \end{bmatrix}e^{i\omega (t' + R(t')/c)}

    .. math::
        B_x &= B_{x0}\sin(2\pi z /\lambda_u + \phi),\\
        B_y &= B_{y0}\sin(2\pi z /\lambda_u)

the corresponding velosity components are

    .. math::
        \beta_x &= \frac{K_y}{\gamma}\cos(\omega_u t),\\
        \beta_y &= -\frac{K_x}{\gamma}\cos(\omega_u t + \phi),

where :math:`\omega_u = 2\pi c /\lambda_u` - undulator frequency,
:math:`\phi` - phase shift between the magnetic field components. In this
simple case one can consider only one period in the integral phase term
replacing the exponential series by a factor of
:math:`\frac{\sin(N\pi\omega_1/\omega)}{\sin(\pi\omega_1/\omega)}`, where
:math:`\omega_1 = \frac{2\gamma^2}{1+K_x^2/2+K_y^2/2+\gamma^2(\theta^2+\psi^2)}
\omega_u`.

In the far-field approximation we consider the undulator as a point source and
replace the vector :math:`\mathbf{R}` by a projection
:math:`-\mathbf{n}\cdot\mathbf{r}`, where :math:`\mathbf{n} =
[\theta,\psi,1-\frac{1}{2}(\theta^2+\psi^2)]` - direction to the observation
point, :math:`\mathbf{r} = [\frac{K_y}{\gamma}\sin(\omega_u t'),
-\frac{K_x}{\gamma}\sin(\omega_u t' + \phi)`,
:math:`\beta_m\omega t'-\frac{1}{8\gamma^2}
(K_y^2\sin(2\omega_u t')+K_x^2\sin(2\omega_u t'+2\phi))]` - electron
trajectory, :math:`\beta_m = 1-\frac{1}{2\gamma^2}-\frac{K_x^2}{4\gamma^2}-
\frac{K_y^2}{4\gamma^2}`.

We directly calculate the integral using the Gauss-Legendre method. The
integration grid is generated around the center of each undulator period. Most
typical cases with reasonably narrow solid angles require around 10 integration
points per period for satisfactory convergence.

Custom field configurations i.e. tapered undulator require integration over
full undulator length, therefore the integration grid is multiplied by the
number of periods. Similar approach is used for the near field calculations,
where the undulator extension is taken into account.

.. [Kim] K.-J. Kim, Characteristics of Synchrotron Radiation, AIP Conference
   Proceedings, **184** (AIP, 1989) 565.

.. [Walker] R. Walker, Insertion devices: undulators and wigglers, CAS - CERN
   Accelerator School: Synchrotron Radiation and Free Electron Lasers,
   Grenoble, France, 22-27 Apr 1996: proceedings (CERN. Geneva, 1998) 129-190.

.. autoclass:: UndulatorUrgent()
   :members: __init__
.. autoclass:: WigglerWS()
   :members: __init__
.. autoclass:: BendingMagnetWS()
   :members: __init__

.. autoclass:: Undulator()
   :members: __init__, tuning_curves
.. autoclass:: Wiggler()
   :members: __init__
.. autoclass:: BendingMagnet()
   :members: __init__

.. _comparison-synchrotron-sources:

Comparison of synchrotron source codes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

There are several codes that can calculate undulator spectra: Urgent [Urgent]_,
US [US]_, SPECTRA [SPECTRA]_. There is a common problem about them that the
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

.. image:: _images/compareUndulators.png
   :scale: 50 %

.. _tapering_comparison:

Undulator spectrum with tapering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectrum can be broadened by tapering the magnetic gap. The figure below
shows a comparison of xrt with SPECTRA [SPECTRA]_, YAUP [YAUP]_ and
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

.. image:: _images/compareTaper.png
   :scale: 50 %

Notice that not only the band width is affected by tapering. Also the
transverse distribution attains inhomogeneity which varies with energy, see the
animation below. Notice also that such a picture is sensitive to emittance; the
one below was calculated for the emittance of Petra III ring.

.. image:: _images/taperingEnergyScan.swf
   :width: 300
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/taperingEnergyScan.swf

Undulator spectrum in transverse plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The codes calculating undulator spectra -- Urgent [Urgent]_, US [US]_, SPECTRA
[SPECTRA]_ -- calculate either the spectrum of flux through a given aperture
*or* the transversal distribution at a fixed energy. It is not possible to
simultaneously have two dependencies: on energy *and* on transversal
coordinates.

Whereas xrt gives equal results to other codes in such univariate
distributions as flux through an aperture:

.. image:: _images/compareFlux.png
   :scale: 50 %

... and transversal distribution at a fixed energy:

+----------------+--------------------+-------------+
|                | SPECTRA [SPECTRA]_ |     xrt     |
+================+====================+=============+
| *E* = 4850 eV  |                    |             |
| (3rd harmonic) |  |spectra_lowE|    |  |xrt_lowE| |
+----------------+--------------------+-------------+
| *E* = 11350 eV |                    |             |
| (7th harmonic) |   |spectra_highE|  | |xrt_highE| |
+----------------+--------------------+-------------+

.. |spectra_lowE| image:: _images/undulator-E=04850eV-spectra.*
   :scale: 50 %
   :align: bottom
.. |spectra_highE| image:: _images/undulator-E=11350eV-spectra.*
   :scale: 50 %
   :align: bottom
.. |xrt_lowE| image:: _images/undulator-E=04850eV-xrt.*
   :scale: 50 %
.. |xrt_highE| image:: _images/undulator-E=11350eV-xrt.*
   :scale: 50 %

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

.. |xrtLo| image:: _images/oneHarmonic-E=04850eV-xrt.*
   :scale: 50 %
.. |xrtHi| image:: _images/oneHarmonic-E=11350eV-xrt.*
   :scale: 50 %
.. |xrtLo5| image:: _images/oneHarmonic-E=04850eV-xrt_x5.*
   :scale: 50 %
.. |xrtHi5| image:: _images/oneHarmonic-E=11350eV-xrt_x5.*
   :scale: 50 %

In particular, it is seen that divergence strongly depends on energy, even for
such a narrow energy band within one harmonic. It is also seen that the maximum
flux corresponds to slightly off-axis radiation (greenish color) but not to the
on-axis radiation (bluish color).

.. _near_field_comparison:

Undulator radiation in near field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice that on the following pictures the p-polarized flux is almost 3 orders
of magnitude lower than the full flux.

+------------------------+--------------------+----------------+
|                        | SPECTRA [SPECTRA]_ |       xrt      |
+========================+====================+================+
|   near field at 5 m,   |                    |                |
|   full flux            |  |spectra_n05m0|   |  |xrt_n05m0|   |
+------------------------+--------------------+----------------+
|   near field at 5 m,   |                    |                |
|   p-polarized          |  |spectra_n05mP|   |  |xrt_n05mP|   |
+------------------------+--------------------+----------------+
|   far field at 5 m,    |                    |                |
|   full flux            |  |spectra_f05m0|   |  |xrt_f05m0|   |
+------------------------+--------------------+----------------+
|   far field at 5 m,    |                    |                |
|   p-polarized          |  |spectra_f05mP|   |  |xrt_f05mP|   |
+------------------------+--------------------+----------------+

.. |spectra_n05m0| image:: _images/spectra_near_R0=05m.*
   :scale: 50 %
   :align: bottom
.. |spectra_n05mP| image:: _images/spectra_near_R0=05m_p.*
   :scale: 50 %
   :align: bottom
.. |spectra_f05m0| image:: _images/spectra_far_R0=05m.*
   :scale: 50 %
   :align: bottom
.. |spectra_f05mP| image:: _images/spectra_far_R0=05m_p.*
   :scale: 50 %
   :align: bottom

.. |xrt_n05m0| image:: _images/1u_xrt4-n-near05m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_n05mP| image:: _images/1u_xrt4-n-near05m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f05m0| image:: _images/1u_xrt4-n-far05m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f05mP| image:: _images/1u_xrt4-n-far05m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom

+------------------------+--------------------+----------------+
|                        | SPECTRA [SPECTRA]_ |       xrt      |
+========================+====================+================+
|   near field at 25 m   |                    |                |
|   full flux            |  |spectra_n25m0|   |  |xrt_n25m0|   |
+------------------------+--------------------+----------------+
|   near field at 25 m,  |                    |                |
|   p-polarized          |  |spectra_n25mP|   |  |xrt_n25mP|   |
+------------------------+--------------------+----------------+
|   far field at 25 m,   |                    |                |
|   full flux            |  |spectra_f25m0|   |  |xrt_f25m0|   |
+------------------------+--------------------+----------------+
|   far field at 25 m,   |                    |                |
|   p-polarized          |  |spectra_f25mP|   |  |xrt_f25mP|   |
+------------------------+--------------------+----------------+

.. |spectra_n25m0| image:: _images/spectra_near_R0=25m.*
   :scale: 50 %
   :align: bottom
.. |spectra_n25mP| image:: _images/spectra_near_R0=25m_p.*
   :scale: 50 %
   :align: bottom
.. |spectra_f25m0| image:: _images/spectra_far_R0=25m.*
   :scale: 50 %
   :align: bottom
.. |spectra_f25mP| image:: _images/spectra_far_R0=25m_p.*
   :scale: 50 %
   :align: bottom

.. |xrt_n25m0| image:: _images/1u_xrt4-n-near25m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_n25mP| image:: _images/1u_xrt4-n-near25m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f25m0| image:: _images/1u_xrt4-n-far25m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f25mP| image:: _images/1u_xrt4-n-far25m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "12 Apr 2016"
__all__ = ('GeometricSource', 'MeshSource', 'BendingMagnet', 'Wiggler',
           'Undulator')

from .sources_beams import Beam, copy_beam, rotate_coherency_matrix,\
    defaultEnergy
from .sources_geoms import GeometricSource, MeshSource, NESWSource,\
    CollimatedMeshSource, shrink_source
from .sources_legacy import UndulatorUrgent, WigglerWS, BendingMagnetWS
from .sources_synchr import BendingMagnet, Wiggler, Undulator

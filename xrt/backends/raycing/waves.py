# -*- coding: utf-8 -*-
r"""
.. _waves:

Wave propagation (diffraction)
------------------------------

Time dependent diffraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

We start from the Kirchhoff integral theorem in the general (time-dependent)
form [Born & Wolf]:

    .. math::
        V(r,t)=\frac 1{4\pi }\int _S\left\{[V]\frac{\partial }{\partial n}
        \left(\frac 1 s\right)-\frac 1{\mathit{cs}}\frac{\partial s}{\partial
        n}\left[\frac{\partial V}{\partial t}\right]-\frac 1 s\left[\frac
        {\partial V}{\partial n}\right]\right\}\mathit{dS},

where the integration is performed over the selected surface :math:`S`,
:math:`s` is the distance between the point :math:`r` and the running point on
surface :math:`S`, :math:`\frac{\partial }{\partial n}` denotes differentiation
along the normal on the surface and the square brackets on :math:`V` terms
denote retarded values, i.e. the values at time :math:`t − s/c`. :math:`V` is a
scalar wave here but can represent any component of the actual electromagnetic
wave provided that the observation point is much further than the wave length
(surface currents are neglected here). :math:`V` depends on position and time;
this is something what we do not have in ray tracing. We obtain it from ray
characteristics by:

    .. math::
        V(s,t)=\frac 1{\sqrt{2\pi }}\int U_{\omega }(s)e^{-i\omega t}d\omega,

where :math:`U_{\omega }(s)` is interpreted as a monochromatic wave field and
therefore can be associated with a ray. Here, this is any component of the ray
polarization vector times its propagation factor :math:`e^{ikl(s)}`.
Substituting it into the Kirchhoff integral yields :math:`V(r,t)`. As we
visualize the wave fields in space and energy, the obtained :math:`V(r,t)` must
be back-Fourier transformed to the frequency domain represented by a new
re-sampled energy grid:

    .. math::
        U_{\omega '}(r)=\frac 1{\sqrt{2\pi }}\int V(r,t)e^{i\omega 't}
        \mathit{dt}.

Ingredients:

    .. math::
        [V]\frac{\partial }{\partial n}\left(\frac 1 s\right)=-\frac{(\hat
        {\vec s}\cdot \vec n)}{s^2}[V],

        \frac 1{\mathit{cs}}\frac{\partial s}{\partial n}\left[\frac{\partial
        V}{\partial t}\right]=\frac{ik} s(\hat{\vec s}\cdot \vec n)[V],

        \frac 1 s\left[\frac{\partial V}{\partial n}\right]=\frac{ik} s(\hat
        {\vec l}\cdot \vec n)[V],

where the hatted vectors are unit vectors: :math:`\hat {\vec l}` for the
incoming direction, :math:`\hat {\vec s}` for the outgoing direction, both
being variable over the diffracting surface. As :math:`1/s\ll k`, the 1st term
is negligible as compared to the second one.

Finally,

    .. math::
        U_{\omega '}(r)=\frac{-i}{8\pi ^2\hbar ^2c}\int e^{i(\omega '-\omega)t}
        \mathit{dt}\int \frac E s\left((\hat{\vec s}\cdot \vec n)+(\hat{\vec l}
        \cdot \vec n)\right)U_{\omega }(s)e^{ik(l(s)+s)}\mathit{dS}\mathit{dE}.

The time-dependent diffraction integral is not yet implemented in xrt.

Stationary diffraction
~~~~~~~~~~~~~~~~~~~~~~

If the time interval :math:`t` is infinite, the forward and back Fourier
transforms give unity. The Kirchhoff integral theorem is reduced then to its
monochromatic form. In this case the energy of the reconstructed wave is the
same as that of the incoming one. We can still use the general equation, where
we substitute:

    .. math::
        \delta (\omega -\omega ')=\frac 1{2\pi }\int e^{i(\omega '-\omega )t}
        \mathit{dt},

which yields:

    .. math::
        U_{\omega }(r)=-\frac {i k}{4\pi }\int \frac1 s\left((\hat{\vec s}\cdot
        \vec n)+(\hat{\vec l}\cdot \vec n)\right)U_{\omega }(s)e^{ik(l(s)+s)}
        \mathit{dS}.

How we treat non-monochromaticity? We repeat the sequence of ray-tracing
from the source down to the diffracting surface for each energy individually.
For synchrotron sources, we also assume a single electron trajectory (so called
"filament beam"). This single energy contributes fully coherently into the
diffraction integral. Different energies contribute incoherently, i.e. we add
their intensities, not amplitudes.

The input field amplitudes can, in principle, be taken from ray-tracing, as it
was done by [Shi_Reininger]_ as :math:`U_\omega(s) = \sqrt{I_{ray}(s)}`. This
has, however, a fundamental difficulty. The notion "intensity" in many ray
tracing programs, as in Shadow used in [Shi_Reininger]_, is different from the
physical meaning of intensity: "intensity" in Shadow is a placeholder for
reflectivity and transmittivity. The real intensity is represented by the
*density* of rays – this is the way the rays were sampled, while each ray has
:math:`I_{ray}(x, z) = 1` at the source [shadowGuide]_, regardless of the
intensity profile. Therefore the actual intensity must be reconstructed.
We tried to overcome this difficulty by computing the density of rays by (a)
histogramming and (b) kernel density estimation [KDE]_. However, the easiest
approach is to sample the source with uniform ray density (and not proportional
to intensity) and to assign to each ray its physical wave amplitudes as *s* and
*p* projections. In this case we do not have to reconstruct the physical
intensity. The uniform ray density is an option for the geometric and
synchrotron sources in :mod:`~xrt.backends.raycing.sources`.

Notice that this formulation does not require paraxial propagation and thus xrt
is more general than other wave propagation codes. For instance, it can work
with gratings and FZPs where the deflection angles may become large.

.. [Shi_Reininger] X. Shi, R. Reininger, M. Sanchez del Rio & L. Assoufid,
   A hybrid method for X-ray optics simulation: combining geometric ray-tracing
   and wavefront propagation, J. Synchrotron Rad. **21** (2014) 669–678.

.. [shadowGuide] F. Cerrina, "SHADOW User’s Guide" (1998).

.. [KDE] Michael G. Lerner (mglerner) (2013) http://www.mglerner.com/blog/?p=28

Normalization
~~~~~~~~~~~~~

The amplitude factors in the Kirchhoff integral assure that the diffracted wave
has correct intensity and flux. This fact appears to be very handy in
calculating the efficiency of a grating or an FZP in a particular diffraction
order. Without proper amplitude factors one would need to calculate all the
significant orders and renormalize their total flux to the incoming one.

The resulting amplitude is correct provided that the amplitudes on the
diffracting surface are properly normalized. The latter are normalized as
follows. First, the normalization constant :math:`X` is found from the flux
integral:

    .. math::
        F = X^2 \int \left(|E_s|^2 + |E_p|^2 \right) (\hat{\vec l}\cdot \vec n)
        \mathit{dS}

by means of its Monte-Carlo representation:

    .. math::
        X^2 = \frac{F N }{\sum \left(|E_s|^2 + |E_p|^2 \right)
        (\hat{\vec l}\cdot \vec n) S} \equiv \frac{F N }{\Sigma(J\angle)S}.

The area :math:`S` can be calculated by the user or it can be calculated
automatically by constructing a convex hull over the impact points of the
incoming rays. The voids, as in the case of a grating (shadowed areas) or an
FZP (the opaque zones) cannot be calculated by the convex hull and such cases
must be carefully considered by the user.

With the above normalization factors, the Kirchhoff integral calculated by
Monte-Carlo sampling gives the polarization components :math:`E_s` and
:math:`E_p` as (:math:`\gamma = s, p`):

    .. math::
        E_\gamma(r) = \frac{\sum K(r, s) E_\gamma(s) X S}{N} =
        \sum{K(r, s) E_\gamma(s)} \sqrt{\frac{F S} {\Sigma(J\angle) N}}.

Finally, the Kirchhoff intensity :math:`\left(|E_s|^2 + |E_p|^2\right)(r)` must
be integrated over the screen area to give the flux.

    .. note::
        The above normalization steps are automatically done inside
        :meth:`diffract`.

.. _seq_prop:

Sequential propagation
~~~~~~~~~~~~~~~~~~~~~~

In order to continue the propagation of a diffracted field to the next optical
element, not only the field distribution but also local propagation directions
are needed, regardless how the radiation is supposed to be propagated further
downstream: as rays or as a wave. The directions are given by the gradient
applied to the field amplitude. Because :math:`1/s\ll k` (validity condition
for the Kirchhoff integral), the by far most significant contribution to the
gradient is from the exponent function, while the gradient of the pre-exponent
factor is neglected. The new wave directions are thus given by a Kirchhoff-like
integral:

    .. math::
        {\vec \nabla } U_{\omega }(r) = \frac {k^2}{4\pi }
        \int \frac {\hat{\vec s}} s
        \left((\hat{\vec s}\cdot \vec n)+(\hat{\vec l}\cdot \vec n)\right)
        U_{\omega }(s)e^{ik(l(s)+s)} \mathit{dS}.

The resulted vector is complex valued. Taking the real part is not always
correct as the vector components may happen to be (almost) purely imaginary.
Our solution is to multiply the vector components by a conjugate phase factor
of the largest vector component and only then to take the real part.

The correctness of this approach to calculating the local wave directions can
be verified as follows. We first calculate the field distribution on a flat
screen, together with the local directions. The wave front, as the surface
normal to these directions, is calculated by linear integrals of the
corresponding angular projections. The calculated wave front surface is then
used as a new screen, where the diffracted wave is supposed to have a constant
phase. This is indeed demonstrated in :ref:`the example <wavefronts>` applying
phase as color axis. The sharp phase distribution is indicative of a true wave
front, which in turn justifies the correct local propagation directions.

After having found two polarization components and new directions on the
receiving surface, the last step is to reflect or refract the wave samples as
rays locally, taking the complex refractive indices for both polarizations.
This approach is applicable also to wave propagation regime, as the
reflectivity values are purely local to the impact points.

.. _usage_GPU_warnings:

Usage
~~~~~

    .. warning::
        You need a good graphics card for running these calculations!

    .. note::
        OpenCL platforms/devices can be inspected in xrtQook ('GPU' button)

    .. warning::
        Long calculation on GPU in Windows may result in the system message
        “Display driver stopped responding and has recovered” or python
        RuntimeError: out of resources (yes, it's about the driver response,
        not the lack of memory). The solution is to change TdrDelay registry
        key from the default value of 2 seconds to some hundreds or even
        thousands. Please refer to
        https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918(v=vs.85).aspx

    .. tip::
        If you plan to utilize xrt on a multi-GPU platform under Linux, we
        recommend KDE based systems (e.g. Kubuntu). It is enough to install the
        package ``fglrx-updates-dev`` without the complete graphics driver.

In contrast to ray propagation, where passing through an optical element
requires only one method (“reflect” for a reflective/refractive surface or
“propagate” for a slit), wave propagation in our implementation requires two or
three methods:

1. ``prepare_wave`` creates points on the receiving surface where the
   diffraction integral will be calculated. For an optical element or a slit
   the points are uniformly randomly distributed (therefore reasonable physical
   limits must be given); for a screen the points are determined by the screen
   pixels. The returned *wave* is an instance of class :class:`Beam`, where the
   arrays x, y, z are local to the receiving surface.
2. ``diffract`` (as imported function from xrt.backends.raycing.waves *or* as a
   method of the diffracted beam) takes a beam local to the diffracting surface
   and calculates the diffraction integrals at the points prepared by
   ``prepare_wave``. There are five scalar integrals: two for Es and Ep and
   three for the components of the diffracted direction (see the previous
   Section). All five integrals are calculated in the coordinates local to the
   diffracting surface but at the method's end these are transformed to the
   local coordinates of the receiving surface (remained in the *wave*
   container) and to the global coordinate system (a Beam object returned by
   ``diffract``).

For slits and screens, the two steps above are sufficient. For an optical
element, another step is necessary.

3. ``reflect`` method of the optical element is meant to take into account its
   material properties. The intersection points are already known, as provided
   by the previous ``prepare_wave``, which can be reported to ``reflect`` by
   ``noIntersectionSearch=True``. Here, ``reflect`` takes the beam right before
   the surface and propagates it to right after it. As a result of such a
   zero-length travel, the wave gets no additional propagation phase but only a
   complex-valued reflectivity coefficient and a new propagation direction.

These three methods are enough to describe wave propagation through the
complete beamline. The first two methods, ``prepare_wave`` and ``diffract``,
are split from each other because the diffraction calculations may need several
repeats in order to accumulate enough wave samples for attaining dark field at
the image periphery. The second method can reside in a loop that will
accumulate the complex valued field amplitudes in the same beam arrays defined
by the first method. In the supplied examples, ``prepare_wave`` for the last
screen is done before a loop and all the intermediate diffraction steps are
done within that loop.

The quality of the resulting diffraction images is mainly characterized by the
blackness of the dark field – the area of expected zero intensity. If the
statistics is not sufficient, the dark area is not black, and can even be
bright enough to mask the main spot. The contrast depends both on beamline
geometry (distances) and on the number of wave field samples (a parameter for
``prepare_wave``). Shorter distances require more samples for the same quality,
and for the distances shorter than a few meters one may have to reduce the
problem dimensionality by cutting in horizontal or vertical, see the
:ref:`examples of SoftiMAX<SoftiMAX>`. In the console output, ``diffract``
reports on *samples per zone* (meaning per Fresnel zone). As a rule of thumb,
this figure should be greater than ~10\ :sup:`4` for a good resulting quality.


.. automethod:: xrt.backends.raycing.oes.OE.prepare_wave
   :noindex:

.. automethod:: xrt.backends.raycing.apertures.RectangularAperture.prepare_wave
   :noindex:

.. automethod:: xrt.backends.raycing.screens.Screen.prepare_wave
   :noindex:

.. autofunction:: diffract

.. _coh_signs:

Coherence signatures
~~~~~~~~~~~~~~~~~~~~

A standard way to define coherence properties is via *mutual intensity J* and
*complex degree of coherence j* (DoC,  normalized *J*):

    .. math::
        J(x_1, y_1, x_2, y_2) \equiv J_{12} =
        \left<E(x_1, y_1)E^{*}(x_2, y_2)\right>\\
        j(x_1, y_1, x_2, y_2) \equiv j_{12} =
        \frac{J_{12}}{\left(J_{11}J_{22}\right)^{1/2}},

where the averaging :math:`\left<\ \right>` in :math:`J_{12}` is over different
realizations of filament electron beam (one realization per ``repeat``) and is
done for the field components :math:`E_s` or :math:`E_p` of a field diffracted
onto a given :math:`(x, y)` plane.

Both functions are Hermitian in respect to the exchange of points 1 and 2.
There are two common ways of working with them:

1. The horizontal and vertical directions are considered independently by
   placing the points 1 and 2 on a horizontal or vertical line symmetrically
   about the optical axis. DoC thus becomes a 1D function dependent on the
   distance between the points, e.g. as :math:`j_{12}^{\rm hor}=j(x_1-x_2)`.
   The intensity distribution is also determined over the same line as a 1D
   positional function, e.g. as :math:`I(x)`.

   The widths :math:`\sigma_x` and :math:`\xi_x` of the distributions
   :math:`I(x)` and :math:`j(x_1-x_2)` give the *coherent fraction*
   :math:`\zeta_x` [Vartanyants2010]_

    .. math::
        \zeta_x = \left(4\sigma_x^2/\xi_x^2 + 1\right)^{-1/2}.

2. The transverse field distribution can be analized integrally (not split into
   the horizontal and vertical projections) by performing the modal analysis
   consisting of solving the eigenvalue problem for the matrix
   :math:`J^{tr=1}_{12}` -- the matrix :math:`J_{12}` normalized to its trace
   -- and doing the standard eigendecomposition:

    .. math::
        J^{tr=1}(x_1, y_1, x_2, y_2) =
        \sum_i{w_i V_i(x_1, y_1)V_i^{+}(x_2, y_2)},

    with :math:`w_i, V_i` being the *i*\ th eigenvalue and eigenvector.
    :math:`w_0` is the fraction of the total flux contained in the 0th
    (coherent) mode or *coherent flux fraction*.

    .. note::
        The matrix :math:`J_{12}` is of the size
        (N\ :sub:`x`\ ×N\ :sub:`y`)², i.e. *squared* total pixel size of the
        image! In the current implementation, we use :meth:`eigh` method from
        ``scipy.linalg``, where a feasible image size should not exceed
        ~100×100 pixels (i.e. ~10\ :sup:`8` size of :math:`J_{12}`).

    .. note::
        For a fully coherent field :math:`j_{12}\equiv1` and
        :math:`w_0=1, w_i=0\ \forall i>0`, :math:`V_0` being the coherent
        field.

.. _coh_signs_PCA:

We also propose a third method that results in the same figures as the second
method above.

3. It uses Principal Component Analysis (PCA) applied to the filament images
   :math:`E(x, y)`. It consists of the following steps.

   a) Out of *r* repeats of :math:`E(x, y)` build a data matrix :math:`D` with
      N\ :sub:`x`\ ×N\ :sub:`y` rows and *r* columns.
   b) The matrix :math:`J_{12}` is equal to the product :math:`DD^{+}`. Instead
      of solving this huge eigenvalue problem of (N\ :sub:`x`\ ×N\ :sub:`y`)²
      size, we solve a typically smaller matrix :math:`D^{+}D` of the size
      *r*\ ².
   c) The biggest *r* eigenvalues of :math:`J_{12}` are equal to those of
      :math:`D^{+}D` [proof to present in the coming paper]. To find the
      primary (biggest) eigenvalue is the main objective of the modal analysis
      (item 2 above); PCA can provide it much easier due to the smaller size of
      the problem.
   d) Also the *eigen modes* of :math:`J_{12}=DD^{+}` can be found by PCA via
      the eigenvectors :math:`v`'s of :math:`D^{+}D`. The matrix
      :math:`Dv_iv_i^{+}` is of the size of :math:`D` and has all the columns
      proportional to each other [proof to present in the coming paper]. These
      columns are the *i*\ th principal components for the corresponding
      columns of :math:`D`. Being normalized, all the columns become equal and
      give the *i*\ th eigenvector of :math:`J_{12}`.

   Finally, PCA gives exactly the same information as the direct modal analysis
   (method No 2 above) but is cheaper to calculate by many orders of magnitude.

.. _coh_signs_DoTC:

One can define another measure of coherence as a single number, termed as
*degree of transverse coherence* (DoTC) [Saldin2008]_:

.. math::
    {\rm DoTC} = \frac{\iiiint |J_{12}|^2 dx_1 dy_1 dx_2 dy_2}
    {\left[\iint J_{11} dx_1 dy_1\right]^2}

.. [Saldin2008] E.L. Saldin, E.A. Schneidmiller, M.V. Yurkov, *Coherence
   properties of the radiation from X-ray free electron laser*, Opt. Commun.
   **281** (2008) 1179–88.

.. [Vartanyants2010] I.A. Vartanyants and A. Singer, *Coherence properties of
   hard x-ray synchrotron sources and x-ray free-electron lasers*,
   New Journal of Physics **12** (2010) 035004.

We propose to calculate DoTC from the matrix traces [derivation to present in
the coming paper] as:

4. a) DoTC = Tr(*J²*)/Tr²(*J*).

   b) DoTC = Tr(*D*\ :sup:`+`\ *DD*\ :sup:`+`\ *D*)/Tr²(*D*\ :sup:`+`\ *D*),
      with the matrix *D* defined above. The exactly same result as in (a) but
      obtained with smaller matrices.


.. note::
    A good test for the correctness of the obtained coherent fraction is to
    find it at various positions on propagating in free space, where the result
    is expected to be invariant. As appears in the
    :ref:`examples of SoftiMAX<SoftiMAX>`, the analysis based on DoC never
    gives an invariant coherent fraction at the scanned positions around the
    focus. The primary reason for this is the difficulty in the determination
    of the width of DoC, for the latter typically being a complex-shaped
    oscillatory curve. In contrast, the modal analysis (the PCA implementation
    is recommended) and the DoTC give the expected invariance.

Coherence analysis and related plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xrt.backends.raycing.coherence


Typical logic for a wave propagation study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. According to [Wolf]_, the visibility is not affected by a small bandwidth.
   It is therefore recommended to first work with strictly monochromatic
   radiation and check non-monochromaticity at the end.
2. Do a hybrid propagation (waves start somewhere closer to the end) at *zero
   emittance*. Put the screen at various positions around the main focus for
   seeing the depth of focus.
3. Examine the focus and the footprints and decide if vertical and horizontal
   cuts are necessary (when the dark field does not become black enough at a
   reasonably high number of wave samples).
4. Run non-zero electron emittance, which spoils both the focal size and the
   degree of coherence.
5. Try various ways to improve the focus and the degree of coherence: e.g.
   decrease the exit slit or elongate inter-optics distances.
6. Do hybrid propagation for a finite energy band to study the chromaticity in
   focus position.

.. [Wolf] E. Wolf. Introduction to the theory of coherence and polarization of
   light. Cambridge University Press, Cambridge, 2007.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "26 Mar 2016"
import numpy as np
from scipy.spatial import ConvexHull

import time
import os
from .. import raycing
from . import myopencl as mcl
from . import sources as rs
from .physconsts import CHBAR, CH
try:
    import pyopencl as cl  # analysis:ignore
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    isOpenCL = True
except ImportError:
    isOpenCL = False
_DEBUG = 20


if isOpenCL:
    waveCL = mcl.XRT_CL(r'diffract.cl')
    if waveCL.lastTargetOpenCL is None:
        waveCL = None
else:
    waveCL = None


def prepare_wave(fromOE, wave, xglo, yglo, zglo):
    """Creates the beam arrays used in wave diffraction calculations. The
    arrays reside in the container *wave*. *fromOE* is the diffracting element:
    a descendant from
    :class:`~xrt.backends.raycing.oes.OE`,
    :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
    :class:`~xrt.backends.raycing.apertures.RoundAperture`.
    *xglo*, *yglo* and *zglo* are global coordinates of the receiving points.
    This function is typicaly caled by ``prepare_wave`` methods of the class
    that represents the receiving surface:
    :class:`~xrt.backends.raycing.oes.OE`,
    :class:`~xrt.backends.raycing.apertures.RectangularAperture`,
    :class:`~xrt.backends.raycing.apertures.RoundAperture`,
    :class:`~xrt.backends.raycing.screens.Screen` or
    :class:`~xrt.backends.raycing.screens.HemisphericScreen`.
    """
    if not hasattr(wave, 'Es'):
        nrays = len(wave.x)
        wave.Es = np.zeros(nrays, dtype=complex)
        wave.Ep = np.zeros(nrays, dtype=complex)
    else:
        wave.Es[:] = 0
        wave.Ep[:] = 0
    wave.EsAcc = np.zeros_like(wave.Es)
    wave.EpAcc = np.zeros_like(wave.Es)
    wave.aEacc = np.zeros_like(wave.Es)
    wave.bEacc = np.zeros_like(wave.Es)
    wave.cEacc = np.zeros_like(wave.Es)
    wave.Jss[:] = 0
    wave.Jpp[:] = 0
    wave.Jsp[:] = 0

# global xglo, yglo, zglo are transformed into x, y, z which are fromOE-local:
    x, y, z = np.array(xglo), np.array(yglo), np.array(zglo)
    x -= fromOE.center[0]
    y -= fromOE.center[1]
    z -= fromOE.center[2]
    a0, b0 = fromOE.bl.sinAzimuth, fromOE.bl.cosAzimuth
    x[:], y[:] = raycing.rotate_z(x, y, b0, a0)

    if hasattr(fromOE, 'rotationSequence'):  # OE
        raycing.rotate_xyz(
            x, y, z, rotationSequence=fromOE.rotationSequence,
            pitch=-fromOE.pitch, roll=-fromOE.roll-fromOE.positionRoll,
            yaw=-fromOE.yaw)
        if fromOE.extraPitch or fromOE.extraRoll or fromOE.extraYaw:
            raycing.rotate_xyz(
                x, y, z, rotationSequence=fromOE.extraRotationSequence,
                pitch=-fromOE.extraPitch, roll=-fromOE.extraRoll,
                yaw=-fromOE.extraYaw)

    wave.xDiffr = x  # in fromOE local coordinates
    wave.yDiffr = y  # in fromOE local coordinates
    wave.zDiffr = z  # in fromOE local coordinates
    wave.rDiffr = (wave.xDiffr**2 + wave.yDiffr**2 + wave.zDiffr**2)**0.5
    wave.a[:] = wave.xDiffr / wave.rDiffr
    wave.b[:] = wave.yDiffr / wave.rDiffr
    wave.c[:] = wave.zDiffr / wave.rDiffr
    wave.path[:] = 0.
    wave.fromOE = fromOE
    wave.beamReflRays = np.long(0)
    wave.beamReflSumJ = 0.
    wave.beamReflSumJnl = 0.
    wave.diffract_repeats = np.long(0)
    return wave


def qualify_sampling(wave, E, goodlen):
    a = wave.xDiffr / wave.rDiffr  # a and c of wave change in diffract
    c = wave.zDiffr / wave.rDiffr
    if hasattr(wave, 'amin') and hasattr(wave, 'amax'):
        NAx = (wave.amax - wave.amin) * 0.5
    else:
        NAx = (a.max() - a.min()) * 0.5
    if hasattr(wave, 'cmin') and hasattr(wave, 'cmax'):
        NAz = (wave.cmax - wave.cmin) * 0.5
    else:
        NAz = (c.max() - c.min()) * 0.5
    invLambda = E / CH * 1e7
    fn = (NAx**2 + NAz**2) * wave.rDiffr.mean() * invLambda  # Fresnel number
    samplesPerZone = goodlen / fn
    return fn, samplesPerZone


def diffract(oeLocal, wave, targetOpenCL=raycing.targetOpenCL,
             precisionOpenCL=raycing.precisionOpenCL):
    r"""
    Calculates the diffracted field – the amplitudes and the local directions –
    contained in the *wave* object. The field on the diffracting surface is
    given by *oeLocal*. You can explicitly change OpenCL settings
    *targetOpenCL* and *precisionOpenCL* which are initially set to 'auto', see
    their explanation in :class:`xrt.backends.raycing.sources.Undulator`.
    """
    oe = wave.fromOE
    if _DEBUG > 10:
        t0 = time.time()

    good = oeLocal.state == 1
    goodlen = good.sum()
    if goodlen < 1e2:
        print("Not enough good rays at {0}: {1} of {2}".format(
              oe.name, goodlen, len(oeLocal.x)))
        return
#        goodIn = beam.state == 1
#        self.beamInRays += goodIn.sum()
#        self.beamInSumJ += (beam.Jss[goodIn] + beam.Jpp[goodIn]).sum()

    # this would be better in prepare_wave but energy is unknown there yet
    nf, spz = qualify_sampling(wave, oeLocal.E[0], goodlen)
    print("Effective Fresnel number = {0:.3g}, samples per zone = {1:.3g}".
          format(nf, spz))

    shouldCalculateArea = False
    if not hasattr(oeLocal, 'area'):
        shouldCalculateArea = True
    else:
        if oeLocal.area is None or (oeLocal.area <= 0):
            shouldCalculateArea = True

    if shouldCalculateArea:
        if _DEBUG > 20:
            print("The area of {0} under the beam will now be calculated".
                  format(oe.name))
        if hasattr(oe, 'rotationSequence'):  # OE
            secondDim = oeLocal.y
        elif hasattr(oe, 'propagate'):  # aperture
            secondDim = oeLocal.z
        elif hasattr(oe, 'shine'):  # source
            secondDim = oeLocal.z
        else:
            raise ValueError('Unknown diffracting element!')
        impactPoints = np.vstack((oeLocal.x[good], secondDim[good])).T
        try:  # convex hull
            hull = ConvexHull(impactPoints)
        except:  # QhullError  # analysis:ignore
            raise ValueError('cannot normalize this way!')
        outerPts = impactPoints[hull.vertices, :]
        lines = np.hstack([outerPts, np.roll(outerPts, -1, axis=0)])
        oeLocal.area = 0.5 * abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
        if hasattr(oeLocal, 'areaFraction'):
            oeLocal.area *= oeLocal.areaFraction
    if _DEBUG > 10:
        print("The area of {0} under the beam is {1:.4g}".format(
              oe.name, oeLocal.area))

    if hasattr(oe, 'rotationSequence'):  # OE
        if oe.isParametric:
            s, phi, r = oe.xyz_to_param(oeLocal.x, oeLocal.y, oeLocal.z)
            n = oe.local_n(s, phi)
        else:
            n = oe.local_n(oeLocal.x, oeLocal.y)[-3:]
        nl = (oeLocal.a*np.asarray([n[-3]]) + oeLocal.b*np.asarray([n[-2]]) +
              oeLocal.c*np.asarray([n[-1]])).flatten()
#    elif hasattr(oe, 'propagate'):  # aperture
    else:
        n = [0, 1, 0]
        nl = oeLocal.a*n[0] + oeLocal.b*n[1] + oeLocal.c*n[2]

    wave.diffract_repeats += 1
    wave.beamReflRays += goodlen
    wave.beamReflSumJ += (oeLocal.Jss[good] + oeLocal.Jpp[good]).sum()
    wave.beamReflSumJnl += abs(((oeLocal.Jss[good] + oeLocal.Jpp[good]) *
                               nl[good]).sum())

    if waveCL is None:
        kcode = _diffraction_integral_conv
    else:
        if targetOpenCL not in ['auto']:  # raycing.is_sequence(targetOpenCL):
            waveCL.set_cl(targetOpenCL, precisionOpenCL)
        kcode = _diffraction_integral_CL

    Es, Ep, aE, bE, cE = kcode(oeLocal, n, nl, wave, good)

    wave.EsAcc += Es
    wave.EpAcc += Ep
    wave.aEacc += aE
    wave.bEacc += bE
    wave.cEacc += cE
    wave.E[:] = oeLocal.E[0]
    wave.Es[:] = wave.EsAcc
    wave.Ep[:] = wave.EpAcc
    wave.Jss[:] = (wave.Es * np.conj(wave.Es)).real
    wave.Jpp[:] = (wave.Ep * np.conj(wave.Ep)).real
    wave.Jsp[:] = wave.Es * np.conj(wave.Ep)

    if hasattr(oe, 'rotationSequence'):  # OE
        toRealComp = wave.cEacc if abs(wave.cEacc[0]) > abs(wave.bEacc[0]) \
            else wave.bEacc
        toReal = np.exp(-1j * np.angle(toRealComp))
#    elif hasattr(oe, 'propagate'):  # aperture
    else:
        toReal = np.exp(-1j * np.angle(wave.bEacc))
    wave.a[:] = (wave.aEacc * toReal).real
    wave.b[:] = (wave.bEacc * toReal).real
    wave.c[:] = (wave.cEacc * toReal).real
#    wave.a[:] = np.sign(wave.aEacc.real) * np.abs(wave.aEacc)
#    wave.b[:] = np.sign(wave.bEacc.real) * np.abs(wave.bEacc)
#    wave.c[:] = np.sign(wave.cEacc.real) * np.abs(wave.cEacc)

    norm = (wave.a**2 + wave.b**2 + wave.c**2)**0.5
    norm[norm == 0] = 1.
    wave.a /= norm
    wave.b /= norm
    wave.c /= norm

    norm = wave.dS * oeLocal.area * wave.beamReflSumJ
    de = wave.beamReflRays * wave.beamReflSumJnl * wave.diffract_repeats
    if de > 0:
        norm /= de
    else:
        norm = 0
    wave.Jss *= norm
    wave.Jpp *= norm
    wave.Jsp *= norm
    wave.Es *= norm**0.5
    wave.Ep *= norm**0.5
    if hasattr(oeLocal, 'accepted'):  # for calculating flux
        wave.accepted = oeLocal.accepted
        wave.acceptedE = oeLocal.acceptedE
        wave.seeded = oeLocal.seeded
        wave.seededI = oeLocal.seededI * len(wave.x) / len(oeLocal.x)

    glo = rs.Beam(copyFrom=wave)
    glo.x[:] = wave.xDiffr
    glo.y[:] = wave.yDiffr
    glo.z[:] = wave.zDiffr
    if hasattr(oe, 'local_to_global'):
        oe.local_to_global(glo)

# rotate abc, coh.matrix, Es, Ep in the local system of the receiving surface
    if hasattr(wave, 'toOE'):
        # rotate coherency from oe local back to global for later transforming
        # it to toOE local:
        if hasattr(oe, 'rotationSequence'):  # OE
            rollAngle = oe.roll + oe.positionRoll
            cosY, sinY = np.cos(rollAngle), np.sin(rollAngle)
            Es[:], Ep[:] = raycing.rotate_y(Es, Ep, cosY, sinY)
        toOE = wave.toOE
        wave.a[:], wave.b[:], wave.c[:] = glo.a, glo.b, glo.c
        wave.Jss[:], wave.Jpp[:], wave.Jsp[:] = glo.Jss, glo.Jpp, glo.Jsp
        wave.Es[:], wave.Ep[:] = glo.Es, glo.Ep
        a0, b0 = toOE.bl.sinAzimuth, toOE.bl.cosAzimuth
        wave.a[:], wave.b[:] = raycing.rotate_z(wave.a, wave.b, b0, a0)

        if hasattr(toOE, 'rotationSequence'):  # OE
            if toOE.isParametric:
                s, phi, r = toOE.xyz_to_param(wave.x, wave.y, wave.z)
                oeNormal = list(toOE.local_n(s, phi))
            else:
                oeNormal = list(toOE.local_n(wave.x, wave.y))
            rollAngle = toOE.roll + toOE.positionRoll +\
                np.arctan2(oeNormal[-3], oeNormal[-1])
            wave.Jss[:], wave.Jpp[:], wave.Jsp[:] = \
                rs.rotate_coherency_matrix(wave, slice(None), -rollAngle)
            cosY, sinY = np.cos(rollAngle), np.sin(rollAngle)
            wave.Es[:], wave.Ep[:] = raycing.rotate_y(
                wave.Es, wave.Ep, cosY, -sinY)

            raycing.rotate_xyz(
                wave.a, wave.b, wave.c, rotationSequence=toOE.rotationSequence,
                pitch=-toOE.pitch, roll=-toOE.roll-toOE.positionRoll,
                yaw=-toOE.yaw)
            if toOE.extraPitch or toOE.extraRoll or toOE.extraYaw:
                raycing.rotate_xyz(
                    wave.a, wave.b, wave.c,
                    rotationSequence=toOE.extraRotationSequence,
                    pitch=-toOE.extraPitch, roll=-toOE.extraRoll,
                    yaw=-toOE.extraYaw)

            norm = -wave.a*oeNormal[-3] - wave.b*oeNormal[-2] -\
                wave.c*oeNormal[-1]
            norm[norm < 0] = 0
            for b in (wave, glo):
                b.Jss *= norm
                b.Jpp *= norm
                b.Jsp *= norm
                b.Es *= norm**0.5
                b.Ep *= norm**0.5

    if _DEBUG > 10:
        print("diffract on {0} completed in {1:.4f} s".format(
              oe.name, time.time()-t0))

    return glo


def _diffraction_integral_conv(oeLocal, n, nl, wave, good):
    # rows are by image and columns are by inBeam
    a = wave.xDiffr[:, np.newaxis] - oeLocal.x[good]
    b = wave.yDiffr[:, np.newaxis] - oeLocal.y[good]
    c = wave.zDiffr[:, np.newaxis] - oeLocal.z[good]
    pathAfter = (a**2 + b**2 + c**2)**0.5
    ns = (a*n[0] + b*n[1] + c*n[2]) / pathAfter
    k = oeLocal.E[good] / CHBAR * 1e7  # [mm^-1]
#    U = k*1j/(4*np.pi) * (nl[good]+ns) *\
#        np.exp(1j*k*(pathAfter+oeLocal.path[good])) / pathAfter
    U = k*1j/(4*np.pi) * (nl[good]+ns) * np.exp(1j*k*(pathAfter)) / pathAfter
    Es = (oeLocal.Es[good] * U).sum(axis=1)
    Ep = (oeLocal.Ep[good] * U).sum(axis=1)
    abcU = k**2/(4*np.pi) * (oeLocal.Es[good]+oeLocal.Ep[good]) * U / pathAfter
    aE = (abcU * a).sum(axis=1)
    bE = (abcU * b).sum(axis=1)
    cE = (abcU * c).sum(axis=1)
    return Es, Ep, aE, bE, cE


def _diffraction_integral_CL(oeLocal, n, nl, wave, good):
    myfloat = waveCL.cl_precisionF
    mycomplex = waveCL.cl_precisionC
#    myfloat = np.float64
#    mycomplex = np.complex128

    imageRays = np.int32(len(wave.xDiffr))
    frontRays = np.int32(len(oeLocal.x[good]))
    n = np.array(n)
    if len(n.shape) < 2:
        n = n[:, None] * np.ones(oeLocal.x.shape)

    scalarArgs = [frontRays]

    slicedROArgs = [myfloat(wave.xDiffr),  # x_mesh
                    myfloat(wave.yDiffr),  # y_mesh
                    myfloat(wave.zDiffr)]  # z_mesh

    k = oeLocal.E[good] / CHBAR * 1e7
    nonSlicedROArgs = [myfloat(nl[good]),  # nl_loc
                       mycomplex(oeLocal.Es[good]),  # Es_loc
                       mycomplex(oeLocal.Ep[good]),  # Ep_loc
                       myfloat(k),
                       np.array(
                           [oeLocal.x[good], oeLocal.y[good],
                            oeLocal.z[good], 0*oeLocal.z[good]],
                           order='F', dtype=myfloat),  # bOElo_coord
                       np.array(
                           [n[0][good], n[1][good],
                            n[2][good], 0*n[2][good]],
                           order='F', dtype=myfloat)]  # surface_normal

    slicedRWArgs = [np.zeros(imageRays, dtype=mycomplex),  # Es_res
                    np.zeros(imageRays, dtype=mycomplex),  # Ep_res
                    np.zeros(imageRays, dtype=mycomplex),  # aE_res
                    np.zeros(imageRays, dtype=mycomplex),  # bE_res
                    np.zeros(imageRays, dtype=mycomplex)]  # cE_res

    Es_res, Ep_res, aE_res, bE_res, cE_res = waveCL.run_parallel(
        'integrate_kirchhoff', scalarArgs, slicedROArgs,
        nonSlicedROArgs, slicedRWArgs, None, imageRays)

    return Es_res, Ep_res, aE_res, bE_res, cE_res

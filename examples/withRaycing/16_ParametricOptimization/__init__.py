# -*- coding: utf-8 -*-
r"""
Automatic optimization of monochromator detuning
------------------------------------------------

Files in ``examples/withRaycing/16_ParametricOptimization``.

``16.1_autoOptimization_detuning.py`` finds the relative pitch of the second
crystal of a double crystal monochromator that optimizes the energy
resolution. We wrap :func:`runner.run_ray_tracing() <xrt.runner.run_ray_tracing>`
and plot property extraction into an evaluation function that serves as input
for a standard SciPy optimizer (Brent).

Detuning the second crystal narrows the overlapping angular acceptance of the
DCM crystals, downstream slits restrict the transmitted divergence. The
smallest transmitted energy bandwidth therefore occurs at a nonzero detuning.

+-------------------------+-------------------------+
|       DCM Detuning      |       Convergence       |
+=========================+=========================+
|   |detuningAnimation|   |    |convergencePlot|    |
+-------------------------+-------------------------+

.. |detuningAnimation| animation:: _images/DCM_detuning_FSM
   :alt: &ensp;FSM footprint and energy spectrum during optimization of the
       second Si(111) crystal's relative pitch. Detuning angles are shown in
       microradians.

.. |convergencePlot| imagezoom:: _images/DCM_detuning_Convergence.png
   :loc: upper-right-corner
   :alt: &ensp;Energy FWHM versus objective evaluation index, starting at zero.

Resolution and flux
~~~~~~~~~~~~~~~~~~~

The sampled energy widths reach a minimum of about 0.47 eV, compared with
about 1.4 eV at zero detuning. Only energy width is minimized, flux is
recorded as a diagnostic.

+-------------------------+-------------------------+
| Energy width vs. angle  | Photon flux vs. angle   |
+=========================+=========================+
|       |detuningDE|      |      |detuningFlux|     |
+-------------------------+-------------------------+

.. |detuningDE| imagezoom:: _images/DCM_detuning_dE_vs_dTheta.png
   :alt: &ensp;Energy FWHM at every detuning angle evaluated by Brent's method.

.. |detuningFlux| imagezoom:: _images/DCM_detuning_Flux_vs_dTheta.png
   :loc: upper-right-corner
   :alt: &ensp;Transmitted photon flux at the same evaluated detuning angles.

"""

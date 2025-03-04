# -*- coding: utf-8 -*-
r"""
.. _modes:

Coherent mode decomposition and propagation
-------------------------------------------

The most straightforward way for studying the *coherent portion* of synchrotron
light is first to propagate a large collection of filament beams
(or *macro-electrons*) onto a surface of interest (typically, the focal plane
at the sample position) and then perform the modal analysis. The propagation
itself may become expensive when optical elements are situated close to each
other or when the analysis plane is out of focus. These cases require a large
number of wave samples which would allow only a small number of filament beams
processed in a reasonable time. In such cases, an inverse approach should be
preferred: first do the modal analysis of the source radiation and then
propagate the 0th or a few main modes corresponding to the biggest eigenvalues
(flux fractions).

In xrt, we calculate a collection of undulator field realizations at the first
beamline slit (a front-end slit), transform these fields into modes and save
them for the further propagation along the beamline. A selected number of
modes, also optionally a selected number of original fields, are prepared for
the following three ways of propagation:

1) As waves. The wave samples are sequentially diffracted from the first slit
   by calculating the Kirchhoff diffraction integral, see
   :ref:`Sequential propagation <seq_prop>`.

2) As hybrid waves. The wave samples at the first slit are treated as rays with
   the directions projected from the source center (for the modes) or from the
   filament beam position (for the fields). The beams of these rays can be
   propagated in ray tracing and then diffracted closer to the end of the
   beamline. Note that if the beams were propagated as rays down to the image
   plane, the individual fields would be seen as filament beam dots and the
   individual modes would be seen as centered dots, i.e. the source would have
   zero size.

3) As rays. The wave samples at the first slit are propagated backward to the
   source plane, which gives the field distribution in real space. The field at
   the first slit is sampled by its intensity; these samples give the ray
   directions. These beams are suitable for ray traycing down to the image
   plane. They are not suitable for wave propagation, as the propagation phase
   of each ray is destroyed by the above resampling procedure.

.. autofunction:: make_and_save_modes

.. autofunction:: use_saved

See a usage example in
``/examples/withRaycing/11_Waves/coherentModePropagation.py``.

A model case of 5 macro-electrons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. imagezoom:: _images/bl-modes.*

This example was made for the following beamline scheme, where the mirror M1 is
an ideal ellipsoid providing 1:1 image of the source.

The eigenmode decomposition at the first slit is accurate within the machine
precision, i.e. the total flux summed over all individual fields is equal to
the total flux summed over all modes, here down to ~1e-16 relative error. This
fact can be proven by comparing the last animation frames of the ‘field’ images
vs. ‘mode’ images. This comparison “all fields” vs. “all modes” was the main
objective of this “a few macro-electrons” test; in usual studies one would
typically need a few thousand macro-electrons and just a small number of modes.

.. rubric:: Individual fields

+-----+----------------------------+----------------------+
|     |        focus/source        |        FE slit       |
+=====+============================+======================+
| |r| |           |rfid|           |        |rffe|        |
+-----+----------------------------+----------------------+
| |h| |           |hfid|           |        |hffe|        |
+-----+----------------------------+----------------------+
| |w| |           |wfid|           |        |wffe|        |
+-----+----------------------------+----------------------+

.. |rfid| imagezoom:: _images/decomp-rays-fields-beamScreenF
.. |rffe| imagezoom:: _images/decomp-rays-fields-beamFElocal
   :loc: upper-right-corner
.. |hfid| imagezoom:: _images/decomp-hybr-fields-beamScreenF
.. |hffe| imagezoom:: _images/decomp-hybr-fields-beamFElocal
   :loc: upper-right-corner
.. |wfid| imagezoom:: _images/decomp-wave-fields-beamScreenF
   :loc: lower-left-corner
.. |wffe| imagezoom:: _images/decomp-wave-fields-beamFElocal
   :loc: lower-right-corner

.. rubric:: Coherent modes

+-----+----------------------------+----------------------+
|     |        focus/source        |        FE slit       |
+=====+============================+======================+
| |r| |           |rmid|           |        |rmfe|        |
+-----+----------------------------+----------------------+
| |h| |           |hmid|           |        |hmfe|        |
+-----+----------------------------+----------------------+
| |w| |           |wmid|           |        |wmfe|        |
+-----+----------------------------+----------------------+

.. |rmid| imagezoom:: _images/decomp-rays-modes-beamScreenF
.. |rmfe| imagezoom:: _images/decomp-rays-modes-beamFElocal
   :loc: upper-right-corner
.. |hmid| imagezoom:: _images/decomp-hybr-modes-beamScreenF
.. |hmfe| imagezoom:: _images/decomp-hybr-modes-beamFElocal
   :loc: upper-right-corner
.. |wmid| imagezoom:: _images/decomp-wave-modes-beamScreenF
   :loc: lower-left-corner
.. |wmfe| imagezoom:: _images/decomp-wave-modes-beamFElocal
   :loc: lower-right-corner

A model case of 5000 macro-electrons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Coherent radiation

The coherent radiation in the 0th mode has a weight (*coherent fraction*)
≈3.66%. Here it was individually propagated in three different ways:

+-----+----------------------------+----------------------+
|     |        focus/source        |        FE slit       |
+=====+============================+======================+
| |r| |           |rcid|           |        |rcfe|        |
+-----+----------------------------+----------------------+
| |h| |           |hcid|           |        |hcfe|        |
+-----+----------------------------+----------------------+
| |w| |           |wcid|           |        |wcfe|        |
+-----+----------------------------+----------------------+

.. |rcid| imagezoom:: _images/decomp-rays-modes-atFE-5000-beamScreenF.png
.. |rcfe| imagezoom:: _images/decomp-rays-modes-atFE-5000-beamFElocal.png
   :loc: upper-right-corner
.. |hcid| imagezoom:: _images/decomp-hybr-modes-atFE-5000-beamScreenF.png
.. |hcfe| imagezoom:: _images/decomp-hybr-modes-atFE-5000-beamFElocal.png
   :loc: upper-right-corner
.. |wcid| imagezoom:: _images/decomp-wave-modes-atFE-5000-beamScreenF.png
   :loc: lower-left-corner
.. |wcfe| imagezoom:: _images/decomp-wave-modes-atFE-5000-beamFElocal.png
   :loc: lower-right-corner

.. |r| replace:: rays
.. |h| replace:: hybrid
.. |w| replace:: wave

"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "6 Dec 2021"
import numpy as np
import scipy.linalg as spl
import scipy.interpolate as spi

# import matplotlib.pyplot as plt
import pickle
import time

from . import sources as rs
from . import apertures as ra
from . import waves as rw
from .physconsts import CHBAR


def _solve_modes(fields, nModes, phaseEsEp=0, name=''):
    """Calculates eigenmodes from a list of fields."""
    nElectrons = len(fields)
    if nModes > nElectrons:
        nModes = nElectrons

    Es = np.array([f[0] for f in fields]).T
    Ep = np.array([f[1] for f in fields]).T
    fluxField = (Es*np.conj(Es)).real.sum(axis=0)
    fluxField += (Ep*np.conj(Ep)).real.sum(axis=0)
    fluxFields = fluxField.sum()

    DE = Es + Ep*np.exp(1j*phaseEsEp)
    start = time.time()
    DTD = np.dot(DE.T.conjugate(), DE)
    DTD /= np.diag(DTD).sum()
    wAll = spl.eigh(DTD, eigvals_only=True)
    print("Eigenvalues 0 to {0} w={1}".format(nModes-1, wAll[::-1][:nModes]))
    eigvals = nElectrons-nModes, nElectrons-1
    try:
        wE, vE = spl.eigh(DTD, eigvals=eigvals)
    except TypeError:  # the kw 'eigvals' is gone
        wE, vE = spl.eigh(DTD, subset_by_index=eigvals)
    stop = time.time()
    if name:
        print("{0}:".format(name))
    print("PCA problem has taken {0} s".format(stop-start))
    print("Eigenvalues 0 to {0} w={1}".format(len(wE)-1, wE[::-1]))
    del DTD
    del DE

    modes = []
    fluxModes = 0.
    fluxMode = []
    for iMode in range(nModes):
        vv = vE[:, -1-iMode]
        mEs = np.dot(Es, vv)
        mEp = np.dot(Ep, vv)
        flux = (mEs*np.conj(mEs)).real.sum()
        flux += (mEs*np.conj(mEs)).real.sum()
        fluxMode.append(flux)
        fluxModes += flux
        modes.append((mEs, mEp))

    print('total flux in {0} field{1} = {2:.7g}'.format(
        nElectrons, 's' if nElectrons > 1 else '', fluxFields))
    print('total flux in {0} mode{1}  = {2:.7g}'.format(
        nModes, 's' if nModes > 1 else '', fluxModes))
    # if nElectrons > 1:
    #     fluxFieldStr = ', '.join('{0:.7g}'.format(f) for f in fluxField)
    #     print('flux in {0} field{1} = {2}'.format(
    #         nElectrons, 's' if nElectrons > 1 else '', fluxFieldStr))
    # if nModes > 1:
    #     fluxModeStr = ', '.join('{0:.7g}'.format(f) for f in fluxMode)
    #     print('flux in {0} mode{1}  = {2}'.format(
    #         nModes, 's' if nModes > 1 else '', fluxModeStr))
    return modes, wAll, fluxFields


def _sample_real_space(wave, lim):
    xx, zz = wave.x, wave.z
    II = wave.Jss + wave.Jpp
    mcSamples = len(II)
    nrep = 0
    Imax = II.max()
    # interp = LinearNDInterpolator((xx, zz), II, fill_value=0)  # not faster!
    interp = spi.CloughTocher2DInterpolator((xx, zz), II, fill_value=0)
    while True:
        rnds = np.random.rand(mcSamples, 3)
        rx = rnds[:, 0] * (lim[1]-lim[0]) + lim[0]
        rz = rnds[:, 1] * (lim[3]-lim[2]) + lim[2]
        rI = interp(rx, rz)
        Ipass = np.where(Imax*rnds[:, 2] < rI)[0]
        if len(Ipass) == 0:
            raise ValueError('No good samples in this seed!')

        if nrep == 0:
            x, z = rx[Ipass], rz[Ipass]
        else:
            x = np.concatenate((x, rx[Ipass]))
            z = np.concatenate((z, rz[Ipass]))

        if len(x) > mcSamples:
            x = x[:mcSamples]
            z = z[:mcSamples]
            print("{0} samples of {1}".format(len(x), mcSamples))
            break
        elif len(x) == mcSamples:
            print("{0} samples of {1}. Bingo!".format(len(x), mcSamples))
            break
        nrep += 1
        print("{0} samples of {1}".format(len(x), mcSamples))
    return x, z


def _make_hybr(fieldsFar, beamFar, waveFar, nElectronsSave):
    mPh = np.exp(-1e7j * waveFar.E/CHBAR * waveFar.rDiffr)
    beamsOrigin = []
    for iField, (Es, Ep) in enumerate(fieldsFar[:nElectronsSave]):
        resBeam = rs.BeamProxy(beamFar)
        resBeam.Es[:] = Es * mPh
        resBeam.Ep[:] = Ep * mPh
        resBeam.Jss[:] = (Es*np.conj(Es)).real
        resBeam.Jpp[:] = (Ep*np.conj(Ep)).real
        resBeam.Jsp[:] = (Es*np.conj(Ep))
        beamsOrigin.append(resBeam)
    return beamsOrigin


def _make_rays(bl, fieldsFar, waveFar, nElectronsSave, limitsOrigin, figStr):
    limitsFar = bl.slits[0].opening
    distanceFar = bl.slits[0].center[1] - bl.sources[0].center[1]
    waveBack = rs.BeamProxy(waveFar)
    waveBack.a *= -1
    waveBack.b *= -1
    waveBack.c *= -1
    waveBack.E[:] = -abs(waveBack.E[0])
    waveBack.area = (limitsFar[1]-limitsFar[0]) * (limitsFar[3]-limitsFar[2])
    slitOrigin = ra.RectangularAperture(bl, 'sourceFrame', (0, 0, 0),
                                        opening=limitsOrigin)
    beamsOrigin = []
    start = time.time()
    for iField, (Es, Ep) in enumerate(fieldsFar[:nElectronsSave]):
        pstr = '{0}beam {1} of {2}'.format(figStr+': ' if figStr else '',
                                           iField+1, nElectronsSave)
        print(pstr)
        waveBack.Es[:] = -Es
        waveBack.Ep[:] = -Ep
        waveBack.Jss[:] = (waveBack.Es*np.conj(waveBack.Es)).real
        waveBack.Jpp[:] = (waveBack.Ep*np.conj(waveBack.Ep)).real

        xFar, zFar = _sample_real_space(waveBack, limitsFar)
        # # plot far field
        # waveBackI = waveBack.Jss + waveBack.Jpp
        # ratio = int(limitsFar[1] / limitsFar[3])
        # fig = plt.figure(figsize=(2*ratio, 2))
        # ax = fig.add_subplot(111, aspect=1)
        # ax.scatter(waveBack.x, waveBack.z, c=waveBackI, s=0.5)
        # # ax.scatter(a*distFar, c*distFar, c='r', s=0.01)
        # ax.set_xlim(limitsFar[:2])
        # ax.set_ylim(limitsFar[2:])
        # fig.savefig('waveBack-{0}-{1}.png'.format(figStr, iField))

        waveOrigin = slitOrigin.prepare_wave(bl.slits[0], len(waveFar.x))
        rw.diffract(waveBack, waveOrigin)
        # we can diffract waveOrigin again but it would need many more samples

        # # plot origin field
        # waveOriginI = waveOrigin.Jss + waveOrigin.Jpp
        # ratio = int(limitsOrigin[1] / limitsOrigin[3])
        # fig = plt.figure(figsize=(2*ratio, 2))
        # ax = fig.add_subplot(111, aspect=1)
        # ax.scatter(waveOrigin.x, waveOrigin.z, c=waveOriginI, s=5)
        # ax.set_xlim(limitsOrigin[:2])
        # ax.set_ylim(limitsOrigin[2:])
        # fig.savefig('waveOrigin-{0}-{1}.png'.format(figStr, iField))

        resBeam = rs.BeamProxy(waveOrigin)
        resBeam.E[:] = abs(waveBack.E[0])
        resBeam.a[:] = xFar / distanceFar
        resBeam.c[:] = zFar / distanceFar
        resBeam.b[:] = (1 - resBeam.a**2 - resBeam.c**2)**0.5
        resBeam.path[:] = 0
        beamsOrigin.append(resBeam)
    stop = time.time()
    print("{0} beam{1[0]} ha{1[1]} been generated in {2} s"
          .format(nElectronsSave, ('', 's') if nElectronsSave == 1
                  else ('s', 've'), stop-start))
    return beamsOrigin


def make_and_save_modes(bl, nsamples, nElectrons, nElectronsSave, nModes,
                        fixedEnergy, phaseEsEp=0, output='all',
                        basename='local', limitsOrigin=[]):
    """
    Produces pickled files of *nModes* wave modes and *nElectronsSave* wave
    fields. The beamline object *bl* must have at least one aperture; the first
    of them fill be used to generate *nElectrons* fields for the eigenmode
    decomposition. The aperture will be sampled by *nsamples* wave samples.
    The fields are normalized such that the intensity sum over the sumples and
    over *nElectrons* gives the total flux returned as *totalFlux* in
    `use_saved()`. Note that the total flux is made independent (in average) of
    *nElectrons*.

    *fixedEnergy* is photon energy.

    *phaseEsEp* is phase difference between Es and Ep components.

    *output* can be 'all' or any combination of words 'wave', 'hybr', 'rays' in
    one string. The case 'rays' can take quite long time for field resampling.

    *basename* is the output file name that will be prepended by the selected
    'wave-', 'hybr-', 'rays-' output modes and appended by `.pickle`.

    *limitsOrigin* as [xmin, xmax, zmin, zmax] must be given for generating
    'rays'.
    """
    waveFar = bl.slits[0].prepare_wave(bl.sources[0], nsamples)
    fieldsFar = []
    start = time.time()
    limitsFar = bl.slits[0].opening
    area = (limitsFar[1]-limitsFar[0]) * (limitsFar[3]-limitsFar[2])
    dS = area / len(waveFar.x)
    norm = nElectrons**0.5  # then the sum flux is independent of nElectrons
    for iElectron in range(nElectrons):
        beamFar = bl.sources[0].shine(wave=waveFar, fixedEnergy=fixedEnergy)
        fieldsFar.append((np.copy(waveFar.Es)*dS**0.5 / norm,
                          np.copy(waveFar.Ep)*dS**0.5 / norm))
        pstr = 'macro-electron {0} of {1}'.format(iElectron+1, nElectrons)
        if iElectron % 10 == 9:
            pstr += ' in {0:.1f} s'.format(time.time() - start)
        print(pstr)
    stop = time.time()
    print("{0} wave{1[0]} ha{1[1]} been generated in {2} s".format(
        nElectrons, ('', 's') if nElectrons == 1 else ('s', 've'), stop-start))

    waveModes, wAll, fluxFields = _solve_modes(
        fieldsFar, nModes, phaseEsEp, 'waves')

    if 'wave' in output or 'all' in output:
        pickleName = 'wave-{0}.pickle'.format(basename)
        with open(pickleName, 'wb') as f:
            pickle.dump(
                (fieldsFar[:nElectronsSave], waveModes, wAll, fluxFields,
                 rs.BeamProxy(waveFar), fixedEnergy), f)

    if 'hybr' in output or 'all' in output:
        beamsOrigin = _make_hybr(fieldsFar, beamFar, waveFar, nElectronsSave)
        modesOrigin = _make_hybr(waveModes, beamFar, waveFar, nElectronsSave)
        pickleName = 'hybr-{0}.pickle'.format(basename)
        with open(pickleName, 'wb') as f:
            pickle.dump(
                (beamsOrigin, modesOrigin, wAll, fluxFields,
                 rs.BeamProxy(beamFar), fixedEnergy), f)

    if limitsOrigin:
        if 'rays' in output or 'all' in output:
            beamsOrigin = _make_rays(
                bl, fieldsFar, waveFar, nElectronsSave, limitsOrigin, 'fields')
            modesOrigin = _make_rays(
                bl, waveModes, waveFar, nElectronsSave, limitsOrigin, 'modes')
            pickleName = 'rays-{0}.pickle'.format(basename)
            with open(pickleName, 'wb') as f:
                pickle.dump(
                    (beamsOrigin, modesOrigin, wAll, fluxFields,
                     rs.BeamProxy(beamFar), fixedEnergy), f)

    print("Done with make_and_save_modes()")


def use_saved(what, basename):
    """Loads the saved modes and fields produced by `make_and_save_modes()`.

    *what* is a string starting with one of 'wave', 'hybr', 'rays' and ending
    with one of 'modes', 'fields'.

    *basename* is the same parameter used in `make_and_save_modes()`.

    Returnes a tuple (savedBeams, wAll, totalFlux), where
    *savedBeams* is a list of beam objects corresponding to *what*.
    *wAll* is a list of all eigenvalues (flux weights) of the eigenmode
    decomposition.
    *totalFlux* is described in `make_and_save_modes()`.
    """
    for header in ['wave', 'hybr', 'rays']:
        if what.startswith(header):
            break
    else:
        raise ValueError('Unknown type of saved object')

    pickleName = '{0}-{1}.pickle'.format(header, basename)
    with open(pickleName, 'rb') as f:
        fields, modes, wAll, totalFlux, wave, E0 = pickle.load(f)

    if what.startswith('wave'):
        savedWaves = modes if what.endswith('modes') else fields
        res = []
        for savedWave in savedWaves:
            w = rs.BeamProxy(wave)
            w.Es[:], w.Ep[:] = savedWave
            w.Jss[:] = (w.Es * np.conj(w.Es)).real
            w.Jpp[:] = (w.Ep * np.conj(w.Ep)).real
            w.Jsp[:] = (w.Es * np.conj(w.Ep))
            res.append(w)

    elif what.startswith('hybr') or what.startswith('rays'):
        res = modes if what.endswith('modes') else fields

    print('total flux = {0:.7g}'.format(totalFlux))
    if len(wAll) > 1:
        print('biggest {0} eigenvalues: {1}'.format(
            len(modes), wAll[::-1][:len(modes)]))
    return res, wAll, totalFlux

# -*- coding: utf-8 -*-
r"""
The module :mod:`~xrt.backends.raycing.coherence` has functions for 1D and 2D
analysis of coherence and functions for 1D plotting of degree of coherence and
and 2D plotting of eigen modes.

The input for the analysis functions is a 3D stack of field images. It can be
obtained directly from the undulator class, or from a plot object after several
repeats of wave propagation of a filament beam through a beamline. Examples can
be found in ``...\tests\raycing\test_coherent_fraction_stack.py`` and in
:ref:`SoftiMAX`.

.. autofunction:: calc_1D_coherent_fraction

.. autofunction:: plot_1D_degree_of_coherence

.. autofunction:: calc_degree_of_transverse_coherence_4D

.. autofunction:: calc_degree_of_transverse_coherence_PCA

.. autofunction:: calc_eigen_modes_4D

.. autofunction:: calc_eigen_modes_PCA

.. autofunction:: plot_eigen_modes

"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "21 Jan 2018"
import numpy as np
import scipy.linalg as spl
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


def calc_1D_coherent_fraction(U, axisName, axis, p=0):
    """
    Calculates 1D degree of coherence (DoC). From its width in respect to the
    width of intensity distribution also infers the coherent fraction. Both
    widths are rms. The one of intensity is calculated over the whole axis, the
    one of DoC is calculated between the first local minima around the center
    provided that these minima are lower than 0.5.

    *U*: complex valued ndarray, shape(repeats, nx, ny)
        3D stack of field images. For a 1D cut along *axis*, the middle of the
        other dimension is sliced out.

    *axis*: str, one of 'x' or ('y' or 'z')
        Specifies the 1D axis of interest.

    *p*: float, distance to screen
        If non-zero, the calculated mutual intensity will be divided by *p*\ ².
        This is useful to get proper physical units of the returned intensity
        if the function is applied directly to the field stacks given by
        Undulator.multi_electron_stack() that is calculated in angular units.

    Returns a tuple of mutual intensity, 1D intensity, 1D DoC, rms width of
    intensity, rms width of DoC (between the local minima, see above), the
    position of the minima (only the positive side) and the coherent fraction.
    This tuple can be fed to the plotting function
    :func:`plot_1D_degree_of_coherence`.

    """
    repeats, binsx, binsz = U.shape
    if axisName == 'x':
        Uc = U[:, :, binsz//2]
    elif axisName in ['y', 'z']:
        Uc = U[:, binsx//2, :]
    else:
        raise ValueError("unknown axis")
    J = np.dot(Uc.T.conjugate(), Uc) / repeats
    if p > 0:  # p is distance
        J /= p**2
    II = np.abs(np.diag(J))  # intensity as the main diagonal
#    II = (np.abs(Uc)**2).sum(axis=0) / repeats  # intensity by definition
    J /= II**0.5 * II[:, np.newaxis]**0.5
    Jd = np.abs(np.diag(np.fliplr(J)))  # DoC as the cross-diagonal

    varI = (II * axis**2).sum() / II.sum()
    axisEx = 2 * axis

    lm = argrelextrema(Jd, np.less)[0]  # local min
    # for variance up to the 1st local minimum which is < 0.5:
    lm = lm[(axisEx[lm] > 0) & (Jd[lm] < 0.5)]
    if len(lm) > 0:
        cond = np.abs(axisEx) <= axisEx[lm[0]]
        limJd = axisEx[lm[0]]
    else:
        cond = slice(None)  # for unconstrained variance calculation
        limJd = None
    varJd = (Jd * axisEx**2)[cond].sum() / Jd[cond].sum()
    cohFr = (4*varI/varJd + 1)**(-0.5)
    return J, II, Jd, varI, varJd, limJd, cohFr


def plot_1D_degree_of_coherence(data1D, axisName, axis, unit="mm", fig2=None,
                                isIntensityNormalized=False, locLegend='auto'):
    """
    Provides two plots: a 2D plot of mutual intensity and a 1D plot of
    intensity and DoC. The latter plot can be shared between the two 1D axes if
    this function is invoked two times.

    *data1D*: tuple returned by :func:`calc_1D_coherent_fraction`.

    *axisName*: str, used in labels.

    *axis*: 1D array of abscissa.

    *unit*: str, used in labels.

    *fig2*: matplotlib figure object, if needed for shared plotting of two 1D
    axes.

    *isIntensityNormalized*: bool, controls the intensity axis label.

    *locLegend*: str, legend location in matplotlib style.

    Returns the two figure objects for the user to add suptitles and to export
    to image files.

    Plot examples for one and two 1D curves:

    .. imagezoom:: _images/_DOC-1D-1.*
    .. imagezoom:: _images/_DOC-1D-2.*

    """
    Jx, Ix, Jdx, varIx, varJdx, limJd, cohFrx = data1D

    fig1 = plt.figure()

    ax = fig1.add_subplot(111, aspect=1)
    am = axis.max()
    unitStr = "({0})".format(unit) if unit else ""
    ax.set_xlabel("{0}$_1$ {1}".format(axisName, unitStr))
    ax.set_ylabel("{0}$_2$ {1}".format(axisName, unitStr))
    ax.imshow(np.abs(Jx), origin='upper', extent=[-am, am, am, -am],
              interpolation='none')

    if fig2 is None:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        if isIntensityNormalized:
            ax2.set_ylabel("normalized intensity (a.u.)", color="C0")
        else:
            ax2.set_ylabel("intensity (ph/s/{0}²/0.1%BW)".format(unit),
                           color="C0")
        ax2.tick_params(axis='y', colors="C0")
        ax2.myLabel = "{0} or {0}$_1-${0}$_2$".format(axisName)
        ax2.myUnit = " ({0})".format(unit)
        ax3 = ax2.twinx()
        ax3.set_ylabel("degree of coherence", color="C1")
        ax3.tick_params(axis='y', colors="C1")
        ax3.spines['left'].set_color("C0")
        ax3.spines['right'].set_color("C1")
        lstyle = '-'
        loc = "center left" if locLegend == 'auto' else locLegend
        rHeight = 1.02
    else:
        ax2, ax3 = fig2.axes
        ax2.myLabel += " or {0} or {0}$_1-${0}$_2$".format(axisName)
        lstyle = '--'
        loc = "center right" if locLegend == 'auto' else locLegend
        rHeight = 1.0
    ax2.set_xlabel(ax2.myLabel + ax2.myUnit)

    l1, = ax2.plot(axis, Ix, "C0"+lstyle)
    ax2.set_ylim([0, 1.02 if isIntensityNormalized else None])

    l3, = ax3.plot(axis*2, Jdx, "C1"+lstyle)
    if limJd is not None:
        rect = mpatches.Rectangle(
            (-limJd, 0), width=2*limJd, height=rHeight, color='C1', alpha=0.1)
        ax3.add_patch(rect)

    ax3.set_ylim([0, 1.02])

    pp = mpatches.Patch(color='none')  # a fake element to add text to legends
    legendX = ax3.legend(
        (l1, l3, pp),
        ("I({0}),\nσ$_{0}$ = {1:.2f} {2}".format(axisName, varIx**0.5, unit),
         "DoC({0}$_1-${0}$_2$),\nξ$_{0}$ = {1:.2f} {2}".format(
             axisName, varJdx**0.5, unit),
         "\nζ$_{0}$ = {1:.1f} %".format(axisName, cohFrx*100)),
        loc=loc)
    ax3.add_artist(legendX)

    return fig1, fig2


def calc_degree_of_transverse_coherence_4D(J):
    """
    Calculates DoTC from the mutual intensity *J* as Tr(*J²*)/Tr²(*J*). This
    function should only be used for demonstration purpose. There is a faster
    alternative: :func:`calc_degree_of_transverse_coherence_PCA`.
    """
    res = np.diag(np.dot(J, J)).sum() / np.diag(J).sum()**2
    return res.real


def calc_degree_of_transverse_coherence_PCA(U):
    """
    Calculates DoTC from the field stack *U*. The field images of *U* are
    flattened to form the matrix D shaped as (repeats, nx×ny).
    DoTC = Tr(*D*\ :sup:`+`\ *DD*\ :sup:`+`\ *D*)/Tr²(*D*\ :sup:`+`\ *D*),
    which is equal to the original DoTC for the mutual intensity *J*:
    DoTC = Tr(*J²*)/Tr²(*J*).
    """
    repeats, binsx, binsz = U.shape
    k = binsx * binsz
    D = np.array(U).reshape((repeats, k), order='F').T
    DTD = np.dot(D.T.conjugate(), D)
    res = np.diag(np.dot(DTD, DTD)).sum() / np.diag(DTD).sum()**2
    return res.real


def calc_eigen_modes_4D(J, eigenN=4):
    """
    Solves the eigenvalue problem for the mutual intensity *J*. This function
    should only be used for demonstration purpose. There is a much faster
    alternative: :func:`calc_eigen_modes_PCA`.
    """
    k = J.shape[0]
    J /= np.diag(J).sum()
    kwargs = dict(eigvals=(k-eigenN, k-1)) if eigenN else {}
    w, v = spl.eigh(J, **kwargs)
    if eigenN is None:
        rE = np.dot(np.dot(v, np.diag(w)), np.conj(v.T))
        print("diff J--decomposed(J) = {0}".format(np.abs(J-rE).sum()))
    return w, v


def calc_eigen_modes_PCA(U, eigenN=4, maxRepeats=None, normalize=False):
    """
    Solves the PCA problem for the field stack *U* shaped as (repeats, nx, ny).
    The field images are flattened to form the matrix D shaped as
    (repeats, nx×ny). The eigenvalue problem is solved for the matrix
    *D*\ :sup:`+`\*D*.

    Returns a tuple of two arrays: eigenvalues in a 1D array and eigenvectors
    as columns of a 2D array. This is a much faster and exact replacement of
    the full eigen mode decomposition by :func:`calc_eigen_modes_4D`.

    *eigenN* sets the number of returned eigen modes. If None, the number of
    modes is inferred from the shape of the field stack *U* and is equal to the
    number of macroelectrons (repeats).

    if *maxRepeats* are given, the stack is sliced up to that number. This
    option is introduced in order not to make the eigenvalue problem too big.

    If *normalize* is True, the eigenvectors are normalized, otherwise
    they are the PCs of the field stack in the original field units.
    """
    locU = U if maxRepeats is None else U[:maxRepeats, :, :]
    repeats, binsx, binsz = locU.shape
    if eigenN is None:
        eigenN = repeats
    if repeats < eigenN:
        raise ValueError('"repeats" must be >= {0}'.format(eigenN))
    k = binsx * binsz
    D = np.array(locU).reshape((repeats, k), order='F').T
    DTD = np.dot(D.T.conjugate(), D)
    DTD /= np.diag(DTD).sum()
    kwargs = dict(eigvals=(repeats-eigenN, repeats-1))
    wPCA, vPCA = spl.eigh(DTD, **kwargs)
    outPCA = np.zeros((k, eigenN), dtype=np.complex128)
    for i in range(eigenN):
        mPCA = np.outer(vPCA[:, -1-i], vPCA[:, -1-i].T.conjugate())
        vv = np.dot(D, mPCA)[:, 0]
        if normalize:
            outPCA[:, -1-i] = vv / np.dot(vv, vv.conj())**0.5
        else:
            outPCA[:, -1-i] = vv
    if (eigenN is None) and normalize:
        rE = np.dot(np.dot(vPCA, np.diag(wPCA)), np.conj(vPCA.T))
        print("diff DTD--decomposed(DTD) = {0}".format(np.abs(DTD-rE).sum()))
    return wPCA, outPCA
calc_eigen_modes = calc_eigen_modes_PCA


def plot_eigen_modes(x, y, w, v, xlabel='', ylabel=''):
    """
    Provides 2D plots of the first 4 eigen modes in the given *x* and *y*
    coordinates. The eigen modes are specified by eigenvalues and eigenvectors
    *w* and *v* returned by :func:`calc_eigen_modes_PCA` or
    :func:`calc_eigen_modes_4D`.

    Plot example:

    .. imagezoom:: _images/_Modes-2D.*

    """
    limx, limz = [x.min(), x.max()], [y.min(), y.max()]
    isSliceX, isSliceZ = False, False
    if limx[0] == limx[1]:
        limx = [l*0.1 for l in limz]
        isSliceX = True
    if limz[0] == limz[1]:
        limz = [l*0.1 for l in limx]
        isSliceZ = True
    dlimx, dlimz = limx[1] - limx[0], limz[1] - limz[0]
    fxz = float(dlimz)/dlimx

    xMargin = 80
    yMargin = xMargin - 30
    space = 4
    dX = 256
    dY = dX * fxz
    dpi = 100

    cmap = cm.get_cmap('cubehelix')
#    cmap = None  # default to rc image.cmap value

    extent = limx + limz

    if w is None:
        return
#    norm = (abs(v[:, -4:])**2).sum(axis=0)
#    v[:, -4:] /= norm**0.5

    xFigSize = float(2*xMargin + space + 2*dX)
#    yFigSize = float(2*yMargin + space + 2*dY + 20/fxz)
    yFigSize = float(2*yMargin + space + 2*dY)
    figMs = plt.figure(figsize=(xFigSize/dpi, yFigSize/dpi), dpi=dpi)

    rect2d = [xMargin/xFigSize, (yMargin+dY+space)/yFigSize,
              dX/xFigSize, dY/yFigSize]
    ax0 = figMs.add_axes(rect2d, aspect=1)
    rect2d = [(xMargin+dX+space)/xFigSize, (yMargin+dY+space)/yFigSize,
              dX/xFigSize, dY/yFigSize]
    ax1 = figMs.add_axes(rect2d, aspect=1)
    rect2d = [xMargin/xFigSize, yMargin/yFigSize,
              dX/xFigSize, dY/yFigSize]
    ax2 = figMs.add_axes(rect2d, aspect=1)
    rect2d = [(xMargin+dX+space)/xFigSize, yMargin/yFigSize,
              dX/xFigSize, dY/yFigSize]
    ax3 = figMs.add_axes(rect2d, aspect=1)
    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')
        if isSliceX:
            ax.set_xticks([0])
        if isSliceZ:
            ax.set_yticks([0])

    for ax in [ax0, ax1]:
        ax.xaxis.tick_top()
    for ax in [ax1, ax3]:
        ax.yaxis.tick_right()
    modeName = 'mode'
    labels = ['0th (coherent) {0}: w={1:.3f}', '1st residual {0}: w={1:.3f}',
              '2nd residual {0}: w={1:.3f}', '3rd residual {0}: w={1:.3f}']
    for iax, (ax, lab) in enumerate(zip([ax0, ax1, ax2, ax3], labels)):
        assert v[:, -iax-1].shape[0] == len(y)*len(x)
        im = (v[:, -iax-1]).reshape(len(y), len(x))
        imA = (im.real**2 + im.imag**2).astype(float)
        ax.imshow(imA, extent=extent, cmap=cmap, interpolation='none')
        plt.text(
            0.5, 0.95, lab.format(modeName, w[-iax-1]),
            transform=ax.transAxes, ha='center', va='top', color='w', size=10)

    if xlabel:
        figMs.text((xMargin+dX+space/2.)/xFigSize, yMargin*0.3/yFigSize,
                   xlabel, ha='center', va='center')
    if ylabel:
        figMs.text(xMargin*0.3/xFigSize, (yMargin+dY+space/2.)/yFigSize,
                   ylabel, ha='center', va='center', rotation='vertical')

    return figMs

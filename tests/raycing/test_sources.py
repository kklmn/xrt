# -*- coding: utf-8 -*-
"""
The module provides visualization routines for displaying spatial and
energy distributions of synchrotron sources in 2D and 3D.

For heavy calculations, I recomend running calculations with `wantPickle = True`
and then adjusting plotting properties in the next runs.
"""

__author__ = "Konstantin Klementiev"
__date__ = "1 Oct 2026"

import time
import copy
import numpy as np
import pickle
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
# import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs

vmin = 1e-4

dpi = 96
xOrigin2d = 84  # all sizes are in pixels
yOrigin2d = 48
space2dto1d = 8
height1d = 100
xSpaceExtra = 20
ySpaceExtra = 28


def visualize2D(source, data, title, saveName=None, sign=1):
    def one_fig(data2D, ts, tChar, otherChar, sh, title1D):
        xFigSize = float(xOrigin2d + sh[0] + space2dto1d +
                         height1d + xSpaceExtra)
        yFigSize = float(yOrigin2d + sh[1] + space2dto1d +
                         height1d + ySpaceExtra)
        fig = plt.figure(figsize=(xFigSize/dpi, yFigSize/dpi), dpi=dpi*1.25)
        rect_2D = [xOrigin2d / xFigSize, yOrigin2d / yFigSize,
                   (sh[0]-1) / xFigSize, (sh[1]-1) / yFigSize]
        rect_1DE = copy.deepcopy(rect_2D)
        rect_1DE[1] = rect_2D[1] + rect_2D[3] + space2dto1d/yFigSize
        rect_1DE[3] = height1d / yFigSize
        rect_1Dx = copy.deepcopy(rect_2D)
        rect_1Dx[0] = rect_2D[0] + rect_2D[2] + space2dto1d/xFigSize
        rect_1Dx[2] = height1d / xFigSize

        extent = [source.eMin, source.eMax, ts[0], ts[-1]]
        ax2D = fig.add_axes(rect_2D)
        dataMax = data2D.max()
        data2D[data2D < vmin] = vmin
        ax2D.imshow(
            data2D.T, aspect='auto', cmap='hot', extent=extent,
            # interpolation='nearest', origin='lower', figure=fig,
            interpolation=None, origin='lower', figure=fig,
            norm=LogNorm(vmin=dataMax*vmin, vmax=dataMax))
        ax2D.set_xlabel('$E$ (eV)')
        ax2D.set_ylabel(r"${0}'$ (mrad)".format(tChar))

        ax1DE = fig.add_axes(rect_1DE, sharex=ax2D)
        if title1D is not None:
            ax1DE.set_ylabel(title1D)
        ax1Dt = fig.add_axes(rect_1Dx, sharey=ax2D)
        plt.setp(ax1DE.get_xticklabels() + ax1Dt.get_yticklabels(),
                 visible=False)
        dE = energies[1] - energies[0]
        dt = ts[1] - ts[0]
        up = np.sum(data2D, axis=1)*dt
        ax1DE.plot(energies, up, 'r')
        # ax1DE.set_yscale('log')
        right = np.sum(data2D/energies[:, None]*dE, axis=0)
        ax1Dt.plot(right, ts, 'r')
        # ax1Dt.set_xscale('log')
        ax1DE.set_ylim(bottom=0)
        # ax1DE.set_yticks(np.array([0, 2, 4, 6, 8])*1e16)

        ax2D.set_xlim(extent[0], extent[1])
        ax2D.set_ylim(extent[2], extent[3])

        ax1DE.text(
            0.2, 1.0, r"Angular flux density {0} at ".format(title) +
            r"${0}'=0$".format(otherChar), transform=ax1DE.transAxes,
            size=12, color='r', ha='left', va='bottom')
        ax1DE.text(
            0.7, 0.95, "integrated over " + r"$d{0}'$".format(tChar),
            transform=ax1DE.transAxes, size=12, color='k', ha='center',
            va='top')
        ax1Dt.text(
            0.25, 0.5, "integrated\nover " + r"$dE/E$", rotation=-90,
            transform=ax1Dt.transAxes, size=12, color='k', ha='center',
            va='center')
        return fig, ax2D, ax1DE, ax1Dt

    if isinstance(source, rs.UndulatorUrgent):
        data = np.concatenate((data[:, :0:-1, :], data), axis=1)
        data = np.concatenate((data[:, :, :0:-1], sign*data), axis=2)
        xs = np.concatenate((-source.xs[:0:-1], source.xs))
        zs = np.concatenate((-source.zs[:0:-1], source.zs))
        energies = source.energies
    else:
        xs = np.mgrid[source.Theta_min:source.Theta_max + 0.5*source.dTheta:
                      source.dTheta] * 1e3  # from rad to mrad
        zs = np.mgrid[source.Psi_min:source.Psi_max + 0.5*source.dPsi:
                      source.dPsi] * 1e3  # from rad to mrad
        energies = np.mgrid[source.E_min:source.E_max + 0.5*source.dE:
                            source.dE]

    xSlice = (data.shape[1]-1) // 2
    zSlice = (data.shape[2]-1) // 2

    size = [350, 300]
    title1D = r"$dI_0/dz'$"+"\n(ph/s/mrad/0.1%bw)"
    figX, ax2EX, ax1EX, ax1XX = one_fig(data[:, :, zSlice], xs, 'x', 'z', size,
                                        title1D)
    size[1] = (size[1] * data.shape[2]) // data.shape[1]
    title1D = r"$dI_0/dx'$"+"\n(ph/s/mrad/0.1%bw)"
    figZ, ax2EZ, ax1EZ, ax1ZZ = one_fig(data[:, xSlice, :], zs, 'z', 'x', size,
                                        title1D)
    dE = energies[1] - energies[0]
    integralEvsX = np.sum(data[:, :, zSlice]/energies[:, None], axis=0)*dE
    integralEvsZ = np.sum(data[:, xSlice, :]/energies[:, None], axis=0)*dE

    maxIntegral = max(np.max(integralEvsX), np.max(integralEvsZ))
    ax1XX.set_xlim(maxIntegral*(sign-1)*0.55, maxIntegral*1.1)
    ax1ZZ.set_xlim(maxIntegral*(sign-1)*0.55, maxIntegral*1.1)

    if saveName is not None:
        fName = "{0}_{1}'E-" + source.prefix_save_name() + ".png"
        figX.savefig(fName.format(saveName, 'x'))
        figZ.savefig(fName.format(saveName, 'z'))


def imshow3d(ax, exz, colors, cut, norm=None, zorder=1):
    e, x, z = exz
    ce, cx, cz = cut
    icut = next(i for (i, v) in enumerate(cut) if isinstance(v, int))
    if icut == 0:
        x2, z2 = np.meshgrid(x[cx], z[cz], indexing='ij')
        e2 = np.full_like(x2, e[ce])
    elif icut == 1:
        e2, z2 = np.meshgrid(e[ce], z[cz], indexing='ij')
        x2 = np.full_like(e2, x[cx])
    elif icut == 2:
        e2, x2 = np.meshgrid(e[ce], x[cx], indexing='ij')
        z2 = np.full_like(e2, z[cz])
    else:
        raise ValueError("Invalid data cut")
    ax.plot_surface(x2, e2, z2, rstride=1, cstride=1, facecolors=colors,
                    shade=False, zorder=zorder)


def make_fig_3d(cdata, nexz, exz, cexz, wantZplane=True):
    e, x, z = exz
    ce, cx, cz = cexz
    ne, nx, nz = nexz

    fig = plt.figure(num=1, clear=True, figsize=(12, 8))  # "num" and "clear"!
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set(xlabel="x", ylabel="e", zlabel="z")
    ax.set_box_aspect((nx, 1.25*nx, nz))
    ax.set_axis_off()

    de = e[1] - e[0]
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    if wantZplane:
        cut = slice(ce, None), slice(None, nx//2+1), cz
        colors = cdata[*cut]
        imshow3d(ax, exz, colors, cut, zorder=1)
        lx = x[0]-dx, x[0]-dx, 0
        le = e[ce], e[-1]+de, e[-1]+de
        lz = [z[cz]] * 3
        ax.plot(lx, le, lz, color='gray', lw=1, zorder=1.5)

    cut = slice(ce, None), cx, slice(None, nz//2+1)
    colors = cdata[*cut]
    imshow3d(ax, exz, colors, cut, zorder=2)
    lx = [0] * 3
    le = e[ce], e[-1]+de, e[-1]+de
    lz = z[0]-dz, z[0]-dz, 0
    ax.plot(lx, le, lz, color='gray', lw=1, zorder=2.5)

    cut = ce, slice(None), slice(None)
    colors = cdata[*cut]
    imshow3d(ax, exz, colors, cut, zorder=3)
    lx = x[0]-dx, x[0]-dx, x[-1]+dx, x[-1]+dx, x[0]-dx
    le = [e[ce]] * 5
    lz = z[0]-dz, z[-1]+dz, z[-1]+dz, z[0]-dz, z[0]-dz
    ax.plot(lx, le, lz, color='gray', lw=1, zorder=3.5)

    if wantZplane:
        cut = slice(None, ce+1), slice(None, nx//2+1), cz
        colors = cdata[*cut]
        imshow3d(ax, exz, colors, cut, zorder=4)
        if ce > 0:
            lx = x[0]-dx, x[0]-dx, 0
            le = e[ce], e[0]-de, e[0]-de
            lz = [z[cz]] * 3
            ax.plot(lx, le, lz, color='gray', lw=1, zorder=4.5)
    cut = slice(None, ce+1), cx, slice(None, nz//2+1)
    colors = cdata[*cut]
    imshow3d(ax, exz, colors, cut, zorder=5)
    if ce > 0:
        lx = [0] * 3
        le = e[ce], e[0]-de, e[0]-de
        lz = z[0]-dz, z[0]-dz, 0
        ax.plot(lx, le, lz, color='gray', lw=1, zorder=5.5)

    ax.text(-0.01, e[ce], z[0]-0.01, f'{e[ce]:.0f} eV', zdir='x',
            color='gray', fontsize=14, ha='left', va='top', zorder=100)

    fig.tight_layout()
    return fig


def visualize3D(source, data, isZplane=True, saveName=None):
    if isinstance(source, rs.UndulatorUrgent):
        data = np.concatenate((data[:, :0:-1, :], data), axis=1)
        data = np.concatenate((data[:, :, :0:-1], data), axis=2)
        xs = np.concatenate((-source.xs[:0:-1], source.xs))
        zs = np.concatenate((-source.zs[:0:-1], source.zs))
        es = source.energies
    else:
        xs = np.mgrid[source.Theta_min:source.Theta_max + 0.5*source.dTheta:
                      source.dTheta] * 1e3  # from rad to mrad
        zs = np.mgrid[source.Psi_min:source.Psi_max + 0.5*source.dPsi:
                      source.dPsi] * 1e3  # from rad to mrad
        es = np.mgrid[source.E_min:source.E_max + 0.5*source.dE: source.dE]

    cx = (len(xs)-1) // 2
    cz = (len(zs)-1) // 2

    wantDark = False
    if wantDark:
        plt.style.use('dark_background')

    cmapName = 'hot'
    if True:  # want logarithmic colors
        dataMax = np.max(data)
        norm = LogNorm(vmin=dataMax*vmin, vmax=dataMax)  # accepts only 1D input
        cdata = plt.get_cmap(cmapName)(norm(data.flatten())).reshape(
            list(data.shape)+[4])  # rgba
    else:  # want linear colors
        norm = Normalize()
        data[data < vmin] = vmin
        cdata = plt.get_cmap(cmapName)(norm(data))

    # prepare pictures for the docs:
    # a = np.arange(15, 50)*100
    # b = np.arange(232, 250, 2)*10
    # c = np.arange(472, 490, 2)*10
    # wanted = sorted(np.unique(np.concatenate((a, b, c, [4990]))))

    # for ce, e in zip([len(es)//2], [es[len(es)//2]]):  # just a central cut
    for ce, e in enumerate(es):
        # if e not in wanted:
        #     continue
        fig = make_fig_3d(cdata, data.shape, (es, xs, zs), (ce, cx, cz),
                          wantZplane=True)
        fname = f"{source.prefix_save_name()}_{saveName}_{e:.0f}.png"
        fig.savefig(fname)
        print(fname)


def test_synchrotron_source(SourceClass, **kwargs):
    t0 = time.time()

    source = SourceClass(**kwargs)

    wantPickle = False  # for long calculations like srw, remove after use
    pickleName = f'tmp-{source.prefix_save_name()}.pickle'
    if wantPickle and os.path.isfile(pickleName):
        with open(pickleName, 'rb') as f:
            # I0, l1, l2, l3, grid = pickle.load(f)[:5]
            I0, grid = pickle.load(f)[:2]
            (source.Theta_min, source.Theta_max, source.dTheta,
             source.Psi_min, source.Psi_max, source.dPsi,
             source.E_min, source.E_max, source.dE) = grid
    else:
        es = np.linspace(kwargs['eMin'], kwargs['eMax'], kwargs['eN']+1)
        for ie, ee in enumerate(es):
            ti = time.time()
            print(f"E = {ee:.1f} eV, {ie+1} of {len(es)} in {ti-t0:.1f} s")
            I0t, l1t, l2t, l3t = source.intensities_on_mesh(
                energy=[ee], eSpreadNSamples=11)
            if ie == 0:
                I0 = I0t
                # I0, l1, l2, l3 = I0t, l1t, l2t, l3t
            else:
                I0 = np.concatenate([I0, I0t], axis=0)
                # l1 = np.concatenate([l1, l1t], axis=0)
                # l2 = np.concatenate([l2, l2t], axis=0)
                # l3 = np.concatenate([l3, l3t], axis=0)

    te = time.time()
    print('calculations took {0:.1f} s'.format(te - t0))

    if wantPickle and not os.path.isfile(pickleName):
        with open(pickleName, 'wb') as f:
            grid = [source.Theta_min, source.Theta_max, source.dTheta,
                    source.Psi_min, source.Psi_max, source.dPsi,
                    source.E_min, source.E_max, source.dE]
            # pickle.dump((I0, l1, l2, l3, grid, tstop-tstart), f, protocol=4)
            pickle.dump((I0, grid, te-t0), f, protocol=4)

    if 'xrt' in source.prefix_save_name():
        I0 *= 1e-6  # from /sr to /mrad²

    visualize2D(source, I0, r"$dI_0/dx'dz'$", 'I0')
    # visualize2D(source, I0*(1+l1)/2., r"$dI_{\sigma\sigma}/dx'dz'$", 'Is')
    # visualize2D(source, I0*(1-l1)/2., r"$dI_{\pi\pi}/dx'dz'$", Ip')
    # visualize2D(source, I0*l2/2., r"$\Re{dI_{\sigma\pi}/dx'dz'}$", IspRe')
    # sign = -1
    # if hasattr(source, 'Kx'):
    #     if source.Kx > 0:
    #         sign = 1
    # visualize2D(source, I0*l3/2., r'$\Im{I_{\sigma\pi}}$', 'IspIm', sign=sign)

    # select only one visualize3D at a time:
    # visualize3D(source, I0, isZplane=True, saveName='Itot')
    # visualize3D(source, I0*(1+l1)/2., isZplane=False, saveName='IsPol')
    # visualize3D(source, I0*(1-l1)/2., isZplane=False, saveName='IpPol')
    # visualize3D(source, I0*l2/2., saveName='IspRe')
    # visualize3D(source, I0*l3/2., saveName='IspIm')


def run_test(what):
    if what.lower().startswith('bm'):  # bending magnet
        kwargs = dict(
            B0=1.7, eE=3., xPrimeMax=2.5, zPrimeMax=0.3,
            eMin=1500, eMax=16500, eN=3000, nx=20, nz=20)

        if 'legacy' in what:  # by ws
            Source = rs.BendingMagnetWS
        else:  # by xrt:
            Source = rs.BendingMagnet
            kwargs['distE'] = 'BW'

    elif what.lower().startswith('w'):  # wiggler
        kwargs = dict(
            period=80., K=13., n=12, eE=3.,
            xPrimeMax=2.5, zPrimeMax=0.3, eMin=1500, eMax=16500,
            eN=3000, nx=20, nz=20)

        if 'legacy' in what:  # by ws
            Source = rs.WigglerWS
        else:  # by xrt:
            Source = rs.Wiggler
            kwargs['distE'] = 'BW'

    elif what.lower().startswith('u'):  # undulator
        # kwargs = dict(
        #     period=31.4, K=2.7, n=63, eE=6.08,
        #     xPrimeMax=0.3, zPrimeMax=0.3,
        #     eMin=500, eMax=16500, eN=1000, nx=20, nz=20)

        # Kmax = 1.92
        # thetaMax, psiMax = 100e-6, 50e-6
        # kwargs = dict(
        #     name='IVU18.5', eE=3.0, eI=0.5,
        #     eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
        #     period=18.5, n=108, K=Kmax,
        #     eMin=1500, eMax=16500, eN=1000, nx=40, nz=4,
        #     xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3)

        kwargs = dict(  # creates pictures for the docs, rather heavy! pickle it
            period=31.4, K=2.7, n=63, eE=6.08, eI=0.5,
            xPrimeMax=0.25, zPrimeMax=0.15,
            eSigmaX=134.2, eSigmaZ=6.325, eEpsilonX=1., eEpsilonZ=0.01,
            eMin=1500, eMax=5000-10, eN=350-1, nx=25*16, nz=15*16)

        if 'legacy' in what:  # by urgent
            Source = rs.UndulatorUrgent
            kwargs['icalc'] = 3  # 0 emittance
        elif 'srw' in what.lower():  # untested in xrt 2.0.0
            import srw.xrtSRW as xrtSRW
            Source = xrtSRW.UndulatorSRW
            kwargs['R0'] = 50000
            # 974 s - single electron
            # 65501 s - zero spread
            # 66180 s -nonzero spread
            # 0 emittance:
            kwargs['eSigmaX'] = 0
            kwargs['eSigmaZ'] = 0
            kwargs['eEpsilonX'] = 0
            kwargs['eEpsilonZ'] = 0
            # kwargs['eEspread'] = 1e-3
            kwargs['harmonicStart'] = 1
            kwargs['harmonicFin'] = 4
        else:  # by xrt:
            Source = rs.Undulator
            # kwargs['R0'] = 50000
            # kwargs['eSigmaX'] = 0
            # kwargs['eSigmaZ'] = 0
            # kwargs['eEpsilonX'] = 0
            # kwargs['eEpsilonZ'] = 0
            kwargs['eEspread'] = 1e-3  # increases calculation time!
            kwargs['distE'] = 'BW'
            kwargs['xPrimeMaxAutoReduce'] = False
            kwargs['zPrimeMaxAutoReduce'] = False
            # kwargs['targetOpenCL'] = "CPU"
            # kwargs['filamentBeam'] = True

    elif what.lower().startswith('e'):  # elliptical undulator
        kwargs = dict(
            period=31.4, Ky=2.7, Kx=2.7, n=63, eE=6.08,
            xPrimeMax=0.3, zPrimeMax=0.3,
            eMin=1000, eMax=4500, eN=350, nx=50, nz=50)

        if 'legacy' in what:  # by urgent
            Source = rs.UndulatorUrgent
            kwargs['icalc'] = 3  # 0 emittance
        else:  # by xrt:
            Source = rs.Undulator
            kwargs['phaseDeg'] = 90
            kwargs['distE'] = 'BW'
            kwargs['xPrimeMaxAutoReduce'] = False
            kwargs['zPrimeMaxAutoReduce'] = False
            # 0 emittance:
            kwargs['eSigmaX'] = 0
            kwargs['eSigmaZ'] = 0
            kwargs['eEpsilonX'] = 0
            kwargs['eEpsilonZ'] = 0

    test_synchrotron_source(Source, **kwargs)


if __name__ == '__main__':
    """ select a test """

    # run_test('BM legacy')
    # run_test('BM')

    # run_test('wiggler legacy')
    # run_test('wiggler')

    # run_test('undulator legacy')
    run_test('undulator')

    # run_test('elliptical undulator legacy')
    # run_test('elliptical undulator')

    plt.show()

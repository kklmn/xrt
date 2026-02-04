# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "1 Oct 2015"

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def plot_NOM_2D(fname):
    xL, yL, zL = np.loadtxt(fname, unpack=True)
    nX = (yL == yL[0]).sum()
    nY = (xL == xL[0]).sum()
    x = xL[:nX]
    y = yL[::nX]
    print(nX, nY)
    z = zL.reshape((nY, nX))

    zmax = abs(z).max()
    print(z.shape)
#    print(zmax)

    fig = plt.figure(figsize=(16, 8))
    rect_2D = [0.1, 0.1, 0.72, 0.6]
    rect_1Dx = [0.1, 0.72, 0.72, 0.26]
    rect_1Dy = [0.83, 0.1, 0.13, 0.6]
    extent = [x[0], x[-1], y[0], y[-1]]
    ax2D = plt.axes(rect_2D)
    ax2D.set_xlabel('x (mm)')
    ax2D.set_ylabel('y (mm)')
    ax2D.imshow(
        z, aspect='auto', cmap='jet', extent=extent,
        # interpolation='nearest',
        interpolation='none',
        origin='lower', figure=fig)

    ax1Dx = plt.axes(rect_1Dx, sharex=ax2D)
    ax1Dy = plt.axes(rect_1Dy, sharey=ax2D)
    ax1Dx.set_ylabel('h (nm)')
    ax1Dy.set_xlabel('h (nm)')
    plt.setp(ax1Dx.get_xticklabels() + ax1Dy.get_yticklabels(),
             visible=False)
#    ax1Dx.plot(x, x*0, 'gray')
    kl, = ax1Dx.plot(x, z.sum(axis=0)/nY, 'k')
    ax1Dx.plot(x, z[0, :], 'r')
    ax1Dx.plot(x, z[nY//2, :], 'g')
    ax1Dx.plot(x, z[nY-1, :], 'b')
    ax1Dx.legend([kl], ['average over y'], loc='upper left', frameon=False)

#    ax1Dy.plot(y*0, y, 'gray')
    ax1Dy.plot(z.sum(axis=1)/nX, y, 'k')
    ax1Dy.plot(z[:, 0], y, 'y')
    ax1Dy.plot(z[:, nX//2], y, 'c')
    ax1Dy.plot(z[:, nX-1], y, 'm')

    ax2D.set_xlim(extent[0], extent[1])
    ax2D.set_ylim(extent[2], extent[3])
    ax1Dx.set_ylim(-zmax, zmax)
    ax1Dy.set_xlim(-zmax, zmax)

    ax2D.annotate('', (0, 0), (-0.03, 0), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='r', ec='r', headwidth=10,
                                  headlength=0.4))
    ax2D.annotate('', (0, 0.5), (-0.03, 0.5), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='g', ec='g', headwidth=10,
                                  headlength=0.4))
    ax2D.annotate('', (0, 1), (-0.03, 1), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='b', ec='b', headwidth=10,
                                  headlength=0.4))

    ax2D.annotate('', (0, 0), (0, -0.06), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='y', ec='y', headwidth=10,
                                  headlength=0.4))
    ax2D.annotate('', (0.5, 0), (0.5, -0.06), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='c', ec='c', headwidth=10,
                                  headlength=0.4))
    ax2D.annotate('', (1, 0), (1, -0.06), size=10,
                  xycoords="axes fraction",
                  arrowprops=dict(alpha=1, fc='m', ec='m', headwidth=10,
                                  headlength=0.4))

    b, a = np.gradient(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    a /= dx
    b /= dy
    rmsA = ((a**2).sum() / (nX * nY))**0.5
    rmsB = ((b**2).sum() / (nX * nY))**0.5
    aveZ = z.sum() / (nX * nY)
    fig.text(0.91, 0.95,
             u'rms slope errors:\ndz/dx = {0:.2f} µrad\n'
             u'dz/dy = {1:.2f} µrad\n\n'
             u'mean figure error:\n<z> = {2:.2f} pm\n'
             .format(rmsA, rmsB, aveZ*1e3),
             transform=fig.transFigure, size=12, color='r', ha='center',
             va='top')

# this is the way how the surface is used in ray-tracing/wave-propagation:
# the hight and the directions are spline-interpolated (the spline coefficients
# are pre-calculated) and then the spline piece-wise polinomials are used to
# reconstruct the height and the two directions and arbitrary (x, y) points.
    splineZ = ndimage.spline_filter(z.T)
    splineA = ndimage.spline_filter(a.T)
    splineB = ndimage.spline_filter(b.T)

    nrays = 1000
    xnew = np.random.uniform(x[0], x[-1], nrays)
    ynew = np.random.uniform(y[0], y[-1], nrays)
    coords = np.array([(xnew-x[0]) / (x[-1]-x[0]) * (nX-1),
                       (ynew-y[0]) / (y[-1]-y[0]) * (nY-1)])

    znew = ndimage.map_coordinates(splineZ, coords, prefilter=True)
    anew = ndimage.map_coordinates(splineA, coords, prefilter=True)
    bnew = ndimage.map_coordinates(splineB, coords, prefilter=True)

    ax2D.scatter(xnew, ynew, c=znew, marker='o', s=50, cmap='jet')
    ax2D.quiver(xnew, ynew, -anew, -bnew, edgecolor='gray', color='gray',
                # headaxislength=5,
                scale=200, lw=0.2)

    fig.savefig(fname+'.png')
    plt.show()


def plot_NOM_3D(fname):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    xL, yL, zL = np.loadtxt(fname, unpack=True)
    nX = (yL == yL[0]).sum()
    nY = (xL == xL[0]).sum()
    x = xL.reshape((nY, nX))
    y = yL.reshape((nY, nX))
    z = zL.reshape((nY, nX))
    x1D = xL[:nX]
    y1D = yL[::nX]
#    z += z[::-1, :]
    zmax = abs(z).max()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.5)
    ax.set_zlim(-zmax, zmax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    splineZ = ndimage.spline_filter(z.T)
    nrays = 1000
    xnew = np.random.uniform(x1D[0], x1D[-1], nrays)
    ynew = np.random.uniform(y1D[0], y1D[-1], nrays)
    coords = np.array([(xnew-x1D[0]) / (x1D[-1]-x1D[0]) * (nX-1),
                       (ynew-y1D[0]) / (y1D[-1]-y1D[0]) * (nY-1)])
    znew = ndimage.map_coordinates(splineZ, coords, prefilter=True)
    ax.scatter(xnew, ynew, znew, c=znew, marker='o', s=50, cmap=cm.coolwarm)

    fig.savefig(fname+'_3d.png')
    plt.show()


if __name__ == '__main__':
    fname = '../../examples/withRaycing/13_Warping/mock_surface.dat'
    plot_NOM_2D(fname)
    # plot_NOM_3D(fname)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:21:26 2016

@author: rchernik
"""

import numpy as np
#import matplotlib as mpl
#mpl.use('webAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
from xrt.backends.raycing.physconsts import SIM0, SIE0, SIC, E2W, K2B, PI

compare2reference = False
suffix = '_average'

# Integration grid (per period)
nRKPoins = 30

kwargs = dict(eE=1.5, period=84., n=36)
eEpsilonX = 6.0e-9
eEpsilonZ = 0.06e-9
betaX = 5.66
betaZ = 2.85
sigmaX = (eEpsilonX * betaX)**0.5
sigmaXp = (eEpsilonX / betaX)**0.5
print(u'sigmaX={} µm'.format(sigmaX*1e6))
print(u'sigmaXp={} mrad'.format(sigmaXp*1e6))
sigmaZ = (eEpsilonZ * betaZ)**0.5
sigmaZp = (eEpsilonZ / betaZ)**0.5
print(u'sigmaZ={} µm'.format(sigmaZ*1e6))
print(u'sigmaZp={} mrad'.format(sigmaZp*1e6))

#sheet, prefix = 'EPU_HP_mode', '1'
#sheet, prefix = 'QEPU_HP_mode', '2'
#sheet, prefix = 'EPU_VP_mode', '3'
sheet, prefix = 'QEPU_VP_mode', '4'
customField = ['B-Hamed.xlsx', dict(sheetname=sheet, skiprows=0)]

#kwargs = dict(eE=3, period=18.5*3, n=108/3, targetE=[9000, 7])
#customField = 20.

eE = kwargs['eE']  # [GeV]
gamma = eE * 1e9 * SIE0 / (SIM0 * SIC**2)

# Undulator parameters
period = kwargs['period']
Lu = period/1000.  # [m]

periods = kwargs['n']
targetE = kwargs.get('targetE', None)
if targetE is not None:
    w = targetE[0]  # (eV)
    K = np.sqrt(targetE[1] * 8 * PI * SIC * 1e3 * gamma**2 /
                period / targetE[0] / E2W - 2)
    print("K = {0}".format(K))
    Ky = K
    Kx = 0
    if np.isnan(K):
        raise ValueError("Cannot calculate K, try to increase the "
                         "undulator harmonic number")
    if len(targetE) > 2:
        isElliptical = targetE[2]
        if isElliptical:
            Kx = Ky = K / 2**0.5
            print("Kx = Ky = {0}".format(Kx))
else:
    w = 10200  # (eV)
    Kx = kwargs.get('Kx', 0.)
    Ky = kwargs.get('Ky', 0.)
    K = kwargs.get('K', 10.)
    if Kx == 0 and Ky == 0:
        Ky = K
B0x = K2B * Kx / period
B0y = K2B * Ky / period
phase = np.pi*0.5

# derivative parameters
ECML = Lu * SIE0 / SIM0 / SIC / 2. / np.pi
EG = ECML / gamma

print(u"Xmax = {0} µm".format(Ky/gamma*Lu/2/np.pi*1e6))
print(u"Ymax = {0} µm".format(Kx/gamma*Lu/2/np.pi*1e6))

betam = 1. - (1. + 0.5 * Kx**2 + 0.5*Ky**2) / 2. / gamma**2
wu = np.pi * SIC / Lu / gamma**2 * \
                (2*gamma**2 - 1 - 0.5*Kx**2 - 0.5*Ky**2) / E2W
tgw = 1. / wu
tgwmm = 2 * np.pi / Lu / 1e3

zgrid, zstep = np.linspace(-np.pi*periods+0.5*np.pi,
                           np.pi*periods+0.5*np.pi,
                           nRKPoins*periods, retstep=True)


def read_custom_field(fname, kwargs={}):
    if fname.endswith('.xls') or fname.endswith('.xlsx'):
        import pandas
        data = pandas.read_excel(fname, **kwargs).values
    else:
        data = np.loadtxt(fname)
    return data


def By(z, customFieldData):
    if isinstance(customFieldData, (float, int)):
        return B0y*np.sin(z)
    else:
        return np.interp(z,
                         customFieldData[:, 0] / Lu * 2 * np.pi * 0.001,
                         customFieldData[:, 2])


def Bx(z, customFieldData):
    if isinstance(customFieldData, (float, int)):
        return B0x*np.sin(z + phase)
    else:
        return np.interp(z,
                         customFieldData[:, 0] / Lu * 2 * np.pi * 0.001,
                         customFieldData[:, 1])


def Bz(z, customFieldData):
    if isinstance(customFieldData, (float, int)):
        return customFieldData*np.ones_like(z)
    else:
        if customFieldData.shape[1] > 3:
            return np.interp(z,
                             customFieldData[:, 0] / Lu * 2 * np.pi * 0.001,
                             customFieldData[:, 3])
        else:
            return np.zeros_like(z)


def f1(u, v, z, B0z):  # v', x'' gives beta_x
    return EG * (-By(z, B0z) + u * Bz(z, B0z))


def f2(u, v, z, B0z):  # u', y'' gives beta_y
    return EG * (-v * Bz(z, B0z) + Bx(z, B0z))


def fz(u, v, z, B0z):  # beta_z', z'' gives beta_z
    return EG * (-v * By(z, B0z) + u * Bx(z, B0z))


def f3(u, v, z, B0z):  # v, x' gives r_x
    return v


def f4(u, v, z, B0z):  # u, y' gives r_y
    return u


def f5(u, v, z, B0z):
    return np.sqrt(1. - 1./gamma**2 - u**2 - v**2)


def iterate_rk():
    if customField is not None:
        if isinstance(customField, (tuple, list)):
            fname = customField[0]
            kwargs = customField[1]
        elif isinstance(customField, (float, int)):
            fname = None
            customFieldData = customField
        else:
            fname = customField
            kwargs = {}
        if fname:
            customFieldData = read_custom_field(fname, kwargs)
        tB0z = customFieldData
        # tB0z = customFieldData[:, 0] / Lu * 2 * np.pi
    else:
        raise ValueError("no custom field!")

    for B0z in np.linspace(0, 0, 1):

        ugrid = [0]
        vgrid = [0]
        xgrid = [0]
        ygrid = [0]
        ztgrid = [0]
        dzgrid = [0]
        # betazgrid = [0]
        u = 0
        v = 0
        z = 0
        x = 0
        y = 0
        zt = 0
        dz = 0
        h = zstep
        betaXav = 0
        betaYav = 0
        betaZav = 0

        for iz, z in enumerate(zgrid):
            k1f1 = h * f1(u, v, z, tB0z)
            k1f2 = h * f2(u, v, z, tB0z)
            # k1fz = h * fz(u, v, z, tB0z)

            k2f1 = h * f1(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f2 = h * f2(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            # k2fz = h * fz(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)

            k3f1 = h * f1(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f2 = h * f2(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            # k3fz = h * fz(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)

            k4f1 = h * f1(u + k3f2, v + k3f1, z + h, tB0z)
            k4f2 = h * f2(u + k3f2, v + k3f1, z + h, tB0z)
            # k4fz = h * f5(u + k3f2, v + k3f1, z + h, tB0z)

            v += (k1f1 + 2.*k2f1 + 2.*k3f1 + k4f1)/6.
            u += (k1f2 + 2.*k2f2 + 2.*k3f2 + k4f2)/6.
            # dz += (k1f2 + 2.*k2f2 + 2.*k3f2 + k4f2)/6.

            ugrid.append(u)
            vgrid.append(v)
            # dzgrid.append(dz)
            betaXav += v*h
            betaYav += u*h
            # betaZav += dz*h

        betaXav /= (zgrid[-1] - zgrid[0])
        betaYav /= (zgrid[-1] - zgrid[0])
        # betaZav /= (zgrid[-1] - zgrid[0])

        betaXMin = min(vgrid[:-1])
        betaXMax = max(vgrid[:-1])

        betaYMin = min(ugrid[:-1])
        betaYMax = max(ugrid[:-1])

        # betaZMin = min(dzgrid[:-1])
        # betaZMax = max(dzgrid[:-1])

        if suffix == '_average':
            betaX0 = -betaXav
            betaY0 = -betaYav
            # betaZ0 = -betaZav
        else:
            betaX0 = -0.5*(betaXMin+betaXMax)
            betaY0 = -0.5*(betaYMin+betaYMax)
            # betaZ0 = -0.5*(betaZMin+betaZMax)

        ugrid = [betaY0]
        vgrid = [betaX0]
        # dzgrid = [betaZ0]
        u = betaY0
        v = betaX0
        # dz = betaZ0

        xAv = 0
        yAv = 0
        zAv = 0

        for iz, z in enumerate(zgrid):
            k1f1 = h * f1(u, v, z, tB0z)
            k1f2 = h * f2(u, v, z, tB0z)
            k1f3 = h * f3(u, v, z, tB0z)
            k1f4 = h * f4(u, v, z, tB0z)
            k1f5 = h * f5(u, v, z, tB0z)
            # k1fz = h * fz(u, v, z, tB0z)

            k2f1 = h * f1(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f2 = h * f2(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f3 = h * f3(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f4 = h * f4(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f5 = h * f5(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            # k2fz = h * fz(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)

            k3f1 = h * f1(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f2 = h * f2(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f3 = h * f3(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f4 = h * f4(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f5 = h * f5(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            # k3fz = h * fz(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)

            k4f1 = h * f1(u + k3f2, v + k3f1, z + h, tB0z)
            k4f2 = h * f2(u + k3f2, v + k3f1, z + h, tB0z)
            k4f3 = h * f3(u + k3f2, v + k3f1, z + h, tB0z)
            k4f4 = h * f4(u + k3f2, v + k3f1, z + h, tB0z)
            k4f5 = h * f5(u + k3f2, v + k3f1, z + h, tB0z)
            # k4fz = h * f5(u + k3f2, v + k3f1, z + h, tB0z)

            v += (k1f1 + 2.*k2f1 + 2.*k3f1 + k4f1)/6.
            u += (k1f2 + 2.*k2f2 + 2.*k3f2 + k4f2)/6.
            x += (k1f3 + 2.*k2f3 + 2.*k3f3 + k4f3)/6.
            y += (k1f4 + 2.*k2f4 + 2.*k3f4 + k4f4)/6.
            zt += (k1f5 + 2.*k2f5 + 2.*k3f5 + k4f5)/6.
            # dz += (k1fz + 2.*k2fz + 2.*k3fz + k4fz)/6.

            ugrid.append(u)
            vgrid.append(v)
            xgrid.append(x)
            ygrid.append(y)
            ztgrid.append(zt)
            dzgrid.append(dz)

            xAv += x*h
            yAv += y*h
            zAv += zt*h

        xAv /= (zgrid[-1] - zgrid[0])
        yAv /= (zgrid[-1] - zgrid[0])
        zAv /= (zgrid[-1] - zgrid[0])

        xMin = min(xgrid[:-1])
        xMax = max(xgrid[:-1])

        yMin = min(ygrid[:-1])
        yMax = max(ygrid[:-1])

        ztMin = min(ztgrid[:-1])
        ztMax = max(ztgrid[:-1])

        if suffix == '_average':
            x0 = -xAv
            y0 = -yAv
            zt0 = -zAv
        else:
            x0 = -0.5*(xMin+xMax)
            y0 = -0.5*(yMin+yMax)
            zt0 = -0.5*(ztMin+ztMax)

        # dz0 = -0.5*(dzMin+dzMax)

        ugrid = [betaY0]
        vgrid = [betaX0]
        # dzgrid = [betaZ0]
        # betazgrid = np.sqrt(1 - 1/gamma**2 - betaX0**2 - betaY0**2)
        u = betaY0
        v = betaX0
        # dz = betaZ0
        xgrid = [x0]
        ygrid = [y0]
        ztgrid = [zt0]
        x = x0
        y = y0
        zt = zt0
        betaZav = 0
        betaZpath = 0

        for iz, z in enumerate(zgrid):
            k1f1 = h * f1(u, v, z, tB0z)
            k1f2 = h * f2(u, v, z, tB0z)
            k1f3 = h * f3(u, v, z, tB0z)
            k1f4 = h * f4(u, v, z, tB0z)
            k1f5 = h * f5(u, v, z, tB0z)
            k1fz = h * fz(u, v, z, tB0z)

            k2f1 = h * f1(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f2 = h * f2(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f3 = h * f3(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f4 = h * f4(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2f5 = h * f5(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)
            k2fz = h * fz(u + 0.5*k1f2, v + 0.5*k1f1, z + 0.5*h, tB0z)

            k3f1 = h * f1(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f2 = h * f2(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f3 = h * f3(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f4 = h * f4(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3f5 = h * f5(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)
            k3fz = h * fz(u + 0.5*k2f2, v + 0.5*k2f1, z + 0.5*h, tB0z)

            k4f1 = h * f1(u + k3f2, v + k3f1, z + h, tB0z)
            k4f2 = h * f2(u + k3f2, v + k3f1, z + h, tB0z)
            k4f3 = h * f3(u + k3f2, v + k3f1, z + h, tB0z)
            k4f4 = h * f4(u + k3f2, v + k3f1, z + h, tB0z)
            k4f5 = h * f5(u + k3f2, v + k3f1, z + h, tB0z)
            k4fz = h * f5(u + k3f2, v + k3f1, z + h, tB0z)

            v += (k1f1 + 2.*k2f1 + 2.*k3f1 + k4f1)/6.
            u += (k1f2 + 2.*k2f2 + 2.*k3f2 + k4f2)/6.
            betaZav += h * np.sqrt(1. - 1./gamma**2 - u**2 - v**2)
            betaZpath += h
            x += (k1f3 + 2.*k2f3 + 2.*k3f3 + k4f3)/6.
            y += (k1f4 + 2.*k2f4 + 2.*k3f4 + k4f4)/6.
            zt += (k1f5 + 2.*k2f5 + 2.*k3f5 + k4f5)/6.
            dz += (k1fz + 2.*k2fz + 2.*k3fz + k4fz)/6.

            ugrid.append(u)
            vgrid.append(v)
            xgrid.append(x)
            ygrid.append(y)
            ztgrid.append(zt)
            dzgrid.append(dz)

        betaZav /= betaZpath
        print "betaZav", betaZav, ", betam", betam
        wuAv = 2. * np.pi * SIC * betaZav / Lu / E2W
        print "wuAv", wuAv, ", wu", wu

        # test of the integration: taking arbitrary direction
        norm_ref = np.array(
                [1.e-5*np.ones_like(zgrid[:-1]),
                 1.e-5*np.ones_like(zgrid[:-1]),
                 (1. - 1.e-10)*np.ones_like(zgrid[:-1])])
        # concatenating the integrated coordinates
        r_int = np.array([xgrid[:-2], ygrid[:-2], ztgrid[:-2]])
        phase_int = w/wuAv*(zgrid[:-1] - (norm_ref[0]*r_int[0] +
                            norm_ref[1]*r_int[1] + norm_ref[2]*r_int[2]))
        if compare2reference:
            # comparison to the reference plane undulator
            beta_ref = np.array(
                [Ky / gamma * np.cos(zgrid[:-1]),
                 -Kx / gamma * np.cos(zgrid[:-1] + phase)])
            r_ref = np.array(
                    [Ky / wu / gamma * np.sin(zgrid[:-1]),
                     -Kx / wu / gamma * np.sin(zgrid[:-1] + phase),
                     betam * zgrid[:-1] / wu - 0.125 / wu / gamma**2 *
                     (Ky**2 * np.sin(2*zgrid[:-1]) +
                      Kx**2 * np.sin(2*(zgrid[:-1] + phase)))])
            phase_ref = w*(zgrid[:-1]/wu - (norm_ref[0]*r_ref[0] +
                           norm_ref[1]*r_ref[1] + norm_ref[2]*r_ref[2]))

        # print "dz", dzgrid[0], dzgrid[-1]
        # print "zt", ztgrid[0], ztgrid[-1]

        z = np.array(zgrid) * Lu / 2. / np.pi * 1e3
        plt.figure(1)
        plt.plot(z, Bx(zgrid, tB0z))
        plt.title(r"B$_x$")
        plt.xlabel(r"z, mm")
        plt.ylabel(r"B$_x$, T")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("Bx"+suffix+'.png')

        plt.figure(2)
        plt.plot(z, By(zgrid, tB0z))
        plt.title(r"B$_y$")
        plt.xlabel(r"z, mm")
        plt.ylabel(r"B$_y$, T")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("By"+suffix+'.png')

        plt.figure(3)
        plt.plot(z, Bz(zgrid, tB0z))
        plt.title(r"B$_z$")
        plt.xlabel(r"z, mm")
        plt.ylabel(r"B$_z$, T")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("Bz"+suffix+'.png')

        plt.figure(4)
        plt.plot(z, vgrid[:-1])
        if compare2reference:
            plt.plot(z[:-1], beta_ref[0])
        plt.title(r"$\beta_x$")
        plt.xlabel(r"z, mm")
        plt.ylabel(r"$\beta_x$, c")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("beta_x"+suffix+'.png')

        plt.figure(5)
        plt.plot(z, ugrid[:-1])
        if compare2reference:
            plt.plot(z[:-1], beta_ref[1])
        plt.title(r"$\beta_y$")
        plt.xlabel(r"z, mm")
        plt.ylabel(r"$\beta_y$, c")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("beta_y"+suffix+'.png')

        # plt.figure(17)
        # plt.plot(z,
        #         np.sqrt(1. - 1/gamma*2 - np.array(ugrid[:-1])**2 -
        #             np.array(vgrid[:-1])**2))
        # plt.plot(np.array(zgrid) * Lu / 2. / np.pi * 1e3, dzgrid[:-1])
        # if compare2reference:
        #   plt.plot(zgrid[:-1], beta_ref[1])
        # plt.title("beta_z")
        # plt.gca().set_xlim(z[0], z[-1])
        # plt.savefig("beta_z"+suffix+'.png')

        plt.figure(6)
        plt.plot(z, np.array(xgrid[:-1]) / tgwmm * 1e3)
        if compare2reference:
            plt.plot(z, np.array(r_ref[0]) / tgwmm * wu * 1e3)
        plt.title("Trajectory, $x$ plane")
        plt.xlabel(r"z, mm")
        plt.ylabel(u"x, µm")
        plt.gca().set_xlim(z[0], z[-1])
        plt.gca().set_ylim(-60, 60)
        plt.savefig("x"+suffix+'.png')

        plt.figure(7)
        plt.plot(z, np.array(ygrid[:-1]) / tgwmm * 1e3)
        if compare2reference:
            plt.plot(z[:-1], np.array(r_ref[1]) / tgwmm * wu * 1e3)
        plt.title("Trajectory, $y$ plane")
        plt.xlabel(r"z, mm")
        plt.ylabel(u"y, µm")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("y"+suffix+'.png')

        plt.figure(8)
        plt.plot(z, np.array(ztgrid[:-1]) / tgwmm)
        if compare2reference:
            plt.plot(z[:-1], np.array(r_ref[2]) / tgwmm * wu)
        plt.title("z")
        plt.gca().set_xlim(z[0], z[-1])
        plt.savefig("z"+suffix+'.png')

        fig = plt.figure(9)
        ax = fig.gca(projection='3d')
        ax.plot(np.array(ztgrid[:-1]) / tgwmm / 1e3,
                np.array(xgrid[:-1]) / tgwmm * 1e3,
                np.array(ygrid[:-1]) / tgwmm * 1e3,
                label='parametric curve')
        ax.set_xlabel(r"z, m")
        ax.set_ylabel(u"x, µm")
        ax.set_zlabel(u"y, µm")
        plt.title("trajectory 3D")
        plt.savefig("trajectory"+suffix+'.png')

#        plt.figure(10)
#        plt.plot(zgrid[:-1], phase_int)
#        if compare2reference:
#            plt.plot(zgrid[:-1], phase_ref)
#        plt.title("phase (wt(t'))")
#        plt.savefig("phase (wt(t'))"+suffix+'.png')

        if compare2reference:
            plt.figure(11)
            plt.plot(z[:-1], vgrid[:-2] - beta_ref[0])
            plt.title(r"$\beta_x$ $_{integrated}$ - $\beta_x$ $_{direct}$")
            plt.xlabel(r"z, mm")
            plt.ylabel(u"$\Delta\beta_x$, µm")
            plt.savefig("beta_x_calc - beta_x_ref"+suffix+'.png')

            plt.figure(12)
            plt.plot(z[:-1], ugrid[:-2] - beta_ref[1])
            plt.title(r"$\beta_y$ $_{integrated}$ - $\beta_y$ $_{direct}$")
            plt.xlabel(r"z, mm")
            plt.ylabel(u"$\Delta\beta_y$, µ$m")
            plt.savefig("beta_y_calc - beta_y_ref"+suffix+'.png')

            plt.figure(13)
            plt.plot(z[:-1],
                     (np.array(xgrid[:-2])-np.array(r_ref[0])*wu)/tgwmm * 1e3)
            plt.title(r"x$_{integrated}$ - x$_{direct}$")
            plt.xlabel(r"z, mm")
            plt.ylabel(u"$\Delta$x, µm")
            plt.savefig("x_calc - x_ref"+suffix+'.png')

            plt.figure(14)
            plt.plot(z[:-1],
                     (np.array(ygrid[:-2])-np.array(r_ref[1])*wu)/tgwmm * 1e3)
            plt.title(r"y$_{integrated}$ - y$_{direct}$")
            plt.xlabel(r"z, mm")
            plt.ylabel(u"$\Delta$y, µm")
            plt.savefig("y_calc - y_ref"+suffix+'.png')

            plt.figure(15)
            plt.plot(z[:-1],
                     (np.array(ztgrid[:-2])-np.array(r_ref[2])*wu)/tgwmm)
            plt.title(r"z$_{integrated}$ - z$_{direct}$")
            plt.savefig("z_calc - z_ref"+suffix+'.png')

#            plt.figure(16)
#            plt.plot(z[:-1],
#                     phase_int - phase_ref)
#            plt.title("phase$_{integrated}$ - phase$_{direct}$")
#            plt.savefig("phase_calc - phase_ref"+suffix+'.png')
    plt.show()


iterate_rk()

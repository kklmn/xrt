# -*- coding: utf-8 -*-
"""
The module provides visualization routines for displaying spatial and
energy distributions of synchrotron sources in 2D and 3D."""

__author__ = "Konstantin Klementiev"
__date__ = "12 Mar 2014"

#import cmath
import time
import copy
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore
#import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs

vmin = 2.5e-3

dpi = 82
xOrigin2d = 84  # all sizes are in pixels
yOrigin2d = 48
space2dto1d = 8
height1d = 100
xSpaceExtra = 20
ySpaceExtra = 28


def visualize(source, data, title, saveName=None, sign=1):
    def one_fig(what, ts, tChar, otherChar):
        sh = what.shape
        xFigSize = float(xOrigin2d + sh[0] + space2dto1d +
                         height1d + xSpaceExtra)
        yFigSize = float(yOrigin2d + sh[1] + space2dto1d +
                         height1d + ySpaceExtra)
        fig = plt.figure(figsize=(xFigSize/dpi, yFigSize/dpi), dpi=dpi)
        rect_2D = [xOrigin2d / xFigSize, yOrigin2d / yFigSize,
                   (sh[0]-1) / xFigSize, (sh[1]-1) / yFigSize]
        rect_1DE = copy.deepcopy(rect_2D)
        rect_1DE[1] = rect_2D[1] + rect_2D[3] + space2dto1d/yFigSize
        rect_1DE[3] = height1d / yFigSize
        rect_1Dx = copy.deepcopy(rect_2D)
        rect_1Dx[0] = rect_2D[0] + rect_2D[2] + space2dto1d/xFigSize
        rect_1Dx[2] = height1d / xFigSize

        extent = [source.eMin, source.eMax, ts[0], ts[-1]]
        ax2D = plt.axes(rect_2D)
        dataMax = what.max()
        ax2D.imshow(
            what.T, aspect='auto', cmap='hot', extent=extent,
#            interpolation='nearest', origin='lower', figure=fig,
            interpolation=None, origin='lower', figure=fig,
            norm=LogNorm(vmin=dataMax*vmin, vmax=dataMax))
        ax2D.set_xlabel('energy (eV)')
        ax2D.set_ylabel(r"${0}'$ (mrad)".format(tChar))

        ax1DE = plt.axes(rect_1DE, sharex=ax2D)
        ax1Dt = plt.axes(rect_1Dx, sharey=ax2D)
        plt.setp(ax1DE.get_xticklabels() + ax1Dt.get_yticklabels(),
                 visible=False)
        dE = source.energies[1] - source.energies[0]
        dt = ts[1] - ts[0]
        ax1DE.plot(
            source.energies, np.sum(what, axis=1)*dt, 'r')
#            ax1DE.set_yscale('log')
        ax1Dt.plot(np.sum(what/source.energies[:, None]*dE, axis=0), ts, 'r')
#            ax1Dt.set_xscale('log')
        ax1DE.set_ylim(bottom=0)
        ax1DE.set_yticks(np.array([0, 2, 4, 6, 8])*1e16)

        ax2D.set_xlim(extent[0], extent[1])
        ax2D.set_ylim(extent[2], extent[3])

        ax1DE.text(
            0.65, 1.0, r"Angular flux density {0} at ".format(title) +
            r"${0}'=0$".format(otherChar) +
            r" (ph/s/mrad$^2$/0.1%bw)", transform=ax1DE.transAxes,
            size=12, color='r', ha='center', va='bottom')
        ax1DE.text(
            0.7, 0.95, "integrated\nover " + r"$d{0}'$".format(tChar),
            transform=ax1DE.transAxes, size=12, color='k', ha='center',
            va='top')
        ax1Dt.text(
            0.25, 0.5, "integrated\nover " + r"$d$ln(energy)", rotation=-90,
            transform=ax1Dt.transAxes, size=12, color='k', ha='center',
            va='center')
        return fig, ax2D, ax1DE, ax1Dt

    xs = source.xs * 1e3  # from rad to mrad
    zs = source.zs * 1e3  # from rad to mrad
    xSlice = 0
    zSlice = 0
    if 'xrt' in source.prefix_save_name() or\
            'srw' in source.prefix_save_name():
        pass
    else:
        data = np.concatenate((data[:, :0:-1, :], data), axis=1)
        data = np.concatenate((data[:, :, :0:-1], sign*data), axis=2)
        xs = np.concatenate((-xs[:0:-1], xs), axis=1)
        zs = np.concatenate((-zs[:0:-1], zs), axis=1)
    xSlice = (data.shape[1]-1) / 2
    zSlice = (data.shape[2]-1) / 2

    figX, ax2EX, ax1EX, ax1XX = one_fig(data[:, :, zSlice], xs, 'x', 'z')
    figZ, ax2EZ, ax1EZ, ax1ZZ = one_fig(data[:, xSlice, :], zs, 'z', 'x')
    dE = source.energies[1] - source.energies[0]
    integralEvsX = \
        np.sum(data[:, :, zSlice]/source.energies[:, None], axis=0)*dE
    integralEvsZ = \
        np.sum(data[:, xSlice, :]/source.energies[:, None], axis=0)*dE

    maxIntegral = max(np.max(integralEvsX), np.max(integralEvsZ))
    ax1XX.set_xlim(maxIntegral*(sign-1)*0.5, maxIntegral)
    ax1ZZ.set_xlim(maxIntegral*(sign-1)*0.5, maxIntegral)

    if saveName is not None:
        fName = "{0}_{1}'E-" + source.prefix_save_name() + ".png"
        figX.savefig(fName.format(saveName, 'x'))
        figZ.savefig(fName.format(saveName, 'z'))


def visualize3D(source, data, isZplane=True, saveName=None):
# Enthought library imports
    from mayavi.scripts import mayavi2
    from mayavi.sources.array_source import ArraySource
#        from mayavi.modules.outline import Outline
#        from mayavi.modules.volume import Volume
    from mayavi.modules.text3d import Text3D
    from mayavi.modules.image_plane_widget import ImagePlaneWidget
    from mayavi.tools.camera import view
    from mayavi.tools.animator import animate

    @mayavi2.standalone
    def view_data(data):
        """Example showing how to view a 3D numpy array in mayavi2.
        """
        def set_labelE(ind):
            return '{0:.0f} eV'.format(source.energies[ind])

        def move_view(obj, evt):
            labelE.text = set_labelE(ipwX.ipw.slice_index)
            pos = labelE.position
            labelE.position = ipwX.ipw.slice_index * src.spacing[0] + 1,\
                pos[1], pos[2]
            labelE.vector_text.update()

        def set_lut(ipw):
            lutM = ipw.module_manager.scalar_lut_manager
#                lutM.show_scalar_bar = True
#                lutM.number_of_labels = 9
            lutM.lut.scale = 'log10'
            lutM.lut.range = [dataMax*vmin, dataMax]
            lutM.lut_mode = 'hot'

        @animate()
        def anim(data, ipwX):
            scene.scene.off_screen_rendering = True
            scene.scene.anti_aliasing_frames = 0
            for i in range(0, data.shape[0], 1):
                ipwX.ipw.slice_index = i
                move_view(None, None)
                if saveName is not None:
                    scene.scene.save('{0}{1:04d}.png'.format(saveName, i))
                yield

        # 'mayavi' is always defined on the interpreter.
        scene = mayavi.new_scene()  # analysis:ignore
        scene.scene.background = (0, 0, 0)
        print(source.prefix_save_name())

        src = ArraySource(transpose_input_array=True)
        sh = data.shape
#        print(sh)
        if 'xrt' in source.prefix_save_name() or\
                'srw' in source.prefix_save_name():
            src.scalar_data = data[:, :sh[1]//2+1, :sh[2]//2+1].copy()
        else:
            src.scalar_data = data[:, ::-1, ::-1].copy()
#        src.spacing = np.array([-0.05, 1, 1])
#        src.spacing = np.array([-0.25, 1, 1])
        src.spacing = np.array([-0.25, 0.25, 0.25])
        mayavi.add_source(src)  # analysis:ignore
        # Visualize the data.
#            o = Outline()
#            mayavi.add_module(o)

        ipwY = ImagePlaneWidget()
        mayavi.add_module(ipwY)  # analysis:ignore
        ipwY.ipw.plane_orientation = 'y_axes'  # our x-axis
        ipwY.ipw.slice_index = int(data.shape[1] - 1)
#        if 'xrt' in source.prefix_save_name():
#            ipwY.ipw.slice_index /= int(2)
        ipwY.ipw.left_button_action = 0
        set_lut(ipwY)

        if isZplane:
            ipwZ = ImagePlaneWidget()
            mayavi.add_module(ipwZ)  # analysis:ignore
            ipwZ.ipw.plane_orientation = 'z_axes'  # our z-axis
            ipwZ.ipw.slice_index = int(data.shape[2] - 1)
#            if 'xrt' in source.prefix_save_name():
#                ipwZ.ipw.slice_index /= int(2)
            ipwZ.ipw.left_button_action = 0

        if 'xrt' in source.prefix_save_name() or\
                'srw' in source.prefix_save_name():
            pass
        else:
            data = np.concatenate((data[:, :0:-1, :], data), axis=1)
            data = np.concatenate((data[:, :, :0:-1], data), axis=2)
        sh = data.shape
        print(sh)
        src = ArraySource(transpose_input_array=True)
        src.scalar_data = data.copy()
#        src.spacing = np.array([-0.05, 1, 1])
#        src.spacing = np.array([-0.25, 1, 1])
        src.spacing = np.array([-0.25, 0.25, 0.25])
        mayavi.add_source(src)  # analysis:ignore

        ipwX = ImagePlaneWidget()
        mayavi.add_module(ipwX)  # analysis:ignore
        ipwX.ipw.plane_orientation = 'x_axes'  # energy
        set_lut(ipwX)
        ipwX.ipw.add_observer('WindowLevelEvent', move_view)
        ipwX.ipw.add_observer('StartInteractionEvent', move_view)
        ipwX.ipw.add_observer('EndInteractionEvent', move_view)

        labelE = Text3D()
        mayavi.add_module(labelE)  # analysis:ignore
        labelE.position = (1, data.shape[1]*0.73*src.spacing[1],
                           data.shape[2]*0.85*src.spacing[2])
        labelE.orientation = 90, 0, 90
        labelE.text = 'Energy'
        labelE.scale = 3, 3, 1
        labelE.actor.property.color = 0, 1, 1
        labelE.orient_to_camera = False
        labelE.text = set_labelE(0)

        view(45, 70, 200)
        wantToAnimate = True
        if wantToAnimate:
            anim(data, ipwX)
        else:
            ipwX.ipw.slice_index = data.shape[0]-1
            move_view(None, None)

    data[data < 1e-7] = 1e-7
    dataMax = np.max(data)
    view_data(data)


def test_synchrotron_source(SourceClass, **kwargs):
    tstart = time.time()

    source = SourceClass(**kwargs)

#    if source.prefix_save_name().startswith('srw'):
#        import pickle
#        pickleName = 'srw-und-non0em.pickle'
#        with open(pickleName, 'rb') as f:
#            I0, l1, l2, l3 = pickle.load(f)[0:4]

    print('started')
    I0, l1, l2, l3 = source.intensities_on_mesh()
    I0 *= 1e-6  # from /sr to /mrad²
    print('finished')
    tstop = time.time()
    print('calculations took {0:.1f} s'.format(tstop - tstart))

##for long calculations like srw:
#    if source.prefix_save_name().startswith('srw'):
#        import pickle
#        pickleName = source.prefix_save_name()+'.pickle'
#        with open(pickleName, 'wb') as f:
#            pickle.dump((I0, l1, l2, l3, tstop-tstart), f, protocol=2)

##visualize in 2D:
    visualize(source, I0, r'$I_0$', 'I0')
#    visualize(source, I0*(1+l1)/2., r'$I_{\sigma\sigma}$', 'Is')
#    visualize(source, I0*(1-l1)/2., r'$I_{\pi\pi}$', 'Ip')
#    visualize(source, I0*l2/2., r'$\Re{I_{\sigma\pi}}$', 'IspRe')
#    sign = -1
#    if hasattr(source, 'Kx'):
#        if source.Kx > 0:
#            sign = 1
#    visualize(source, I0*l3/2., r'$\Im{I_{\sigma\pi}}$', 'IspIm', sign=sign)

##select only one visualize3D at a time:
#    visualize3D(source, I0, isZplane=False, saveName='Itot')
#    visualize3D(source, I0*(1+l1)/2., isZplane=False, saveName='IsPol')
#    visualize3D(source, I0*(1-l1)/2., isZplane=False, saveName='IpPol')
#    visualize3D(source, I0*l2/2., saveName='IspRe')
#    visualize3D(source, I0*l3/2., saveName='IspIm')
#
if __name__ == '__main__':
    """Uncomment the block you want to test."""

##*********** Bending Magnet ***************
#    kwargs = dict(B0=1.7, eE=3., xPrimeMax=2.5, zPrimeMax=0.3,
#                  eMin=1500, eMax=31500, eN=3000, nx=1, nz=10)
###by WS:
##    Source = rs.BendingMagnetWS
##by xrt:
#    kwargs['distE'] = 'BW'
#    Source = rs.BendingMagnet

##*********** Wiggler ***************
#    kwargs = dict(period=80., K=13., n=12, eE=3., xPrimeMax=2.5,
#                  zPrimeMax=0.3, eMin=1500, eMax=31500, eN=3000, nx=20, nz=20)
##by WS:
#    Source = rs.WigglerWS
##by xrt:
#    kwargs['distE'] = 'BW'
#    Source = rs.Wiggler

#*********** undulator ***************
#    kwargs = dict(
#        period=31.4, K=2.7, n=63, eE=6.08,
#        xPrimeMax=0.3, zPrimeMax=0.3,
#        eMin=500, eMax=31500, eN=1000, nx=20, nz=20)
#    Kmax = 1.92
#    thetaMax, psiMax = 100e-6, 50e-6
#    kwargs = dict(name='IVU18.5', eE=3.0, eI=0.5,
#                  eEpsilonX=0.263, eEpsilonZ=0.008, betaX=9., betaZ=2.,
#                  period=18.5, n=108, K=Kmax,
#                  eMin=1500, eMax=31500, eN=1000, nx=40, nz=4,
#                  xPrimeMax=thetaMax*1e3, zPrimeMax=psiMax*1e3, distE='BW')
    kwargs = dict(
        period=31.4, K=2.7, n=63, eE=6.08, eI=0.5, xPrimeMax=0.3, zPrimeMax=0.15,
        eSigmaX=134.2, eSigmaZ=6.325, eEpsilonX=1., eEpsilonZ=0.01,
        eMin=1500, eMax=5000, eN=350, nx=40*4, nz=20*4)
##by Urgent:
#    kwargs['icalc'] = 3  # 0 emittance
#    Source = rs.UndulatorUrgent
###by SRW:
#    import srw.xrtSRW as xrtSRW
#    kwargs['R0'] = 50000
## 974 s - single electron
## 65501 s - zero spread
## 66180 s -nonzero spread
#    kwargs['eSigmaX'] = 0
#    kwargs['eSigmaZ'] = 0
#    kwargs['eEpsilonX'] = 0
#    kwargs['eEpsilonZ'] = 0
##    kwargs['eEspread'] = 1e-3
#    kwargs['harmonicStart'] = 1
#    kwargs['harmonicFin'] = 4
#    Source = xrtSRW.UndulatorSRW
#by xrt:
    kwargs['R0'] = 50000
#    kwargs['eSigmaX'] = 0
#    kwargs['eSigmaZ'] = 0
#    kwargs['eEpsilonX'] = 0
#    kwargs['eEpsilonZ'] = 0
    kwargs['eEspread'] = 1e-3*0
    kwargs['distE'] = 'BW'
    kwargs['xPrimeMaxAutoReduce'] = False
    kwargs['zPrimeMaxAutoReduce'] = False
#    kwargs['targetOpenCL'] = "CPU"
    kwargs['filamentBeam'] = True
    Source = rs.Undulator

##*** helical undulator **************
#    kwargs = dict(
#        period=31.4, Ky=2.7, Kx=2.7, n=63, eE=6.08,
#        xPrimeMax=0.3, zPrimeMax=0.3,
#        eMin=500, eMax=10500, eN=1000, nx=20, nz=20)
###by Urgent:
##    Source = rs.UndulatorUrgent
##by xrt:
#    kwargs['phaseDeg'] = 90
#    kwargs['distE'] = 'BW'
#    kwargs['xPrimeMaxAutoReduce'] = False
#    kwargs['zPrimeMaxAutoReduce'] = False
#    Source = rs.Undulator

    test_synchrotron_source(Source, **kwargs)

    plt.show()

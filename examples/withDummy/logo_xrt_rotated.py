# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "18 Sep 2026"

import sys
import os, sys; sys.path.append(os.path.join('..', '..'))  # analysis:ignore

import numpy as np
import scipy.ndimage as sndi
import matplotlib as mpl
# mpl.style.use('classic')
# mpl.use('agg')
import matplotlib.pyplot as plt

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.dummy as dummy
import copy


def main():
    logo = plt.imread('logo-python.png')  # load 2D template

    inishape = logo.shape
    logo = sndi.rotate(logo, 45)
    rotshape = logo.shape
    logo = logo[(rotshape[0]-inishape[0])//2:, (rotshape[1]-inishape[1])//2:]
    logo = logo[:inishape[0], :inishape[1]]
    dy, dx = logo.shape[:2]
    print(logo.shape)

    logo_mono = logo[:, :, 0] + logo[:, :, 1] + logo[:, :, 2]*2
    logo_inty = copy.deepcopy(logo_mono)
    logo_blue = copy.deepcopy(logo_mono)
    logo_blue[logo[:, :, 2] < 0.4] = 0
    logo_yellow = copy.deepcopy(logo_mono)
    logo_yellow[logo[:, :, 0] < 0.4] = 0
    xrtp.height1d = 80
    xrtp.heightE1d = 80
    xrtp.xspace1dtoE1d = 4
    xrtp.heightE1dbar = 12
    xrtp.xOrigin2d = 4
    xrtp.yOrigin2d = 4
    xrtp.xSpaceExtra = 6
    xrtp.ySpaceExtra = -80

    # make "ray-tracing" arrays: x, y, intensity and cData
    locNrays = dy * dx
    YY, XX = np.mgrid[0:dx, 0:dy]
    x = XX.flatten()
    y = dy - YY.flatten()
    ymax = y.max()
    logo_inty += np.where(ymax - YY > ymax*0.4, ymax - YY, ymax*0.4)
    intensity = logo_inty.flatten()
    cData = x * np.log(abs(y)+1.5)
    # cData = x*y
    cDatamax = np.max(cData)

    blue_area = logo_blue.T.flatten() > 0.1
    cData[blue_area] = cDatamax*0.5 + (0.5*y[blue_area]-dy*0.5)**2*0.1
    yellow_area = logo_yellow.T.flatten() > 0.1
    cData[yellow_area] = cDatamax*0.14 + (2.5*y[yellow_area]-dy*0.5)**2*0.01
    intensity[~blue_area & ~yellow_area] = 0.

    def local_output():
        return x, y, intensity, cData, locNrays
    dummy.run_process = local_output  # invoked by pyXRayTrcaer to get rays

    plot1 = xrtp.XYCPlot(
        'dummy',
        xaxis=xrtp.XYCAxis('', '', fwhmFormatStr=None, bins=dx,
                           ppb=1, limits=[0.5, dx+0.5]),
        yaxis=xrtp.XYCAxis('', '', fwhmFormatStr=None, bins=dy,
                           ppb=1, limits=[0.5, dy+0.5]),
        caxis=xrtp.XYCAxis('', '', fwhmFormatStr=None, bins=dy//2, ppb=2,
                           limits=[10, cDatamax*0.8], outline=1),
        # negative=True, invertColorMap=True, xPos=0,
        # saveName=['logo-xrt.png', 'logo-xrt.pdf'],
        negative=False, invertColorMap=False, xPos=0,
        saveName=['logo-xrt-inv.png', 'logo-xrt-inv.pdf'],
        aspect='auto')
    fontProp = mpl.font_manager.FontProperties(
        fname=r'C:\Windows\Fonts\timesbd.ttf', weight=960, size=100)
    # xpos, ypos = 0.28, 0.58
    xpos, ypos = 0.26, 0.04
    plot1.textPanelX = plot1.fig.text(
        xpos, ypos, 'x', transform=plot1.fig.transFigure, color='r',
        ha='center', fontproperties=fontProp)
    # xpos, ypos = 0.58, 0.58
    xpos, ypos = 0.58, 0.04
    plot1.textPanelR = plot1.fig.text(
        xpos, ypos, 'r', transform=plot1.fig.transFigure, color='r',
        ha='center', fontproperties=fontProp)
    # xpos, ypos = 0.82, 0.58
    xpos, ypos = 0.84, 0.04
    plot1.textPanelT = plot1.fig.text(
        xpos, ypos, 't', transform=plot1.fig.transFigure, color='r',
        ha='center', fontproperties=fontProp)

    # with no labels:
    plot1.textNrays = None
    plot1.textGoodrays = None
    plot1.textI = None
    # ... and no tick labels:
    plt.setp(
        plot1.ax1dHistEbar.get_yticklabels() +
        plot1.ax2dHist.get_xticklabels() + plot1.ax2dHist.get_yticklabels(),
        visible=False)
    # ... and no ticks:
    allAxes = [plot1.ax1dHistX, plot1.ax1dHistY, plot1.ax2dHist,
               plot1.ax1dHistE, plot1.ax1dHistEbar]
    for ax in allAxes:
        for axXY in (ax.xaxis, ax.yaxis):
            plt.setp(axXY.get_ticklines(), visible=False)
    # end of no ticks

    xrtr.run_ray_tracing(plot1, repeats=2, backend='dummy')


# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()

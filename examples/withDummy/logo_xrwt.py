r"""
Logo
----

Uses :mod:`dummy` backend.
File: `\\examples\\withDummy\\logo_xrt.py`

:mod:`xrt` logo (on the right, in two versions) created from a "flat" python
logo (on the left).

 |ini_image|  --------------  |logo_image|     |logo_image_inv|

.. |ini_image| image:: _images/logo-python.*
   :scale: 50 %
   :align: bottom

.. |logo_image| image:: _images/logo-xrt.*
   :scale: 50 %
   :align: bottom

.. |logo_image_inv| image:: _images/logo-xrt-inv.*
   :scale: 50 %
   :align: bottom
"""
__author__ = "Konstantin Klementiev"
__date__ = "1 Mar 2012"

import sys
sys.path.append(r"c:\Ray-tracing")
#sys.path.append(r"/media/sf_Ray-tracing")
import numpy as np
#import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.dummy as dummy
import copy


def main():
    logo = plt.imread('logo-python.png')  # load 2D template
#    logo = plt.imread('logo_test0.png')  # load 2D template
    logo_mono = logo[:, :, 0] + logo[:, :, 1] + logo[:, :, 2]*2
    logo_blue = copy.deepcopy(logo_mono)
    logo_blue[logo[:, :, 2] < 0.4] = 0
    logo_yellow = copy.deepcopy(logo_mono)
    logo_yellow[logo[:, :, 0] < 0.4] = 0
    xrtp.height1d = 64
    xrtp.heightE1d = 64
    xrtp.xspace1dtoE1d = 4
    xrtp.xOrigin2d = 4
    xrtp.yOrigin2d = 4
    xrtp.xSpaceExtra = 6
    xrtp.ySpaceExtra = 4

#    make "ray-tracing" arrays: x, y, intensity and cData
    locNrays = logo.shape[0] * logo.shape[1]
    xy = np.mgrid[0:logo.shape[1], 0:logo.shape[0]]
    x = xy[0, ...].flatten()
    y = logo.shape[0] - xy[1, ...].flatten()
    print(logo.shape)
    intensity = logo_mono.T.flatten()
    cData = y * np.log(abs(x)+1.5)
    cDatamax = np.max(cData)

    def local_output():
        return x, y, intensity, cData, locNrays
    dummy.run_process = local_output  # invoked by pyXRayTrcaer to get the rays

    blue_area = logo_blue.T.flatten() > 0.1
    cData[blue_area] = cDatamax*0.5 + (y[blue_area] - logo.shape[0]*0.5)*1.2
    yellow_area = logo_yellow.T.flatten() > 0.1
    cData[yellow_area] = cDatamax*0.15 - \
        (y[yellow_area] - logo.shape[0]*0.5)*1.2
    intensity[~blue_area & ~yellow_area] = 0.

    plot1 = xrtp.XYCPlot(
        'dummy',
        saveName=['logo-xrwt.png', 'logo_xrwt.pdf'],
#        saveName=['logo-xrt-inv.png', 'logo_xrt-inv.pdf'],
        xaxis=xrtp.XYCAxis('', '', fwhmFormatStr=None, bins=logo.shape[1],
                           ppb=1, limits=[0.5, logo.shape[1]+0.5]),
        yaxis=xrtp.XYCAxis('', '', fwhmFormatStr=None, bins=logo.shape[0],
                           ppb=1, limits=[0.5, logo.shape[0]+0.5]),
        caxis=xrtp.XYCAxis(
            '', '', fwhmFormatStr=None, bins=logo.shape[0]/2, ppb=2,
            limits=[10, cDatamax*0.8]),
        negative=True, invertColorMap=True,
        aspect='auto')
    plot1.textPanel = plot1.fig.text(
        0.54, 0.77, 'xr', transform=plot1.fig.transFigure, size=64, color='r',
        ha='left', fontname='times new roman', weight=660)
    plot1.textPanel = plot1.fig.text(
        0.74, 0.73, 'w', transform=plot1.fig.transFigure, size=58, color='r',
        ha='left', fontname='times new roman', weight=660)
    plot1.textPanel = plot1.fig.text(
        0.9, 0.77, 't', transform=plot1.fig.transFigure, size=64, color='r',
        ha='left', fontname='times new roman', weight=660)
#     with no labels:
    plot1.textNrays = None
    plot1.textGoodrays = None
    plot1.textI = None
#    ... and no tick labels:
    plt.setp(
        plot1.ax1dHistEbar.get_yticklabels() +
        plot1.ax2dHist.get_xticklabels() + plot1.ax2dHist.get_yticklabels(),
        visible=False)
#    ... and no ticks:
    allAxes = [plot1.ax1dHistX, plot1.ax1dHistY, plot1.ax2dHist,
               plot1.ax1dHistE, plot1.ax1dHistEbar]
    for ax in allAxes:
        for axXY in (ax.xaxis, ax.yaxis):
            plt.setp(axXY.get_ticklines(), visible=False)
#     end of no ticks:

    xrtr.run_ray_tracing(plot1, repeats=2, backend='dummy')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()

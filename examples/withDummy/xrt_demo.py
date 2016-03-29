# -*- coding: utf-8 -*-
"""
Module :mod:`demo` represents the simplest example of xrt. Its main objective
is to demonstrate the output graphics, where the user can play with several
options given by the parameters of :class:`XYCPlot` constructors. The default
ray distributions provided by the 'dummy' backend have no physical meaning but
can be changed in module :mod:`dummy`.
"""
#import matplotlib as mpl
#mpl.use('wx')
import sys
sys.path.append(r"c:\Ray-tracing")

import xrt.plotter as plotter
import xrt.runner as runner


def demo():
    """The main body of demo."""
    plots = []

#create a plot...
    plot = plotter.XYCPlot(title='normal energy distribution')
#notice the energy offset and how it is displayed
    plot.caxis.offset = 5000
#    plot1.persistentName='01.pickle' #for saving and restoring
    plot.xaxis.limits = [-2.5, 2.5]
    plot.yaxis.limits = [-2.5, 2.5]
    filename = 'offset5000'
    plot.saveName = [filename + '.pdf', filename + '.png']
#an example of creating a label:
#the first 2 values are x and y in figure coordinates
    plot.textPanel = plot.fig.text(
        0.76, 0.8, '', transform=plot.fig.transFigure, size=12, color='r')
    plot.caxis.fwhmFormatStr = '%.1f'
    plots.append(plot)

#... and another plot. The 2nd plot is negative and of inverted colors (i.e.
#the energy color map is inverted back to that of plot1 after making the graph
#negative)
    plot = plotter.XYCPlot(
        invertColorMap=True, negative=True, title='normal energy distribution,\
         negative+{inverted colors}')
    plot.xaxis.limits = [-1, 1]
    plot.yaxis.limits = [-1, 1]
#an example of creating a label:
    plot.textPanel = plot.fig.text(
        0.76, 0.8, '', transform=plot.fig.transFigure, size=12, color='b')
    plot.xaxis.fwhmFormatStr = '%.1f'
    plots.append(plot)

#a dummy text:
    for plot in plots:
        plot.textPanel.set_text(
            'test label1 = {0}\ntest label2 = {1}'.format(0.1, r'$e^{i\pi}$'))

    runner.run_ray_tracing(
        plots, repeats=40, updateEvery=2, backend='dummy', processes='all')

if __name__ == '__main__':
    demo()

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:13:08 2024

@author: roman
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import xrt
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

import json


def deserialize_plots(data):
    plotsList = []
    plotDefArgs = dict(raycing.get_params("xrt.plotter.XYCPlot"))
    axDefArgs = dict(raycing.get_params("xrt.plotter.XYCAxis"))

    if not isinstance(data['Project']['plots'], dict):
        print("Plots are not defined")
        return []

    for plotName, plotProps in data['Project']['plots'].items():
        plotKwargs = {}

        for pname, pval in plotProps.items():
            if pname in ['_object']:
                continue
            if pname in ['xaxis', 'yaxis', 'caxis']:
                axKwargs = {}
                for axname, axval in pval.items():
                    if axname == '_object':
                        continue
                    if axname in axDefArgs and axval != str(axDefArgs[axname]):
                        axKwargs[axname] = raycing.parametrize(axval)
                plotKwargs[pname] = xrtplot.XYCAxis(**axKwargs)

            else:
                if pname in plotDefArgs and pval != str(plotDefArgs[pname]):
                    plotKwargs[pname] = raycing.parametrize(pval)
        try:
            newPlot = xrtplot.XYCPlot(**plotKwargs)
            plotsList.append(newPlot)
        except:
            print("Plot init failed")
    return plotsList


def get_xrt_run_args(data, beamLine, plots):
    runnerArgsStr = data['Project']['run_ray_tracing']
    runnerKWArgs = {}
    runnerKWArgs['plots'] = plots
    runnerKWArgs['beamLine'] = beamLine
    for arg in ['repeats', 'updateEvery', 'pickleEvery',
                'threads', 'processes']:
        if arg in runnerArgsStr:
            runnerKWArgs[arg] = raycing.get_init_val(runnerArgsStr[arg])
    return runnerKWArgs

rrun.run_process = raycing.run_process_from_file


def main():
    fileName = r"BioXAS_Main.xml"
#    fileName = r"BioXAS_Main.xml.json"  # Uncomment two lines below to convert
#    fileName = r"1crystal.xml"
#    fileName = r"4crystals.xml"  # Saves plots to PNG
#    fileName = r"lens1.xml"
#    fileName = r"lens3.xml"
#    fileName = r"testAlignment.xml"
    fileName = r"testGrating.xml"
    beamLine = raycing.BeamLine(fileName=fileName)

    plots = deserialize_plots(beamLine.layoutStr)
    runnerKwargs = get_xrt_run_args(beamLine.layoutStr, beamLine, plots)
    xrtrun.run_ray_tracing(**runnerKwargs)

#    with open(fileName+'.json', 'w') as json_file:  # Simple XML-to-JSON
#        json.dump(beamLine.layoutStr, json_file, indent=4)

if __name__ == "__main__":
    main()

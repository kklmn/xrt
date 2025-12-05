# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-11-21"

Created with xrtQook


None

"""

import numpy as np
import sys
sys.path.append(r"C:\GitHub\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials_elemental as rmatsel
import xrt.backends.raycing.materials_compounds as rmatsco
import xrt.backends.raycing.materials_crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
from xrt.backends.raycing.figure_error import RandomRoughness, GaussianBump, Waviness 

rghR = RandomRoughness(name='RRandom',limPhysX=[-10, 10], limPhysY=[-40, 40],
                      rms=100., corrLength=1, seed=20251201)

rghG = GaussianBump(name='RGauss', base=rghR, limPhysX=[-10, 10], limPhysY=[-40, 40],
                   bumpHeight=300., sigmaX=10, sigmaY=30)

rghW = Waviness(name='RWavy', base=rghG, limPhysX=[-10, 10], limPhysY=[-40, 40],
               amplitude=200., xWaveLength=40, yWaveLength=1)


def build_beamline():
    BeamLine = raycing.BeamLine(
        name=r"BeamLine",
        description=None)

    BeamLine.geometricSource01 = raycing.sources_geoms.GeometricSource(
        bl=BeamLine,
        name=r"geometricSource01",
        center=[0, 0, 0],
        dx=0.1,
        dxprime=0.001,
        dz=0.1,
        dzprime=0.001)

    BeamLine.geometricSource02 = raycing.sources_geoms.GeometricSource(
        bl=BeamLine,
        name=r"geometricSource02",
        center=[100, 0, 0],
        dx=0.1,
        dxprime=0.001,
        dz=0.1,
        dzprime=0.001)

    BeamLine.toroidMirror01 = roes.ToroidMirror(
        bl=BeamLine,
        name=r"toroidMirror01",
        center=[0.0, 100.0, 0.0],
        pitch=r"30deg",
        limPhysX=[-10.0, 10.0],
        limPhysY=[-40.0, 40.0],
        order=1,
        R=200,
        r=50)

    BeamLine.toroidMirrorSTL = roes.ToroidMirror(
        bl=BeamLine,
        name=r"toroidMirrorSTL",
        center=[100.0, 100.0, 0.0],
        pitch=r"30deg",
        limPhysX=[-10.0, 10.0],
        limPhysY=[-40.0, 40.0],
        order=1,
        roughness=rghW,
        R=200,
        r=50)

#    BeamLine.toroidMirrorSTL = roes.OEfrom3DModel(
#        bl=BeamLine,
#        name=r"toroidMirrorSTL",
#        center=[100, 100.0, 0.0],
#        pitch=r"30deg",
#        filename="mirror_cut_400x100.stl",
##        filename="toroidMirror01.stl",
#        orientation='XYZ',
#        )


#    print("lim X", BeamLine.toroidMirrorSTL.limPhysX,
#          "\nlim Y", BeamLine.toroidMirrorSTL.limPhysY)
#    nd = 100    
#    xl, yl = np.meshgrid(np.linspace(-2, 2, nd), np.linspace(-2, 2, nd))
#    zl = BeamLine.toroidMirrorSTL.local_z(xl.flatten(), yl.flatten())
#    print(zl)
    
#    from matplotlib import pyplot as plt
#    from matplotlib import cm
#    
#    plt.figure()
#    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#    surf = ax.plot_surface(xl, yl, zl.reshape(nd, nd), cmap=cm.jet,
#                       linewidth=0, antialiased=False)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#
#    plt.show()
    

    BeamLine.screen01 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen01",
        center=[0, 150, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    BeamLine.screen02 = rscreens.Screen(
        bl=BeamLine,
        name=r"screen02",
        center=[100, 150, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    return BeamLine


def run_process(BeamLine):
    geometricSource01_global = BeamLine.geometricSource01.shine()
    geometricSource02_global = BeamLine.geometricSource02.shine()

    toroidMirror01_global, toroidMirror01_local = BeamLine.toroidMirror01.reflect(
        beam=geometricSource01_global)
    toroidMirrorSTL_global, toroidMirrorSTL_local = BeamLine.toroidMirrorSTL.reflect(
        beam=geometricSource02_global)

    screen01_local = BeamLine.screen01.expose(
        beam=toroidMirror01_global)

    screen02_local = BeamLine.screen02.expose(
        beam=toroidMirrorSTL_global)

    outDict = {
        'geometricSource01_global': geometricSource01_global,
        'geometricSource02_global': geometricSource02_global,

        'toroidMirror01_global': toroidMirror01_global,
        'toroidMirror01_local': toroidMirror01_local,

        'toroidMirrorSTL_global': toroidMirrorSTL_global,
        'toroidMirrorSTL_local': toroidMirrorSTL_local,

        'screen01_local': screen01_local,
        'screen02_local': screen02_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"geometricSource01_global",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot01-geometricSource01_global-energy")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen01_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Screen 1 - Toroid")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen02_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Screen2 - STL")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"toroidMirror01_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"y profile - Toroid",
        aspect="auto")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"toroidMirrorSTL_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"y profile - STL",
        aspect="auto")
    plots.append(plot05)

    plot06 = xrtplot.XYCPlot(
        beam=r"toroidMirror01_local",
        xaxis=xrtplot.XYCAxis(
            label=r"y"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"x profile (YZ) - Toroid",
        aspect="auto")
    plots.append(plot06)

    plot07 = xrtplot.XYCPlot(
        beam=r"toroidMirrorSTL_local",
        xaxis=xrtplot.XYCAxis(
            label=r"y"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"x profile (YZ) - STL",
        aspect="auto")
    plots.append(plot07)

    return plots

BeamLine = build_beamline()

def main():
    BeamLine = build_beamline()
#    BeamLine.glow(v2=True)
    plots = define_plots()
    xrtrun.run_ray_tracing(plots, repeats=5, beamLine=BeamLine)
    
if __name__ == '__main__':
    main()


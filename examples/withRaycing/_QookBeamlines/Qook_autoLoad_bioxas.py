# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-09-22"

Created with xrtQook


BioXAS-Main
____________

This example is based on the layout of the BioXAS-Main beamline (Canadian Light Source) and incorporates most commonly used optical components: the wiggler source, flat and curved X-ray mirrors with different coatings, filters, slits and screens.

.. imagezoom:: _images/ex03_glow.png
   :scale: 60%

Features.
________

1) Thin mirrors on bulk substrate, see the RhOnSi and CVDonSi materials.

.. imagezoom:: _images/ex03_p01.png
   :scale: 60%

2) Automatic alignment procedures, see the beamline alignE, alignMode, SSRL_DCM bragg properties and the centers of the optical elements. Note that the value of SSRL_DCM.bragg overrides that beamLine.alignE value, which can be used to introduce misalignment on certain elements.

.. imagezoom:: _images/ex03_p02.png
   :scale: 60%

3) Absorbed power distributons. See how the parameters of the Mirror1.reflect() function.

.. imagezoom:: _images/ex03_p03a.png
   :scale: 60%

Check the plot properties

.. imagezoom:: _images/ex03_p03b.png
   :scale: 60%

Increase the energy range...

.. imagezoom:: _images/ex03_p03c.png
   :scale: 60%

...to see the effect on the absorbed power spectrum.

.. imagezoom:: _images/ex03_p03d.png
   :scale: 60%



"""
import os, sys; sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore

import numpy as np

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

CVD = rmats.Material(
    elements=['C'],
    quantities=[1.0],
    kind=r"plate",
    rho=3.52,
    name=r"CVD")

Rh = rmats.Material(
    elements=['Rh'],
    quantities=[1.0],
    kind=r"mirror",
    rho=12.41,
    name=r"Rh")

Si220 = rmats.CrystalSi(
    a=5.4307717932001225,
    hkl=[2, 2, 0],
    d=1.9200677810242166,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    name=r"Si220",
    kind=r"crystal")

CVDcoating = rmats.Material(
    elements=['C'],
    quantities=[1.0],
    kind=r"mirror",
    rho=3.52,
    name=r"CVDcoating")

Si = rmats.Material(
    elements=['Si'],
    quantities=[1.0],
    kind=r"mirror",
    rho=2.33,
    name=r"Si")

RhOnSi = rmats.Coated(
    coating=Rh,
    surfaceRoughness=2,
    substrate=Si,
    name=r"RhOnSi")

CVDonSi = rmats.Coated(
    coating=CVDcoating,
    surfaceRoughness=2,
    substrate=Si,
    name=r"CVDonSi")

Si220harm = rmats.CrystalHarmonics(
    Nmax=2,
    name=r"Si220harm",
    hkl=[2, 2, 0],
    a=5.41949,
    b=5.41949,
    c=5.41949,
    alpha=1.5707963267948966,
    beta=1.5707963267948966,
    gamma=1.5707963267948966,
    atomsFraction=[1, 1, 1, 1, 1, 1, 1, 1],
    d=1.916079064786341,
    V=159.1751463370933,
    elements=['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si'],
    quantities=[1, 1, 1, 1, 1, 1, 1, 1],
    rho=2.3439368026411915,
    kind=r"crystal harmonics")


def build_beamline():
    BioXAS_Main = raycing.BeamLine(
        alignE=8000,
        name=r"BioXAS_Main")

    BioXAS_Main.Wiggler = raycing.sources_synchr.Wiggler(
        bl=BioXAS_Main,
        name=r"Wiggler",
        center=[0, 0, 0],
        nrays=200000,
        eE=2.9,
        eI=0.25,
        eSigmaX=405.84479792157003,
        eSigmaZ=10.067770358922576,
        eEpsilonX=18.099999999999998,
        eEpsilonZ=0.0362,
        betaX=9.1,
        betaZ=2.8000000000000003,
        xPrimeMax=1.0,
        zPrimeMax=0.375,
        eMin=7998.0,
        eMax=8002.0,
        eN=52,
        K=35,
        period=150,
        n=11,
        nx=51,
        nz=51)

    BioXAS_Main.FEMask = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"FEMask",
        center=[0, 12000, 0],
        opening=[-12.0, 12.0, -1.75, 1.75],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.DiamondFilter = roes.Plate(
        bl=BioXAS_Main,
        name=r"DiamondFilter",
        center=[0, 13600, 0],
        pitch=1.5707963267948966,
        material=CVD,
        limPhysX=[-5.0, 5.0],
        limOptX=[-5, 5],
        limPhysY=[-5.0, 5.0],
        limOptY=[-5, 5],
        t=0.05)

    BioXAS_Main.WhiteBeamSlits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"WhiteBeamSlits",
        center=[0, 14000, 0],
        opening=[-10.0, 10.0, -1.0, 1.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.Mirror1 = roes.ToroidMirror(
        bl=BioXAS_Main,
        name=r"Mirror1",
        center=[0, 14600, 0],
        pitch=r"0.15 deg",
        material=RhOnSi,
        limPhysX=[-12.0, 12.0],
        limOptX=[-12, 12],
        limPhysY=[-495.0, 495.0],
        limOptY=[-495, 495],
        order=1,
        R=7120000.0,
        r=69.81)

    BioXAS_Main.CM_Slits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"CM_Slits",
        center=[0, 15600, r"auto"],
        opening=[-5.0, 5.0, -2.0, 2.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.SSRL_DCM = raycing.oes.DCM(
        bl=BioXAS_Main,
        name=r"SSRL_DCM",
        center=[0, 25300, r"auto"],
        bragg=[8000],
        material=Si220,
        material2=Si220,
        limPhysX=[-20.0, 20.0],
        limOptX=[-20, 20],
        limPhysY=[-72.3913, 3.8087],
        limOptY=[-72.3913, 3.8087],
        limPhysX2=[-20.0, 20.0],
        limPhysY2=[-1.1951, 94.0549],
        limOptX2=[-20, 20],
        limOptY2=[-1.1951, 94.0549],
        order=1,
        cryst2perpTransl=6.5023)

    BioXAS_Main.PreM2Screen = rscreens.Screen(
        bl=BioXAS_Main,
        name=r"PreM2Screen",
        center=[0, 26000, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    BioXAS_Main.Mirror2 = roes.ToroidMirror(
        bl=BioXAS_Main,
        name=r"Mirror2",
        center=[0, 26900, r"auto"],
        pitch=r"-0.15 degree",
        positionRoll=3.141592653589793,
        material=RhOnSi,
        limOptX=[-12, 12],
        limOptY=[-550, 550],
        order=1,
        R=2500000.0,
        r=35.9)

    BioXAS_Main.PhotonShutter = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"PhotonShutter",
        center=[0, 28300, r"auto"],
        opening=[-5.0, 5.0, -2.0, 2.0],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.DBHR1 = raycing.oes_base.OE(
        bl=BioXAS_Main,
        name=r"DBHR1",
        center=[0, 29900, r"auto"],
        pitch=r"0.2 deg",
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10, 10],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75, 75],
        order=1)

    BioXAS_Main.DBHR2 = raycing.oes_base.OE(
        bl=BioXAS_Main,
        name=r"DBHR2",
        center=[0, 30075, r"auto"],
        pitch=r"-0.2 deg",
        positionRoll=3.141592653589793,
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10, 10],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75, 75],
        order=1)

    BioXAS_Main.JJslits = rapts.RectangularAperture(
        bl=BioXAS_Main,
        name=r"JJslits",
        center=[0, 30350, r"auto"],
        opening=[-5.0, 5.0, -0.2, 0.2],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0])

    BioXAS_Main.SampleScreen = rscreens.Screen(
        bl=BioXAS_Main,
        name=r"SampleScreen",
        center=[0, 30650, r"auto"],
        x=[1.0, -0.0, 0.0],
        z=[0.0, 0.0, 1.0],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    return BioXAS_Main


def run_process(BioXAS_Main):
    Wiggler_global = BioXAS_Main.Wiggler.shine()

    FEMask_local = BioXAS_Main.FEMask.propagate(
        beam=Wiggler_global)

    DiamondFilter_global, DiamondFilter_local1, DiamondFilter_local2 = BioXAS_Main.DiamondFilter.double_refract(
        beam=Wiggler_global,
        returnLocalAbsorbed=0)

    WhiteBeamSlits_local = BioXAS_Main.WhiteBeamSlits.propagate(
        beam=DiamondFilter_global)

    Mirror1_global, Mirror1_local = BioXAS_Main.Mirror1.reflect(
        beam=DiamondFilter_global,
        returnLocalAbsorbed=0)

    CM_Slits_local = BioXAS_Main.CM_Slits.propagate(
        beam=Mirror1_global)

    SSRL_DCM_global, SSRL_DCM_local1, SSRL_DCM_local2 = BioXAS_Main.SSRL_DCM.double_reflect(
        beam=Mirror1_global)

    PreM2Screen_local = BioXAS_Main.PreM2Screen.expose(
        beam=SSRL_DCM_global)

    Mirror2_global, Mirror2_local = BioXAS_Main.Mirror2.reflect(
        beam=SSRL_DCM_global)

    PhotonShutter_local = BioXAS_Main.PhotonShutter.propagate(
        beam=Mirror2_global)

    DBHR1_global, DBHR1_local = BioXAS_Main.DBHR1.reflect(
        beam=Mirror2_global)

    DBHR2_global, DBHR2_local = BioXAS_Main.DBHR2.reflect(
        beam=DBHR1_global)

    JJslits_local = BioXAS_Main.JJslits.propagate(
        beam=DBHR2_global)

    SampleScreen_local = BioXAS_Main.SampleScreen.expose(
        beam=DBHR2_global)

    outDict = {
        'Wiggler_global': Wiggler_global,
        'FEMask_local': FEMask_local,
        'DiamondFilter_global': DiamondFilter_global,
        'DiamondFilter_local1': DiamondFilter_local1,
        'DiamondFilter_local2': DiamondFilter_local2,
        'WhiteBeamSlits_local': WhiteBeamSlits_local,
        'Mirror1_global': Mirror1_global,
        'Mirror1_local': Mirror1_local,
        'CM_Slits_local': CM_Slits_local,
        'SSRL_DCM_global': SSRL_DCM_global,
        'SSRL_DCM_local1': SSRL_DCM_local1,
        'SSRL_DCM_local2': SSRL_DCM_local2,
        'PreM2Screen_local': PreM2Screen_local,
        'Mirror2_global': Mirror2_global,
        'Mirror2_local': Mirror2_local,
        'PhotonShutter_local': PhotonShutter_local,
        'DBHR1_global': DBHR1_global,
        'DBHR1_local': DBHR1_local,
        'DBHR2_global': DBHR2_global,
        'DBHR2_local': DBHR2_local,
        'JJslits_local': JJslits_local,
        'SampleScreen_local': SampleScreen_local}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"Wiggler_global",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"plot01",
        fluxFormatStr=r"%g")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"DiamondFilter_local2",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot02",
        fluxKind=r"power",
        fluxFormatStr=r"%g")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"Mirror1_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"plot03",
        fluxKind=r"power",
        fluxFormatStr=r"%g")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"PreM2Screen_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"plot04",
        fluxFormatStr=r"%g")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"SampleScreen_local",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            limits=[7998, 8002]),
        title=r"plot05",
        fluxFormatStr=r"%g")
    plots.append(plot05)
    return plots


def main():
    BioXAS_Main = build_beamline()
    E0 = 0.5 * (BioXAS_Main.Wiggler.eMin +
                BioXAS_Main.Wiggler.eMax)
    BioXAS_Main.alignE=E0
    plots = define_plots()
    BioXAS_Main.explore(plots=xrtplot.serialize_plots(plots))
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        backend=r"raycing",
        beamLine=BioXAS_Main)


if __name__ == '__main__':
    main()

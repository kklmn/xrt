# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2026-10-09"

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

import numpy as np
import sys
sys.path.append(r"C:\GitHub\xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials.elemental as rmatsel
import xrt.backends.raycing.materials.compounds as rmatsco
import xrt.backends.raycing.materials.crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.figure_error as rfe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

CVD = rmats.material.Material(
    elements=['C'],
    quantities=[1.0],
    kind='plate',
    rho=3.52,
    name='CVD')

Rh = rmats.material.Material(
    elements=['Rh'],
    quantities=[1.0],
    kind='mirror',
    rho=12.41,
    name='Rh')

Si220 = rmats.crystals_basic.CrystalSi(
    a=5.4307717932001225,
    hkl=[2, 2, 0],
    d=1.9200677810242166,
    V=160.17128543981727,
    elements=['Si'],
    quantities=[1.0],
    table='Chantler',
    name='Si220')

CVDcoating = rmats.material.Material(
    elements=['C'],
    quantities=[1.0],
    kind='mirror',
    rho=3.52,
    name='CVDcoating')

Si = rmats.material.Material(
    elements=['Si'],
    quantities=[1.0],
    kind='mirror',
    rho=2.33,
    name='Si')

RhOnSi = rmats.multilayer.Coated(
    coating=Rh,
    cThickness=30.0,
    surfaceRoughness=2,
    substrate=Si,
    substRoughness=2.0,
    name='RhOnSi')

CVDonSi = rmats.multilayer.Coated(
    coating=CVDcoating,
    cThickness=20.0,
    surfaceRoughness=2,
    substrate=Si,
    substRoughness=2.0,
    name='CVDonSi')

Si220harm = rmats.crystals_basic.CrystalHarmonics(
    name='Si220harm',
    hkl=[2, 2, 0],
    a=5.41949,
    b=5.41949,
    c=5.41949,
    atomsFraction=[1, 1, 1, 1, 1, 1, 1, 1],
    elements=['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si'],
    quantities=[1, 1, 1, 1, 1, 1, 1, 1],
    rho=2.3439368026411915)


def build_beamline():
    bl = raycing.BeamLine(
        alignE=8000,
        name='BioXAS_Main')

    bl.Wiggler = rsources.synchr.Wiggler(
        bl=bl,
        name='Wiggler',
        center=[0.0, 0.0, 0.0],
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
        B0=2.498940853962759,
        K=35.0,
        period=150,
        n=11)

    bl.FEMask = rapts.RectangularAperture(
        bl=bl,
        name='FEMask',
        center=[0.0, 12000.0, 0.0],
        blades={'left': -12, 'right': 12, 'bottom': -1.75, 'top': 1.75})

    bl.DiamondFilter = roes.refractive.Plate(
        t=0.05,
        bl=bl,
        name='DiamondFilter',
        center=[0.0, 13600.0, 0.0],
        pitch=1.5707963267948966,
        material=CVD,
        limPhysX=[-5.0, 5.0],
        limOptX=[-5.0, 5.0],
        limPhysY=[-5.0, 5.0],
        limOptY=[-5.0, 5.0],
        precisionOpenCL='float64')

    bl.WhiteBeamSlits = rapts.RectangularAperture(
        bl=bl,
        name='WhiteBeamSlits',
        center=[0.0, 14000.0, 0.0],
        blades={'left': -10, 'right': 10, 'bottom': -1, 'top': 1})

    bl.Mirror1 = roes.ToroidMirror(
        bl=bl,
        name='Mirror1',
        center=[0.0, 14600.0, 0.0],
        pitch='0.15 deg',
        material=RhOnSi,
        limPhysX=[-12.0, 12.0],
        limOptX=[-12.0, 12.0],
        limPhysY=[-495.0, 495.0],
        limOptY=[-495.0, 495.0],
        order=1,
        R=7120000.0,
        r=69.81,
        precisionOpenCL='float64')

    bl.CM_Slits = rapts.RectangularAperture(
        bl=bl,
        name='CM_Slits',
        center=[0, 15600, "auto"],
        blades={'left': -5, 'right': 5, 'bottom': -2, 'top': 2})

    bl.SSRL_DCM = roes.dcm.DCM(
        bragg='8000 eV',
        cryst2perpTransl=6.5023,
        limPhysX2=[-20.0, 20.0],
        limPhysY2=[-1.1951, 94.0549],
        limOptX2=[-20.0, 20.0],
        limOptY2=[-1.1951, 94.0549],
        material2=Si220,
        bl=bl,
        name='SSRL_DCM',
        center=[0, 25300, "auto"],
        material=Si220,
        limPhysX=[-20.0, 20.0],
        limOptX=[-20.0, 20.0],
        limPhysY=[-72.3913, 3.8087],
        limOptY=[-72.3913, 3.8087],
        order=1,
        precisionOpenCL='float64')

    bl.PreM2Screen = rscreens.Screen(
        bl=bl,
        name='PreM2Screen',
        center=[0, 26000, "auto"],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    bl.Mirror2 = roes.ToroidMirror(
        bl=bl,
        name='Mirror2',
        center=[0, 26900, "auto"],
        pitch='-0.15 degree',
        positionRoll=3.141592653589793,
        material=RhOnSi,
        limOptX=[-12.0, 12.0],
        limOptY=[-550.0, 550.0],
        order=1,
        R=2500000.0,
        r=35.9,
        precisionOpenCL='float64')

    bl.PhotonShutter = rapts.RectangularAperture(
        bl=bl,
        name='PhotonShutter',
        center=[0, 28300, "auto"],
        blades={'left': -5, 'right': 5, 'bottom': -2, 'top': 2})

    bl.DBHR1 = roes.base.OE(
        bl=bl,
        name='DBHR1',
        center=[0, 29900, "auto"],
        pitch='0.2 deg',
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10.0, 10.0],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75.0, 75.0],
        order=1,
        precisionOpenCL='float64')

    bl.DBHR2 = roes.base.OE(
        bl=bl,
        name='DBHR2',
        center=[0, 30075, "auto"],
        pitch='-0.2 deg',
        positionRoll=3.141592653589793,
        material=CVDcoating,
        limPhysX=[-10.0, 10.0],
        limOptX=[-10.0, 10.0],
        limPhysY=[-75.0, 75.0],
        limOptY=[-75.0, 75.0],
        order=1,
        precisionOpenCL='float64')

    bl.JJslits = rapts.RectangularAperture(
        bl=bl,
        name='JJslits',
        center=[0, 30350, "auto"],
        blades={'left': -5, 'right': 5, 'bottom': -0.2, 'top': 0.2})

    bl.SampleScreen = rscreens.Screen(
        bl=bl,
        name='SampleScreen',
        center=[0, 30650, "auto"],
        limPhysX=[0.0, 0.0],
        limPhysY=[0.0, 0.0],
        cLimits=[0.0, 0.0])

    return bl


def run_process(bl):
    Wiggler_global = bl.Wiggler.shine()

    FEMask_local = bl.FEMask.propagate(
        beam=Wiggler_global)

    DiamondFilter_global, DiamondFilter_local1, DiamondFilter_local2 = bl.DiamondFilter.double_refract(
        beam=Wiggler_global)

    WhiteBeamSlits_local = bl.WhiteBeamSlits.propagate(
        beam=DiamondFilter_global)

    Mirror1_global, Mirror1_local = bl.Mirror1.reflect(
        beam=DiamondFilter_global)

    CM_Slits_local = bl.CM_Slits.propagate(
        beam=Mirror1_global)

    SSRL_DCM_global, SSRL_DCM_local1, SSRL_DCM_local2 = bl.SSRL_DCM.double_reflect(
        beam=Mirror1_global)

    PreM2Screen_local = bl.PreM2Screen.expose(
        beam=SSRL_DCM_global)

    Mirror2_global, Mirror2_local = bl.Mirror2.reflect(
        beam=SSRL_DCM_global)

    PhotonShutter_local = bl.PhotonShutter.propagate(
        beam=Mirror2_global)

    DBHR1_global, DBHR1_local = bl.DBHR1.reflect(
        beam=Mirror2_global)

    DBHR2_global, DBHR2_local = bl.DBHR2.reflect(
        beam=DBHR1_global)

    JJslits_local = bl.JJslits.propagate(
        beam=DBHR2_global)

    SampleScreen_local = bl.SampleScreen.expose(
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


def define_plots(bl):
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam='Wiggler_global',
        rayFlag=[1],
        xaxis=xrtplot.XYCAxis(
            label='x'),
        yaxis=xrtplot.XYCAxis(
            label='z'),
        caxis=xrtplot.XYCAxis(
            label='energy',
            unit='eV'),
        aspect='auto',
        title='01 - Wiggler Source',
        fluxFormatStr='%g',
        contourFactor=1,
        name='plot01')
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam='DiamondFilter_local2',
        rayFlag=[1],
        xaxis=xrtplot.XYCAxis(
            label='x'),
        yaxis=xrtplot.XYCAxis(
            label='y'),
        caxis=xrtplot.XYCAxis(
            label='energy',
            unit='eV'),
        title='02 - CVD Filter Absorbed Power',
        fluxKind='power',
        fluxFormatStr='%g',
        contourFactor=1,
        name='plot02')
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam='Mirror1_local',
        rayFlag=[1],
        xaxis=xrtplot.XYCAxis(
            label='x'),
        yaxis=xrtplot.XYCAxis(
            label='y'),
        caxis=xrtplot.XYCAxis(
            label='energy',
            unit='eV'),
        aspect='auto',
        title='03 - Mirror1 Absorbed Power',
        fluxKind='power',
        fluxFormatStr='%g',
        contourFactor=1,
        name='plot03')
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam='PreM2Screen_local',
        rayFlag=[1],
        xaxis=xrtplot.XYCAxis(
            label='x'),
        yaxis=xrtplot.XYCAxis(
            label='z'),
        caxis=xrtplot.XYCAxis(
            label='energy',
            unit='eV'),
        title='04 - preM2 Screen Footprint',
        fluxFormatStr='%g',
        contourFactor=1,
        saveName=r"c:\nsls\data1.png",
        name='plot04')
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam='SampleScreen_local',
        rayFlag=[1],
        xaxis=xrtplot.XYCAxis(
            label='x'),
        yaxis=xrtplot.XYCAxis(
            label='z'),
        caxis=xrtplot.XYCAxis(
            label='energy',
            unit='eV',
            limits=[7998, 8002]),
        title='05 - Sample',
        fluxFormatStr='%g',
        contourFactor=1,
        saveName=r"c:\blabla\path",
        name='plot05')
    plots.append(plot05)
    return plots


def main():
    BioXAS_Main = build_beamline()
    E0 = 0.5 * (BioXAS_Main.Wiggler.eMin +
                BioXAS_Main.Wiggler.eMax)
    BioXAS_Main.alignE = E0
    plots = define_plots(BioXAS_Main)
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        backend='raycing',
        beamLine=BioXAS_Main)


if __name__ == '__main__':
    main()

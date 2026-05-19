"""
Trace the synthetic TXM sample with a geometric source.

The Plate geometry is taken from the HDF5 TXM sample limits. The screen is 5 m
downstream of the sample.
"""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.materials.compounds as xcomp
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources
import xrt.plotter as xrtp
import xrt.runner as xrtr

showIn3D = False

HERE = Path(__file__).resolve().parent
SAMPLE_FILE = HERE / "txm_sample_50um_500.h5"
SAMPLE_Y = 10.0  # mm
SCREEN_DISTANCE = 5000.0  # mm
ENERGY_SIGMA = 5.0  # eV
ROTATION_VALUES = range(-45, 46, 2)  # degrees
ENERGY_VALUES = range(220, 650, 5)  # eV
BINS = 512


def make_material_table():
    return {
        0: xcomp.Water(kind="plate"),
        1: xcomp.RockSalt(kind="plate"),
        2: xcomp.Air(kind="plate"),
        3: xcomp.Polyimide(kind="plate"),
        4: xcomp.Fluorite(kind="plate"),
        5: xcomp.Mylar(kind="plate"),
    }


def load_sample_limits(file_name):
    with h5py.File(file_name, "r") as h5:
        limits = h5["limits"]
        xLimits = np.asarray(limits["x"][:], dtype=float)
        yLimits = np.asarray(limits["y"][:], dtype=float)
        zLimits = np.asarray(limits["z"][:], dtype=float)
    return xLimits, yLimits, zLimits


def build_beamline(nrays=500000, energy=520.0):
    beamLine = raycing.BeamLine()
    xLimits, yLimits, zLimits = load_sample_limits(SAMPLE_FILE)
    sampleThickness = float(zLimits[1] - zLimits[0])

    txmMaterial = rm.TXMMaterial(
        str(SAMPLE_FILE), make_material_table(), name="indexed TXM sample")

    beamLine.source = rsources.GeometricSource(
        bl=beamLine,
        name="TXM source",
        center=(0, 0, 0),
        nrays=nrays,
        distx="normal",
        dx=0.001,
        distz="normal",
        dz=0.001,
        distxprime="normal",
        dxprime=2e-3,
        distzprime="normal",
        dzprime=2e-3,
        distE="normal",
        energies=(energy, ENERGY_SIGMA),
        polarization="horizontal")

    beamLine.sample = roes.Plate(
        bl=beamLine,
        name="TXM sample",
        center=(0, SAMPLE_Y, 0),
        pitch="90deg",
        t=sampleThickness,
        limPhysX=tuple(xLimits),
        limPhysY=tuple(yLimits),
        material=txmMaterial)

    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="TXM screen",
        center=(0, SAMPLE_Y + SCREEN_DISTANCE, 0),
        limPhysX=[-20, 20],
        limPhysY=[-20, 20])

    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine(withAmplitudes=True)
    beamSampleGlobal, beamSampleLocal1, beamSampleLocal2 = \
        beamLine.sample.double_refract(beamSource)
    beamScreenLocal = beamLine.screen.expose(beamSampleGlobal)

    return {
        "beamSource": beamSource,
        "beamSampleGlobal": beamSampleGlobal,
        "beamSampleLocal1": beamSampleLocal1,
        "beamSampleLocal2": beamSampleLocal2,
        "beamScreenLocal": beamScreenLocal,
    }


rrun.run_process = run_process


def define_plots(output_dir, energy=520.0, screenOnly=False):
    output_dir = Path(output_dir)
    energyLimits = [energy - 4*ENERGY_SIGMA, energy + 4*ENERGY_SIGMA]

    plots = []
    if not screenOnly:
        samplePlot = xrtp.XYCPlot(
            "beamSampleLocal2",
            (1,),
            xaxis=xrtp.XYCAxis(
                "x", "um", factor=-1e3, limits=[-40, 40], bins=BINS, ppb=1),
            yaxis=xrtp.XYCAxis(
                "y", "um", factor=1e3, limits=[-40, 40], bins=BINS, ppb=1),
            caxis=xrtp.XYCAxis(
                "energy", "eV", limits=energyLimits, bins=BINS, ppb=1),
            title="TXM sample local2",
            saveName=[str(output_dir / "sample.png")])
        plots.append(samplePlot)

    screenPlot = xrtp.XYCPlot(
        "beamScreenLocal",
        (1,),
        xaxis=xrtp.XYCAxis(
            "x", "mm", limits=[-20, 20], bins=BINS, ppb=1),
        yaxis=xrtp.XYCAxis(
            "z", "mm", limits=[-20, 20], bins=BINS, ppb=1),
        caxis=xrtp.XYCAxis(
            "energy", "eV", limits=energyLimits, bins=BINS, ppb=1),
        title="TXM screen",
        saveName=[str(output_dir / "screen.png")])
    plots.append(screenPlot)

    return plots


def _scan_frame_name(output_dir, index, parameter, value):
    return Path(output_dir) / "frame_{0:03d}_{1}_{2}.png".format(
        index, parameter, value)


def make_glow_scan(scanName, output_dir=HERE):
    """Build a compact xrtGlow track matching the runner scan filenames."""
    output_dir = Path(output_dir).resolve()

    if scanName == "rotation":
        values = ["{0:+d}deg".format(yawDeg)
                  for yawDeg in ROTATION_VALUES]
        output = {
            "glowFrameName": str(
                output_dir / "frame_{index:03d}_yaw_{value}.png")}
        item = {
            "type": "track",
            "id": "TXM sample.yaw",
            "start": 0,
            "duration": len(values),
            "target": "TXM sample",
            "property": "yaw",
            "values": {"type": "list", "values": values},
            "output": output,
        }
    elif scanName == "energy":
        values = ["[{0:d}, {1:g}]".format(e0, ENERGY_SIGMA)
                  for e0 in ENERGY_VALUES]
        energyLabels = ["{0:d}eV".format(e0) for e0 in ENERGY_VALUES]
        output = {
            "glowFrameName": str(
                output_dir / "frame_{index:03d}_energy_{energy}.png")}
        item = {
            "type": "track",
            "id": "TXM source.energies",
            "start": 0,
            "duration": len(values),
            "target": "TXM source",
            "property": "energies",
            "values": {"type": "list", "values": values},
            "vars": {
                "energy": {"type": "list", "values": energyLabels}},
            "output": output,
        }
    else:
        return None

    return {"version": 1, "kind": "timeline_recipe",
            "frames": len(values), "output": output, "items": [item]}


def _set_screen_frame(plots, output_dir, index, parameter, value):
    plots[-1].saveName = [
        str(_scan_frame_name(output_dir, index, parameter, value))]


def _set_plot_energy_limits(plots, energy):
#    limits = [energy - 4*ENERGY_SIGMA, energy + 4*ENERGY_SIGMA]
    limits = [min(ENERGY_VALUES)- 4*ENERGY_SIGMA,
              max(ENERGY_VALUES)- 4*ENERGY_SIGMA]
    for plot in plots:
        plot.caxis.limits = limits


def scan_rotation(plots=None, beamLine=None, output_dir=HERE, energy=520.0):
    for index, yawDeg in enumerate(ROTATION_VALUES):
        beamLine.sample.yaw = np.radians(yawDeg)
        if hasattr(beamLine.sample, "get_orientation"):
            beamLine.sample.get_orientation()
        _set_plot_energy_limits(plots, energy)
        _set_screen_frame(
            plots, output_dir, index, "yaw", "{0:+d}deg".format(yawDeg))
        yield


def scan_energy(plots=None, beamLine=None, output_dir=HERE, energy=520.0):
    for index, e0 in enumerate(ENERGY_VALUES):
        beamLine.source.energies = (float(e0), ENERGY_SIGMA)
        _set_plot_energy_limits(plots, float(e0))
        _set_screen_frame(
            plots, output_dir, index, "energy", "{0:d}eV".format(e0))
        yield


def plot_generator(
        plots=None, beamLine=None, scanName=None, output_dir=HERE,
        energy=520.0):
    if scanName == "rotation":
        for _ in scan_rotation(plots, beamLine, output_dir, energy):
            yield
    elif scanName == "energy":
        for _ in scan_energy(plots, beamLine, output_dir, energy):
            yield
    else:
        yield


def _axis_edges(axis):
    if getattr(axis, "binEdges", None) is not None and axis.binEdges.any():
        return axis.binEdges
    return [
        axis.limits[0] - axis.offset,
        axis.limits[1] - axis.offset,
    ]


def save_total_intensity_image(plot, fileName):
    image = plot.total2D.real.copy()
    xedges = _axis_edges(plot.xaxis)
    yedges = _axis_edges(plot.yaxis)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    positive = image[image > 0]
    norm = None
    if positive.size:
        norm = mplcolors.LogNorm(vmin=1e-3, vmax=1)
    image[image <= 0] = float("nan")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(
        image, origin="lower", extent=extent, interpolation="nearest",
        cmap="jet", norm=norm, aspect=plot.aspect)
    ax.set_xlabel(plot.xaxis.displayLabel)
    ax.set_ylabel(plot.yaxis.displayLabel)
    ax.set_title("TXM screen total intensity")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("total intensity (log scale)")
    fig.savefig(fileName, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace and plot the synthetic TXM sample.")
    parser.add_argument(
        "--nrays", type=int, default=500000,
        help="Number of rays generated by the geometric source.")
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="Number of ray-tracing repeats.")
    parser.add_argument(
        "--processes", type=int, default=4,
        help="Number of worker processes for ray tracing.")
    parser.add_argument(
        "--energy", type=float, default=820.0,
        help="Central source energy in eV. The sigma is fixed at 5 eV.")
    parser.add_argument(
        "--scan", choices=("none", "rotation", "energy"), default="energy",
        help="Optional screen-only scan generator.")
    parser.add_argument(
        "--showIn3d", "--showIn3D", dest="showIn3D", action="store_true",
        help="Open xrtGlow instead of running the scripted trace.")
    parser.add_argument(
        "--output-dir", type=Path, default=HERE,
        help="Directory for sample.png and screen.png.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scanName = None if args.scan == "none" else args.scan
    beamLine = build_beamline(args.nrays, args.energy)
    if showIn3D or args.showIn3D:
        scan = make_glow_scan(scanName, args.output_dir) if scanName else None
        beamLine.glow(
            scale=[1000, 1, 1000], centerAt="TXM sample", scan=scan)
        return

    plots = define_plots(args.output_dir,
                         args.energy, screenOnly=bool(scanName))
    generatorKWargs = {
        "plots": plots,
        "beamLine": beamLine,
        "scanName": scanName,
        "output_dir": args.output_dir,
        "energy": args.energy,
    }
    xrtr.run_ray_tracing(
        plots=plots, repeats=args.repeats, backend="raycing",
        processes=args.processes, beamLine=beamLine,
        generator=plot_generator if scanName else None,
        generatorKWargs=generatorKWargs if scanName else "auto")
#    save_total_intensity_image(
#        plots[1], args.output_dir / "screen_total_jet.png")
    if scanName:
        print("Wrote scan frames: {0}".format(
            args.output_dir / "frame_*.png"))
    else:
        print("Wrote: {0}".format(args.output_dir / "sample.png"))
        print("Wrote: {0}".format(args.output_dir / "screen.png"))
#    print("Wrote: {0}".format(args.output_dir / "screen_total_jet.png"))


if __name__ == "__main__":
    main()

"""
Trace the synthetic TXM sample with a geometric source.

The Plate geometry is taken from the HDF5 TXM sample limits. The screen is 5 m
downstream of the sample. Edit the settings below to choose
the ray count, source energy, scan and viewer mode. Scripted rotation scans
also save numerical projections and flat/dark references in a Data Exchange
HDF5 file readable by dxchange.read_aps_32id() for use with TomoPy. Rotation
scans include unchanged rays that miss both sample surfaces. Rays hitting only
one surface remain lost under Plate's two-surface model; this does not implement
propagation through side faces.
"""

from __future__ import print_function

import sys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import h5py
import numpy as np

sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore

import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.materials.compounds as xcomp
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources
import xrt.plotter as xrtp
import xrt.runner as xrtr


showIn3D = True

sampleFile = "txm_sample_50um_500.h5"
outputDir = "."
tomopyFile = "txm_projections.h5"  # Rotation export, relative to outputDir.
nrays = 500000
repeats = 20
processes = 4
energy = 520.0  # eV
scanName = "rotation"  # None, "rotation" or "energy"
sampleY = 10.0  # mm
screenDistance = 5000.0  # mm
energySigma = 5.0  # eV
rotationValues = range(-45, 46, 2)  # degrees
energyValues = range(220, 650, 5)  # eV
bins = 512


txmMaterialIndex = {
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


def build_beamline(nrays=nrays, energy=energy):
    beamLine = raycing.BeamLine()
    beamLine.flatField = False
    xLimits, yLimits, zLimits = load_sample_limits(sampleFile)
    sampleThickness = float(zLimits[1] - zLimits[0])

    txmMaterial = rm.TXMMaterial(
        str(sampleFile), txmMaterialIndex, name="indexed TXM sample")

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
        energies=(energy, energySigma),
        polarization="horizontal")

    beamLine.sample = roes.Plate(
        bl=beamLine,
        name="TXM sample",
        center=(0, sampleY, 0),
        pitch="90deg",
        t=sampleThickness,
        limPhysX=tuple(xLimits),
        limPhysY=tuple(yLimits),
        material=txmMaterial)

    beamLine.screen = rscreens.Screen(
        bl=beamLine,
        name="TXM screen",
        center=(0, sampleY + screenDistance, 0),
        limPhysX=[-20, 20],
        limPhysY=[-20, 20])

    return beamLine


def run_process(beamLine):
    beamSource = beamLine.source.shine(withAmplitudes=True)
    if beamLine.flatField:
        return {
            "beamSource": beamSource,
            "beamScreenLocal": beamLine.screen.expose(beamSource),
        }

    beamSampleGlobal, beamSampleLocal1, beamSampleLocal2 = \
        beamLine.sample.double_refract(
            beamSource, lostAsOver=(scanName == "rotation"))
    beamScreenLocal = beamLine.screen.expose(beamSampleGlobal)

    return {
        "beamSource": beamSource,
        "beamSampleGlobal": beamSampleGlobal,
        "beamSampleLocal1": beamSampleLocal1,
        "beamSampleLocal2": beamSampleLocal2,
        "beamScreenLocal": beamScreenLocal,
    }


rrun.run_process = run_process


def define_plots(output_dir, energy=energy, screenOnly=False):
    output_dir = os.fspath(output_dir)
    energyLimits = [energy - 4*energySigma, energy + 4*energySigma]

    plots = []
    if not screenOnly:
        samplePlot = xrtp.XYCPlot(
            "beamSampleLocal2",
            (1,),
            xaxis=xrtp.XYCAxis(
                "x", "um", factor=-1e3, limits=[-40, 40], bins=bins, ppb=1),
            yaxis=xrtp.XYCAxis(
                "y", "um", factor=1e3, limits=[-40, 40], bins=bins, ppb=1),
            caxis=xrtp.XYCAxis(
                "energy", "eV", limits=energyLimits, bins=bins, ppb=1),
            title="TXM sample local2",
            saveName=[str(os.path.join(output_dir, "sample.png"))])
        plots.append(samplePlot)

    screenPlot = xrtp.XYCPlot(
        "beamScreenLocal",
        (1, 3) if scanName == "rotation" else (1,),
        xaxis=xrtp.XYCAxis(
            "x", "mm", limits=[-20, 20], bins=bins, ppb=1),
        yaxis=xrtp.XYCAxis(
            "z", "mm", limits=[-20, 20], bins=bins, ppb=1),
        caxis=xrtp.XYCAxis(
            "energy", "eV", limits=energyLimits, bins=bins, ppb=1),
        title="TXM screen",
        saveName=[str(os.path.join(output_dir, "screen.png"))])
    plots.append(screenPlot)

    return plots


def _scan_frame_name(output_dir, index, parameter, value):
    return os.path.join(output_dir,
                        "frame_{0:03d}_{1}_{2}.png".format(
                            index, parameter, value))


def make_glow_scan(scanName, output_dir=outputDir):
    """Build a compact xrtGlow track matching the runner scan filenames."""
    output_dir = os.path.abspath(output_dir)

    if scanName == "rotation":
        values = ["{0:+d}deg".format(yawDeg)
                  for yawDeg in rotationValues]
        output = {
            "glowFrameName": str(
                os.path.join(output_dir, "frame_{index:03d}_yaw_{value}.png"))}
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
        values = ["[{0:d}, {1:g}]".format(e0, energySigma)
                  for e0 in energyValues]
        energyLabels = ["{0:d}eV".format(e0) for e0 in energyValues]
        output = {
            "glowFrameName": str(
                os.path.join(output_dir, "frame_{index:03d}_energy_{energy}.png"))}
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
    if scanName == 'rotation':
        limits = [energy - 4*energySigma, energy + 4*energySigma]
    else:
        limits = [min(energyValues) - 4*energySigma,
                  max(energyValues) + 4*energySigma]
    for plot in plots:
        plot.caxis.limits = limits


def start_tomopy_export(fileName, plot):
    """Start a Data Exchange file using the completed flat-field histogram."""
    image = np.asarray(plot.total2D.real, dtype=np.float32)
    rows, columns = image.shape
    with h5py.File(fileName, "w") as h5:
        exchange = h5.create_group("exchange")
        data = exchange.create_dataset(
            "data", shape=(0, rows, columns),
            maxshape=(None, rows, columns), dtype="f4",
            chunks=(1, rows, columns), compression="gzip")
        data.attrs["axes"] = "theta:y:x"
        data.attrs["units"] = "arbitrary"
        data.attrs["ray_states"] = "1: transmitted; 3: missed both surfaces"
        data.attrs["side_faces"] = (
            "Plate two-surface refraction; side-face losses remain excluded")
        theta = exchange.create_dataset(
            "theta", shape=(0,), maxshape=(None,), dtype="f8")
        theta.attrs["units"] = "degrees"
        flat = exchange.create_dataset("data_white", data=image[None, :, :])
        flat.attrs["nrays"] = plot.nRaysAll
        exchange.create_dataset("data_dark", data=np.zeros_like(flat[:]))
        exchange.create_dataset(
            "nrays", shape=(0,), maxshape=(None,), dtype="i8")
        for name, axis in [("x_bin_edges", plot.xaxis),
                           ("z_bin_edges", plot.yaxis)]:
            edges = exchange.create_dataset(name, data=axis.binEdges)
            edges.attrs["units"] = "mm"


def save_tomopy_projection(fileName, plot, yawDeg):
    """Append one completed projection; close the file between scan steps."""
    with h5py.File(fileName, "r+") as h5:
        exchange = h5["exchange"]
        index = exchange["data"].shape[0]
        for name in ["data", "theta", "nrays"]:
            exchange[name].resize(index + 1, axis=0)
        exchange["data"][index] = plot.total2D.real.astype(np.float32)
        exchange["theta"][index] = yawDeg
        exchange["nrays"][index] = plot.nRaysAll


def scan_rotation(plots=None, beamLine=None, output_dir=outputDir, energy=energy):
    fileName = os.path.join(output_dir, tomopyFile)
    screenPlot = plots[-1]
    _set_plot_energy_limits(plots, energy)
    screenPlot.saveName = [os.path.join(output_dir, "flat.png")]
    beamLine.flatField = True
    try:
        yield  # Accumulate the same repeats with the sample bypassed.
    finally:
        beamLine.flatField = False
    start_tomopy_export(fileName, screenPlot)

    for index, yawDeg in enumerate(rotationValues):
        beamLine.sample.yaw = np.radians(yawDeg)
        if hasattr(beamLine.sample, "get_orientation"):
            beamLine.sample.get_orientation()
        _set_plot_energy_limits(plots, energy)
        _set_screen_frame(
            plots, output_dir, index, "yaw", "{0:+d}deg".format(yawDeg))
        yield
        # The runner clears the histogram only after this generator resumes.
        save_tomopy_projection(fileName, screenPlot, yawDeg)
    print("Wrote tomography data: {0}".format(fileName))


def scan_energy(plots=None, beamLine=None, output_dir=outputDir, energy=energy):
    for index, e0 in enumerate(energyValues):
        beamLine.source.energies = (float(e0), energySigma)
        _set_plot_energy_limits(plots, float(e0))
        _set_screen_frame(
            plots, output_dir, index, "energy", "{0:d}eV".format(e0))
        yield


def plot_generator(
        plots=None, beamLine=None, scanName=None, output_dir=outputDir,
        energy=energy):
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


def main():
    output_dir = os.fspath(outputDir)
    os.makedirs(output_dir, exist_ok=True)
    beamLine = build_beamline(nrays, energy)
    if showIn3D:
        scan = make_glow_scan(scanName, output_dir) if scanName else None
        beamLine.glow(
            scale=[1000, 1, 1000], centerAt="TXM sample", scan=scan)
        return

    plots = define_plots(output_dir,
                         energy, screenOnly=bool(scanName))
    generatorKWargs = {
        "plots": plots,
        "beamLine": beamLine,
        "scanName": scanName,
        "output_dir": output_dir,
        "energy": energy,
    }
    xrtr.run_ray_tracing(
        plots=plots, repeats=repeats, backend="raycing",
        processes=processes, beamLine=beamLine,
        generator=plot_generator if scanName else None,
        generatorKWargs=generatorKWargs if scanName else "auto")
#    save_total_intensity_image(
#        plots[1], os.path.join(output_dir, "screen_total_jet.png"))
    if scanName:
        print("Wrote scan frames: {0}".format(
            os.path.join(output_dir, "frame_*.png")))
    else:
        print("Wrote: {0}".format(os.path.join(output_dir, "sample.png")))
        print("Wrote: {0}".format(os.path.join(output_dir, "screen.png")))
#    print("Wrote: {0}".format(os.path.join(output_dir, "screen_total_jet.png")))


if __name__ == "__main__":
    main()

"""
Create a synthetic indexed-volume TXM sample.

The HDF5 layout is:

    /indexGrid       uint8, shape (nz, ny, nx), axisOrder="zyx"
    /limits/x        [xmin, xmax] in mm
    /limits/y        [ymin, ymax] in mm
    /limits/z        [zmin, zmax] in mm
"""

from __future__ import print_function

import argparse
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE / "txm_sample_50um_500.h5"


def make_grid(n=500):
    grid = np.zeros((n, n, n), dtype=np.uint8)  # z, y, x
    zz, yy, xx = np.ogrid[:n, :n, :n]

    # Coordinates are in point indices: x left-right, y bottom-top, z near-far.
    sphere_center = (125, 375, 125)
    sphere_radius = 100
    sphere = (
        (xx - sphere_center[0])**2 +
        (yy - sphere_center[1])**2 +
        (zz - sphere_center[2])**2 <= sphere_radius**2)
    grid[sphere] = 1

#    cube_center = (375, 125, 375)
    cube_center = (125, 375, 125)
    cube_side = 100
    cube_half = cube_side // 2
    xs = slice(cube_center[0] - cube_half, cube_center[0] + cube_half)
    ys = slice(cube_center[1] - cube_half, cube_center[1] + cube_half)
    zs = slice(cube_center[2] - cube_half, cube_center[2] + cube_half)
    grid[zs, ys, xs] = 2

    cylinder_center = (250, 125, 125)
    cylinder_length = int(round(n * 2.0 / 3.0))
    cylinder_radius = n / 16.0
    x0 = cylinder_center[0] - cylinder_length // 2
    x1 = x0 + cylinder_length
    cylinder = (
        (yy[:, :, 0] - cylinder_center[1])**2 +
        (zz[:, :, 0] - cylinder_center[2])**2 <= cylinder_radius**2)
    cylinder_view = grid[:, :, x0:x1]
    cylinder_view[cylinder, :] = 3

    bar_center = (250, 375, 375)
    bar_length = int(round(n * 2.0 / 3.0))
    bar_side = n // 10
    bar_half = bar_side // 2
    x0 = bar_center[0] - bar_length // 2
    x1 = x0 + bar_length
    y0 = bar_center[1] - bar_half
    y1 = y0 + bar_side
    z0 = bar_center[2] - bar_half
    z1 = z0 + bar_side
    grid[z0:z1, y0:y1, x0:x1] = 4

    cube_b_center = (375, 125, 375)
    cube_b_side = 180
    cube_b_half = cube_b_side // 2
    xs = slice(cube_b_center[0] - cube_b_half, cube_b_center[0] + cube_b_half)
    ys = slice(cube_b_center[1] - cube_b_half, cube_b_center[1] + cube_b_half)
    zs = slice(cube_b_center[2] - cube_b_half//8,
               cube_b_center[2] + cube_b_half//8)
    grid[zs, ys, xs] = 5
    return grid


def write_sample(file_name, n=500, width=0.05, height=0.05, thickness=0.01):
    grid = make_grid(n)
    xlim = np.array([-width*0.5, width*0.5], dtype=np.float64)
    ylim = np.array([-height*0.5, height*0.5], dtype=np.float64)
    zlim = np.array([0.0, thickness], dtype=np.float64)

    with h5py.File(file_name, "w") as h5:
        dset = h5.create_dataset(
            "indexGrid", data=grid, dtype="u1", chunks=(1, n, n),
            compression="gzip", compression_opts=4, shuffle=True)
        dset.attrs["axisOrder"] = "zyx"
        dset.attrs["description"] = (
            "500x500x500 indexed TXM sample, 50 microns per side")
        dset.attrs["backgroundIndex"] = 0

        limits = h5.create_group("limits")
        limits.create_dataset("x", data=xlim)
        limits.create_dataset("y", data=ylim)
        limits.create_dataset("z", data=zlim)
        limits.attrs["units"] = "mm"

        h5.attrs["units"] = "mm"
        h5.attrs["shapeDescriptions"] = (
            "1 sphere, 2 cube, 4 cylinder, 8 bar")
        h5.attrs["sphere"] = (
            "center=(125, 375, 125) points, diameter=200 points")
        h5.attrs["cube"] = (
            "center=(375, 125, 375) points, side=180 points")
        h5.attrs["cylinder"] = (
            "center=(250, 375, 375) points, x length=2*n/3, "
            "diameter=n/8")
        h5.attrs["bar"] = (
            "center=(250, 125, 125) points, x length=2*n/3, "
            "cross-section=n/10 by n/10")

    return grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a synthetic TXM indexed-volume HDF5 sample.")
    parser.add_argument(
        "fileName", nargs="?", type=Path, default=DEFAULT_FILE,
        help="Output HDF5 file. Defaults to txm_sample_50um_500.h5.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.fileName.parent.mkdir(parents=True, exist_ok=True)
    grid = write_sample(args.fileName)
    unique, counts = np.unique(grid, return_counts=True)
    print("Wrote: {0}".format(args.fileName))
    print("Grid shape (z, y, x): {0}".format(grid.shape))
    print("Voxel counts: {0}".format(dict(zip(unique.tolist(),
          counts.tolist()))))


if __name__ == "__main__":
    main()

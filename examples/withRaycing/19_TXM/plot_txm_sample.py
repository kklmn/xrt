"""
Quick visualization helpers for the synthetic TXM indexed-volume sample.

The 3D scatter plot is intentionally sampled. The full occupied volume has
millions of voxels, which is not a good target for matplotlib's 3D scatter.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE / "txm_sample_50um_500.h5"
MATERIAL_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:cyan",
}


def load_sample(file_name):
    with h5py.File(file_name, "r") as h5:
        grid = h5["indexGrid"][:]
        xlim = h5["limits/x"][:]
        ylim = h5["limits/y"][:]
        zlim = h5["limits/z"][:]
    return grid, xlim, ylim, zlim


def voxel_centers(lim, n):
    step = (lim[1] - lim[0]) / n
    return np.linspace(lim[0], lim[1], n, endpoint=False) + 0.5 * step


def save_scatter(grid, xlim, ylim, zlim, output, max_points=80000, seed=11):
    nz, ny, nx = grid.shape
    x = voxel_centers(xlim, nx)
    y = voxel_centers(ylim, ny)
    z = voxel_centers(zlim, nz)

    occupied = np.flatnonzero(grid)
    if occupied.size > max_points:
        rng = np.random.default_rng(seed)
        occupied = rng.choice(occupied, max_points, replace=False)

    iz, iy, ix = np.unravel_index(occupied, grid.shape)
    values = grid[iz, iy, ix]

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for value in sorted(MATERIAL_COLORS):
        mask = values == value
        if not np.any(mask):
            continue
        ax.scatter(
            x[ix[mask]] * 1e3, z[iz[mask]] * 1e3, y[iy[mask]] * 1e3,
            s=1, alpha=0.45, color=MATERIAL_COLORS[value],
            label="material {0}".format(value))

    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_zlabel("y (um)")
    ax.set_xlim(xlim * 1e3)
    ax.set_ylim(zlim * 1e3)
    ax.set_zlim(ylim * 1e3)
    ax.set_title("TXM sample occupied voxels (sampled)")
    ax.legend(loc="upper left", markerscale=6)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return occupied.size


def save_xz_projection(grid, xlim, ylim, zlim, output):
    dy = (ylim[1] - ylim[0]) / grid.shape[1]
    projection = grid.astype(np.float32).sum(axis=1) * dy

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(
        projection,
        origin="lower",
        extent=(xlim[0] * 1e3, xlim[1] * 1e3,
                zlim[0] * 1e3, zlim[1] * 1e3),
        interpolation="nearest",
        cmap="viridis")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title("TXM sample x-z projection")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("integrated material index (index mm)")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return projection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a synthetic TXM indexed-volume HDF5 sample.")
    parser.add_argument(
        "fileName", nargs="?", type=Path, default=DEFAULT_FILE,
        help="HDF5 sample file. Defaults to txm_sample_50um_500.h5.")
    parser.add_argument(
        "--output-dir", type=Path, default=HERE,
        help="Directory for generated PNG files.")
    parser.add_argument(
        "--max-scatter-points", type=int, default=80000,
        help="Maximum occupied voxels shown in the 3D scatter plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    grid, xlim, ylim, zlim = load_sample(args.fileName)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scatter_file = args.output_dir / "txm_sample_scatter.png"
    projection_file = args.output_dir / "txm_sample_xz_projection.png"

    shown = save_scatter(
        grid, xlim, ylim, zlim, scatter_file,
        max_points=args.max_scatter_points)
    projection = save_xz_projection(grid, xlim, ylim, zlim, projection_file)

    unique, counts = np.unique(grid, return_counts=True)
    print("Loaded: {0}".format(args.fileName))
    print("Grid shape (z, y, x): {0}".format(grid.shape))
    print("Voxel counts: {0}".format(dict(zip(unique.tolist(), counts.tolist()))))
    print("Scatter voxels shown: {0}".format(shown))
    print("Projection range: {0:g} to {1:g}".format(
        projection.min(), projection.max()))
    print("Wrote: {0}".format(scatter_file))
    print("Wrote: {0}".format(projection_file))


if __name__ == "__main__":
    main()

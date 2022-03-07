"""Numba accelerated zonal statistics."""
from numba import njit
import numpy as np

from ._helper import lazycompile


@lazycompile(njit)
def do_mean(pixels, z_pixels, nodata, num_zones, dtype="float64"):
    """Calculate the zonal mean.

    The mean for each pixel of `pixels` is calculated for each zone in
    `z_pixels`.

    The zones in `z_pixels` have to be numbered in a linear fashion,
    starting with 0 for the first zone and num_zones for the last zone.

    Args:
        pixels: input value pixels (T,Y,X)
        z_pixels: zonal pixels (Y,X)
        nodata: nodata value in values
        num_zones: number of zones in z_pixels
        dtype: datatype
    """
    t, nr, nc = pixels.shape

    sums = np.zeros((t, num_zones), dtype=dtype)
    valids = np.zeros((t, num_zones), dtype=dtype)

    for tix in range(t):
        for rw in range(nr):
            for cl in range(nc):
                pix = pixels[tix, rw, cl]
                z_idx = z_pixels[rw, cl]
                if pix != nodata and z_idx >= 0:
                    sums[tix, z_idx] += pix
                    valids[tix, z_idx] += 1

        for idx in range(sums.shape[1]):
            if valids[tix, idx] > 0:
                sums[tix, idx] = sums[tix, idx] / valids[tix, idx]
            else:
                sums[tix, idx] = np.nan

    return sums

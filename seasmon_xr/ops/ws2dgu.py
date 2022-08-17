"""Whittaker smoother with fixed lambda (S)."""
import numpy as np
from numba import guvectorize
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, int16[:])], "(n),(),() -> (n)", nopython=True
    )
)
def ws2dgu(y, lmda, nodata, out):
    """
    Whittaker smoother with fixed smoothing coefficient.

    The Whittaker Smoother is a penalized least square algorithm for smoothing and interpolation
    of noisy data. The smoothing coefficient optimization allows to automate the right amount of
    penalty.
    References:
    - Eilers, A perfect smoother, https://doi.org/10.1021/ac034173t
    - Eilers, Pesendorfer and Bonifacio, Automatic smoothing of remote sensing data,
      https://doi.org/10.1016/j.csda.2009.09.020)
    - Garcia, Robust smoothing of gridded data in one and higher dimensions with missing values,
      https://doi.org/10.1016/j.csda.2009.09.020

    Args:
        y: time-series numpy array
        l: smoothing parameter lambda (S)
        w: weights numpy array
        p: "Envelope" value
    Returns:
        Smoothed time-series array z
    """
    if lmda != 0.0:

        # Compute weights for nodata values
        w = 1 - np.array(
            [((x == nodata) or np.isnan(x) or np.isinf(x)) for x in y], dtype=float64
        )
        n = np.sum(w)

        if n > 1:
            z = ws2d(y, lmda, w)
            np.round_(z, 0, out)
        else:
            out[:] = y[:]
    else:
        out[:] = y[:]

"""Whittaker smoother with asymmetric smoothing and fixed lambda (S)."""
import numpy as np
from numba import guvectorize
from numba.core.types import float64, int16

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64, int16[:])],
        "(n),(),(),() -> (n)",
        nopython=True,
    )
)
def ws2dpgu(y, lmda, nodata, p, out):
    """
    Whittaker smoother with asymmetric smoothing and fixed lambda (S).

    Args:
        y: time-series numpy array
        l: smoothing parameter lambda (S)
        w: weights numpy array
        p: "Envelope" value
    Returns:
        Smoothed time-series array z
    """
    if lmda != 0.0:

        m = y.shape[0]

        # Compute weights for nodata values
        w = 1 - np.array(
            [((x == nodata) or np.isnan(x) or np.isinf(x)) for x in y], dtype=float64
        )
        n = np.sum(w)

        if n > 1:
            p1 = 1 - p
            z = np.zeros(m)
            znew = np.zeros(m)
            wa = np.zeros(m)

            # Calculate weights
            for _ in range(10):
                envelope = y > z
                wa[envelope] = p
                wa[~envelope] = p1
                ww = w * wa

                znew[:] = ws2d(y, lmda, ww)

                z_tmp = np.sum(np.abs(znew - z))
                if z_tmp == 0.0:
                    break

                z[:] = znew[:]

            z = ws2d(y, lmda, ww)
            np.round_(z, 0, out)

        else:
            out[:] = y[:]
    else:
        out[:] = y[:]

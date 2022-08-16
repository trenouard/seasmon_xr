"""Whittaker filter V-curve optimization os S."""

import numpy as np
from numba import guvectorize
from numba.core.types import float64, int16, boolean

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64[:], boolean, int16[:], float64[:])],
        "(n),(),(m),() -> (n),()",
        nopython=True,
    )
)
def ws2dwcv(y, nodata, llas, robust, out, lopt):
    """
    Whittaker filter GCV optimization of S.

    Args:
        y (np.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        llas (np.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]
    w = np.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if (y[ii] == nodata) or np.isnan(y[ii]) or np.isinf(y[ii]):
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    # Eigenvalues
    d_eigs = -2 + 2 * np.cos(np.arange(m) * np.pi / m)
    d_eigs[0] = 1e-15

    if n > 5:

        z = np.zeros(m)
        r_weights = np.ones(m)

        # Setting number of robust iterations to perform
        if not robust:
            r_its = 1
        else:
            r_its = 4

        # Initialising list for writing to
        robust_gcv = []

        gcv_temp = [1e15, 0]
        for it in range(r_its):
            if it > 1:
                s_opt_val = robust_gcv[1][1]
            else:
                s_opt_val = 0.0

            if not s_opt_val:
                smoothing = 10**llas
            else:
                smoothing = np.array([s_opt_val])

            w_temp = w * r_weights
            for s in smoothing:

                y_smoothed = ws2d(y, s, w_temp)

                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                tr_H = gamma.sum()
                wsse = (((w_temp**0.5) * (y - y_smoothed)) ** 2).sum()
                denominator = w_temp.sum() * (1 - (tr_H / (w_temp.sum()))) ** 2
                gcv_score = wsse / denominator

                gcv = [gcv_score, s]

                if gcv[0] < gcv_temp[0]:
                    gcv_temp = gcv
                    y_temp = y_smoothed

            best_gcv = gcv_temp
            s = best_gcv[1]

            if robust:
                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                r_arr = y - y_temp

                mad = np.median(
                    np.abs(
                        r_arr[r_weights != 0] - np.median(r_arr[r_weights != 0])
                    )
                )
                u_arr = r_arr / (1.4826 * mad * np.sqrt(1 - gamma.sum() / n))

                r_weights = (1 - (u_arr / 4.685) ** 2) ** 2
                r_weights[(np.abs(u_arr / 4.685) > 1)] = 0

                r_weights[r_arr > 0] = 1

            robust_weights = w * r_weights

            robust_gcv.append(best_gcv)

        robust_gcv = np.array(robust_gcv)

        if robust:
            lopt[0] = robust_gcv[1, 1]
        else:
            lopt[0] = robust_gcv[0, 1]

        z[:] = 0.0
        z = ws2d(y, lopt[0], robust_weights)
        np.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0

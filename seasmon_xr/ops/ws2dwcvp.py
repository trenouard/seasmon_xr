"""Whittaker filter V-curve optimization of S and asymmetric weights."""
# pyright: reportGeneralTypeIssues=false
import numpy as np
from numba import guvectorize, jit
from numba.core.types import float64, int16, boolean

from ._helper import lazycompile
from .ws2d import ws2d


@lazycompile(
    guvectorize(
        [(float64[:], float64, float64, float64[:], boolean, int16[:], float64[:])],
        "(n),(),(),(m),() -> (n),()",
        nopython=True,
    )
)
def ws2dwcvp(y, nodata, p, llas, robust, out, lopt):
    """
    Whittaker filter Generalized Cross Validation optimization of lambda and asymmetric weights.

    Whittaker Cross Validation (WCV)
    The Whittaker Smoother is a penalized least square algorithm for smoothing and interpolation
    of noisy data. The smoothing coefficient optimization allows to automate the right amount of
    penalty.
    References:
    - Eilers, A perfect smoother, https://doi.org/10.1021/ac034173t
    - Eilers, Pesendorfer and Bonifacio, Automatic smoothing of remote sensing data,
      https://doi.org/10.1109/Multi-Temp.2017.8076705
    - Garcia, Robust smoothing of gridded data in one and higher dimensions with missing values,
      https://doi.org/10.1016/j.csda.2009.09.020

    Args:
        y (np.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        llas (np.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]

    # Compute weights for nodata values
    w = 1 - np.array(
        [((x == nodata) or np.isnan(x) or np.isinf(x)) for x in y], dtype=float64
    )
    n = np.sum(w)

    # Eigenvalues
    d_eigs = -2 + 2 * np.cos(np.arange(m) * np.pi / m)
    d_eigs[0] = 1e-15

    if n > 4:

        z = np.zeros(m)
        znew = np.zeros(m)
        wa = np.zeros(m)
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
                lambda_range = np.array([robust_gcv[1][1]])
            else:
                lambda_range = 10**llas

            w_temp = w * r_weights
            for s in lambda_range:

                z = ws2d(y, s, w_temp)

                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                tr_H = gamma.sum()
                wsse = (((w_temp**0.5) * (y - z)) ** 2).sum()
                denominator = w_temp.sum() * (1 - (tr_H / (w_temp.sum()))) ** 2
                gcv_score = wsse / denominator

                gcv = [gcv_score, s]

                if gcv[0] < gcv_temp[0]:
                    gcv_temp = gcv
                    y_temp = z

            best_gcv = gcv_temp
            s = best_gcv[1]

            if robust:
                gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
                r_arr = y - y_temp

                mad = np.median(
                    np.abs(r_arr[r_weights != 0] - np.median(r_arr[r_weights != 0]))
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

        for _ in range(10):

            envelope = y > z
            wa[envelope] = p
            wa[~envelope] = 1 - p
            ww = robust_weights * wa

            znew[0:m] = ws2d(y, lopt[0], ww)

            z_tmp = np.sum(np.abs(znew - z))
            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        np.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@jit(nopython=True)
def _ws2dwcvp(y, w, p, llas, robust):
    """
    Whittaker filter Generalized Cross Validation optimization of lambda and asymmetric weights.

    Whittaker Cross Validation (WCV)
    The Whittaker Smoother is a penalized least square algorithm for smoothing and interpolation
    of noisy data. The smoothing coefficient optimization allows to automate the right amount of
    penalty.
    References:
    - Eilers, A perfect smoother, https://doi.org/10.1021/ac034173t
    - Eilers, Pesendorfer and Bonifacio, Automatic smoothing of remote sensing data,
      https://doi.org/10.1109/Multi-Temp.2017.8076705
    - Garcia, Robust smoothing of gridded data in one and higher dimensions with missing values,
      https://doi.org/10.1016/j.csda.2009.09.020

    Args:
        y (np.array): raw data array (1d, expected in float64)
        w (np.array): weights same size as y
        p (float): Envelope value for asymmetric weights
        llas (np.array): 1d array of s values to use for optimization
        robust (boolean): performs a robust fitting by computing robust weights if True
    """
    m = y.shape[0]
    n = w.sum()

    # Eigenvalues
    d_eigs = -2 + 2 * np.cos(np.arange(m) * np.pi / m)
    d_eigs[0] = 1e-15

    z = np.zeros(m)
    znew = np.zeros(m)
    wa = np.zeros(m)
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
            lambda_range = np.array([robust_gcv[1][1]])
        else:
            lambda_range = 10**llas

        w_temp = w * r_weights
        for s in lambda_range:

            z = ws2d(y, s, w_temp)

            gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
            tr_H = gamma.sum()
            wsse = (((w_temp**0.5) * (y - z)) ** 2).sum()
            denominator = w_temp.sum() * (1 - (tr_H / (w_temp.sum()))) ** 2
            gcv_score = wsse / denominator

            gcv = [gcv_score, s]

            if gcv[0] < gcv_temp[0]:
                gcv_temp = gcv
                y_temp = z

        best_gcv = gcv_temp
        s = best_gcv[1]

        if robust:
            gamma = w_temp / (w_temp + s * ((-1 * d_eigs) ** 2))
            r_arr = y - y_temp

            mad = np.median(
                np.abs(r_arr[r_weights != 0] - np.median(r_arr[r_weights != 0]))
            )
            u_arr = r_arr / (1.4826 * mad * np.sqrt(1 - gamma.sum() / n))

            r_weights = (1 - (u_arr / 4.685) ** 2) ** 2
            r_weights[(np.abs(u_arr / 4.685) > 1)] = 0

            r_weights[r_arr > 0] = 1

        robust_weights = w * r_weights

        robust_gcv.append(best_gcv)

    robust_gcv = np.array(robust_gcv)

    if robust:
        lopt = robust_gcv[1, 1]
    else:
        lopt = robust_gcv[0, 1]

    z[:] = 0.0

    for _ in range(10):
        envelope = y > z
        wa[envelope] = p
        wa[~envelope] = 1 - p
        ww = robust_weights * wa

        znew[0:m] = ws2d(y, lopt, ww)

        z_tmp = np.sum(np.abs(znew - z))
        if z_tmp == 0.0:
            break

        z[0:m] = znew[0:m]

    z = ws2d(y, lopt, ww)
    z = np.round_(z, 0)

    return z, lopt

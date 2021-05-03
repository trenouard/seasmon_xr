"""Tooling to calculate Standardized Precipitation Index (SPI)"""
from math import log, sqrt

import numba
import numpy as np
import numba_scipy # pylint: disable=unused-import
import scipy.special as sc

@numba.njit
def brentq(xa, xb, s):
    """Root finding optimization using Brent's method

    adapted from:

    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/optimize/Zeros/brentq.c#L37
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.

    """

    xpre = xa
    xcur = xb
    xblk = 0.
    fblk = 0.
    spre = 0.
    scur = 0.
    maxiter = 100
    xtol = 2e-12
    rtol= 8.881784197001252e-16

    func = lambda a: log(a) - sc.digamma(a) - s

    fpre = func(xpre)
    fcur = func(xcur)


    if (fpre*fcur) > 0:
        return 0.0

    if fpre == 0:
        return xpre

    if fcur == 0:
        return xcur

    iterations = 0

    for _ in range(maxiter):
        iterations += 1

        if (fpre * fcur) < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) /2

        if fcur == 0 or (abs(sbis) < delta):
            return xcur

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre)/(dblk * dpre * (fblk - fpre))

            if (2 * abs(stry)) < min(abs(spre), 3 * abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis

        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur

        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = func(xcur)

    return xcur

@numba.njit
def spifun(x):
    """Calculate SPI with gamma distribution for 3d array

    Fitting gamma distribution adapted from:
    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/stats/_continuous_distns.py#L2554
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.
    """
    t,r,c = x.shape

    for ri in range(r):
        for ci in range(c):

            xt = x[:, ri, ci]
            valid_ix = []

            xts = 0

            for tix in range(t):
                if xt[tix] > 0:
                    xts += xt[tix]
                    valid_ix.append(tix)

            n_valid = len(valid_ix)

            p_zero = 1 - (n_valid / t)

            if p_zero > 0.9:
                x[:, ri, ci] = -9999
                continue

            xtsbar = xts / n_valid

            logs = 0

            for tix in valid_ix:
                logs+= log(xt[tix])

            s = log(xtsbar) - (logs / n_valid)

            if s == 0:
                x[:, ri, ci] = -9999
                continue

            a_est = (3-s + sqrt((s-3)**2 + 24*s)) / (12*s)
            xa = a_est * (1 - 0.4)
            xb = a_est * (1 + 0.4)
            a = brentq(xa, xb, s)

            if a == 0:
                x[:, ri, ci] = -9999
                continue

            scale = xtsbar / a

            spi = np.full(t, p_zero, dtype=numba.float64)

            for tix in valid_ix:
                spi[tix] = p_zero + ((1-p_zero) * sc.gammainc(a, xt[tix]/scale))

            for tix in range(t):
                spi[tix] = (sc.ndtri(spi[tix]) * 1000)

            np.round_(spi, 0, spi)

            x[:,ri, ci] = spi[:]

    return x

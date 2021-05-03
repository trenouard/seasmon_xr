from numba import jit
from numpy import zeros

from ._helper import lazycompile

@lazycompile(jit(nopython=True))
def autocorr(x):
    """Calculates Lag-1 autocorrelation.
    Adapted from https://stackoverflow.com/a/29194624/5997555
    Args:
        x: 3d data array
    Returns:
        Lag-1 autocorrelation array
    """
    r, c, t = x.shape
    z = zeros((r, c), dtype="float32")

    M = t-1

    for rr in range(r):
        for cc in range(c):

            data1 = x[rr, cc, 1:]
            data2 = x[rr, cc, :-1]

            sum1 = 0.
            sum2 = 0.
            for i in range(M):
                sum1 += data1[i]
                sum2 += data2[i]

                mean1 = sum1 / M
                mean2 = sum2 / M

            var_sum1 = 0.
            var_sum2 = 0.
            cross_sum = 0.
            for i in range(M):
                var_sum1 += (data1[i] - mean1) ** 2
                var_sum2 += (data2[i] - mean2) ** 2
                cross_sum += (data1[i] * data2[i])

            std1 = (var_sum1 / M) ** .5
            std2 = (var_sum2 / M) ** .5
            cross_mean = cross_sum / M

            if std1 != 0 and std2 != 0:

                lc = (cross_mean - mean1 * mean2) / (std1 * std2)
            else:
                lc = 0.0
            z[rr, cc] = lc
    return z
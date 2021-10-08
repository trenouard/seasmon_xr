from numba import jit, float64, float32
from numpy import zeros

from ._helper import lazycompile


@lazycompile(jit(nopython=True))
def autocorr(x):
    """Calculates Lag-1 autocorrelation.
    Adapted from https://stackoverflow.com/a/29194624/5997555
    Args:
        x: 3d data array (Y,X,T)
    Returns:
        Lag-1 autocorrelation array (Y,X)
    """
    r, c, t = x.shape
    z = zeros((r, c), dtype="float32")

    M = t - 1

    for rr in range(r):
        for cc in range(c):

            data1 = x[rr, cc, 1:]
            data2 = x[rr, cc, :-1]

            sum1 = 0.0
            sum2 = 0.0
            for i in range(M):
                sum1 += data1[i]
                sum2 += data2[i]

                mean1 = sum1 / M
                mean2 = sum2 / M

            var_sum1 = 0.0
            var_sum2 = 0.0
            cross_sum = 0.0
            for i in range(M):
                var_sum1 += (data1[i] - mean1) ** 2
                var_sum2 += (data2[i] - mean2) ** 2
                cross_sum += data1[i] * data2[i]

            std1 = (var_sum1 / M) ** 0.5
            std2 = (var_sum2 / M) ** 0.5
            cross_mean = cross_sum / M

            if std1 != 0 and std2 != 0:

                lc = (cross_mean - mean1 * mean2) / (std1 * std2)
            else:
                lc = 0.0
            z[rr, cc] = lc
    return z


@jit(nopython=True)
def autocorr_1d(data):
    assert data.ndim == 1
    M = data.shape[0] - 1
    xx = data[1:]
    yy = data[:-1]

    # Sum((Xi - Sum(Xi)/M)^2) = Sum(Xi^2) - (Sum(Xi)^2)/M
    # This allows for single pass over data computation
    Sx = float64(0)
    Sy = float64(0)
    Sxx = float64(0)
    Syy = float64(0)
    Sxy = float64(0)
    for i in range(M):
        y = float64(yy[i])
        x = float64(xx[i])
        Sx += x
        Sy += y
        Sxx += x * x
        Syy += y * y
        Sxy += x * y

    # Sxx_ = Sum((Xi - Mean)^2)  # Sxx'
    Sxx_ = Sxx - Sx * Sx / M
    Syy_ = Syy - Sy * Sy / M

    #  (Sxy/M) - (Sx/M)*(Sy/M)
    # -----------------------------
    # sqrt(Sxx'/M) * sqrt(Syy'/M)
    #

    # Since M > 0, Sxx'>0, Syy' >0
    #
    #    1/(sqrt(Sxx'/M) * sqrt(Syy'/M))
    # => M/sqrt(Sxx'*Syy')
    # => M*((Sxx'*Syy')^-0.5)
    # => M*Z  where Z = ((Sxx'*Syy')^-0.5)
    #
    #    (Sxx/M - (Sx/M)*(Sy/M)*M*Z
    # => (Sxx - Sx*Sy/M)*Z
    #
    # this is mostly for numeric stability, but it helps
    # with performance too as we remove a bunch of divides and
    # replace 2 sqrt + 2 div ops with one 1/sqrt op
    var_sum = Sxx_ * Syy_
    result = float64(0)

    if var_sum >= 1e-8:
        result = (Sxy - Sx * Sy / M) * (var_sum ** (-0.5))

    return result


@lazycompile(jit(nopython=True))
def autocorr_tyx(tyx):
    """Calculates Lag-1 autocorrelation.
    Adapted from https://stackoverflow.com/a/29194624/5997555
    Args:
        tyx: 3d data array in T,Y,X order
    Returns:
        Lag-1 autocorrelation array Y,X order
    """
    _, nr, nc = tyx.shape
    z = zeros((nr, nc), dtype="float32")

    for rr in range(nr):
        for cc in range(nc):
            data = tyx[:, rr, cc]
            z[rr, cc] = autocorr_1d(data)

    return z

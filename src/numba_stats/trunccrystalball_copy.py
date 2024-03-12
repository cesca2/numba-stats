import numba as nb
import numpy as np
from .crystalball_copy import _pdf as _cb_pdf, _cdf

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.njit(cache=True)
def _pdf(z, zmin, zmax, beta, m):
    if z < zmin or z > zmax:
        return 0.0
    return _cb_pdf(z, beta, m) / (_cdf(zmax, beta, m) - _cdf(zmin, beta, m))


@nb.vectorize(_signatures, cache=True)
def pdf(x, xmin, xmax, beta, m, loc, scale):
    if x < xmin or x > xmax:
        return 0
    if beta < 0:
        beta *= -1
        x = xmax - x + xmin
        loc = xmax - loc + xmin
    z = (x - loc) / scale
    zmin = (xmin - loc) / scale
    zmax = (xmax - loc) / scale
    return _pdf(z, zmin, zmax, beta, m) / scale


@nb.vectorize(_signatures, cache=True)
def cdf(x, xmin, xmax, beta, m, loc, scale):
    if x < xmin:
        return 0.0
    elif x > xmax:
        return 1.0
    rside=False
    if beta < 0:
        rside = True
        beta *= -1
        x = xmax - x + xmin
        loc = xmax - loc + xmin
    z = (x - loc) / scale
    zmin = (xmin - loc) / scale
    zmax = (xmax - loc) / scale
    pmin = _cdf(zmin, beta, m)
    pmax = _cdf(zmax, beta, m)
    ret = (_cdf(z, beta, m) - pmin) / (pmax - pmin)
    if rside:
        return 1-ret
    else:
        return ret

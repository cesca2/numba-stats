import numba as nb
import numpy as np
from .crystalball_copy import pdf as _cb_pdf, cdf as _cb_cdf
from .trunccrystalball_copy import pdf as _tcb_pdf, cdf as _tcb_cdf

_signatures = [
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32),
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
]


@nb.vectorize(_signatures, cache=True)
def pdf(x, xmin, xmax, f, betal, ml, betar, mr, locl, scalel, locr, scaler):
    if x < xmin or x > xmax:
        return 0
    cbl = _tcb_pdf(x, xmin, xmax, betal, ml, locl, scalel)
    invx = xmax - x + xmin
    invr = xmax - locr + xmin
    cbr = _tcb_pdf(invx, xmin, xmax, betar, mr, invr, scaler)
    return f*cbl + (1-f)*cbr


@nb.vectorize(_signatures, cache=True)
def cdf(x, xmin, xmax, f, betal, ml, betar, mr, locl, scalel, locr, scaler):
    if x < xmin:
        return 0
    elif x > xmax:
        return 1
    cbl = _tcb_cdf(x, xmin, xmax, betal, ml, locl, scalel)
    invx = xmax - x + xmin
    invr = xmax - locr + xmin
    cbr = 1-_tcb_cdf(invx, xmin, xmax, betar, mr, invr, scaler)
    return f*cbl + (1-f)*cbr

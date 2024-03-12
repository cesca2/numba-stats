"""
Johnson's S_U PDF

See Also
--------
https://root.cern/doc/master/classRooJohnson.html: RooFit equivalent
"""
import numpy as np
from ._util import _jit, _generate_wrappers, _prange

_doc_par = """
x : ArrayLike
    Random variate.
mu : float
    Centre of Gaussian component.
lb : float
    Width of Gaussian component.
    Must be greater than 0.
gamma : float
    Distorts distribution to left or right.
delta : float
    Strength of Gaussian component.
    Must be greater than 0.
massThreshold : float
    PDF set to 0 below (to the left of) this parameter.
"""


@_jit(2)
def _trans_arg(x, mu, lb):
    return (x-mu)/lb

@_jit(4)
def _trans_expo(x, mu, lb, gamma, delta):
    arg = _trans_arg(x, mu, lb)
    return gamma + delta * np.arcsinh(arg)

@_jit(5)
def _pdf(x, mu, lb, gamma, delta, massThreshold):
    out = np.empty_like(x)
    for i in _prange(len(x)):
        x_i = x[i]
        if x_i< massThreshold:
            out[i] = 0.0
        else:
            arg = _trans_arg(x_i, mu, lb)
            expo = _trans_expo(x_i, mu, lb, gamma, delta)
            out[i] = delta / np.sqrt(2*np.pi) / (lb * np.sqrt(1 + arg * arg)) * np.exp(-0.5 * expo * expo)
    
    return out

_generate_wrappers(globals())

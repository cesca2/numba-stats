"""
Hypatia distribution.

Hypatia distribution uses a generalised hyperbolic core with power law tails
on both sides.

https://arxiv.org/abs/1312.5000

There is no analytic integral available.

This version is ported from the ROOT RooHypatia2 implementation.

See here: https://root.cern.ch/doc/master/classRooHypatia2.html
"""
from ._util import _jit, _generate_wrappers, _prange, _trans
from numba import njit
import numba as nb
from ._special import gamma, kv
import numpy as np
from math import erf as _erf

_doc_par = """
x : Array-like
    Random variate.
lb : float
    Lambda parameter. Roughly the concavity of the hyperbolic core.
    Usually negative.
zeta : float
    Usually set to a very small value e.g. 1e-5.
    Must be > 0.
    Note: One requires beta^2 < zeta^2/(sigma^2 * A_lambda^2(zeta))
    where A_lambda^2(zeta) = zeta*K_lambda(zeta)/K_(lambda+1)(zeta)
    with K_lambda(zeta) the modified Bessel function of the second kind
    to prevent the tails from raising.
beta : float
    Related to the asymmetry of the hyperbolic core.
    Usually set to 0.
    Note: One requires beta^2 < zeta^2/(sigma^2 * A_lambda^2(zeta))
    to prevent the tails from raising.
sigma : float
    Width of the hyperbolic core.
    Note: One requires beta^2 < zeta^2/(sigma^2 * A_lambda^2(zeta))
    to prevent the tails from raising.
mu : float
    Centre of the distribution.
al : float
    Distance from mu in units of the standard deviation where the
    powerlaw tail switches on for the left (lower) end of distribution.
nl : float
    Absolute value of the slope of the left (lower) powerlaw tail.
    Must be larger than 1.
ar : float
    Distance from mu in units of the standard deviation where the
    powerlaw tail switches on for the right (upper) end of distribution.
nr : float
    Absolute value of the slope of the right (upper) powerlaw tail.
    Must be larger than 1.
"""

def _jit_nc(arg, cache=False):
    return _jit(arg, cache)

@_jit_nc(-2)
def _besselK(x, ni):
    return kv(ni, x)

@_jit_nc(-2)
def _LnBesselK(x, ni):
    return np.log(_besselK(x, ni))

@_jit_nc(-5)
def _LogEval(d, l, alpha, beta, delta):
    gamma = alpha
    dg = delta*gamma
    #This variable is really called "thing" in the RooFit code...
    thing = delta*delta + d*d
    logno = l*np.log(gamma/delta) - np.log(np.sqrt(2*np.pi)) - _LnBesselK(dg, l)

    return np.exp(logno + beta*d + (0.5-l)*(np.log(alpha)-0.5*np.log(thing)) + _LnBesselK(alpha*np.sqrt(thing), l-0.5))

@_jit_nc(-5)
def _diff_eval(d, l, alpha, beta, delta):
    gamma = alpha
    dg = delta*gamma
    thing = delta*delta + d*d
    sqrthing = np.sqrt(thing)
    alphasq = alpha*sqrthing
    no = np.power(gamma/delta, l)/_besselK(dg, l)*(1./np.sqrt(2*np.pi))
    ns1 = 0.5-l

    return no*np.power(alpha, ns1)*np.power(thing, 0.5*l-1.25) * (-d*alphasq*(_besselK(alphasq, l-1.5) + _besselK(alphasq, l+0.5)) + (2*(beta*thing+d*l)-d) * _besselK(alphasq, ns1)) * np.exp(beta*d) * 0.5

#Not in the RooFit version, define some functions to handle the if/else statements for numpy
@_jit_nc(-7)
def _zeta0al(d, alsigma, lb, alpha, beta, delta, nl):
    k1 = _LogEval(-alsigma, lb, alpha, beta, delta)
    k2 = _diff_eval(-alsigma, lb, alpha, beta, delta)
    B = -alsigma + nl*k1/k2
    A = k1*np.power(B+alsigma, nl)

    return A * np.power(B-d, -nl)

@_jit_nc(-7)
def _zeta0ar(d, arsigma, lb, alpha, beta, delta, nr):
    k1 = _LogEval(arsigma, lb, alpha, beta, delta)
    k2 = _diff_eval(arsigma, lb, alpha, beta, delta)
    B = -arsigma - nr*k1/k2
    A = k1*np.power(B+arsigma, nr)

    return A * np.power(B+d, -nr)

@_jit_nc(-7)
def _lb0al(d, beta, alsigma, al, lb, delta, nl):
    cons1 = np.exp(-beta*alsigma)
    phi = 1. + al*al
    k1 = cons1 * np.power(phi, lb-0.5)
    k2 = beta*k1 - cons1*(lb-0.5) * np.power(phi, lb-1.5) * 2*al/delta
    B = -alsigma + nl*k1/k2
    A = k1*np.power(B+alsigma, nl)

    return A*np.power(B-d, -nl)

@_jit_nc(-7)
def _lb0ar(d, beta, arsigma, ar, lb, delta, nr):
    cons1 = np.exp(beta*arsigma)
    phi = 1. + ar*ar
    k1 = cons1 * np.power(phi, lb-0.5)
    k2 = beta*k1 - cons1*(lb-0.5) * np.power(phi, lb-1.5) * 2*ar/delta
    B = -arsigma - nr*k1/k2
    A = k1*np.power(B+arsigma, nr)

    return A*np.power(B+d, -nr)

@_jit_nc(-4)
def _lb0Other(d, beta, delta, lb):

    return np.exp(beta*d) * np.power(1 + d*d/(delta*delta), lb - 0.5)

@_jit_nc(9)
def _pdf(x, lb, zeta, beta, sigma, mu, al, nl, ar, nr):
    d = _trans(x, mu, 1.)
    out = np.empty_like(x)

    cons0 = np.sqrt(zeta)
    alsigma = al*sigma
    arsigma = ar*sigma

    if zeta > 0.:
        phi = _besselK(zeta, lb+1)/_besselK(zeta, lb)
        cons1 = sigma/np.sqrt(phi)
        alpha = cons0/cons1
        delta = cons0*cons1

        for i in _prange(len(x)):
            d_i = d[i]
            if d_i < -alsigma:
                out[i] = _zeta0al(d_i, alsigma, lb, alpha, beta, delta, nl)
            elif d_i > arsigma:
                out[i] = _zeta0ar(d_i, arsigma, lb, alpha, beta, delta, nr)
            elif d_i >=-alsigma and d_i <= arsigma:
                out[i] = _LogEval(d_i, lb, alpha, beta, delta)

    elif zeta < 0.:
        print('Zeta cannot be < 0.')
    elif lb < 0.:
        delta = sigma

        for i in _prange(len(x)):
            d_i = d[i]
            if d_i < -alsigma:
                out[i] = _lb0al(d_i, beta, alsigma, al, lb, delta, nl) 
            elif d_i > arsigma:
                out[i] = _lb0ar(d_i, beta, arsigma, ar, lb, delta, nr)
            elif d_i >=-alsigma and d_i <= arsigma:
                out[i] = _lb0Other(d_i, beta, delta, lb)

    else:
        print(f'Zeta = 0 only supported for lb < 0.')

    return out


_generate_wrappers(globals())

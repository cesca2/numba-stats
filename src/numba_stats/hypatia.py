"""
Hypatia distribution.

Hypatia distribution uses a generalised hyperbolic core with power law tails
on both sides.

https://arxiv.org/abs/1312.5000

There is no analytic integral available.

This version is ported from the ROOT RooHypatia2 implementation.

See here: https://root.cern.ch/doc/master/classRooHypatia2.html
"""
from ._util import _jit, _generate_wrappers, _prange
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

#Used for Bessel fcn when x is small
@_jit(-2)
def _low_x_BK(nu, x):

    return gamma(nu)*np.power(2, nu-1)*np.power(x, -nu)

#Used for Bessel fcn when x is small
@_jit(-2)
def _low_x_LnBK(nu, x):

    return np.log(gamma(nu)) + np.log(2)*(nu-1) - np.log(x)*nu

#Bessel functions
@_jit(-2)
def _besselK(ni, x):
    nu = np.abs(ni)
    #Looks a bit different to get the if conditions to work with numpy
    lowest_mask = x<1.e-6
    middle_mask = np.intersect1d(np.where(x>=1.e-6), np.where(x<1.e-4))
    upper_mask = np.intersect1d(np.where(x>=1.e-4), np.where(x<0.1))
    other_mask = x>=1.e-4
    to_return = np.zeros(x.shape)

    to_return[lowest_mask] = _low_x_BK(nu, x[lowest_mask]) if nu>0. else kv(nu, x[lowest_mask])
    to_return[middle_mask] = _low_x_BK(nu, x[middle_mask]) if (nu>0. and nu<55.) else kv(nu, x[middle_mask])
    to_return[upper_mask] = _low_x_BK(nu, x[upper_mask]) if nu>=55. else kv(nu, x[upper_mask])
    to_return[other_mask] = kv(nu, x[other_mask])

    return to_return

#Log Bessel fcn
@_jit(-2)
def _LnBesselK(ni, x):
    nu = np.abs(ni)
    lowest_mask = x<1.e-6
    middle_mask = np.intersect1d(np.where(x>=1.e-6), np.where(x<1.e-4))
    upper_mask = np.intersect1d(np.where(x>=1.e-4), np.where(x<0.1))
    other_mask = x>=1.e-4
    to_return = np.zeros(x.shape)

    to_return[lowest_mask] = _low_x_LnBK(nu, x[lowest_mask]) if nu>0. else np.log(kv(nu, x[lowest_mask]))
    to_return[middle_mask] = _low_x_LnBK(nu, x[middle_mask]) if (nu>0. and nu<55.) else np.log(kv(nu, x[middle_mask]))
    to_return[upper_mask] = _low_x_LnBK(nu, x[upper_mask]) if nu>=55. else np.log(kv(nu, x[upper_mask]))
    to_return[other_mask] = np.log(kv(nu, x[other_mask]))

    return to_return

@_jit(-5)
def _LogEval(d, l, alpha, beta, delta):
    gamma = alpha
    dg = delta*gamma
    #This variable is really called "thing" in the RooFit code...
    thing = delta*delta + d*d
    logno = l*np.log(gamma/delta) - np.log(np.sqrt(2*np.pi)) - _LnBesselK(l, dg)

    return np.exp(logno + beta*d + (0.5-l)*(np.log(alpha)-0.5*np.log(thing)) + _LnBesselK(l-0.5, alpha*np.sqrt(thing)))

@_jit(-5)
def _diff_eval(d, l, alpha, beta, delta):
    gamma = alpha
    dg = delta*gamma
    thing = delta*delta + d*d
    sqrthing = np.sqrt(thing)
    alphasq = alpha*sqrthing
    no = np.power(gamma/delta, l)/_besselK(l, dg)*(1./np.sqrt(2*np.pi))
    ns1 = 0.5-l

    return no*np.power(alpha, ns1)*np.power(thing, 0.5*l-1.25) * (-d*alphasq*(_besselK(l-1.5, alphasq) + _besselK(l+0.5, alphasq)) + (2*(beta*thing+d*l)-d) * _besselK(ns1, alphasq)) * np.exp(beta*d) * 0.5

#Not in the RooFit version, define some functions to handle the if/else statements for numpy
@_jit(-7)
def _zeta0al(d, alsigma, lb, alpha, beta, delta, nl):
    k1 = _LogEval(-alsigma, lb, alpha, beta, delta)
    k2 = _diff_eval(-alsigma, lb, alpha, beta, delta)
    B = -alsigma + nl*k1/k2
    A = k1*np.power(B+alsigma, nl)

    return A * np.power(B-d, -nl)

@_jit(-7)
def _zeta0ar(d, arsigma, lb, alpha, beta, delta, nr):
    k1 = _LogEval(arsigma, lb, alpha, beta, delta)
    k2 = _diff_eval(arsigma, lb, alpha, beta, delta)
    B = -arsigma - nr*k1/k2
    A = k1*np.power(B+arsigma, nr)

    return A * np.power(B+d, -nr)

@_jit(-7)
def _lb0al(beta, alsigma, al, lb, delta, nl, d):
    cons1 = np.exp(-beta*alsigma)
    phi = 1. + a*a
    k1 = cons1 * np.power(phi, lb-0.5)
    k2 = beta*k1 - cons1*(lb-0.5) * np.power(phi, lb-1.5) * 2*a/delta
    B = -alsigma + nl*k1/k2
    A = k1*np.power(B+alsigma, nl)

    return A*np.power(B-d, -nl)

@_jit(-7)
def _lb0ar(beta, arsigma, ar, lb, delta, nr, d):
    cons1 = np.exp(beta*arsigma)
    phi = 1. + ar*ar
    k1 = cons1 * np.power(phi, lb-0.5)
    k2 = beta*k1 - cons1*(lb-0.5) * np.power(phi, lb-1.5) * 2*ar/delta
    B = -arsigma - nr*k1/k2
    A = k1*np.power(B+arsigma, nr)

    return A*np.power(B+d, -nr)

@_jit(-4)
def _lb0Other(beta, d, delta, lb):

    return np.exp(beta*d) * np.power(1 + d*d/(delta*delta), lb - 0.5)

@_jit(10)
def _pdf(x, lb, zeta, beta, sigma, mu, al, nl, ar, nr):
    d = x - mu
    out = np.zeros(x.shape)

    cons0 = np.sqrt(zeta)
    alsigma = al*sigma
    arsigma = ar*sigma
    #Define some masks for the different regions of the Hypatia function
    al_mask = d < -alsigma
    ar_mask = d > arsigma
    other_mask = np.intersect1d(np.where(d>=-alsigma), np.where(d<=arsigma))

    if zeta > 0.:
        phi = _besselK(lb+1, zeta)/besselK(lb, zeta)
        cons1 = sigma/np.sqrt(phi)
        alpha = cons0/cons1
        delta = cons0*cons1

        out[al_mask] = _zeta0al(d[al_mask], alsigma, lb, alpha, beta, delta, nl)
        out[ar_mask] = _zeta0ar(d[ar_mask], arsigma, lb, alpha, beta, delta, nr)
        out[other_mask] = _LogEval(d[other_mask], lb, alpha, beta, delta)

    elif zeta < 0.:
        print('Zeta cannot be < 0.')
    elif lb < 0.:
        delta = sigma

        out[al_mask] = _lb0al(beta, alsigma, al, lb, delta, nl, d[al_mask])
        out[ar_mask] = _lb0ar(beta, arsigma, ar, lb, delta, nr, d[ar_mask])
        out[other_mask] = _lb0Other(beta, d[other_mask], delta, lb)

    else:
        print(f'Zeta = 0 only supported for lb < 0. lb = {_lb}')

    return out


_generate_wrappers(globals())

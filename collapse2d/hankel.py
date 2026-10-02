"""Zeroth-order Hankel transforms by Ogata's double-exponential quadrature.

    I(r) = int_0^inf f(q) J0(q r) dq

H. Ogata, Publ. RIMS Kyoto Univ. 41 (2005) 949.  With x = q r the integral is
(1/r) int_0^inf f(x/r) J0(x) dx; the quadrature nodes sit at transformed zeros
of J0 so that the oscillatory integrand is resolved with a few hundred points
for any r.
"""
import numpy as np
from scipy.special import j0, j1, jn_zeros, y0


class OgataJ0:
    def __init__(self, n=600, h=0.004):
        xi = jn_zeros(0, n) / np.pi
        self.w = y0(np.pi * xi) / j1(np.pi * xi)
        t = h * xi
        psi = t * np.tanh(0.5 * np.pi * np.sinh(t))
        dpsi = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (1.0 + np.cosh(np.pi * np.sinh(t)))
        self.x = np.pi * psi / h
        self.weights = np.pi * self.w * j0(self.x) * dpsi

    def __call__(self, f, r):
        """f: vectorised callable of q; r: scalar or array of radii (> 0)."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        q = self.x[None, :] / r[:, None]
        return (f(q) @ self.weights) / r

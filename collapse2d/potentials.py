"""Impurity potentials for a charge +Ze near a gapped 2D Dirac layer.

All potentials are *potential energies of an electron*, in units hbar v = 1
(nm^-1), as functions of the in-plane distance r (nm).

Screened impurity (linear response, static RPA):

    U(q) = -2 pi Z alpha0 e^{-q d} G(q) / [ q eps(q) ],
    eps(q) = kappa + r_K q + 2 pi alpha0 N Pi(q) G(q) / q,
    G(q)   = 1 - exp(-2 q D)            (metallic gate at distance D, G = 1 if D = inf)

with d the height of the charge above the layer and Pi(q) the static interband
polarization of one gapped Dirac flavour,

    Pi(q) = [ m + (q^2 - 4 m^2)/(2q) * arctan(q/2m) ] / (4 pi).

Real space: U(r) = -Z alpha0 * int_0^inf J0(q r) f(q) dq, f = e^{-qd} G / eps.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import j0, jn_zeros

# -------------------------------------------------------------------------
# polarization of a gapped 2D Dirac flavour
# -------------------------------------------------------------------------


def polarization(q, m):
    """Static interband polarization Pi(q) of one massive 2D Dirac flavour (hbar v = 1)."""
    q = np.asarray(q, dtype=float)
    if m == 0:
        return q / 16.0
    x = q / (2.0 * m)
    out = np.empty_like(q)
    small = x < 1e-3
    xs = x[small]
    # series: Pi = (m/4pi)[(4/3)x^2 - (8/15)x^4 + (12/35) x^6]
    out[small] = m / (4 * np.pi) * (4.0 / 3.0 * xs ** 2 - 8.0 / 15.0 * xs ** 4 + 12.0 / 35.0 * xs ** 6)
    qb = q[~small]
    out[~small] = (m + (qb ** 2 - 4 * m * m) / (2 * qb) * np.arctan(qb / (2 * m))) / (4 * np.pi)
    return out


# -------------------------------------------------------------------------
# simple analytic potentials
# -------------------------------------------------------------------------


class CutoffCoulomb:
    """U = -beta/r for r > r0, -beta/r0 inside (Pereira et al. 2008 regularization)."""

    def __init__(self, beta, r0):
        self.beta, self.r0 = float(beta), float(r0)
        self.kinks = (self.r0,)
        self.tail_beta = self.beta
        self.analytic_from = self.r0

    def analytic_tail(self, r):
        return -self.beta / r

    def __call__(self, r):
        return -self.beta / np.maximum(r, self.r0)


class HeightCoulomb:
    """Charge at height d above the layer: U = -beta/sqrt(r^2 + d^2)."""

    def __init__(self, beta, d):
        self.beta, self.d = float(beta), float(d)
        self.kinks = ()
        self.tail_beta = self.beta
        self.analytic_from = 0.0

    def analytic_tail(self, r):
        return -self.beta / np.sqrt(r * r + self.d * self.d)

    def __call__(self, r):
        return -self.beta / np.sqrt(r * r + self.d * self.d)


class PointCoulomb:
    """U = -beta/r (subcritical only)."""
    point = True

    def __init__(self, beta=1.0):
        self.beta = float(beta)
        self.kinks = ()
        self.tail_beta = self.beta

    def __call__(self, r):
        return -self.beta / r


# -------------------------------------------------------------------------
# numerically screened impurity
# -------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_J0_ZEROS = jn_zeros(0, 200000)


def _hankel_gl(f, r, qmax, extra_breaks=()):
    """int_0^qmax J0(q r) f(q) dq by Gauss-Legendre between the zeros of J0(q r)."""
    nz = int(np.searchsorted(_J0_ZEROS, qmax * r))
    br = np.concatenate([[0.0], _J0_ZEROS[:nz] / r, np.asarray(extra_breaks, float), [qmax]])
    br = np.unique(br[(br >= 0) & (br <= qmax)])
    a, b = br[:-1], br[1:]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    q = mid[:, None] + half[:, None] * _GL_X[None, :]
    return float(np.sum(half[:, None] * _GL_W[None, :] * j0(q * r) * f(q)))


class ScreenedImpurity:
    """Linear-response screened impurity potential, tabulated and splined.

    U(r) = -Z alpha0 [ U_inf(r) + int J0(qr) h(q) dq ],  h = f - f_inf,
    f_inf(q) = e^{-qd} G(q) / (kappa_inf + r_K q),  kappa_inf = kappa + pi alpha0 N/8,
    where U_inf (the large-q, i.e. short-distance, part) is evaluated in closed
    form or through the Laplace representation
        int_0^inf J0(qr) e^{-qz}/(k + a q) dq = int_0^inf e^{-k t} / sqrt(r^2 + (z + a t)^2) dt,
    and the smooth remainder h by Gauss-Legendre quadrature between Bessel zeros.
    Beyond the table, U is continued with the asymptotic form A/r + B/r^3.
    """

    def __init__(self, Z, mat, d, D=np.inf, rpa=True, r_min=1e-4, r_max=4000.0, n_r=500):
        self.Z, self.mat, self.d, self.D = float(Z), mat, float(d), float(D)
        self.N = mat.N if rpa else 0
        self.kappa, self.rK, self.alpha0, self.m = mat.kappa, mat.r_K, mat.alpha0, mat.m
        self.kappa_inf = self.kappa + np.pi * self.alpha0 * self.N / 8.0
        self.kinks = ()
        self.range = (2 * self.D + self.d) if np.isfinite(self.D) else np.inf
        rr = np.geomspace(r_min, r_max, n_r)
        self.r_tab = rr
        self.U_tab = np.array([self._U_exact(r) for r in rr])
        self._spline = CubicSpline(np.log(rr), self.U_tab)
        # asymptotic continuation A/r + B/r^3 fitted on the last table points
        r1, r2 = rr[-1], rr[-12]
        M = np.array([[1 / r1, 1 / r1 ** 3], [1 / r2, 1 / r2 ** 3]])
        self._A, self._B = np.linalg.solve(M, [self.U_tab[-1], self.U_tab[-12]])
        # effective Coulomb strength of the tail (exactly zero with a gate: the
        # image charge cancels the monopole; the fit would only return noise)
        self.tail_beta = -self._A if np.isinf(self.D) else 0.0
        if not np.isinf(self.D):
            self._A = 0.0
            self._B = self.U_tab[-1] * r1 ** 3
        self.U0 = self.U_tab[0]
        # beyond this radius A/r + B/r^3 reproduces the table to < 1e-6 (relative)
        rel = np.abs(self._A / rr + self._B / rr ** 3 - self.U_tab) / np.abs(self.U_tab)
        bad = np.nonzero(rel > 1e-6)[0]
        self.analytic_from = rr[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(rr) else rr[0]

    def analytic_tail(self, r):
        return self._A / r + self._B / r ** 3

    # --- dielectric function and kernels ---------------------------------
    def G(self, q):
        return np.ones_like(q) if np.isinf(self.D) else -np.expm1(-2.0 * q * self.D)

    def eps(self, q):
        q = np.asarray(q, dtype=float)
        e = self.kappa + self.rK * q
        if self.N:
            e = e + 2 * np.pi * self.alpha0 * self.N * polarization(q, self.m) / np.where(q > 0, q, 1.0) * self.G(q)
        return e

    def f(self, q):
        return np.exp(-q * self.d) * self.G(q) / self.eps(q)

    def f_inf(self, q):
        return np.exp(-q * self.d) * self.G(q) / (self.kappa_inf + self.rK * q)

    def _laplace_line(self, r, z):
        """int_0^inf J0(qr) e^{-qz}/(kappa_inf + r_K q) dq."""
        if self.rK == 0:
            return 1.0 / (self.kappa_inf * np.hypot(r, z))
        k, a = self.kappa_inf, self.rK
        val, _ = quad(lambda t: np.exp(-k * t) / np.hypot(r, z + a * t), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)
        return val

    def _U_inf(self, r):
        out = self._laplace_line(r, self.d)
        if not np.isinf(self.D):
            out -= self._laplace_line(r, self.d + 2 * self.D)
        return out

    def _qmax(self):
        q = 40.0 / self.d
        scales = []
        if self.N and self.m > 0:
            scales.append(3000.0 * self.m)
        if not np.isinf(self.D):
            scales.append(40.0 / (2 * self.D))
        if not scales:
            return 0.0                     # h vanishes identically
        return min(q, max(scales))

    def _U_exact(self, r):
        total = self._U_inf(r)
        qmax = self._qmax()
        if qmax > 0:
            breaks = np.geomspace(1e-6 * qmax, qmax, 120)
            total += _hankel_gl(lambda q: self.f(q) - self.f_inf(q), r, qmax, breaks)
        return -self.Z * self.alpha0 * total

    # --- evaluation ------------------------------------------------------
    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        out = np.empty_like(r)
        lo = r < self.r_tab[0]
        hi = r > self.r_tab[-1]
        mid = ~(lo | hi)
        out[lo] = self.U0
        out[mid] = self._spline(np.log(r[mid]))
        out[hi] = self._A / r[hi] + self._B / r[hi] ** 3
        return out

    def effective_charge(self, r):
        """Z_eff(r) = -kappa r U(r)/alpha0: the charge seen at distance r through kappa alone."""
        return -self.kappa * r * self(r) / self.alpha0

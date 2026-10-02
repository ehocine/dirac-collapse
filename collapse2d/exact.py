"""Closed-form results for the 2D massive Dirac-Coulomb problem."""
import mpmath as mp
import numpy as np


def dirac_coulomb_levels(beta, j, n_max=4, m=1.0):
    """E_{n j} = m (n + s)/sqrt(beta^2 + (n + s)^2), s = sqrt(j^2 - beta^2) (beta < |j|)."""
    s = np.sqrt(j * j - beta * beta)
    n = np.arange(n_max)
    return m * (n + s) / np.sqrt(beta ** 2 + (n + s) ** 2)


def cutoff_threshold_function(beta, j, m, r0, dps=30):
    """Matching function at E = -m for V = -beta/max(r, r0) (beta > |j|).

    Outside: F = K_{2i gamma}(x), x = sqrt(8 beta m r), gamma = sqrt(beta^2 - j^2),
             G = (j F - r F')/beta.
    Inside:  F = sqrt(r) J_{j-1/2}(p r), G = (p/b) sqrt(r) J_{j+1/2}(p r),
             b = beta/r0, p^2 = b (b - 2m).
    Returns the normalised Wronskian F_in G_out - F_out G_in at r0.
    """
    with mp.workdps(dps):
        beta, m, r0 = mp.mpf(beta), mp.mpf(m), mp.mpf(r0)
        jj = mp.mpf(j)
        gam = mp.sqrt(beta ** 2 - jj ** 2)
        mu = 2j * gam
        x = mp.sqrt(8 * beta * m * r0)
        K = mp.besselk(mu, x)
        dK = -mp.besselk(mu - 1, x) - mu / x * K
        Fo = mp.re(K)
        Go = mp.re(jj * K - x * dK / 2) / beta
        b = beta / r0
        p = mp.sqrt(b * (b - 2 * m))
        mm = int(round(float(jj - mp.mpf(1) / 2)))
        Fi = mp.sqrt(r0) * mp.besselj(mm, p * r0)
        Gi = p / b * mp.sqrt(r0) * mp.besselj(mm + 1, p * r0)
        if mp.im(p) != 0:
            # p = i q (forbidden core): J_n(i x) = i^n I_n(x); remove the common phase i^mm
            ph = mp.power(1j, -mm)
            Fi, Gi = mp.re(Fi * ph), mp.re(Gi * ph)
        return float((Fi * Go - Fo * Gi) / (mp.sqrt(Fi ** 2 + Gi ** 2) * mp.sqrt(Fo ** 2 + Go ** 2)))


def cutoff_critical_coupling(j, m, r0, lo=None, hi=3.0, n_scan=400):
    """All roots beta in (|j|, hi) of the threshold condition, ascending."""
    from scipy.optimize import brentq
    lo = lo if lo is not None else abs(j) * (1 + 1e-9)
    bs = np.linspace(lo, hi, n_scan)
    f = np.array([cutoff_threshold_function(b, j, m, r0) for b in bs])
    roots = []
    for i in np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
        roots.append(brentq(lambda b: cutoff_threshold_function(b, j, m, r0), bs[i], bs[i + 1], xtol=1e-14))
    return np.array(roots)

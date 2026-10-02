"""Radial 2D Dirac equation with a central potential.

For H = sigma.p + Delta sigma_z + V(r) (hbar v = 1) and total angular momentum
j = m + 1/2 = +-1/2, +-3/2, ..., the spinor
    psi = (2 pi r)^{-1/2} ( F(r) e^{i m theta},  i G(r) e^{i (m+1) theta} )
obeys
    F' =  (j/r) F - (E - V + Delta) G
    G' = -(j/r) G + (E - V - Delta) F
with |psi|^2 = (F^2 + G^2)/(2 pi r).

The equations are integrated with a fixed-step fourth-order Runge-Kutta scheme
on the variable u = ln r + r/L (logarithmic near the origin, linear beyond L),
vectorised over many energies -- or many coupling strengths, since every
potential used here scales linearly with the impurity charge, V = lam * v(r).
"""
import numpy as np

# -------------------------------------------------------------------------
# grids
# -------------------------------------------------------------------------


def _u_of_r(r, L):
    return np.log(r) + r / L


def _r_of_u(u, L):
    """Invert u = ln r + r/L by Newton iteration (vectorised)."""
    u = np.asarray(u, dtype=float)
    r = np.where(u < np.log(L), np.exp(np.minimum(u, 700.0)), np.maximum(L * (u - np.log(L)), L))
    for _ in range(60):
        g = np.log(r) + r / L - u
        dg = 1.0 / r + 1.0 / L
        rn = r - g / dg
        rn = np.where(rn <= 0, 0.5 * r, rn)
        if np.max(np.abs(rn - r) / r) < 1e-15:
            r = rn
            break
        r = rn
    return r


def make_grid(r_a, r_b, L, h, breaks=()):
    """Points (with half points) from r_a to r_b, with every r in `breaks` on the grid.

    Returns arrays (r, du): r has 2n+1 entries (full and half points), du the n
    signed step sizes in u.  Works for r_b < r_a (inward integration).
    """
    pts = [r_a] + sorted([b for b in breaks if min(r_a, r_b) < b < max(r_a, r_b)],
                         reverse=r_b < r_a) + [r_b]
    rs, dus = [np.array([r_a])], []
    for a, b in zip(pts[:-1], pts[1:]):
        ua, ub = _u_of_r(a, L), _u_of_r(b, L)
        n = max(2, int(np.ceil(abs(ub - ua) / h)))
        uu = np.linspace(ua, ub, 2 * n + 1)
        rr = _r_of_u(uu, L)
        rr[0], rr[-1] = a, b
        rs.append(rr[1:])
        dus.append(np.full(n, (ub - ua) / n))
    return np.concatenate(rs), np.concatenate(dus)


# -------------------------------------------------------------------------
# propagation
# -------------------------------------------------------------------------


def propagate(v, Delta, E, lam, j, y0, r, du, L, record_r=(), norm_upto=None, invariant_from=None):
    """RK4 propagation of (F, G) for arrays E and lam (same length) along the grid.

    v      : callable, potential shape; V = lam * v(r)
    y0     : (F0, G0) arrays at r[0]
    record : radii (must be grid points) at which to store (F, G)
    norm_upto : accumulate int (F^2 + G^2) dr over the grid points with r <= norm_upto
    invariant_from : average Q = |a F^2 + b G^2| / k over r >= this radius
                     (continuum normalisation, see `continuum`)
    Returns dict with final y, records, accumulators and log-scale per column.
    """
    E = np.asarray(E, float)
    lam = np.asarray(lam, float)
    j = np.asarray(j, float)
    F, G = (np.array(y0[0], float), np.array(y0[1], float))
    vr = v(r)
    c = L / (L + r)          # dr/du / r
    rc = r * c               # dr/du
    n = len(du)
    logscale = np.zeros_like(F)
    rec_idx = {int(np.argmin(np.abs(r - rr))): rr for rr in record_r}
    records = {}
    acc_norm = np.zeros_like(F)
    q_sum = np.zeros_like(F)
    q_len = 0.0
    if 0 in rec_idx:
        records[0] = (F.copy(), G.copy())

    def rhs(F, G, i):
        a = j * c[i]
        w = E - lam * vr[i]
        return a * F - rc[i] * (w + Delta) * G, -a * G + rc[i] * (w - Delta) * F

    for s in range(n):
        i0, i1, i2 = 2 * s, 2 * s + 1, 2 * s + 2
        h = du[s]
        k1F, k1G = rhs(F, G, i0)
        k2F, k2G = rhs(F + 0.5 * h * k1F, G + 0.5 * h * k1G, i1)
        k3F, k3G = rhs(F + 0.5 * h * k2F, G + 0.5 * h * k2G, i1)
        k4F, k4G = rhs(F + h * k3F, G + h * k3G, i2)
        Fn = F + h / 6.0 * (k1F + 2 * k2F + 2 * k3F + k4F)
        Gn = G + h / 6.0 * (k1G + 2 * k2G + 2 * k3G + k4G)
        dr = r[i2] - r[i0]
        if norm_upto is not None and r[i2] <= norm_upto * (1 + 1e-12):
            acc_norm += 0.5 * abs(dr) * (F * F + G * G + Fn * Fn + Gn * Gn)
        F, G = Fn, Gn
        if invariant_from is not None and r[i2] >= invariant_from:
            w = E - lam * vr[i2]
            k = np.sqrt(np.maximum(w * w - Delta * Delta, 1e-300))
            q_sum += abs(dr) * np.abs((w - Delta) * F * F + (w + Delta) * G * G) / k
            q_len += abs(dr)
        if s % 10 == 0:
            big = np.maximum(np.abs(F), np.abs(G)) > 1e50
            if np.any(big):
                F[big] *= 1e-50
                G[big] *= 1e-50
                acc_norm[big] *= 1e-100
                q_sum[big] *= 1e-100
                logscale[big] += 50 * np.log(10)
                for key in records:
                    records[key][0][big] *= 1e-50
                    records[key][1][big] *= 1e-50
        if i2 in rec_idx:
            records[i2] = (F.copy(), G.copy())
    out = {"F": F, "G": G, "logscale": logscale, "norm": acc_norm, "records": records, "r": r}
    if invariant_from is not None:
        out["Q"] = q_sum / max(q_len, 1e-300)
    return out


def regular_start(v, Delta, E, lam, j, r_s):
    """Regular solution at small r for a potential finite at the origin
    (or, for a point Coulomb shape v = -1/r, the r^s solution, s = sqrt(j^2 - lam^2)).
    E, lam and j may be arrays of a common length."""
    E = np.asarray(E, float)
    j = np.broadcast_to(np.asarray(j, float), E.shape)
    lam = np.broadcast_to(np.asarray(lam, float), E.shape)
    if getattr(v, "point", False):
        s = np.sqrt(j * j - lam * lam)
        F = r_s ** s
        return F, (j - s) / lam * F
    w = E - lam * v(np.array([r_s]))[0]
    aj = np.abs(j)
    pos = j > 0
    F = np.where(pos, r_s ** aj, -(w + Delta) * r_s ** (aj + 1) / (2 * aj + 1))
    G = np.where(pos, (w - Delta) * r_s ** (aj + 1) / (2 * aj + 1), r_s ** aj)
    return F, G


def decaying_start(v, Delta, E, lam, j, R):
    """Exponentially decaying solution at large R (|E - V(R)| < Delta)."""
    w = np.asarray(E, float) - np.asarray(lam, float) * v(np.array([R]))[0]
    kap = np.sqrt(np.maximum(Delta * Delta - w * w, 0.0))
    F = np.ones_like(w)
    G = (j / R + kap) / (w + Delta)
    return F, G


# -------------------------------------------------------------------------
# solver object
# -------------------------------------------------------------------------


class RadialDirac:
    """Bound states, critical couplings and continuum states for V = lam * v(r).

    v must be a callable potential *shape* with attributes `kinks` (radii where
    v is not smooth) and `tail_beta` (strength of the -beta/r tail; 0 for a
    short-range tail).  Delta is the mass in nm^-1.
    """

    def __init__(self, v, Delta, r_core, h_log=0.01, h_lin=0.02, r_s=None):
        self.v, self.Delta, self.r_core = v, float(Delta), float(r_core)
        self.h_log, self.h_lin = h_log, h_lin
        self.r_s = r_s if r_s is not None else 1e-6 * r_core
        self.kinks = tuple(getattr(v, "kinks", ()))

    # ---- Wronskian at the matching point --------------------------------
    def wronskian(self, E, lam, j, R=None, r_m=None):
        E = np.atleast_1d(np.asarray(E, float))
        lam = np.broadcast_to(np.asarray(lam, float), E.shape).copy()
        D = self.Delta
        w_inf = E                                   # potential vanishes at infinity
        kap = np.sqrt(np.maximum(D * D - w_inf ** 2, 1e-30))
        kmin = np.min(kap)
        tb = getattr(self.v, "tail_beta", 0.0) * np.max(lam)
        if R is None:
            # decay length: exponential (1/kappa); at threshold a Coulomb tail gives a
            # stretched exponential exp(-sqrt(8 beta Delta r)), while a short-range
            # (gate-screened) tail gives the power law G ~ r^{-|j|}, F -> 0, which the
            # local start below reproduces once V(R) is negligible
            if kmin > 1e-3 * D:
                R = 60.0 / kmin
            elif tb * D > 1e-9:
                R = 2000.0 / (tb * D)
            else:
                R = 400.0 * getattr(self.v, "range", self.r_core)
            R = max(R, 50 * self.r_core)
        if r_m is None:
            r_m = min(max(5 * self.r_core, 1.0 / max(np.max(kap), 1e-12)), R / 4)
        L = max(min(1.0 / max(kmin, 1e-12), R / 20), 2 * self.r_core)
        r1, du1 = make_grid(self.r_s, r_m, L, self.h_log, self.kinks)
        y0 = regular_start(self.v, D, E, lam, j, self.r_s)
        out = propagate(self.v, D, E, lam, j, y0, r1, du1, L)
        r2, du2 = make_grid(R, r_m, L, self.h_lin, self.kinks)
        y1 = decaying_start(self.v, D, E, lam, j, R)
        inn = propagate(self.v, D, E, lam, j, y1, r2, du2, L)
        Fo, Go, Fi, Gi = out["F"], out["G"], inn["F"], inn["G"]
        return (Fo * Gi - Fi * Go) / (np.hypot(Fo, Go) * np.hypot(Fi, Gi))

    @staticmethod
    def _ksection(fun, a, b, fa, tol, k=24, maxit=40):
        """Vectorised k-section root refinement of a sign change of fun on [a, b]."""
        for _ in range(maxit):
            x = np.linspace(a, b, k + 2)[1:-1]
            fx = fun(x)
            xs = np.concatenate([[a], x, [b]])
            fs = np.concatenate([[fa], fx, [-fa]])
            cand = np.nonzero(np.sign(fs[:-1]) * np.sign(fs[1:]) <= 0)[0]
            if len(cand) == 0:
                break
            idx = cand[0]
            a, b, fa = xs[idx], xs[idx + 1], fs[idx]
            if b - a < tol:
                break
        # final linear interpolation
        x = np.array([a, b])
        fx = fun(x)
        return a - fx[0] * (b - a) / (fx[1] - fx[0]) if fx[1] != fx[0] else 0.5 * (a + b)

    # ---- bound states -----------------------------------------------------
    def bound_states(self, lam, j, n_scan=240, edge=1e-7, tol=1e-13):
        """All bound energies in (-Delta, Delta) for coupling lam, ascending."""
        D = self.Delta
        th = np.linspace(np.arccos(1 - edge), np.arccos(-1 + edge), n_scan)
        Eg = D * np.cos(th)[::-1]
        # scan in batches of comparable decay constant
        W = np.empty_like(Eg)
        for chunk in np.array_split(np.arange(len(Eg)), 12):
            W[chunk] = self.wronskian(Eg[chunk], lam, j)
        roots = []
        for i in np.nonzero(np.sign(W[:-1]) * np.sign(W[1:]) < 0)[0]:
            a, b = Eg[i], Eg[i + 1]
            # discard spurious sign flips where the scan was too coarse (|W| ~ 1 on both sides)
            fun = lambda x: self.wronskian(x, lam, j)
            fa = fun(np.array([a]))[0]
            fb = fun(np.array([b]))[0]
            if np.sign(fa) == np.sign(fb):
                continue
            roots.append(self._ksection(fun, a, b, fa, tol * D))
        return np.array(roots)

    # ---- critical coupling: level reaches E = -Delta ----------------------
    def critical_couplings(self, j, lam_lo, lam_hi, n_scan=120, tol=1e-12):
        """Couplings lam at which successive levels of channel j reach E = -Delta.

        Uses the threshold (E = -Delta) solution, which decays as
        exp(-sqrt(8 beta Delta r)) for a Coulomb tail.
        """
        D = self.Delta
        # logarithmic spacing: successive critical couplings are separated by a
        # roughly constant ratio, so a log grid resolves all of them
        lams = np.geomspace(lam_lo, lam_hi, n_scan)
        fun = lambda L_: self.wronskian(np.full(len(L_), -D), L_, j)
        W = np.concatenate([fun(c) for c in np.array_split(lams, 6)])
        roots = []
        good = np.isfinite(W[:-1]) & np.isfinite(W[1:])
        for i in np.nonzero(good & (np.sign(W[:-1]) * np.sign(W[1:]) < 0))[0]:
            a, b = lams[i], lams[i + 1]
            fa = fun(np.array([a]))[0]
            if np.sign(fa) == np.sign(fun(np.array([b]))[0]):
                continue
            roots.append(self._ksection(fun, a, b, fa, tol))
        return np.array(roots)

    # ---- continuum ----------------------------------------------------------
    def continuum(self, E, lam, j, r_in=None, record_r=(), n_waves=150, k_ratio=1.6):
        """Energy-normalised continuum solutions (|E| > Delta).

        Normalisation: <F_E, F_E'> + <G_E, G_E'> = delta(E - E').  Far away the
        WKB invariant Q = |a F^2 + b G^2| / k (a, b = E - V -+ Delta,
        k = local momentum) is constant and the energy-normalised amplitude
        follows from Q = 1/pi.  E, lam, j may be arrays (broadcast together);
        they are propagated in batches of similar momentum.  Returns dict with
        records (normalised F, G at the requested radii), the inner weight
        int_0^{r_in} (F^2 + G^2) dr and the normalisation factor.
        """
        E = np.atleast_1d(np.asarray(E, float))
        lam = np.broadcast_to(np.asarray(lam, float), E.shape).copy()
        j = np.broadcast_to(np.asarray(j, float), E.shape).copy()
        D = self.Delta
        k = np.sqrt(E * E - D * D)
        group = np.floor(np.log(k) / np.log(k_ratio)).astype(int)
        inner = np.zeros_like(E)
        Nn = np.zeros_like(E)
        recs = {}
        for g in np.unique(group):
            idx = np.nonzero(group == g)[0]
            kmin, kmax = np.min(k[idx]), np.max(k[idx])
            L = max(1.0 / kmax, self.r_core)
            # the WKB invariant has O((j/kR)^2) corrections: go far enough for high j
            R = max(n_waves * 2 * np.pi, 300.0 * np.max(np.abs(j[idx]))) / kmin
            R = max(R, 100 * self.r_core)
            breaks = list(self.kinks) + list(record_r) + ([r_in] if r_in else [])
            rr, du = make_grid(self.r_s, R, L, self.h_lin if R > L else self.h_log, breaks)
            y0 = regular_start(self.v, D, E[idx], lam[idx], j[idx], self.r_s)
            out = propagate(self.v, D, E[idx], lam[idx], j[idx], y0, rr, du, L, record_r=record_r,
                            norm_upto=r_in, invariant_from=0.6 * R)
            N_ = 1.0 / np.sqrt(np.pi * out["Q"])
            Nn[idx] = N_
            inner[idx] = out["norm"] * N_ ** 2
            for i, (Fv, Gv) in out["records"].items():
                key = min(record_r, key=lambda x: abs(x - rr[i]))
                if key not in recs:
                    recs[key] = (np.zeros_like(E), np.zeros_like(E))
                recs[key][0][idx] = Fv * N_
                recs[key][1][idx] = Gv * N_
        return {"records": recs, "inner": inner, "N": Nn}

    # ---- resonances ---------------------------------------------------------
    def quasi_bound(self, lam, j, E_lo, E_hi, n_scan=120):
        """Real quasi-bound energies below -Delta: the regular solution is matched
        to the solution that decays into the Coulomb barrier.  The barrier is the
        region |E - V| < Delta, i.e. beta/(|E|+Delta) < r < beta/(|E|-Delta); the
        decaying solution is started at r_b = beta/(|E| - Delta/2), beyond the
        barrier centre, so that the neglected tunnelling correction is of the
        order of the resonance width.  The result seeds the Lorentzian search."""
        beta = getattr(self.v, "tail_beta", 0.0) * lam
        if beta <= 0:
            return np.array([])

        def W(E):
            E = np.atleast_1d(E)
            out = np.empty_like(E)
            for i, e in enumerate(E):
                rb = beta / (abs(e) - 0.5 * self.Delta)
                r1 = beta / (abs(e) + self.Delta)
                out[i] = self.wronskian(np.array([e]), lam, j, R=rb, r_m=0.5 * r1)[0]
            return out

        Eg = np.linspace(E_lo, E_hi, n_scan)
        w = W(Eg)
        roots = []
        for i in np.nonzero(np.sign(w[:-1]) * np.sign(w[1:]) < 0)[0]:
            # genuine zero crossings have |W| small on at least one side
            if min(abs(w[i]), abs(w[i + 1])) > 0.5:
                continue
            roots.append(self._ksection(W, Eg[i], Eg[i + 1], w[i], 1e-12 * self.Delta, k=8, maxit=12))
        return np.array(roots)

    def resonances(self, lam, j, E_lo, E_hi, r_in, n_scan=240, n_waves=40, min_prominence=1.5):
        """Quasi-bound states in the lower continuum (E_lo < E < E_hi < -Delta).

        The energy-normalised inner weight w(E) = int_0^{r_in}(F^2+G^2) dr is a
        Lorentzian (Gamma/2pi)/((E-E_r)^2 + Gamma^2/4) times the inner
        probability of the resonance.  Candidates come from local maxima of w on
        a scan and from the quasi-bound energies; each is refined by successive
        zooming and fitted with a Lorentzian plus linear background.
        Returns a list of dicts (E_r, Gamma, w_max, Gamma_peak = 2/(pi w_max)).
        """
        fun = lambda Ex: self.continuum(Ex, lam, j, r_in=r_in, n_waves=n_waves)["inner"]
        Eg = np.linspace(E_lo, E_hi, n_scan)
        w = fun(Eg)
        cands = []
        for i in range(1, n_scan - 1):
            if w[i] > w[i - 1] and w[i] >= w[i + 1] and \
                    w[i] >= min_prominence * min(w[max(i - 8, 0)], w[min(i + 8, n_scan - 1)]):
                cands.append((Eg[i - 1], Eg[i + 1]))
        dE = Eg[1] - Eg[0]
        for e in self.quasi_bound(lam, j, E_lo, E_hi):
            if not any(a - dE <= e <= b + dE for a, b in cands):
                h = 0.02 * (abs(e) - self.Delta)
                cands.append((e - h, min(e + h, -self.Delta * (1 + 1e-9))))
        out = []
        for a, b in cands:
            for _ in range(40):
                xs = np.linspace(a, b, 17)
                ws = fun(xs)
                k = int(np.argmax(ws))
                a, b = xs[max(k - 1, 0)], xs[min(k + 1, 16)]
                gam = 2.0 / (np.pi * ws[k])
                if b - a < 0.05 * gam:
                    break
            E0, wmax = xs[k], ws[k]
            gam = 2.0 / (np.pi * wmax)
            xs = E0 + gam * np.linspace(-3, 3, 41)
            xs = xs[xs < -self.Delta]
            ws = fun(xs)
            if not np.all(np.isfinite(ws)) or not np.isfinite(wmax):
                continue
            res = _fit_lorentzian(xs, ws, E0, gam, wmax)
            res.update({"w_max": wmax, "Gamma_peak": gam})
            depth = abs(res["E_r"]) - self.Delta
            ok = (res["Gamma"] < 2 * depth and np.all(np.isfinite(res["fit_err"]))
                  and res["fit_err"][1] < 0.2 * res["Gamma"])
            if ok and not any(abs(res["E_r"] - o["E_r"]) < 0.5 * (res["Gamma"] + o["Gamma"]) for o in out):
                out.append(res)
        return sorted(out, key=lambda o: -o["E_r"])


def _fit_lorentzian(x, y, E0, gam, ymax):
    from scipy.optimize import curve_fit

    def model(x, Er, G, A, c0, c1):
        return A * (G / 2) ** 2 / ((x - Er) ** 2 + (G / 2) ** 2) + c0 + c1 * (x - E0)

    try:
        p, cov = curve_fit(model, x, y, p0=[E0, gam, ymax, 0.0, 0.0], maxfev=20000)
        return {"E_r": p[0], "Gamma": abs(p[1]), "A": p[2], "fit_err": np.sqrt(np.diag(cov))[:2]}
    except Exception:
        return {"E_r": E0, "Gamma": gam, "A": ymax, "fit_err": np.array([np.nan, np.nan])}


def kmax_safe(k):
    return max(np.max(k), 1e-12)


# -------------------------------------------------------------------------
# Siegert (complex-energy) resonances by exterior complex rotation
# -------------------------------------------------------------------------


def _rk4_path(Vc, Delta, E, j, path, y0):
    """RK4 along a polyline of complex radii path[0..2n] (full and half points)."""
    F, G = complex(y0[0]), complex(y0[1])
    Vp = Vc(path)
    for s in range(0, len(path) - 1, 2):
        r0, rm, r1 = path[s], path[s + 1], path[s + 2]
        h = r1 - r0

        def f(F, G, r, V):
            return (j / r) * F - (E - V + Delta) * G, -(j / r) * G + (E - V - Delta) * F

        k1 = f(F, G, r0, Vp[s])
        k2 = f(F + 0.5 * h * k1[0], G + 0.5 * h * k1[1], rm, Vp[s + 1])
        k3 = f(F + 0.5 * h * k2[0], G + 0.5 * h * k2[1], rm, Vp[s + 1])
        k4 = f(F + h * k3[0], G + h * k3[1], r1, Vp[s + 2])
        F = F + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        G = G + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        nrm = abs(F) + abs(G)
        if nrm > 1e50:
            F, G = F / nrm, G / nrm
    return F, G


def siegert(solver, lam, j, E0, Gamma0=None, R0=None, theta=0.4, tol=1e-14, maxit=60):
    """Complex resonance energy E_r - i Gamma/2 of channel j for V = lam * v(r).

    The regular solution (real r) is matched at r_m to the outgoing solution,
    which is continued along the rotated ray r = R0 + s e^{-i theta} (lower
    continuum, E < -Delta; for E < -Delta the outgoing wave is e^{-ikr}) where it
    decays.  The potential beyond R0 must be analytic: `solver.v.analytic_tail`
    (a callable valid for complex r >= R0) is used there.
    """
    v, D = solver.v, solver.Delta
    beta = getattr(v, "tail_beta", 0.0) * lam
    tail = lambda r: lam * v.analytic_tail(r)
    E0 = complex(E0)
    sgn = -1.0 if E0.real < 0 else 1.0
    if R0 is None:
        r2 = beta / max(abs(E0.real) - D, 1e-12 * D)
        R0 = max(1.5 * r2, getattr(v, "analytic_from", 0.0), 20 * solver.r_core)

    def W(E):
        k = np.sqrt(E * E - D * D + 0j)
        if k.real < 0:
            k = -k
        r_m = min(max(5 * solver.r_core, 0.25 * beta / max(abs(E.real) + D, 1e-30)), 0.5 * R0)
        # outward regular solution on the real axis (complex E)
        L = max(1.0 / max(abs(k), 1e-12), solver.r_core)
        rr, _ = make_grid(solver.r_s, r_m, L, solver.h_log, solver.kinks)
        y0 = regular_start(v, D, np.array([E.real]), np.array([lam]), np.array([j]), solver.r_s)
        # start values depend weakly on E; use the complex E in the leading term
        Fo, Go = _rk4_path(lambda r: lam * v(np.real(r)), D, E, j, rr.astype(complex), (y0[0][0], y0[1][0]))
        # outgoing solution: far end of the rotated ray
        ray_len = 40.0 / max(abs(k) * np.sin(theta), 1e-12)
        n_ray = int(max(400, 30 * abs(k) * ray_len))
        sray = np.linspace(ray_len, 0.0, 2 * n_ray + 1)
        path_ray = R0 + sray * np.exp(1j * sgn * theta)
        rf = path_ray[0]
        wv = E - tail(np.array([rf]))[0]
        kk = np.sqrt(wv * wv - D * D + 0j)
        # pick the root for which exp(i kk (r - rf)) decays along the ray towards larger s
        if (1j * kk * np.exp(1j * sgn * theta)).real > 0:
            kk = -kk
        # F' = i kk F for the outgoing wave e^{i kk r}; G from F' = (j/r)F - (w + D) G
        Fi = 1.0 + 0j
        Gi = ((j / rf) - 1j * kk) * Fi / (wv + D)
        Fi, Gi = _rk4_path(tail, D, E, j, path_ray, (Fi, Gi))
        # real axis from R0 inward to r_m
        rr2, _ = make_grid(R0, r_m, max(R0 / 30, solver.r_core), solver.h_lin, solver.kinks)
        Fi, Gi = _rk4_path(lambda r: lam * v(np.real(r)), D, E, j, rr2.astype(complex), (Fi, Gi))
        return (Fo * Gi - Fi * Go) / (np.sqrt(abs(Fo) ** 2 + abs(Go) ** 2) * np.sqrt(abs(Fi) ** 2 + abs(Gi) ** 2))

    E1 = E0 - 0.5j * (Gamma0 if Gamma0 else 1e-3 * D)
    E2 = E1 * (1 + 1e-6) + 1e-6 * D
    f1, f2 = W(E1), W(E2)
    for _ in range(maxit):
        if f2 == f1:
            break
        E3 = E2 - f2 * (E2 - E1) / (f2 - f1)
        E1, f1 = E2, f2
        E2, f2 = E3, W(E3)
        if abs(E2 - E1) < tol * D:
            break
    return E2.real, -2 * E2.imag

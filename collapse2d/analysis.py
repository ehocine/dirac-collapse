"""Derived quantities: critical charges, asymptotic laws, resonance tracking, LDOS."""
import mpmath as mp
import numpy as np
from scipy.special import j0, j1

from .dirac import RadialDirac, make_grid, propagate, regular_start, decaying_start, siegert
from .potentials import HeightCoulomb, ScreenedImpurity

EULER = float(mp.euler)

# constants of the critical-coupling law  gamma_c = pi / [ln(1/(m r0)) + c]
# (j = 1/2): c = a_in - 2 C_E, with a_in = A/B for the zero-mass, beta = 1/2
# core solution F = A + B ln(r/r0) beyond the core.
C_CUTOFF = 2 * j0(0.5) / (j0(0.5) - j1(0.5)) - 2 * EULER
C_HEIGHT = np.log(8.0) - 2 * EULER


def beta_c_asymptotic(m_r0, c):
    """Leading-log critical coupling for j = 1/2: beta_c = sqrt(1/4 + pi^2/(ln(1/(m r0)) + c)^2)."""
    L = np.log(1.0 / np.asarray(m_r0, float)) + c
    return np.sqrt(0.25 + (np.pi / L) ** 2)


def core_constant(v, r_core, R=1e7):
    """a_in for a general regularised core of size r_core (potential shape v,
    with V = -v_tail_strength * ...): integrate the zero-energy, massless,
    beta = j = 1/2 problem and fit F = A + B ln(r/r_core) far outside."""
    from scipy.integrate import solve_ivp
    scale = 0.5 / v.tail_beta

    def rhs(t, y):
        r = np.exp(t)
        V = scale * v(np.array([r]))[0]
        F, G = y
        return [r * ((0.5 / r) * F + V * G), r * (-(0.5 / r) * G - V * F)]

    r0 = 1e-8 * r_core
    sol = solve_ivp(rhs, (np.log(r0), np.log(R * r_core)), [r0 ** 0.5, 0.0], rtol=1e-12, atol=1e-40)
    F, G = sol.y[:, -1]
    B = (F - G) / 2
    A = F - B * np.log(R)
    return A / B


def gamow_exponent(beta, E, m):
    """2S = 2 pi beta (|E|/k - 1): WKB tunnelling exponent through the Coulomb
    barrier beta/(|E|+m) < r < beta/(|E|-m) at energy E < -m, k = sqrt(E^2 - m^2)."""
    k = np.sqrt(E * E - m * m)
    return 2 * np.pi * beta * (np.abs(E) / k - 1.0)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def make_solver(mat, d, model="rpa", D=np.inf, h_log=0.01, h_lin=0.02):
    """RadialDirac for an impurity of unit charge; the coupling lam is then Z.

    model: 'bare'   charge at height d, constant kappa       (V = -Z alpha0/(kappa sqrt(r^2+d^2)))
           'rpa'    + interband polarization of the gapped Dirac sea (+ Keldysh r_K, + gate D)
           'env'    Keldysh/gate screening only (N = 0)
    """
    if model == "bare" and np.isinf(D) and mat.r_K == 0:
        v = HeightCoulomb(mat.alpha0 / mat.kappa, d)
    else:
        v = ScreenedImpurity(1.0, mat, d, D=D, rpa=(model == "rpa"))
    return RadialDirac(v, mat.m, r_core=d, h_log=h_log, h_lin=h_lin)


def critical_charge(solver, Z_lo=0.1, Z_hi=12.0, j=0.5, n_scan=120, which=0):
    roots = solver.critical_couplings(j, Z_lo, Z_hi, n_scan=n_scan)
    return roots[which] if len(roots) > which else np.nan


# ---------------------------------------------------------------------------
# resonances
# ---------------------------------------------------------------------------


def resonance_pole(solver, Z, j, E_seed, G_seed, rel_jump=0.3):
    """Siegert pole near the seed; None if the iteration fails or jumps away."""
    m = solver.Delta
    try:
        Er, G = siegert(solver, Z, j, E_seed, G_seed)
    except Exception:
        return None
    if not (np.isfinite(Er) and np.isfinite(G)) or G <= 0 or Er >= -m:
        return None
    if abs(Er - E_seed) > rel_jump * (abs(E_seed) - m) + 3 * G_seed:
        return None
    return Er, G


def track_resonance(solver, Zs, j=0.5, gamow_c=-1.7, G_floor=1e-12, E_window=4.0):
    """E_r(Z) and Gamma(Z) for the lowest diving level above Z_c.

    Seeds: the quasi-bound energy (accurate near threshold where the barrier is
    thick) and the previous pole, with the width seeded by the Gamow law.  When
    the Gamow estimate is below G_floor (relative to m) the width is not
    resolvable in double precision; the quasi-bound energy and the Gamow width
    are then reported with flag 'gamow'.
    """
    m = solver.Delta
    beta_tail = solver.v.tail_beta
    out = []
    prev = None
    for Z in Zs:
        qb = solver.quasi_bound(Z, j, -E_window * m, -m * (1 + 1e-7), n_scan=60)
        seeds = []
        if len(qb):
            seeds.append(qb[-1] if prev is None else qb[np.argmin(np.abs(qb - prev[0]))])
        if prev is not None:
            seeds.append(prev[0])
        rec = None
        for Es in seeds:
            twoS = gamow_exponent(beta_tail * Z, Es, m)
            Gs = m * np.exp(gamow_c - twoS)
            if Gs < G_floor * m:
                rec = (Es, Gs, "gamow")
                break
            res = resonance_pole(solver, Z, j, Es, max(Gs, 1e-14 * m))
            if res is not None:
                rec = (res[0], res[1], "siegert")
                break
        if rec is None and prev is not None:
            res = resonance_pole(solver, Z, j, prev[0], prev[1], rel_jump=1.0)
            if res is not None:
                rec = (res[0], res[1], "siegert")
        out.append((Z,) + (rec if rec else (np.nan, np.nan, "fail")))
        if rec:
            prev = rec
    return out


# ---------------------------------------------------------------------------
# local density of states
# ---------------------------------------------------------------------------


def bound_density(solver, E, lam, j, r_eval):
    """F^2 + G^2 at the radii r_eval for the normalised bound state at energy E."""
    D = solver.Delta
    kap = np.sqrt(D * D - E * E)
    R = max(60.0 / kap, 50 * solver.r_core)
    r_m = min(max(5 * solver.r_core, 1.0 / kap), R / 4)
    L = max(min(1.0 / kap, R / 20), 2 * solver.r_core)
    rin = [r for r in r_eval if r < r_m]
    rout = [r for r in r_eval if r_m <= r < R]
    args = (np.array([E]), np.array([lam]), np.array([j]))
    r1, du1 = make_grid(solver.r_s, r_m, L, solver.h_log, list(solver.kinks) + rin)
    o = propagate(solver.v, D, *args, regular_start(solver.v, D, *args, solver.r_s),
                  r1, du1, L, record_r=rin, norm_upto=r_m)
    r2, du2 = make_grid(R, r_m, L, solver.h_lin, list(solver.kinks) + rout)
    i = propagate(solver.v, D, *args, decaying_start(solver.v, D, *args, R),
                  r2, du2, L, record_r=rout, norm_upto=R)
    Fo, Go, Fi, Gi = o["F"][0], o["G"][0], i["F"][0], i["G"][0]
    s = Fo / Fi if abs(Fi) > abs(Gi) else Go / Gi
    norm = o["norm"][0] + s * s * i["norm"][0]
    dens = {}
    for idx, (F, G) in o["records"].items():
        dens[min(rin, key=lambda x: abs(x - r1[idx]))] = (F[0] ** 2 + G[0] ** 2) / norm
    for idx, (F, G) in i["records"].items():
        dens[min(rout, key=lambda x: abs(x - r2[idx]))] = s * s * (F[0] ** 2 + G[0] ** 2) / norm
    return np.array([dens.get(r, 0.0) for r in r_eval])


def ldos(solver, lam, E_grid, r_list, j_max=7.5, eta=0.0, bound_E=None, n_waves=60, continuum=True):
    """LDOS per flavour (states per area per energy, hbar v = 1) at radii r_list.

    Continuum: sum_j (F^2 + G^2)/(2 pi r) with energy-normalised solutions.
    Bound states (energies bound_E[j] for each j) are broadened by a Lorentzian
    of half-width eta.  Energies inside the gap contribute only through bound
    states.  Returns array (len(E_grid), len(r_list)).
    """
    D = solver.Delta
    js = np.concatenate([-np.arange(0.5, j_max + 0.1, 1.0), np.arange(0.5, j_max + 0.1, 1.0)])
    E_grid = np.asarray(E_grid, float)
    out = np.zeros((len(E_grid), len(r_list)))
    cont = np.abs(E_grid) > D * (1 + 1e-9)
    if continuum and np.any(cont):
        Ec = E_grid[cont]
        EE = np.repeat(Ec, len(js))
        JJ = np.tile(js, len(Ec))
        res = solver.continuum(EE, lam, JJ, record_r=list(r_list), n_waves=n_waves)
        for ir, r in enumerate(r_list):
            F, G = res["records"][r]
            dens = ((F ** 2 + G ** 2) / (2 * np.pi * r)).reshape(len(Ec), len(js)).sum(1)
            out[cont, ir] = dens
    if bound_E is not None and eta > 0:
        for j, Es in bound_E.items():
            for Eb in Es:
                rho = bound_density(solver, Eb, lam, j, list(r_list)) / (2 * np.pi * np.asarray(r_list))
                lor = (eta / np.pi) / ((E_grid - Eb) ** 2 + eta ** 2)
                out += lor[:, None] * rho[None, :]
    return out

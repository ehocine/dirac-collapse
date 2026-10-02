"""Table I: validation of every numerical ingredient against closed forms.

1. interband polarization Pi(q) vs direct evaluation of the bubble integral
2. screened potentials vs closed forms (gapless RPA, gate image charge, Keldysh)
3. 2D Dirac-Coulomb levels (point charge, subcritical)
4. critical coupling of the cutoff Coulomb model vs the exact Bessel-function condition
5. LDOS sum rule of the free massive Dirac fermion, sum_j |psi_j|^2 = |E|/(2 pi)
6. resonances: Siegert poles (two rotation angles) vs Lorentzian fits of the
   energy-normalised inner weight on the real axis
"""
import sys
import time

import numpy as np
from common import load_json, save_json, sci, write_table
from scipy.integrate import dblquad
from scipy.special import struve, y0

from collapse2d import exact
from collapse2d.dirac import RadialDirac, siegert
from collapse2d.materials import gapped_graphene
from collapse2d.potentials import CutoffCoulomb, PointCoulomb, ScreenedImpurity, _hankel_gl, polarization


class Zero:
    kinks = ()
    tail_beta = 0.0

    def __call__(self, r):
        return np.zeros_like(np.asarray(r, float))


def bubble(q, m):
    def integrand(phi, k):
        E = np.sqrt(k * k + m * m)
        Ep = np.sqrt(k * k + q * q + 2 * k * q * np.cos(phi) + m * m)
        dot = k * k + k * q * np.cos(phi)
        return k * (1 - (dot + m * m) / (E * Ep)) / (E + Ep) / (4 * np.pi ** 2)
    return dblquad(integrand, 0, np.inf, 0, 2 * np.pi, epsabs=1e-13, epsrel=1e-10)[0]


def main():
    rows, rec = [], {}

    # 1 polarization
    errs = []
    for q, m in ((0.5, 0.0), (0.05, 0.1), (0.3, 0.1), (2.0, 0.1), (20.0, 0.1)):
        errs.append(abs(polarization(np.array([q]), m)[0] / bubble(q, m) - 1))
    rows.append(("Interband polarization $\\Pi(q)$", "direct bubble integral", max(errs)))
    rec["polarization"] = errs

    # 2 potentials
    mat = gapped_graphene(0.015)
    d = 0.3
    r = np.array([0.01, 0.5, 3, 30, 300, 3000.0])
    P = ScreenedImpurity(1.0, mat.with_(Delta=1e-12), d)
    e_gapless = np.max(np.abs(P(r) / (-mat.alpha0 / (P.kappa_inf * np.hypot(r, d))) - 1))
    G = ScreenedImpurity(1.0, mat.with_(N=0), d, D=20.0)
    ex = -mat.alpha0 / mat.kappa * (1 / np.hypot(r, d) - 1 / np.hypot(r, d + 40))
    e_gate = np.max(np.abs(G(r) / ex - 1))
    K = ScreenedImpurity(1.0, mat.with_(N=0, r_K=4.0, kappa=2.0), 1e-9)
    exK = -mat.alpha0 * np.pi / (2 * 4.0) * (struve(0, 2.0 * r / 4.0) - y0(2.0 * r / 4.0))
    e_keld = np.max(np.abs(K(r) / exK - 1))
    R = ScreenedImpurity(1.0, mat, d)
    brute = [abs(-mat.alpha0 * _hankel_gl(R.f, x, 40 / d, np.geomspace(1e-7, 40 / d, 200)) / R(np.array([x]))[0] - 1)
             for x in (1.0, 20.0, 150.0)]
    rows.append(("Screened potential, gapless RPA", "$-Z\\alpha_0/(\\kappa_\\infty\\sqrt{r^2+d^2})$", e_gapless))
    rows.append(("Screened potential, metallic gate", "image charge", e_gate))
    rows.append(("Screened potential, Keldysh", "Struve--Neumann form", e_keld))
    rows.append(("Screened potential, gapped RPA", "direct Hankel quadrature", max(brute)))
    rec["potentials"] = {"gapless": e_gapless, "gate": e_gate, "keldysh": e_keld, "gapped_brute": brute}

    # 3 Dirac-Coulomb levels
    S = RadialDirac(PointCoulomb(1.0), 1.0, r_core=1.0)
    errs = []
    for beta in (0.2, 0.45):
        for j in (0.5, -0.5, 1.5, -1.5):
            E = np.sort(S.bound_states(beta, j, edge=1e-3))[:4]
            exl = np.sort(exact.dirac_coulomb_levels(beta, j, 8))
            exl = exl[1:] if j < 0 else exl          # n = 0 exists only for j > 0
            errs.append(np.max(np.abs(E - exl[:len(E)])))
    rows.append(("Dirac--Coulomb levels ($\\beta=0.2,0.45$; $|j|\\le 3/2$)", "exact spectrum", max(errs)))
    rec["coulomb_levels"] = errs

    # 4 critical couplings, cutoff model
    errs, table4 = [], []
    for mr0, j in ((0.0667, 0.5), (1e-3, 0.5), (0.0667, 1.5)):
        exs = exact.cutoff_critical_coupling(j, mr0, 1.0, hi=4.0)[:2]
        S = RadialDirac(CutoffCoulomb(1.0, 1.0), mr0, r_core=1.0)
        num = S.critical_couplings(j, abs(j) * (1 + 1e-6), 4.0, n_scan=120)[:2]
        errs.append(np.max(np.abs(num - exs)))
        table4.append((mr0, j, exs, num))
    rows.append(("Critical couplings, cutoff Coulomb", "exact $K_{2i\\gamma}$ matching", max(errs)))
    rec["cutoff_critical"] = table4

    # 5 free LDOS sum rule
    S = RadialDirac(Zero(), 1.0, r_core=1.0)
    E0 = np.array([-1.5, -2.0, -3.0, 1.2])
    js = np.arange(-25.5, 26, 1.0)
    out = S.continuum(np.repeat(E0, len(js)), 0.0, np.tile(js, len(E0)), record_r=[0.5, 2.0, 7.0])
    errs = []
    for rr, (F, Gv) in out["records"].items():
        dens = ((F ** 2 + Gv ** 2) / (2 * np.pi * rr)).reshape(len(E0), len(js)).sum(1)
        errs.append(np.max(np.abs(dens / (np.abs(E0) / (2 * np.pi)) - 1)))
    rows.append(("Continuum normalisation (free LDOS)", "$|E|/(2\\pi\\hbar^2v^2)$", max(errs)))
    rec["ldos_sum_rule"] = errs

    # 6 resonances
    m = 0.0667
    S = RadialDirac(CutoffCoulomb(1.0, 1.0), m, r_core=1.0)
    res_rows = []
    for beta in (0.98, 1.0, 1.1):
        fit = S.resonances(beta, 0.5, -3.5 * m, -1.0005 * m, r_in=beta / (2 * m))[0]
        p1 = siegert(S, beta, 0.5, fit["E_r"], fit["Gamma"], theta=0.3)
        p2 = siegert(S, beta, 0.5, fit["E_r"], fit["Gamma"], theta=0.6)
        res_rows.append((beta, fit["E_r"] / m, fit["Gamma"] / m, p1[0] / m, p1[1] / m, abs(p2[1] / p1[1] - 1)))
        print("res", res_rows[-1])
    rec["resonances"] = res_rows

    body = ["\\begin{tabular}{llc}", "\\hline\\hline", "quantity & reference & max.\\ rel./abs.\\ deviation\\\\", "\\hline"]
    for name, ref, err in rows:
        body.append(f"{name} & {ref} & {sci(err)}\\\\")
    body += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_validation", "\n".join(body))

    resonance_table(res_rows)
    save_json(rec, "s1_validation")
    for r_ in rows:
        print(r_)


def resonance_table(res_rows):
    body = ["\\begin{tabular}{lccccc}", "\\hline\\hline",
            "$\\beta$ & \\multicolumn{2}{c}{Lorentzian fit (real $E$)} & \\multicolumn{2}{c}{Siegert pole} & $\\theta$-dependence\\\\",
            " & $E_r/m$ & $\\Gamma/m$ & $E_r/m$ & $\\Gamma/m$ & of $\\Gamma$\\\\", "\\hline"]
    for b, Ef, Gf, Es, Gs, th in res_rows:
        body.append(f"{b} & ${Ef:.6f}$ & {sci(Gf, 5)} & ${Es:.6f}$ & {sci(Gs, 5)} & {sci(th)}\\\\")
    body += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_resonance_check", "\n".join(body))


if __name__ == "__main__":
    if "--tables" in sys.argv:          # rewrite the resonance table from saved data
        resonance_table(load_json("s1_validation")["resonances"])
        sys.exit()
    t = time.time()
    main()
    print("done in %.0fs" % (time.time() - t))

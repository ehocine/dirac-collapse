"""Fig. 4 + Tables IV and VI: robustness checks.

(a) Is the Gamow prefactor c_G = ln(Gamma/Delta) + 2S universal?  We extract it
    for the cut-off and height regularisations over a wide range of m r0, i.e.
    of beta_c from close to 1/2 up to ~1.2, and compare with the beta_c -> 1/2
    limit implied by Gamayun, Gorbar & Gusynin, PRB 80, 165429 (2009):
        Gamma = (3 pi/4) m exp(-pi/sqrt(2 xi))  ->  c_G = ln(3 pi/4) - pi.
(b) Sensitivity of Z_c to the impurity height d (graphene/hBN, 30 meV gap).
"""
import sys
import time

import numpy as np
from common import C, SINGLE, plt, save_json, savefig, write_table

from collapse2d import exact
from collapse2d.analysis import critical_charge, gamow_exponent, make_solver, track_resonance
from collapse2d.dirac import RadialDirac
from collapse2d.materials import gapped_graphene
from collapse2d.potentials import CutoffCoulomb, HeightCoulomb

C_G_LIMIT = np.log(3 * np.pi / 4) - np.pi


def gamow_case(kind, mr0):
    m = 1.0
    v = CutoffCoulomb(1.0, mr0) if kind == "cutoff" else HeightCoulomb(1.0, mr0)
    S = RadialDirac(v, m, r_core=mr0)
    if kind == "cutoff":
        bc = exact.cutoff_critical_coupling(0.5, m, mr0, lo=0.5 * (1 + 1e-12), hi=3.0, n_scan=300)[0]
    else:
        bc = S.critical_couplings(0.5, 0.5 + 1e-9, 3.0, n_scan=160)[0]
    Zs = bc * np.array([1.002, 1.005, 1.01, 1.02, 1.04, 1.07])
    tr = track_resonance(S, Zs)
    pts = []
    for Z, E, G, flag in tr:
        if flag == "siegert" and np.isfinite(G) and G > 0:
            s2 = gamow_exponent(Z, E, m)
            pts.append((s2, np.log(G / m) + s2))
    pts = np.array(pts) if pts else np.zeros((0, 2))
    sel = pts[pts[:, 0] > 8] if len(pts) else pts
    cG = float(np.mean(sel[:, 1])) if len(sel) else np.nan
    return {"kind": kind, "mr0": mr0, "beta_c": bc, "points": pts, "c_G": cG,
            "spread": float(np.std(sel[:, 1])) if len(sel) else np.nan}


def main(part):
    if part == "gamow":
        cases = [("cutoff", 1e-6), ("cutoff", 1e-4), ("cutoff", 1e-2), ("cutoff", 0.0667),
                 ("height", 1e-4), ("height", 1e-2), ("height", 0.1)]
        out = []
        for kind, x in cases:
            t = time.time()
            r = gamow_case(kind, x)
            out.append(r)
            print(kind, x, "beta_c=%.5f c_G=%.3f +- %.3f (%d pts, %.0fs)"
                  % (r["beta_c"], r["c_G"], r["spread"], len(r["points"]), time.time() - t), flush=True)
        save_json({"cases": out, "c_G_limit": C_G_LIMIT}, "s8_gamow")
    elif part == "height":
        out = {}
        for d in (0.2, 0.3, 0.5):
            for model in ("bare", "rpa"):
                mat = gapped_graphene(0.015)
                out[f"{model}/{d}"] = critical_charge(make_solver(mat, d, model))
                print(model, d, out[f"{model}/{d}"], flush=True)
        save_json(out, "s8_height")
    else:
        plot()


def asymptotic_cG(points):
    """c_G from the deepest-barrier points.  c_G drifts slowly with 2S towards
    its asymptotic value, so we average the points with 2S > 8, or take the
    deepest point (flagged approximate) when none is that deep.  Points with
    c_G > 0 belong to a different pole and are discarded."""
    p = np.array([q for q in points if q[1] < 0 and q[0] > 0]) if len(points) else np.zeros((0, 2))
    if len(p) == 0:
        return np.nan, np.nan, False
    deep = p[p[:, 0] > 8]
    if len(deep):
        return float(np.mean(deep[:, 1])), float(np.std(deep[:, 1])), True
    k = int(np.argmax(p[:, 0]))
    return float(p[k, 1]), np.nan, False


def length(x):
    """m r0 for the table: powers of ten below 0.01 written as such."""
    e = np.log10(x)
    return f"$10^{{{int(round(e))}}}$" if x < 0.01 and abs(e - round(e)) < 1e-9 else f"{x:g}"


def plot():
    from common import load_json
    g = load_json("s8_gamow")
    h = load_json("s8_height")
    s4 = load_json("s4_resonances")
    fig, ax = plt.subplots(figsize=SINGLE)
    rows = ["\\begin{tabular}{lccc}", "\\hline\\hline", "core & $mr_0$ or $md$ & $\\beta_c$ & $c_G$\\\\", "\\hline"]
    summary = []
    for kind, mk, col in (("cutoff", "o", C["blue"]), ("height", "s", C["red"])):
        xs, ys, es = [], [], []
        for c in g["cases"]:
            if c["kind"] != kind:
                continue
            cG, sd, ok = asymptotic_cG(c["points"])
            summary.append((kind, c["mr0"], c["beta_c"], cG, sd, ok))
            if ok and sd > 0.005:
                cell = f"${cG:.2f}\\pm{sd:.2f}$"
            elif ok:
                cell = f"${cG:.2f}$"
            else:
                cell = f"$\\approx{cG:.2f}$"
            rows.append(f"{kind} & {length(c['mr0'])} & {c['beta_c']:.4f} & {cell}\\\\")
            if np.isfinite(cG):
                xs.append(c["beta_c"]); ys.append(cG); es.append(sd if ok else 0.0)
                if not ok:
                    ax.annotate("", xy=(c["beta_c"], cG - 0.12), xytext=(c["beta_c"], cG),
                                arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
        ax.errorbar(xs, ys, yerr=es, fmt=mk, ms=4, color=col, mfc="none", capsize=2, label=f"Coulomb tail, {kind} core")
    for key, o in s4["cases"].items():
        b = o["beta_tail"] * o["Zc"]
        pts = [(s, np.log(G) + s) for G, s, f in zip(o["G_res"], o["twoS"], o["flag"])
               if f == "siegert" and G and G > 0 and s > 8]
        if pts:
            rpa = key.startswith("rpa")
            ax.plot(b, np.mean([q[1] for q in pts]), "^" if rpa else "v", ms=4,
                    color=C["green"] if rpa else C["black"])
    ax.plot([], [], "^", color=C["green"], ms=4, label="graphene, Dirac-sea RPA")
    ax.plot([], [], "v", color=C["black"], ms=4, label="graphene, constant $\\kappa$")
    ax.plot([0.5], [g["c_G_limit"]], "*", ms=7, color=C["grey"])
    ax.text(0.53, g["c_G_limit"] + 0.01, "$\\beta_c\\to1/2$ limit", fontsize=6, color=C["grey"], va="bottom")
    ax.set_xlabel("$\\beta_c$ (strength of the Coulomb tail at $Z_c$)")
    ax.set_ylabel("$c_G=\\ln(\\Gamma/\\Delta)+2S$")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    savefig(fig, "fig_gamow_prefactor")
    rows += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_gamow", "\n".join(rows))
    rows = ["\\begin{tabular}{lccc}", "\\hline\\hline", "$d$ (nm) & 0.2 & 0.3 & 0.5\\\\", "\\hline"]
    for model, lab in (("bare", "constant $\\kappa$"), ("rpa", "Dirac-sea RPA")):
        rows.append(lab + " & " + " & ".join(f"{h[f'{model}/{d}']:.3f}" for d in (0.2, 0.3, 0.5)) + "\\\\")
    rows += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_height", "\n".join(rows))
    for r in summary:
        print(r)


if __name__ == "__main__":
    # no argument: run everything (as `make data` does); "gamow", "height", "plot": one part
    parts = sys.argv[1:] or ["gamow", "height", "plot"]
    for part in parts:
        main(part)

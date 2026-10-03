"""Fig. 3 + Table VII: diving of the j = 1/2 level and atomic-collapse resonances.

Below Z_c the level is bound (E > -Delta); above Z_c it continues as a Siegert
pole E_r - i Gamma/2 in the lower continuum.  Near threshold the width obeys the
Gamow law  ln(Gamma/Delta) = c_G - 2S,  2S = 2 pi beta (|E_r|/k - 1), where
beta is the strength of the *unscreened* Coulomb tail.
"""
import sys
import time

import numpy as np
from common import C, DOUBLE, load_json, plt, save_json, savefig, sci, write_table

from collapse2d.analysis import critical_charge, gamow_exponent, make_solver, track_resonance
from collapse2d.materials import gapped_graphene

D_IMP = 0.3
CASES = [("bare", 0.03), ("rpa", 0.03), ("bare", 0.1), ("rpa", 0.1), ("bare", 0.01), ("rpa", 0.01)]


def main():
    out = {}
    for model, gp in CASES:
        t = time.time()
        mat = gapped_graphene(gp / 2)
        S = make_solver(mat, D_IMP, model)
        Zc = critical_charge(S)
        Zb = Zc * np.linspace(0.55, 0.995, 10)
        Eb = []
        for Z in Zb:
            E = S.bound_states(Z, 0.5, n_scan=100, edge=1e-6)
            Eb.append(E[0] if len(E) else np.nan)
        Zr = Zc * np.array([1.003, 1.006, 1.01, 1.02, 1.035, 1.05, 1.075, 1.1, 1.15, 1.2, 1.25, 1.3])
        tr = track_resonance(S, Zr)
        key = f"{model}/{gp}"
        out[key] = {"Zc": Zc, "Z_bound": Zb, "E_bound": np.array(Eb) / mat.m,
                    "Z_res": [x[0] for x in tr], "E_res": [x[1] / mat.m for x in tr],
                    "G_res": [x[2] / mat.m for x in tr], "flag": [x[3] for x in tr],
                    "beta_tail": S.v.tail_beta, "m": mat.m, "Delta_eV": mat.Delta}
        out[key]["twoS"] = [gamow_exponent(S.v.tail_beta * Z, E * mat.m, mat.m) if np.isfinite(E) else np.nan
                            for Z, E in zip(out[key]["Z_res"], out[key]["E_res"])]
        print(key, "Zc=%.5f" % Zc, "%.0fs" % (time.time() - t))
        for Z, E, G, f, s in zip(out[key]["Z_res"], out[key]["E_res"], out[key]["G_res"], out[key]["flag"], out[key]["twoS"]):
            print("   Z=%.4f E/D=%.6f G/D=%.3e 2S=%.2f lnG+2S=%.3f %s" % (Z, E, G, s, np.log(G) + s if G > 0 else np.nan, f))

    # ---- Gamow constant from Siegert points with 2S > 3 ----
    pts = []
    for key, o in out.items():
        for G, s, f in zip(o["G_res"], o["twoS"], o["flag"]):
            if f == "siegert" and np.isfinite(G) and s > 3:
                pts.append((s, np.log(G) + s))
    pts = np.array(pts)
    cG = float(np.mean(pts[pts[:, 0] > 8, 1])) if np.any(pts[:, 0] > 8) else float(np.mean(pts[:, 1]))
    print("Gamow constant c_G =", cG, "spread", np.std(pts[pts[:, 0] > 8, 1]) if np.any(pts[:, 0] > 8) else 0)

    save_json({"cases": out, "gamow_c": cG, "gamow_points": pts}, "s4_resonances")
    plot()


def consistent(o):
    """Mask of resonance points that continue the tracked pole.

    The pole tracker occasionally jumps to another branch far above threshold.
    E_r must decrease and Gamma must not spike as Z grows: a point is dropped if
    it lies above the last accepted E_r, or if its width is a strict local
    maximum among its finite neighbours."""
    E = np.array(o["E_res"], float)
    G = np.array(o["G_res"], float)
    ok = np.array([f != "fail" for f in o["flag"]]) & np.isfinite(E) & np.isfinite(G)
    keep = np.zeros(len(E), bool)
    last = np.inf
    for i in np.nonzero(ok)[0]:
        prev = [k for k in range(i - 1, -1, -1) if ok[k]][:1]
        nxt = [k for k in range(i + 1, len(E)) if ok[k]][:1]
        spike = prev and nxt and G[i] > G[prev[0]] and G[i] > G[nxt[0]]
        if E[i] < last and not spike:
            keep[i] = True
            last = E[i]
    return keep


def plot():
    d = load_json("s4_resonances")
    out, cG = d["cases"], d["gamow_c"]
    # ---- figure ----
    fig, axs = plt.subplots(1, 2, figsize=DOUBLE)
    ax = axs[0]
    for model, col in (("bare", C["black"]), ("rpa", C["blue"])):
        o = out[f"{model}/0.03"]
        lab = "constant $\\kappa$" if model == "bare" else "Dirac-sea RPA"
        ax.plot(o["Z_bound"], o["E_bound"], "-", color=col, label=lab)
        ok = [i for i in np.nonzero(consistent(o))[0] if o["G_res"][i] < 2 * (abs(o["E_res"][i]) - 1)]
        Z = np.array(o["Z_res"])[ok]
        E = np.array(o["E_res"])[ok]
        G = np.array(o["G_res"])[ok]
        ax.plot(Z, E, "--", color=col)
        ax.fill_between(Z, E - G / 2, E + G / 2, color=col, alpha=0.2, lw=0)
        ax.axvline(o["Zc"], color=col, lw=0.5, ls=":")
    ax.axhspan(-1, 1, color=C["grey"], alpha=0.08, lw=0)
    ax.axhline(-1, color=C["grey"], lw=0.5)
    ax.set_xlabel("impurity charge $Z$")
    ax.set_ylabel("$E/\\Delta$")
    ax.set_ylim(-3.2, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_title("(a) $2\\Delta=30$ meV: bound level (solid), resonance (dashed, band $\\pm\\Gamma/2$)", fontsize=7)

    ax = axs[1]
    mk = {"0.01": "^", "0.03": "o", "0.1": "s"}
    for key, o in out.items():
        model, gp = key.split("/")
        col = C["black"] if model == "bare" else C["blue"]
        s = np.array(o["twoS"])
        G = np.array(o["G_res"])
        siegert_pts = np.array([f == "siegert" for f in o["flag"]])
        ok = siegert_pts & consistent(o) & (G > 0)
        ax.semilogy(s[ok], G[ok], mk[gp], ms=3.5, color=col, mfc="none" if model == "bare" else col,
                    label=f"{'bare' if model == 'bare' else 'RPA'}, {float(gp) * 1000:.0f} meV")
    ss = np.linspace(0, 40, 100)
    ax.semilogy(ss, np.exp(cG - ss), "-", color=C["red"], lw=0.8, label=f"$e^{{{cG:.2f}-2S}}$")
    ax.set_xlabel("$2S=2\\pi\\beta(|E_r|/k-1)$")
    ax.set_ylabel("$\\Gamma/\\Delta$")
    ax.set_xlim(0, 36)
    ax.set_ylim(1e-17, 2)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.set_title("(b) Gamow law for the resonance width", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_resonances")

    resonance_table(out["rpa/0.03"])


def resonance_table(o):
    m_eV = o["Delta_eV"]
    body = ["\\begin{tabular}{lcccc}", "\\hline\\hline",
            "$Z/Z_c$ & $E_r/\\Delta$ & $E_r$ (meV) & $\\Gamma$ (meV) & method\\\\", "\\hline"]
    for Z, E, G, f, ok in zip(o["Z_res"], o["E_res"], o["G_res"], o["flag"], consistent(o)):
        if not ok:
            continue
        body.append(f"{Z / o['Zc']:.3f} & ${E:.5f}$ & ${E * m_eV * 1000:.3f}$ & {width(G * m_eV * 1000)} & {f}\\\\")
    body += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_resonances", "\n".join(body))


def width(G):
    """Widths of order 1 meV as plain decimals, the rest in powers of ten."""
    return f"${G:.2f}$" if 0.1 <= G < 100 else sci(G, 2)


if __name__ == "__main__":
    if "--plot" in sys.argv or "--tables" in sys.argv:   # figure and table from saved data
        plot()
    else:
        main()

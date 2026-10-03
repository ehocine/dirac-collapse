"""Fig. 1 + Table III: critical charge of an impurity on gapped graphene.

Models (impurity at height d = 0.3 nm, graphene on hBN, kappa = 2.5):
  bare    constant dielectric screening kappa
  RPA     + static interband polarization of the gapped Dirac sea (N = 4)
  RPA+gate  + metallic gate at distance D below the layer
"""
import time

import numpy as np
from common import C, DOUBLE, load_json, plt, save_json, savefig, write_table

from collapse2d.analysis import critical_charge, make_solver
from collapse2d.materials import gapped_graphene
from collapse2d.potentials import ScreenedImpurity

D_IMP = 0.3
KAPPA = 2.5
GAPS = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])     # 2 Delta in eV
MODELS = [("bare", "bare", np.inf), ("RPA", "rpa", np.inf), ("RPA, D=30 nm", "rpa", 30.0),
          ("RPA, D=10 nm", "rpa", 10.0), ("RPA, D=3 nm", "rpa", 3.0)]


def main():
    res = {name: {"Zc1": [], "Zc2": []} for name, _, _ in MODELS}
    eps_ratio = []
    for gp in GAPS:
        mat = gapped_graphene(gp / 2, kappa=KAPPA)
        for name, model, D in MODELS:
            t = time.time()
            S = make_solver(mat, D_IMP, model, D=D)
            roots = S.critical_couplings(0.5, 0.2, 14.0, n_scan=56)
            z1 = roots[0] if len(roots) > 0 else np.nan
            z2 = roots[1] if len(roots) > 1 else np.nan
            res[name]["Zc1"].append(z1)
            res[name]["Zc2"].append(z2)
            print("2D=%.3f %-14s Zc=%.5f Zc2=%.5f (%.1fs)" % (gp, name, z1, z2, time.time() - t))
        P = ScreenedImpurity(1.0, mat, D_IMP)
        eps_ratio.append([P.eps(np.array([c * mat.m]))[0] / KAPPA for c in (0.5, 1.0, 2.0)])
    eps_ratio = np.array(eps_ratio)

    save_json({"gaps": GAPS, "results": res, "eps_ratio_at_m_2m": eps_ratio, "kappa": KAPPA, "d": D_IMP},
              "s3_dirac_sea")
    ratio = np.array(res["RPA"]["Zc1"]) / np.array(res["bare"]["Zc1"])
    print("Zc(RPA)/Zc(bare):", np.round(ratio, 3))
    print("eps(q=m/2, m, 2m)/kappa:", np.round(eps_ratio, 3))

    plot()


def plot():
    d = load_json("s3_dirac_sea")
    res = d["results"]
    gaps = np.array(d["gaps"])
    # ---- figure ----
    fig, axs = plt.subplots(1, 2, figsize=DOUBLE)
    ax = axs[0]
    rr = np.geomspace(0.05, 2000, 300)
    for gp, col in ((0.01, C["blue"]), (0.03, C["green"]), (0.1, C["orange"]), (0.3, C["red"])):
        mat = gapped_graphene(gp / 2, kappa=KAPPA)
        P = ScreenedImpurity(1.0, mat, D_IMP)
        ax.semilogx(rr, P.effective_charge(rr), color=col, label=f"$2\\Delta={gp * 1000:.0f}$ meV")
        ax.axvline(mat.compton, color=col, lw=0.5, ls=":")
    mat = gapped_graphene(0.015, kappa=KAPPA)
    Pg = ScreenedImpurity(1.0, mat, D_IMP, D=10.0)
    ax.semilogx(rr, Pg.effective_charge(rr), "--", color=C["green"], label="30 meV, gate $D=10$ nm")
    ax.axhline(KAPPA / (KAPPA + np.pi * mat.alpha0 / 2), color=C["grey"], lw=0.5, ls="--")
    ax.text(0.06, KAPPA / (KAPPA + np.pi * mat.alpha0 / 2) + 0.02, "$\\kappa/\\kappa_\\infty$", fontsize=7, color=C["grey"])
    ax.set_xlabel("$r$ (nm)")
    ax.set_ylabel("$Z_{\\rm eff}(r)/Z=-\\kappa rU(r)/(Z\\alpha_0)$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(a) Dirac-sea screening (dotted: $\\lambda_C=\\hbar v/\\Delta$)", fontsize=8)

    ax = axs[1]
    cols = [C["black"], C["blue"], C["green"], C["orange"], C["red"]]
    for (name, _, _), col in zip(MODELS, cols):
        ax.semilogx(gaps * 1000, res[name]["Zc1"], "o-", ms=3, color=col, label=name)
    mat = gapped_graphene(0.015, kappa=KAPPA)
    ax.axhline(KAPPA / (2 * mat.alpha0), color=C["grey"], lw=0.5, ls=":")
    ax.axhline((KAPPA + np.pi * mat.alpha0 / 2) / (2 * mat.alpha0), color=C["grey"], lw=0.5, ls="--")
    ax.set_xlabel("gap $2\\Delta$ (meV)")
    ax.set_ylabel("critical charge $Z_c$")
    ax.text(5.5, KAPPA / (2 * mat.alpha0) + 0.03, "$\\kappa/2\\alpha_0$", fontsize=6.5, color=C["grey"])
    ax.text(5.5, (KAPPA + np.pi * mat.alpha0 / 2) / (2 * mat.alpha0) + 0.03, "$\\kappa_\\infty/2\\alpha_0$",
            fontsize=6.5, color=C["grey"])
    ax.legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
    ax.set_ylim(0.4, 3.0)
    ax.set_title("(b) $j=1/2$ level reaches $E=-\\Delta$", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_dirac_sea")

    sel = [0.01, 0.03, 0.1, 0.3]
    body = ["\\begin{tabular}{lcccccc}", "\\hline\\hline",
            "$2\\Delta$ (meV) & $\\lambda_C$ (nm) & bare & RPA & RPA, $D=30$ nm & RPA, $D=10$ nm & RPA, $D=3$ nm\\\\", "\\hline"]
    for gp in sel:
        i = int(np.argmin(np.abs(gaps - gp)))
        mat = gapped_graphene(gp / 2)
        cells = [f"{res[n]['Zc1'][i]:.3f}" for n, _, _ in MODELS]
        body.append(f"{gp * 1000:.0f} & {mat.compton:.1f} & " + " & ".join(cells) + "\\\\")
    body += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_zc", "\n".join(body))


if __name__ == "__main__":
    import sys
    plot() if "--plot" in sys.argv else main()

"""Table VIII: critical charges for representative gapped 2D Dirac systems.

Impurity at height d = 0.3 nm.  For graphene the Dirac-sea polarization (N = 4)
is included; for the TMDs the Keldysh length r_K already contains the interband
polarization (N = 0).  Gate: metallic plane at distance D below the layer.
Gaps: graphene/hBN 15 and 30 meV bracket the measured transport gaps (16-28 meV,
Hunt et al. 2013; 31 meV, Woods et al. 2014); kappa = 2.5 uses eps_hBN = 4, the
anisotropic row uses sqrt(eps_par eps_perp) = sqrt(6.93 x 3.76) = 5.10 (Laturia et
al. 2018), kappa = 3.05; SiC: eps ~ 9.7, kappa = 5.35; gap 0.26 eV (Zhou et al. 2007).
"""
import time

import numpy as np
from common import load_json, save_json, write_table

from collapse2d.analysis import critical_charge, make_solver
from collapse2d.materials import gapped_graphene, mos2, wse2

D_IMP = 0.3
SYSTEMS = [
    ("graphene/hBN (aligned)", gapped_graphene(0.0075, kappa=2.5), "15 meV"),
    ("graphene/hBN (aligned)", gapped_graphene(0.015, kappa=2.5), "30 meV"),
    ("graphene/hBN, anisotropic $\\kappa$", gapped_graphene(0.015, kappa=3.05), "30 meV"),
    ("graphene, suspended", gapped_graphene(0.015, kappa=1.0), "30 meV"),
    ("epitaxial graphene/SiC", gapped_graphene(0.13, kappa=5.35), "0.26 eV"),
    ("MoS$_2$/hBN", mos2(kappa=2.5), "1.66 eV"),
    ("WSe$_2$/hBN", wse2(kappa=2.5), "1.60 eV"),
]


def main():
    rows = []
    for name, mat, gap in SYSTEMS:
        t = time.time()
        is_tmd = mat.N == 0
        cells = {}
        for label, model, D in (("const", "bare", np.inf), ("scr", "env" if is_tmd else "rpa", np.inf),
                                ("gate", "env" if is_tmd else "rpa", 30.0)):
            m_ = mat.with_(r_K=0.0) if (label == "const") else mat
            S = make_solver(m_, D_IMP, model, D=D)
            cells[label] = critical_charge(S, Z_lo=0.1, Z_hi=80.0, n_scan=240)
        rows.append((name, gap, mat.kappa, mat.alpha0 / mat.kappa, cells))
        print(name, gap, cells, "%.0fs" % (time.time() - t))
    save_json({"rows": [(r[0], r[1], r[2], r[3], r[4]) for r in rows]}, "s6_materials")

    table()


def table():
    rows = load_json("s6_materials")["rows"]
    body = ["\\begin{tabular}{llccccc}", "\\hline\\hline",
            "system & $2\\Delta$ & $\\kappa$ & $\\alpha_0/\\kappa$ & $Z_c$ (const.\\ $\\kappa$) & $Z_c$ (screened) & $Z_c$ (+ gate, 30 nm)\\\\",
            "\\hline"]
    for name, gap, kap, a, c in rows:
        body.append(f"{name} & {gap} & {kap} & {a:.2f} & {c['const']:.2f} & {c['scr']:.2f} & {c['gate']:.2f}\\\\")
    body += ["\\hline\\hline", "\\end{tabular}"]
    write_table("tab_materials", "\n".join(body))


if __name__ == "__main__":
    import sys
    table() if "--tables" in sys.argv else main()

"""Gate effect on the atomic-collapse resonance (2 Delta = 30 meV, Dirac-sea RPA).

With a gate the Coulomb tail, and hence the Gamow barrier, is cut off beyond
~D.  We compare the j = 1/2 resonance at the same Z/Z_c with and without a gate
(D = 10 nm) from real-axis profiles of the energy-normalised inner weight.
"""
import numpy as np
from common import C, SINGLE, load_json, plt, save_json, savefig

from collapse2d.analysis import critical_charge, make_solver, track_resonance
from collapse2d.materials import gapped_graphene

D_IMP = 0.3


def main():
    mat = gapped_graphene(0.015)
    m = mat.m
    out = {}
    for label, D, col in (("no gate", np.inf, C["blue"]), ("gate, D = 10 nm", 10.0, C["red"])):
        S = make_solver(mat, D_IMP, "rpa", D=D)
        Zc = critical_charge(S, Z_hi=14.0, n_scan=56)
        out[label] = {"Zc": Zc}
        for f in (1.1, 1.2):
            Z = f * Zc
            Eg = -m * np.linspace(1.0005, 3.5, 300)
            r_in = 3.0 if np.isinf(D) else 3.0
            w = S.continuum(Eg, Z, 0.5, r_in=r_in, n_waves=40)["inner"]
            res = S.resonances(Z, 0.5, -3.5 * m, -1.0005 * m, r_in=r_in)
            if np.isinf(D):
                tr = track_resonance(S, [Z])
                pole = {"E_r": tr[0][1] / m, "Gamma": tr[0][2] / m, "flag": tr[0][3]}
            else:
                pole = None
            out[label][f"{f}"] = {"E": Eg / m, "w": w,
                                  "fits": [{"E_r": r_["E_r"] / m, "Gamma": r_["Gamma"] / m} for r_ in res],
                                  "pole": pole}
            print(label, "Z/Zc", f, "Zc", Zc, "fits", out[label][f"{f}"]["fits"], "pole", pole)
    save_json(out, "s7_gate_resonance")
    plot()


def fwhm(E, w):
    i = int(np.argmax(w))
    h = w[i] / 2
    lo, hi = i, i
    while lo > 0 and w[lo] > h:
        lo -= 1
    while hi < len(w) - 1 and w[hi] > h:
        hi += 1
    return E[i], E[hi] - E[lo], (lo == 0 or hi == len(w) - 1)


def plot():
    o = load_json("s7_gate_resonance")
    fig, ax = plt.subplots(figsize=SINGLE)
    summary = {}
    for (label, col) in (("no gate", C["blue"]), ("gate, D = 10 nm", C["red"])):
        for f, ls in (("1.1", "-"), ("1.2", "--")):
            E = np.array(o[label][f]["E"])[::-1]
            w = np.array(o[label][f]["w"])[::-1]
            Ep, G, edge = fwhm(E, w)
            summary[f"{label}/{f}"] = {"E_peak": Ep, "FWHM": G, "edge": edge}
            ax.plot(E, w / np.max(w), ls, color=col, lw=0.9, label=f"{label}, $Z/Z_c={f}$")
            print(label, f, "peak %.3f FWHM %.3f" % (Ep, G), "(at range edge)" if edge else "")
    ax.set_xlabel("$E/\\Delta$")
    ax.set_ylabel("inner weight $w(E)/w_{\\max}$")
    ax.set_xlim(-3.5, -1.0)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    savefig(fig, "fig_gate_resonance")
    save_json({**o, "summary": summary}, "s7_gate_resonance")


if __name__ == "__main__":
    import sys
    plot() if "--plot" in sys.argv else main()

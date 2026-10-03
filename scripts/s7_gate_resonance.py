"""Fig. 5: gate effect on the atomic-collapse resonance (2 Delta = 30 meV, Dirac-sea RPA).

With a gate the Coulomb tail, and hence the Gamow barrier, is cut off beyond
~D.  We compare the j = 1/2 resonance at the same Z/Z_c with and without a gate
(D = 10 nm) from real-axis profiles of the energy-normalised inner weight.
Widths are compared as full widths at half maximum of these profiles; the grid is
refined around each peak so that the narrow ungated resonance is resolved.
"""
import numpy as np
from common import C, SINGLE, load_json, plt, save_json, savefig

from collapse2d.analysis import critical_charge, make_solver, track_resonance
from collapse2d.materials import gapped_graphene

D_IMP = 0.3
R_IN = 3.0
E_MAX = 5.0          # profiles cover -E_MAX Delta < E < -Delta


def profile(S, Z, m):
    """Inner weight on a coarse grid, refined around the peak."""
    Eg = -m * np.linspace(1.0005, E_MAX, 400)
    w = S.continuum(Eg, Z, 0.5, r_in=R_IN, n_waves=40)["inner"]
    i = int(np.argmax(w))
    dE = abs(Eg[1] - Eg[0])
    Ef = np.linspace(Eg[i] - 4 * dE, Eg[i] + 4 * dE, 161)
    Ef = Ef[(Ef < -1.0005 * m) & (Ef > -E_MAX * m)]
    wf = S.continuum(Ef, Z, 0.5, r_in=R_IN, n_waves=40)["inner"]
    E = np.concatenate([Eg, Ef])
    w = np.concatenate([w, wf])
    o = np.argsort(E)
    return E[o], w[o]


def main():
    mat = gapped_graphene(0.015)
    m = mat.m
    out = {}
    for label, D in (("no gate", np.inf), ("gate, D = 10 nm", 10.0)):
        S = make_solver(mat, D_IMP, "rpa", D=D)
        Zc = critical_charge(S, Z_hi=14.0, n_scan=56)
        out[label] = {"Zc": Zc}
        for f in (1.1, 1.2):
            Z = f * Zc
            E, w = profile(S, Z, m)
            res = S.resonances(Z, 0.5, -E_MAX * m, -1.0005 * m, r_in=R_IN)
            if np.isinf(D):
                tr = track_resonance(S, [Z])
                pole = {"E_r": tr[0][1] / m, "Gamma": tr[0][2] / m, "flag": tr[0][3]}
            else:
                pole = None
            out[label][f"{f}"] = {"E": E / m, "w": w,
                                  "fits": [{"E_r": r_["E_r"] / m, "Gamma": r_["Gamma"] / m} for r_ in res],
                                  "pole": pole}
            print(label, "Z/Zc", f, "Zc", Zc, "fits", out[label][f"{f}"]["fits"], "pole", pole)
    save_json(out, "s7_gate_resonance")
    plot()


def fwhm(E, w):
    """Peak position and full width at half maximum (linear interpolation);
    the flag is True if a half-maximum point lies outside the grid."""
    i = int(np.argmax(w))
    h = w[i] / 2
    lo, hi = i, i
    while lo > 0 and w[lo] > h:
        lo -= 1
    while hi < len(w) - 1 and w[hi] > h:
        hi += 1
    edge = w[lo] > h or w[hi] > h
    El = E[lo] + (h - w[lo]) * (E[lo + 1] - E[lo]) / (w[lo + 1] - w[lo]) if w[lo] <= h else E[lo]
    Eh = E[hi - 1] + (h - w[hi - 1]) * (E[hi] - E[hi - 1]) / (w[hi] - w[hi - 1]) if w[hi] <= h else E[hi]
    return E[i], Eh - El, edge


def plot():
    o = load_json("s7_gate_resonance")
    fig, ax = plt.subplots(figsize=SINGLE)
    summary = {}
    for (label, col) in (("no gate", C["blue"]), ("gate, D = 10 nm", C["red"])):
        for f, ls in (("1.1", "-"), ("1.2", "--")):
            E = np.array(o[label][f]["E"])
            w = np.array(o[label][f]["w"])
            Ep, G, edge = fwhm(E, w)
            summary[f"{label}/{f}"] = {"E_peak": Ep, "FWHM": G, "edge": edge}
            ax.plot(E, w / np.max(w), ls, color=col, lw=0.9, label=f"{label}, $Z/Z_c={f}$")
            pole = o[label][f]["pole"]
            print(label, f, "peak %.3f FWHM %.4f" % (Ep, G), "(at range edge)" if edge else "",
                  "Siegert Gamma %.4f" % pole["Gamma"] if pole else "")
    ax.set_xlabel("$E/\\Delta$")
    ax.set_ylabel("inner weight $w(E)/w_{\\max}$")
    ax.set_xlim(-E_MAX, -1.0)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    savefig(fig, "fig_gate_resonance")
    o.pop("summary", None)
    save_json({**o, "summary": summary}, "s7_gate_resonance")


if __name__ == "__main__":
    import sys
    plot() if "--plot" in sys.argv else main()

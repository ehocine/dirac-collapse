"""Fig. 5: local density of states around the impurity (STM observable).

Gapped graphene, 2 Delta = 100 meV, kappa = 2.5, d = 0.3 nm, Dirac-sea RPA.
Total LDOS = 4 (spin x valley) x single-flavour LDOS summed over j.
Continuum states are energy normalised; bound states are added as Lorentzians;
the whole spectrum is convolved with a Lorentzian of half-width ETA (STM
resolution).
"""
import time

import numpy as np
from common import C, load_json, plt, save_json, savefig

from collapse2d.analysis import critical_charge, ldos, make_solver
from collapse2d.materials import gapped_graphene

GAP = 0.10
D_IMP = 0.3
ETA = 0.003            # eV
J_MAX = 6.5


def spectrum(S, mat, Z, E_eV, r_list):
    """Total LDOS (states / eV / nm^2) on the energy grid E_eV at radii r_list."""
    m = mat.m
    Eint = E_eV / mat.hbar_v
    bound = {}
    for j in np.concatenate([-np.arange(0.5, 3.6, 1.0), np.arange(0.5, 3.6, 1.0)]):
        Eb = S.bound_states(Z, j, n_scan=100, edge=1e-4)
        if len(Eb):
            bound[j] = Eb
    eta_int = ETA / mat.hbar_v
    bnd = ldos(S, Z, Eint, r_list, eta=eta_int, bound_E=bound, continuum=False)
    cont = ldos(S, Z, Eint, r_list, j_max=J_MAX, n_waves=50)
    # broaden the continuum with the same Lorentzian
    dE = Eint[1] - Eint[0]
    kern_x = np.arange(-60, 61) * dE
    kern = (eta_int / np.pi) / (kern_x ** 2 + eta_int ** 2) * dE
    cont_b = np.array([np.convolve(cont[:, i], kern, mode="same") for i in range(len(r_list))]).T
    total = bnd + cont_b
    # units: per flavour, hbar v = 1 -> states / (nm^-1 * nm^2); convert to /eV/nm^2 and x4 flavours
    return 4.0 * total / mat.hbar_v, bound


def main():
    mat = gapped_graphene(GAP / 2)
    S = make_solver(mat, D_IMP, "rpa")
    Zc = critical_charge(S)
    print("Zc =", Zc)
    E_eV = np.linspace(-0.30, 0.10, 201)
    r_map = list(np.round(np.linspace(0.4, 20.0, 50), 6))
    out = {"Zc": Zc, "E": E_eV, "r": r_map}
    maps = {}
    for Z in (1.2, round(1.25 * Zc, 3)):
        t = time.time()
        rho, bound = spectrum(S, mat, Z, E_eV, r_map)
        maps[Z] = rho
        out[f"map_{Z}"] = rho
        out[f"bound_{Z}"] = {str(k): v * mat.hbar_v for k, v in bound.items()}
        print("map Z=%.3f done %.0fs" % (Z, time.time() - t))
    # point spectra at r = 1 nm for a sequence of charges
    Zs = [0.0, 0.8, 1.4, round(Zc, 2), round(1.15 * Zc, 2), round(1.3 * Zc, 2)]
    spec = {}
    for Z in Zs:
        t = time.time()
        rho, _ = spectrum(S, mat, max(Z, 1e-6), E_eV, [1.0])
        spec[Z] = rho[:, 0]
        print("spectrum Z=%.2f %.0fs" % (Z, time.time() - t))
    out["spectra"] = {str(k): v for k, v in spec.items()}

    save_json(out, "s5_ldos")
    plot()


def plot():
    o = load_json("s5_ldos")
    Zc, E_eV, r_map = o["Zc"], np.array(o["E"]), o["r"]
    maps = {float(k[4:]): np.array(v) for k, v in o.items() if k.startswith("map_")}
    spec = {float(k): np.array(v) for k, v in o["spectra"].items()}
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.7, 1.1], wspace=0.12)
    for k, (Z, rho) in enumerate(maps.items()):
        ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(r_map, E_eV * 1000, np.log10(np.maximum(rho, 1e-4)), shading="auto",
                           cmap="magma", vmin=-2.5, vmax=0.5)
        ax.axhline(-GAP / 2 * 1000, color="w", lw=0.4, ls=":")
        ax.axhline(GAP / 2 * 1000, color="w", lw=0.4, ls=":")
        ax.set_xlabel("$r$ (nm)")
        if k == 0:
            ax.set_ylabel("$E$ (meV)")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"({'ab'[k]}) $Z={Z:.2f}$ ($Z/Z_c={Z / Zc:.2f}$)", fontsize=8)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\log_{10}$ LDOS (eV$^{-1}$nm$^{-2}$)", fontsize=7)
    ax = fig.add_subplot(gs[0, 4])
    off = 0.0
    cols = [C["grey"], C["black"], C["blue"], C["green"], C["orange"], C["red"]]
    for (Z, rho), col in zip(spec.items(), cols):
        ax.plot(E_eV * 1000, rho + off, color=col, lw=0.9, label=f"$Z={Z:.2f}$")
        off += 0.6
    ax.axvspan(-GAP / 2 * 1000, GAP / 2 * 1000, color=C["grey"], alpha=0.1, lw=0)
    ax.set_xlabel("$E$ (meV)")
    ax.set_ylabel("LDOS, $r=1$ nm (offset)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_title("(c) model $dI/dV$ spectra", fontsize=8)
    savefig(fig, "fig_ldos")


if __name__ == "__main__":
    import sys
    plot() if "--plot" in sys.argv else main()

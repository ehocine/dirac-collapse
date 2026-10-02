"""Fig. 2 + Table II: the critical coupling for a regularised Coulomb centre.

beta_c(m r0) for the cutoff model (exact Bessel matching, mpmath) and for a
charge at height d (numerical), compared with the leading-log law
    beta_c = sqrt(1/4 + pi^2 / [ln(1/(m r0)) + c]^2),
c_cutoff = 2 J0(1/2)/(J0(1/2) - J1(1/2)) - 2 C_E,   c_height = ln 8 - 2 C_E.
Panel (c) compares 2 beta_c(gap) for d = 0.5 nm with the empirical tight-binding
fit Z_c/Z_c(0) = (1 + 29.8 gap[eV])^0.38 of Wang et al. (arXiv:2501.15049).
"""
import numpy as np
from common import C, DOUBLE, plt, save_json, savefig, write_table

from collapse2d import exact
from collapse2d.analysis import C_CUTOFF, C_HEIGHT, beta_c_asymptotic, core_constant
from collapse2d.dirac import RadialDirac
from collapse2d.potentials import HeightCoulomb


def main():
    mr = np.logspace(-10, -0.5, 20)
    cut, hgt = [], []
    for x in mr:
        g_guess = np.pi / (np.log(1 / x) + C_CUTOFF)
        hi = min(np.sqrt(0.25 + (3 * g_guess) ** 2), 4.0)
        cut.append(exact.cutoff_critical_coupling(0.5, x, 1.0, lo=0.5 * (1 + 1e-12), hi=hi, n_scan=300)[0])
        if x >= 1e-7:
            S = RadialDirac(HeightCoulomb(1.0, 1.0), x, r_core=1.0)
            hgt.append(S.critical_couplings(0.5, 0.5 + 1e-9, min(np.sqrt(0.25 + (3 * g_guess) ** 2), 4.0), n_scan=90)[0])
        else:
            hgt.append(np.nan)
        print("m r0 %.1e  cutoff %.10f  height %.10f" % (x, cut[-1], hgt[-1]))
    cut, hgt = np.array(cut), np.array(hgt)
    a_height_num = core_constant(HeightCoulomb(1.0, 1.0), 1.0)
    print("a_in(height) numerical", a_height_num, "ln 8 =", np.log(8))

    # empirical next-order coefficient: pi/gamma = ln(1/m r0) + c - c2 gamma^2
    def c2_fit(b, x):
        g = np.sqrt(b ** 2 - 0.25)
        ok = np.isfinite(g) & (x < 1e-3)
        return np.polyfit(g[ok] ** 2, (np.pi / g - np.log(1 / x))[ok], 1)
    fit_cut = c2_fit(cut, mr)
    fit_hgt = c2_fit(hgt, mr)
    print("fit cutoff (slope, intercept)", fit_cut, "analytic c", C_CUTOFF)
    print("fit height (slope, intercept)", fit_hgt, "analytic c", C_HEIGHT)

    # Wang et al. comparison (gap in eV, d = 0.5 nm, hbar v = 0.6582 eV nm)
    hv, d = 0.6582, 0.5
    gaps = np.linspace(0.02, 0.5, 13)
    wang = []
    for gp in gaps:
        S = RadialDirac(HeightCoulomb(1.0, d), (gp / 2) / hv, r_core=d)
        wang.append(2 * S.critical_couplings(0.5, 0.5 + 1e-9, 2.5, n_scan=80)[0])
    wang = np.array(wang)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax = axs[0]
    xx = np.logspace(-10, -0.5, 300)
    ax.semilogx(mr, cut, "o", ms=3, color=C["blue"], mfc="none", label="cutoff, exact")
    ax.semilogx(mr, hgt, "s", ms=3, color=C["red"], mfc="none", label="height $d$, numerical")
    ax.semilogx(xx, beta_c_asymptotic(xx, C_CUTOFF), "-", color=C["blue"], lw=0.8, label="leading-log law, cutoff")
    ax.semilogx(xx, beta_c_asymptotic(xx, C_HEIGHT), "--", color=C["red"], lw=0.8, label="leading-log law, height")
    ax.axhline(0.5, color=C["grey"], lw=0.5, ls=":")
    ax.set_xlabel("$m r_0$  ($m d$)")
    ax.set_ylabel("$\\beta_c$")
    ax.set_title("(a)", fontsize=8)
    ax.legend(frameon=False, fontsize=6)

    ax = axs[1]
    for b, col, mk, c in ((cut, C["blue"], "o", C_CUTOFF), (hgt, C["red"], "s", C_HEIGHT)):
        g = np.sqrt(b ** 2 - 0.25)
        ax.plot(g ** 2, np.pi / g - np.log(1 / mr), mk, ms=3, color=col, mfc="none")
        ax.axhline(c, color=col, lw=0.6, ls="--")
    ax.set_xlabel("$\\gamma_c^2=\\beta_c^2-1/4$")
    ax.set_ylabel("$\\pi/\\gamma_c-\\ln(1/m r_0)$")
    ax.set_xlim(0, 0.35)
    ax.set_title("(b) approach to the constants $c$", fontsize=8)

    ax = axs[2]
    gg = np.linspace(0.0, 0.5, 200)
    ax.plot(gg, (1 + 29.8 * gg) ** 0.38, "-", color=C["grey"], lw=1.0, label="TB fit, Wang et al.")
    ax.plot(gaps, wang, "o", ms=3, color=C["red"], label="continuum, $d=0.5$ nm")
    ax.plot(gg[1:], 2 * beta_c_asymptotic((gg[1:] / 2) / hv * d, C_HEIGHT), ":", color=C["red"], lw=0.8,
            label="leading-log law")
    ax.set_xlabel("gap $2\\Delta$ (eV)")
    ax.set_ylabel("$Z_c(\\Delta)/Z_c(0)=2\\beta_c$")
    ax.set_title("(c)", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig_critical_law")

    body = ["\\begin{tabular}{lccc}", "\\hline\\hline",
            "regularisation & $a_{\\rm in}$ & $c=a_{\\rm in}-2C_E$ & fitted $c$ (intercept) \\\\", "\\hline",
            f"cutoff, $V=-\\beta/\\max(r,r_0)$ & $2J_0(\\frac12)/[J_0(\\frac12)-J_1(\\frac12)]={C_CUTOFF + 2 * 0.5772156649:.6f}$ & {C_CUTOFF:.6f} & {fit_cut[1]:.4f}\\\\",
            f"height, $V=-\\beta/\\sqrt{{r^2+d^2}}$ & $\\ln 8={np.log(8):.6f}$ (num.\\ {a_height_num:.10f}) & {C_HEIGHT:.6f} & {fit_hgt[1]:.4f}\\\\",
            "\\hline\\hline", "\\end{tabular}"]
    write_table("tab_constants", "\n".join(body))
    save_json({"m_r0": mr, "cutoff": cut, "height": hgt, "a_height_numerical": a_height_num,
               "fit_cutoff": fit_cut, "fit_height": fit_hgt, "c_cutoff": C_CUTOFF, "c_height": C_HEIGHT,
               "wang_gaps": gaps, "wang_2betac": wang}, "s2_critical_law")


if __name__ == "__main__":
    main()

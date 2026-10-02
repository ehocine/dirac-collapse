"""Regression tests: every closed-form statement used in the paper."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collapse2d import exact  # noqa: E402
from collapse2d.analysis import (C_CUTOFF, C_HEIGHT, bound_density, core_constant,  # noqa: E402
                                 gamow_exponent, make_solver)
from collapse2d.dirac import RadialDirac, siegert  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402
from collapse2d.potentials import (CutoffCoulomb, HeightCoulomb, PointCoulomb,  # noqa: E402
                                   ScreenedImpurity, polarization)


def test_polarization_limits():
    q = np.array([1e-4, 1e-2, 0.3, 5.0])
    assert np.allclose(polarization(q, 0.0), q / 16)
    m = 0.2
    small = polarization(np.array([1e-5]), m)[0]
    assert abs(small / (1e-10 / (12 * np.pi * m)) - 1) < 1e-8
    big = polarization(np.array([1e4]), m)[0]
    assert abs(big / (1e4 / 16) - 1) < 1e-6
    # series / closed-form crossover is continuous
    x = 2 * m * np.array([0.999e-3, 1.001e-3])
    p = polarization(x, m)
    assert abs(p[1] / p[0] - (x[1] / x[0]) ** 2) < 1e-6


def test_screened_potential_limits():
    mat = gapped_graphene(0.015)
    d = 0.3
    r = np.array([0.05, 2.0, 50.0, 800.0])
    P = ScreenedImpurity(1.0, mat.with_(Delta=1e-12), d)
    assert np.allclose(P(r), -mat.alpha0 / (P.kappa_inf * np.hypot(r, d)), rtol=1e-7)
    G = ScreenedImpurity(1.0, mat.with_(N=0), d, D=20.0)
    ex = -mat.alpha0 / mat.kappa * (1 / np.hypot(r, d) - 1 / np.hypot(r, d + 40))
    assert np.allclose(G(r), ex, rtol=1e-6)
    R = ScreenedImpurity(1.0, mat, d)
    assert abs(R.tail_beta / (mat.alpha0 / mat.kappa) - 1) < 1e-6      # long-range charge unscreened


@pytest.mark.parametrize("j", [0.5, -0.5, 1.5])
def test_dirac_coulomb_levels(j):
    S = RadialDirac(PointCoulomb(1.0), 1.0, r_core=1.0)
    E = np.sort(S.bound_states(0.3, j, edge=1e-3))[:3]
    ex = np.sort(exact.dirac_coulomb_levels(0.3, j, 6))
    ex = ex[1:] if j < 0 else ex
    assert np.allclose(E, ex[:3], atol=1e-9)


def test_cutoff_critical_coupling_pereira():
    ex = exact.cutoff_critical_coupling(0.5, 0.0667, 1.0, hi=2.0)[0]
    assert abs(ex - 0.94912768) < 1e-7           # Pereira, Kotov & Castro Neto: 0.949
    S = RadialDirac(CutoffCoulomb(1.0, 1.0), 0.0667, r_core=1.0)
    num = S.critical_couplings(0.5, 0.55, 1.5, n_scan=40)[0]
    assert abs(num - ex) < 1e-8


def test_core_constants():
    assert abs(core_constant(HeightCoulomb(1.0, 1.0), 1.0) - np.log(8)) < 1e-9
    assert abs(C_HEIGHT - (np.log(8) - 2 * 0.5772156649015329)) < 1e-12
    # the leading-log law approaches the exact result as m r0 -> 0
    b = exact.cutoff_critical_coupling(0.5, 1e-8, 1.0, lo=0.5 * (1 + 1e-12), hi=0.6, n_scan=200)[0]
    g = np.sqrt(b * b - 0.25)
    assert abs(np.pi / g - np.log(1e8) - C_CUTOFF) < 0.05


def test_free_ldos_sum_rule():
    class Zero:
        kinks = ()
        tail_beta = 0.0

        def __call__(self, r):
            return np.zeros_like(np.asarray(r, float))
    S = RadialDirac(Zero(), 1.0, r_core=1.0)
    E0 = np.array([-1.5, 2.0])
    js = np.arange(-15.5, 16, 1.0)
    out = S.continuum(np.repeat(E0, len(js)), 0.0, np.tile(js, len(E0)), record_r=[1.0], n_waves=60)
    F, G = out["records"][1.0]
    dens = ((F ** 2 + G ** 2) / (2 * np.pi)).reshape(len(E0), len(js)).sum(1)
    assert np.allclose(dens, np.abs(E0) / (2 * np.pi), rtol=1e-4)


def test_bound_density_normalised():
    S = RadialDirac(HeightCoulomb(1.0, 0.3), 0.05, r_core=0.3)
    E = S.bound_states(0.9, 0.5, edge=1e-3)[0]
    r = np.geomspace(1e-4, 4000, 3000)
    assert abs(np.trapezoid(bound_density(S, E, 0.9, 0.5, list(r)), r) - 1) < 1e-4


def test_siegert_matches_real_axis_fit_and_is_theta_independent():
    m = 0.0667
    S = RadialDirac(CutoffCoulomb(1.0, 1.0), m, r_core=1.0)
    E1, G1 = siegert(S, 1.0, 0.5, -1.2264 * m, 2.45e-3 * m, theta=0.3)
    E2, G2 = siegert(S, 1.0, 0.5, -1.2264 * m, 2.45e-3 * m, theta=0.6)
    assert abs(G1 / G2 - 1) < 1e-8 and abs(E1 - E2) < 1e-12
    assert abs(G1 / m - 2.4528e-3) / 2.4528e-3 < 1e-3      # Lorentzian fit on the real axis


def test_gamow_exponent_limits():
    m, beta = 1.0, 1.0
    assert gamow_exponent(beta, -1.0 - 1e-8, m) > 1e3          # infinite barrier at threshold
    assert abs(gamow_exponent(beta, -1e6, m)) < 1e-5            # vanishes deep in the continuum


def test_critical_charge_rpa_larger_than_bare():
    mat = gapped_graphene(0.015)
    zb = make_solver(mat, 0.3, "bare").critical_couplings(0.5, 0.3, 4.0, n_scan=30)[0]
    zr = make_solver(mat, 0.3, "rpa").critical_couplings(0.5, 0.3, 6.0, n_scan=30)[0]
    assert abs(zb - 0.84833) < 1e-4 and abs(zr - 1.67139) < 1e-4


def test_critical_couplings_match_kuleshov_2015():
    """Kuleshov et al., JETP Lett. 101, 264 (2015), Table 1: cut-off radius R = 1/(25 m)."""
    for j, ref in ((0.5, (0.87, 1.54)), (-0.5, (1.09, 1.82))):
        ex = exact.cutoff_critical_coupling(j, 1 / 25, 1.0, hi=2.5, n_scan=400)[:2]
        assert np.allclose(ex, ref, atol=0.006)

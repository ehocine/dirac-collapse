# Atomic collapse in gapped 2D Dirac materials

Paper, code and data for

> E. Hocine, *Atomic collapse in gapped two-dimensional Dirac materials: Dirac-sea
> screening, gates, and resonance widths* (manuscript, 2026; REVTeX/PRB format)

Repository: https://github.com/ehocine/dirac-collapse · License: MIT (see `LICENSE`) · Citation: `CITATION.cff`

## Quick start

```bash
pip install -r requirements.txt     # numpy, scipy, mpmath, matplotlib, pytest
make test                           # 13 regression tests (~1.5 min)
# the scripts are independent; to use several cores:
# cd scripts && for s in s*_*.py; do python3 $s & done; wait
make data                           # all tables, figures, raw data (~1.5 h sequential on a laptop)
make paper                          # compiles paper/main.tex if present (manuscript not included)
```

## Layout

```
collapse2d/
  materials.py    material/environment parameters (gapped graphene, MoS2, WSe2), units
  potentials.py   interband polarization Pi(q) of gapped Dirac fermions; cutoff, height and
                  point Coulomb; linear-response screened impurity (RPA + Keldysh + gate)
                  via a subtracted Hankel transform
  dirac.py        radial 2D Dirac solver: RK4 on u = ln r + r/L, vectorised over E, Z and j;
                  bound states, critical couplings, energy-normalised continuum, real-axis
                  resonance fits, Siegert poles by exterior complex rotation
  exact.py        exact Dirac-Coulomb levels; exact threshold condition of the cutoff model (mpmath)
  analysis.py     critical-coupling law and its constants, Gamow exponent, resonance tracking,
                  bound-state densities, LDOS
scripts/
  s1_validation.py        Tables 1, 2   validation against closed forms; Siegert vs real-axis fits
  s2_critical_law.py      Fig. 1, Table 3  logarithmic law, constants, comparison with Wang et al. 2025
  s3_dirac_sea.py         Fig. 2, Table 4  Z_c vs gap: bare / Dirac-sea RPA / gates
  s4_resonances.py        Fig. 3, Table 5  diving level, Siegert widths, Gamow law
  s7_gate_resonance.py    Fig. 4           resonance broadening by a metallic gate
  s5_ldos.py              Fig. 5           LDOS maps and model STM spectra (`--plot` replots from data)
  s6_materials.py         Table 8          material survey (incl. anisotropic-hBN sensitivity row)
  s8_robustness.py        Fig. 4, Tables 4, 6  prefactor c_G vs beta_c; Z_c vs impurity height
tests/test_collapse2d.py
data/            raw results (JSON) and run logs
paper/           output folder: generated tables/ and figures/ (the manuscript itself is not
                 distributed with the code)
```

## Main results

* The critical coupling of the j = 1/2 level obeys the law of Gamayun et al. (2009)
  `beta_c^2 = 1/4 + pi^2 / [ln(hbar v / (Delta r0)) + c]^2`. The constant `c` depends on the core:
  * cutoff core: `c = 2 J0(1/2)/(J0(1/2) - J1(1/2)) - 2 C_E = 1.5415`
  * charge at height d: `c = ln 8 - 2 C_E = 0.9250`, from an exact hypergeometric core solution.

  This law explains the sublinear gap dependence of the tight-binding critical charges of Wang et al.
  (2025) and matches their empirical fit `(1 + 29.8 Delta)^0.38` within 2 % for gaps >= 0.3 eV.
* The gapped Dirac sea screens like a Keldysh layer with length `N alpha0 hbar v/(6 Delta)`. This
  doubles `Z_c` of gapped graphene on hBN (30 meV gap: 0.85 -> 1.67). A gate closer than
  `hbar v/Delta` raises it further and makes it almost gap-independent.
* Resonance widths follow `Gamma = Delta exp(c_G - 2S)`, where `2S = 2 pi beta (|E|/k - 1)` holds for
  any coupling. For an unscreened Coulomb tail, c_G is a function of beta_c alone, independent of
  the core shape. It rises from -2.0 at beta_c = 0.58 to -1.45 at beta_c = 1.17 and tends to
  Gamayun et al.'s ln(3 pi/4) - pi at beta_c -> 1/2. With Dirac-sea screening, c_G ~ -1.7.
* Nonlinear vacuum polarization can only raise Z_c, by at most ~0.34 (bound from Terekhov et
  al. 2008, whose one-loop limit coincides with ours).

## Citation

If you use this code, please cite the accompanying paper (see `CITATION.cff`).

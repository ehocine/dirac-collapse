"""Material and environment parameters.

Internal units: hbar*v = 1, lengths in nm, energies in nm^-1.  An energy E
in these units corresponds to E * hbar_v eV.
"""
from dataclasses import dataclass, replace

E2 = 1.43996448  # e^2/(4 pi eps0) in eV nm


@dataclass(frozen=True)
class Material:
    """Gapped 2D Dirac material on/in a dielectric environment.

    name     label
    hbar_v   hbar * v_F in eV nm
    Delta    Dirac mass = half the band gap, in eV
    kappa    environmental dielectric constant (eps_top + eps_bottom)/2
    r_K      Rytova-Keldysh screening length of the layer's *other* bands (nm),
             entering eps(q) = kappa + r_K q + ...  (0 for graphene)
    N        number of Dirac flavours whose interband polarization screens the
             impurity (N = 4 for graphene: spin x valley; 0 to switch it off)
    """
    name: str
    hbar_v: float
    Delta: float
    kappa: float = 1.0
    r_K: float = 0.0
    N: int = 4

    @property
    def alpha0(self):
        """Vacuum 'fine-structure constant' e^2/(4 pi eps0 hbar v)."""
        return E2 / self.hbar_v

    @property
    def m(self):
        """Dirac mass in nm^-1."""
        return self.Delta / self.hbar_v

    @property
    def compton(self):
        """Compton wavelength hbar v / Delta in nm."""
        return self.hbar_v / self.Delta

    def beta(self, Z):
        """Bare coupling Z e^2/(kappa hbar v) for an impurity of charge Z."""
        return Z * self.alpha0 / self.kappa

    def eV(self, E):
        return E * self.hbar_v

    def with_(self, **kw):
        return replace(self, **kw)


# graphene: v_F = 1.0e6 m/s; kappa = (1 + eps_hBN)/2 for graphene on hBN, eps_hBN ~ 4
def gapped_graphene(Delta, kappa=2.5, N=4, hbar_v=0.6582):
    return Material("gapped graphene", hbar_v, Delta, kappa, 0.0, N)


# monolayer MoS2 as a massive Dirac system: Xiao et al. PRL 108, 196802 (2012)
# (a = 3.193 A, t = 1.10 eV, so hbar v = a t = 0.351 eV nm; gap 1.66 eV);
# Keldysh length r0 = 41.47 A from
# Berkelbach, Hybertsen & Reichman, PRB 88, 045318 (2013).  N = 0: the Keldysh
# length already contains the interband polarization.
def mos2(kappa=4.5):
    return Material("MoS2", 0.3512, 0.83, kappa, 4.147, 0)


# monolayer WSe2: Xiao et al. (a = 3.310 A, t = 1.19 eV, gap 1.60 eV);
# Keldysh length 45.11 A (Berkelbach et al. 2013)
def wse2(kappa=2.5):
    return Material("WSe2", 0.3939, 0.80, kappa, 4.511, 0)

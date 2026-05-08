"""
Bose-Fermi Mixture Density-Profile Solver
=========================================

P452 Project 2 - Phase 2 Checkpoint 2.2

Coupled Thomas-Fermi (bosons) + Local-Density-Approximation (fermions)
for a co-trapped BEC + degenerate Fermi gas in spherical harmonic traps.

Equations
---------
    mu_B = V_B(r) + g_B  n_B(r) + g_BF n_F(r)
    mu_F = V_F(r) + E_F[n_F(r)] + g_BF n_B(r),    E_F[n] = (h^2 / 2 m_F) (6 pi^2 n)^{2/3}

Inverting,
    n_B(r) = max(0, (mu_B - V_B - g_BF n_F) / g_B )
    n_F(r) = max(0, [(mu_F - V_F - g_BF n_B) / K_F]^{3/2}),
                K_F = (6 pi^2)^{2/3} / (2 alpha)        with alpha = m_F / m_B.

mu_B and mu_F are fixed by  integrate(4 pi r^2 n_X dr) = N_X.

Units (boson harmonic-oscillator units)
---------------------------------------
    h = m_B = omega_B = 1
    length:   a_ho_B = sqrt(h / (m_B omega_B))
    energy:   h omega_B
    density:  1 / a_ho_B^3

In these units
    V_B(r) = r^2 / 2
    V_F(r) = (m_F/m_B) (omega_F/omega_B)^2  r^2 / 2
    g_B    = 4 pi  (a_B  / a_ho_B)
    g_BF   = 2 pi  (a_BF / a_ho_B)  (1 + alpha) / alpha,   alpha = m_F/m_B
    K_F    = (6 pi^2)^{2/3} / (2 alpha)

Stability matrix
----------------
    M = [[ d mu_B / d n_B, d mu_B / d n_F ],
         [ d mu_F / d n_B, d mu_F / d n_F ]]
      = [[ g_B, g_BF ],
         [ g_BF, (2/3) E_F / n_F ]]
det(M) > 0  <=>  g_B * (2/3) E_F[n_F] / n_F  >  g_BF^2
The most-stringent point is the peak n_F (trap centre).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
#  unit conversions:  scattering lengths (in a_ho_B) -> dimensionless g
# ---------------------------------------------------------------------------

def g_B_from_a(a_B_over_aho: float) -> float:
    """g_B  =  4 pi  (a_B / a_ho_B)."""
    return 4.0 * np.pi * a_B_over_aho


def g_BF_from_a(a_BF_over_aho: float, mass_ratio: float) -> float:
    """g_BF = 2 pi (a_BF/a_ho_B) (1+alpha)/alpha,   alpha = m_F/m_B."""
    return 2.0 * np.pi * a_BF_over_aho * (1.0 + mass_ratio) / mass_ratio


def fermi_K(mass_ratio: float) -> float:
    """Coefficient K_F such that  E_F[n] = K_F * n^{2/3}  in boson HO units."""
    return (6.0 * np.pi ** 2) ** (2.0 / 3.0) / (2.0 * mass_ratio)


# ---------------------------------------------------------------------------
#  density-from-mu and number integrals
# ---------------------------------------------------------------------------

def n_B_of_r(mu_B, V_B, n_F, g_B, g_BF):
    if g_B <= 0.0:
        return np.zeros_like(V_B)
    x = (mu_B - V_B - g_BF * n_F) / g_B
    return np.where(x > 0.0, x, 0.0)


def n_F_of_r(mu_F, V_F, n_B, K_F, g_BF):
    x = (mu_F - V_F - g_BF * n_B) / K_F
    x_pos = np.maximum(x, 0.0)            # avoid (-)**1.5 warning
    return np.where(x > 0.0, x_pos ** 1.5, 0.0)


def particle_number(n_density, r):
    """4 pi int  r^2 n(r) dr."""
    return 4.0 * np.pi * np.trapezoid(n_density * r ** 2, r)


# ---------------------------------------------------------------------------
#  bisection for mu, given the partner density
# ---------------------------------------------------------------------------

def _find_mu(target_N, density_at_mu, mu_lo=-1.0e3, mu_hi=1.0e4):
    """Find mu so that integrate density(mu) = target_N. density is monotone in mu."""
    if target_N <= 0.0:
        return -1.0e10  # density essentially zero everywhere

    def res(mu):
        return density_at_mu(mu) - target_N

    f_lo, f_hi = res(mu_lo), res(mu_hi)
    # expand if accidentally on the wrong side
    while f_lo > 0.0 and mu_lo > -1.0e8:
        mu_lo *= 2.0
        f_lo = res(mu_lo)
    while f_hi < 0.0 and mu_hi < 1.0e8:
        mu_hi *= 2.0
        f_hi = res(mu_hi)
    if f_lo > 0.0 or f_hi < 0.0:
        # could not bracket -- return the best we have
        return mu_lo if f_lo > 0.0 else mu_hi

    return brentq(res, mu_lo, mu_hi, xtol=1.0e-9, rtol=1.0e-10)


# ---------------------------------------------------------------------------
#  main solver
# ---------------------------------------------------------------------------

def solve_bf_mixture(
    g_B: float,
    g_BF: float,
    mass_ratio: float = 1.0,
    freq_ratio: float = 1.0,
    N_B: float = 1.0e4,
    N_F: float = 1.0e3,
    r_max: float = 12.0,
    n_r: int = 401,
    max_iter: int = 400,
    tol: float = 1.0e-7,
    mixing: float = 0.4,
    verbose: bool = False,
):
    """Solve coupled TF+LDA self-consistency. Returns a result dict.

    Notes
    -----
    * `mixing` is the linear-mix coefficient (smaller -> more stable, slower).
    * If the iteration diverges/oscillates, `converged` will be False and the
      returned profiles correspond to the last accepted iterate.  This is
      typically a signal of the collapse instability (g_BF too negative) or
      phase separation (g_BF too positive).
    """
    r = np.linspace(0.0, r_max, n_r)
    V_B = 0.5 * r ** 2
    V_F = 0.5 * mass_ratio * freq_ratio ** 2 * r ** 2
    K_F = fermi_K(mass_ratio)

    # initial guess: independent profiles
    n_F = np.zeros_like(r)
    n_B = np.zeros_like(r)
    mu_B = mu_F = 0.0
    history = []
    converged = False
    last_change = np.inf

    for it in range(max_iter):
        # update bosons given current n_F
        def density_B(mu):
            return particle_number(n_B_of_r(mu, V_B, n_F, g_B, g_BF), r)
        mu_B_new = _find_mu(N_B, density_B)
        n_B_target = n_B_of_r(mu_B_new, V_B, n_F, g_B, g_BF)

        # update fermions given proposed n_B (we damp below)
        def density_F(mu):
            return particle_number(n_F_of_r(mu, V_F, n_B_target, K_F, g_BF), r)
        mu_F_new = _find_mu(N_F, density_F)
        n_F_target = n_F_of_r(mu_F_new, V_F, n_B_target, K_F, g_BF)

        # linear mixing
        n_B_new = mixing * n_B_target + (1.0 - mixing) * n_B
        n_F_new = mixing * n_F_target + (1.0 - mixing) * n_F

        # convergence diagnostic
        scale = max(np.max(n_B_new), np.max(n_F_new), 1.0e-30)
        change = (np.max(np.abs(n_B_new - n_B))
                  + np.max(np.abs(n_F_new - n_F))) / scale
        history.append(change)
        if verbose and it % 20 == 0:
            print(f"  it {it:3d}  mu_B={mu_B_new:.4f}  mu_F={mu_F_new:.4f}  "
                  f"rel.change={change:.2e}")
        n_B, n_F = n_B_new, n_F_new
        mu_B, mu_F = mu_B_new, mu_F_new
        last_change = change
        if change < tol and it > 5:
            converged = True
            break

    # diagnostics
    def edge_radius(n):
        if np.max(n) <= 0.0:
            return 0.0
        thresh = 1.0e-4 * np.max(n)
        idx = np.where(n > thresh)[0]
        return r[idx[-1]] if len(idx) else 0.0

    R_B = edge_radius(n_B)
    R_F = edge_radius(n_F)

    # stability at peak n_F
    n_F_peak = float(np.max(n_F))
    if n_F_peak > 0.0:
        # d E_F / d n  =  (2/3) K_F  n^{-1/3}
        dEF_dn_peak = (2.0 / 3.0) * K_F * n_F_peak ** (-1.0 / 3.0)
        det_M_peak = g_B * dEF_dn_peak - g_BF ** 2
    else:
        dEF_dn_peak = np.inf
        det_M_peak = g_B * np.inf - g_BF ** 2  # -> +inf if g_B>0

    return {
        "r": r, "n_B": n_B, "n_F": n_F,
        "V_B": V_B, "V_F": V_F,
        "mu_B": mu_B, "mu_F": mu_F,
        "g_B": g_B, "g_BF": g_BF,
        "mass_ratio": mass_ratio, "freq_ratio": freq_ratio,
        "N_B": N_B, "N_F": N_F,
        "R_B": R_B, "R_F": R_F,
        "K_F": K_F,
        "n_B_peak": float(np.max(n_B)),
        "n_F_peak": n_F_peak,
        "dEF_dn_peak": dEF_dn_peak,
        "det_M_peak": det_M_peak,
        "stable_peak": det_M_peak > 0.0,
        "converged": converged,
        "iterations": it + 1,
        "last_change": last_change,
        "history": history,
    }


# ---------------------------------------------------------------------------
#  pure-component analytical references (used for unit tests)
# ---------------------------------------------------------------------------

def pure_boson_TF(g_B, N_B):
    """Standard TF result: mu = (g_B N_B / 4.737)^{2/5}, R = sqrt(2 mu)."""
    coeff = 8.0 * np.pi * 2.0 ** 1.5 / 15.0   # = 4.737
    mu = (g_B * N_B / coeff) ** 0.4
    R = np.sqrt(2.0 * mu)
    n0 = mu / g_B
    return mu, R, n0


def pure_fermion_LDA(N_F, mass_ratio=1.0, freq_ratio=1.0):
    """Spinless ideal Fermi gas in 3D harmonic trap:  N = (mu / h omega_F)^3 / 6."""
    omega_F_eff = np.sqrt(freq_ratio ** 2 / mass_ratio) * 1.0  # the effective
    # omega in boson units is sqrt(m_F omega_F^2 / m_F) = omega_F.
    # but the formula for N depends on "h omega_F" in its own units;
    # in our boson units, h omega_F = freq_ratio.
    mu = freq_ratio * (6.0 * N_F) ** (1.0 / 3.0)
    K_F = fermi_K(mass_ratio)
    n0 = (mu / K_F) ** 1.5
    R = np.sqrt(2.0 * mu / (mass_ratio * freq_ratio ** 2))
    return mu, R, n0


# ---------------------------------------------------------------------------
#  self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pure boson TF reference (g_B = 0.0628, N_B = 10000):")
    mu_B_ref, R_B_ref, n0_B_ref = pure_boson_TF(0.0628, 1.0e4)
    print(f"  analytic   mu_B = {mu_B_ref:.4f}  R_TF = {R_B_ref:.4f}  n0 = {n0_B_ref:.3f}")
    out = solve_bf_mixture(g_B=0.0628, g_BF=0.0, N_B=1.0e4, N_F=0.0,
                           verbose=False)
    print(f"  numerical  mu_B = {out['mu_B']:.4f}  R   = {out['R_B']:.4f}  n0 = {out['n_B'][0]:.3f}")
    print(f"  converged in {out['iterations']} iters")

    print("\nPure fermion LDA reference (N_F = 1000, alpha=1, freq_ratio=1):")
    mu_F_ref, R_F_ref, n0_F_ref = pure_fermion_LDA(1000.0)
    print(f"  analytic   mu_F = {mu_F_ref:.4f}  R    = {R_F_ref:.4f}  n0 = {n0_F_ref:.3f}")
    out = solve_bf_mixture(g_B=0.0628, g_BF=0.0, N_B=0.0, N_F=1.0e3,
                           verbose=False)
    print(f"  numerical  mu_F = {out['mu_F']:.4f}  R    = {out['R_F']:.4f}  n0 = {out['n_F'][0]:.3f}")
    print(f"  converged in {out['iterations']} iters")

    print("\nMixed, g_BF = +0.05 (mild repulsion):")
    out = solve_bf_mixture(g_B=0.0628, g_BF=0.05, N_B=1.0e4, N_F=1.0e3,
                           verbose=False)
    print(f"  mu_B = {out['mu_B']:.4f}  mu_F = {out['mu_F']:.4f}")
    print(f"  R_B  = {out['R_B']:.3f}  R_F  = {out['R_F']:.3f}")
    print(f"  n_B(0) = {out['n_B'][0]:.3f}  n_F(0) = {out['n_F'][0]:.3f}")
    print(f"  det(M) at peak n_F = {out['det_M_peak']:+.4f}  ->  "
          f"{'STABLE' if out['stable_peak'] else 'UNSTABLE'}")
    print(f"  converged in {out['iterations']} iters  (last change = {out['last_change']:.1e})")

    print("\nMixed, g_BF = -0.05 (mild attraction):")
    out = solve_bf_mixture(g_B=0.0628, g_BF=-0.05, N_B=1.0e4, N_F=1.0e3,
                           verbose=False)
    print(f"  mu_B = {out['mu_B']:.4f}  mu_F = {out['mu_F']:.4f}")
    print(f"  R_B  = {out['R_B']:.3f}  R_F  = {out['R_F']:.3f}")
    print(f"  n_B(0) = {out['n_B'][0]:.3f}  n_F(0) = {out['n_F'][0]:.3f}")
    print(f"  det(M) at peak n_F = {out['det_M_peak']:+.4f}  ->  "
          f"{'STABLE' if out['stable_peak'] else 'UNSTABLE'}")
    print(f"  converged in {out['iterations']} iters")

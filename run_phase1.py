"""
run_phase1.py
=============

Driver for Phase 1 / Checkpoint 1.2 numerical scaling:

    1. 2x2 lattice  (N = 4)   -- validate against the analytical 4-ring
                                 spectrum:  E in {-2, -1 (x3), 0 (x7), +1 (x5)} J
    2. 4x4 lattice  (N = 16)
    3. 6x6 lattice  (N = 36)  -- only sectors with D <= 2 000 000 are kept
                                 (full S^z = 0 sector has 9.07e9 states).

For each size we plot
    * the lowest 8 eigenvalues  vs  H/J   in the range  0 <= H/J <= 10
    * the average magnetisation <Mz> = <S^z_total> / N   vs   H/J
"""

import os
import pickle
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from heisenberg_ed import (
    HeisenbergSquare,
    magnetization_curve,
    per_sector_ground_states,
)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"spectra_{name}.pkl")


def cached_spectra(name: str, model, k_per_sector: int = 6,
                   sector_max_D=None):
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    spectra = analyse(name, model, k_per_sector=k_per_sector,
                      sector_max_D=sector_max_D)
    with open(path, "wb") as f:
        pickle.dump(spectra, f)
    return spectra


# ---------------------------------------------------------------------------
#  analytical reference for the 2x2 lattice  (4-site Heisenberg ring)
# ---------------------------------------------------------------------------
#
# 2x2 PBC has 4 unique bonds forming a 4-cycle, and one can show that
#     H = J * (S_a . S_b),   with S_a = S_1 + S_3,   S_b = S_2 + S_4
# Each of S_a, S_b is the coupling of two spin-1/2 -> 0 or 1.
# Casework gives the spectrum
#     E = -2 J        (multiplicity 1,  total S = 0)
#     E = -  J        (multiplicity 3,  total S = 1)
#     E =   0         (multiplicity 7)
#     E = +  J        (multiplicity 5,  total S = 2)
ANALYTIC_2x2_LEVELS = [(-2.0, 1), (-1.0, 3), (0.0, 7), (1.0, 5)]


def analyse(name: str, model: HeisenbergSquare, k_per_sector: int = 6,
            sector_max_D=None):
    print("=" * 64)
    print(f"  {name}  (N = {model.N},  bonds = {len(model.bonds)})")
    print("=" * 64)
    t0 = time.time()
    spectra = model.all_sectors(k_per=k_per_sector,
                                sector_max_D=sector_max_D)
    print(f"  total time: {time.time() - t0:.2f} s\n")
    return spectra


def verify_2x2(spectra: dict):
    """Check 2x2 numerical spectrum against the analytic levels."""
    all_eigs = []
    for n_up, eigs in spectra.items():
        for e in eigs:
            all_eigs.append(round(float(e), 6))
    counts = Counter(all_eigs)
    print("  numerical spectrum (energy : multiplicity):")
    for E in sorted(counts):
        print(f"    E = {E:+.4f}   x {counts[E]}")
    print("  analytic spectrum:")
    for E, m in ANALYTIC_2x2_LEVELS:
        print(f"    E = {E:+.4f}   x {m}")
    # tolerance comparison
    ok = True
    for E, m in ANALYTIC_2x2_LEVELS:
        # tally numerical states close to E
        n_num = sum(c for e, c in counts.items() if abs(e - E) < 1e-6)
        if n_num != m:
            ok = False
            print(f"    [!] mismatch at E = {E}: got {n_num}, expected {m}")
    print("  ->", "PASSED" if ok else "FAILED")
    print()


# ---------------------------------------------------------------------------
#  plotting
# ---------------------------------------------------------------------------

def make_plots(model, spectra, name, h_array, save_dir,
               unreliable_below: float | None = None):
    """Plot lowest-8-level + magnetisation panels.

    Parameters
    ----------
    unreliable_below : float, optional
        If given, shade the H/J range below this value to indicate that the
        curve there is not trustworthy because some sectors were skipped
        (relevant for 6x6).
    """
    N = model.N

    levels, sz_levels, _, _ = per_sector_ground_states(spectra, h_array, N,
                                                       n_per_sector=1)
    Egs, Mz, Sz_gs = magnetization_curve(spectra, h_array, N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ----- left panel: lowest 8 levels vs H/J
    cmap = plt.get_cmap("viridis")
    for j in range(levels.shape[1]):
        axes[0].plot(h_array, levels[:, j] / model.J,
                     lw=1.4, color=cmap(j / max(1, levels.shape[1] - 1)))
    axes[0].plot(h_array, Egs / model.J, "k--", lw=1.0, alpha=0.5,
                 label="ground state")
    axes[0].set_xlabel(r"$H / J$")
    axes[0].set_ylabel(r"$E / J$")
    axes[0].set_title(f"Lowest 8 eigenvalues — {name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    # ----- right panel: magnetisation
    axes[1].plot(h_array, Mz, lw=2.0, color="C3")
    axes[1].set_xlabel(r"$H / J$")
    axes[1].set_ylabel(r"$\langle M^z \rangle  =  \langle S^z_\mathrm{tot} \rangle / N$")
    axes[1].set_title(f"Magnetisation curve — {name}")
    axes[1].set_ylim(-0.02, 0.55)
    axes[1].grid(True, alpha=0.3)
    for s in range(0, N // 2 + 1):
        axes[1].axhline(s / N, color="grey", ls=":", lw=0.5, alpha=0.4)

    if unreliable_below is not None:
        for ax in axes:
            ax.axvspan(0.0, unreliable_below, color="red", alpha=0.10,
                       zorder=-1)
            ax.text(unreliable_below / 2, ax.get_ylim()[1] * 0.92,
                    "missing\nsectors",
                    ha="center", va="top", fontsize=9,
                    color="darkred", alpha=0.8)

    fig.suptitle(f"{name} square Heisenberg AFM, PBC", y=1.02)
    fig.tight_layout()
    out = os.path.join(save_dir, f"heisenberg_{name}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
#  combined scaling figure
# ---------------------------------------------------------------------------

def combined_magnetisation_plot(results, h_array, save_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (name, model, spectra), color in zip(results, ["C0", "C1", "C2"]):
        _, Mz, _ = magnetization_curve(spectra, h_array, model.N)
        ax.plot(h_array, Mz, lw=2, label=f"{name}  (N = {model.N})",
                color=color)
    ax.set_xlabel(r"$H / J$")
    ax.set_ylabel(r"$\langle M^z \rangle$")
    ax.set_title("Magnetisation staircase — finite-size scaling")
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(save_dir, "magnetisation_scaling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    save_dir = os.path.dirname(os.path.abspath(__file__))
    h_array = np.linspace(0.0, 10.0, 401)

    # ---- 2x2 ----
    m2 = HeisenbergSquare(2, 2, J=1.0)
    s2 = cached_spectra("2x2", m2, k_per_sector=20)     # ALL 16 levels
    print("Validation against analytical 4-ring spectrum:")
    verify_2x2(s2)
    make_plots(m2, s2, "2x2", h_array, save_dir)

    # ---- 4x4 ----
    m4 = HeisenbergSquare(4, 4, J=1.0)
    s4 = cached_spectra("4x4", m4, k_per_sector=6)
    make_plots(m4, s4, "4x4", h_array, save_dir)

    # ---- 6x6 ----
    # Only the tractable sectors (D <= 2e6).  This covers the high-field
    # regime (large |S^z|).  The low-field part of the curve requires
    # additional symmetries (translation, point group) -- see QuSpin.
    m6 = HeisenbergSquare(6, 6, J=1.0)
    s6 = cached_spectra("6x6", m6, k_per_sector=4,
                        sector_max_D=2_000_000)
    make_plots(m6, s6, "6x6_partial", h_array, save_dir,
               unreliable_below=3.3)

    # ---- combined comparison ----
    combined_magnetisation_plot(
        [("2x2", m2, s2), ("4x4", m4, s4), ("6x6 (partial)", m6, s6)],
        h_array, save_dir,
    )
    print("\nDone.")

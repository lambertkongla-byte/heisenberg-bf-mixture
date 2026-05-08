"""
Exact Diagonalization of the S = 1/2 Heisenberg Model on a Square Lattice
=========================================================================

P452 Project 2 - Phase 1 Checkpoint 1.2

Hamiltonian
-----------
    H = J * sum_<i,j> S_i . S_j   -   h * sum_i S^z_i

with J > 0 (anti-ferromagnetic) on an Lx x Ly square lattice with periodic
boundary conditions. The interaction is expanded as

    S_i . S_j = S^z_i S^z_j + (1/2) (S+_i S-_j  +  S-_i S+_j)

Sign convention for the Zeeman term: -h * S^z_total (h > 0 favours UP spins,
so the ground-state magnetisation grows from 0 to +1/2 as h increases, which
matches the plot expected in the problem set).

Algorithmic ideas
-----------------
1.  [H, S^z_total] = 0  =>  block-diagonalise by n_up (number of up spins).
    Sector dimension = C(N, n_up).
2.  Inside a fixed-n_up sector the Zeeman term is the constant -h * S^z, so
    we diagonalise once at h = 0 and reconstruct E_n(h) = E_n(0) - h * S^z.
3.  The S^z S^z piece is diagonal; the S+S- terms flip a single ↑↓ pair, so
    the sparse Hamiltonian has at most ~2N non-zeros per row.

Memory note
-----------
    L=4,4  ->  n_up=8 sector  D = 12 870     (trivial)
    L=6,6  ->  n_up=18 sector D = 9.07e9     (infeasible without further
                                              symmetries: translation, lattice
                                              point group, spin inversion).
    For 6x6 we restrict to the high-|S^z| sectors that fit in memory and
    note that QuSpin + translation symmetry is needed to reach S^z = 0.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
#  bit helpers
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    """Number of set bits (= number of up spins) in the integer state."""
    return bin(x).count("1")


# ---------------------------------------------------------------------------
#  Heisenberg square lattice
# ---------------------------------------------------------------------------

class HeisenbergSquare:
    """S = 1/2 Heisenberg on Lx x Ly torus."""

    # ------------------------------------------------------------- construction
    def __init__(self, Lx: int, Ly: int, J: float = 1.0):
        self.Lx, self.Ly = Lx, Ly
        self.N = Lx * Ly
        self.J = J
        self.bonds = self._build_bonds()

    def _build_bonds(self):
        """Unique nearest-neighbour bonds on a Lx x Ly torus.

        Site index s = x*Ly + y for 0 <= x < Lx, 0 <= y < Ly.
        We use a `set` so that the L=2 PBC self-identification (which would
        otherwise double-count the wrap-around bond) is automatically resolved
        and each pair appears exactly once.
        """
        bset = set()
        for x in range(self.Lx):
            for y in range(self.Ly):
                s = x * self.Ly + y
                r = ((x + 1) % self.Lx) * self.Ly + y           # +x neighbour
                u = x * self.Ly + (y + 1) % self.Ly             # +y neighbour
                if s != r:
                    bset.add((min(s, r), max(s, r)))
                if s != u:
                    bset.add((min(s, u), max(s, u)))
        return sorted(bset)

    # ------------------------------------------------------------- basis & H
    def basis(self, n_up: int):
        """Sorted list of basis-state integers with `n_up` up-spins.

        Returns (states, idx) where idx[state] -> position in `states`.
        """
        states = []
        for combo in combinations(range(self.N), n_up):
            v = 0
            for c in combo:
                v |= 1 << c
            states.append(v)
        states.sort()
        idx = {v: i for i, v in enumerate(states)}
        return states, idx

    def H0_block(self, n_up: int):
        """Sparse h=0 Hamiltonian in the sector with `n_up` up spins."""
        states, idx = self.basis(n_up)
        D = len(states)
        if D == 0:
            return None

        rows, cols, data = [], [], []
        J = self.J
        bonds = self.bonds

        for k, s in enumerate(states):
            diag = 0.0
            for (i, j) in bonds:
                bi = (s >> i) & 1
                bj = (s >> j) & 1
                if bi == bj:
                    # parallel spins  ->  S^z S^z = +1/4
                    diag += 0.25 * J
                else:
                    # anti-parallel  ->  S^z S^z = -1/4   AND a flip-flop term
                    diag -= 0.25 * J
                    sf = s ^ ((1 << i) | (1 << j))   # flip both spins
                    rows.append(k)
                    cols.append(idx[sf])
                    data.append(0.5 * J)
            rows.append(k)
            cols.append(k)
            data.append(diag)

        return csr_matrix((data, (rows, cols)), shape=(D, D), dtype=np.float64)

    # ------------------------------------------------------------- eigenvalues
    def lowest_eigs(self, n_up: int, k: int = 8, dense_thresh: int = 400):
        """Lowest `k` eigenvalues at h = 0 in the `n_up` sector.

        Uses dense eigvalsh for tiny blocks and sparse Lanczos otherwise.
        """
        H = self.H0_block(n_up)
        if H is None:
            return np.array([])
        D = H.shape[0]
        if D <= max(k + 5, dense_thresh):
            return np.sort(np.linalg.eigvalsh(H.toarray()))[:k]
        kk = min(k, D - 2)
        eigs = eigsh(H, k=kk, which="SA", return_eigenvectors=False)
        return np.sort(eigs)

    def all_sectors(self, k_per: int = 6, sector_max_D: int | None = None,
                    verbose: bool = True):
        """Lowest eigenvalues in every n_up sector.

        Sectors with dimension > sector_max_D are skipped.
        Returns dict { n_up -> sorted eigenvalues at h = 0 }.
        """
        out = {}
        for n_up in range(self.N + 1):
            D = comb(self.N, n_up)
            if sector_max_D is not None and D > sector_max_D:
                if verbose:
                    print(f"  n_up={n_up:2d}  Sz={n_up - self.N/2:+5.1f}  "
                          f"D={D:>15,d}   SKIPPED")
                continue
            if verbose:
                print(f"  n_up={n_up:2d}  Sz={n_up - self.N/2:+5.1f}  "
                      f"D={D:>15,d}", end="", flush=True)
            eigs = self.lowest_eigs(n_up, k=k_per)
            out[n_up] = eigs
            if verbose:
                print(f"   E0 = {eigs[0]: .6f}")
        return out


# ---------------------------------------------------------------------------
#  field-dependent observables  (algebraic, no extra diagonalisation)
# ---------------------------------------------------------------------------

def magnetization_curve(spectra: dict, h_array: np.ndarray, N: int):
    """Ground-state energy, <Mz>, and S^z_GS as a function of h.

    Convention: H_Zeeman = -h * S^z_total  (h > 0 favours up-spins).
    """
    Egs = np.empty_like(h_array, dtype=float)
    Mz = np.empty_like(h_array, dtype=float)
    Sz_gs = np.empty_like(h_array, dtype=float)
    for k, h in enumerate(h_array):
        best_E, best_Sz = np.inf, 0.0
        for n_up, eigs in spectra.items():
            Sz = n_up - N / 2.0
            E = eigs[0] - h * Sz
            if E < best_E:
                best_E, best_Sz = E, Sz
        Egs[k] = best_E
        Sz_gs[k] = best_Sz
        Mz[k] = best_Sz / N
    return Egs, Mz, Sz_gs


def eigenstate_trajectories(spectra: dict, h_array: np.ndarray, N: int,
                            n_states: int | None = None):
    """Energies of *fixed* eigenstates as functions of h.

    Each eigenstate of H is also an eigenstate of S^z_total (because they
    commute), so its energy at field h is exactly

        E_alpha(h)  =  E_alpha(h=0)  -  h * S^z_alpha .

    We collect every (E0, Sz) pair from all the cached sectors, sort by E0
    (ascending), keep the lowest `n_states`, and return their full
    trajectories.

    Returns
    -------
    energies : ndarray (len(h_array), n_states_kept)
        Eigenstate energies vs h.
    sz_vals : ndarray (n_states_kept,)
        S^z_total label of each tracked eigenstate.
    e0_vals : ndarray (n_states_kept,)
        Energy of each tracked eigenstate at h = 0.
    """
    states = []  # list of (E0, Sz)
    for n_up, eigs in spectra.items():
        Sz = n_up - N / 2.0
        for e in eigs:
            states.append((float(e), float(Sz)))
    states.sort(key=lambda t: t[0])  # by E at h=0
    if n_states is not None:
        states = states[:n_states]
    e0 = np.array([s[0] for s in states])
    sz = np.array([s[1] for s in states])
    energies = e0[None, :] - h_array[:, None] * sz[None, :]
    return energies, sz, e0


def per_sector_ground_states(spectra: dict, h_array: np.ndarray, N: int,
                             n_per_sector: int = 1):
    """Trajectories of the lowest `n_per_sector` states *in each S^z sector*.

    This is the right plot for visualising the magnetic level structure of
    a Heisenberg model: one line per (sector, level), with the ground state
    of the full system being exactly the lower envelope of these lines.

    Returns
    -------
    energies : ndarray (len(h_array), n_lines)
    sz_vals  : ndarray (n_lines,)
    e0_vals  : ndarray (n_lines,)
    labels   : list of strings  ("Sz = +k" etc.) for legend
    """
    states = []
    for n_up in sorted(spectra.keys()):
        Sz = n_up - N / 2.0
        for e in sorted(spectra[n_up])[:n_per_sector]:
            states.append((float(e), float(Sz)))
    e0 = np.array([s[0] for s in states])
    sz = np.array([s[1] for s in states])
    labels = [f"Sz={int(s):+d}" if float(int(s)) == s else f"Sz={s:+.1f}"
              for s in sz]
    energies = e0[None, :] - h_array[:, None] * sz[None, :]
    return energies, sz, e0, labels


def reliability_threshold(spectra: dict, N: int):
    """Smallest |Sz| sector that we have data for, expressed as the inner
    boundary of the Sz-set we possess.

    For full-spectrum lattices (4x4, 2x2) this is 0 -- everything is
    reliable.  For 6x6 with skipped central sectors, returns the smallest
    Sz value present in `spectra`.  The data is reliable for fields H at
    which the GS Sz strictly exceeds this boundary; below that field, a
    missing inner-sector state would have lower energy and our displayed
    "GS" is wrong.
    """
    sz_vals = sorted(n_up - N / 2.0 for n_up in spectra.keys())
    # find largest gap from 0; the unreliable region is |Sz| <= s_inner
    s_inner_pos = min((s for s in sz_vals if s >= 0), default=0.0)
    s_inner_neg = max((s for s in sz_vals if s <= 0), default=0.0)
    return float(max(abs(s_inner_pos), abs(s_inner_neg)))


def trust_field_h(spectra: dict, h_array: np.ndarray, N: int):
    """Smallest H in `h_array` at which the data-implied ground state has
    |Sz| > inner-boundary, i.e. the field above which the magnetisation
    curve from this (possibly partial) data is trustworthy.

    Returns inf if no such H exists in the array (i.e. data never escapes
    the partial-data regime), and 0.0 if the data is fully reliable.
    """
    s_inner = reliability_threshold(spectra, N)
    if s_inner == 0.0:
        return 0.0
    _, _, Sz_gs = magnetization_curve(spectra, h_array, N)
    above = np.where(np.abs(Sz_gs) > s_inner + 0.5)[0]  # strict, with tolerance
    if len(above) == 0:
        return float("inf")
    return float(h_array[above[0]])

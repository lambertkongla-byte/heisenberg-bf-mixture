"""
P452 Project 2  -  Streamlit UI
================================

Single-app interface for both checkpoints:

    Phase 1 : Heisenberg AFM exact diagonalisation on 2x2, 4x4, 6x6 squares.
    Phase 2 : Self-consistent Thomas-Fermi + LDA Bose-Fermi mixture in a trap.

Run locally with
    streamlit run app.py

or deploy on https://streamlit.io/cloud  with this directory as the repo.
"""

from __future__ import annotations

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ensure local imports work regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from heisenberg_ed import (                                # noqa: E402
    HeisenbergSquare,
    magnetization_curve,
    per_sector_ground_states,
    reliability_threshold,
    trust_field_h,
)
from bf_mixture import (                                   # noqa: E402
    solve_bf_mixture,
    pure_boson_TF,
    pure_fermion_LDA,
    fermi_K,
    g_B_from_a,
    g_BF_from_a,
)

CACHE_DIR = os.path.join(HERE, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def safe_tight_layout(fig):
    """Use tight_layout when possible, but do not let Matplotlib crash Streamlit."""
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.10, right=0.95, bottom=0.14, top=0.90,
                            wspace=0.30)


# =============================================================================
#  Phase-1 helpers  (cached)
# =============================================================================

@st.cache_data(show_spinner=False)
def get_phase1_spectra(Lx: int, Ly: int, k_per: int, sector_max_D: int | None):
    """Compute (or load from disk) the per-sector h=0 eigenvalues."""
    tag = f"{Lx}x{Ly}_k{k_per}_D{sector_max_D}"
    path = os.path.join(CACHE_DIR, f"spectra_{tag}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["N"], data["nbonds"], data["spectra"]

    model = HeisenbergSquare(Lx, Ly, J=1.0)
    spectra = model.all_sectors(
        k_per=k_per, sector_max_D=sector_max_D, verbose=False
    )
    with open(path, "wb") as f:
        pickle.dump({"N": model.N, "nbonds": len(model.bonds),
                     "spectra": spectra}, f)
    return model.N, len(model.bonds), spectra


def saturation_field_estimate(Lx: int, Ly: int) -> float:
    """H_sat / J = 2 z S in spin-wave theory.  z = coordination number."""
    if Lx == 2 and Ly == 2:
        return 2.0          # z = 2 on the 4-cycle
    return 4.0              # z = 4 on a true square lattice (4x4, 6x6)


# =============================================================================
#  Phase-2 helpers  (cached)
# =============================================================================

@st.cache_data(show_spinner=False)
def get_bf_solution(g_B, g_BF, mass_ratio, freq_ratio,
                    N_B, N_F, r_max, n_r, mixing):
    return solve_bf_mixture(
        g_B=g_B, g_BF=g_BF,
        mass_ratio=mass_ratio, freq_ratio=freq_ratio,
        N_B=N_B, N_F=N_F,
        r_max=r_max, n_r=n_r, mixing=mixing,
        max_iter=500,
    )


# =============================================================================
#  Streamlit page setup
# =============================================================================

st.set_page_config(
    page_title="P452 Project 2 - Many-Body Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("P452 Project 2")
page = st.sidebar.radio(
    "Section",
    ["Phase 1 - Heisenberg ED",
     "Phase 2 - Bose-Fermi Mixture",
     "About"],
)
st.sidebar.markdown("---")


# =============================================================================
#  PHASE 1
# =============================================================================

if page == "Phase 1 - Heisenberg ED":
    st.title("Phase 1 - Heisenberg AFM on a Square Lattice")
    st.markdown(
        "Spin-1/2 antiferromagnetic Heisenberg model on an "
        r"$L_x \times L_y$ square lattice with periodic boundary "
        "conditions:"
    )
    st.latex(
        r"\hat H \;=\; J \sum_{\langle i,j\rangle}"
        r"\hat{\mathbf S}_i \cdot \hat{\mathbf S}_j"
        r"\;-\; h \sum_i \hat S^z_i"
    )
    st.markdown(
        r"Sign convention: $-h\,\hat S^z_{\rm tot}$, so that $h>0$ favours "
        r"up-spins (the ground-state magnetisation grows from $0$ to "
        r"$+\tfrac12$ as $h$ increases). Block-diagonalised by "
        r"$[\hat H,\hat S^z_{\rm tot}]=0$."
    )

    # ------ controls ------
    st.sidebar.markdown("### Phase 1 parameters")
    size_label = st.sidebar.selectbox(
        "Lattice size",
        ["2 x 2  (N = 4, full ED)",
         "4 x 4  (N = 16, full ED)",
         "6 x 6  (N = 36, partial)"],
    )
    h_max = st.sidebar.slider("Maximum $H/J$", 1.0, 20.0, 10.0, 0.5)

    if size_label.startswith("2"):
        Lx, Ly, k_per, sector_max_D = 2, 2, 20, None
    elif size_label.startswith("4"):
        Lx, Ly, k_per, sector_max_D = 4, 4, 6, None
    else:
        Lx, Ly, k_per, sector_max_D = 6, 6, 4, 2_000_000

    # ------ compute ------
    if Lx == 6:
        st.warning(
            "6x6 partial ED can be slow or memory-heavy on Streamlit Cloud. "
            "If the app resets, use 2x2 or 4x4 online and run 6x6 locally."
        )

    with st.spinner(f"Diagonalising {size_label}..."):
        N, n_bonds, spectra = get_phase1_spectra(
            Lx, Ly, k_per, sector_max_D
        )

    # Format Hilbert dim: comma-separated for small N, scientific for large
    hd = 2 ** N
    if hd < 10 ** 7:
        hd_str = f"{hd:,}"
    else:
        # 3 sig figs in 6.87e10 form, then convert "e10" -> "x10^10"
        sci = f"{hd:.2e}"  # e.g. '6.87e+10'
        mant, exp = sci.split("e")
        hd_str = f"{mant} x 10^{int(exp)}"

    summary = st.columns(4)
    summary[0].metric("Sites N", N)
    summary[1].metric("Bonds", n_bonds)
    summary[2].metric("Sectors solved", len(spectra))
    summary[3].metric(f"Hilbert dim (2^{N})", hd_str)

    # ------ field-dependent observables ------
    h_array = np.linspace(0.0, h_max, 401)

    # Plot strategy:
    #   - small system (N <= 4): plot ALL eigenstates of H (all 16 for 2x2)
    #   - larger systems: plot the ground state of each S^z sector
    # In either case each line is a fixed eigenstate, with energy
    # E_alpha(h) = E_alpha(0) - h * Sz_alpha   (since [H, S^z_total] = 0).
    if N <= 4:
        # all stored eigenstates from each sector
        n_per_sector = max(len(v) for v in spectra.values())
        plot_label_kind = "All eigenstates"
    else:
        n_per_sector = 1
        plot_label_kind = "Ground state of each Sz sector"

    levels, sz_levels, e0_levels, _ = per_sector_ground_states(
        spectra, h_array, N, n_per_sector=n_per_sector
    )
    n_lines_drawn = levels.shape[1]
    Egs, Mz, Sz_gs = magnetization_curve(spectra, h_array, N)

    # how trustworthy is the data at low H?
    h_trust = trust_field_h(spectra, h_array, N)
    s_inner = reliability_threshold(spectra, N)
    has_skipped = (h_trust > 0.0 and np.isfinite(h_trust))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ----- left panel: eigenenergies vs H/J ---------------------------
    sz_max_abs = max(abs(sz_levels.min()), abs(sz_levels.max()))
    norm = plt.Normalize(vmin=-sz_max_abs, vmax=+sz_max_abs)
    cmap = plt.get_cmap("coolwarm")
    for j in range(n_lines_drawn):
        axes[0].plot(h_array, levels[:, j], lw=1.2,
                     color=cmap(norm(sz_levels[j])), alpha=0.8)
    axes[0].plot(h_array, Egs, "k--", lw=1.8, alpha=0.95,
                 label="ground state (envelope)")
    axes[0].set_xlabel("H / J")
    axes[0].set_ylabel("E / J")
    if N <= 4:
        title = f"All {n_lines_drawn} eigenstates"
    else:
        title = (f"Ground state of each Sz sector  "
                 f"({n_lines_drawn} of {N + 1} sectors)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.02, shrink=0.85)
    cbar.set_label("Sz_tot")

    # ----- right panel: magnetization curve ---------------------------
    if has_skipped:
        # Split into reliable / unreliable parts
        # H >= h_trust : data is the true GS magnetisation (solid red)
        # H <  h_trust : data is wrong (lowest available sector is plotted dashed grey)
        mask_rel = h_array >= h_trust
        mask_unr = ~mask_rel
        axes[1].plot(h_array[mask_unr], Mz[mask_unr],
                     ls="--", color="grey", lw=1.5,
                     label="data (NOT true GS — sectors skipped)")
        axes[1].plot(h_array[mask_rel], Mz[mask_rel],
                     color="C3", lw=2, label="data (true GS)")
        # mark the true-GS expected point at H=0
        axes[1].plot([0], [0], marker="*", color="black", markersize=14,
                     zorder=5,
                     label="true M(H=0) = 0 (Sz=0 not computed)")
        axes[1].annotate("", xy=(0.05, 0.005), xytext=(0.5, 0.18),
                         arrowprops=dict(arrowstyle="->",
                                         color="black", alpha=0.6, lw=1.2))
        axes[1].legend(loc="lower right", fontsize=8)
    else:
        axes[1].plot(h_array, Mz, lw=2, color="C3")
    axes[1].set_xlabel("H / J")
    axes[1].set_ylabel("<Mz>")
    axes[1].set_title("Magnetisation curve")
    axes[1].set_ylim(-0.02, 0.55)
    axes[1].grid(alpha=0.3)
    for s in range(0, N // 2 + 1):
        axes[1].axhline(s / N, color="grey", ls=":", lw=0.5, alpha=0.4)

    # mark the unreliable region precisely on both panels (light blue)
    if has_skipped:
        for ax in axes:
            ax.axvspan(0.0, h_trust, color="#cfe2ff", alpha=0.5, zorder=-1)
            ax.axvline(h_trust, color="#4a8de0", lw=1, alpha=0.6, ls=":")

    safe_tight_layout(fig)
    st.pyplot(fig)

    # ----- explanation text ------------------------------------------
    if has_skipped:
        st.info(
            f"The $S^z_{{\\rm tot}} = 0$ sector has too many degenerate "
            f"basis states to diagonalise directly, so we only fit the "
            f"extreme cases $|S^z_{{\\rm tot}}| \\geq {int(s_inner)}$. "
            f"Therefore the magnetisation at $H=0$ should be exactly "
            f"$\\langle M^z\\rangle = 0$ (because $S^z_{{\\rm tot}} = 0$ "
            f"in the true ground state). The dashed grey portion below "
            f"$H/J = {h_trust:.2f}$ shows the artificial value $1/3$ that "
            f"comes from the lowest sector we *did* compute "
            f"($S^z = \\pm{int(s_inner)}$); the black star marks where "
            f"the true point at $H=0$ lies."
        )
    else:
        st.caption(
            "Each coloured curve is a fixed eigenstate (same $S^z$ for all "
            r"$H$, slope $-S^z$). The black dashed line is the ground-state "
            "envelope, tracing successive level crossings as the Zeeman "
            "term grows."
        )

    obs = st.columns(4)
    if has_skipped:
        # report what we actually computed (lowest among kept sectors), but label honestly
        obs[0].metric(f"$E_0$ in $|S^z| \\geq {int(s_inner)}$",
                      f"{e0_levels.min():.4f} J")
        obs[1].metric("True $E_0$ at $H=0$",
                      "not computed")
    else:
        obs[0].metric("$E_0$ at $H=0$", f"{Egs[0]:.4f} J")
        obs[1].metric("$E_0/N$ at $H=0$", f"{Egs[0]/N:.4f} J")
    h_sat_th = saturation_field_estimate(Lx, Ly)
    obs[2].metric(r"$H_{\rm sat}/J$ (spin wave)", f"{h_sat_th:.1f}")
    obs[3].metric(r"$H_{\rm sat}/J$ (numerical)",
                  f"{h_array[np.where(Mz >= 0.4999)[0][0]]:.2f}"
                  if np.any(Mz >= 0.4999) else "not reached")

    # ------ plateau analysis -----------------------------------------
    st.markdown("### Magnetisation plateaus")

    # Sz_gs steps through integer multiples of 1 between 0 and N/2 as H grows.
    # Each plateau is a contiguous H interval with the same Sz_gs value.
    plateaus = []
    if len(Sz_gs):
        start_idx = 0
        for i in range(1, len(Sz_gs)):
            if Sz_gs[i] != Sz_gs[start_idx]:
                plateaus.append({
                    "Sz": float(Sz_gs[start_idx]),
                    "Mz": float(Sz_gs[start_idx]) / N,
                    "H_lo": float(h_array[start_idx]),
                    "H_hi": float(h_array[i - 1]),
                })
                start_idx = i
        plateaus.append({
            "Sz": float(Sz_gs[start_idx]),
            "Mz": float(Sz_gs[start_idx]) / N,
            "H_lo": float(h_array[start_idx]),
            "H_hi": float(h_array[-1]),
        })

    if plateaus:
        import pandas as pd
        df = pd.DataFrame({
            "Sz_tot":    [int(p['Sz']) for p in plateaus],
            "<M^z>":     [round(p['Mz'], 4) for p in plateaus],
            "H/J start": [round(p['H_lo'], 3) for p in plateaus],
            "H/J end":   [round(p['H_hi'], 3) for p in plateaus],
            "width":     [round(p['H_hi'] - p['H_lo'], 3) for p in plateaus],
        })
        st.dataframe(df, hide_index=True)
        st.caption(
            f"As H increases, $S^z_{{\\rm tot}}$ steps through integer "
            f"values $0, 1, 2, \\dots, N/2 = {N//2}$. Each step gives a "
            f"plateau of height $1/N = {1.0/N:.4f}$ in $\\langle M^z\\rangle$."
        )
        if Lx == 6:
            st.caption(
                "**6×6 caveat:** because the central $S^z$ sectors were skipped, "
                "the small-$|S^z|$ plateaus are merged into a single artificial "
                "plateau. The high-field plateaus and the saturation field "
                "$H_{\\rm sat}/J = 4$ are reliable."
            )

    # ------ thermodynamic-limit discussion ---------------------------
    with st.expander(r"Thermodynamic limit: how does $\langle M^z\rangle(H)$ behave as $N \to \infty$?",
                     expanded=False):
        st.markdown(
            r"""
**Plateau heights and widths.** On a finite lattice each plateau has
height $1/N$ (one extra flipped spin) and a width that is set by the
energy gap between adjacent $S^z$ sectors,

$$
\Delta H_S \;=\; E_0(S+1) - E_0(S)\, .
$$

For $0 \le S \le N/2$ on the antiferromagnet, this gap *decreases* with
$N$ for the small-$S$ sectors (gapless spin-wave excitations on the
square lattice in the AFM phase) and approaches a finite spin-wave
saturation gap near $S = N/2$.

**As $N\to\infty$.**

* Plateau heights shrink as $1/N \to 0$,
* and plateau widths shrink in the AFM region,
* so the staircase smooths into a *continuous* curve
  $\langle M^z\rangle(H)$ that grows monotonically from $0$ at $H=0$ to
  the saturation value $1/2$ at a single critical field
  $H_{\rm sat}$.

**The saturation field.** The transition from the AFM phase to the
fully polarised FM phase is a true quantum phase transition that the
finite-size data already pin down sharply. Linear spin-wave theory (or
the exact one-magnon energy on the FM background) gives, on the square
lattice with coordination number $z = 4$,

$$
\bigl(H/J\bigr)_c \;=\; H_{\rm sat}/J \;=\; 2\,z\,S \;=\; 4\, ,
$$

independent of $N$. The 4×4 numerics already give $H_{\rm sat}/J
\approx 4$ to within the field-grid resolution; the 6×6 high-field
sector confirms the same value. In the thermodynamic limit
$\langle M^z\rangle$ approaches $1/2$ continuously as $H \to 4J$ from
below and is identically $1/2$ for all $H \ge 4J$.

**Where the staircase comes from physically.** Each step is a level
crossing in the ground state from one $S^z$ sector to the next. On a
finite lattice these crossings are discrete and visible; in the
thermodynamic limit they accumulate into a continuous magnetisation
density.
            """
        )

    # ------ finite-size overlay --------------------------------------
    with st.expander("Finite-size overlay: 2x2 vs 4x4 vs 6x6", expanded=False):
        st.caption(
            "This comparison is not computed automatically on Streamlit Cloud. "
            "Code inside an expander still executes during page load, so an "
            "automatic 6x6 calculation can make the app fail its health check."
        )
        include_6x6 = st.checkbox(
            "Include 6x6 curve (slow; use only after 2x2 and 4x4 work)",
            value=False,
        )
        run_overlay = st.button("Compute finite-size overlay")

        if run_overlay:
            with st.spinner("Computing finite-size overlay..."):
                _, _, sp2 = get_phase1_spectra(2, 2, 20, None)
                _, _, sp4 = get_phase1_spectra(4, 4, 6, None)
                sp6 = None
                if include_6x6:
                    _, _, sp6 = get_phase1_spectra(6, 6, 4, 2_000_000)

                h_overlay = np.linspace(0.0, 6.0, 601)
                _, Mz2, _ = magnetization_curve(sp2, h_overlay, 4)
                _, Mz4, _ = magnetization_curve(sp4, h_overlay, 16)

            fig2, ax2 = plt.subplots(figsize=(8, 4.6))
            ax2.plot(h_overlay, Mz2, lw=2, label="2x2  (N=4)")
            ax2.plot(h_overlay, Mz4, lw=2, label="4x4  (N=16)")

            if sp6 is not None:
                _, Mz6, _ = magnetization_curve(sp6, h_overlay, 36)
                h_trust6 = trust_field_h(sp6, h_overlay, 36)
                if np.isfinite(h_trust6) and h_trust6 > 0.0:
                    mask6 = h_overlay >= h_trust6
                    ax2.plot(h_overlay[mask6], Mz6[mask6], lw=2,
                             label=f"6x6  (reliable for H/J >= {h_trust6:.2f})")
                else:
                    ax2.plot(h_overlay, Mz6, lw=2, label="6x6  (N=36)")

            ax2.axvline(4.0, color="k", ls="--", lw=1, alpha=0.5,
                        label="H_sat / J = 4")
            ax2.set_xlabel("H / J")
            ax2.set_ylabel("<Mz>")
            ax2.set_ylim(-0.02, 0.55)
            ax2.set_title("Finite-size scaling of the magnetisation curve")
            ax2.grid(alpha=0.3)
            ax2.legend(loc="lower right")
            safe_tight_layout(fig2)
            st.pyplot(fig2)
        else:
            st.info("Click the button above to generate the overlay. Start without 6x6 on Streamlit Cloud.")


# =============================================================================
#  PHASE 2
# =============================================================================

elif page == "Phase 2 - Bose-Fermi Mixture":
    st.title("Phase 2 - Trapped Bose-Fermi Mixture")
    st.markdown(
        "Self-consistent Thomas-Fermi (bosons) + Local-Density-Approximation "
        "(fermions) for two species in spherical harmonic traps:"
    )
    st.latex(
        r"\mu_B = V_B(r) + g_B\,n_B(r) + g_{BF}\,n_F(r)"
        r"\qquad\;"
        r"\mu_F = V_F(r) + E_F[n_F(r)] + g_{BF}\,n_B(r)"
    )
    st.latex(
        r"E_F[n] \;=\; \frac{\hbar^2}{2 m_F}\,(6\pi^2 n)^{2/3}"
    )
    st.markdown(
        r"Units: boson harmonic-oscillator units "
        r"($\hbar = m_B = \omega_B = 1$).  Coupling constants are computed "
        r"from scattering lengths via $g_B = 4\pi\,\tilde a_B$ and "
        r"$g_{BF} = 2\pi\,\tilde a_{BF}\,(1+\alpha)/\alpha$ with "
        r"$\alpha = m_F / m_B$ and $\tilde a = a / a_{\rm ho}^{(B)}$."
    )

    # ------ controls ------
    st.sidebar.markdown("### Phase 2 parameters")
    input_mode = st.sidebar.radio(
        "Coupling input style",
        ["scattering lengths $\\tilde a / a_{\\rm ho}$",
         "coupling constants $g$ directly"],
        index=0,
    )

    mass_ratio = st.sidebar.slider(r"$m_F / m_B$", 0.1, 5.0, 1.0, 0.05)
    freq_ratio = st.sidebar.slider(r"$\omega_F / \omega_B$", 0.1, 5.0, 1.0, 0.05)
    N_B = st.sidebar.number_input("$N_B$", 0, 1_000_000, 10000, 1000)
    N_F = st.sidebar.number_input("$N_F$", 0, 1_000_000, 1000, 100)

    if input_mode.startswith("scattering"):
        a_B = st.sidebar.slider(r"$\tilde a_B$", 0.0001, 0.05, 0.005,
                                0.0005, format="%.4f")
        a_BF = st.sidebar.slider(r"$\tilde a_{BF}$", -0.05, 0.05, 0.0,
                                 0.0005, format="%.4f")
        g_B = g_B_from_a(a_B)
        g_BF = g_BF_from_a(a_BF, mass_ratio)
        st.sidebar.caption(f"-> $g_B = {g_B:.4f}$,  $g_{{BF}} = {g_BF:+.4f}$")
    else:
        g_B = st.sidebar.slider(r"$g_B$", 0.001, 1.0, 0.0628, 0.001,
                                format="%.4f")
        g_BF = st.sidebar.slider(r"$g_{BF}$", -0.5, 0.5, 0.0, 0.005,
                                 format="%.4f")

    with st.sidebar.expander("Numerics"):
        r_max = st.slider("Grid $r_{\\max}$", 5.0, 30.0, 12.0, 1.0)
        n_r = st.slider("Grid points", 100, 800, 400, 100)
        mixing = st.slider("Linear-mix coefficient", 0.05, 0.9, 0.4, 0.05)

    # ------ solve ------
    with st.spinner("Solving self-consistent equations..."):
        out = get_bf_solution(g_B, g_BF, mass_ratio, freq_ratio,
                              float(N_B), float(N_F),
                              r_max, n_r, mixing)

    if out["converged"]:
        st.success(f"Converged in {out['iterations']} iterations  "
                   f"(rel. change = {out['last_change']:.1e})")
    else:
        st.warning(
            f"Did **not** converge in {out['iterations']} iterations.  "
            "This typically signals collapse ($g_{BF}$ too negative) or "
            "phase separation ($g_{BF}$ too positive)."
        )

    # ------ density profile ------
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(out["r"], out["n_B"], "b-", lw=2,
            label=f"Bosons, N_B = {int(N_B)}")
    ax.plot(out["r"], out["n_F"], "r-", lw=2,
            label=f"Fermions, N_F = {int(N_F)}")
    if g_BF == 0.0:
        regime = "non-interacting"
    elif g_BF > 0.0:
        regime = "repulsive  (BEC pushes fermions outward)"
    else:
        regime = "attractive  (BEC pulls fermions inward)"
    ax.set_title(f"Density profiles - g_BF={g_BF:+.4f} ({regime})")
    ax.set_xlabel("r / a_ho(B)")
    ax.set_ylabel("density [a_ho(B)]^-3")
    ax.grid(alpha=0.3)
    ax.legend()
    safe_tight_layout(fig)
    st.pyplot(fig)

    # ------ observables ------
    cols = st.columns(4)
    cols[0].metric(r"$\mu_B$", f"{out['mu_B']:.3f}")
    cols[1].metric(r"$\mu_F$", f"{out['mu_F']:.3f}")
    cols[2].metric(r"$R_B$ (TF)", f"{out['R_B']:.3f}")
    cols[3].metric(r"$R_F$", f"{out['R_F']:.3f}")
    cols2 = st.columns(4)
    cols2[0].metric(r"$n_B(0)$", f"{out['n_B'][0]:.2f}")
    cols2[1].metric(r"$n_F(0)$", f"{out['n_F'][0]:.2f}")
    cols2[2].metric(
        "stability det(M) at peak $n_F$",
        f"{out['det_M_peak']:+.4f}",
        delta="stable" if out["stable_peak"] else "UNSTABLE",
        delta_color=("normal" if out["stable_peak"] else "inverse"),
    )

    # baseline (non-interacting) reference
    mu_B_ref, R_B_ref, n0_B_ref = pure_boson_TF(g_B, N_B) if N_B else (0, 0, 0)
    mu_F_ref, R_F_ref, n0_F_ref = (
        pure_fermion_LDA(N_F, mass_ratio, freq_ratio) if N_F else (0, 0, 0)
    )
    with st.expander("Compare with non-interacting baseline"):
        ref_table = [
            ["mu_B", f"{mu_B_ref:.3f}", f"{out['mu_B']:.3f}"],
            ["mu_F", f"{mu_F_ref:.3f}", f"{out['mu_F']:.3f}"],
            ["R_B",  f"{R_B_ref:.3f}",  f"{out['R_B']:.3f}"],
            ["R_F",  f"{R_F_ref:.3f}",  f"{out['R_F']:.3f}"],
            ["n_B(0)", f"{n0_B_ref:.2f}", f"{out['n_B'][0]:.2f}"],
            ["n_F(0)", f"{n0_F_ref:.2f}", f"{out['n_F'][0]:.2f}"],
        ]
        st.table({"quantity": [r[0] for r in ref_table],
                  "g_BF = 0 (independent)": [r[1] for r in ref_table],
                  "current g_BF": [r[2] for r in ref_table]})

    # ------ three-regime comparison ------
    st.markdown("### Three-regime comparison")
    st.caption(
        "Repeat the calculation at a chosen attraction, zero coupling, and a "
        "matching repulsion to see the size and central-density change clearly."
    )
    base_g = st.slider(r"$|g_{BF}|$ for the comparison", 0.0, 0.3, 0.05, 0.005,
                       format="%.3f")
    if st.button("Run comparison"):
        labels = [
            (f"attractive  g_BF=-{base_g:.3f}", -base_g),
            ("non-interacting  g_BF=0",         0.0),
            (f"repulsive  g_BF=+{base_g:.3f}", +base_g),
        ]
        results = []
        with st.spinner("Solving three regimes..."):
            for label, gbf in labels:
                results.append((
                    label, gbf,
                    get_bf_solution(g_B, gbf, mass_ratio, freq_ratio,
                                    float(N_B), float(N_F),
                                    r_max, n_r, mixing)
                ))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, (label, gbf, res) in zip(axes, results):
            ax.plot(res["r"], res["n_B"], "b-", lw=2, label="bosons")
            ax.plot(res["r"], res["n_F"], "r-", lw=2, label="fermions")
            ax.set_xlabel("r")
            ax.set_title(label)
            ax.grid(alpha=0.3)
            ax.legend()
            box = (
                f"mu_B={res['mu_B']:.2f}\n"
                f"mu_F={res['mu_F']:.2f}\n"
                f"R_B={res['R_B']:.2f}\n"
                f"R_F={res['R_F']:.2f}"
            )
            ax.text(0.97, 0.97, box, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc="white", ec="grey", alpha=0.85))
        axes[0].set_ylabel("density")
        safe_tight_layout(fig)
        st.pyplot(fig)


# =============================================================================
#  ABOUT
# =============================================================================

else:
    st.title("About this app")
    st.markdown(
        """
        This is the companion UI for **P452 Project 2 — Simulation of
        Many-Body Systems**.

        **Phase 1.**  Exact diagonalisation of the spin-1/2 antiferromagnetic
        Heisenberg model on $L_x\\times L_y$ square lattices with periodic
        boundary conditions.  The Hilbert space is block-diagonalised by
        $[\\hat H,\\hat S^z_{\\rm tot}]=0$; sparse Lanczos
        (`scipy.sparse.linalg.eigsh`) is used for the larger blocks.  The
        Zeeman field appears as an additive constant inside each fixed-$S^z$
        block, so a single $h=0$ diagonalisation per sector yields the entire
        $H/J\\in[0,h_{\\max}]$ curve algebraically.

        **Phase 2.**  Self-consistent Thomas-Fermi (bosons) + Local-Density
        Approximation (fermions) for a co-trapped Bose-Fermi mixture.  The
        chemical potentials are tuned at every iteration to enforce
        $\\int n_X\\,d^3r = N_X$ via bisection.  Stability against collapse /
        phase separation is monitored through the $2\\times 2$ stability matrix
        at the peak of $n_F$.

        **File map**

        * `heisenberg_ed.py` -- ED engine + observables (Phase 1)
        * `bf_mixture.py`    -- TF+LDA solver (Phase 2)
        * `run_phase1.py`    -- non-UI driver that produces the static plots
        * `app.py`           -- this Streamlit UI

        **Deploying.**  `pip install -r requirements.txt`, then
        `streamlit run app.py` locally, or push the directory to a public
        repository and deploy on https://streamlit.io/cloud .
        """
    )

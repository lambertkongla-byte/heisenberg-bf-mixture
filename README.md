# P452 Project 2 — Many-Body Simulation

Interactive simulation tool for two many-body quantum systems, built as a single Streamlit app:

- **Phase 1** — Exact diagonalisation of the spin-1/2 antiferromagnetic Heisenberg model on 2×2, 4×4, and 6×6 square lattices with periodic boundary conditions.
- **Phase 2** — Self-consistent Thomas-Fermi + Local-Density-Approximation solver for a trapped Bose-Fermi mixture.

## Live demo

Deployed on Streamlit Community Cloud: **[(https://heisenberg-bf-mixture-e4ra4wffpc5jr63fcx6xfb.streamlit.app/]**



## Run locally

You'll need Python 3.9 or newer.

```bash
# clone the repo
git clone https://github.com/YOUR-USERNAME/p452-project2.git
cd p452-project2

# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## Project structure

| file | purpose |
|---|---|
| `app.py` | Streamlit UI integrating both phases. |
| `heisenberg_ed.py` | Sparse exact-diagonalisation engine for the square-lattice Heisenberg model. Block-diagonalises by `Sᶻ_total`; uses `scipy.sparse.linalg.eigsh` for large blocks. |
| `bf_mixture.py` | Self-consistent TF + LDA solver in spherical harmonic traps. Bisection on `μ_B`, `μ_F`; linear-mix iteration. |
| `run_phase1.py` | Non-UI driver that produces static plots used in the write-up. |
| `requirements.txt` | Pip dependencies (streamlit, numpy, scipy, matplotlib, pandas). |

## Phase 1 — Heisenberg ED

Hamiltonian:

$$
\hat H \;=\; J\sum_{\langle i,j\rangle}\hat{\mathbf S}_i\!\cdot\!\hat{\mathbf S}_j \;-\; h\sum_i \hat S^z_i
$$

Two ideas make the computation fast:

1. **`[H, Ŝᶻ_total] = 0`** so the Hamiltonian block-diagonalises by `Sᶻ`. Each block has dimension `C(N, n_up)`.
2. The Zeeman term is a constant `−h·Sᶻ` *inside* each block, so a single `h = 0` diagonalisation per sector yields the entire `H/J` curve algebraically — no per-field re-diagonalisation needed.

**Validation**: 2×2 numerical spectrum matches the analytic 4-ring decomposition `{−2 (×1), −1 (×3), 0 (×7), +1 (×5)}`. 4×4 ground-state energy `E₀/J = −11.2285` matches the literature value.

**Limitation for 6×6**: the `Sᶻ_total = 0` sector has `C(36,18) ≈ 9.07 × 10⁹` states — too large to diagonalise without translation symmetry (which would require QuSpin or equivalent). The app computes only the high-`|Sᶻ|` sectors and clearly marks the low-field region of the magnetisation curve as not the true ground state.

## Phase 2 — Bose-Fermi mixture

Coupled equations:

$$
\mu_B = V_B(r) + g_B\,n_B(r) + g_{BF}\,n_F(r)
$$
$$
\mu_F = V_F(r) + E_F[n_F(r)] + g_{BF}\,n_B(r), \qquad E_F[n] = \frac{\hbar^2}{2 m_F}(6\pi^2 n)^{2/3}
$$

Solved self-consistently by alternating bisection on `μ_B` and `μ_F` to enforce particle-number constraints `∫ n_X d³r = N_X`, with linear mixing for stability.

Stability against collapse / phase separation is monitored via the `2×2` stability matrix evaluated at the peak fermion density. Numerical results match analytic Thomas-Fermi (bosons) and ideal-Fermi-gas-in-trap (fermions) references to four+ decimals when `g_BF = 0`.

The app supports inputs in either dimensionless coupling constants `g_B, g_BF` or scattering lengths `ã_B, ã_BF` (in units of the boson harmonic-oscillator length).

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (must be public for the free tier).
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **Create app** → pick the repo → set main file to `app.py` → **Deploy**.

The first build takes ~2–3 minutes. Subsequent pushes auto-redeploy within ~30 seconds.

**Memory note**: the free tier provides ~1 GB RAM, which is enough for 2×2 and 4×4 but tight for 6×6. To work around this, run 6×6 once locally to populate `_cache/spectra_6x6_*.pkl`, remove `_cache/` from `.gitignore`, and commit the pickle files. Streamlit Cloud will then load the spectra instantly instead of recomputing.

## License

MIT — feel free to use, modify, or fork.

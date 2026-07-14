# Tissue Growth with Phenotypic Heterogeneity — Finite-Volume Solvers

Finite-volume simulations of a continuum model for tissue growth in which many
cell phenotypes move, grow and compete for space. Seminar project (Modelling &
Simulation, B.Sc. Mathematics, TU Dresden), based on the model of
Dębiec, Mandal & Schmidtchen (2025).

## The model

Each phenotype `i = 1, …, N` is described by a density `n_i(x, t)`. All cells
move down the gradient of a shared pressure `p` that depends only on the total
density `rho = Σ_i n_i` (Darcy's law), and each phenotype grows logistically
until the tissue is locally full:

```
∂_t n_i + ∇·(n_i v) = n_i G_i(rho),     v = -∇p(rho),     p(rho) = rho^γ
```

With a logistic growth term `G_i(rho) = r_i (1 - rho)`, growth is fast in empty
regions and stops once `rho = 1`. Because all phenotypes share the same
velocity field but grow at different rates, faster phenotypes displace slower
ones, and initially mixed populations separate into distinct regions
(phase separation).

## What's implemented

Two solvers, both using a finite-volume discretization with a first-order
upwind flux and explicit-Euler time stepping:

- **1D, N species** (`DarcyDiscrete1D`) — the model above for an arbitrary
  number of phenotypes on an interval, with two initial conditions:
  strictly separated blocks and overlapping Gaussian bells.
- **2D, two species** (`TissueSim2D`) — two competing populations on the unit
  square, additionally including isotropic self-diffusion and asymmetric
  competition `n_k G_k (1 - n_k - c_k n_other)`, with Neumann or Dirichlet
  boundaries and an interactive viewer (time slider, mass-fraction contours).

## Project structure

```
src/tissuegrowth/
  model1d.py             # DarcyDiscrete1D — 1D N-species solver
  model2d.py             # TissueSim2D — 2D two-species solver
  initial_conditions.py  # separated_blocks, gaussian_bells (1D)
examples/
  run_1d_separated.py    # 1D, two separated blocks
  run_1d_gaussian.py     # 1D, overlapping Gaussian bells
  run_2d.py              # 2D run + interactive viewer
docs/
  tissue_growth_de.tex   # write-up, German  (+ .pdf)
  tissue_growth_en.tex   # write-up, English (+ .pdf)
  figures/               # figures, generated from the code in examples/
```

## Install & run

```bash
pip install -r requirements.txt

python examples/run_1d_gaussian.py     # 1D N-species
python examples/run_1d_separated.py    # 1D phase separation
python examples/run_2d.py              # 2D + interactive viewer
```

Or use the package directly:

```python
from tissuegrowth import DarcyDiscrete1D
from tissuegrowth.initial_conditions import gaussian_bells

model = DarcyDiscrete1D(n_species=5, L=30.0, J=100, T=20.0,
                        initial_condition=gaussian_bells)
model.simulate()
model.plot_snapshots(num_snapshots=5)
```

## Numerical scheme (short version)

Densities live at cell centers; velocities and fluxes at cell faces. The
interface velocity is `v = -Δp / Δx`, the flux is upwinded
(`F = v · n_upwind`), and each cell is updated by its net flux plus the
reaction term. The 2D solver adds a 5-point Laplacian for self-diffusion. The
time step must satisfy a CFL-type stability condition; the defaults in the
examples are chosen accordingly.

## Write-up

The full derivation, discretization and results are written up in two languages,
LaTeX sources and compiled PDFs alike:

- English — [`docs/tissue_growth_en.pdf`](docs/tissue_growth_en.pdf)
- Deutsch — [`docs/tissue_growth_de.pdf`](docs/tissue_growth_de.pdf)

Both versions are kept in sync and share the figures in `docs/figures/`, which
are produced by the scripts in `examples/`.

## Reference

T. Dębiec, A. Mandal, M. Schmidtchen, *Incompressible limit for a tissue growth
model with phenotypic heterogeneity* (2025).

## License

MIT — see [LICENSE](LICENSE).

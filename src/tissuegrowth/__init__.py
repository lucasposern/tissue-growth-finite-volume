"""Finite-volume solvers for a tissue-growth model with phenotypic heterogeneity.

The package implements the Debiec-Mandal-Schmidtchen phenotype model, in which
every phenotype moves along the negative gradient of a shared pressure (Darcy's
law) and grows logistically until the tissue is locally full:

    d_t n_i + div(n_i v) = n_i r_i (1 - rho),   v = -grad p(rho),   rho = sum_i n_i.

Modules
-------
model1d      : ``DarcyDiscrete1D`` -- 1D N-species solver.
model2d      : ``TissueSim2D``     -- 2D two-species solver with competition.
initial_conditions : reusable initial density profiles for the 1D model.
"""

from .model1d import DarcyDiscrete1D
from .model2d import TissueSim2D
from . import initial_conditions

__all__ = ["DarcyDiscrete1D", "TissueSim2D", "initial_conditions"]

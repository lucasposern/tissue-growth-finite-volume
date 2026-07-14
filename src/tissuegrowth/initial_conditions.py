"""Initial density profiles for the 1D N-species tissue-growth model.

Every initializer has the signature ``f(x, n_species, L, rng) -> ndarray`` and
returns an array ``n`` of shape ``(n_species, J)`` with the initial density of
each species in each finite-volume cell. The total density ``rho = sum_i n_i``
is rescaled to never exceed 1 (a full tissue).
"""
from __future__ import annotations

import numpy as np


def _finalize(n, noise, rng):
    """Add small multiplicative noise, clip to >= 0 and rescale rho to <= 1."""
    n = n * (1.0 + noise * rng.standard_normal(n.shape))
    n = np.clip(n, 0.0, None)
    total_max = n.sum(axis=0).max()
    if total_max > 1.0:
        n = n / total_max
    return n


def separated_blocks(x, n_species, L, rng, amplitude=0.8, noise=0.05):
    """Two spatially separated blocks, each occupied by a single species.

    Species 1 sits in a smooth parabolic bump around ``0.25 * L`` and species 2
    around ``0.75 * L``; any further species start empty. The multiplicative
    noise only acts where the density is already positive, so no species is
    created in an empty cell. Requires ``n_species >= 2``.
    """
    if n_species < 2:
        raise ValueError("separated_blocks needs at least two species.")
    n = np.zeros((n_species, x.size))
    left_shape = np.maximum(0.0, 1.0 - ((x - 0.25 * L) / (0.12 * L)) ** 2)
    right_shape = np.maximum(0.0, 1.0 - ((x - 0.75 * L) / (0.12 * L)) ** 2)
    n[0] = amplitude * left_shape
    n[1] = amplitude * right_shape
    return _finalize(n, noise, rng)


def gaussian_bells(x, n_species, L, rng, amplitude=0.5, noise=0.05):
    """Overlapping Gaussian bells so that all species compete from the start.

    Species ``i`` is centered at ``(i + 1) / (n_species + 1) * L`` with width
    ``L / (3 * n_species)``, wide enough that neighbouring profiles overlap and
    the phenotypes immediately compete for space.
    """
    n = np.zeros((n_species, x.size))
    width = L / (3 * n_species)
    for i in range(n_species):
        center = L * (i + 1) / (n_species + 1)
        n[i] = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return _finalize(n, noise, rng)

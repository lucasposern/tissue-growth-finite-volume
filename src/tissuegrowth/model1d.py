"""1D finite-volume solver for the N-species Darcy tissue-growth model.

For each phenotype ``i = 1, ..., N`` the density ``n_i(x, t)`` obeys

    d_t n_i + d_x (n_i v) = n_i r_i (1 - rho),
    rho = sum_j n_j,   p(rho) = rho**gamma,   v = -d_x p(rho)   (Darcy's law).

Space is discretized with a finite-volume scheme on ``J`` cells using a
first-order upwind flux; time is advanced with explicit Euler. The two initial
conditions used in the report (strictly separated blocks and overlapping
Gaussian bells) live in :mod:`tissuegrowth.initial_conditions` and are passed in
as the ``initial_condition`` argument, so the same solver serves both cases.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .initial_conditions import gaussian_bells


class DarcyDiscrete1D:
    """Explicit finite-volume solver for the 1D N-species model."""

    def __init__(self, n_species=5, L=10.0, J=200, T=5.0, dt=1e-4, gamma=2.0,
                 r_rates=None, r_max=1.0, r_min=0.2, random_seed=0,
                 initial_condition=gaussian_bells):
        """
        Parameters
        ----------
        n_species : int
            Number of phenotypes ``N``.
        L : float
            Length of the interval ``[0, L]``.
        J : int
            Number of finite-volume cells.
        T : float
            Final simulation time.
        dt : float
            Time-step size.
        gamma : float
            Exponent of the pressure law ``p(rho) = rho**gamma``.
        r_rates : None, float or array-like, optional
            Growth rates ``r_i``. ``None`` draws a linearly spaced set between
            ``r_max`` and ``r_min`` and randomly permutes it; a scalar assigns
            the same rate to every species; an array of length ``N`` is used
            directly.
        r_max, r_min : float
            Bounds for the default growth rates.
        random_seed : int
            Seed for the initial data and the default growth-rate assignment.
        initial_condition : callable
            ``f(x, n_species, L, rng) -> ndarray`` of shape ``(N, J)``.
        """
        self.N = n_species
        self.L = L
        self.J = J
        self.T = T
        self.dt = dt
        self.gamma = gamma

        self.dx = L / J
        self.x = (np.arange(J) + 0.5) * self.dx  # cell centers

        self.r = self._make_rates(r_rates, r_max, r_min, random_seed)

        rng = np.random.default_rng(random_seed)
        self.n = np.asarray(initial_condition(self.x, self.N, self.L, rng),
                            dtype=float)
        if self.n.shape != (self.N, self.J):
            raise ValueError(
                f"initial_condition returned shape {self.n.shape}, "
                f"expected {(self.N, self.J)}."
            )

        self.time = 0.0
        self.history = []
        self.history_times = []

    def _make_rates(self, r_rates, r_max, r_min, random_seed):
        if r_rates is None:
            base = np.linspace(r_max, r_min, self.N)
            return np.random.default_rng(random_seed + 999).permutation(base)
        r = np.asarray(r_rates, dtype=float)
        if r.ndim == 0:
            return np.full(self.N, float(r))
        if r.size == self.N:
            return r
        raise ValueError(f"r_rates has length {r.size}, but n_species = {self.N}.")

    # -- spatial operators ------------------------------------------------
    def compute_velocity(self):
        """Interface velocity ``v = -d_x p(rho)`` at the ``J + 1`` cell faces."""
        rho = self.n.sum(axis=0)
        p = rho ** self.gamma
        dp_dx = np.zeros(self.J + 1)
        dp_dx[1:-1] = (p[1:] - p[:-1]) / self.dx  # inner faces; walls stay 0
        return -dp_dx

    def compute_flux(self):
        """Upwind flux ``F_i = n_i v`` at the faces; zero flux at the walls.

        Upwinding takes the density from the upwind cell: the left value where
        ``v >= 0`` and the right value where ``v < 0``.
        """
        v = self.compute_velocity()
        v_inner = v[1:-1]
        take_left = v_inner >= 0.0
        flux = np.zeros((self.N, self.J + 1))
        for i in range(self.N):
            n_i = self.n[i]
            flux[i, 1:-1] = np.where(take_left,
                                     v_inner * n_i[:-1],
                                     v_inner * n_i[1:])
        return flux

    def compute_reaction(self):
        """Logistic growth ``n_i r_i (1 - rho)``; growth stops once ``rho = 1``."""
        rho = self.n.sum(axis=0)
        return self.n * (self.r[:, None] * (1.0 - rho)[None, :])

    # -- time stepping ----------------------------------------------------
    def step(self):
        """Advance one explicit-Euler step ``d_t n_i = -d_x F_i + reaction``."""
        flux = self.compute_flux()
        reaction = self.compute_reaction()
        self.n += self.dt * (-(flux[:, 1:] - flux[:, :-1]) / self.dx + reaction)
        np.clip(self.n, 0.0, None, out=self.n)
        self.time += self.dt

    def simulate(self, store_every=200, verbose=True):
        """Run until ``T``, storing the state every ``store_every`` steps."""
        num_steps = int(self.T / self.dt)
        self.history = []
        self.history_times = []
        for step in range(num_steps):
            if step % store_every == 0:
                self.history.append(self.n.copy())
                self.history_times.append(self.time)
                if verbose and step % (20 * store_every) == 0:
                    print(f"t = {self.time:.3f} / {self.T:.3f}")
            self.step()
        self.history.append(self.n.copy())
        self.history_times.append(self.time)
        return self.history

    # -- visualization ----------------------------------------------------
    def plot_snapshots(self, num_snapshots=5):
        """Plot densities (top) and species fractions (bottom) at snapshots."""
        if not self.history:
            raise RuntimeError("Call simulate() first.")
        indices = np.linspace(0, len(self.history) - 1, num_snapshots, dtype=int)
        fig, axes = plt.subplots(2, num_snapshots,
                                 figsize=(5 * num_snapshots, 6), sharex=True)
        if num_snapshots == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        colors = [plt.get_cmap("tab10")(i) for i in range(self.N)]

        for ax_top, ax_bot, idx in zip(axes[0], axes[1], indices):
            state = self.history[idx]
            total = state.sum(axis=0)
            for i in range(self.N):
                ax_top.plot(self.x, state[i], color=colors[i], linewidth=1.5,
                            label=f"Species {i + 1}")
            ax_top.plot(self.x, total, "k--", linewidth=2.0, label="Total density")
            ax_top.set_ylabel("Density")
            ax_top.set_title(f"t = {self.history_times[idx]:.2f}")
            ax_top.grid(True)

            fractions = np.zeros_like(state)
            mask = total > 0
            fractions[:, mask] = state[:, mask] / total[mask]
            bottom = np.zeros(self.J)
            for i in range(self.N):
                ax_bot.bar(self.x, fractions[i], width=self.dx * 0.9, bottom=bottom,
                           color=colors[i], edgecolor="k", linewidth=0.2)
                bottom += fractions[i]
            ax_bot.set_ylim(0.0, 1.0)
            ax_bot.set_ylabel("Fraction")
            ax_bot.grid(True, axis="y", linestyle=":", linewidth=0.5)

        axes[1, -1].set_xlabel("Position x")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.show()

    def animate_simulation(self, history=None, interval=100):
        """Return a matplotlib animation of the density profiles over time."""
        if history is None:
            history = self.history
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [plt.get_cmap("tab10")(i) for i in range(self.N)]

        def update(frame):
            ax.clear()
            state = history[frame]
            total = state.sum(axis=0)
            for i in range(self.N):
                ax.plot(self.x, state[i], color=colors[i], linewidth=1.5,
                        label=f"Species {i + 1}")
            ax.plot(self.x, total, "k--", linewidth=2.0, label="Total density")
            ax.set_ylim(0.0, max(1.0, total.max() * 1.1))
            ax.set_xlabel("Position x")
            ax.set_ylabel("Density")
            ax.set_title(f"t = {self.history_times[frame]:.2f}")
            ax.grid(True)
            ax.legend(loc="upper right")

        anim = FuncAnimation(fig, update, frames=len(history), interval=interval)
        plt.close(fig)
        return anim

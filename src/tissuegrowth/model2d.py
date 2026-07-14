"""2D finite-volume solver for two competing cell populations.

Two densities ``n1, n2`` evolve on the unit square under

    d_t n_k = -div(F_k) + eps_k * lap(n_k) + n_k G_k (1 - n_k - c_k n_other),

where the advective flux ``F_k = n_k v_k`` follows the shared pressure through
``v_k = -beta_k * grad(n_bar)`` with ``n_bar = (n1 + n2) / 2``. The flux uses a
first-order upwind scheme, diffusion a 5-point Laplacian, and time is advanced
with explicit Euler. Neumann (no-flux) or Dirichlet (zero) boundaries are
supported.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import LinearSegmentedColormap


class TissueSim2D:
    """2D simulation of two competing cell populations ``n1`` and ``n2``.

    The model couples growth, interspecific competition, pressure-driven
    advection and self-diffusion.

    Examples
    --------
    >>> sim = TissueSim2D(grid_size=80, G1=1.0, beta1=1.0)
    >>> sim.run()
    >>> sim.visualize()
    """

    def __init__(self,
                 grid_size=50,          # number of cells per axis (grid_size**2 control volumes)
                 count_timesteps=1_000_000,  # number of time steps to simulate
                 dt=5e-6,               # time-step size
                 snapshot_every=100,    # store every k-th step for visualization
                 boundary="neumann",    # boundary condition: "neumann" or "dirichlet"
                 G1=1.0, G2=1.0,        # growth rates of n1, n2
                 comp1=1.0, comp2=1.0,  # inhibition of n1 by n2 and of n2 by n1
                 beta1=1.0, beta2=1.0,  # pressure response (higher -> more advection)
                 eps1=0.0, eps2=0.0,    # isotropic self-diffusion of n1, n2
                 center1=(0.55, 0.55), center2=(0.45, 0.45),  # initial peak centers
                 peak1=0.03, peak2=0.03,      # initial peak densities
                 sigma1=0.05, sigma2=0.05,    # initial Gaussian widths
                 background1=0.0, background2=0.0):  # uniform background densities
        # -- grid --
        self.grid_size = grid_size
        self.dx = 1.0 / (grid_size - 1)
        self.dy = self.dx

        # -- time --
        self.count_timesteps = count_timesteps
        self.dt = dt
        self.current_timestep = 0
        self.snapshot_every = snapshot_every

        # -- advection / diffusion / boundary --
        self.beta1, self.beta2 = beta1, beta2
        self.eps1, self.eps2 = eps1, eps2
        self.boundary = boundary

        # -- growth and competition --
        self.G1, self.G2 = G1, G2
        self.comp1, self.comp2 = comp1, comp2

        # -- densities (full precision for the update) --
        self.n1 = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.n2 = np.zeros((grid_size, grid_size), dtype=np.float64)
        # stored snapshots (reduced precision to save memory)
        n_snaps = count_timesteps // self.snapshot_every
        self.n1_snap = np.zeros((grid_size, grid_size, n_snaps), dtype=np.float32)
        self.n2_snap = np.zeros((grid_size, grid_size, n_snaps), dtype=np.float32)

        # -- initial data: two 2D Gaussians plus a uniform background --
        self.center1, self.center2 = center1, center2
        self.peak1, self.peak2 = peak1, peak2
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.background1, self.background2 = background1, background2

        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y, indexing="ij")
        gauss1 = peak1 * np.exp(-((X - center1[1]) ** 2 + (Y - center1[0]) ** 2) / (2 * sigma1 ** 2))
        gauss2 = peak2 * np.exp(-((X - center2[1]) ** 2 + (Y - center2[0]) ** 2) / (2 * sigma2 ** 2))
        self.n1[:, :] = gauss1 + background1
        self.n2[:, :] = gauss2 + background2

    # -- one explicit-Euler update ---------------------------------------
    def update_n(self):
        """Advance the densities one explicit-Euler step.

        Combines pressure-driven advection (upwind flux), self-diffusion and
        logistic growth with interspecific competition.
        """
        # shared (rescaled) pressure variable
        n_bar = (self.n1 + self.n2) / 2

        # face velocities  v = -beta * grad(n_bar)
        v1_x = np.zeros((self.grid_size + 1, self.grid_size))
        v1_x[1:-1, :] = -self.beta1 * (n_bar[1:, :] - n_bar[:-1, :]) / self.dx
        v1_y = np.zeros((self.grid_size, self.grid_size + 1))
        v1_y[:, 1:-1] = -self.beta1 * (n_bar[:, 1:] - n_bar[:, :-1]) / self.dy
        v2_x = np.zeros((self.grid_size + 1, self.grid_size))
        v2_x[1:-1, :] = -self.beta2 * (n_bar[1:, :] - n_bar[:-1, :]) / self.dx
        v2_y = np.zeros((self.grid_size, self.grid_size + 1))
        v2_y[:, 1:-1] = -self.beta2 * (n_bar[:, 1:] - n_bar[:, :-1]) / self.dy

        # upwind fluxes  F = v * n  (density taken from the upwind cell)
        F1_x = np.zeros_like(v1_x)
        F1_x[1:-1, :] = np.where(v1_x[1:-1, :] >= 0, v1_x[1:-1, :] * self.n1[:-1, :],
                                 v1_x[1:-1, :] * self.n1[1:, :])
        F1_y = np.zeros_like(v1_y)
        F1_y[:, 1:-1] = np.where(v1_y[:, 1:-1] >= 0, v1_y[:, 1:-1] * self.n1[:, :-1],
                                 v1_y[:, 1:-1] * self.n1[:, 1:])
        F2_x = np.zeros_like(v2_x)
        F2_x[1:-1, :] = np.where(v2_x[1:-1, :] >= 0, v2_x[1:-1, :] * self.n2[:-1, :],
                                 v2_x[1:-1, :] * self.n2[1:, :])
        F2_y = np.zeros_like(v2_y)
        F2_y[:, 1:-1] = np.where(v2_y[:, 1:-1] >= 0, v2_y[:, 1:-1] * self.n2[:, :-1],
                                 v2_y[:, 1:-1] * self.n2[:, 1:])

        # flux divergence (finite-volume advection term)
        div1 = (F1_x[1:, :] - F1_x[:-1, :]) / self.dx + (F1_y[:, 1:] - F1_y[:, :-1]) / self.dy
        div2 = (F2_x[1:, :] - F2_x[:-1, :]) / self.dx + (F2_y[:, 1:] - F2_y[:, :-1]) / self.dy

        # self-diffusion (5-point Laplacian, interior only)
        lap1 = np.zeros_like(self.n1)
        lap2 = np.zeros_like(self.n2)
        lap1[1:-1, 1:-1] = (self.n1[2:, 1:-1] + self.n1[:-2, 1:-1]
                            + self.n1[1:-1, 2:] + self.n1[1:-1, :-2]
                            - 4 * self.n1[1:-1, 1:-1]) / self.dx ** 2
        lap2[1:-1, 1:-1] = (self.n2[2:, 1:-1] + self.n2[:-2, 1:-1]
                            + self.n2[1:-1, 2:] + self.n2[1:-1, :-2]
                            - 4 * self.n2[1:-1, 1:-1]) / self.dx ** 2

        # logistic growth with asymmetric competition
        R1 = self.n1 * (self.G1 * (1 - self.n1 - self.comp1 * self.n2))
        R2 = self.n2 * (self.G2 * (1 - self.n2 - self.comp2 * self.n1))

        # explicit-Euler update
        n1_new = self.n1 + self.dt * (-div1 + self.eps1 * lap1 + R1)
        n2_new = self.n2 + self.dt * (-div2 + self.eps2 * lap2 + R2)

        # boundary conditions
        if self.boundary == "neumann":
            n1_new[0, :], n1_new[-1, :] = n1_new[1, :], n1_new[-2, :]
            n1_new[:, 0], n1_new[:, -1] = n1_new[:, 1], n1_new[:, -2]
            n2_new[0, :], n2_new[-1, :] = n2_new[1, :], n2_new[-2, :]
            n2_new[:, 0], n2_new[:, -1] = n2_new[:, 1], n2_new[:, -2]
        elif self.boundary == "dirichlet":
            n1_new[0, :] = n1_new[-1, :] = n1_new[:, 0] = n1_new[:, -1] = 0
            n2_new[0, :] = n2_new[-1, :] = n2_new[:, 0] = n2_new[:, -1] = 0

        # no negative densities
        np.maximum(n1_new, 0.0, out=n1_new)
        np.maximum(n2_new, 0.0, out=n2_new)

        self.n1, self.n2 = n1_new, n2_new

    # -- run --------------------------------------------------------------
    def run(self, verbose=True):
        """Run the full simulation, storing snapshots every ``snapshot_every`` steps."""
        for t in range(self.count_timesteps):
            self.current_timestep = t
            self.update_n()
            if t % self.snapshot_every == 0:
                k = t // self.snapshot_every
                self.n1_snap[:, :, k] = self.n1
                self.n2_snap[:, :, k] = self.n2
                if verbose:
                    print(f"{t / self.count_timesteps * 100:.2f}%")
        if verbose:
            print("100.00%")
            print(f"Simulation complete. total_mass={self.n1.sum() + self.n2.sum():.6e}")

    # -- interactive visualization ---------------------------------------
    def visualize(self):
        """Interactive viewer: density maps, an RGB overlay and mass-fraction contours."""
        timesteps = self.n1_snap.shape[2]
        n1_max = self.n1_snap.max()
        n2_max = self.n2_snap.max()
        n12_max = (self.n1_snap + self.n2_snap).max()
        max_val = max(n1_max, n2_max)

        from skimage.measure import find_contours

        n1 = self.n1_snap[:, :, 0]
        n2 = self.n2_snap[:, :, 0]
        n12 = n1 + n2

        red = LinearSegmentedColormap.from_list("red", [(1, 1, 1), (1, 0, 0)])
        blue = LinearSegmentedColormap.from_list("blue", [(1, 1, 1), (0, 0, 1)])
        purple = LinearSegmentedColormap.from_list("purple", [(1, 1, 1), (0.5, 0, 0.5)])

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        plt.subplots_adjust(bottom=0.40, right=0.85)

        im0 = axes[0].imshow(n1, origin="lower", extent=[0, 1, 0, 1], cmap=red, vmin=0, vmax=n1_max)
        axes[0].set_title("n1"); fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(n2, origin="lower", extent=[0, 1, 0, 1], cmap=blue, vmin=0, vmax=n2_max)
        axes[1].set_title("n2"); fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(n12, origin="lower", extent=[0, 1, 0, 1], cmap=purple, vmin=0, vmax=n12_max)
        axes[2].set_title("n1 + n2"); fig.colorbar(im2, ax=axes[2])

        mixed_rgb = np.ones((self.grid_size, self.grid_size, 3))
        n1_norm = np.clip(n1 / max_val, 0, 1)
        n2_norm = np.clip(n2 / max_val, 0, 1)
        mixed_rgb[..., 1] -= n1_norm; mixed_rgb[..., 2] -= n1_norm
        mixed_rgb[..., 0] -= n2_norm; mixed_rgb[..., 1] -= n2_norm
        mixed_rgb = np.clip(mixed_rgb, 0, 1)
        im3 = axes[3].imshow(mixed_rgb, origin="lower", extent=[0, 1, 0, 1])
        axes[3].set_title("n1 (red) / n2 (blue)")

        param_text = (
            f"grid_size: {self.grid_size}\n"
            f"boundary: {self.boundary}\n"
            f"G1: {self.G1:.2f}, G2: {self.G2:.2f}\n"
            f"comp1: {self.comp1:.2f}, comp2: {self.comp2:.2f}\n"
            f"beta1: {self.beta1:.2f}, beta2: {self.beta2:.2f}\n"
            f"eps1: {self.eps1:.2f}, eps2: {self.eps2:.2f}"
        )
        axes[4].axis("off")
        axes[4].text(0.05, 0.5, param_text, fontsize=12, verticalalignment="center",
                     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))
        axes[4].set_title("Parameters")

        slider_t = Slider(plt.axes([0.25, 0.22, 0.5, 0.03]), "Timestep", 0, timesteps - 1, valinit=0, valstep=1)
        slider_p = Slider(plt.axes([0.25, 0.17, 0.5, 0.03]), "Mass fraction [%]", 50, 99, valinit=85)
        check = CheckButtons(plt.axes([0.05, 0.12, 0.12, 0.06]), ["Contours"], [True])

        state = {"show_contours": True}
        contours = {"n1": [], "n2": [], "n12": []}

        def mass_threshold(data, fraction):
            flat = data.ravel()
            idx = np.argsort(flat)[::-1]
            cumsum = np.cumsum(flat[idx])
            return flat[idx][np.searchsorted(cumsum, fraction * cumsum[-1])]

        def clear_contours():
            for group in contours.values():
                for line in group:
                    line.remove()
                group.clear()

        def update(_=None):
            t = int(slider_t.val)
            frac = slider_p.val / 100.0
            n1 = self.n1_snap[:, :, t]
            n2 = self.n2_snap[:, :, t]
            n12 = n1 + n2
            im0.set_data(n1); im1.set_data(n2); im2.set_data(n12)

            n1_norm = np.clip(n1 / max_val, 0, 1)
            n2_norm = np.clip(n2 / max_val, 0, 1)
            mixed_rgb[:] = 1.0
            mixed_rgb[..., 1] -= n1_norm; mixed_rgb[..., 2] -= n1_norm
            mixed_rgb[..., 0] -= n2_norm; mixed_rgb[..., 1] -= n2_norm
            mixed_rgb[:] = np.clip(mixed_rgb, 0, 1)
            im3.set_data(mixed_rgb)

            clear_contours()
            if state["show_contours"]:
                for c in find_contours(n1, mass_threshold(n1, frac)):
                    c /= (self.grid_size - 1)
                    contours["n1"].append(axes[0].plot(c[:, 1], c[:, 0], color=(0.5, 0, 0), lw=1.5)[0])
                    contours["n1"].append(axes[3].plot(c[:, 1], c[:, 0], color=(0.5, 0, 0), lw=1.5)[0])
                for c in find_contours(n2, mass_threshold(n2, frac)):
                    c /= (self.grid_size - 1)
                    contours["n2"].append(axes[1].plot(c[:, 1], c[:, 0], color=(0, 0, 0.5), lw=1.5)[0])
                    contours["n2"].append(axes[3].plot(c[:, 1], c[:, 0], color=(0, 0, 0.5), lw=1.5)[0])
                for c in find_contours(n12, mass_threshold(n12, frac)):
                    c /= (self.grid_size - 1)
                    contours["n12"].append(axes[2].plot(c[:, 1], c[:, 0], "k", lw=1.5)[0])
            fig.canvas.draw_idle()

        def toggle(_label):
            state["show_contours"] = not state["show_contours"]
            update()

        slider_t.on_changed(update)
        slider_p.on_changed(update)
        check.on_clicked(toggle)
        update()
        plt.show()

    # -- persistence ------------------------------------------------------
    def save(self, filename=""):
        """Serialize the whole simulation object with pickle."""
        if filename == "":
            filename = input("Enter filename to save: ")
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename=""):
        """Replace the current state with a previously saved simulation."""
        if filename == "":
            filename = input("Enter filename to load: ")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, TissueSim2D):
            raise TypeError("Loaded file does not contain a TissueSim2D object.")
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)

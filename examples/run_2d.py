"""2D two-species run followed by the interactive viewer.

Two populations start as Gaussian blobs and compete for space. After the run,
an interactive window lets you scrub through time, adjust the mass-fraction
contour level, and toggle contours on the density maps.
"""
from tissuegrowth import TissueSim2D


def main():
    sim = TissueSim2D(
        grid_size=60,
        count_timesteps=200_000,
        dt=5e-6,
        snapshot_every=1000,
        G1=1.0, G2=1.0,
        comp1=1.0, comp2=1.0,
        beta1=1.0, beta2=1.0,
        boundary="neumann",
    )
    sim.run()
    sim.visualize()
    return sim


if __name__ == "__main__":
    main()

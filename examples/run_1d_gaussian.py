"""1D N-species run starting from overlapping Gaussian bells.

All species start in the same region and compete immediately; over time the
faster-growing phenotypes displace the slower ones.
"""
from tissuegrowth import DarcyDiscrete1D
from tissuegrowth.initial_conditions import gaussian_bells


def main():
    model = DarcyDiscrete1D(
        n_species=5, L=30.0, J=100, T=20.0, dt=5e-4, gamma=2.0,
        initial_condition=gaussian_bells,
    )
    model.simulate(store_every=200)
    model.plot_snapshots(num_snapshots=5)
    return model


if __name__ == "__main__":
    main()

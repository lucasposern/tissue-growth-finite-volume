"""1D N-species run starting from two strictly separated blocks.

Species 1 and species 2 start in disjoint regions and grow into the free space
between them; the lower panels show the emerging phase separation.
"""
from tissuegrowth import DarcyDiscrete1D
from tissuegrowth.initial_conditions import separated_blocks


def main():
    model = DarcyDiscrete1D(
        n_species=2, L=10.0, J=100, T=20.0, dt=5e-4, gamma=2.0,
        initial_condition=separated_blocks,
    )
    model.simulate(store_every=200)
    model.plot_snapshots(num_snapshots=5)
    return model


if __name__ == "__main__":
    main()

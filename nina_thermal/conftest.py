"""Shared fixtures for the training test suite.

The synthetic-dataset builders in :mod:`nina_thermal.training.fixtures` do real
sensor simulation, so the multi-root tree is built once per session and reused.
"""

import pytest

from nina_thermal.training import fixtures


@pytest.fixture(scope="session")
def single_root(tmp_path_factory):
    """One ``04_dataset/`` tree (fixed AGC, profile B) with its summary.

    Returns:
        ``(dataset_dir: Path, summary: dict)``.
    """
    dataset_dir = tmp_path_factory.mktemp("single") / "04_dataset"
    spec = fixtures.FixtureSpec(
        n_granules=10,
        patches_per_granule=6,
        agc_modes=("fixed",),
        profiles=("B",),
        seed=7,
    )
    summary = fixtures.generate_synthetic_dataset(dataset_dir, spec)
    return dataset_dir, summary


@pytest.fixture(scope="session")
def multiroot(tmp_path_factory):
    """Three overlapping roots + a FIRMS CSV each.

    Returns:
        A :class:`nina_thermal.training.fixtures.MultiRootFixture`.
    """
    base = tmp_path_factory.mktemp("multiroot")
    return fixtures.generate_multiroot_fixture(base, seed=42)

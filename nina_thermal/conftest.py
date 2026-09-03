"""Shared fixtures for the nina_thermal test suites.

The synthetic-dataset builders in :mod:`nina_thermal.training.fixtures` do real
sensor simulation, so the multi-root trees are built once per session and reused.
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
    """Three small overlapping roots + a FIRMS CSV each (fast; conflicts guaranteed).

    Returns:
        A :class:`nina_thermal.training.fixtures.MultiRootFixture`.
    """
    base = tmp_path_factory.mktemp("multiroot")
    return fixtures.generate_multiroot_fixture(base, seed=42)


@pytest.fixture(scope="session")
def eval_multiroot(tmp_path_factory):
    """Larger multi-root fixture whose test split clears ``MIN_FIRE_FOR_EVAL``.

    Tuned (seed 2, 20 granules/root, 14 patches/granule) so the profile-B test
    slice holds >= 20 fire patches — the non-degenerate path for the evaluation
    report.

    Returns:
        A :class:`nina_thermal.training.fixtures.MultiRootFixture`.
    """
    base = tmp_path_factory.mktemp("eval_multiroot")
    return fixtures.generate_multiroot_fixture(
        base, seed=2, granules_per_root=20, patches_per_granule=14, overlap=6
    )


@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory):
    """A bare MobileNetV3-Small state dict + a matching ``training_report.json``.

    Random weights (``pretrained=False``) — enough to exercise the evaluation
    and quantization plumbing without an ImageNet download.

    Returns:
        ``(checkpoint_path: Path, out_dir: Path)`` — ``out_dir`` also holds the
        stub ``training_report.json`` (``selected_threshold`` 0.45, beta sqrt(3)).
    """
    import json
    import math

    import torch

    from nina_thermal.training import train_thermal

    out_dir = tmp_path_factory.mktemp("ckpt")
    ckpt = out_dir / train_thermal.CHECKPOINT_NAME
    torch.save(train_thermal.build_model(pretrained=False).state_dict(), ckpt)
    (out_dir / "training_report.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.45,
                "hyperparameters": {"beta": round(math.sqrt(3.0), 6), "pos_weight": 3.0},
            }
        )
    )
    return ckpt, out_dir

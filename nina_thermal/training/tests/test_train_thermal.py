"""Tests for nina_thermal.training.train_thermal — model + transforms + loops.

These exercise the library functions with random weights
(``pretrained=False``); the ImageNet download and the full argparse/checkpoint
path are out of scope for a wiring test.
"""

import math

import torch
from torch.utils.data import DataLoader

from nina_thermal.training import train_thermal
from nina_thermal.training.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)


class TestBuildModel:
    def test_single_logit_head_and_param_count(self):
        model = train_thermal.build_model(pretrained=False)
        x = torch.zeros(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        out = model(x)
        assert out.shape == (2, 1)
        n_params = sum(p.numel() for p in model.parameters())
        # MobileNetV3-Small with a Dropout->Linear(576,1) head.
        assert 900_000 < n_params < 1_000_000

    def test_head_is_dropout_then_linear(self):
        model = train_thermal.build_model(pretrained=False)
        assert isinstance(model.classifier[0], torch.nn.Dropout)
        assert isinstance(model.classifier[-1], torch.nn.Linear)
        assert model.classifier[-1].out_features == 1

    def test_freeze_toggles_feature_grads(self):
        model = train_thermal.build_model(pretrained=False)
        train_thermal.set_backbone_frozen(model, True)
        assert all(not p.requires_grad for p in model.features.parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())
        train_thermal.set_backbone_frozen(model, False)
        assert all(p.requires_grad for p in model.features.parameters())


class TestBuildTransforms:
    def test_native_resolution_no_resize(self):
        tf = train_thermal.build_transforms(augment=False)
        names = [type(t).__name__ for t in tf.transforms]
        assert "Resize" not in names
        assert "Normalize" in names and "Grayscale" in names

    def test_output_is_three_channel_native_size(self):
        from PIL import Image

        img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=128)
        out = train_thermal.build_transforms(augment=False)(img)
        assert out.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_augment_adds_flips(self):
        aug = [type(t).__name__ for t in train_thermal.build_transforms(True).transforms]
        plain = [type(t).__name__ for t in train_thermal.build_transforms(False).transforms]
        assert len(aug) > len(plain)
        assert any("Flip" in n for n in aug)
        assert not any("ColorJitter" in n or "Brightness" in n for n in aug)


class TestEpochLoops:
    def _loader(self, multiroot):
        records, _ = load_merged_records(multiroot.roots)
        samples = split_samples(expand_to_samples(records, ["B"], ["fixed"]))["train"]
        ds = ThermalDataset(samples, train_thermal.build_transforms(augment=False))
        return DataLoader(ds, batch_size=8)

    def test_run_epoch_trains_and_returns_finite_loss(self, multiroot):
        train_thermal.set_seed(0)
        model = train_thermal.build_model(pretrained=False)
        loader = self._loader(multiroot)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [p.clone() for p in model.classifier.parameters()]
        loss = train_thermal.run_epoch(
            model, loader, crit, opt, torch.device("cpu"), training=True
        )
        assert loss > 0 and math.isfinite(loss)
        after = list(model.classifier.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(after, before))

    def test_eval_epoch_does_not_update_weights(self, multiroot):
        model = train_thermal.build_model(pretrained=False)
        loader = self._loader(multiroot)
        crit = torch.nn.BCEWithLogitsLoss()
        snapshot = [p.clone() for p in model.parameters()]
        train_thermal.run_epoch(model, loader, crit, None, torch.device("cpu"), False)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), snapshot))

    def test_collect_scores_shapes_and_range(self, multiroot):
        model = train_thermal.build_model(pretrained=False)
        loader = self._loader(multiroot)
        y_true, y_score = train_thermal.collect_scores(
            model, loader, torch.device("cpu")
        )
        assert len(y_true) == len(y_score) == len(loader.dataset)
        assert all(v in (0, 1) for v in y_true)
        assert all(0.0 <= s <= 1.0 for s in y_score)


class TestParseArgs:
    def test_defaults_match_module_constants(self):
        args = train_thermal.parse_args([])
        assert args.profiles == train_thermal.TRAIN_PROFILES
        assert args.agc_modes == train_thermal.TRAIN_AGC_MODES
        assert args.seed == train_thermal.SEED

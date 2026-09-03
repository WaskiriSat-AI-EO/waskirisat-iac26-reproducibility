"""Tests for nina_thermal.quantization.qat — PT2E QAT plumbing.

``ai-edge-torch`` was renamed to ``litert-torch`` and PT2E moved from
``torch.ao`` to ``torchao``; the module probes for both.  These tests confirm
the probing resolves on the installed stack and that the wrapper / capture /
convert steps run end to end on a tiny graph.
"""

import torch
from torch.utils.data import DataLoader

from nina_thermal.quantization import qat
from nina_thermal.training import train_thermal
from nina_thermal.training.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)


class TestToolchainResolution:
    def test_converter_module_has_convert(self):
        assert hasattr(qat._converter_module(), "convert")

    def test_pt2e_api_returns_three_callables(self):
        prepare, prepare_qat, convert = qat._pt2e_api()
        assert all(callable(fn) for fn in (prepare, prepare_qat, convert))

    def test_build_quantizer_is_configured(self):
        q = qat.build_quantizer(per_channel=True)
        assert q is not None


class TestSampleInput:
    def test_shape_is_native_geometry(self):
        x = qat.sample_input()
        assert x.shape == (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert x.min() >= 0.0 and x.max() <= 1.0


class TestNormalizedFireNet:
    def test_folds_imagenet_normalization(self):
        backbone = train_thermal.build_model(pretrained=False).eval()
        wrapped = qat.NormalizedFireNet(backbone).eval()
        x = torch.full((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), 0.5)
        mean = torch.tensor(train_thermal._IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(train_thermal._IMAGENET_STD).view(1, 3, 1, 1)
        with torch.no_grad():
            assert torch.allclose(wrapped(x), backbone((x - mean) / std), atol=1e-5)

    def test_output_is_single_logit(self):
        wrapped = qat.NormalizedFireNet(train_thermal.build_model(pretrained=False))
        assert wrapped(qat.sample_input()).shape == (1, 1)


class TestBuildModelForQat:
    def test_loads_checkpoint_and_unfreezes(self, tmp_path):
        ckpt = tmp_path / "mobilenetv3s_fire_best.pt"
        torch.save(train_thermal.build_model(pretrained=False).state_dict(), ckpt)
        model = qat.build_model_for_qat(ckpt, torch.device("cpu"))
        assert isinstance(model, qat.NormalizedFireNet)
        assert all(p.requires_grad for p in model.parameters())


class TestSetMode:
    def test_plain_module_toggles(self):
        m = torch.nn.Linear(2, 1)
        qat.set_mode(m, training=True)
        assert m.training
        qat.set_mode(m, training=False)
        assert not m.training


class TestCalibrate:
    def _loader(self, multiroot, n=16):
        records, _ = load_merged_records(multiroot.roots)
        samples = split_samples(expand_to_samples(records, ["B"], ["fixed"]))["train"][
            :n
        ]
        ds = ThermalDataset(samples, train_thermal.build_transforms(augment=False))
        return DataLoader(ds, batch_size=4)

    def test_runs_without_grad(self, multiroot):
        model = qat.NormalizedFireNet(
            train_thermal.build_model(pretrained=False)
        ).eval()
        qat.calibrate(model, self._loader(multiroot), torch.device("cpu"), batches=2)

    def test_empty_loader_is_not_fatal(self):
        model = qat.NormalizedFireNet(
            train_thermal.build_model(pretrained=False)
        ).eval()
        qat.calibrate(model, DataLoader([]), torch.device("cpu"), batches=4)


class TestQatCaptureAndConvert:
    """The load-bearing claim of PR #7's quantization work: PT2E QAT runs."""

    def test_prepare_calibrate_step_convert(self, multiroot):
        train_thermal.set_seed(0)
        backbone = train_thermal.build_model(pretrained=False)
        for p in backbone.parameters():
            p.requires_grad = True
        model = qat.NormalizedFireNet(backbone).eval()
        example = qat.sample_input()

        prepared, quantizer = qat.prepare_for_qat(model, example)
        assert quantizer is not None

        records, _ = load_merged_records(multiroot.roots)
        samples = split_samples(expand_to_samples(records, ["B"], ["fixed"]))["train"][
            :8
        ]
        loader = DataLoader(
            ThermalDataset(samples, train_thermal.build_transforms(augment=False)),
            batch_size=4,
        )
        qat.calibrate(prepared, loader, torch.device("cpu"), batches=1)

        qat.set_mode(prepared, training=True)
        opt = torch.optim.SGD(
            [p for p in prepared.parameters() if p.requires_grad], lr=1e-4
        )
        crit = torch.nn.BCEWithLogitsLoss()
        images, labels = next(iter(loader))
        opt.zero_grad()
        loss = crit(prepared(images), labels.unsqueeze(1))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)

        converted = qat.convert(prepared)
        with torch.no_grad():
            out = converted(example)
        assert out.shape == (1, 1) and torch.isfinite(out).all()

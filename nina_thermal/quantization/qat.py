"""Quantization-aware training for the fire classifier — fp32 → int8 TFLite.

Mirrors ``nina_cloud/quantization/qat.py`` for the MobileNetV3-Small thermal
backbone.  The design decision carried over unchanged is
:class:`NormalizedFireNet`: ImageNet normalization is baked into the model's
forward pass rather than applied in the data transform, so the exported graph
consumes raw uint8 pixels directly.

On the OpenMV N6 the Lepton hands MicroPython an 8-bit grayscale frame.  The
TFLite input tensor is uint8 with ``scale ≈ 1/255, zero_point = 0``, so the
runtime dequantizes to float [0, 1] and the in-graph normalization converts to
the range the ImageNet-pretrained stem expects — no host-side preprocessing::

    on-device:  uint8 [0,255] --dequant(1/255, 0)--> float [0,1]
                              --NormLayer(mean,std)--> float [-2.1, 2.6]
                              --> backbone --> int8 logit

NOTE: the backbone built here must stay in sync with
``nina_thermal/training/train_thermal.py::build_model``.  The checkpoint is a
bare state dict, so an architecture mismatch surfaces as a load error rather
than as silent misbehaviour.

MobileNetV3 caveat
------------------
Unlike MobileNetV2, this backbone contains ``hardswish`` activations and
squeeze-excite blocks.  Both are representable in TFLite, but SE blocks carry
per-channel multiplies whose dynamic range quantizes less gracefully than plain
convolutions.  The PTQ smoke test exists to surface conversion failures in
seconds rather than after a full QAT fine-tune, and the parity check in
``run_quantize`` is what decides whether the int8 artifact is acceptable.
"""

import hashlib
import importlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nina_thermal.training.dataset import IMAGE_HEIGHT, IMAGE_WIDTH
from nina_thermal.training.train_thermal import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    build_model,
)

logger = logging.getLogger(__name__)


# ── Wrapped model ─────────────────────────────────────────────────────────────


class NormalizedFireNet(nn.Module):
    """Fire classifier with ImageNet normalization folded into the graph.

    Accepts float input in [0, 1] — i.e. plain ``ToTensor()`` output with no
    ``transforms.Normalize`` — and normalizes internally.  Any data pipeline
    feeding this wrapper must therefore omit the external Normalize step, or
    normalization is applied twice.

    Args:
        backbone: MobileNetV3-Small with the single-logit fire head.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize then classify.

        Args:
            x: Float tensor in [0, 1], shape ``(B, 3, H, W)``.

        Returns:
            Unnormalized logits, shape ``(B, 1)``.  Sigmoid is applied by the
            caller (in MicroPython: ``1 / (1 + exp(-logit))``).
        """
        return self.backbone((x - self.mean) / self.std)


def build_model_for_qat(
    checkpoint_path: Path, device: torch.device
) -> NormalizedFireNet:
    """Load the trained fp32 checkpoint and wrap it for QAT.

    The backbone is left fully trainable: quantization noise injected in the
    early convolutions cannot be compensated by the head alone.

    Args:
        checkpoint_path: Path to ``mobilenetv3s_fire_best.pt`` (bare state dict).
        device: Target device.

    Returns:
        Wrapped model in ``eval()`` mode on ``device``.
    """
    backbone = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone.load_state_dict(state)
    for param in backbone.parameters():
        param.requires_grad = True

    model = NormalizedFireNet(backbone).to(device).eval()
    logger.info(
        "QAT model loaded from %s — %s params",
        checkpoint_path.name,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


def sample_input(device: torch.device | None = None) -> torch.Tensor:
    """Return a representative input tensor at the native Lepton geometry.

    Args:
        device: Optional device for the tensor.

    Returns:
        Zero-filled tensor ``(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)`` in [0, 1].
    """
    tensor = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    return tensor if device is None else tensor.to(device)


# ── PT2E preparation ──────────────────────────────────────────────────────────


def _converter_module():
    """Return the PyTorch→TFLite converter module.

    ``ai-edge-torch`` was renamed to ``litert-torch``; the installed
    ``ai_edge_torch`` 0.7.x is a deprecation shim with no functionality, so the
    new name is preferred and the old one kept only for older environments.

    Returns:
        The imported converter module exposing ``convert``.

    Raises:
        ImportError: If neither package provides a working converter.
    """
    try:
        import litert_torch

        return litert_torch
    except ImportError:
        pass
    try:
        import ai_edge_torch

        if hasattr(ai_edge_torch, "convert"):
            return ai_edge_torch
    except ImportError:
        pass
    raise ImportError(
        "A PyTorch→TFLite converter is required.  Install with: "
        "pip install litert-torch   (ai-edge-torch is deprecated and the "
        "0.7.x wheel is an empty shim)."
    )


def _pt2e_api():
    """Return ``(prepare_pt2e, prepare_qat_pt2e, convert_pt2e)``.

    PyTorch 2.12 moved the PT2E entry points out of
    ``torch.ao.quantization.quantize_pt2e`` into ``torchao``.  Both locations
    are probed so the module works either side of that move.

    Returns:
        The three PT2E functions.

    Raises:
        ImportError: If neither location provides them.
    """
    for module_name in (
        "torchao.quantization.pt2e.quantize_pt2e",
        "torch.ao.quantization.quantize_pt2e",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return (
            module.prepare_pt2e,
            module.prepare_qat_pt2e,
            module.convert_pt2e,
        )
    raise ImportError(
        "PT2E quantization API not found in torchao or torch.ao.  "
        "Install with: pip install torchao"
    )


def build_quantizer(per_channel: bool = True):
    """Build the TFLite PT2E quantizer.

    Args:
        per_channel: Per-output-channel symmetric int8 weights when True,
            per-tensor when False.  Per-channel is more accurate; per-tensor
            produces a slightly smaller artifact.

    Returns:
        A configured PT2E quantizer.

    Raises:
        ImportError: If no PT2E quantizer implementation is installed.
    """
    for module_name, class_name in (
        ("litert_torch.quantize.pt2e_quantizer", "PT2EQuantizer"),
        ("ai_edge_torch.quantize.pt2e_quantizer", "TFLiteQuantizer"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        quantizer = getattr(module, class_name)()
        quantizer.set_global(
            module.get_symmetric_quantization_config(is_per_channel=per_channel)
        )
        return quantizer
    raise ImportError(
        "No PT2E quantizer found.  Install with: pip install litert-torch"
    )


def set_mode(model: nn.Module, training: bool) -> nn.Module:
    """Switch a model between train and eval mode, exported-graph safe.

    PT2E-exported ``GraphModule`` objects reject ``.train()`` / ``.eval()`` and
    require ``move_exported_model_to_{train,eval}`` so that dropout and
    BatchNorm nodes are rewritten in the graph.  Plain modules still take the
    ordinary path.

    Args:
        model: A plain ``nn.Module`` or a PT2E-exported ``GraphModule``.
        training: True for train mode, False for eval.

    Returns:
        ``model``, for chaining.
    """
    try:
        from torchao.quantization.pt2e import (
            move_exported_model_to_eval,
            move_exported_model_to_train,
        )

        if getattr(model, "_is_exported_program", False) or hasattr(model, "graph"):
            (move_exported_model_to_train if training else move_exported_model_to_eval)(
                model
            )
            return model
    except ImportError:
        pass
    model.train(training)
    return model


def _export_graph(
    model: NormalizedFireNet, example: torch.Tensor, dynamic_batch: bool = False
):
    """Capture ``model`` as a training-mode FX GraphModule for PT2E.

    The PT2E entry points take a ``GraphModule``, not the ``ExportedProgram``
    that ``torch.export.export`` returns, so ``.module()`` is required.
    ``export_for_training`` is preferred where available because it preserves
    BatchNorm as a separate node — QAT needs to observe and fold it, which a
    pre-folded inference graph makes impossible.

    ``dynamic_batch`` is required whenever the captured graph will later see
    real mini-batches: export otherwise specialises the batch dimension to the
    example's size and every subsequent call raises
    ``Guard failed: x.size()[0] == 1``.

    Args:
        model: Wrapped model to capture.
        example: Representative input ``(1, 3, H, W)`` in [0, 1].
        dynamic_batch: Leave dimension 0 unspecialised.

    Returns:
        An FX ``GraphModule`` ready for ``prepare_pt2e`` / ``prepare_qat_pt2e``.
    """
    import torch.export as te

    exporter = getattr(te, "export_for_training", te.export)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: te.Dim("batch", min=1, max=8192)},)
    return exporter(model.eval(), (example,), dynamic_shapes=dynamic_shapes).module()


def prepare_for_ptq(
    model: NormalizedFireNet, example: torch.Tensor, per_channel: bool = True
) -> tuple[nn.Module, object]:
    """Export and prepare the model for post-training static quantization.

    Args:
        model: Wrapped model in ``eval()`` mode.
        example: Representative input ``(1, 3, H, W)`` in [0, 1].
        per_channel: Weight quantization granularity.

    Returns:
        ``(prepared_model, quantizer)``.  The quantizer must be handed to
        :func:`export_tflite`, which needs it to emit the quantized graph.
    """
    prepare_pt2e, _, _ = _pt2e_api()
    quantizer = build_quantizer(per_channel)
    return prepare_pt2e(_export_graph(model, example), quantizer), quantizer


def prepare_for_qat(
    model: NormalizedFireNet, example: torch.Tensor, per_channel: bool = True
) -> tuple[nn.Module, object]:
    """Export and prepare the model for quantization-aware training.

    Inserts fake-quant nodes using the TFLite scheme: per-output-channel
    symmetric int8 weights, per-tensor asymmetric uint8 activations.

    Args:
        model: Wrapped model in ``eval()`` mode, all params trainable.
        example: Representative input ``(1, 3, H, W)`` in [0, 1].
        per_channel: Weight quantization granularity.

    Returns:
        ``(prepared_model, quantizer)``.  Switch the model to train mode with
        :func:`set_mode` before fine-tuning.
    """
    _, prepare_qat_pt2e, _ = _pt2e_api()
    quantizer = build_quantizer(per_channel)
    # torch.export specialises any dimension whose example size is 1, so a
    # batch-1 example cannot produce a dynamic batch dim ("Constraints
    # violated (batch)").  Capture with batch 2; the range still allows 1.
    capture_example = example.expand(2, *example.shape[1:]).contiguous()
    graph = _export_graph(model, capture_example, dynamic_batch=True)
    return prepare_qat_pt2e(graph, quantizer), quantizer


def calibrate(
    model: nn.Module, loader, device: torch.device, batches: int = 16
) -> None:
    """Warm up activation observers on real data before QAT fine-tuning.

    Starting the fine-tune with uninitialised observer ranges lets the first
    optimizer steps see wildly wrong quantization scales, which can move the
    weights somewhere the observers then have to chase.

    Args:
        model: Prepared (observer-inserted) model.
        loader: Loader yielding ``(images, labels)`` with images in [0, 1].
        device: Compute device.
        batches: Number of batches to run.
    """
    set_mode(model, training=False)
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= batches:
                break
            model(images.to(device))
    logger.info("Calibrated observers on %d batch(es).", min(batches, len(loader)))


def convert(model: nn.Module) -> nn.Module:
    """Convert a prepared model from fake-quant to true int8 operations.

    ``fold_quantize`` is forced off.  With folding on, weights are baked into
    plain int8 tensors that the LiteRT converter then rejects
    ("operand #0 must be ranked tensor of ... quantized values, but got
    tensor<16x3x3x3xi8>").  Leaving the quantize nodes in the graph preserves
    the quantization parameters the converter needs.

    Args:
        model: Prepared PTQ or QAT model.

    Returns:
        Converted model in eval mode.
    """
    _, _, convert_pt2e = _pt2e_api()

    return set_mode(convert_pt2e(model, fold_quantize=False), training=False)


# ── TFLite export ─────────────────────────────────────────────────────────────


def export_tflite(
    model: nn.Module,
    example: torch.Tensor,
    out_path: Path,
    quantizer: object | None = None,
) -> Path:
    """Convert a converted PT2E model to an int8 TFLite file.

    Args:
        model: Model returned by :func:`convert`.
        example: Representative input ``(1, 3, H, W)`` used for shape inference.
        out_path: Destination ``.tflite`` path.
        quantizer: The quantizer used to prepare the model.  Required for an
            int8 artifact — without it the converter emits a float graph.

    Returns:
        The written path.

    Raises:
        ImportError: If no converter package is installed.
    """
    converter = _converter_module()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The converter traces through numpy and cannot see CUDA tensors, so the
    # graph and its example input are moved to host memory first.
    model = set_mode(model, training=False).to("cpu")
    kwargs = {}
    if quantizer is not None:
        from litert_torch.quantize.quant_config import QuantConfig

        kwargs["quant_config"] = QuantConfig(pt2e_quantizer=quantizer)
    converter.convert(model, (example.cpu(),), **kwargs).export(str(out_path))
    logger.info(
        "TFLite written: %s (%.3f MB)",
        out_path,
        out_path.stat().st_size / 1024 / 1024,
    )
    return out_path


def write_checksum(tflite_path: Path) -> Path:
    """Write a ``sha256sum``-format checksum beside the TFLite artifact.

    Args:
        tflite_path: Path to the ``.tflite`` file.

    Returns:
        Path to the ``.sha256`` file.
    """
    digest = hashlib.sha256(tflite_path.read_bytes()).hexdigest()
    checksum_path = tflite_path.with_suffix(".sha256")
    checksum_path.write_text(f"{digest}  {tflite_path.name}\n", encoding="utf-8")
    return checksum_path


# ── Host-side TFLite inference ────────────────────────────────────────────────


def tflite_scores(tflite_path: Path, loader, limit: int | None = None) -> list[float]:
    """Run the exported TFLite model on host and return sigmoid scores.

    Used to measure fp32-vs-int8 parity: the quantized model is only acceptable
    if its ranking of fire above no-fire survives the conversion.

    Args:
        tflite_path: Path to the int8 ``.tflite`` file.
        loader: Loader yielding ``(images, labels)`` with images in [0, 1].
        limit: Optional cap on the number of samples scored.

    Returns:
        Sigmoid scores in dataset order.

    Raises:
        ImportError: If ``ai-edge-litert`` is unavailable.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise ImportError(
            "ai-edge-litert is required for host-side TFLite inference.  "
            "Install with: pip install ai-edge-litert"
        ) from exc

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    scores: list[float] = []
    for images, _ in loader:
        for image in images:
            if limit is not None and len(scores) >= limit:
                return scores
            array = image.unsqueeze(0).numpy()
            scale, zero_point = in_detail["quantization"]
            if in_detail["dtype"] in (np.uint8, np.int8):
                array = np.round(array / scale + zero_point).astype(in_detail["dtype"])
            else:
                array = array.astype(in_detail["dtype"])
            interpreter.set_tensor(in_detail["index"], array)
            interpreter.invoke()
            raw = interpreter.get_tensor(out_detail["index"]).astype(np.float32)
            out_scale, out_zero = out_detail["quantization"]
            if out_scale:
                raw = (raw - out_zero) * out_scale
            scores.append(float(1.0 / (1.0 + np.exp(-raw.reshape(-1)[0]))))
    return scores


def describe_tflite_io(tflite_path: Path) -> dict:
    """Report the exported artifact's actual input/output tensor types.

    The deployment contract depends on whether the graph really consumes uint8
    (NPU-resident) or float32 (a dequantize runs on the Cortex-M55 first), so
    this is read back from the flatbuffer rather than assumed.

    Args:
        tflite_path: Path to the ``.tflite`` file.

    Returns:
        Input/output dtype, shape and quantization parameters, plus a
        ``fully_quantized`` flag that is True only when both ends are integer.
    """
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    def describe(detail: dict) -> dict:
        scale, zero_point = detail["quantization"]
        return {
            "dtype": detail["dtype"].__name__,
            "shape": [int(v) for v in detail["shape"]],
            "scale": float(scale),
            "zero_point": int(zero_point),
        }

    described_in = describe(in_detail)
    described_out = describe(out_detail)
    return {
        "input": described_in,
        "output": described_out,
        "fully_quantized": described_in["dtype"] in ("uint8", "int8")
        and described_out["dtype"] in ("uint8", "int8"),
    }

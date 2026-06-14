"""Core QAT module: model wrapping, PT2E preparation, calibration, fine-tuning.

This module centralises all PyTorch-side quantization logic.  The key design
decision is ``NormalizedMobileNetV2``: rather than applying ImageNet mean/std
normalization in the data transform (as in training/evaluate), normalization is
baked into the model's forward pass.  When the graph is exported to int8 TFLite
the runtime dequantizes the uint8 camera feed (scale=1/255, zero_point=0) to
float [0,1] and the in-graph subtraction/division converts it to the ImageNet-
normalized range the backbone expects — so raw uint8 camera output feeds the
OpenMV model with zero host-side preprocessing.

NOTE: The backbone architecture defined in ``NormalizedMobileNetV2`` must stay
in sync with ``nina_cloud/training/train_cloud.py`` (MobileNetV2 IMAGENET1K_V1,
frozen in training but fully unfrozen here for QAT) and
``nina_cloud/evaluation/evaluate.py::build_model``.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── Constants — must match train_cloud.py ─────────────────────────────────────
IMAGE_SIZE: int = 224
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


# ── Wrapped model ─────────────────────────────────────────────────────────────


class NormalizedMobileNetV2(nn.Module):
    """MobileNetV2 with ImageNet normalization baked into the forward graph.

    Accepts float input in [0, 1] (standard ``ToTensor()`` output) and applies
    ImageNet mean/std normalization internally before the backbone.  When the
    exported int8 TFLite is loaded by the OpenMV ``ml.Model``, the Neural-ART
    NPU dequantizes the uint8 camera feed (scale≈1/255, zero_point=0) to float
    [0, 1] and then executes the normalization ops — so the MicroPython inference
    script feeds raw uint8 pixels with no preprocessing::

        img = sensor.snapshot()
        scores = model.predict([img])[0].flatten()

    This is equivalent to the external ``transforms.Normalize`` applied during
    training; only the location (transform vs. graph) differs.

    Args:
        backbone: Fully constructed MobileNetV2 with the cloud classifier head
            (``Dropout(0.2) → Linear(1280, 1)``).
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with internal ImageNet normalization.

        Args:
            x: Float tensor in [0, 1], shape ``(B, 3, H, W)``.  Do NOT apply
               ``transforms.Normalize`` in the data pipeline when using this
               wrapper — normalization is handled here.

        Returns:
            Unnormalized logit tensor, shape ``(B, 1)``.  Apply sigmoid
            externally (in MicroPython: ``score = 1 / (1 + exp(-logit))``).
        """
        x = (x - self.mean) / self.std
        return self.backbone(x)


# ── Model construction ────────────────────────────────────────────────────────


def build_model_for_qat(
    checkpoint_path: Path,
    device: torch.device,
) -> "NormalizedMobileNetV2":
    """Load the float32 checkpoint and wrap it for QAT.

    Unlike ``evaluate.build_model``, this function **unfreezes the backbone**
    so that fake-quant gradients flow through all convolutional layers.  QAT
    requires the backbone to be trainable because quantization noise in early
    convolutional layers cannot be corrected by adjusting only the head.

    The returned model is in ``eval()`` mode.  Call ``model.train()`` before
    starting the QAT fine-tune loop.

    Args:
        checkpoint_path: Path to ``mobilenetv2_cloud_best.pt`` (bare state_dict,
            as written by ``train_cloud.py``).
        device: Target device.

    Returns:
        ``NormalizedMobileNetV2`` with all parameters trainable, in ``eval()``
        mode, moved to ``device``.
    """
    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(backbone.last_channel, 1),
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone.load_state_dict(state)

    # Unfreeze backbone for QAT — all layers need gradients so fake-quant can
    # compensate quantization error in convolutional weights.
    for p in backbone.parameters():
        p.requires_grad = True

    model = NormalizedMobileNetV2(backbone).to(device)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "QAT model loaded: %s  trainable=%d / total=%d params",
        checkpoint_path.name,
        trainable,
        total,
    )
    return model


# ── Data transforms ───────────────────────────────────────────────────────────


def qat_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmented transform for QAT training — NO external Normalize step.

    Mirrors the training augmentation from ``train_cloud.py`` but omits
    ``transforms.Normalize`` because ``NormalizedMobileNetV2`` applies
    ImageNet normalization internally.

    Args:
        image_size: Spatial dimension for resizing (default: 224).

    Returns:
        Composed torchvision transform suitable for QAT training.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            # NOTE: No Normalize — NormalizedMobileNetV2 handles it.
        ]
    )


def qat_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic eval/calibration transform — NO external Normalize step.

    Args:
        image_size: Spatial dimension for resizing (default: 224).

    Returns:
        Deterministic torchvision transform for validation / calibration.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # NOTE: No Normalize — NormalizedMobileNetV2 handles it.
        ]
    )


# ── PT2E preparation ──────────────────────────────────────────────────────────


def prepare_for_ptq(
    model: "NormalizedMobileNetV2",
    sample_input: torch.Tensor,
) -> nn.Module:
    """Export and prepare model for post-training static quantization (smoke test).

    This is used only as a fast sanity check: if the graph can be traced and
    converted via PT2E PTQ, it will almost certainly survive QAT.  The PTQ
    artifact itself is not used for deployment — QAT produces the final model.

    Args:
        model: ``NormalizedMobileNetV2`` in ``eval()`` mode.
        sample_input: Representative input tensor ``(1, 3, H, W)`` in [0, 1].

    Returns:
        PTQ-prepared model (``nn.Module`` with observer nodes inserted).

    Raises:
        ImportError: If ``ai-edge-torch`` is not installed.
        torch.fx.proxy.TraceError: If the graph contains unsupported ops.
    """
    try:
        from ai_edge_torch.quantize.pt2e_quantizer import (
            TFLiteQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError as exc:
        raise ImportError(
            "ai-edge-torch is required for PTQ/QAT.  "
            "Install with: pip install ai-edge-torch"
        ) from exc

    from torch.ao.quantization.quantize_pt2e import prepare_pt2e
    from torch.export import export as torch_export

    model = model.eval()
    exported = torch_export(model, (sample_input,))
    quantizer = TFLiteQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(exported, quantizer)
    logger.debug("PTQ preparation complete (smoke test).")
    return prepared


def prepare_for_qat(
    model: "NormalizedMobileNetV2",
    sample_input: torch.Tensor,
) -> nn.Module:
    """Export and prepare model for QAT using the PT2E TFLite quantizer.

    Inserts fake-quant nodes throughout the graph using the TFLite quantization
    scheme: per-output-channel symmetric int8 for weights and per-tensor
    asymmetric uint8 for activations.  The returned model is a regular
    ``nn.Module`` and can be trained with a standard optimizer.

    Call ``calibrate_observers`` before training to warm up observer min/max
    statistics, then call ``model.train()`` and run the fine-tune loop.

    Args:
        model: ``NormalizedMobileNetV2`` in ``eval()`` mode, all params trainable.
        sample_input: Representative input tensor ``(1, 3, H, W)`` in [0, 1].

    Returns:
        QAT-prepared model (``nn.Module``) with fake-quant nodes, in ``eval()``
        mode.  Switch to ``train()`` before the fine-tune loop.

    Raises:
        ImportError: If ``ai-edge-torch`` is not installed.
    """
    try:
        from ai_edge_torch.quantize.pt2e_quantizer import (
            TFLiteQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError as exc:
        raise ImportError(
            "ai-edge-torch is required for PTQ/QAT.  "
            "Install with: pip install ai-edge-torch"
        ) from exc

    from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
    from torch.export import export as torch_export

    model = model.eval()
    exported = torch_export(model, (sample_input,))
    quantizer = TFLiteQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
    )
    prepared = prepare_qat_pt2e(exported, quantizer)
    logger.info("QAT preparation complete — fake-quant nodes inserted.")
    return prepared


# ── Calibration ───────────────────────────────────────────────────────────────


def calibrate_observers(
    model_prepared: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> None:
    """Warm up quantization observers before enabling fake-quant.

    Runs ``n_batches`` calibration batches through the prepared model (no
    gradients) so that per-tensor activation observers accumulate accurate
    min/max statistics before QAT fine-tuning starts.  Without this step,
    initial fake-quant scales may be poorly set and the first training epoch
    suffers unnecessarily large gradient variance.

    Args:
        model_prepared: Model returned by ``prepare_for_qat``.
        dataloader: DataLoader yielding ``(images, labels)`` where images are
            float [0, 1] (from ``qat_eval_transform``).
        device: Target device.
        n_batches: Number of batches to run for calibration.
    """
    model_prepared.eval()
    n_seen = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model_prepared(imgs.to(device))
            n_seen += imgs.size(0)
    logger.info(
        "Observer calibration complete — %d samples (%d batches).",
        n_seen,
        min(i + 1, n_batches),
    )


# ── QAT fine-tuning ───────────────────────────────────────────────────────────


def qat_fine_tune(
    model_prepared: nn.Module,
    train_records: list,
    val_records: list,
    image_dir: Path,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 16,
    pos_weight: float = 3.0,
    seed: int = 42,
) -> nn.Module:
    """QAT fine-tuning loop with asymmetric BCE loss.

    Fine-tunes the fake-quant model using the same asymmetric loss as the
    original training run.  A very low learning rate (1e-5 default) prevents
    catastrophic forgetting of the float32 weights while allowing fake-quant
    nodes to adjust scales and compensate for quantization error.

    The backbone is fully unfrozen (``build_model_for_qat`` ensures this) so
    that fake-quant gradients flow through all convolutional layers — essential
    for recovering accuracy in early layers where quantization noise accumulates.

    Args:
        model_prepared: QAT-prepared model from ``prepare_for_qat``.
        train_records: Training manifest records (``split=="train"``).
        val_records: Validation manifest records (``split=="val"``).
        image_dir: Directory containing PNG images.
        device: Training device.
        epochs: Number of QAT fine-tune epochs (5 is typically sufficient).
        lr: Learning rate — much smaller than original LR (1e-3) to avoid
            disrupting the float32 operating point.
        batch_size: DataLoader batch size.
        pos_weight: FN:FP cost ratio — must match the original ``pos_weight``
            from training to preserve the asymmetric decision boundary.
        seed: RNG seed for reproducible augmentation.

    Returns:
        ``model_prepared`` after in-place fine-tuning (also returned for chaining).
    """
    from nina_cloud.training.train_cloud import CloudDataset

    torch.manual_seed(seed)

    train_loader = DataLoader(
        CloudDataset(train_records, image_dir, qat_train_transform()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        CloudDataset(val_records, image_dir, qat_eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    beta = math.sqrt(pos_weight)
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
    logger.info(
        "QAT fine-tune — epochs=%d  lr=%.2e  pos_weight=%.1f  beta=%.3f",
        epochs,
        lr,
        pos_weight,
        beta,
    )

    for epoch in range(1, epochs + 1):
        # ── Training epoch ────────────────────────────────────────────────────
        model_prepared.train()
        total_loss = 0.0
        n = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model_prepared(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = total_loss / max(n, 1)

        # ── Validation loss ───────────────────────────────────────────────────
        model_prepared.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model_prepared(imgs).squeeze(1)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                n_val += imgs.size(0)
        val_loss /= max(n_val, 1)

        logger.info(
            "QAT epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

    return model_prepared


# ── Conversion ────────────────────────────────────────────────────────────────


def convert_qat_model(model_prepared: nn.Module) -> nn.Module:
    """Convert fake-quant QAT model to a quantized int8 model.

    Freezes fake-quant observers and replaces fake-quant nodes with true
    integer quantize/dequantize operations, producing a model whose weights
    are quantized to int8 and whose activations are represented as uint8.

    Args:
        model_prepared: Fine-tuned model from ``qat_fine_tune``.

    Returns:
        Quantized model ready for ``export_tflite.export_tflite``.
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e

    model_prepared.eval()
    quantized = convert_pt2e(model_prepared)
    logger.info("PT2E conversion complete — model is now fully int8.")
    return quantized


# ── Val-score collection (for threshold re-sweep) ─────────────────────────────


def collect_val_scores_fp32(
    model: nn.Module,
    val_records: list,
    image_dir: Path,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[list[int], list[float]]:
    """Run fp32 model inference on val records and return (y_true, y_score).

    Used for the fp32 baseline in the parity comparison and for re-sweeping
    the operating threshold on the float32 model as a reference.

    Args:
        model: Float32 model (``evaluate.build_model`` or ``NormalizedMobileNetV2``).
        val_records: Manifest records with ``label`` key.
        image_dir: Directory containing PNG images.
        device: Inference device.
        batch_size: DataLoader batch size.

    Returns:
        ``(y_true, y_score)`` — ground-truth labels and sigmoid scores.
    """
    from nina_cloud.training.train_cloud import CloudDataset

    # NormalizedMobileNetV2 uses qat_eval_transform (no external normalize);
    # standard build_model uses eval_transform (with normalize).
    # Detect which wrapper is in use by checking for the 'mean' buffer.
    if hasattr(model, "mean"):
        tf = qat_eval_transform()
    else:
        from nina_cloud.evaluation.evaluate import eval_transform

        tf = eval_transform()

    loader = DataLoader(
        CloudDataset(val_records, image_dir, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    y_true: list[int] = []
    y_score: list[float] = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            y_score.extend(logits.sigmoid().cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_score

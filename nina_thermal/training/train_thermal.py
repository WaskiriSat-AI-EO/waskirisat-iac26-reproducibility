"""MobileNetV3-Small transfer-learning baseline for fire/no-fire classification.

This is the ``nina_thermal`` counterpart of
``nina_cloud/training/train_cloud.py``.  It consumes the balanced manifests
produced by the thermal datagen pipeline across one or more acquisition-window
roots, trains a single-logit binary classifier at the native Lepton resolution,
sweeps the decision threshold on the validation set under a fire-recall floor,
and writes a ``training_report.json`` carrying full provenance.

Deliberate departures from ``train_cloud.py``
----------------------------------------------
**Backbone — MobileNetV3-Small, not MobileNetV2.**  The deployment target is a
32 MB QSPI flash on the OpenMV N6, but the mission budget calls for a model
under 1 MB.  Measured parameter counts with a single-logit head at 160×120:
MobileNetV2 α=1.0 is 2,225,153 params (~2.12 MB int8); MobileNetV3-Small is
927,585 (~0.88 MB int8).  MobileNetV3-Small is the smallest torchvision
backbone that still ships ImageNet weights — the α<1.0 MobileNetV2 variants do
not, and training a 400k-parameter network from scratch on ~640 images is not
viable.

**No photometric augmentation.**  ``train_cloud.py`` applies ``ColorJitter``.
Here it would be actively harmful: under the flight ``fixed`` AGC a pixel value
*is* an absolute temperature, quantised at ``FIXED_TEMP_RANGE_K / 255 ≈
0.392 K`` across 258.15–358.15 K.  Perturbing brightness or contrast severs
that mapping and teaches the model that the radiometric scale is arbitrary —
precisely the cue that separates a fire from warm bare rock.  Only geometric
augmentation (flips, small translations) is applied, and the degradation
profiles already supply the sensor-noise axis.

**Two-phase fine-tuning.**  ``train_cloud.py`` trains the head over a frozen
backbone.  Thermal grayscale at 3.5 km GSD sits far outside the ImageNet
manifold, so frozen features underfit.  Phase 1 trains the head alone to
convergence; phase 2 unfreezes the backbone at a much lower learning rate.

**Checkpoint selection on val ROC-AUC, not accuracy or F-beta.**  Under an
asymmetric cost the accuracy-optimal model is not the mission-optimal one, but
F-beta at a recall-floor operating point is not usable either: with ~50 fire
images in the val split and a 0.95 recall floor, the degenerate "predict fire
everywhere" model scores a competitive F-beta, so F-beta selection actively
rewards a useless classifier.  ROC-AUC measures whether the model *ranks* fire
above no-fire at all, is invariant to prevalence, and cannot be gamed that way.
The operating point is then swept once, on the selected checkpoint.
``--select_metric`` exposes ``pr_auc`` and ``fbeta`` for comparison.

Usage::

    # Inspect the merge and split reconciliation without training:
    python -m nina_thermal.training.train_thermal --audit

    python -m nina_thermal.training.train_thermal \\
        --roots data data_202409 data_202509 \\
        --epochs 20 --finetune_epochs 15 --pos_weight 3.0 --recall_target 0.95
"""

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_thermal.datagen import config as datagen_config
from nina_thermal.training.dataset import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ThermalDataset,
    describe_samples,
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.frp_labels import (
    DEFAULT_FRP_THRESHOLD_MW,
    apply_frp_labels,
)
from nina_thermal.training.threshold import (
    average_precision,
    metrics_at_threshold,
    precision_at_recall,
    roc_auc,
    select_threshold,
)

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ROOTS = ["data", "data_202408", "data_202409", "data_202509"]

# Profile B is the nominal mid-FFC-cycle condition and the deployment
# assumption.  A (post-FFC best case) and C (SAA-transit worst case) are held
# out entirely so evaluation can bound robustness in both directions.
TRAIN_PROFILES = ["B"]

# 'fixed' is the flight AGC configuration (datagen_config.FLIGHT_AGC_MODE).
# The 'global' variants are a per-granule adaptive stretch the satellite will
# never produce, and their metadata is stamped flight_config=false.
TRAIN_AGC_MODES = [datagen_config.FLIGHT_AGC_MODE]

HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 15
BATCH_SIZE = 32
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
POS_WEIGHT = 3.0
RECALL_TARGET = 0.95
PATIENCE = 10
SEED = datagen_config.DEFAULT_SEED

SELECT_METRIC = "roc_auc"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

ARCHITECTURE = "mobilenet_v3_small"
CHECKPOINT_NAME = "mobilenetv3s_fire_best.pt"


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Seed every RNG that influences training.

    Args:
        seed: Global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.

    Without this, augmentation draws depend on worker scheduling order and runs
    are not reproducible.

    Args:
        worker_id: Worker index supplied by the DataLoader.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Small with a single-logit binary head.

    The stock classifier (``Linear(576,1024) → Hardswish → Dropout →
    Linear(1024,1000)``) is replaced wholesale by ``Dropout → Linear(576,1)``.
    Dropping the 1024-unit bottleneck removes ~1.6M parameters — the single
    change that brings the int8 artifact under 1 MB.

    NOTE: ``nina_thermal/evaluation`` and ``nina_thermal/quantization`` must
    rebuild this exact architecture to load the checkpoint.  The architecture
    name is recorded in ``training_report.json`` for that purpose.

    Args:
        pretrained: Load ImageNet weights.  False is for tests only.

    Returns:
        Model with all parameters trainable; freeze via :func:`set_backbone_frozen`.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 1),
    )
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze the feature extractor.

    Args:
        model: Model from :func:`build_model`.
        frozen: True to disable gradients on ``features``.
    """
    for param in model.features.parameters():
        param.requires_grad = not frozen


# ── Transforms ────────────────────────────────────────────────────────────────


def build_transforms(augment: bool):
    """Build the preprocessing pipeline for one split.

    No resize is performed — the exported PNGs are already 160×120, the native
    Lepton geometry.  ``Grayscale(3)`` replicates the single AGC channel so the
    RGB-pretrained stem accepts the input without altering its content.

    Args:
        augment: Apply geometric augmentation (train split only).

    Returns:
        A composed torchvision transform.
    """
    steps = []
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Sub-tile translation only: fires are small and off-centre in
            # flight, and a fire near the edge must not be croppable away.
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        ]
    steps += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ── Train / eval loops ────────────────────────────────────────────────────────


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> float:
    """Run one pass over a loader.

    Args:
        model: Model to run.
        loader: Data loader for the split.
        criterion: Loss function.
        optimizer: Required when ``training`` is True, ignored otherwise.
        device: Compute device.
        training: Whether to backpropagate.

    Returns:
        Mean loss over the split.
    """
    model.train(training)
    total_loss = 0.0
    seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
    return total_loss / seen


def collect_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[float]]:
    """Run inference and return ``(y_true, y_score)`` with sigmoid applied.

    Args:
        model: Model to evaluate.
        loader: Loader over an unshuffled split.
        device: Compute device.

    Returns:
        Parallel lists of integer labels and float scores in [0, 1].
    """
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True)).squeeze(1)
            y_score.extend(logits.sigmoid().cpu().tolist())
            y_true.extend(int(v) for v in labels.tolist())
    return y_true, y_score


def _train_phase(
    *,
    phase: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    recall_target: float,
    beta: float,
    checkpoint_path: Path,
    select_metric: str,
    best_score: float,
    history: list[dict],
) -> float:
    """Run one training phase, checkpointing on validation F-beta.

    Args:
        phase: Phase label recorded in the history ("head" or "finetune").
        model: Model to train.
        train_loader: Shuffled training loader.
        val_loader: Unshuffled validation loader.
        criterion: Loss function.
        optimizer: Optimizer over the currently trainable parameters.
        scheduler: LR scheduler stepped once per epoch.
        device: Compute device.
        epochs: Maximum epochs for this phase.
        patience: Stop after this many epochs without F-beta improvement.
        recall_target: Fire-recall floor for the per-epoch threshold sweep.
        beta: F-beta weighting.
        checkpoint_path: Destination for the best state dict.
        select_metric: Validation metric driving checkpointing — one of
            ``"roc_auc"``, ``"pr_auc"`` or ``"fbeta"``.
        best_score: Best selection score carried in from previous phases.
        history: Mutable list receiving one dict per epoch.

    Returns:
        The best validation selection score after this phase.
    """
    logger.info(
        "%-8s %-10s %-10s %-8s %-8s %-8s %-8s",
        "Epoch",
        "TrainLoss",
        "ValLoss",
        "Recall",
        "Prec",
        "F-beta",
        "ROC-AUC",
    )
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss = run_epoch(model, val_loader, criterion, None, device, training=False)
        scheduler.step()

        y_true, y_score = collect_scores(model, val_loader, device)
        threshold, degenerate = select_threshold(y_true, y_score, recall_target)
        point = metrics_at_threshold(y_true, y_score, threshold, beta)
        auc = roc_auc(y_true, y_score)
        prauc = average_precision(y_true, y_score)
        score = {
            "roc_auc": auc,
            "pr_auc": prauc,
            "fbeta": point["fbeta"],
        }[select_metric]
        # A single-class val batch yields None; treat it as no improvement
        # rather than crashing mid-run.
        score = -1.0 if score is None else float(score)

        improved = score > best_score
        if improved:
            best_score = score
            torch.save(model.state_dict(), checkpoint_path)
            stale = 0
        else:
            stale += 1

        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "lr": round(optimizer.param_groups[0]["lr"], 10),
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_threshold": round(threshold, 6),
                "val_recall": point["recall"],
                "val_precision": point["precision"],
                "val_fbeta": point["fbeta"],
                "val_roc_auc": auc,
                "val_pr_auc": prauc,
                "val_select_score": round(score, 6),
                "degenerate_val": degenerate,
                "checkpointed": improved,
            }
        )
        logger.info(
            "%-8d %-10.4f %-10.4f %-8.3f %-8.3f %-8.3f %-8.3f%s",
            epoch,
            train_loss,
            val_loss,
            point["recall"],
            point["precision"],
            point["fbeta"],
            auc if auc is not None else float("nan"),
            "  <- best" if improved else "",
        )

        if stale >= patience:
            logger.info(
                "Early stop: %d epochs without %s improvement.",
                patience,
                select_metric,
            )
            break

    return best_score


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV3-Small fire/no-fire classifier"
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help=(
            "Data roots or 04_dataset dirs to merge.  Duplicate patches are "
            f"dropped and splits reconciled conservatively.  (default: {DEFAULT_ROOTS})"
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=TRAIN_PROFILES,
        help=(
            "Degradation profiles to train on.  Default B only (nominal "
            "mid-FFC condition); A and C stay held out as best/worst-case "
            f"robustness bounds for evaluation.  (default: {TRAIN_PROFILES})"
        ),
    )
    parser.add_argument(
        "--agc_modes",
        nargs="+",
        default=TRAIN_AGC_MODES,
        help=(
            "AGC modes to train on.  Default is the flight configuration only; "
            "'global' is an adaptive stretch not produced on orbit.  "
            f"(default: {TRAIN_AGC_MODES})"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="models/nina_thermal",
        help="Directory for the checkpoint and training report.",
    )
    parser.add_argument(
        "--epochs", type=int, default=HEAD_EPOCHS, help="Phase-1 (head) epochs."
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=FINETUNE_EPOCHS,
        help="Phase-2 (unfrozen backbone) epochs.  0 disables phase 2.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=HEAD_LR, help="Phase-1 learning rate."
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=FINETUNE_LR,
        help="Phase-2 learning rate; must be far lower than --lr.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=POS_WEIGHT,
        help=(
            "Weight on the fire-class loss term.  Encodes the FN:FP mission "
            "cost ratio — a missed fire forfeits the early-warning window, a "
            "false alarm costs one downlink slot.  This is only incidentally "
            "an imbalance correction (train is roughly 58%% no-fire).  "
            "beta = sqrt(pos_weight) is derived for F-beta."
        ),
    )
    parser.add_argument(
        "--recall_target",
        type=float,
        default=RECALL_TARGET,
        help=(
            "Minimum fire recall for the val threshold sweep.  Kept below the "
            "cloud model's 0.99 because the val split holds only ~50 fire "
            "images, where 0.99 demands a perfect score."
        ),
    )
    parser.add_argument(
        "--select_metric",
        choices=["roc_auc", "pr_auc", "fbeta"],
        default=SELECT_METRIC,
        help=(
            "Validation metric for checkpoint selection.  Default roc_auc: "
            "threshold-independent and not gameable by predicting fire "
            "everywhere, unlike fbeta on a small val split."
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=PATIENCE, help="Early-stopping patience."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Global RNG seed.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument(
        "--log_dir", default="data/logs", help="Directory for the run log."
    )
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=None,
        help=(
            "Relabel from FIRMS as 'large fire complex in view': positive when "
            "summed FRP over the 640x480 km patch footprint >= this many MW, "
            "negative only when the footprint holds zero detections, ambiguous "
            "patches dropped.  The datagen labels ask whether ANY fire is "
            "present, which is not answerable at 4 km GSD.  Suggested value: "
            f"{DEFAULT_FRP_THRESHOLD_MW:g}.  Omit to keep the datagen labels."
        ),
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Report the merge, reconciliation and split composition, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "train_thermal",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )

    set_seed(args.seed)

    # ── Assemble the dataset ──────────────────────────────────────────────────
    records, merge_stats = load_merged_records(args.roots)
    samples = expand_to_samples(records, args.profiles, args.agc_modes)

    frp_summary = None
    if args.frp_threshold is not None:
        samples, frp_summary = apply_frp_labels(samples, args.roots, args.frp_threshold)

    by_split = split_samples(samples)
    composition = {
        split: describe_samples(rows) for split, rows in sorted(by_split.items())
    }

    logger.info("Profiles=%s  AGC modes=%s", args.profiles, args.agc_modes)
    for split, stats in composition.items():
        logger.info(
            "  %-5s %4d images  %4d patches  %3d granules  fire=%d no_fire=%d",
            split,
            stats["images"],
            stats["patches"],
            stats["granules"],
            stats["fire"],
            stats["no_fire"],
        )

    if args.audit:
        logger.info("Merge stats: %s", json.dumps(merge_stats, indent=2))
        if frp_summary is not None:
            logger.info("FRP relabel: %s", json.dumps(frp_summary, indent=2))
        logger.info("Audit only — no training performed.")
        return log_run_footer(start_time, exit_code=0)

    train_samples = by_split.get("train", [])
    val_samples = by_split.get("val", [])
    if not train_samples:
        logger.error("No training samples.  Run the datagen split stage first.")
        return log_run_footer(start_time, exit_code=1)
    if not val_samples:
        logger.error(
            "No validation samples — checkpoint selection and the threshold "
            "sweep both require a val split.  Run run_split for every root."
        )
        return log_run_footer(start_time, exit_code=1)

    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        ThermalDataset(train_samples, build_transforms(augment=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        ThermalDataset(val_samples, build_transforms(augment=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    beta = math.sqrt(args.pos_weight)
    logger.info("Device: %s", device)
    logger.info(
        "Architecture: %s  input=%dx%d (WxH)  params=%s  int8 lower bound=%.2f MB",
        ARCHITECTURE,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        f"{total_params:,}",
        total_params / 1024 / 1024,
    )
    logger.info(
        "pos_weight=%.2f  beta=%.3f  recall_target=%.2f",
        args.pos_weight,
        beta,
        args.recall_target,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME

    history: list[dict] = []
    best_score = -1.0

    # ── Phase 1: head only ────────────────────────────────────────────────────
    logger.info("── Phase 1: frozen backbone, head only (lr=%g) ──", args.lr)
    set_backbone_frozen(model, True)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    best_score = _train_phase(
        phase="head",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5),
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        recall_target=args.recall_target,
        beta=beta,
        checkpoint_path=checkpoint_path,
        select_metric=args.select_metric,
        best_score=best_score,
        history=history,
    )

    # ── Phase 2: full fine-tune from the best head checkpoint ─────────────────
    if args.finetune_epochs > 0:
        logger.info("── Phase 2: unfrozen backbone (lr=%g) ──", args.finetune_lr)
        if checkpoint_path.is_file():
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True)
            )
        set_backbone_frozen(model, False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        best_score = _train_phase(
            phase="finetune",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.finetune_epochs, 1)
            ),
            device=device,
            epochs=args.finetune_epochs,
            patience=args.patience,
            recall_target=args.recall_target,
            beta=beta,
            checkpoint_path=checkpoint_path,
            select_metric=args.select_metric,
            best_score=best_score,
            history=history,
        )

    if not checkpoint_path.is_file():
        logger.error(
            "No checkpoint was written — no epoch produced a usable %s.",
            args.select_metric,
        )
        return log_run_footer(start_time, exit_code=1)

    # ── Final sweep on the best checkpoint ────────────────────────────────────
    logger.info(
        "Best val %s: %.4f  → %s", args.select_metric, best_score, checkpoint_path
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    y_true, y_score = collect_scores(model, val_loader, device)
    threshold, degenerate = select_threshold(y_true, y_score, args.recall_target)
    val_metrics = metrics_at_threshold(y_true, y_score, threshold, beta)

    logger.info(
        "Selected threshold %.4f — val recall=%.3f precision=%.3f specificity=%.3f",
        threshold,
        val_metrics["recall"],
        val_metrics["precision"],
        val_metrics["specificity"],
    )
    if degenerate:
        logger.warning(
            "Degenerate val sweep — threshold fell back to 0.5 and is NOT a "
            "validated operating point."
        )

    report = {
        "stage": "train_thermal",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": ARCHITECTURE,
        "input_size": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "channels": 3},
        "normalization": {"mean": _IMAGENET_MEAN, "std": _IMAGENET_STD},
        "parameters": total_params,
        "hyperparameters": {
            "epochs": args.epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "finetune_lr": args.finetune_lr,
            "pos_weight": args.pos_weight,
            "beta": round(beta, 6),
            "recall_target": args.recall_target,
            "patience": args.patience,
            "select_metric": args.select_metric,
            "seed": args.seed,
        },
        "data": {
            "roots": [str(r) for r in args.roots],
            "profiles": args.profiles,
            "agc_modes": args.agc_modes,
            "merge": merge_stats,
            "frp_relabel": frp_summary,
            "composition": composition,
        },
        "checkpoint": checkpoint_path.name,
        "selected_threshold": threshold,
        "degenerate_val": degenerate,
        "best_val_select_score": best_score,
        "val_metrics_at_threshold": val_metrics,
        "val_roc_auc": roc_auc(y_true, y_score),
        "val_average_precision": average_precision(y_true, y_score),
        "val_fire_prevalence": round(sum(y_true) / len(y_true), 6),
        "precision_at_recall": precision_at_recall(y_true, y_score, [0.90, 0.95, 0.99]),
        "history": history,
        "notes": (
            "Trained on profile B / fixed AGC only; profiles A and C are held "
            "out as out-of-distribution robustness bounds.  pos_weight encodes "
            "the FN:FP mission cost ratio.  selected_threshold is the highest "
            "val threshold reaching recall >= recall_target.  Splits are "
            "reconciled across roots with test > val > train precedence, so a "
            "granule held out in any root is held out everywhere."
        ),
    }
    report_path = out_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Training report → %s", report_path)

    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    raise SystemExit(main())

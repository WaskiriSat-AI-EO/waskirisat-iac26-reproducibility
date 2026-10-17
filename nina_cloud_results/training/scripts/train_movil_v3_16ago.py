"""
Entrenamiento robusto MobileNetV2 — CLEAR vs CLOUD

Política:
TRAIN -> B
VAL   -> B
TEST  -> A+B+C (NO se usa durante entrenamiento ni calibración)

Mejoras:
- Data augmentation SOLO en TRAIN.
- Transfer learning + fine-tuning parcial.
- BatchNormalization congelada durante fine-tuning.
- Salida binaria sigmoid = P(CLOUD).
- Selección de checkpoint por VAL PR-AUC.
- Threshold calibrado SOLO con VAL B.
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

DEFAULT_SEED = 42
IMG_SIZE = 224
NUM_CHANNELS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS_HEAD = 10
DEFAULT_EPOCHS_FINETUNE = 8
DEFAULT_LR_HEAD = 1e-4
DEFAULT_LR_FINETUNE = 1e-5
DEFAULT_UNFREEZE_LAST = 30
DEFAULT_RECALL_TARGET = 0.95
DEFAULT_CLOUD_WEIGHT = 1.0
CLASS_NAMES = ["clear", "cloud"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MobileNetV2_Robusto_Cloud_Clear")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_pngs(directory: Path):
    return (
        len(list((directory / "clear").glob("*.png"))),
        len(list((directory / "cloud").glob("*.png"))),
    )


def load_final_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe final_manifest.json:\n{path}")
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("final_manifest.json debe contener una lista.")
    return records


def verify_dataset_policy(records):
    summary = {"train": {}, "val": {}, "test": {}}
    scenes = {"train": set(), "val": set(), "test": set()}

    for r in records:
        split = str(r.get("split", "")).lower()
        if split not in summary:
            continue
        profile = str(r.get("profile", "")).upper()
        summary[split][profile] = summary[split].get(profile, 0) + 1
        scenes[split].add(r.get("scene_id"))

    if set(summary["train"]) != {"B"}:
        raise RuntimeError(f"TRAIN debe contener solo perfil B: {summary['train']}")
    if set(summary["val"]) != {"B"}:
        raise RuntimeError(f"VAL debe contener solo perfil B: {summary['val']}")
    if set(summary["test"]) != {"A", "B", "C"}:
        raise RuntimeError(f"TEST debe contener A+B+C: {summary['test']}")

    if scenes["train"] & scenes["val"]:
        raise RuntimeError("Scene leakage TRAIN/VAL")
    if scenes["train"] & scenes["test"]:
        raise RuntimeError("Scene leakage TRAIN/TEST")
    if scenes["val"] & scenes["test"]:
        raise RuntimeError("Scene leakage VAL/TEST")

    return summary, scenes


def create_data_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
            layers.RandomRotation(0.03, fill_mode="reflect", name="random_rotation"),
            layers.RandomBrightness(0.10, value_range=(0.0, 255.0), name="random_brightness"),
            layers.RandomContrast(0.10, name="random_contrast"),
        ],
        name="train_augmentation",
    )


def build_directory_dataset(directory, batch_size, shuffle, seed):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"No existe:\n{directory}")

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    ds = ds.map(
        lambda image, label: (tf.cast(image, tf.float32), tf.cast(label, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


















def create_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS),
        include_top=False,
        weights="imagenet",
        alpha=0.85,
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    base_model.trainable = False

    augmentation = create_data_augmentation()
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="image")
    x = augmentation(inputs)
    
    
    
   # x = layers.Lambda(preprocess_input, name="mobilenet_preprocess")(x)
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocess",
    )(x)
        
    
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.20, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="cloud_probability")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Robusto_Cloud_Clear")
    return model, base_model


def metric_list():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
    ]


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metric_list(),
    )


def enable_partial_finetuning(base_model, unfreeze_last):
    base_model.trainable = True
    total_layers = len(base_model.layers)
    boundary = max(0, total_layers - min(unfreeze_last, total_layers))

    for i, layer in enumerate(base_model.layers):
        if i < boundary:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_layers = [l.name for l in base_model.layers if l.trainable]
    logger.info("Backbone fine-tuning: %d capas entrenables", len(trainable_layers))
    logger.info("Últimas entrenables: %s", trainable_layers[-10:])


def print_trainable_params(model, title):
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_trainable = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    logger.info("%s | trainable=%s | non_trainable=%s | total=%s",
                title, f"{trainable:,}", f"{non_trainable:,}", f"{trainable + non_trainable:,}")


def create_callbacks(checkpoint_path, csv_path, patience_early=5, patience_lr=2):
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(csv_path)),
    ]


def collect_probabilities(model, dataset):
    y_true, y_score = [], []
    for images, labels in dataset:
        prob = model.predict(images, verbose=0).reshape(-1)
        y_score.extend(prob.astype(float).tolist())
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision_cloud": float(precision),
        "recall_cloud": float(recall),
        "specificity_clear": float(specificity),
        "f1_cloud": float(f1),
        "confusion_matrix": {
            "TN_clear_correct": tn,
            "FP_clear_as_cloud": fp,
            "FN_cloud_as_clear": fn,
            "TP_cloud_correct": tp,
        },
    }


def select_threshold(y_true, y_score, recall_target):
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("VAL necesita CLEAR y CLOUD para calibrar threshold.")

    candidates = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    valid = []
    for thr in candidates:
        m = metrics_at_threshold(y_true, y_score, float(thr))
        if m["recall_cloud"] >= recall_target:
            valid.append(m)

    if not valid:
        raise RuntimeError(f"Ningún threshold cumple recall_target={recall_target}")

    # Primero mayor precision; en empate mayor F1; luego threshold más alto.
    valid.sort(
        key=lambda m: (m["precision_cloud"], m["f1_cloud"], m["threshold"]),
        reverse=True,
    )
    return valid[0]


def evaluate_checkpoint(path, val_dataset):
    model = tf.keras.models.load_model(path)
    result = model.evaluate(val_dataset, verbose=0, return_dict=True)
    return model, {k: float(v) for k, v in result.items()}


def merge_histories(h1, h2):
    merged = {}
    for key in set(h1.history) | set(h2.history):
        merged[key] = list(h1.history.get(key, [])) + list(h2.history.get(key, []))
    return merged


def plot_metric(history, train_key, val_key, title, ylabel, path, phase1_epochs):
    train_values = history.get(train_key, [])
    val_values = history.get(val_key, [])
    epochs = np.arange(1, len(train_values) + 1)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.78])
    ax.plot(epochs, train_values, marker="o", label=f"TRAIN {train_key}")
    ax.plot(epochs, val_values, marker="o", label=f"VAL {train_key}")
    if phase1_epochs:
        ax.axvline(phase1_epochs + 0.5, linestyle="--", alpha=0.6, label="Inicio fine-tuning")
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(y_true, y_score, selected, path):
    thresholds = np.linspace(0.0, 1.0, 201)
    precision, recall, f1 = [], [], []
    for thr in thresholds:
        m = metrics_at_threshold(y_true, y_score, thr)
        precision.append(m["precision_cloud"])
        recall.append(m["recall_cloud"])
        f1.append(m["f1_cloud"])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.78])
    ax.plot(thresholds, precision, label="Precision CLOUD")
    ax.plot(thresholds, recall, label="Recall CLOUD")
    ax.plot(thresholds, f1, label="F1 CLOUD")
    ax.axvline(selected["threshold"], linestyle="--",
               label=f"Threshold={selected['threshold']:.4f}")
    ax.set_title("Calibración de threshold — VALIDATION B")
    ax.set_xlabel("Threshold P(CLOUD)")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_val_predictions(y_true, y_score, threshold, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "cloud_probability", "predicted_label", "threshold"])
        for truth, score in zip(y_true, y_score):
            writer.writerow([int(truth), float(score), int(score >= threshold), float(threshold)])


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 robusto CLEAR vs CLOUD")
    parser.add_argument("--dataset", default="data/cloud/dataset_final_repo")
    parser.add_argument("--output", default="data/cloud/training_mobilenetv2_robusto_v2")
    parser.add_argument("--epochs-head", type=int, default=DEFAULT_EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int, default=DEFAULT_EPOCHS_FINETUNE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr-head", type=float, default=DEFAULT_LR_HEAD)
    parser.add_argument("--lr-finetune", type=float, default=DEFAULT_LR_FINETUNE)
    parser.add_argument("--unfreeze-last", type=int, default=DEFAULT_UNFREEZE_LAST)
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument("--cloud-weight", type=float, default=DEFAULT_CLOUD_WEIGHT,
                        help="1.0=sin ponderación adicional de CLOUD")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    manifest_path = dataset_root / "final_manifest.json"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"No existe:\n{d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_final_manifest(manifest_path)
    profile_summary, _ = verify_dataset_policy(records)

    logger.info("=" * 78)
    logger.info("POLÍTICA VERIFICADA")
    logger.info("TRAIN: %s", profile_summary["train"])
    logger.info("VAL:   %s", profile_summary["val"])
    logger.info("TEST:  %s", profile_summary["test"])
    logger.info("Scene leakage: 0")
    logger.info("TEST NO se usa en este script.")
    logger.info("=" * 78)

    train_clear, train_cloud = count_pngs(train_dir)
    val_clear, val_cloud = count_pngs(val_dir)
    logger.info("TRAIN B | clear=%d cloud=%d total=%d", train_clear, train_cloud, train_clear + train_cloud)
    logger.info("VAL B   | clear=%d cloud=%d total=%d", val_clear, val_cloud, val_clear + val_cloud)

    train_ds = build_directory_dataset(train_dir, args.batch_size, True, args.seed)
    val_ds = build_directory_dataset(val_dir, args.batch_size, False, args.seed)

    model, base_model = create_model()
    class_weight = {0: 1.0, 1: float(args.cloud_weight)}

    phase1_ckpt = output_dir / "phase1_head_best.keras"
    phase2_ckpt = output_dir / "phase2_finetune_best.keras"
    best_model_path = output_dir / "mobilenetv2_best.keras"
    report_path = output_dir / "training_report.json"
    phase1_csv = output_dir / "history_phase1.csv"
    phase2_csv = output_dir / "history_phase2.csv"

    # ---------------- FASE 1 ----------------
    logger.info("\n" + "=" * 78)
    logger.info("FASE 1 — TRANSFER LEARNING | backbone congelada")
    logger.info("=" * 78)
    compile_model(model, args.lr_head)
    print_trainable_params(model, "FASE 1")

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=create_callbacks(phase1_ckpt, phase1_csv),
        verbose=1,
    )

    # ---------------- FASE 2 ----------------
    logger.info("\n" + "=" * 78)
    logger.info("FASE 2 — FINE-TUNING PARCIAL")
    logger.info("=" * 78)

    model = tf.keras.models.load_model(phase1_ckpt)
    
    
    
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise RuntimeError("No se pudo localizar MobileNetV2 dentro del modelo.")

    enable_partial_finetuning(base_model, args.unfreeze_last)
    compile_model(model, args.lr_finetune)
    print_trainable_params(model, "FASE 2")

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weight,
        callbacks=create_callbacks(phase2_ckpt, phase2_csv),
        verbose=1,
    )

    # -------- Elegir mejor checkpoint por VAL PR-AUC --------
    m1, r1 = evaluate_checkpoint(phase1_ckpt, val_ds)
    m2, r2 = evaluate_checkpoint(phase2_ckpt, val_ds)

    logger.info("FASE 1 | VAL PR-AUC=%.6f ACC=%.6f REC=%.6f", r1["pr_auc"], r1["accuracy"], r1["recall"])
    logger.info("FASE 2 | VAL PR-AUC=%.6f ACC=%.6f REC=%.6f", r2["pr_auc"], r2["accuracy"], r2["recall"])

    if r2["pr_auc"] >= r1["pr_auc"]:
        selected_stage = "phase2_finetune"
        selected_model = m2
        selected_val_keras = r2
    else:
        selected_stage = "phase1_head"
        selected_model = m1
        selected_val_keras = r1

    selected_model.save(best_model_path)
    logger.info("Checkpoint seleccionado: %s", selected_stage)
    logger.info("Mejor modelo → %s", best_model_path)

    # -------- Threshold SOLO con VAL B --------
    y_true_val, y_score_val = collect_probabilities(selected_model, val_ds)
    selected_thr = select_threshold(y_true_val, y_score_val, args.recall_target)
    threshold = selected_thr["threshold"]

    logger.info("\n" + "=" * 78)
    logger.info("THRESHOLD CALIBRADO SOLO CON VALIDATION B")
    logger.info("Recall target: %.3f", args.recall_target)
    logger.info("Threshold: %.6f", threshold)
    logger.info("VAL ACC: %.2f%%", 100 * selected_thr["accuracy"])
    logger.info("VAL PREC CLOUD: %.2f%%", 100 * selected_thr["precision_cloud"])
    logger.info("VAL REC CLOUD: %.2f%%", 100 * selected_thr["recall_cloud"])
    logger.info("VAL SPEC CLEAR: %.2f%%", 100 * selected_thr["specificity_clear"])
    logger.info("VAL F1 CLOUD: %.2f%%", 100 * selected_thr["f1_cloud"])
    logger.info("VAL CM: %s", selected_thr["confusion_matrix"])
    logger.info("=" * 78)

    save_val_predictions(y_true_val, y_score_val, threshold, output_dir / "val_predictions.csv")

    merged = merge_histories(h1, h2)
    phase1_epochs = len(h1.history["loss"])
    plot_metric(merged, "accuracy", "val_accuracy", "Accuracy — TRAIN y VAL", "Accuracy",
                output_dir / "01_training_accuracy.png", phase1_epochs)
    plot_metric(merged, "loss", "val_loss", "Loss — TRAIN y VAL", "Loss",
                output_dir / "02_training_loss.png", phase1_epochs)
    plot_metric(merged, "pr_auc", "val_pr_auc", "PR-AUC — TRAIN y VAL", "PR-AUC",
                output_dir / "03_training_pr_auc.png", phase1_epochs)
    plot_threshold_curve(y_true_val, y_score_val, selected_thr,
                         output_dir / "04_val_threshold_calibration.png")

    report = {
        "model": "MobileNetV2",
        "weights": "ImageNet",
        "task": "clear_vs_cloud",
        "classes": {"0": "clear", "1": "cloud"},
        "dataset_policy": {
            "train": "B",
            "validation": "B",
            "test": "A+B+C",
            "test_used_during_training": False,
            "scene_leakage": 0,
        },
        "augmentation_train_only": {
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_factor": 0.03,
            "brightness_factor": 0.10,
            "contrast_factor": 0.10,
        },
        "phase1": {
            "epochs_requested": args.epochs_head,
            "epochs_executed": len(h1.history["loss"]),
            "learning_rate": args.lr_head,
            "backbone_frozen": True,
            "checkpoint": str(phase1_ckpt),
            "val_metrics_at_0_5": r1,
        },
        "phase2": {
            "epochs_requested": args.epochs_finetune,
            "epochs_executed": len(h2.history["loss"]),
            "learning_rate": args.lr_finetune,
            "unfreeze_last": args.unfreeze_last,
            "batchnorm_frozen": True,
            "checkpoint": str(phase2_ckpt),
            "val_metrics_at_0_5": r2,
        },
        "selected_checkpoint": {
            "stage": selected_stage,
            "path": str(best_model_path),
            "selection_metric": "validation PR-AUC",
            "val_metrics_at_0_5": selected_val_keras,
        },
        "threshold_calibration": {
            "source": "VALIDATION B only",
            "recall_target": args.recall_target,
            "selected_threshold": threshold,
            "metrics": selected_thr,
        },
        "training": {
            "batch_size": args.batch_size,
            "cloud_weight": args.cloud_weight,
            "seed": args.seed,
            "train_clear": train_clear,
            "train_cloud": train_cloud,
            "val_clear": val_clear,
            "val_cloud": val_cloud,
        },
        "note": "TEST A/B/C permanece aislado. Evaluarlo después con mobilenetv2_best.keras y selected_threshold.",
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 78)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("Mejor modelo: %s", best_model_path)
    logger.info("Reporte: %s", report_path)
    logger.info("Threshold final: %.6f", threshold)
    logger.info("TEST A/B/C NO fue utilizado.")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

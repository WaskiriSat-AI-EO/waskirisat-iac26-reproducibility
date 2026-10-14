#!/usr/bin/env python3
"""Validate the cloud and fire TFLite classifiers on a labelled PNG dataset.

Install: python -m pip install ai-edge-litert pillow numpy
Run:     python validate_tflite.py --root paper_tests
Compare: python validate_tflite.py --root paper_tests --openmv-results paper_tests/test_1
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter


TESTS = {
    "cloud_detection": {
        "classes": {"clear": 0, "cloud": 1},
        "shape": (1, 224, 224, 3),
        "layout": "NHWC",
        "input": "RGB uint8 values quantized with the tensor scale and zero point",
        "output": "probability",
    },
    "fire_detection": {
        "classes": {"no_fire": 0, "fire": 1},
        "shape": (1, 3, 120, 160),
        "layout": "NCHW",
        "input": "RGB float32 scaled to [0,1]; ImageNet normalization is inside the model",
        "output": "logit",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantization(detail):
    scale, zero = detail["quantization"]
    return float(scale), int(zero)


def tensor_contract(detail):
    scale, zero = quantization(detail)
    return {
        "name": detail["name"],
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "scale": scale,
        "zero_point": zero,
    }


def sigmoid(value):
    factor = math.exp(-abs(value))
    return 1.0 / (1.0 + factor) if value >= 0 else factor / (1.0 + factor)


def resize_rgb(path, width, height):
    with Image.open(path) as image:
        source_size = list(image.size)
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8), source_size


def prepare_input(path, config, detail):
    shape = config["shape"]
    height, width = (shape[1], shape[2]) if config["layout"] == "NHWC" else (shape[2], shape[3])
    rgb, source_size = resize_rgb(path, width, height)

    if config["layout"] == "NCHW":
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    else:
        scale, zero = quantization(detail)
        if scale <= 0:
            raise ValueError("The quantized cloud input has no valid scale")
        limits = np.iinfo(detail["dtype"])
        tensor = np.clip(np.rint(rgb.astype(np.float32) / scale + zero), limits.min, limits.max)
        tensor = tensor.astype(detail["dtype"])[None, ...]

    if tuple(tensor.shape) != tuple(detail["shape"]) or tensor.dtype != detail["dtype"]:
        raise ValueError("Prepared tensor does not match the model input contract")
    return tensor, source_size


def dequantize_output(tensor, detail):
    raw = tensor.reshape(-1)
    if raw.size != 1:
        raise ValueError("Expected one scalar output, received %d values" % raw.size)
    quantized = int(raw[0]) if np.issubdtype(raw.dtype, np.integer) else None
    scale, zero = quantization(detail)
    value = float((quantized - zero) * scale) if quantized is not None and scale > 0 else float(raw[0])
    return quantized, value


def auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    ok = [row for row in rows if row["status"] == "ok"]
    tp = sum(row["expected_id"] == 1 and row["prediction"]["predicted_id"] == 1 for row in ok)
    tn = sum(row["expected_id"] == 0 and row["prediction"]["predicted_id"] == 0 for row in ok)
    fp = sum(row["expected_id"] == 0 and row["prediction"]["predicted_id"] == 1 for row in ok)
    fn = sum(row["expected_id"] == 1 and row["prediction"]["predicted_id"] == 0 for row in ok)
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elapsed = [row["timing_ms"]["inference"] for row in ok]
    labels = [row["expected_id"] for row in ok]
    scores = [row["prediction"]["positive_probability"] for row in ok]
    return {
        "images_discovered": len(rows),
        "images_processed": len(ok),
        "image_errors": len(rows) - len(ok),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "accuracy": divide(tp + tn, len(ok)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": divide(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "mcc": divide(tp * tn - fp * fn, denominator),
            "roc_auc": auc(labels, scores),
        },
        "inference_ms": {
            "mean": float(np.mean(elapsed)) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "p95": float(np.percentile(elapsed, 95)) if elapsed else None,
            "min": min(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
    }


def compare_openmv(rows, path):
    if not path or not path.is_file():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    by_key = {
        (row["expected_class"], row["filename"]): row
        for row in reference.get("results", []) if row.get("status") == "ok"
    }
    pairs = [(row, by_key.get((row["expected_class"], row["filename"]))) for row in rows]
    pairs = [(desktop, board) for desktop, board in pairs if board is not None and desktop["status"] == "ok"]
    if not pairs:
        return {"reference_file": str(path), "matched_images": 0}
    differences = [
        abs(desktop["prediction"]["positive_probability"] - board["prediction"]["positive_probability"])
        for desktop, board in pairs
    ]
    agreements = sum(
        desktop["prediction"]["predicted_id"] == board["prediction"]["predicted_id"]
        for desktop, board in pairs
    )
    return {
        "reference_file": str(path),
        "matched_images": len(pairs),
        "decision_agreement": agreements / len(pairs),
        "probability_mae": float(np.mean(differences)),
        "probability_max_abs_error": max(differences),
    }


def evaluate(root, output_dir, openmv_dir, test_id, threshold):
    config = TESTS[test_id]
    model_path = root / "models" / (test_id + ".tflite")
    dataset_dir = root / test_id
    if not model_path.is_file() or not dataset_dir.is_dir():
        raise FileNotFoundError("Missing model or dataset for " + test_id)

    started = time.perf_counter()
    interpreter = Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    model_load_ms = (time.perf_counter() - started) * 1000
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(test_id + " must expose exactly one input and one output")
    input_detail, output_detail = inputs[0], outputs[0]
    if tuple(input_detail["shape"]) != config["shape"]:
        raise ValueError("Unexpected %s input shape: %s" % (test_id, input_detail["shape"]))

    items = []
    for class_name, class_id in config["classes"].items():
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError("Missing class directory: " + str(class_dir))
        items.extend((path, class_name, class_id) for path in sorted(class_dir.glob("*.png")))

    rows = []
    for index, (path, expected_class, expected_id) in enumerate(items, 1):
        row = {"index": index, "filename": path.name, "expected_class": expected_class, "expected_id": expected_id}
        try:
            start = time.perf_counter()
            tensor, source_size = prepare_input(path, config, input_detail)
            preprocessing_ms = (time.perf_counter() - start) * 1000
            interpreter.set_tensor(input_detail["index"], tensor)
            start = time.perf_counter()
            interpreter.invoke()
            inference_ms = (time.perf_counter() - start) * 1000
            quantized, value = dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
            probability = value if config["output"] == "probability" else sigmoid(value)
            probability = min(1.0, max(0.0, probability))
            predicted_id = int(probability >= threshold)
            predicted_class = next(name for name, identifier in config["classes"].items() if identifier == predicted_id)
            row.update({
                "status": "ok",
                "source_size": source_size,
                "prediction": {
                    "raw_quantized": quantized,
                    "raw_dequantized": value,
                    "positive_probability": probability,
                    "predicted_id": predicted_id,
                    "predicted_class": predicted_class,
                },
                "correct": predicted_id == expected_id,
                "timing_ms": {"preprocessing": preprocessing_ms, "inference": inference_ms},
            })
        except Exception as error:
            row.update({"status": "error", "error": "%s: %s" % (type(error).__name__, error)})
        rows.append(row)
        if row["status"] == "error" or index % 10 == 0 or index == len(items):
            print("%s: %d/%d %s" % (test_id, index, len(items), row["status"]))

    report = {
        "schema_version": 1,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__},
        "model": {
            "id": test_id,
            "path": str(model_path),
            "sha256": sha256(model_path),
            "load_ms": model_load_ms,
            "input": tensor_contract(input_detail) | {"layout": config["layout"], "preprocessing": config["input"]},
            "output": tensor_contract(output_detail) | {"interpretation": config["output"], "threshold": threshold},
        },
        "results": rows,
        "summary": metrics(rows),
    }
    reference_path = openmv_dir / (test_id + "_results.json") if openmv_dir else None
    comparison = compare_openmv(rows, reference_path)
    if comparison:
        report["openmv_comparison"] = comparison

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (test_id + "_python_results.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("paper_tests"), help="Directory containing models and labelled image folders")
    parser.add_argument("--output", type=Path, help="Output directory; default: ROOT/test_1")
    parser.add_argument("--openmv-results", type=Path, help="Optional directory containing the OpenMV JSON files")
    parser.add_argument("--model", choices=("all", *TESTS), default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    selected = TESTS if args.model == "all" else (args.model,)
    output_dir = args.output or args.root / "test_1"
    for test_id in selected:
        report = evaluate(args.root, output_dir, args.openmv_results, test_id, args.threshold)
        values = report["summary"]["metrics"]
        print("%s: accuracy=%.3f, F1=%.3f, ROC-AUC=%.3f" %
              (test_id, values["accuracy"], values["f1"], values["roc_auc"]))


if __name__ == "__main__":
    main()

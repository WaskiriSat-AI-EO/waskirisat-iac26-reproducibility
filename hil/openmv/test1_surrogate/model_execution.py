import binascii
import gc
import hashlib
import math
import os
import time
import ujson

import image
import ml
from ulab import numpy as np


ROOTS = ("/sdcard/paper_tests", "/sd/paper_tests", "/paper_tests", "paper_tests")
OUTPUT_DIR = "test_1"
THRESHOLD = 0.5
SCHEMA_VERSION = "6.3"

# Cloud runs first because its larger tensor arena requires the least
# fragmented memory. Fire then reuses one float32 input tensor for its batch.
RUN_ORDER = (1, 0)

TESTS = (
    {
        "id": "fire_detection",
        "model": "models/fire_detection.tflite",
        "model_size": 1452472,
        "classes": (("no_fire", 0), ("fire", 1)),
        "positive": "fire",
        "input_shape": (1, 3, 120, 160),
        "input_dtype": "f",
        "input_scale": 1.0,
        "input_zero": 0,
        "layout": "NCHW",
        "output_shape": (1, 1),
        "output_dtype": "b",
        "output_scale": 0.04537102,
        "output_zero": -5,
        "output_transform": "sigmoid",
        "manifest_sha256": "c6e1489a88323230a97e3273d21ac1c469a8a847bcd78adc07397f8d28781be5",
        "desktop_raw": (
            10, 15, -4, 4, 21, 12, -15, 1, 7, -42,
            -4, 15, -26, -21, -30, -24, -21, -13, -31, -38,
            -12, -40, -12, -17, -27, -38, -1, -5, -15, -48,
            -36, -19, -45, -29, -30, -41, -16, -5, 10, 23,
            -18, 21, 5, -35, -32, -43, -30, -1, -18, -35,
            2, 8, -10, -3, -3, 8, 4, 0, 12, -13,
            -30, 11, -11, 1, -10, 5, 3, -3, 4, -24,
            0, 0, 4, 0, -21, -6, 5, -23, -8, 0,
            -1, 5, -5, 5, -8, 4, -15, -7, -4, -5,
            -21, 2, 0, -1, -13, -1, 0, -3, -9, 4,
        ),
    },
    {
        "id": "cloud_detection",
        "model": "models/cloud_detection.tflite",
        "model_size": 1941072,
        "classes": (("clear", 0), ("cloud", 1)),
        "positive": "cloud",
        "input_shape": (1, 224, 224, 3),
        "input_dtype": "b",
        "input_scale": 1.0,
        "input_zero": -128,
        "layout": "NHWC",
        "output_shape": (1, 1),
        "output_dtype": "b",
        "output_scale": 0.00390625,
        "output_zero": -128,
        "output_transform": "probability",
        "manifest_sha256": "595211136ed731e4085fc60cd8db5f2fd14e96fb697acfa1d354f2ba36c4b553",
        "desktop_raw": (
            -128, -128, -128, -126, -8, -127, -128, -128, -128, -51,
            -88, -128, -107, -118, -125, -57, -125, -128, -128, -128,
            -128, -128, -128, -125, -127, -125, -121, -124, -118, -126,
            -44, 63, -121, -127, -126, -125, -127, -23, 74, -109,
            57, 8, -127, -121, -125, -63, 51, -102, -74, 8,
            127, 127, 127, -74, -23, 127, 127, 120, 127, 127,
            127, 127, 122, 44, 127, 127, 127, 122, 99, 127,
            127, 79, 126, 104, 127, 102, 109, 57, 127, 126,
            88, 37, -88, 0, -114, -15, 123, -104, 63, 109,
            -57, 109, 114, 121, 63, 15, 84, -37, -84, 37,
        ),
    },
)

STAGES = ("input_read", "image_convert", "tensor_prepare", "predict_total",
          "inference_estimated", "postprocess", "sample_total")


def join(left, right):
    return left + right if left.endswith("/") else left + "/" + right


def is_dir(path):
    try:
        os.listdir(path)
        return True
    except Exception:
        return False


def root_dir():
    for path in ROOTS:
        if is_dir(path):
            return path
    raise OSError("paper_tests directory not found")


def elapsed_us(start):
    return int(time.ticks_diff(time.ticks_us(), start))


def milliseconds(value):
    return float(value) / 1000.0


def dtype_code(value):
    return chr(value) if isinstance(value, int) else value


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / denominator


def sigmoid(value):
    factor = math.exp(-abs(value))
    return 1.0 / (1.0 + factor) if value >= 0 else factor / (1.0 + factor)


def scalar(tensor):
    values = tensor.flatten().tolist()
    if len(values) != 1:
        raise ValueError("Expected one scalar output")
    return float(values[0])


def class_name(config, class_id):
    for name, identifier in config["classes"]:
        if identifier == class_id:
            return name
    return "unknown"


def image_items(root, config):
    items = []
    manifest = hashlib.sha256()
    for expected_class, expected_id in config["classes"]:
        directory = join(join(root, config["id"]), expected_class)
        if not is_dir(directory):
            raise OSError("Missing image directory: " + directory)
        names = [name for name in os.listdir(directory)
                 if name.lower().endswith(".png")]
        names.sort()
        for filename in names:
            manifest.update((expected_class + "/" + filename + "\n").encode())
            items.append((join(directory, filename), filename,
                          expected_class, expected_id))
    manifest_hex = binascii.hexlify(manifest.digest()).decode()
    if len(items) != len(config["desktop_raw"]):
        raise ValueError("%s dataset contains %d images; expected %d" %
                         (config["id"], len(items), len(config["desktop_raw"])))
    if manifest_hex != config["manifest_sha256"]:
        raise ValueError(config["id"] + " dataset manifest mismatch")
    return items, manifest_hex


def check_model(model, config, load_ms):
    actual = (
        tuple(model.input_shape[0]), dtype_code(model.input_dtype[0]),
        float(model.input_scale[0]), int(model.input_zero_point[0]),
        tuple(model.output_shape[0]), dtype_code(model.output_dtype[0]),
        float(model.output_scale[0]), int(model.output_zero_point[0]),
    )
    expected = (
        config["input_shape"], config["input_dtype"], config["input_scale"],
        config["input_zero"], config["output_shape"], config["output_dtype"],
        config["output_scale"], config["output_zero"],
    )
    for index, (found, required) in enumerate(zip(actual, expected)):
        equal = abs(found - required) <= 1e-6 if index in (2, 6) else found == required
        if not equal:
            raise ValueError("Model contract mismatch at field %d: %s != %s" %
                             (index, found, required))
    return {
        "id": config["id"], "path": config["model"],
        "size_bytes": int(model.len),
        "arena_bytes": int(model.ram), "load_ms": load_ms,
        "input": {
            "shape": list(actual[0]), "dtype": actual[1],
            "scale": actual[2], "zero_point": actual[3],
            "layout": config["layout"],
        },
        "output": {
            "shape": list(actual[4]), "dtype": actual[5],
            "scale": actual[6], "zero_point": actual[7],
            "transform": config["output_transform"],
            "threshold": THRESHOLD, "positive_class": config["positive"],
        },
    }


def desktop_prediction(config, index):
    raw = config["desktop_raw"][index]
    value = (raw - config["output_zero"]) * config["output_scale"]
    probability = value if config["output_transform"] == "probability" else sigmoid(value)
    predicted_id = int(probability >= THRESHOLD)
    return raw, value, probability, predicted_id


class CloudImageInput:
    def __init__(self, frame, config):
        self.frame = frame
        self.config = config
        self.prepare_us = 0

    def __call__(self, buffer, shape, dtype):
        start = time.ticks_us()
        config = self.config
        if tuple(shape) != config["input_shape"] or dtype_code(dtype) != "b":
            raise ValueError("Unexpected Cloud input contract")
        tensor = self.frame.to_ndarray("b", buffer=buffer)
        if tuple(tensor.shape) != (224, 224, 3):
            raise ValueError("Unexpected Cloud tensor shape")
        tensor = None
        self.prepare_us = elapsed_us(start)


def load_frame(path, width, height, pixformat):
    gc.collect()
    start = time.ticks_us()
    source = image.Image(path, copy_to_fb=True)
    input_read_us = elapsed_us(start)
    source_size = [int(source.width()), int(source.height())]

    start = time.ticks_us()
    frame = image.Image(width, height, pixformat)
    frame.draw_image(
        source, 0, 0,
        x_scale=float(width) / source_size[0],
        y_scale=float(height) / source_size[1],
        hint=image.BILINEAR,
    )
    image_convert_us = elapsed_us(start)
    source = None
    gc.collect()
    return frame, source_size, input_read_us, image_convert_us


def load_fire_tensor(path, audit, tensor):
    frame, source_size, input_read_us, image_convert_us = load_frame(
        path, 160, 120, image.GRAYSCALE
    )

    start = time.ticks_us()
    gray = frame.to_ndarray("f")
    gray *= 1.0 / 255.0
    minimum = float(np.min(gray))
    maximum = float(np.max(gray))
    if not (minimum >= 0.0 and maximum <= 1.0 and minimum <= maximum):
        raise ValueError("Fire tensor values must be finite and within 0..1")
    for channel in range(3):
        for row in range(120):
            tensor[0, channel, row, :] = gray[row, :]
    if tuple(tensor.shape) != (1, 3, 120, 160):
        raise ValueError("Unexpected Fire tensor shape")

    audit_data = None
    if audit:
        samples = (0, 1, 159, 160, 9600, 19199)
        flat = gray.flatten()
        mean = float(np.mean(gray))
        sample_values = [float(flat[index]) for index in samples]
        audit_data = {
            "shape": [1, 3, 120, 160], "range": [minimum, maximum],
            "channel_mean": [mean, mean, mean],
            "channel_min": [minimum, minimum, minimum],
            "channel_max": [maximum, maximum, maximum],
            "sample_indexes": list(samples),
            "channel_samples": [sample_values, sample_values, sample_values],
        }
        flat = None
    tensor_prepare_us = elapsed_us(start)
    frame = gray = None
    gc.collect()
    return tensor, source_size, audit_data, input_read_us, image_convert_us, tensor_prepare_us


def predict(model, model_input, config):
    start = time.ticks_us()
    outputs = model.predict([model_input])
    predict_us = elapsed_us(start)
    if len(outputs) != 1:
        raise ValueError("Expected one output tensor")

    start = time.ticks_us()
    value = scalar(outputs[0])
    probability = value if config["output_transform"] == "probability" else sigmoid(value)
    probability = max(0.0, min(1.0, probability))
    raw = int(round(value / config["output_scale"])) + config["output_zero"]
    raw = max(-128, min(127, raw))
    predicted_id = int(probability >= THRESHOLD)
    result = {
        "raw_dequantized": value,
        "raw_quantized_estimate": raw,
        "positive_probability": probability,
        "predicted_id": predicted_id,
        "predicted_class": class_name(config, predicted_id),
    }
    postprocess_us = elapsed_us(start)
    outputs = None
    return result, predict_us, postprocess_us


def new_stats():
    return {
        "ok": 0, "errors": 0, "correct": 0,
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        "min_memory": None, "write_us": 0,
        "comparison": {
            "count": 0, "agreements": 0, "exact": 0,
            "sum_abs": 0.0, "sum_sq": 0.0, "sum_error": 0.0,
            "sum_x": 0.0, "sum_y": 0.0,
            "sum_xx": 0.0, "sum_yy": 0.0, "sum_xy": 0.0,
        },
        "times": {name: [0, None, 0] for name in STAGES},
    }


def update_stats(stats, expected_id, prediction, times, free_memory,
                 desktop_probability, desktop_raw):
    predicted_id = prediction["predicted_id"]
    stats["ok"] += 1
    stats["correct"] += int(expected_id == predicted_id)
    key = "tp" if expected_id == predicted_id == 1 else \
          "tn" if expected_id == predicted_id == 0 else \
          "fp" if expected_id == 0 else "fn"
    stats[key] += 1
    stats["min_memory"] = (free_memory if stats["min_memory"] is None
                           else min(stats["min_memory"], free_memory))
    for name, value in times.items():
        entry = stats["times"][name]
        entry[0] += value
        entry[1] = value if entry[1] is None else min(entry[1], value)
        entry[2] = max(entry[2], value)

    current = prediction["positive_probability"]
    error = current - desktop_probability
    comp = stats["comparison"]
    comp["count"] += 1
    comp["agreements"] += int(predicted_id == int(desktop_probability >= THRESHOLD))
    comp["exact"] += int(prediction["raw_quantized_estimate"] == desktop_raw)
    comp["sum_abs"] += abs(error)
    comp["sum_sq"] += error * error
    comp["sum_error"] += error
    comp["sum_x"] += current
    comp["sum_y"] += desktop_probability
    comp["sum_xx"] += current * current
    comp["sum_yy"] += desktop_probability * desktop_probability
    comp["sum_xy"] += current * desktop_probability


def comparison_summary(comp):
    count = comp["count"]
    left = count * comp["sum_xx"] - comp["sum_x"] * comp["sum_x"]
    right = count * comp["sum_yy"] - comp["sum_y"] * comp["sum_y"]
    denominator = math.sqrt(max(0.0, left * right))
    correlation = None if denominator == 0 else (
        count * comp["sum_xy"] - comp["sum_x"] * comp["sum_y"]
    ) / denominator
    return {
        "count": count,
        "exact_quantized_outputs": comp["exact"],
        "exact_output_rate": ratio(comp["exact"], count),
        "decision_agreement": ratio(comp["agreements"], count),
        "probability_mae": ratio(comp["sum_abs"], count),
        "probability_rmse": math.sqrt(comp["sum_sq"] / count) if count else None,
        "probability_bias_openmv_minus_desktop": ratio(comp["sum_error"], count),
        "probability_correlation": correlation,
    }


def make_summary(stats, discovered, batch_us, memory_before, memory_after,
                 input_source):
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    timing = {}
    for name, values in stats["times"].items():
        timing[name + "_ms"] = {
            "mean": milliseconds(values[0] / stats["ok"]) if stats["ok"] else None,
            "min": milliseconds(values[1]) if values[1] is not None else None,
            "max": milliseconds(values[2]) if stats["ok"] else None,
            "total": milliseconds(values[0]),
        }
    timing["result_records_write_ms"] = milliseconds(stats["write_us"])
    return {
        "input_source": input_source,
        "images_discovered": discovered, "images_processed": stats["ok"],
        "image_errors": stats["errors"], "correct": stats["correct"],
        "incorrect": stats["ok"] - stats["correct"],
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "accuracy": ratio(tp + tn, tp + tn + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
        },
        "desktop_comparison": comparison_summary(stats["comparison"]),
        "timing": timing, "batch_total_ms": milliseconds(batch_us),
        "memory": {
            "free_before_model_bytes": memory_before,
            "minimum_free_bytes": stats["min_memory"],
            "free_after_model_release_bytes": memory_after,
        },
    }


def write_record(stream, record, first, stats):
    start = time.ticks_us()
    if not first:
        stream.write(",")
    stream.write(ujson.dumps(record))
    stream.flush()
    stats["write_us"] += elapsed_us(start)


def validate_inputs(root, config, items):
    model_path = join(root, config["model"])
    model_size = int(os.stat(model_path)[6])
    if model_size != config["model_size"]:
        raise ValueError("%s model size mismatch: %d != %d" %
                         (config["id"], model_size, config["model_size"]))
    reference = {
        "dataset_manifest_sha256": config["manifest_sha256"],
        "desktop_runtime": "LiteRT 2.2.0",
    }
    if config["id"] == "cloud_detection":
        reference.update({
            "type": "openmv_rgb565_bilinear_png",
            "image_count": len(items),
            "preprocessing": "PNG to RGB565, explicit bilinear resize to 224x224, then int8 x-128",
        })
    else:
        reference.update({
            "type": "png_grayscale_images",
            "image_count": len(items),
            "preprocessing": "Grayscale values duplicated into RGB NCHW float32 and divided by 255",
        })
    return model_path, reference


def run_model(root, config):
    print("\nValidating:", config["id"])
    items, manifest = image_items(root, config)
    validation_start = time.ticks_us()
    model_path, input_reference = validate_inputs(root, config, items)
    input_reference["dataset_manifest_sha256"] = manifest
    input_reference["validation_ms"] = milliseconds(elapsed_us(validation_start))

    gc.collect()
    memory_before = int(gc.mem_free())
    print("Free memory before model:", memory_before)
    print("Loading:", model_path)
    start = time.ticks_us()
    model = ml.Model(model_path)
    model_info = check_model(model, config, milliseconds(elapsed_us(start)))

    output_dir = join(root, OUTPUT_DIR)
    final_path = join(output_dir, config["id"] + "_results.json")
    temp_path = final_path + ".tmp"
    try:
        os.remove(temp_path)
    except Exception:
        pass

    stats = new_stats()
    batch_start = time.ticks_us()
    fire_tensor = (np.empty((1, 3, 120, 160), dtype=np.float)
                   if config["id"] == "fire_detection" else None)
    output = open(temp_path, "w")
    output.write('{"schema_version":"' + SCHEMA_VERSION + '",')
    output.write('"purpose":"final_embedded_batch_evaluation","model":')
    output.write(ujson.dumps(model_info))
    output.write(',"input_reference":')
    output.write(ujson.dumps(input_reference))
    output.write(',"results":[')

    try:
        for index, item in enumerate(items):
            path, filename, expected_class, expected_id = item
            sample_start = time.ticks_us()
            model_input = None
            audit_data = None
            source_size = None
            stage = "input_preparation"
            try:
                if config["id"] == "cloud_detection":
                    (frame, source_size, input_read_us,
                     image_convert_us) = load_frame(path, 224, 224, image.RGB565)
                    model_input = CloudImageInput(frame, config)
                    stage = "inference"
                    prediction, predict_us, postprocess_us = predict(model, model_input, config)
                    tensor_prepare_us = model_input.prepare_us
                    inference_us = max(0, predict_us - tensor_prepare_us)
                else:
                    audit = index == 0 or index == len(items) // 2
                    (model_input, source_size, audit_data, input_read_us,
                     image_convert_us, tensor_prepare_us) = load_fire_tensor(
                         path, audit, fire_tensor
                     )
                    stage = "inference"
                    prediction, predict_us, postprocess_us = predict(model, model_input, config)
                    inference_us = predict_us

                desktop_raw, desktop_value, desktop_probability, desktop_id = \
                    desktop_prediction(config, index)
                times = {
                    "input_read": input_read_us,
                    "image_convert": image_convert_us,
                    "tensor_prepare": tensor_prepare_us,
                    "predict_total": predict_us,
                    "inference_estimated": inference_us,
                    "postprocess": postprocess_us,
                    "sample_total": elapsed_us(sample_start),
                }
                free_memory = int(gc.mem_free())
                update_stats(stats, expected_id, prediction, times, free_memory,
                             desktop_probability, desktop_raw)
                record = {
                    "index": index + 1, "status": "ok", "filename": filename,
                    "expected_class": expected_class, "expected_id": expected_id,
                    "prediction": prediction,
                    "desktop_reference": {
                        "raw_quantized": desktop_raw,
                        "raw_dequantized": desktop_value,
                        "positive_probability": desktop_probability,
                        "predicted_id": desktop_id,
                    },
                    "comparison": {
                        "raw_quantized_difference":
                            prediction["raw_quantized_estimate"] - desktop_raw,
                        "absolute_probability_error":
                            abs(prediction["positive_probability"] - desktop_probability),
                        "decision_matches_desktop": prediction["predicted_id"] == desktop_id,
                    },
                    "correct": expected_id == prediction["predicted_id"],
                    "timing_ms": {name: milliseconds(value)
                                  for name, value in times.items()},
                    "memory_free_bytes": free_memory,
                }
                record["image"] = path
                record["source_size"] = source_size
                if audit_data is not None:
                    record["input_tensor_stats"] = audit_data
            except Exception as error:
                stats["errors"] += 1
                record = {
                    "index": index + 1, "status": "error", "filename": filename,
                    "expected_class": expected_class, "expected_id": expected_id,
                    "error": repr(error), "error_stage": stage,
                    "elapsed_ms": milliseconds(elapsed_us(sample_start)),
                }
            finally:
                if config["id"] == "cloud_detection" and model_input is not None:
                    model_input.frame = None
                model_input = audit_data = None
                frame = None
                gc.collect()

            write_record(output, record, index == 0, stats)
            if record["status"] == "error" and stats["errors"] == 1:
                print(config["id"] + " first error:", record["error"])
            if index == 0 or (index + 1) % 10 == 0:
                print("%s: %d/%d %s" %
                      (config["id"], index + 1, len(items), record["status"]))

        model = fire_tensor = None
        gc.collect()
        final_summary = make_summary(
            stats, len(items), elapsed_us(batch_start), memory_before,
            int(gc.mem_free()), input_reference["type"]
        )
        output.write('],"summary":')
        output.write(ujson.dumps(final_summary))
        output.write("}\n")
        output.flush()
    finally:
        output.close()
        model = fire_tensor = None
        gc.collect()

    try:
        os.remove(final_path)
    except Exception:
        pass
    os.rename(temp_path, final_path)
    print("Saved:", final_path)
    print("Free memory after release:", gc.mem_free())


def main():
    print("Final OpenMV batch evaluation v6.3 starting")
    root = root_dir()
    output_dir = join(root, OUTPUT_DIR)
    if not is_dir(output_dir):
        os.mkdir(output_dir)
    print("paper_tests:", root)
    print("Initial free memory:", gc.mem_free())
    for index in RUN_ORDER:
        run_model(root, TESTS[index])
        gc.collect()
    print("\nCompleted. Free memory:", gc.mem_free())


if __name__ == "__main__":
    main()

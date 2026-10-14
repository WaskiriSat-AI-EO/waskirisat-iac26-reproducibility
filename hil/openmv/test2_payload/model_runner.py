"""Validated phase-1 preprocessing and transactional sample selection.

The Cloud and Fire paths intentionally mirror model_execution.py schema 6.3.
Do not replace them with a shared Normalization object: their tensor contracts
and output transforms are different.
"""

import gc
import math
import os
import time
import ujson

import image
import ml
from ulab import numpy as np

import config

_root = None
_active_name = None
_model_ctx = None
_catalogs = {}
_positions = {}
_reserved = None
_state_generation = 0


def _join(left, right):
    return left + right if left.endswith("/") else left + "/" + right


def _is_dir(path):
    try:
        os.listdir(path)
        return True
    except Exception:
        return False


def data_root():
    global _root
    if _root is None:
        for candidate in config.DATA_ROOT_CANDIDATES:
            if _is_dir(candidate):
                _root = candidate
                break
    if _root is None:
        raise OSError("PAPER_TESTS_ROOT_NOT_FOUND")
    return _root


def _mode_by_name(name):
    for mode in config.MODES:
        if mode["name"] == name:
            return mode
    raise ValueError("UNKNOWN_MODE:" + str(name))


def mode_names():
    return [mode["name"] for mode in config.MODES]


def active_mode():
    if _active_name is None:
        return config.MODES[0]
    return _mode_by_name(_active_name)


def _state_paths():
    base = _join(data_root(), config.STATE_FILENAME)
    return base + "_a.json", base + "_b.json"


def _load_state():
    global _state_generation
    if not config.PERSIST_PROGRESS:
        return
    best = None
    for path in _state_paths():
        try:
            with open(path, "r") as stream:
                state = ujson.loads(stream.read())
            generation = int(state.get("generation", -1))
            if isinstance(state.get("positions"), dict) and (
                    best is None or generation > best[0]):
                best = (generation, state["positions"])
        except Exception:
            pass
    if best is not None:
        _state_generation = max(0, best[0])
        for mode in config.MODES:
            value = int(best[1].get(mode["name"], 0))
            _positions[mode["name"]] = max(0, value)


def _save_state():
    global _state_generation
    if not config.PERSIST_PROGRESS:
        return
    next_generation = _state_generation + 1
    paths = _state_paths()
    final_path = paths[next_generation & 1]
    temp_path = final_path + ".tmp"
    state = {
        "schema_version": config.SCHEMA_VERSION,
        "generation": next_generation,
        "positions": _positions,
    }
    with open(temp_path, "w") as stream:
        stream.write(ujson.dumps(state))
        stream.flush()
    try:
        os.remove(final_path)
    except Exception:
        pass
    os.rename(temp_path, final_path)
    try:
        os.sync()
    except Exception:
        pass
    _state_generation = next_generation


def initialize():
    root = data_root()
    test_output = _join(root, "test_2")
    if not _is_dir(test_output):
        os.mkdir(test_output)
    for mode in config.MODES:
        _positions[mode["name"]] = 0
    _load_state()
    set_mode(config.MODES[0]["name"])


def _catalog(mode):
    name = mode["name"]
    if name in _catalogs:
        return _catalogs[name]
    dataset_dir = _join(data_root(), mode["dataset"])
    items = []
    for class_name, class_id in mode["classes"]:
        class_dir = _join(dataset_dir, class_name)
        if not _is_dir(class_dir):
            raise OSError("MISSING_CLASS_DIR:" + class_dir)
        filenames = [
            value for value in os.listdir(class_dir)
            if value.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        filenames.sort()
        for filename in filenames:
            items.append({
                "path": _join(class_dir, filename),
                "filename": filename,
                "expected_class": class_name,
                "expected_id": class_id,
            })
    _catalogs[name] = items
    return items


def status():
    mode = active_mode()
    items = _catalog(mode)
    position = min(_positions.get(mode["name"], 0), len(items))
    return {
        "schema_version": config.SCHEMA_VERSION,
        "firmware_version": config.FIRMWARE_VERSION,
        "active_mode": mode["name"],
        "available_modes": mode_names(),
        "dataset": mode["dataset"],
        "position": position,
        "images_total": len(items),
        "remaining_images": len(items) - position,
        "exhausted": position >= len(items),
        "reserved_sample_id": None if _reserved is None else _reserved["sample_id"],
        "state_generation": _state_generation,
        "mem_free": gc.mem_free(),
    }


def set_mode(name):
    global _active_name, _reserved
    mode = _mode_by_name(name)
    changed = _active_name != name
    if _reserved is not None and _reserved["mode"] != name:
        raise RuntimeError("SAMPLE_PENDING_COMMIT:" + _reserved["sample_id"])
    _active_name = name
    if name not in _positions:
        _positions[name] = 0
    if changed:
        unload_model()
    _catalog(mode)
    return changed


def reset_progress():
    global _reserved
    _reserved = None
    for mode in config.MODES:
        _positions[mode["name"]] = 0
    _save_state()


def reserve_sample():
    global _reserved
    mode = active_mode()
    if _reserved is not None:
        if _reserved["mode"] != mode["name"]:
            raise RuntimeError("RESERVATION_MODE_MISMATCH")
        return _reserved
    items = _catalog(mode)
    position = _positions.get(mode["name"], 0)
    if position >= len(items):
        return None
    item = dict(items[position])
    item["mode"] = mode["name"]
    item["index"] = position
    item["sample_id"] = "%s:%06d:%s" % (mode["name"], position, item["filename"])
    _reserved = item
    return item


def commit_sample(sample_id):
    global _reserved
    if _reserved is None:
        # Idempotent success when the immediately previous index is already committed.
        for mode in config.MODES:
            prefix = mode["name"] + ":"
            if sample_id.startswith(prefix):
                try:
                    index = int(sample_id[len(prefix):].split(":", 1)[0])
                    return index < _positions.get(mode["name"], 0)
                except Exception:
                    return False
        return False
    if sample_id != _reserved["sample_id"]:
        return False
    mode_name = _reserved["mode"]
    _positions[mode_name] = _reserved["index"] + 1
    _reserved = None
    _save_state()
    return True


def unload_model():
    global _model_ctx
    _model_ctx = None
    gc.collect()


def _dtype_code(value):
    return chr(value) if isinstance(value, int) else value


def _check_contract(model, mode):
    actual = (
        tuple(model.input_shape[0]), _dtype_code(model.input_dtype[0]),
        float(model.input_scale[0]), int(model.input_zero_point[0]),
        tuple(model.output_shape[0]), _dtype_code(model.output_dtype[0]),
        float(model.output_scale[0]), int(model.output_zero_point[0]),
    )
    expected = (
        mode["input_shape"], mode["input_dtype"], mode["input_scale"],
        mode["input_zero"], mode["output_shape"], mode["output_dtype"],
        mode["output_scale"], mode["output_zero"],
    )
    for index, pair in enumerate(zip(actual, expected)):
        found, required = pair
        equal = abs(found - required) <= 1e-6 if index in (2, 6) else found == required
        if not equal:
            raise ValueError("MODEL_CONTRACT_%d:%s!=%s" % (index, found, required))
    return actual


def _load_model(mode):
    global _model_ctx
    if _model_ctx is not None and _model_ctx["mode"] == mode["name"]:
        return _model_ctx, True
    unload_model()
    gc.collect()
    memory_before = gc.mem_free()
    model_path = _join(data_root(), mode["model"])
    file_size = int(os.stat(model_path)[6])
    if file_size != mode["model_size_bytes"]:
        raise ValueError("MODEL_SIZE:%d!=%d" % (file_size, mode["model_size_bytes"]))
    start = time.ticks_us()
    model = ml.Model(model_path)
    load_us = time.ticks_diff(time.ticks_us(), start)
    actual = _check_contract(model, mode)
    fire_tensor = None
    if mode["name"] == "THERMAL_FIRE":
        fire_tensor = np.empty((1, 3, 120, 160), dtype=np.float)
    _model_ctx = {
        "mode": mode["name"],
        "model": model,
        "fire_tensor": fire_tensor,
        "model_path": model_path,
        "model_load_us": load_us,
        "model_ram": int(model.ram),
        "model_len": int(model.len),
        "memory_before_model": int(memory_before),
        "memory_after_model": int(gc.mem_free()),
        "contract": actual,
    }
    return _model_ctx, False


class _CloudImageInput:
    def __init__(self, frame, mode):
        self.frame = frame
        self.mode = mode
        self.prepare_us = 0

    def __call__(self, buffer, shape, dtype):
        start = time.ticks_us()
        if tuple(shape) != self.mode["input_shape"] or _dtype_code(dtype) != "b":
            raise ValueError("UNEXPECTED_CLOUD_INPUT_CONTRACT")
        tensor = self.frame.to_ndarray("b", buffer=buffer)
        if tuple(tensor.shape) != (224, 224, 3):
            raise ValueError("UNEXPECTED_CLOUD_TENSOR_SHAPE")
        tensor = None
        self.prepare_us = time.ticks_diff(time.ticks_us(), start)


def _load_frame(path, width, height, pixformat):
    gc.collect()
    start = time.ticks_us()
    source = image.Image(path, copy_to_fb=True)
    input_read_us = time.ticks_diff(time.ticks_us(), start)
    source_size = [int(source.width()), int(source.height())]
    start = time.ticks_us()
    frame = image.Image(width, height, pixformat)
    frame.draw_image(
        source, 0, 0,
        x_scale=float(width) / source_size[0],
        y_scale=float(height) / source_size[1],
        hint=image.BILINEAR,
    )
    image_convert_us = time.ticks_diff(time.ticks_us(), start)
    source = None
    gc.collect()
    return frame, source_size, input_read_us, image_convert_us


def _sigmoid(value):
    factor = math.exp(-abs(value))
    return 1.0 / (1.0 + factor) if value >= 0 else factor / (1.0 + factor)


def _class_name(mode, identifier):
    for name, value in mode["classes"]:
        if value == identifier:
            return name
    return "unknown"


def _prediction(output, mode):
    start = time.ticks_us()
    values = output.flatten().tolist()
    if len(values) != 1:
        raise ValueError("EXPECTED_SCALAR_OUTPUT")
    raw_dequantized = float(values[0])
    probability = (raw_dequantized if mode["output_transform"] == "probability"
                   else _sigmoid(raw_dequantized))
    probability = max(0.0, min(1.0, probability))
    raw_estimate = int(round(raw_dequantized / mode["output_scale"])) + mode["output_zero"]
    raw_estimate = max(-128, min(127, raw_estimate))
    predicted_id = int(probability >= config.THRESHOLD)
    return {
        "raw_dequantized": raw_dequantized,
        "raw_quantized_estimate": raw_estimate,
        "positive_probability": probability,
        "predicted_id": predicted_id,
        "predicted_class": _class_name(mode, predicted_id),
    }, time.ticks_diff(time.ticks_us(), start)


def run_reserved_inference():
    sample = reserve_sample()
    if sample is None:
        return None
    mode = active_mode()
    sample_start = time.ticks_us()
    gc.collect()
    memory_before_sample = gc.mem_free()
    ctx, model_reused = _load_model(mode)
    model_input = None
    frame = None
    try:
        if mode["name"] == "RGB_CLOUD":
            frame, source_size, input_read_us, image_convert_us = _load_frame(
                sample["path"], 224, 224, image.RGB565
            )
            model_input = _CloudImageInput(frame, mode)
            start = time.ticks_us()
            outputs = ctx["model"].predict([model_input])
            predict_total_us = time.ticks_diff(time.ticks_us(), start)
            tensor_prepare_us = model_input.prepare_us
            inference_estimated_us = max(0, predict_total_us - tensor_prepare_us)
        else:
            frame, source_size, input_read_us, image_convert_us = _load_frame(
                sample["path"], 160, 120, image.GRAYSCALE
            )
            start = time.ticks_us()
            gray = frame.to_ndarray("f")
            gray *= 1.0 / 255.0
            tensor = ctx["fire_tensor"]
            for channel in range(3):
                for row in range(120):
                    tensor[0, channel, row, :] = gray[row, :]
            tensor_prepare_us = time.ticks_diff(time.ticks_us(), start)
            model_input = tensor
            start = time.ticks_us()
            outputs = ctx["model"].predict([model_input])
            predict_total_us = time.ticks_diff(time.ticks_us(), start)
            inference_estimated_us = predict_total_us
            gray = None

        if len(outputs) != 1:
            raise ValueError("EXPECTED_ONE_OUTPUT")
        prediction, postprocess_us = _prediction(outputs[0], mode)
        result = {
            "schema_version": config.SCHEMA_VERSION,
            "firmware_version": config.FIRMWARE_VERSION,
            "sample_id": sample["sample_id"],
            "active_mode": mode["name"],
            "dataset": mode["dataset"],
            "image_name": sample["filename"],
            "image_index": sample["index"],
            "expected_class": sample["expected_class"],
            "expected_id": sample["expected_id"],
            "source_size": source_size,
            "prediction": prediction,
            "correct": prediction["predicted_id"] == sample["expected_id"],
            "threshold": config.THRESHOLD,
            "positive_class": mode["positive_class"],
            "remaining_images_after_commit": len(_catalog(mode)) - sample["index"] - 1,
            "no_more_images": False,
            "metrics": {
                "model_reused": model_reused,
                "model_load_us": 0 if model_reused else ctx["model_load_us"],
                "model_ram_bytes": ctx["model_ram"],
                "model_size_bytes": ctx["model_len"],
                "input_read_us": input_read_us,
                "image_convert_us": image_convert_us,
                "tensor_prepare_us": tensor_prepare_us,
                "predict_total_us": predict_total_us,
                "inference_estimated_us": inference_estimated_us,
                "postprocess_us": postprocess_us,
                "compute_total_us": time.ticks_diff(time.ticks_us(), sample_start),
                "memory_before_sample_bytes": int(memory_before_sample),
                "memory_after_inference_bytes": int(gc.mem_free()),
            },
        }
        return result
    finally:
        if isinstance(model_input, _CloudImageInput):
            model_input.frame = None
        outputs = model_input = frame = None
        gc.collect()


def reserved_path():
    return None if _reserved is None else _reserved["path"]

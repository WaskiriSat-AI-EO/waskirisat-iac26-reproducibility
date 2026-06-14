"""TFLite export and host-side verification via ai-edge-torch / ai-edge-litert.

The output artifact is an int8 TFLite file whose input tensor is uint8
(scale≈1/255, zero_point=0), matching the raw pixel format produced by the
OpenMV N6 camera sensor (PAG7936).  The OpenMV IDE / ``stedgeai`` compile step
converts this TFLite to the Neural-ART ``.bin`` — no additional low-level STM32
toolchain is required.

On-device MicroPython usage after deployment::

    import ml, sensor
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)  # resize handled by model input quant
    model = ml.Model("/rom/mobilenetv2_cloud_int8.tflite")
    img = sensor.snapshot()
    score = ml.sigmoid(model.predict([img])[0].flatten()[0])
    is_cloud = score >= THRESHOLD_INT8
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── TFLite export ─────────────────────────────────────────────────────────────


def export_tflite(
    quantized_model: nn.Module,
    sample_input: torch.Tensor,
    out_path: Path,
) -> Path:
    """Export a converted PT2E model to an int8 TFLite file via ai-edge-torch.

    The output TFLite has:

    - Per-output-channel **symmetric int8** weights.
    - Per-tensor **uint8** activations; input tensor has scale≈1/255,
      zero_point=0 — mapping raw uint8 camera pixels to float [0, 1].
    - Normalization (ImageNet mean/std subtraction/division) baked into the
      graph via ``NormalizedMobileNetV2``.
    - Single **int8 logit** output; the MicroPython script applies sigmoid and
      compares to ``selected_threshold_int8`` from ``quantization_report.json``.

    Args:
        quantized_model: Converted model from ``qat.convert_qat_model``.
        sample_input: Representative float32 input tensor ``(1, 3, H, W)``
            in [0, 1].  Used for shape inference during conversion.
        out_path: Destination path for the ``.tflite`` file.

    Returns:
        Path to the written TFLite file.

    Raises:
        ImportError: If ``ai-edge-torch`` is not installed.
        RuntimeError: If the graph contains unsupported operations.
    """
    try:
        import ai_edge_torch
    except ImportError as exc:
        raise ImportError(
            "ai-edge-torch is required for TFLite export.  "
            "Install with: pip install ai-edge-torch"
        ) from exc

    quantized_model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting quantized model to TFLite …")
    edge_model = ai_edge_torch.convert(quantized_model, (sample_input,))
    edge_model.export(str(out_path))

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info("TFLite written: %s  (%.2f MB)", out_path, size_mb)
    return out_path


def run_ptq_smoke_test(
    model_fp32: nn.Module,
    sample_input: torch.Tensor,
    tmp_path: Path,
) -> Path:
    """PTQ smoke test: verify graph traceability before investing in QAT.

    Applies fast post-training static quantization (no fine-tuning) and
    attempts a full TFLite conversion.  If this succeeds, QAT on the same
    graph is very likely to succeed.  If it raises, the error is re-raised
    immediately so the pipeline aborts before spending time on fine-tuning.

    Args:
        model_fp32: ``NormalizedMobileNetV2`` with float32 weights.
        sample_input: Representative input tensor ``(1, 3, H, W)`` in [0, 1].
        tmp_path: Directory for the temporary smoke-test TFLite artifact.

    Returns:
        Path to the smoke-test ``.tflite`` file.

    Raises:
        ImportError: If ``ai-edge-torch`` is not installed.
        Exception: Propagated from PT2E tracing or TFLite conversion on failure.
    """
    from nina_cloud.quantization.qat import prepare_for_ptq

    try:
        import ai_edge_torch
    except ImportError as exc:
        raise ImportError(
            "ai-edge-torch is required.  Install with: pip install ai-edge-torch"
        ) from exc

    logger.info("Running PTQ smoke test …")
    prepared = prepare_for_ptq(model_fp32, sample_input)

    # Single calibration forward pass (PTQ only needs the graph; no data calibration here)
    prepared.eval()
    with torch.no_grad():
        prepared(sample_input)

    from torch.ao.quantization.quantize_pt2e import convert_pt2e

    quantized = convert_pt2e(prepared)
    quantized.eval()

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    smoke_path = tmp_path / "smoke_test_ptq.tflite"

    edge_model = ai_edge_torch.convert(quantized, (sample_input,))
    edge_model.export(str(smoke_path))

    size_kb = smoke_path.stat().st_size / 1024
    logger.info(
        "PTQ smoke test PASSED — TFLite written: %s (%.1f KB)", smoke_path, size_kb
    )
    return smoke_path


# ── Checksum ──────────────────────────────────────────────────────────────────


def write_checksum(tflite_path: Path) -> Path:
    """Write a SHA-256 checksum file alongside the TFLite artifact.

    The checksum file follows the ``sha256sum`` format::

        <hex_digest>  <filename>

    Args:
        tflite_path: Path to the ``.tflite`` file.

    Returns:
        Path to the written ``.sha256`` file.
    """
    sha = hashlib.sha256(tflite_path.read_bytes()).hexdigest()
    checksum_path = tflite_path.with_suffix(".sha256")
    checksum_path.write_text(f"{sha}  {tflite_path.name}\n", encoding="utf-8")
    logger.info("Checksum written: %s  sha256=%s…", checksum_path, sha[:16])
    return checksum_path


# ── Host-side TFLite inference (parity) ──────────────────────────────────────


def get_interpreter(tflite_path: Path):
    """Load a TFLite interpreter via ai-edge-litert.

    Args:
        tflite_path: Path to the ``.tflite`` file.

    Returns:
        Allocated ``ai_edge_litert.interpreter.Interpreter`` instance.

    Raises:
        ImportError: If ``ai-edge-litert`` is not installed.
    """
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError as exc:
        raise ImportError(
            "ai-edge-litert is required for host-side parity verification.  "
            "Install with: pip install ai-edge-litert"
        ) from exc

    interp = litert.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    return interp


def tflite_logit(interp, img_uint8_hwc: np.ndarray) -> float:
    """Run a single uint8 HWC image through the TFLite interpreter.

    Handles both NHWC and NCHW input layouts as reported by the interpreter's
    input details, and dequantizes the int8 output logit using the output
    tensor's quantization parameters.

    Args:
        interp: Allocated interpreter from ``get_interpreter``.
        img_uint8_hwc: Uint8 numpy array of shape ``(H, W, 3)`` in [0, 255].

    Returns:
        Dequantized float logit (before sigmoid).
    """
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_shape = in_det["shape"]  # e.g. (1, H, W, 3) or (1, 3, H, W)
    in_dtype = in_det["dtype"]

    # Build input array matching the expected layout
    if len(in_shape) == 4 and in_shape[1] == 3:
        # NCHW — transpose HWC → CHW then add batch
        arr = img_uint8_hwc.transpose(2, 0, 1)[np.newaxis]
    else:
        # NHWC — just add batch dim
        arr = img_uint8_hwc[np.newaxis]

    if in_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(in_dtype)

    interp.set_tensor(in_det["index"], arr)
    interp.invoke()
    raw = interp.get_tensor(out_det["index"])

    quant_params = out_det.get("quantization", (0.0, 0))
    out_scale, out_zp = float(quant_params[0]), int(quant_params[1])
    if out_scale > 0:
        return float((int(raw.flat[0]) - out_zp) * out_scale)
    return float(raw.flat[0])

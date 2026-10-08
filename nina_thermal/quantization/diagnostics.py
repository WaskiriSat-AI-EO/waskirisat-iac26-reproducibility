"""Locate where int8 conversion diverges from the fp32 / fake-quant model.

The quantization pipeline has four representations of the same network:

1. fp32 (``NormalizedFireNet``),
2. the prepared PT2E graph — plain observers (PTQ) or fake-quantize modules
   (QAT), which simulate int8 while computing in float,
3. the converted PT2E graph — real quantize/dequantize ops,
4. the exported TFLite file.

A faithful conversion keeps 3 and 4 tracking 2 to within rounding (score
correlation > 0.99).  :func:`stage_report` scores every stage on the same
images and reports ROC-AUC and correlation with fp32; :func:`compare_graphs`
walks two graphs node by node to find the first operation whose output
diverges.

See ``docs/PLAN_int8_fix.md``.
"""

import logging
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.fx import Interpreter

from nina_thermal.quantization import qat as qat_lib
from nina_thermal.training.threshold import roc_auc

logger = logging.getLogger(__name__)


def logits(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    """Return raw logits of ``model`` over ``loader``, batch size as given.

    Args:
        model: Plain module or PT2E graph module.
        loader: Loader yielding ``(images, labels)`` with images in [0, 1].
        device: Device the model lives on.

    Returns:
        1-D array of logits in dataset order.
    """
    out: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            out.extend(model(images.to(device)).flatten().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def tflite_logits(tflite_path: Path, loader) -> np.ndarray:
    """Return logits of a TFLite file over ``loader`` (inverse of its sigmoid).

    Args:
        tflite_path: Exported model.
        loader: Loader yielding ``(images, labels)``.

    Returns:
        1-D array of logits in dataset order.
    """
    scores = np.clip(np.asarray(qat_lib.tflite_scores(tflite_path, loader)), 1e-7, 1 - 1e-7)
    return np.log(scores / (1 - scores))


def agreement(reference: np.ndarray, test: np.ndarray) -> dict:
    """Correlation and logit-space SQNR of ``test`` against ``reference``.

    Args:
        reference: Reference logits.
        test: Logits to compare.

    Returns:
        ``{"corr", "sqnr_db"}``; ``corr`` is NaN when ``test`` is constant.
    """
    noise = test - reference
    corr = float(np.corrcoef(reference, test)[0, 1]) if np.std(test) > 0 else float("nan")
    sqnr = 10 * np.log10((reference**2).sum() / max((noise**2).sum(), 1e-12))
    return {"corr": round(corr, 4), "sqnr_db": round(float(sqnr), 2)}


def calibrate(model: nn.Module, loader, device: torch.device, batches: int) -> None:
    """Run ``batches`` batches through a prepared model in eval mode."""
    qat_lib.set_mode(model, training=False)
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= batches:
                break
            model(images.to(device))


def count_nodes(graph_module: nn.Module, substring: str) -> int:
    """Count call_function nodes whose target contains ``substring``."""
    return sum(
        n.op == "call_function" and substring in str(n.target)
        for n in graph_module.graph.nodes
    )


def stage_report(
    model: nn.Module,
    val_loader,
    cal_loader,
    labels: list[int],
    device: torch.device,
    *,
    qat: bool,
    calibration_batches: int = 64,
    quantizer: object | None = None,
    model_hook: Callable[[nn.Module], nn.Module] | None = None,
    tflite: bool = True,
) -> dict:
    """Score fp32, prepared, converted and TFLite stages on the same images.

    No fine-tuning happens here: the prepared graph is only calibrated, which
    is enough to test whether conversion preserves what the prepared graph
    computes.

    Args:
        model: fp32 ``NormalizedFireNet`` in eval mode.
        val_loader: Unshuffled validation loader (batch size 1 keeps the
            graph's batch dimension out of the comparison).
        cal_loader: Loader for observer calibration.
        labels: Validation labels aligned with ``val_loader``.
        device: Device for PyTorch stages.
        qat: Prepare with the QAT config (fake-quant) instead of PTQ.
        calibration_batches: Calibration batches.
        quantizer: Optional pre-configured quantizer (e.g. modules excluded).
        model_hook: Optional transform applied to a copy of ``model`` before
            export (e.g. disabling in-place ops).
        tflite: Also export and score the TFLite file.

    Returns:
        ``{"stages": [{"stage", "roc_auc", "corr", "sqnr_db"}...],
        "bn_nodes_after_convert": int}``.
    """
    import copy

    model = copy.deepcopy(model).eval()
    if model_hook is not None:
        model = model_hook(model)
    example = qat_lib.sample_input(device)
    fp32 = logits(model, val_loader, device)
    stages = [{"stage": "fp32", "roc_auc": roc_auc(labels, fp32.tolist())}]

    prepare = qat_lib.prepare_for_qat if qat else qat_lib.prepare_for_ptq
    if quantizer is None:
        quantizer = qat_lib.build_quantizer(qat=qat)
    prepared, quantizer = prepare(model, example, quantizer=quantizer)
    calibrate(prepared, cal_loader, device, calibration_batches)
    simulated = logits(prepared, val_loader, device)
    stages.append(
        {"stage": "prepared (" + ("fake-quant" if qat else "observers") + ")",
         "roc_auc": roc_auc(labels, simulated.tolist()), **agreement(fp32, simulated)}
    )

    converted = qat_lib.convert(prepared)
    real = logits(converted, val_loader, device)
    stages.append(
        {"stage": "converted PT2E", "roc_auc": roc_auc(labels, real.tolist()),
         **agreement(fp32, real), "corr_vs_prepared": agreement(simulated, real)["corr"]}
    )
    report = {"stages": stages, "bn_nodes_after_convert": count_nodes(converted, "batch_norm")}

    if tflite:
        with tempfile.TemporaryDirectory() as tmp:
            path = qat_lib.export_tflite(converted, example, Path(tmp) / "m.tflite", quantizer=quantizer)
            exported = tflite_logits(path, val_loader)
        stages.append(
            {"stage": "TFLite", "roc_auc": roc_auc(labels, exported.tolist()),
             **agreement(fp32, exported), "corr_vs_prepared": agreement(simulated, exported)["corr"]}
        )
    return report


class _Recorder(Interpreter):
    """FX interpreter that keeps every floating-point node output."""

    def __init__(self, graph_module: nn.Module) -> None:
        super().__init__(graph_module)
        self.outputs: dict[str, tuple[str, torch.Tensor]] = {}

    def run_node(self, n):
        result = super().run_node(n)
        if n.op == "call_function" and isinstance(result, torch.Tensor) and result.is_floating_point():
            self.outputs[n.name] = (str(n.target), result.detach().clone())
        return result


def compare_graphs(reference: nn.Module, test: nn.Module, image: torch.Tensor) -> list[dict]:
    """Per-node relative error of ``test`` against ``reference`` on one image.

    Nodes are matched by name, which PT2E preserves for the original
    operations through prepare and convert; inserted observer, fake-quant and
    quantize/dequantize nodes are skipped.

    Args:
        reference: Prepared graph (the int8 simulation).
        test: Converted graph.
        image: Input ``(1, 3, H, W)``.

    Returns:
        Rows ``{"node", "target", "rel_err"}`` in the test graph's execution
        order.
    """
    with torch.no_grad():
        ref = _Recorder(reference)
        ref.run(image)
        out = _Recorder(test)
        out.run(image)
    skip = ("quantize", "activation_post_process", "fake_quant")
    rows = []
    for name, (target, tensor) in out.outputs.items():
        if any(s in name or s in target for s in skip) or name not in ref.outputs:
            continue
        expected = ref.outputs[name][1]
        if expected.shape != tensor.shape:
            continue
        rel = (tensor - expected).norm() / (expected.norm() + 1e-12)
        rows.append({"node": name, "target": target, "rel_err": round(float(rel), 4)})
    return rows


def first_jump(rows: list[dict], factor: float = 3.0, floor: float = 0.05) -> dict | None:
    """First row whose error exceeds ``floor`` and ``factor`` × the previous row's."""
    previous = 0.0
    for row in rows:
        if row["rel_err"] > floor and row["rel_err"] > factor * max(previous, 1e-3):
            return row
        previous = row["rel_err"]
    return None

"""Quantization-Aware Training (QAT) pipeline for the cloud/clear classifier.

Produces a per-channel int8 TFLite model (uint8 input) deployable on the
OpenMV N6 / STM32N6 Neural-ART NPU via MicroPython::

    model = ml.Model("/rom/mobilenetv2_cloud_int8.tflite")

Methodology
-----------
The Neural-ART NPU runs INT8 exclusively.  Plain post-training quantization
(PTQ) applied only to the Linear head (as in ``scripts/eval_and_plot.py``) is
insufficient: the convolutional backbone retains float32 weights and cannot be
dispatched to the NPU.  Full-network Quantization-Aware Training (QAT) via
PyTorch PT2E inserts fake-quant nodes throughout the graph and fine-tunes the
model so that all layers tolerate 8-bit precision.  The resulting artifact is
then converted to int8 TFLite via ``ai-edge-torch``.

Pipeline stages
~~~~~~~~~~~~~~~
1. **PTQ smoke test** — verify the fp32 graph exports to int8 TFLite before
   committing to QAT training time.  Aborts early if the graph contains
   unsupported ops.
2. **QAT fine-tune** — load ``mobilenetv2_cloud_best.pt``, unfreeze the
   backbone, insert fake-quant via ``prepare_qat_pt2e`` with the
   ``TFLiteQuantizer`` (per-channel int8 weights, uint8 activations), warm up
   observers, then fine-tune for a small number of epochs with the same
   asymmetric ``BCEWithLogitsLoss(pos_weight=3.0)``.
3. **TFLite export** — convert via ``ai_edge_torch`` to an int8 TFLite.
   ImageNet mean/std normalization is baked into the graph via
   ``NormalizedMobileNetV2`` so the camera's raw uint8 output feeds the model
   directly without any host preprocessing.
4. **Threshold re-derivation** — the float32 operating threshold is invalid
   after quantization; re-run the val-set recall-target sweep on int8 scores
   and persist ``selected_threshold_int8``.
5. **Parity report** — compare fp32 vs int8 on the held-out split:
   Δ recall / precision / F-beta / ROC-AUC, score-distribution KL divergence,
   and Signal-to-Quantization-Noise Ratio (SQNR) as evidence for the IAC paper.

Usage::

    python -m nina_cloud.quantization.pipelines.run_quantize \\
        --dataset_dir data/cloud/04_dataset \\
        --epochs 5 --lr 1e-5 --recall_target 0.99 \\
        --out models/

Dependencies
~~~~~~~~~~~~
This module requires ``torch`` / ``torchvision``, ``ai-edge-torch``, and
``ai-edge-litert`` (host-side TFLite runtime for parity verification).  The
evaluation sub-module ``nina_cloud.evaluation.metrics`` remains torch-free for
CI compatibility; these heavy deps are only pulled in at quantization time.

Asymmetric cost note
--------------------
``pos_weight=3.0`` is preserved in QAT fine-tuning: a missed cloud (FN) wastes
irreplaceable on-orbit downlink bandwidth; a false alarm (FP) merely discards a
clear image.  The asymmetry shifts the operating threshold toward higher recall.
The val-set threshold re-sweep enforces ``recall >= recall_target`` on the *int8*
score distribution, which may differ from the float32 one.
"""

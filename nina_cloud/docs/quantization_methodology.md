# Quantization Methodology: Cloud/Clear Classifier for On-Orbit Deployment

**WaskiriSat Nina-Cloud IQC Payload — IAC Paper Section**

> **Scope of this document.** It describes the PyTorch reference pipeline in this
> package (MobileNetV2 with **α = 1.0**, QAT, `pos_weight` 3.0). The model reported in the
> paper was produced by the Keras scripts in `nina_cloud_results/` instead:
> **α = 0.75**, post-training full-integer INT8, threshold 0.9336. For the paper run read
> `../configs/paper.yaml` and `../MODEL_CARD.md`; see `../../ERRATA.md` #1.

---

## 1. Problem Framing

The cloud/clear classifier is the Image Quality Control (IQC) safety filter for
Bolivia's first CubeSat.  It gates satellite downlink: a cloud prediction on a
tile triggers discard; a clear prediction triggers downlink of 1280×800 px
Sentinel-2-equivalent imagery.  Because downlink windows are irreplaceable, the
mission cost function is **asymmetric**: a missed cloud (false negative, FN) is
more costly than a false alarm (false positive, FP) that discards a clear image.

This asymmetry governs all quantization decisions.  The metric of primary
interest is **cloud recall** (sensitivity), not accuracy.  The operating
threshold is not fixed at 0.5 — it is swept post-training and post-quantization
to enforce a recall floor:

    threshold* = argmax_t { t : recall(t) ≥ recall_target }

with `recall_target = 0.99` by default.

---

## 2. Hardware Target

| Parameter | Value |
|-----------|-------|
| Board | OpenMV N6 (STMicroelectronics STM32N657) |
| MCU | Arm Cortex-M55 @ 800 MHz, Helium SIMD |
| NPU | ST Neural-ART @ 1 GHz, **600 GOPS INT8** |
| External RAM | 64 MB SDRAM (800 MB/s) |
| Model storage | 32 MB QSPI flash (read-only `/rom` filesystem) |
| Deployment API | MicroPython `ml.Model("/rom/model.tflite")` |
| Peak power | < 0.75 W |

The Neural-ART NPU **natively executes INT8 operations only**.  Float32 layers
bypass the NPU and run on the Cortex-M55 at much lower throughput.  Achieving
full-NPU execution therefore requires every layer of the network to be quantized
to INT8.

---

## 3. Baseline Model

**Architecture:** MobileNetV2 (width multiplier α = 1.0, ImageNet pretrained)
with a frozen backbone and a single-logit binary head:

    features (MobileNetV2 backbone, frozen in training, unfrozen in QAT)
    └── classifier
        ├── Dropout(p=0.2)
        └── Linear(1280 → 1)     # single logit; sigmoid applied externally

**Input:** 224 × 224 × 3 RGB, ImageNet-normalized
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

**Loss:** `BCEWithLogitsLoss(pos_weight=3.0)`.
The `pos_weight` encodes the FN:FP cost ratio (not class imbalance — the
training set is balanced 50/50).  `β = √pos_weight = √3 ≈ 1.732` for F-β.

**Operating point:** threshold `t*` selected on the validation set by the
recall-target sweep; stored in `training_report.json`.

**Float32 artifact:** `mobilenetv2_cloud_best.pt` (bare state dict, ~9 MB).

---

## 4. Quantization Strategy: Quantization-Aware Training (QAT)

### 4.1 Why QAT over PTQ

Post-Training Quantization (PTQ) applied after training is simple but can degrade
recall significantly for small-head classifiers where the frozen backbone has
never seen quantization noise.  QAT fine-tunes the model with **simulated integer
arithmetic** (fake-quant nodes) so that all layers, including the frozen
convolutional backbone, adapt to 8-bit precision before conversion.  For
safety-critical applications where recall < 0.99 is unacceptable, QAT is the
preferred strategy.

A PTQ smoke test is nonetheless run first (< 30 s) to verify graph traceability
and TFLite export compatibility before investing in QAT fine-tuning.

### 4.2 Quantization Scheme

| Element | Scheme |
|---------|--------|
| **Weights** | Per-output-channel symmetric **int8** |
| **Activations** | Per-tensor asymmetric **uint8** |
| **Input tensor** | uint8 `[0, 255]`; scale ≈ 1/255, zero_point = 0 |
| **Output tensor** | int8 logit; sigmoid + threshold in MicroPython |
| **BatchNorm** | Folded into preceding Conv2d weights at convert time |

**Input normalization folding.** ImageNet mean/std normalization (subtraction and
division) is baked into the model graph via `NormalizedMobileNetV2`, which wraps
the backbone and applies normalization in its `forward()` pass before the first
convolutional layer.  This means the TFLite model's input quantization maps raw
uint8 camera pixels directly to the normalized float range the backbone expects:

    on-device:  uint8 [0,255] --dequant(scale=1/255, zp=0)--> float [0,1]
                               --NormLayer(mean, std)-------> float [-2.1, 2.6]
                               --> backbone --> logit (int8)

No host-side preprocessing is required in the MicroPython inference script.

### 4.3 Framework and Toolchain

```
PyTorch (train_cloud.py)
    ↓  torch.export.export (PT2E graph capture)
    ↓  prepare_qat_pt2e + TFLiteQuantizer (ai-edge-torch)
    ↓  observer warmup + QAT fine-tune (BCEWithLogitsLoss, pos_weight=3)
    ↓  convert_pt2e (fake-quant → true int8)
    ↓  ai_edge_torch.convert → .export()
mobilenetv2_cloud_int8.tflite   (int8 weights, uint8 input)
    ↓  OpenMV IDE / stedgeai (automatic on drag-and-drop to board)
Neural-ART .bin (NPU-compiled, runs at 600 GOPS)
    ↓  ml.Model("/rom/mobilenetv2_cloud_int8.tflite")
MicroPython inference (OpenMV N6, < 0.75 W)
```

**Library versions (at time of training):**
`torch==2.12.0`, `torchvision==0.27.0`, `ai-edge-torch>=0.3.0`,
`ai-edge-litert>=1.0.0`.

### 4.4 QAT Procedure

1. **Load float32 checkpoint.** Initialize `NormalizedMobileNetV2` from
   `mobilenetv2_cloud_best.pt`.  Unfreeze all backbone parameters — QAT
   requires gradients through convolutional layers to compensate quantization
   noise that accumulates in early layers.

2. **PT2E graph export.** `torch.export.export` captures the full computation
   graph as a flat, traceable representation required by PT2E quantization.

3. **Quantizer attachment.** `TFLiteQuantizer` (from `ai_edge_torch.quantize`)
   with `get_symmetric_quantization_config(is_per_channel=True, is_qat=True)`
   inserts fake-quant nodes at every weight and activation site, using the
   TFLite quantization convention (uint8 activations, int8 weights).

4. **Observer warmup.** Run 10 calibration batches (profile-B train split,
   no augmentation, no gradients) through the prepared model to initialize
   per-tensor min/max statistics before fake-quant is enabled.

5. **Fine-tuning.** Train for **5 epochs** with Adam (`lr = 1×10⁻⁵`) and the
   original asymmetric loss `BCEWithLogitsLoss(pos_weight=3.0)`.  This LR is
   100× smaller than the original training LR to prevent catastrophic
   forgetting while allowing scales to settle.  Full augmentation
   (HFlip, VFlip, ColorJitter) is applied.

6. **PT2E conversion.** `convert_pt2e` freezes observer statistics, folds
   BatchNorm layers, and replaces fake-quant nodes with true QDQ
   (QuantizeLinear / DequantizeLinear) operators.

7. **TFLite export.** `ai_edge_torch.convert` lowers the PT2E graph to a
   TFLite FlatBuffer with per-channel int8 weights and uint8 activations.

---

## 5. Threshold Re-Derivation

After quantization, the sigmoid score distribution shifts.  The float32
operating threshold `t*_{fp32}` from `training_report.json` is therefore
**invalid** on the int8 model.  A fresh threshold sweep is performed on the
int8 model's validation-set scores:

    t*_{int8} = argmax_t { t : recall_{int8}(t) ≥ recall_target }

with `recall_target = 0.99`.  The result is stored as `selected_threshold_int8`
in `quantization_report.json` and must be used in the MicroPython inference
script:

```python
THRESHOLD = 0.512   # selected_threshold_int8 from quantization_report.json
score = ml.sigmoid(model.predict([img])[0].flatten()[0])
is_cloud = score >= THRESHOLD
```

---

## 6. Metric-Preservation Evaluation

### 6.1 Metrics

All metrics are computed by `nina_cloud.evaluation.metrics` (identical to the
float32 evaluation pipeline) on the same held-out split, using the respective
thresholds (`t*_{fp32}` for fp32, `t*_{int8}` for int8):

| Metric | Purpose |
|--------|---------|
| Cloud recall (sensitivity) | Primary mission constraint |
| Specificity (TNR) | Downlink efficiency |
| Precision | Bandwidth utilization |
| F-β (β = √3) | Joint metric coherent with pos_weight |
| ROC-AUC | Threshold-independent discriminability |
| PR-AUC (avg. precision) | Positive-class ranking quality |
| Bayes-projected precision | Operational estimates at π = 0.30, 0.50 |
| Confusion matrix | TN / FP / FN / TP absolute counts |

### 6.2 Quantization-Quality Evidence

Two additional diagnostics are reported for the IAC paper:

**Signal-to-Quantization-Noise Ratio (SQNR, dB):**

    SQNR = 10 · log₁₀(E[s²_{fp32}] / E[(s_{fp32} - s_{int8})²])

where `s` denotes sigmoid scores.  Measured in the score space rather than
weight space because score-level distortion is what directly affects the
operating threshold.  SQNR ≥ 30 dB indicates acceptable distortion for
binary classification.

**KL Divergence KL(fp32 ‖ int8):**

    KL = Σ p_{fp32}(b) · log(p_{fp32}(b) / p_{int8}(b))   (histogram approximation, 64 bins)

A high KL divergence (> 0.05 nats) signals that the int8 score distribution
has shifted substantially from the float32 one, which justifies the threshold
re-derivation step.

### 6.3 Robustness

Metrics are reported separately for each degradation profile:

- **Profile B (nominal):** PAG7936 sensor simulation at nominal attitude — the
  training and deployment condition; the primary headline metric.
- **Profile A (best-case):** minimal motion blur / noise.
- **Profile C (worst-case):** SAA-transit radiation-induced noise peaks.

Profiles A and C serve as out-of-distribution robustness bounds and are not
used for threshold tuning.

### 6.4 Acceptance Gate

A quantized model is accepted for deployment when:

| Criterion | Tolerance |
|-----------|-----------|
| Cloud recall drop (fp32 − int8) | ≤ 0.5 pp |
| F-β drop (fp32 − int8) | ≤ 1.0 pp |

If either constraint is violated, the recommended remedy is increasing QAT
epochs or calibration batches.  The documented fallback is Keras TFMOT QAT
(TensorFlow Model Optimization Toolkit), which produces the same int8 TFLite
artifact via a mature, ST-certified toolchain.

---

## 7. Artifact Registry

| File | Description |
|------|-------------|
| `models/mobilenetv2_cloud_int8.tflite` | INT8 TFLite, uint8 input, ~2–3 MB |
| `models/mobilenetv2_cloud_int8.sha256` | SHA-256 checksum |
| `models/quantization_report.json` | Full parity report + acceptance verdict |

`quantization_report.json` fields used in the paper:

```json
{
  "quantization_strategy": "QAT",
  "framework": "PyTorch PT2E + ai-edge-torch",
  "target": "STM32N6 Neural-ART NPU (OpenMV N6, MicroPython ml.Model)",
  "thresholds": {
    "fp32_selected_threshold": 0.XXXX,
    "int8_selected_threshold": 0.XXXX
  },
  "parity": {
    "sqnr_db": XX.X,
    "kl_divergence": 0.00XXX,
    "nominal_deltas": {
      "recall":   { "fp32": 0.XXX, "int8": 0.XXX, "delta": -0.00X },
      "f_beta":   { "fp32": 0.XXX, "int8": 0.XXX, "delta": -0.00X },
      "roc_auc":  { "fp32": 0.XXX, "int8": 0.XXX, "delta": -0.00X }
    }
  },
  "acceptance_gate": { "passed": true, "violations": [] }
}
```

---

## 8. Reproducibility

```bash
# 1. Ensure a balanced dataset (prerequisite — see datagen pipeline)
python -m nina_cloud.datagen.pipelines.run_dataset \
    --dataset_dir data/cloud/04_dataset

# 2. Train float32 model
python -m nina_cloud.training.train_cloud \
    --dataset_dir data/cloud/04_dataset \
    --epochs 15 --pos_weight 3.0 --recall_target 0.99

# 3. Quantize (QAT → int8 TFLite)
python -m nina_cloud.quantization.pipelines.run_quantize \
    --dataset_dir data/cloud/04_dataset \
    --epochs 5 --lr 1e-5 --recall_target 0.99 \
    --out models/

# 4. Deploy — copy models/mobilenetv2_cloud_int8.tflite to OpenMV board
#    (OpenMV IDE auto-compiles to Neural-ART on flash)
```

Seeding: `--seed 42` (default) ensures deterministic augmentation and
weight initialization throughout.

---

## 9. Fallback: Keras TFMOT QAT

If `ai-edge-torch` op coverage is insufficient for a future model variant,
the alternative is:

1. Rebuild `MobileNetV2` classifier in Keras/TensorFlow (`tf.keras.applications`).
2. Apply `tfmot.quantization.keras.quantize_model` for QAT fine-tuning.
3. Export via `TFLiteConverter` with
   `inference_input_type=tf.uint8` and a representative dataset.

This produces the same `int8 TFLite / uint8 input` artifact and is the
path used by ST's own YOLOv8 reference implementation for the N6.  The cost
is duplicating the classifier training in a second framework; the float32
PyTorch checkpoint becomes reference-only.

---

## References

- ST Edge AI Core — Quantized model support:
  https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html
- ai-edge-torch (Google): https://github.com/google-ai-edge/ai-edge-torch
- PyTorch PT2E Quantization:
  https://pytorch.org/docs/stable/quantization.html#prototype-pt2e-quantization
- OpenMV N6 `ml` module:
  https://docs.openmv.io/library/omv.ml.html
- Howard et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks,"
  CVPR 2018.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference," CVPR 2018.

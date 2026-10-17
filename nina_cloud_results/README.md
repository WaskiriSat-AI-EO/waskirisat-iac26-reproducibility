# NINA Cloud Results

This directory contains the artifacts used to reproduce the training, evaluation, and quantization results of the WaskiriSat Cloud/IQC classifier.

The final model is based on MobileNetV2 with `alpha = 0.75` for binary classification between:

- `CLEAR`
- `CLOUD`

---

## Dataset composition

The final dataset contains 7,910 unique tiles.

| Split | CLEAR | CLOUD | Unique tiles | Profile |
|---|---:|---:|---:|---|
| Train | 2,653 | 2,851 | 5,504 | B |
| Validation | 468 | 530 | 998 | B |
| Test | 834 | 574 | 1,408 | A, B, C |
| **Total** | **3,955** | **3,955** | **7,910** | - |

The test set contains 1,408 unique tiles. Each test tile is evaluated under degradation profiles A, B, and C, resulting in:

`1,408 × 3 = 4,224` physical test images.

The dataset policy recorded during training was:

- Training: profile B
- Validation: profile B
- Test: profiles A + B + C
- Test data used during training: `false`
- Scene leakage: `0`

Dataset metadata and composition are stored under:

- `data/manifests/`
- `nina_cloud_results/dataset/`
- `nina_cloud_results/configs/scene_registry.json`

---

## Training

The final classifier was trained using MobileNetV2 with ImageNet weights.

Main training script:

`train_movil_v3_16ago.py`

Final FP32 model:

`mobilenetv2_best.keras`

Training configuration:

| Parameter | Value |
|---|---:|
| Batch size | 32 |
| Cloud weight | 1.0 |
| Seed | 42 |
| Phase 1 epochs | 10 |
| Phase 1 learning rate | 0.0001 |
| Phase 2 epochs | 8 |
| Phase 2 learning rate | 0.00001 |
| Unfrozen layers in Phase 2 | 30 |
| Batch Normalization in Phase 2 | Frozen |

Training augmentation was applied only to the training subset:

- Horizontal flip
- Vertical flip
- Rotation factor: `0.03`
- Brightness factor: `0.1`
- Contrast factor: `0.1`

The final checkpoint was selected from the fine-tuning stage using validation PR-AUC.

Validation metrics at threshold `0.5` for the selected checkpoint were:

| Metric | Value |
|---|---:|
| Accuracy | 0.9709 |
| Precision | 0.9980 |
| Recall | 0.9472 |
| ROC-AUC | 0.9984 |
| PR-AUC | 0.9988 |

---

## FP32 threshold calibration

The FP32 classification threshold was calibrated using only `VALIDATION B`.

- Recall target: `0.95`
- Selected threshold: `0.2944725156`
- Test data used for tuning: `false`

Validation results at the selected threshold:

| Metric | Value |
|---|---:|
| Accuracy | 0.9820 |
| Precision CLOUD | 0.9981 |
| Recall CLOUD | 0.9679 |
| Specificity CLEAR | 0.9979 |
| F1 CLOUD | 0.9828 |

Confusion matrix:

- TN: 467
- FP: 1
- FN: 17
- TP: 513

---

## FP32 evaluation

The final FP32 model was evaluated independently under degradation profiles A, B, and C.

| Profile | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| A | 0.9389 | 0.9729 | 0.8746 | 0.9832 | 0.9211 |
| B | 0.9382 | 0.9959 | 0.8519 | 0.9976 | 0.9183 |
| C | 0.8558 | 0.9793 | 0.6603 | 0.9904 | 0.7888 |
| Global A+B+C | 0.9110 | 0.9828 | 0.7956 | 0.9904 | 0.8793 |

Global confusion matrix:

- TN: 2,478
- FP: 24
- FN: 352
- TP: 1,370

The FP32 evaluation uses:

- 1,408 images for profile A
- 1,408 images for profile B
- 1,408 images for profile C
- 4,224 images for the global A+B+C evaluation

Original reports and confusion matrices are stored under:

`nina_cloud_results/evaluation/fp32/`

---

## Full Integer INT8 quantization

The final quantized model uses Full Integer INT8 quantization.

Main quantization script:

`quantize_alpha075_fullinteger.py`

Generated model:

`mobilenetv2_alpha075_fullinteger_int8.tflite`

Model information:

| Parameter | Value |
|---|---:|
| Model type | Full Integer INT8 TFLite |
| Size | 1.851 MB |
| Input dtype | int8 |
| Output dtype | int8 |
| Input scale | 1.0 |
| Input zero point | -128 |
| Output scale | 0.00390625 |
| Output zero point | -128 |

---

## INT8 threshold calibration

The INT8 threshold was calibrated using only `VALIDATION B`.

- Recall target: `0.95`
- Selected threshold: `0.93359375`

Validation results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9579 |
| Precision CLOUD | 0.9692 |
| Recall CLOUD | 0.9509 |
| Specificity CLEAR | 0.9658 |
| F1 CLOUD | 0.9600 |
| ROC-AUC | 0.9897 |
| PR-AUC | 0.9926 |

This corresponds to the reported threshold:

`t* ≈ 0.934`

---

## INT8 evaluation

The quantized model was evaluated using the same A, B, and C test profiles.

| Profile | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.8658 | 1.0000 | 0.6707 | 1.0000 | 0.8029 | 0.9758 | 0.9719 |
| B | 0.9354 | 0.9782 | 0.8606 | 0.9868 | 0.9157 | 0.9798 | 0.9768 |
| C | 0.8260 | 0.9940 | 0.5767 | 0.9976 | 0.7299 | 0.9766 | 0.9702 |
| Global A+B+C | 0.8757 | 0.9894 | 0.7027 | 0.9948 | 0.8217 | 0.9719 | 0.9663 |

Global confusion matrix:

- TN: 2,489
- FP: 13
- FN: 512
- TP: 1,210

The per-profile ROC-AUC values are between `0.9758` and `0.9798`, while PR-AUC values are between `0.9702` and `0.9768`.

Original INT8 reports and confusion matrices are stored under:

`nina_cloud_results/evaluation/int8/`

and:

`nina_cloud_results/quantization/reports/`

---

## INT8 inference benchmark

The recorded benchmark for the Full Integer INT8 model used:

- Warm-up runs: 30
- Measured runs: 300
- Batch size: 1
- Threads: 1

Results:

| Metric | Value |
|---|---:|
| Mean inference time | 2.993 ms |
| Median | 2.952 ms |
| Standard deviation | 0.529 ms |
| P95 | 3.114 ms |
| Minimum | 2.703 ms |
| Maximum | 9.710 ms |
| FPS from mean | 334.15 |

---

## Results structure

The original artifacts are preserved under:

```text
nina_cloud_results/
├── configs/
├── dataset/
├── training/
│   ├── models/
│   ├── reports/
│   └── scripts/
├── evaluation/
│   ├── fp32/
│   └── int8/
└── quantization/
    ├── scripts/
    ├── reports/
    └── training_mobilenetv2_alpha075_fullinteger/

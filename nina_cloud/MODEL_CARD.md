# Model card: NINA cloud / clear classifier (IQC)

| | |
|---|---|
| Task | Binary classification of an RGB tile resized to 224 × 224: `clear` (0) or `cloud` (1) |
| Architecture | MobileNetV2, **width multiplier α = 0.75**, ImageNet init, single sigmoid output |
| Input | 224 × 224 × 3 RGB; scaling `x / 127.5 − 1` is inside the model |
| FP32 checkpoint | `nina_cloud_results/training/training_mobilenetv2_alpha075/mobilenetv2_best.keras` |
| INT8 artifact | `mobilenetv2_alpha075_fullinteger_int8.tflite`, sha256 `f528086c…6d5e`, 1,941,072 bytes, the file tested in HIL |
| Config | [`configs/paper.yaml`](configs/paper.yaml); results in `../nina_cloud_results/` |
| Owner | Leyla (WaskiriSat AI-EO) |
| Licence | **Proposal, for the supervisor to confirm:** code BSD-3-Clause (repo `LICENSE`); weights CC BY 4.0 |

## Intended use

Image Quality Control (IQC) filter of the NINA payload: it gates downlink by discarding tiles that are mostly cloud. A `cloud` prediction discards the tile, a `clear` prediction lets it through.

Not intended for: cloud-mask products, cloud-fraction estimation, or any use on real PAG7936 imagery. It is **not flight-cleared**, and has only seen synthetic tiles derived from Sentinel-2.

## Training data

7,910 unique tiles from 41 Sentinel-2 L2A scenes over Bolivia, labelled from the scene cloud mask and degraded to the payload's profiles. Training and validation use profile B only; the test set uses A, B and C. No scene appears in more than one split. See [`DATASET_CARD.md`](DATASET_CARD.md).

## Performance (evidence tier T2: host evaluation of the INT8 file on synthetic data)

Test split: 1,408 tiles per profile (574 cloud / 834 clear). Threshold 0.9336, fixed on validation B.

| Profile | Recall | Precision | ROC-AUC | PR-AUC | Reproduced by |
|---|---|---|---|---|---|
| A | 0.671 | 1.000 | 0.976 | 0.972 | `make table6` |
| B | 0.861 | 0.978 | 0.980 | 0.977 | `make table6` |
| C | 0.577 | 0.994 | 0.977 | 0.970 | `make table6` |

Profile B is the training condition; A and C are out-of-distribution bounds. The high threshold favours precision (almost no clear tile is discarded) at the cost of recall, so cloudy tiles can still be downlinked.

The FP32 model, at its own validation threshold 0.2945, reaches recall 0.852 / precision 0.996 on test B. The delivered reports for it are in `../nina_cloud_results/evaluation/fp32/` (see `ERRATA.md` #3 about one of them).

## Quantization and thresholds

Post-training full-integer INT8 (no QAT), calibrated with 300 random TRAIN B tiles. Input and output are int8 (input scale 1.0, zero point −128; output scale 0.00390625, zero point −128).

Both thresholds were chosen on validation B alone as the highest threshold with recall ≥ 0.95: FP32 0.2945 and INT8 **0.9336** (the paper's t\* = 0.934). The test data was not used for tuning.

No acceptance-gate verdict (recall drop, SQNR, KL) is recorded in the delivered cloud files, so none is claimed here.

## Known limitations

- **Synthetic data only.** Profile multipliers are assumptions (T3).
- **Device preprocessing is not equivalent to training (C-19).** On the board the image goes through RGB565 and a runtime resize, while training saw full-depth tiles. Table 8 quantifies the resulting fidelity.
- **Recall depends strongly on the profile** (0.58 to 0.86), so the figure for profile B should not be quoted as the on-orbit recall.
- The code that trained the model was delivered as files; the original commit is unknown.
- The PyTorch pipeline in `nina_cloud/` (α = 1.0, QAT) is a different implementation and did not produce this model.

## Ladder status

V0–V3 for this model on host data; V5 on the surrogate (Tables 7 and 8, Test 2). V4, V6 and V7 are deferred: see `../docs/evidence-and-ladder.md` and `../docs/deferred-protocols/`.

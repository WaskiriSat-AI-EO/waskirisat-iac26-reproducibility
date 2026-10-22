# Model card: NINA thermal fire-activity classifier (v1, frp20000)

| | |
|---|---|
| Owner | Osvaldo (WaskiriSat AI-EO) |
| Version | v1, trained 2026-08-30 (the model in the paper) |
| Architecture | MobileNetV3-Small, ImageNet init, 1 logit; 927,585 parameters |
| Input | 120 × 160 grayscale brightness temperature (Lepton geometry), duplicated to 3 channels |
| FP32 checkpoint | `mobilenetv3s_fire_best.pt`, sha256 `fe7ff10f…a52e0` |
| INT8 artifact | `mobilenetv3s_fire_int8.tflite`, sha256 `6e486fc7…01e8`, 1.39 MB, the file tested in HIL |
| Config | [`configs/paper.yaml`](configs/paper.yaml); reports in `../models/thermal/frp20000/` |
| Licence | **Proposal, for the supervisor to confirm:** code BSD-3-Clause (repo `LICENSE`); weights CC BY 4.0 |

## Intended use

Flags a 640 × 480 km scene with **high total fire activity**: summed FIRMS fire radiative
power (FRP) ≥ 20,000 MW (20 GW) within ±1 day. It is a scene-level trigger for downlink
prioritization on a small satellite. It is **not** an active-fire detector: it does not
locate or count individual fires, and a single fire is far below what a 640 × 480 km
scene at this resolution can show.

Not intended for: fire perimeter or hotspot mapping, emergency alerting, or any use
outside the training region (eastern Bolivia and its neighbours, dry season).

## Performance (evidence tier T2: host evaluation, not on board)

Test split: 82 patches from 21 granules, 36 fire / 46 no-fire, profile B (nominal sensor
degradation). 90% confidence intervals use a granule-level cluster bootstrap.

| model | threshold | accuracy | recall | precision | specificity | F1 | ROC-AUC [90% CI] | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| FP32 | 0.456 | 0.720 | 0.778 | 0.651 | 0.674 | 0.709 | 0.787 [0.644, 0.943] | 0.702 |
| INT8 (HIL artifact) | 0.2875 | 0.634 | 0.972 | 0.547 | 0.370 | 0.700 | 0.694 [0.508, 0.943] | 0.539 |

Thresholds are the highest validation threshold with recall ≥ 0.95. The HIL firmware
uses a **fixed threshold of 0.5**, not 0.2875. Table 8 (board vs. PC) compares both at 0.5.

Robustness (FP32 ROC-AUC, same test patches): profile A 0.806, B 0.787, C 0.759.
Profiles A and C are out-of-distribution bounds; the model was trained on B only.

Reproduce: `README.md` → Run. `evaluation_summary.json` and `predictions.json` hold every number above.

## INT8 acceptance gate: FAILED, not released for flight

The deployment gate compares FP32 and INT8 on validation (36 patches, 11 granules, 17 fire):

| criterion | tolerance | measured | result |
|---|---|---|---|
| recall drop | ≤ 0.005 | 0.000 | pass |
| F-beta drop (β = √3) | ≤ 0.010 | 0.047 | **fail** |
| score SQNR | ≥ 30 dB | 11.0 dB | **fail** |
| score KL divergence | ≤ 0.05 nats | 3.90 nats | **fail** |

The gate was introduced on 2026-09-02, after this artifact was exported (2026-08-30). It was
computed retroactively on 2026-10-22 (`quantization/pipelines/run_gate.py`). The artifact
is published because it is the one validated in hardware-in-the-loop tests. It is
**not** released for flight.

## Limitations

1. **Labels (v1).** The no-fire class means "zero FIRMS detections in the footprint". Two
   defects in the v1 FIRMS query make many negatives mean "no data" instead:
   - the queried bbox (lon −67.5…−59.7, lat −20.5…−11.8) is smaller than the patch footprints;
   - June 2023 has no FIRMS file, and June 2023 scenes supply **all 46 test negatives**.

   Every positive is from August or September. Part of what the model separates may
   therefore be season or region, not fire activity. See `DATASET_CARD.md`.
2. **Warm-scene confound.** Large fire complexes happen on hot, dry days. At the 20 GW
   threshold, a warm, dry scene without fire cannot be excluded as the thing the model
   detects.
3. **INT8.** The v1 "QAT" never trained against quantization noise: the quantizer ran
   with its post-training configuration and the conv+BN fold failed silently (found
   2026-10-08). This explains most of the FP32 → INT8 loss (ROC-AUC 0.787 → 0.694) and
   the failed gate. QAT is also not bit-reproducible on GPU: re-quantizing the same
   checkpoint gave a different file (`../models/thermal/frp20000/REQUANTIZATION_NOTE.md`).
4. **Small test set.** 82 patches from 21 granules, so intervals are wide.

Limitations 1 and 3 were found after the paper's run. They are fixed in the later v3/v4
models: wide FIRMS bbox, full date coverage, low-activity negatives, real QAT. These are
documented in the v3 training report (`docs/TRAINING_REPORT_v3.md`, pending merge to
`main`). Those models are not the model in the paper.

## Data and attribution

Training data: see [`DATASET_CARD.md`](DATASET_CARD.md). Contains modified Copernicus
Sentinel data [2023–2025]. Fire labels use NASA FIRMS (VIIRS S-NPP 375 m active fires).

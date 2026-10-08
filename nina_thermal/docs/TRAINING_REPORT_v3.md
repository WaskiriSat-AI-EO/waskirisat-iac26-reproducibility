# NINA Thermal — Fire Classifier Training Report v3

**WaskiriSat / NINA payload — fire detection model**
**Date:** 2026-10-03 · **Updated:** 2026-10-08 (int8 fixed, §8–§9) · **Supersedes:**
`TRAINING_REPORT.md` (v1, 2026-08-31) for all model metrics · **Artifacts:**
`models/nina_thermal_frp20000_v3_{b,bc,bc_r90,bc_qat_sel}/`
**Recommended model:** fp32 `models/nina_thermal_frp20000_v3_bc/mobilenetv3s_fire_best.pt`;
int8 `models/nina_thermal_frp20000_v3_bc_qat_sel/mobilenetv3s_fire_int8.tflite`

---

## 1. Executive summary

The model was retrained on **7 acquisition roots** (337 Sentinel-3 granules,
Aug 2022 – Sep 2025) after **three label-construction bugs** were found and fixed.
Those bugs had turned most of the "no-fire" class into patches FIRMS was simply
never queried for. They also affected v1, so **the v1 headline (test ROC-AUC 0.787)
is not a valid measure of fire detection** and should not be cited.

With corrected labels, the recommended model (**v3_bc**: MobileNetV3-Small,
trained on degradation profiles B + C) reaches on the granule-disjoint test split
(profile B, n = 189):

| accuracy | recall | specificity | precision | F1 | ROC-AUC (90% CI) | PR-AUC |
|---|---|---|---|---|---|---|
| **0.894** | **0.940** | **0.858** | **0.839** | **0.886** | **0.959 [0.920, 0.989]** | **0.945** |

The decision threshold now transfers across sensor-degradation profiles. Profile C
(worst case) specificity goes from 0.27 to 0.88, closing v1 open issue #3.

**int8 model fixed (2026-10-08, §8):** the quantization stage had never actually
run quantization-aware training — a config bug plus a torch/torchao
incompatibility made "QAT" fine-tune against plain observers. With real QAT the
int8 artifact (1.237 MB) matches fp32 on the test set, profile B: **accuracy
0.905 vs 0.894, recall 0.940 vs 0.940, ROC-AUC 0.959 vs 0.959** (previous int8:
0.540 / 0.904 / 0.717). Two of four acceptance-gate checks still fail (SQNR, KL),
for reasons discussed in §8.5.

**Still open:** scene mean temperature alone reaches test ROC-AUC 0.903. The CNN
beats it inside months that contain both classes (0.889 vs 0.843), which is the
first evidence it learns more than "hot scene", but the warm-scene confound is
reduced, not eliminated (§7).

---

## 2. What changed since v1

### 2.1 New data

| root | months | granules | role |
|---|---|---|---|
| `data/` | 2023-06, 2023-08 | 150 | existing |
| `data_202408/` | 2024-08 | 50 | existing (strict subset of `data/`) |
| `data_202409/` | 2024-09 | 50 | existing |
| `data_202509/` | 2025-09 | 50 | existing |
| **`data_202208/`** | 2022-08 | 46 | new, peak fire season |
| **`data_202302/`** | 2023-02 (1–25 Feb only) | 57 | new, wet season |
| **`data_202309/`** | 2023-09 | 54 | new, late fire season |

The three new roots ran the full datagen pipeline (extraction → processing →
simulation → split → balance). After de-duplication the merged pool has
**1,785 unique patches from 337 granules** (v1: 924 / 201).

Profile-C images were additionally rendered for the train/val splits of the three
new roots (`run_simulation --profiles C`, split verified identical to the
original), so that B+C training is possible on all roots.

### 2.2 Label bugs found and fixed

Labels come from NASA FIRMS: a patch is **fire** when summed Fire Radiative Power
(FRP) over its 640 × 480 km footprint is ≥ 20,000 MW within ±1 day. In v1 it was
**no-fire** when the footprint held zero FIRMS detections.

| # | bug | effect | fix |
|---|---|---|---|
| 1 | **FIRMS bbox too small.** CSVs covered lon −67.5…−59.7, lat −20.5…−11.8; patch footprints reach lon −76.2…−52.7, lat −28.2…−5.6 | 70% of v1/v2 "no-fire" patches lay **entirely outside** the queried area: zero detections meant "not asked", not "no fire". The model largely learned *lowlands vs. Andes* | Re-fetched FIRMS for all 7 roots over `-77,-29,-52,-5`; each root now declares its queried bbox in `firms/coverage.json`; `frp_labels` marks footprints < 90% inside it as `no_frp_data` |
| 2 | **Missing dates.** No FIRMS CSV for June 2023; Suomi-NPP VIIRS offline 2022-07-20…08-10 | every patch on those dates became a "clean negative" | June 2023 fetched; outage days filled from `VIIRS_NOAA20_SP`; `frp_labels` marks dates with no detection anywhere within ±1 day as `no_frp_data` |
| 3 | **NRT and SP products mixed** for Aug 2023 / Aug 2024 | rows differ slightly, so de-duplication missed them and FRP was double-counted | one product per date (`VIIRS_SNPP_SP`, NOAA-20 for the outage); old CSVs kept in `firms_old_narrow/` |

### 2.3 New negative-class definition

With full FIRMS coverage, **only 7 of 1,785 patches have zero detections**: a
300,000 km² footprint in South America almost always contains at least one small
fire. The zero-detection rule cannot produce a negative class at this footprint
size. v3 uses a **low-activity negative**:

- **fire:** summed FRP ≥ 20,000 MW
- **no-fire:** summed FRP < 1,000 MW (new flag `--frp_negative_max 1000`)
- **ambiguous (excluded):** everything in between

Summed FRP per patch spans five orders of magnitude (5th percentile 32 MW,
median 3.1 GW, 95th percentile 113 GW), and the gap between 1 GW and 20 GW keeps
the two classes physically distinct.

### 2.4 Code changes

| file | change |
|---|---|
| `training/frp_labels.py` | coverage guards (dates and bbox); `negative_max_mw` |
| `train_thermal`, `run_baseline`, `run_evaluate`, `run_quantize` | `--frp_negative_max` (must match across all four) |
| `quantization/qat.py` | model switched to eval mode before `convert_pt2e`; known issue documented |
| `training/fixtures.py`, `tests/test_frp_labels.py` | realistic fixture FIRMS; 4 new tests |
| `docs/RUNBOOK_acquire_low_season.md` | FIRMS step; `--roots` on every command |

Test suite: **274 / 274 passing**.

---

## 3. Dataset

Profile B, fixed AGC, corrected labels (patch counts; B+C training doubles the
image counts with the same patches):

| split | patches | granules | fire | no-fire | prevalence |
|---|---|---|---|---|---|
| train | 566 | 170 | 183 | 383 | 0.323 |
| validation | 128 | 38 | 50 | 78 | 0.391 |
| test | 189 | 49 | 83 | 106 | 0.439 |
| **total** | **883** | 257 | **316** | **567** | |

902 patches fall in the ambiguous band and are excluded. Splits are
granule-disjoint, reconciled across roots (19 granules had conflicting splits,
58 patches reassigned).

Compared with v1 (219 / 36 / 82 images), the training set is 2.6× larger and the
validation set 3.6× larger, which also narrows every confidence interval.

**Labels by month** (all splits). Six of eight months contain both classes;
under the v1 labels almost every month was single-class:

| month | fire | no-fire |
|---|---|---|
| 2022-08 | 39 | 73 |
| 2023-02 | 0 | 295 |
| 2023-06 | 0 | 105 |
| 2023-08 | 3 | 16 |
| 2023-09 | 64 | 22 |
| 2024-08 | 98 | 20 |
| 2024-09 | 107 | 16 |
| 2025-09 | 5 | 20 |

---

## 4. Model and training

Architecture and recipe are unchanged from v1 (see `TRAINING_REPORT.md` §4):
MobileNetV3-Small, single-logit head, 927,585 parameters, 160 × 120 input,
ImageNet init, 20 head epochs (lr 1e-3) + 15 fine-tune epochs (lr 1e-4),
`pos_weight` 3.0 (β = 1.732), checkpoint selection on validation ROC-AUC,
geometric augmentation only, fixed AGC, seed 42.

| model | training profiles | recall target | selected threshold | best checkpoint |
|---|---|---|---|---|
| v3_b | B | 0.95 | 0.3588 | fine-tune epoch 5 |
| **v3_bc** | **B + C** | **0.95** | **0.5164** | fine-tune epoch 9 |
| v3_bc_r90 | B + C | 0.90 | 0.6409 | fine-tune epoch 9 |

**v3_bc and v3_bc_r90 are the same network** (training is deterministic; the
checkpoints are identical). They differ only in the decision threshold swept on
validation, so they are two operating points of one model.

Reproduce:

```bash
ROOTS="data data_202408 data_202409 data_202509 data_202208 data_202302 data_202309"
OUT=models/nina_thermal_frp20000_v3_bc
python -m nina_thermal.training.train_thermal --frp_threshold 20000 --frp_negative_max 1000 \
    --roots $ROOTS --profiles B C --recall_target 0.95 --out_dir $OUT --num_workers 2
python -m nina_thermal.evaluation.pipelines.run_evaluate --checkpoint $OUT/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 --frp_negative_max 1000 --roots $ROOTS --out_dir $OUT --figures
```

---

## 5. Results — validation set

| model | ROC-AUC | PR-AUC | recall | precision | specificity | F-beta | TP / FP / FN / TN |
|---|---|---|---|---|---|---|---|
| v3_b (B) | 0.943 | 0.875 | 0.960 | 0.738 | 0.782 | 0.893 | 48 / 17 / 2 / 61 |
| **v3_bc (B+C)** | **0.934** | **0.874** | **0.950** | **0.725** | **0.769** | **0.882** | **95 / 36 / 5 / 120** |
| v3_bc_r90 (B+C) | 0.935 | 0.878 | 0.900 | 0.804 | 0.859 | 0.874 | 90 / 22 / 10 / 134 |

B+C validation contains both profiles (256 images), so its counts are double those of v3_b.

---

## 6. Results — test set (granule-disjoint)

### 6.1 Profile B (nominal flight condition), n = 189 (83 fire / 106 no-fire)

| metric | v3_b | **v3_bc** | v3_bc_r90 |
|---|---|---|---|
| Accuracy | 0.868 | **0.894** | 0.894 |
| Recall | 0.976 | **0.940** | 0.892 |
| Specificity | 0.783 | **0.858** | 0.896 |
| Precision | 0.779 | **0.839** | 0.871 |
| F1 | 0.866 | **0.886** | 0.881 |
| F-beta (β = 1.732) | 0.918 | **0.912** | 0.886 |
| ROC-AUC | 0.956 | **0.959** | 0.959 |
| PR-AUC | 0.940 | **0.945** | 0.945 |

**Confusion matrices:**

```
v3_b            pred FIRE  pred NO-FIRE      v3_bc           pred FIRE  pred NO-FIRE
actual FIRE         81          2            actual FIRE         78          5
actual NO-FIRE      23         83            actual NO-FIRE      15         91

v3_bc_r90       pred FIRE  pred NO-FIRE
actual FIRE         74          9
actual NO-FIRE      11         95
```

**90% granule-level bootstrap CIs** (n = 1000):

| metric | v3_b | **v3_bc** | v3_bc_r90 |
|---|---|---|---|
| Recall | [0.935, 1.000] | **[0.889, 0.986]** | [0.833, 0.948] |
| Precision | [0.649, 0.882] | **[0.708, 0.941]** | [0.740, 0.973] |
| F-beta | [0.861, 0.957] | **[0.855, 0.955]** | [0.829, 0.936] |
| ROC-AUC | [0.910, 0.992] | **[0.920, 0.989]** | [0.919, 0.989] |

**Operational precision** (Bayes projection at an assumed fire prevalence; v3_bc):
0.42 at 10% prevalence, 0.87 at 50%, 0.98 at 90%. If fire scenes are rare in
operations, roughly half of alarms will be false at this operating point.

### 6.2 Cross-profile robustness (test set)

| model | profile | accuracy | recall | specificity | ROC-AUC | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| v3_b | A | 0.783 | 0.964 | 0.642 | 0.888 | 80 / 38 / 3 / 68 |
| v3_b | C | 0.587 | 0.988 | **0.274** | 0.914 | 82 / 77 / 1 / 29 |
| **v3_bc** | **A** | **0.873** | **0.831** | **0.906** | **0.937** | **69 / 10 / 14 / 96** |
| **v3_bc** | **C** | **0.910** | **0.952** | **0.877** | **0.962** | **79 / 13 / 4 / 93** |
| v3_bc_r90 | A | 0.810 | 0.639 | 0.943 | 0.939 | 53 / 6 / 30 / 100 |
| v3_bc_r90 | C | 0.899 | 0.892 | 0.906 | 0.959 | 74 / 10 / 9 / 96 |

Training on B + C fixes the threshold transfer to C. Profile A (post-FFC best
case, never seen in training) now loses recall instead (0.83), because A images
score lower than B/C. A per-profile threshold, or adding A to training, would
close that gap.

### 6.3 Comparison with v1 and v2

| | v1 | v2 | **v3_bc** |
|---|---|---|---|
| labels | narrow-bbox FIRMS, zero-detection negatives | same | full-coverage FIRMS, < 1 GW negatives |
| roots | 4 | 7 | 7 |
| test patches (B) | 82 | 144 | 189 |
| test ROC-AUC | 0.787 | 0.858 | **0.959** |
| test specificity | 0.674 | 0.520 | **0.858** |
| test recall | 0.778 | 1.000 | **0.940** |
| profile-C specificity | — | 0.010 | **0.877** |

v1 and v2 numbers are not comparable in a strict sense: their negative class was
mostly geography (bug #1), so they measured a different and easier-to-game task.

---

## 7. Is it detecting fire, or hot scenes?

### 7.1 Feasibility baseline (`run_baseline`, train + val only, grouped 5-fold CV)

Logistic regression on hand-crafted brightness-temperature statistics, corrected labels:

| | value |
|---|---|
| n (patches) | 694 (fire fraction 0.336) |
| grouped-CV ROC-AUC | **0.925 ± 0.039** |
| `bt_mean` alone | 0.886 |
| `bt_p50` | 0.890 |
| `top_pixel_excess` (local hotspot) | 0.261 |
| `grad_mean` (local gradients) | 0.440 |

As in v1, all predictive power comes from **whole-scene warmth**; local hotspot
features sit at or below chance. This is the physical limit documented in v1 §2:
a single fire is invisible at 4 km GSD in the 8–14 µm band. The task the sensor
supports is detecting **high total fire activity in the field of view**.

### 7.2 CNN vs. scene mean temperature on the same test patches

| | ROC-AUC |
|---|---|
| scene mean temperature alone (all test patches) | 0.903 |
| v3_bc CNN (all test patches) | 0.959 |
| scene mean, **within-month** (5 mixed months, month-centred) | 0.843 |
| v3_bc CNN, **within-month** | **0.889** |
| v3_b CNN, within-month | 0.918 |

Inside a single month (same season, similar weather), the CNN still ranks fire
above no-fire better than scene temperature does. That is the first evidence
that it uses more than "hot day". The margin is real but modest, and rests on
few patches per month.

### 7.3 Test results by month (v3_bc, profile B)

| month | n | fire | CNN AUC | scene-mean AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 2022-08 | 28 | 13 | 0.964 | 0.918 | 13 | 1 | 0 | 14 |
| 2023-02 | 50 | 0 | — | — | 0 | 2 | 0 | 48 |
| 2023-06 | 26 | 0 | — | — | 0 | 12 | 0 | 14 |
| 2023-08 | 4 | 3 | 1.000 | 1.000 | 3 | 0 | 0 | 1 |
| 2023-09 | 12 | 8 | 1.000 | 0.750 | 6 | 0 | 2 | 4 |
| 2024-08 | 39 | 34 | 1.000 | 0.841 | 33 | 0 | 1 | 5 |
| 2024-09 | 24 | 24 | — | — | 22 | 0 | 2 | 0 |
| 2025-09 | 6 | 1 | 1.000 | 1.000 | 1 | 0 | 0 | 5 |

**June 2023 is the main source of false alarms (12 of 15).** It is early fire
season: warm and dry, but under 1 GW of activity in these footprints. That is
exactly the confound case, and the clearest target for further work.

---

## 8. Quantization

### 8.1 Root cause (found 2026-10-08)

The int8 failure reported on 2026-10-03 (test accuracy 0.540) was **not** a
property of the model and not mainly a conversion defect. The quantization
stage had **never performed quantization-aware training**:

| # | defect | effect | fix |
|---|---|---|---|
| 1 | `build_quantizer()` never passed `is_qat=True` to the litert quantizer | the "QAT" fine-tune used the PTQ config: plain observers, no fake-quantization, no QAT conv+BN patterns. The model never trained against quantization noise | `build_quantizer(qat=True)` for QAT; regression test |
| 2 | `torchao` 0.17 passes plain functions to `torch.export.export`, which torch 2.12 rejects | the real QAT config could not be built at all (likely why defect 1 went unnoticed), and the conv+BN fold at convert time silently failed: 34 BatchNorm nodes survived conversion | compatibility shim in `qat.py` wrapping patterns in an `nn.Module`; regression test asserting 0 BN nodes after convert |

**Correction to the 2026-10-03 version of this section:** the "QAT fake-quant
ROC-AUC 0.912, corr 0.987" figure was measured on that observer-only graph, so
it was not a simulation of int8 at all. With genuine fake-quantization the
untrained simulated model scores AUC 0.50–0.79: MobileNetV3 loses most of its
accuracy at int8 without QAT (per-layer error 6% at the first conv, 19% at the
first depthwise conv, compounding), which is exactly what QAT is meant to fix.

Hypotheses from the debugging plan that were tested and **ruled out** as the
main cause: in-place operations, squeeze-excite blocks (quantized or kept
float), hardswish/hardsigmoid (never quantized by this quantizer), degenerate
observer ranges.

### 8.2 Observer leak (found and fixed 2026-10-08)

QAT fake-quant modules keep updating their activation ranges whenever data
passes through, in eval mode too. Scoring validation therefore (a) leaked
validation statistics into the quantization ranges and (b) changed the model
while it was being measured, so the converted artifact differed from the
checkpoint that was scored (same weights: 0.913 in-loop vs 0.902 after
conversion). `qat.freeze_observers()` now freezes the ranges before any
scoring or conversion. This raised converted-vs-fake-quant score correlation
from ~0.90 to **0.978** and removed most of the "conversion gap".

### 8.3 Result — final pipeline

`run_quantize` on v3_bc: QAT 40 epochs, lr 5e-5, batch 16, B + C, seed 42;
observers frozen for scoring; **checkpoint chosen by real-int8 validation
ROC-AUC** every 5 epochs (selected: epoch 25 of 40). Per-channel symmetric
int8 weights, per-tensor uint8 activations; float32 input / int8 output.
Artifact `models/nina_thermal_frp20000_v3_bc_qat_sel/mobilenetv3s_fire_int8.tflite`,
**1.237 MB**, SHA-256 `cc671cb1…0663d58`. int8 threshold 0.7648, re-swept on
int8 validation scores for recall ≥ 0.95 (fp32: 0.5167).

**Validation (B + C, 256 images: 100 fire / 156 no-fire)**

| metric | fp32 | int8 before fix | **int8 final** |
|---|---|---|---|
| Accuracy | 0.840 | 0.531 | **0.824** |
| Recall | 0.950 | 0.950 | **0.950** |
| Specificity | 0.769 | 0.263 | **0.744** |
| Precision | 0.725 | 0.452 | **0.704** |
| F1 | 0.823 | 0.613 | **0.809** |
| F-beta (β = 1.732) | 0.882 | 0.745 | **0.874** |
| ROC-AUC | 0.934 | 0.768 | **0.907** |
| PR-AUC | 0.874 | 0.643 | **0.809** |
| TP / FP / FN / TN | 95 / 36 / 5 / 120 | 95 / 115 / 5 / 41 | **95 / 40 / 5 / 116** |

**Test set (189 patches per profile: 83 fire / 106 no-fire)** — from
`run_evaluate --tflite` (`models/nina_thermal_frp20000_v3_bc_qat_sel/evaluation_summary.json`),
each model at its own validation threshold:

| profile | model | accuracy | recall | specificity | precision | F1 | ROC-AUC | PR-AUC | TP / FP / FN / TN |
|---|---|---|---|---|---|---|---|---|---|
| A | fp32 | 0.873 | 0.831 | 0.906 | 0.873 | 0.852 | 0.937 | 0.910 | 69 / 10 / 14 / 96 |
| A | **int8** | **0.820** | 0.699 | 0.915 | 0.866 | 0.773 | **0.945** | 0.913 | 58 / 9 / 25 / 97 |
| **B** | fp32 | 0.894 | 0.940 | 0.858 | 0.839 | 0.886 | 0.959 | 0.945 | 78 / 15 / 5 / 91 |
| **B** | **int8** | **0.905** | **0.940** | 0.877 | 0.857 | 0.897 | **0.959** | 0.939 | 78 / 13 / 5 / 93 |
| C | fp32 | 0.910 | 0.952 | 0.877 | 0.859 | 0.903 | 0.962 | 0.943 | 79 / 13 / 4 / 93 |
| C | **int8** | **0.884** | 0.964 | 0.821 | 0.808 | 0.879 | **0.944** | 0.896 | 80 / 19 / 3 / 87 |

```
int8, test profile B   pred FIRE  pred NO-FIRE
actual FIRE                78          5
actual NO-FIRE             13         93
```

int8 test profile B, 90% granule-level bootstrap CIs: recall [0.870, 0.978],
precision [0.724, 0.952], F-beta [0.847, 0.954], ROC-AUC [0.916, 0.989].

**On the flight profile the int8 model equals fp32**: same recall (0.940, same
78 / 5 fire split), two fewer false alarms, identical ROC-AUC. Profile C keeps
recall (0.96) at the cost of specificity. **Profile A loses recall** (0.70 vs
0.83): A is never seen in training and the higher int8 threshold cuts its
lower-scoring fires, even though its ranking (ROC-AUC 0.945) is intact. A
per-profile threshold or adding A to training would recover it (open issue #4).

### 8.4 Run-to-run stability

QAT is not bit-reproducible on GPU. Earlier runs (same settings, before the
observer freeze and checkpoint selection, so with the leak of §8.2):

| run | val int8 ROC-AUC | test B accuracy | test B ROC-AUC | test A accuracy | test C accuracy |
|---|---|---|---|---|---|
| seed 42 | 0.915 | 0.910 | 0.960 | 0.873 | 0.884 |
| seed 1 | 0.865 | 0.884 | 0.959 | 0.831 | 0.873 |
| seed 2 | 0.906 | 0.905 | 0.962 | 0.852 | 0.889 |
| leaky run with selection | 0.898 | 0.862 | 0.913 | 0.847 | 0.873 |
| **final pipeline (§8.3)** | **0.907** | **0.905** | **0.959** | **0.820** | **0.884** |
| fp32 | 0.934 | 0.894 | 0.959 | 0.873 | 0.910 |

Profile B lands at accuracy 0.86–0.91 / ROC-AUC 0.91–0.96 across runs; the
leaky pipeline produced the one clearly weaker run. The final artifact was
chosen by the pipeline on validation only. A 100-epoch run at lr 1e-4 did not
improve validation AUC (0.900): longer QAT overfits.

### 8.5 Acceptance gate

| check | tolerance | measured | verdict |
|---|---|---|---|
| Recall drop | ≤ 0.005 | 0.000 | pass |
| F-beta drop | ≤ 0.010 | 0.008 | pass |
| SQNR (int8 vs fp32 scores) | ≥ 30 dB | 9.2 dB | **fail** |
| Score-distribution KL | ≤ 0.05 nats | 2.49 nats | **fail** |

SQNR and KL compare int8 scores with the **original fp32 checkpoint's** scores.
That is the right test after post-training quantization, where the weights are
unchanged. After QAT the weights are fine-tuned, so the scores legitimately
shift even when decisions are as good, as the operating-point checks and the
test set show. Proposed change for decision: after QAT, measure SQNR/KL
between the int8 artifact and the fake-quant graph it was converted from
(conversion fidelity), and keep recall/F-beta drop against fp32 as the
accuracy gate. Also, the KL estimate rests on 256 scores in 64 bins, which is
unstable (v1 §6.2).

### 8.6 Remaining conversion gap

`run_quantize` records `conversion_parity`: with observers frozen, the
converted graph's validation scores correlate **0.978** with the fake-quant
graph's (fake-quant ROC-AUC 0.909, converted 0.909, TFLite 0.907). Rounding
alone should give > 0.99, so a small residual mismatch remains; it does not
affect accuracy measurably. The artifact also still has a float32 input
boundary (open issue #6).

---

## 9. Debugging plan — status

Plan: `docs/PLAN_int8_fix.md`.

| phase | status | outcome |
|---|---|---|
| 0. Diagnostic harness | **done** | `quantization/diagnostics.py`, `run_quant_diagnose` CLI |
| 1. Isolate defect | **done** | in-place ops, squeeze-excite, hardswish, degenerate ranges ruled out; real cause = no QAT (§8.1) |
| 2. BatchNorm folding | **done** | torch 2.12 shim; 0 BN nodes after convert |
| 3. `ai-edge-quantizer` route | not needed for accuracy | still the path to int8 input/output (full-NPU graph) |
| 4. Harden pipeline | **done** | conversion-parity check; observer freeze (§8.2); checkpoint selection by real-int8 val ROC-AUC (`--select_every`); int8 test evaluation in `run_evaluate --tflite`; regression tests |
| 5. Fallback backbone | not needed | |
| 6. Validate and report | **done** | this section |

Test suite: **277 / 277 passing**.

Reproduce:

```bash
ROOTS="data data_202408 data_202409 data_202509 data_202208 data_202302 data_202309"
OUT=models/nina_thermal_frp20000_v3_bc_qat_sel
cp models/nina_thermal_frp20000_v3_bc/training_report.json $OUT/   # fp32 threshold for run_evaluate
python -m nina_thermal.quantization.pipelines.run_quantize \
    --checkpoint models/nina_thermal_frp20000_v3_bc/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 --frp_negative_max 1000 --roots $ROOTS --profiles B C \
    --out_dir $OUT --epochs 40 --lr 5e-5 --batch_size 16 --num_workers 2 \
    --select_every 5 --no_smoke_test
python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint models/nina_thermal_frp20000_v3_bc/mobilenetv3s_fire_best.pt \
    --tflite $OUT/mobilenetv3s_fire_int8.tflite \
    --frp_threshold 20000 --frp_negative_max 1000 --roots $ROOTS --out_dir $OUT
```

---

## 10. Open issues

| # | issue | impact | proposed action |
|---|---|---|---|
| 1 | int8 gate: SQNR/KL fail after QAT (§8.5); small conversion gap (§8.6) | gate not formally passed although test accuracy matches fp32 | decide gate definition for QAT |
| 2 | Warm-scene confound reduced but not removed (§7) | part of the signal may be season | acquire more mixed-class months (Sep–Oct 2021–2023); month-stratified splits; report within-month AUC as a headline |
| 3 | June 2023 false alarms | 12 of 15 test FPs | more early-season scenes with both classes |
| 4 | Profile A recall 0.83 with B+C training | best-case sensor condition loses recall | per-profile threshold, or train A + B + C |
| 5 | Feb 2023 covers only 1–25 Feb (`--max_products 60` cap) | no Mar–Apr wet-season scenes | re-run acquisition for Mar–Apr 2023 |
| 6 | Float32 graph boundary (v1 issue #2) | NPU cannot run the whole graph | the `ai-edge-quantizer` route fixes this once its calibration is solved |
| 7 | Negative threshold 1 GW chosen by data inspection, not swept | sensitivity unknown | sweep 500 / 1,000 / 2,000 MW and report |

### Deployment prerequisite (unchanged)

The flight Lepton must use the same **fixed AGC** as training (258.15–358.15 K →
0–255). In auto-gain mode pixel values stop mapping to absolute temperature and
the model's basis collapses.

---

## 11. Honest limitations

1. **What the model detects:** large total fire activity in a 640 × 480 km scene
   (≥ 20 GW summed FRP vs < 1 GW), not individual fires. This is a sensor limit,
   not a model limit.
2. **Test set is 189 patches / 49 granules.** Intervals are much narrower than v1
   but still non-trivial (precision 90% CI [0.71, 0.94]).
3. **The 1 GW / 20 GW thresholds exclude 51% of patches.** Performance on the
   ambiguous middle band (moderate fire activity) is not measured.
4. **The int8 model is validated on host only** (TFLite interpreter on CPU);
   it has not run on OpenMV N6 hardware, and on-orbit latency remains a
   calculation (v1 §7).
5. **v1 results are superseded.** Its negative class was dominated by footprints
   outside the FIRMS query area, so its 0.787 measured mainly geography.

---

## 12. Artifacts

```
models/nina_thermal_frp20000_v3_bc/              (recommended)
├── mobilenetv3s_fire_best.pt        fp32 checkpoint — the valid model
├── training_report.json             provenance, history, val metrics
├── evaluation_summary.json          test metrics, CIs, per-profile, per-surface
├── predictions.json                 per-sample test predictions
├── figures/eval_profile_{a,b,c}.png
├── mobilenetv3s_fire_int8.tflite    OLD int8 from broken QAT — superseded, do not deploy
└── quantization_report.json         its gate report (FAILED)

models/nina_thermal_frp20000_v3_bc_qat_sel/      (recommended int8)
├── mobilenetv3s_fire_int8.tflite    final pipeline, 1.237 MB (§8.3)
├── mobilenetv3s_fire_int8.sha256
├── quantization_report.json         val metrics, conversion_parity, checkpoint_selection, gate
├── evaluation_summary.json          fp32 + int8 test metrics, CIs, per-profile
└── predictions.json                 per-sample fp32 and int8 scores

models/nina_thermal_frp20000_v3_bc_qat/          first real-QAT run, pre-freeze (§8.4)
models/nina_thermal_frp20000_v3_bc_qat_s{1,2}/   stability runs (§8.4)
models/nina_thermal_frp20000_v3_bc_qat100/       100-epoch run, not better (§8.4)

models/nina_thermal_frp20000_v3_b/      B-only variant
models/nina_thermal_frp20000_v3_bc_r90/ same network as v3_bc, threshold for recall 0.90
```

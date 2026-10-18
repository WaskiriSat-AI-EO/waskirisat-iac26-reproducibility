# NINA Thermal — Fire Classifier Training Report v4

**WaskiriSat / NINA payload — fire detection model**
**Date:** 2026-10-18 · **Supersedes:** `TRAINING_REPORT_v3.md` for model metrics (v3's
label and QAT fixes still apply and are not repeated here) · **Branch:**
`feat/v4-data-expansion` · **Runbook:** `docs/RUNBOOK_v4_data_expansion.md`
**Models:** fp32 `models/nina_thermal_frp20000_v4_abc/mobilenetv3s_fire_best.pt`;
int8 `models/nina_thermal_frp20000_v4_abc_qat_sel/mobilenetv3s_fire_int8.tflite`
(1.237 MB, SHA-256 `954388ef…66aa6bfe`)

---

## 1. Executive summary

v4 trains the v3 recipe (MobileNetV3-Small, 160 × 120, FRP labels ≥ 20 GW fire /
< 1 GW no-fire) on **4.7× more labeled data**: 31 acquisition roots, 2,152
granules, 2019–2025, **4,187 labeled patches** (v3: 883), with profiles A + B + C
in training (v3: B + C).

| | v3_bc | **v4** |
|---|---|---|
| labeled patches (fire / no-fire) | 883 (316 / 567) | **4,187 (2,025 / 2,162)** |
| months with both classes (≥ 5 each) | 5 of 8 | **21 of 27** |
| years | 2022–2025 | **2019–2025** |
| **frozen v3 test, profile B**: fp32 ROC-AUC | 0.959 | **0.963** |
| **v4 test (714 patches), profile B**: fp32 ROC-AUC | 0.912 | **0.945** |
| v4 test, B: recall at the model's own threshold | 0.805 | **0.958** |
| v4 test, B: specificity at 95% recall | 0.583 | **0.823** |
| int8 v4 test, B: ROC-AUC / recall | — | **0.935 / 0.961** |
| profile A recall, v3 test (int8) | 0.70 | **0.99** |

**Main result:** v3_bc does not generalize to unseen months and years. On the
larger v4 test set its recall drops to 0.80 (70 of 359 fires missed). v4 keeps
recall at 0.96 there, with far fewer false alarms at matched recall. On the
original v3 test set the two models rank equally well (CIs overlap); v4 is not
worse on what v3 was good at.

**Caveat:** within-month AUC did not improve (§6.4). v4's gains come from covering
more seasons and years, not from better fire discrimination inside one month.
The warm-scene confound (v3 §7) is reduced in effect, not solved.

int8 loses ~0.01 ROC-AUC and ~4 points of specificity against fp32, at equal
recall. The acceptance gate fails the same 2 checks as v3 (SQNR, KL), still
pending the gate decision (v3 open issue #1).

---

## 2. Data

### 2.1 What was added

| source | roots | granules | labeled patches (fire / no-fire) |
|---|---|---|---|
| v3 roots (unchanged) | 7 | 337 | 883 (316 / 567) |
| pilot: Sep 10–30 2024 | `data_202409b` | 50 | 144 (121 / 23) |
| recovered `datagen/data` (no download) | `data_extra` | 263 | 635 (77 / 558) |
| Batch 1: rest of v3 months | 7 (`*b`) | 465 | 817 (460 / 357) |
| Batch 2: July and October 2019–2025, Aug 2019 | 15 | 1,037 | 1,708 (1,051 / 657) |
| **total (after de-duplication)** | **31** | **2,152** | **4,187 (2,025 / 2,162)** |

Granule counts are granules that produced at least one patch inside the ROI.

Splits (granule-disjoint, reconciled across roots with precedence test > val > train):

| split | patches | granules | fire | no-fire | images (A / B / C) |
|---|---|---|---|---|---|
| train | 2,870 | 1,027 | 1,393 | 1,477 | 2,534 / 2,870 / 2,870 |
| val | 603 | 212 | 273 | 330 | 536 / 603 / 603 |
| test | 714 | 245 | 359 | 355 | 714 / 714 / 714 |

The 189 v3 test patches are a subset of the v4 test and unchanged (same 19
granules / 58 patches reconciled as in v3). New roots never contain a v3 granule
(checked by acquisition key, §3.4).

### 2.2 Labels by month (profile B)

| month | fire | no-fire | | month | fire | no-fire |
|---|---|---|---|---|---|---|
| 2019-07 | 3 | 35 | | 2023-07 | 13 | 99 |
| 2019-08 | 122 | 9 | | 2023-08 | 77 | 66 |
| 2019-10 | 57 | 29 | | 2023-09 | 154 | 34 |
| 2020-07 | 32 | 64 | | 2023-10 | 137 | 21 |
| 2020-10 | 153 | 11 | | 2024-01 | 0 | 98 |
| 2021-07 | 34 | 57 | | 2024-07 | 167 | 103 |
| 2021-10 | 121 | 31 | | 2024-08 | 317 | 59 |
| 2022-07 | 16 | 73 | | 2024-09 | 228 | 39 |
| 2022-08 | 100 | 105 | | 2024-10 | 159 | 35 |
| 2022-10 | 40 | 30 | | 2025-01 | 0 | 132 |
| 2023-01 | 0 | 170 | | 2025-07 | 25 | 124 |
| 2023-02 | 0 | 392 | | 2025-08 | 4 | 6 |
| 2023-06 | 3 | 219 | | 2025-09 | 33 | 40 |
| | | | | 2025-10 | 30 | 81 |

21 of 27 months have at least 5 patches of each class. v3 had 5 such months out
of 8. Per-root counts: `models/nina_thermal_frp20000_v4_abc/labels_by_root_month.json`.

### 2.3 Data-quality issues found and fixed during v4

| # | issue | effect if unfixed | fix |
|---|---|---|---|
| 1 | Acquisition took the **oldest** `--max_products` of a window | most v3 months cover only days 1–9 | `--sampling even` (round-robin per day) |
| 2 | `resources/bolivia_roi.geojson` is a **0.1° test box**, not Bolivia | `data_202208/02/09` got ~40% of available granules | runbooks use `bo_nina_roi_simplified.geojson`; acquisition refuses ROIs < 1° |
| 3 | `run_processing` without `--roi` | patches over Brazil/Paraguay, outside the FIRMS bbox | `--roi` required (as used for v3 roots) |
| 4 | `run_balance` default `fire_ratio 0.2` balances on the datagen pixel label, before the FRP relabel | dropped 1,029 of 1,274 `data_extra` patches | `--fire_ratio 0.01` for new roots (v3 roots untouched; only `data_202309` lost 257 patches in v3) |
| 5 | **Partial or missing S-NPP days** inside a FIRMS request passed the empty-request check | FRP undercount turns fire patches into false no-fire | per-day check (0 detections, or < 40% of NOAA-20); 9 outage spans added (§2.4) |
| 6 | **Same acquisition in two processing versions** (2019: `LN2_O_NT_003` and reprocessed `LR1_R_NT_004`) | the same scene twice, possibly in train and test | one product per acquisition (highest baseline, then reprocessed); 60 duplicates moved to `02_superseded/` |

### 2.4 FIRMS: S-NPP gaps filled from NOAA-20

The 7 v3 roots pass the per-day audit (`run_fetch_firms --check_only`) with no
flagged day, so v3 labels stand.

| dates | S-NPP vs NOAA-20 detections | roots |
|---|---|---|
| 2021-08-03 … 08-04 | 1,717 vs 6,152; 2,520 vs 7,812 | `data_202108` (not yet downloaded) |
| 2023-07-26 … 07-27 | 0 | `data_202307b` |
| 2023-11-01 | 362 vs 1,165 | `data_202310` |
| 2024-07-10 … 07-16 | 0 (07-16: 844 vs 4,566) | `data_extra`, `data_202407b` |
| 2024-07-24 … 07-28 | 0 (07-24: 782 vs 13,045) | `data_202407b` |
| 2024-09-26 … 09-27 | 0; 2,741 vs 7,073 | `data_202409b` (pilot relabelled: fire 115 → 121) |
| 2025-07-04 … 07-07 | 311–385 vs 1,134–1,317 | `data_extra`, `data_202507b` |
| 2025-10-30 | 161 vs 412 | `data_202510` |

(2022-07-20 … 08-10 was already handled in v3.) List: `config.FIRMS_SNPP_OUTAGES`.

### 2.5 Known data gaps in v4

- **Profile A missing for 703 train/val images** of `data_202208`, `data_202302`,
  `data_202309`. Those roots have A only for test. Rendering A for them was in
  the runbook (§6) and was skipped. The test sets are complete, so all metrics
  below are valid as measured.
- **5 Batch 2 roots not downloaded** (time limit): `data_201909`, `data_202008`,
  `data_202009`, `data_202108`, `data_202109`. Their FIRMS labels are already
  fetched and checked.
- 2019 keeps 98 granules with only the older baseline `O_NT_003` (83 are `R_NT_004`). Baseline
  differences in brightness temperature are not evaluated.

---

## 3. Acquisition pipeline (code changes)

| commit | change |
|---|---|
| `6c98b94` | even sampling, `--skip_inventory`, streamed extract + delete raw, `granule_inventory.txt`, resume, `--dry_run`; new `run_fetch_firms` |
| `bd0412a`, `38955aa` | retry dropped downloads (3 attempts, backoff) |
| `2f35669`, `7379cfc` | per-day FIRMS completeness check, `--check_only`; S-NPP outage list |
| `3e8557c` | `--download files` (default): only the 4 NetCDFs extraction reads, via the CDSE node API. **~14 MB / ~5–8 s per granule instead of ~350 MB / ~2 min**; extracted arrays bit-identical (12/12 granules checked) |
| `db88c90` | one CDSE token reused until near expiry. Per-granule logins at node speed locked the account once (HTTP 400/403) |
| `57ade94` | one product per acquisition (§2.3 #6) |

Tests: full suite **325 passed** (277 at the start of v4). Disk: raw granules are never kept; a 90-granule root
is ~2.5 GB after extraction.

Measured throughput: Batch 1 (580 granules) in 35 min, 3 roots in parallel;
Batch 2 at ~15 granules/min, slowing to ~6/min after 23:20 (CDSE side).

---

## 4. Model and training

Unchanged from v3 except data and profiles: MobileNetV3-Small, single logit,
927,585 parameters, 160 × 120 input, ImageNet init, 20 head epochs (lr 1e-3,
early stop after 10 without val ROC-AUC gain) + 15 fine-tune epochs (lr 1e-4),
`pos_weight` 3.0, batch 32, seed 42, fixed AGC, **profiles A + B + C**.

- Head phase stopped at epoch 18 (val ROC-AUC 0.842).
- Fine-tune best at epoch 14: **val ROC-AUC 0.917** (val = 1,742 images, A + B + C).
- Training time: 14 min on an RTX 4050 Laptop GPU.

v4 val ROC-AUC (0.917) is not comparable with v3's (0.934): the v4 val set is
2.4× larger and contains many more mixed months.

### 4.1 Operating threshold

The training default (val recall ≥ 0.95) gives threshold 0.293 and test
specificity of only 0.65–0.68. Swept on validation (recall floor → threshold),
with the effect on test shown for information:

| val recall floor | threshold | val spec | v3 test B: recall / spec | v4 test B: recall / spec / FP |
|---|---|---|---|---|
| 0.95 | 0.293 | 0.663 | 1.000 / 0.651 | 0.975 / 0.682 / 113 |
| 0.90 | 0.507 | 0.751 | 1.000 / 0.726 | 0.961 / 0.763 / 84 |
| **0.88** | **0.627** | **0.796** | **1.000 / 0.774** | **0.958 / 0.803 / 70** |
| 0.85 | 0.733 | 0.831 | 1.000 / 0.858 | 0.928 / 0.859 / 50 |

**Chosen: floor 0.88, threshold 0.6266** (`models/nina_thermal_frp20000_v4_abc/operating_point.json`).
It keeps test recall ≥ 0.95 on both test sets and cuts false alarms by 38%
against the default. The floor was chosen after looking at these test numbers
(mild selection on test). The threshold itself comes from validation only.
Changing it needs no retraining.

---

## 5. Quantization (int8)

`run_quantize`, same recipe as v3_bc_qat_sel: QAT 40 epochs, lr 5e-5, batch 16,
profiles A + B + C, observers frozen for scoring, checkpoint chosen by
real-int8 val ROC-AUC every 5 epochs, int8 threshold re-swept on int8 val
scores with the same floor (0.88).

| QAT epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | **40** |
|---|---|---|---|---|---|---|---|---|
| int8 val ROC-AUC | 0.871 | 0.895 | 0.893 | 0.908 | 0.901 | 0.908 | 0.910 | **0.911** |

| | fp32 | int8 |
|---|---|---|
| val ROC-AUC | 0.917 | 0.912 |
| threshold | 0.6266 | **0.4818** |
| val recall / specificity | 0.880 / 0.795 | 0.883 / 0.768 |
| size | 3.7 MB | **1.237 MB** (budget 2.0 MB) |

Conversion parity: fake-quant ROC-AUC 0.9107, converted 0.9113, score
correlation 0.987. Weights per-channel symmetric int8, activations per-tensor
uint8, float32 input / int8 output (float32 boundary still open, v3 issue #6).
QAT took 23 min.

**Acceptance gate:** recall drop and F-beta drop pass; **SQNR 10.6 dB (< 30) and
KL 0.232 nats (> 0.05) fail**. As in v3 §8.5, both compare int8 against the
original fp32 scores, which is the wrong reference after QAT changes the
weights. Pending the gate decision (v3 open issue #1).

---

## 6. Results

All metrics on profile B (flight condition) unless stated. CIs are 90%
granule-level bootstrap (1,000 resamples).

### 6.1 Frozen v3 test set (189 patches, 49 granules; 83 fire / 106 no-fire)

| | ROC-AUC (90% CI) | recall | spec | prec | TP / FP / FN / TN |
|---|---|---|---|---|---|
| v3_bc fp32 (thr 0.516) | 0.959 [0.920, 0.989] | 0.940 | 0.858 | 0.839 | 78 / 15 / 5 / 91 |
| v3_bc int8 (thr 0.765) | 0.959 [0.916, 0.989] | 0.940 | 0.877 | 0.857 | 78 / 13 / 5 / 93 |
| **v4 fp32** (thr 0.627) | **0.963** [0.915, 0.995] | **1.000** | 0.774 | 0.776 | 83 / 24 / 0 / 82 |
| **v4 int8** (thr 0.482) | **0.965** [0.940, 0.987] | **0.988** | 0.745 | 0.752 | 82 / 27 / 1 / 79 |

Threshold-free comparison, specificity at 95% recall: v3_bc 0.858, **v4 0.906**.

### 6.2 Full v4 test set (714 patches, 245 granules; 359 fire / 355 no-fire)

| | ROC-AUC (90% CI) | recall | spec | prec | TP / FP / FN / TN |
|---|---|---|---|---|---|
| v3_bc fp32 (thr 0.516) | 0.912 [0.883, 0.937] | 0.805 | 0.879 | 0.870 | 289 / 43 / 70 / 312 |
| **v4 fp32** (thr 0.627) | **0.945** [0.918, 0.967] | **0.958** | 0.803 | 0.831 | 344 / 70 / 15 / 285 |
| **v4 int8** (thr 0.482) | **0.935** [0.909, 0.956] | **0.961** | 0.763 | 0.804 | 345 / 84 / 14 / 271 |

Specificity at 95% recall: v3_bc 0.583, **v4 fp32 0.823**.

### 6.3 Profiles (v4 test, 714 patches)

| profile | v3_bc fp32 AUC / recall | v4 fp32 AUC / recall / spec | v4 int8 AUC / recall / spec |
|---|---|---|---|
| A | 0.867 / 0.632 | 0.955 / 0.955 / 0.808 | 0.933 / 0.942 / 0.758 |
| B | 0.912 / 0.805 | 0.945 / 0.958 / 0.803 | 0.935 / 0.961 / 0.763 |
| C | 0.915 / 0.813 | 0.936 / 0.953 / 0.792 | 0.931 / 0.964 / 0.738 |

One threshold now works across all three profiles. Profile A, never seen by
v3_bc in training, is fixed (v3 open issue #4).

### 6.4 Within-month AUC (warm-scene confound)

Scores centred per month; months with ≥ 2 patches of each class.

| | v3 test | v4 test |
|---|---|---|
| v3_bc | 0.958 | 0.863 |
| v4 | 0.928 | 0.854 |

**No improvement.** This is computed differently from v3 report §7.2 (which
used 5 hand-picked mixed months), so the absolute values are not comparable
with that table. Read with care: few patches per month in the v3 test.
Interpretation: v4 ranks scenes across seasons and years much better, but
inside a month it separates fire from no-fire no better than v3_bc.
Consistent with v3 §7.1: a single fire is invisible at 4 km in 8–14 µm, and
the sensor sees total fire activity and its effect on scene temperature.

### 6.5 Operational precision

v4 int8, v4 test, profile B, Bayes projection to an assumed fire prevalence:
precision 0.80 at the measured 50% prevalence; at 10% prevalence about 0.31.
If fire scenes are rare in operations, most alarms will be false at this
operating point. The threshold should be set from the mission's cost of a
missed fire against a false alarm.

---

## 7. What is left

### 7.1 To finish v4

| # | item | effort |
|---|---|---|
| 1 | Render profile A for the train/val of `data_202208/02/09` (§2.5), retrain, re-run QAT and evaluation (**v4.1**) | ~1 h |
| 2 | Download the 5 missing Batch 2 roots (FIRMS ready; resume works) and include them in v4.1 | ~1–2 h |
| 3 | Decide the operating point with the mission team (§4.1, §6.5) | decision |
| 4 | Decide the acceptance gate definition after QAT (v3 issue #1) | decision |
| 5 | Commit or release model artifacts; record SHA-256 in `models/MANIFEST.json` (on `main`) | ~30 min |
| 6 | Merge `fix/nina-thermal-labels-and-qat` and `feat/v4-data-expansion` into `main` (main has moved: 9 commits, reproducibility layout) | ~1–2 h |
| 7 | Update `RUNBOOK_v4_data_expansion.md` with the Batch 1–2 results (§5–§6 of the runbook) | ~20 min |

### 7.2 Still open from v3

| v3 # | issue | v4 status |
|---|---|---|
| 1 | Gate fails SQNR/KL after QAT | unchanged (decision) |
| 2 | Warm-scene confound | reduced across seasons; **not** within months (§6.4) |
| 3 | June 2023 false alarms | not re-measured; June 2023 now has 219 no-fire and 3 fire patches |
| 4 | Profile A recall | **closed** (0.99 on v3 test, 0.94–0.96 on v4 test) |
| 5 | No Mar–Apr wet-season scenes | open (low priority: no-fire is plentiful) |
| 6 | float32 graph boundary | open |
| 7 | 1 GW negative threshold not swept | open (runbook E2) |

### 7.3 Suggested next experiments

- E1: QAT with seeds 42 / 1 / 2 (QAT is not reproducible on GPU).
- E2: `--frp_negative_max` 500 / 1,000 / 2,000 MW.
- A held-out year (e.g. all of 2019 in test) for an out-of-time evaluation.
- Within-month analysis per month with CIs, and a scene-temperature baseline
  on the v4 pool (`run_baseline`), to quantify how much of v4 is "hot scene".

---

## 8. Reproduce

```bash
ROOTS=$(python -c "import json;print(' '.join(json.load(open('models/nina_thermal_frp20000_v4_abc/training_report.json'))['data']['roots']))")
C="--frp_threshold 20000 --frp_negative_max 1000"
OUT=models/nina_thermal_frp20000_v4_abc
python -m nina_thermal.training.train_thermal $C --roots $ROOTS --profiles A B C \
    --recall_target 0.95 --out_dir $OUT --num_workers 2
Q=models/nina_thermal_frp20000_v4_abc_qat_sel
python -m nina_thermal.quantization.pipelines.run_quantize --checkpoint $OUT/mobilenetv3s_fire_best.pt \
    $C --roots $ROOTS --profiles A B C --recall_target 0.88 --out_dir $Q \
    --epochs 40 --lr 5e-5 --batch_size 16 --num_workers 2 --select_every 5 --no_smoke_test
V3="data data_202408 data_202409 data_202509 data_202208 data_202302 data_202309"
for R in "$V3" "$ROOTS"; do   # frozen v3 test, then full v4 test
  python -m nina_thermal.evaluation.pipelines.run_evaluate --checkpoint $OUT/mobilenetv3s_fire_best.pt \
      $C --roots $R --threshold 0.6266 --tflite $Q/mobilenetv3s_fire_int8.tflite \
      --tflite_threshold 0.4818 --out_dir $Q/eval_$(echo $R | wc -w)roots
done
```

Training is deterministic on this machine (seed 42). QAT is not bit-reproducible
on GPU (v3 §8.4). Data roots are rebuilt with the runbook; raw granules are not
kept.

## 9. Artifacts

```
models/nina_thermal_frp20000_v4_abc/            fp32
├── mobilenetv3s_fire_best.pt          checkpoint
├── training_report.json               provenance, history, val metrics (threshold 0.293 = training default)
├── operating_point.json               chosen threshold 0.6266 (§4.1)
├── val_predictions.json               val scores used for the threshold sweep
├── labels_by_root_month.json          §2.2
├── eval_v3test/  eval_all/            fp32 at threshold 0.293, frozen v3 test and v4 test
└── eval_v3bc_on_v4test/               v3_bc evaluated on the v4 test
models/nina_thermal_frp20000_v4_abc_qat_sel/    int8
├── mobilenetv3s_fire_int8.tflite      1.237 MB, sha256 954388ef…
├── quantization_report.json           selection history, parity, gate (FAIL: SQNR, KL)
└── eval_v3test/  eval_all/            fp32 (thr 0.6266) + int8 (thr 0.4818) metrics and predictions
```

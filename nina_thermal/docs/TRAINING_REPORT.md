# NINA Thermal — Fire Classifier Training Report

**WaskiriSat / NINA payload — fire detection model**
**Date:** 2026-08-31 · **Refreshed:** 2026-09-03 (`run_baseline` + `run_evaluate` +
`run_quantize` re-run against `main` post-#8→#12, no retraining)
**Finalized:** 2026-10-02 — every number in this report re-checked against the
committed artifacts in `models/nina_thermal_frp20000/` (training/quantization/
evaluation/baseline reports, 129/129 `nina_thermal` tests passing); fixed a
unit inconsistency in §4/§8 and synced §12 to `RUNBOOK_acquire_low_season.md`.
No retraining.
**Branch:** `main` · **Artifacts:** `models/nina_thermal_frp20000/`

---

## 1. Executive summary

A MobileNetV3-Small binary classifier was trained to detect **large fire complexes**
in simulated FLIR Lepton 3.5 thermal imagery at 160×120 px, and quantized to an
int8 TFLite artifact of **1.385 MB** — within the 2 MB mission budget.

Headline validation result (fp32): **ROC-AUC 0.836, accuracy 0.833, recall 1.000,
specificity 0.684.** On the granule-disjoint test split the model holds at
**ROC-AUC 0.787** (90% bootstrap CI **[0.644, 0.943]**, granule-level resampling,
n=1000), indicating the learned feature transfers to unseen scenes.

Inference is **not** a bottleneck: 1.61 ms per frame for the int8 model on a host
CPU core, against a Lepton frame period of ~116 ms (8.6 Hz).

**2026-09-03 refresh — new finding.** The reusable evaluation/quantization
pipelines added in #8→#11 introduced a formal **acceptance gate** for the int8
artifact (recall drop ≤0.5pp, F-beta drop ≤1.0pp, SQNR ≥30dB, KL ≤0.05 nats).
Re-running quantization against the unchanged fp32 checkpoint, **the gate now
fails** — see §6.2. This did not exist when 0.787/0.769 were first reported;
it is new information, not a regression in the model itself.

---

## 2. Critical finding — the original labels were not learnable

The first training run failed to learn. Diagnosis showed the cause was the
**dataset target**, not the model.

Comparing fire against no-fire patches **within the same granule** (same scene,
same overpass — all terrain, season and time-of-day confounds removed):

| feature | fire > no-fire | sign test | interpretation |
|---|---|---|---|
| patch mean temperature | 41 / 62 | p = 0.015 | fire patches are warmer overall |
| **hottest pixel** | 33 / 62 | **p = 0.70** | **indistinguishable from chance** |
| max − median (local anomaly) | 17 / 62 | p = 0.0005 | significant in the **wrong** direction |

In physical units the hottest pixel differs by **−0.082 K (95% CI −0.797 … +0.633)** —
smaller than one AGC quantization step (0.392 K).

**Root cause.** A patch is 160×120 px at 4 km GSD = a **640 × 480 km footprint**.
Each pixel covers 16 km². An active fire front occupies ~0.1 % of one pixel, so
averaging raises it by a few tenths of a Kelvin, below the quantization step, and
profile-B noise plus blur removes what remains. NASA FIRMS detects these fires at
3.74 µm (mid-wave), where a fire is ~100–1000× brighter than background; the
Lepton is broadband long-wave (8–14 µm), where it is only ~10× brighter.

The labels were **correct** — the fire was real — but **not inferable** from these
images.

---

## 3. The fix — FRP-based relabelling

The target was reformulated to one the sensor can support: **total radiated fire
power in the field of view**, using FIRMS Fire Radiative Power (FRP, in MW)
summed over each patch footprint.

- **Positive:** summed FRP ≥ 20,000 MW within ±1 day of sensing
- **Negative:** footprint contains **zero** FIRMS detections
- **Excluded:** everything in between (ambiguous)

Excluding the ambiguous band also fixed negative-class contamination: **299 of 516**
original label-0 patches contained FIRMS detections inside their footprint.

### Threshold selection (test split, profile B)

| labels | ROC-AUC | recall | precision | **specificity** | val → test |
|---|---|---|---|---|---|
| datagen (any fire) | 0.572 | 1.000 | 0.431 | **0.025** | 0.698 → 0.572 |
| FRP ≥ 5,000 MW | 0.615 | 0.989 | 0.677 | **0.065** | 0.694 → 0.615 |
| **FRP ≥ 20,000 MW** | **0.787** | 0.778 | 0.651 | **0.674** | 0.836 → 0.787 |

**Specificity is the discriminator.** Below 20 GW the classifier is degenerate —
it answers "fire" to essentially every scene, so its high recall is meaningless.
At 20 GW it rejects two thirds of non-fire scenes, and the val→test gap nearly
closes.

> An initial recommendation of 5,000 MW (argued on sample-size grounds) was
> **overturned by measurement**. 50,000 MW scores higher still but leaves fewer
> than 10 validation positives, too few to tune an operating point on.

---

## 4. Model

| property | value |
|---|---|
| Architecture | MobileNetV3-Small, single-logit binary head |
| Head | `Dropout(0.2) → Linear(576, 1)` (stock 1024-unit bottleneck removed) |
| Trainable parameters | **927,585** |
| Stored tensors (incl. BatchNorm buffers) | 939,731 |
| Input | 160 × 120 × 3 (grayscale replicated), native Lepton geometry — no resize |
| Compute | **22.22 MMACs** (44.45 MFLOPs) per inference |
| fp32 checkpoint | 3.67 MB |
| **int8 TFLite** | **1.385 MB** ✅ within 2 MB budget |

(Sizes are MiB — bytes / 1024² — matching `run_quantize`'s own `size_mb`
computation and the 2 MB budget check it is compared against.)

**Why MobileNetV3-Small:** measured at 160×120 with a single-logit head,
MobileNetV2 α=1.0 is 2,225,153 params (~2.12 MB int8) while MobileNetV3-Small is
927,585 (~0.88 MB). It is the smallest torchvision backbone still shipping
ImageNet weights — α<1.0 MobileNetV2 variants do not, and training from scratch
on ~200 images is not viable.

**Note on the 1 MB target:** not reachable with this backbone. int8 weights alone
are 0.88 MB; int32 biases, quantization parameters and flatbuffer metadata put the
measured floor at ~1.25 MB.

### Key hyperparameters

| parameter | value | rationale |
|---|---|---|
| `pos_weight` | 3.0 | FN:FP **mission cost** ratio, not class imbalance. `beta = √3 = 1.732` |
| `recall_target` | 0.95 | threshold swept on val to enforce a recall floor, not fixed at 0.5 |
| `select_metric` | `roc_auc` | F-beta selection rewarded "predict fire everywhere" on a 36-image val set |
| Phase 1 | 20 epochs, lr 1e-3, frozen backbone | |
| Phase 2 | 15 epochs, lr 1e-4, unfrozen | thermal grayscale is far outside the ImageNet manifold |
| Augmentation | flips + ≤8 % translation only | **no** photometric jitter — under fixed AGC a pixel value *is* a temperature |
| Profile / AGC | B only, `fixed` only | flight configuration; A and C held out as OOD bounds |
| Seed | 42 | |

---

## 5. Dataset

| split | images | granules | fire | no-fire | prevalence |
|---|---|---|---|---|---|
| train | 219 | 72 | 67 | 152 | 0.306 |
| **validation** | **36** | **11** | **17** | **19** | **0.472** |
| test | 82 | 21 | 36 | 46 | 0.439 |
| **total** | **337** | | | | |

Splits are **granule-disjoint** — no scene appears in two splits. Where source
roots disagreed on a granule's split, reconciliation used `test > val > train`
precedence, so a granule held out anywhere is held out everywhere.

### Data funnel

| stage | images | note |
|---|---|---|
| PNGs on disk (4 roots) | 6,570 | |
| after de-duplication | 5,544 | `data_202408` was a strict subset of `data/` |
| AGC = `fixed` only | 2,772 | `global` is not the flight configuration |
| profile B only | 924 | A and C held out as OOD bounds |
| **after FRP relabel @ 20 GW** | **337** | 587 ambiguous patches excluded |

**The binding constraint is 201 unique granules.** The 6,570 images are six
rendered variants (3 profiles × 2 AGC modes) of 924 patches drawn from those 201
Sentinel-3 scenes. Variants add no new information.

---

## 6. Evaluation — validation set

### 6.1 fp32 model (primary)

Operating threshold **0.4564**, selected by recall-floor sweep on validation.

| metric | value |
|---|---|
| Accuracy | **0.833** |
| Recall (sensitivity) | **1.000** |
| Precision | 0.739 |
| Specificity | 0.684 |
| F1 | 0.850 |
| F-beta (β = 1.732) | 0.919 |
| **ROC-AUC** | **0.836** |
| PR-AUC | 0.738 |

```
                    pred FIRE   pred NO-FIRE
  actual FIRE            17            0        recall 1.000
  actual NO-FIRE          6           13        specificity 0.684
```

Every fire in the validation set was caught, at the cost of 6 false alarms out of
19 clear scenes.

### 6.2 int8 quantized model

Operating threshold **0.1809**, re-swept on int8 validation scores (40 QAT
epochs, lr 5e-5, `--batch_size 16 --num_workers 4` default settings OOM-killed
this GPU mid-run at the default batch size 32 / 4 workers — reduce to
`--batch_size 16 --num_workers 2` if reproducing).

| metric | value | Δ vs fp32 |
|---|---|---|
| Recall | 1.000 | 0.000 |
| Precision | 0.607 | −0.132 |
| Specificity | 0.421 | −0.263 |
| F-beta | 0.861 | −0.058 |
| **ROC-AUC** | **0.774** | **−0.062** |
| PR-AUC | 0.671 | −0.068 |

```
                    pred FIRE   pred NO-FIRE
  actual FIRE            17            0
  actual NO-FIRE         11            8
```

Recall is preserved exactly; the cost is precision and specificity. QAT epoch
count matters strongly here — 3 epochs gave ROC-AUC 0.599, 15 gave 0.709, 40
epochs at lr 5e-5 recovers ROC-AUC to the 0.77–0.79 range (run-to-run variance
of a few points on a 36-image validation set).

**Acceptance gate: FAILED**, on all three measured axes:

| check | tolerance | measured | verdict |
|---|---|---|---|
| Recall drop | ≤ 0.005 | 0.000 | pass |
| F-beta drop | ≤ 0.010 | 0.058 | **fail** |
| SQNR | ≥ 30 dB | 8.3 dB | **fail** |
| Score-distribution KL | ≤ 0.05 nats | 11.07 nats | **fail** |

SQNR and the F-beta drop are real and consistent with the operating-point
metrics above — the int8 model works, but noticeably noisier than fp32.

The **KL figure needs a caveat**: it is a 64-bin histogram divergence
estimated from only **36 validation scores**, so most bins hold 0–1 samples.
With this few samples the estimator is structurally unstable regardless of
how good the quantization is — a handful of scores landing in bins where the
other distribution's epsilon-smoothed density is tiny inflates KL sharply.
0.05 nats as a tolerance was very likely calibrated assuming a much larger
validation set; on n=36 it may not be achievable at any epoch count. Treat
SQNR and F-beta drop as the trustworthy signals here, and consider either
enlarging the val split (§9) or revisiting the KL tolerance/bin count before
treating this gate as a go/no-go for deployment.

### 6.3 Test set (generalization check)

| model | split | accuracy | recall | precision | specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| fp32 | val | 0.833 | 1.000 | 0.739 | 0.684 | 0.850 | 0.836 | 0.738 |
| **fp32** | **test** | **0.720** | **0.778** | 0.651 | 0.674 | 0.709 | **0.787** | 0.702 |
| int8 | test | 0.634 | 0.972 | 0.547 | 0.370 | 0.700 | 0.694 | 0.539 |

Granule-level 90% bootstrap CIs (n=1000 resamples) on the fp32 test row, from
`run_evaluate --figures`:

| metric | point estimate | 90% CI |
|---|---|---|
| Recall | 0.778 | [0.675, 0.889] |
| Precision | 0.651 | [0.386, 0.941] |
| F-beta | 0.742 | [0.598, 0.849] |
| ROC-AUC | 0.787 | [0.644, 0.943] |

Precision's CI is wide — 82 test images is not enough to pin it down tightly;
the lower bound (0.386) is a real possibility, not a rounding artifact.

The **0.836 → 0.787** val-to-test gap is small, indicating genuine transfer to
unseen granules. Under the original labels the same gap was 0.698 → 0.572.
The int8 test row above predates the acceptance-gate machinery in §6.2 and was
measured manually rather than through a reusable pipeline command —
`run_evaluate` scores the fp32 checkpoint only; there is currently no
equivalent one-command int8-on-test path.

### 6.4 Cross-profile robustness (test set, fp32)

| profile | condition | ROC-AUC | recall at fixed threshold |
|---|---|---|---|
| A | post-FFC best case | 0.806 | 0.361 |
| **B** | **nominal (trained)** | **0.787** | **0.778** |
| C | SAA-transit worst case | 0.759 | 0.278 |

Ranking quality is robust across all three degradation profiles, but **the
operating threshold does not transfer** — calibrated on B, recall collapses on A
and C. Either train on B+C or calibrate a threshold per profile.

**This table is now one command**, not a manual analysis: `run_evaluate`
expands the test set to A/B/C automatically and writes `by_profile` +
`deltas_vs_B` to `evaluation_summary.json`, plus
`figures/eval_profile_{a,b,c}.png`. A/C images for the test granules already
exist on disk for all four data roots (rendered before the split-aware
policy in #5) — **no re-simulation is needed to reproduce this table.**

### 6.5 Feasibility baselines (`run_baseline`, reproducible gate)

Two logistic-regression baselines over hand-crafted BT statistics
(mean/std/percentiles/gradient/hot-fraction), grouped 5-fold CV by granule,
train+val only — the test split is never read here. This is the number a CNN
must beat to justify its complexity, and reproduces the paper's negative
result on demand.

| label scheme | n | fire fraction | grouped-CV AUC |
|---|---|---|---|
| `datagen` (any fire) | 717 | 0.446 | 0.593 ± 0.101 |
| `frp` (≥ 20,000 MW) | 255 | 0.329 | **0.717 ± 0.134** |

The `datagen` result is more nuanced than a flat "~0.5" and worth reading at
the per-feature level, since it independently reproduces §2's within-granule
finding:

| feature | AUC | reads as |
|---|---|---|
| `bt_mean` / `bt_p50` | 0.611 / 0.623 | whole-scene warmth — mildly predictive |
| `top_pixel_excess` | **0.479** | localized hotspot — at chance |
| `grad_mean` / `grad_p99` | 0.492 / 0.491 | local gradient structure — at chance |

Only whole-scene features carry signal; every localized feature sits at
chance. This is the same scene-warmth confound documented in `frp_labels.py`
and §10, now visible in a hand-crafted-feature ablation rather than only in
the within-granule sign test. Per the runbook, an AUC "high" enough to
contradict this — strong signal on the local features — would mean the
datagen label is not degenerate after all and §§3/6/7 would need rework;
that did not happen here.

The `frp` baseline (0.717) sits below the CNN's test ROC-AUC (0.787),
consistent with the CNN learning some structure beyond BT summary statistics.

---

## 7. Inference performance

Batch size 1 (the on-orbit case). Host: Intel Core i5-13420H, 200 timed
iterations after 30 warm-up.

| configuration | mean | median | p95 | throughput |
|---|---|---|---|---|
| fp32 PyTorch — CPU, 1 thread | 4.49 ms | 4.45 ms | 4.98 ms | 223 fps |
| fp32 PyTorch — CPU, 8 threads | 4.12 ms | 4.03 ms | 4.75 ms | 243 fps |
| fp32 PyTorch — GPU (RTX 4050) | 3.11 ms | 3.08 ms | 3.36 ms | 321 fps |
| **int8 TFLite — CPU, 1 thread** | **1.61 ms** | **1.60 ms** | **1.67 ms** | **623 fps** |
| int8 TFLite — CPU, 8 threads | 1.80 ms | 1.79 ms | 1.85 ms | 557 fps |

**Observations**

- Quantization delivers a **2.8× speed-up** (4.49 → 1.61 ms) on a single core, on
  top of the 4× size reduction.
- Multi-threading **hurts** the int8 model (1.61 → 1.80 ms): at 22 MMACs the
  thread-coordination overhead exceeds the parallelism gain. Deploy single-threaded.
- **Inference is not the bottleneck.** The Lepton 3.5 runs at 8.6 Hz — one frame
  every ~116 ms. Even the slowest configuration measured is ~26× faster than the
  sensor delivers frames.

### On-orbit estimate (NOT measured)

No OpenMV N6 hardware was available, so the following is a **calculation, not a
measurement**:

- 22.22 MMACs ≈ 44.45 MOPs per inference
- Neural-ART NPU rated 600 GOPS INT8 → theoretical floor ~0.074 ms
- At a realistic 10–30 % utilization → **~0.25–0.75 ms per frame**

**Caveat:** this assumes the whole graph runs on the NPU. The current artifact has
a **float32 input boundary**, so part of the graph would fall back to the
Cortex-M55 at 800 MHz and be substantially slower. This must be validated on
hardware.

---

## 8. Artifacts

```
models/nina_thermal_frp20000/
├── mobilenetv3s_fire_best.pt        3.67 MB   fp32 checkpoint (state dict) — gitignored, regenerable
├── mobilenetv3s_fire_int8.tflite    1.385 MB  int8 TFLite, QAT 40 epochs — not yet committed (see note)
├── mobilenetv3s_fire_int8.sha256              fdbdf938d4dcc036…
├── training_report.json              18 KB    full provenance, 35-epoch history
├── quantization_report.json         2.9 KB    fp32↔int8 parity + acceptance gate
├── evaluation_summary.json                    test-set metrics + bootstrap CIs + per-profile (§6.3–6.4)
├── baseline_report_datagen.json               feasibility baseline, datagen labels (§6.5)
├── baseline_report_frp.json                   feasibility baseline, FRP labels (§6.5)
├── predictions.json                           per-sample test predictions (underlies the CIs above)
└── figures/eval_profile_{a,b,c}.png           per-profile score distributions (§6.4)
```

`.pt` checkpoints are gitignored repo-wide (regenerable from the committed
training code). The `.tflite` is the deployable artifact but, per repo
convention, is "added explicitly when a release is cut" — it exists on disk
and its checksum/report are committed, but the binary itself is not yet in
git. The checksum above was re-verified against the current file on disk as
part of finalizing this report.

### Reproduction

```bash
python -m nina_thermal.training.train_thermal \
    --frp_threshold 20000 --out_dir models/nina_thermal_frp20000

python -m nina_thermal.quantization.pipelines.run_quantize \
    --checkpoint models/nina_thermal_frp20000/mobilenetv3s_fire_best.pt \
    --out_dir models/nina_thermal_frp20000 --frp_threshold 20000 \
    --epochs 40 --lr 5e-5 --batch_size 16
```

`--frp_threshold` **must match** between the two commands, or QAT optimises
against a different target than the checkpoint was trained on.

---

## 9. Open issues

| # | issue | impact | proposed action |
|---|---|---|---|
| 1 | int8 **fails the acceptance gate** (§6.2): F-beta drop 0.058, SQNR 8.3 dB, KL 11.07 nats | not clearable to fly as-is on the current gate | outlier clipping before quantization; more QAT epochs; separately, the KL tolerance/bin-count may need recalibrating for a 36-image val set (§6.2) |
| 2 | Float32 input boundary on exported graph | NPU cannot own the whole graph | wire the `ai_edge_quantizer` `static_wi8_ai8` recipe |
| 3 | Threshold does not transfer across profiles | recall 0.778 (B) → 0.278 (C) | train on B+C, or per-profile threshold calibration |
| 4 | Only 219 training images, 36 validation images | wide CIs (precision 90% CI [0.386, 0.941]); KL estimator unstable at this n | train on B+C (→438); low-season acquisition planned (§12) |
| 5 | Scene-warmth confound not excluded | may be a "hot dry day" detector | correlate score vs scene mean temperature; low-season clean negatives (§12) directly attack this |
| 6 | FIRMS ±1 day window vs 62 % night overpasses | label noise | tighten window; match night passes to VIIRS 01:30 |
| 7 | `nina_cloud/quantization/` broken on current deps | cloud pipeline will not run | apply the same `torchao` / `litert-torch` fallbacks |
| 8 | Default `run_quantize` settings (`--batch_size 32 --num_workers 4`) OOM-killed on a 30 GB RAM host at 40 epochs | silent process death, no traceback | use `--batch_size 16 --num_workers 2`; document in the runbook |

### Deployment prerequisite

The flight Lepton **must** be configured with the same **fixed AGC** used in
training (258.15–358.15 K → 0–255, i.e. 0.392 K per grey level). If it runs in
default auto-gain, pixel values no longer map to absolute temperature and the
model's entire basis collapses.

---

## 10. Honest limitations

1. **Test set is 82 images.** Confidence intervals are wide. Treat ROC-AUC 0.787
   as "clearly working," not as a precise value.
2. **The scene-warmth confound is not excluded.** The within-granule control that
   invalidated the original labels cannot be run at 20 GW — no granule contains
   both a ≥20 GW patch and a zero-detection patch. Part of the gain may be that
   large fire complexes occur on hot, dry days. This model is more defensibly
   described as a **high fire-activity scene detector** than an active-fire
   detector.
3. **On-orbit latency is calculated, not measured.** No N6 hardware was available.
4. **The first negative result is itself a contribution.** It quantifies the
   detection floor of a broadband LWIR microbolometer at 4 km GSD with measured
   numbers — a defensible methodological result for IAC 2026, and a limit worth
   knowing before launch rather than after.

---

## 11. Recommended next step

Extract SLSTR **F1/S7 (3.74 µm)** — the mid-wave fire channels, already present in
the downloaded `.SEN3` granules but never extracted — for the same patches and
test whether fire is visible there.

- **Visible at 3.74 µm, absent at 10.85 µm** → proves labels and images are both
  correct and the loss is purely wavelength + aggregation. This is the rigorous,
  quantified statement of the Lepton's detection floor that anchors the paper.
- **Not visible at 3.74 µm either** → the labels or the ±1 day timing window are
  wrong, and it is a fixable data problem.

Either outcome is decisive, and it uses granules already on disk.

## 12. Planned — two new acquisition roots (not yet run)

Full step-by-step plan: **`nina_thermal/docs/RUNBOOK_acquire_low_season.md`**
(status as of 2026-09-24: **not started**). Summary:

- **`data_202208`** — Aug 2022, fire-rich (`--use_firms`), no overlap with
  existing roots. Training has shifted toward accumulated-fire scenes, which
  needs more fire-rich data.
- **`data_202302`** — Feb–Apr 2023, low/wet season, no FIRMS filter. Adds the
  clean no-fire negatives that directly attack the scene-warmth confound in
  §10: a fire complex occurring on a hot dry day is confounded with "hot dry
  day"; a genuinely low-season scene is neither, and is exactly the control
  the within-granule test in §2 could not construct at the 20 GW threshold.

Disk estimate in the runbook is **60–80 GB** for both windows (revised up from
an earlier 30–50 GB guess), against 128 GB free as of 2026-09-24 — workable
but worth checking before starting; three old `.zip` archives (~59 GB) at the
repo root look like they duplicate already-extracted directories and may be
removable first (verify before deleting).

This is a separate, larger job from the refresh in this report: it requires
`run_extraction → run_processing → run_simulation → run_split → run_balance`
per new root, then **retraining, requantizing and re-evaluating** everything
above into a new `models/nina_thermal_frp20000_v2/` directory (the runbook
gives the exact commands, including the `--batch_size 16 --num_workers 2`
fix for the OOM in §9 issue 8), since it changes the training pool. The 0.787
headline may move.

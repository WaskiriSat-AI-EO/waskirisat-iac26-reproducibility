# Runbook — regenerate `models/thermal/frp20000/` with the new reporting

**For:** Osvaldo (the FRP20000 dataset + checkpoint live on your machine; they
are gitignored and too heavy to move, so the pipeline runs where the data is).

**Prerequisite:** PRs #8 → #9 → #10 → #11 merged to `main`.

**What this does NOT require:** re-training. Your existing
`models/thermal/frp20000/mobilenetv3s_fire_best.pt` is still valid — the
merged PRs add tests, an evaluation module, a baseline module and doc fixes;
they do not change `train_thermal.py`'s training logic or the on-disk filename
convention.

---

## 0. Update and sanity-check

```bash
git checkout main && git pull
pip install -r requirements.txt        # now pulls litert-torch + torchao
pytest nina_thermal -q                 # expect ~290 passing; the QAT test takes ~40 s
```

Your 4 data roots must be present as
`data/`, `data_202408/`, `data_202409/`, `data_202509/`, each with
`04_dataset/{balanced_manifest.json,images/,metadata/}` and `firms/*.csv`
(the same layout `train_thermal.py` merged). These are the `DEFAULT_ROOTS`, so
`--roots` can be omitted below.

---

## 1. Feasibility baseline — the paper's negative result, made reproducible

```bash
python -m nina_thermal.training.pipelines.run_baseline \
    --label datagen --out_dir models/thermal/frp20000

python -m nina_thermal.training.pipelines.run_baseline \
    --label frp --frp_threshold 20000 --out_dir models/thermal/frp20000
```

Writes `baseline_report_datagen.json` and `baseline_report_frp.json`.
Expected shape of the result (this is the paper's Section 6.1b / PH-29):

* `--label datagen` → `grouped_cv.auc_mean` near **0.5**, `per_feature_auc` all
  near 0.5, `degenerate` warnings — the datagen label is not learnable at this
  GSD. **If instead the AUC is high, tell Christian before anything else** — it
  would mean the datagen label is not degenerate after all and Sections 3/6/7
  need rework.
* `--label frp` → `grouped_cv.auc_mean` around **0.6–0.7** — the reformulated
  target carries signal a linear model can partly capture, which the CNN then
  beats.

The `train`+`val` granules only are read; the test split is never touched here.

---

## 2. Test-split evaluation with confidence intervals

```bash
python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint models/thermal/frp20000/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 \
    --out_dir models/thermal/frp20000 \
    --figures
```

Writes `evaluation_summary.json`, `predictions.json`, and `figures/*.png`.
Reads `selected_threshold` and `beta` from
`models/thermal/frp20000/training_report.json`.

Expected: `nominal.roc_auc` within the CI of the 0.787 in `TRAINING_REPORT.md`,
and every headline metric (`recall`, `precision`, `fbeta`, `roc_auc`) now
carrying a 90 % granule-level bootstrap interval in `nominal.bootstrap_ci`.

**Cross-profile table (paper figure F5):** `run_evaluate` expands the test set
to profiles A/B/C. If your test granules were rendered at **profile B only**,
`by_profile` will contain just `B` and `deltas_vs_B` will be empty — the run
still succeeds, it just logs `N expected image(s) absent from disk and skipped`.
To get the A/C robustness bounds you would need to re-render the test granules:
`run_simulation` (post-#5, which makes it apply `test → A,B,C fixed`
automatically) → `run_split` → `run_balance` on the test granules, from
`03_patches/`. That is a larger job — do it only if the paper needs F5 filled.

---

## 3. Quantization + acceptance gate

```bash
python -m nina_thermal.quantization.pipelines.run_quantize \
    --checkpoint models/thermal/frp20000/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 \
    --out_dir models/thermal/frp20000 \
    --epochs 40 --lr 5e-5
```

(`--epochs 40 --lr 5e-5` reproduces the setting your `TRAINING_REPORT.md` notes
recovered int8 val ROC-AUC 0.769 against an fp32 reference of 0.836.)

Rewrites `mobilenetv3s_fire_int8.tflite` + `.sha256` and
`quantization_report.json`. The report now has:

* `parity.sqnr_db`, `parity.kl_divergence_nats`, `parity.fbeta_delta`;
* an `acceptance` block — the four-part deployment gate (recall drop ≤ 0.5 pp,
  F-beta drop ≤ 1.0 pp, SQNR ≥ 30 dB, KL ≤ 0.05 nats).

**The exit code is now non-zero if the gate fails** (previously only the size
budget gated it). A failed gate is a *result to report*, not a bug — note the
violations in `TRAINING_REPORT.md` and, if needed, raise `--epochs` /
`--calib_batches`.

---

## 4. Commit the refreshed artifacts

```bash
git checkout -b chore/nina-thermal-frp20000-refresh
git add models/thermal/frp20000/*.json \
        models/thermal/frp20000/*.sha256 \
        nina_thermal/docs/TRAINING_REPORT.md \
        nina_thermal/docs/figures/
git commit -m "chore(nina-thermal): regenerate FRP20000 reports with CIs + acceptance gate"
```

The `.tflite` and `.pt` stay gitignored — only the JSON reports, the checksum,
the figures and the refreshed `TRAINING_REPORT.md` are committed.

While updating `TRAINING_REPORT.md`, keep the honest framing: this is a
**high fire-activity scene detector**, not an active-fire detector — the
scene-warmth confound (large fire complexes occur on hot dry days) cannot be
excluded at the 20 GW threshold, and that limitation is already stated in
`frp_labels.py` and should stay in the paper.

---

## Still open after this (not blocking a first paper draft)

* SEU bit-flip injection harness (`experiments/v5_seu/`) — the paper's
  highest-value novel experiment; does not exist yet.
* FRP-stratified detection-envelope metric (PH-25) — needs FRP propagated into
  `predictions.json`.
* Repo-wide `ruff check .` / `black --check .` are red on `main` today
  (pre-existing, ~434 ruff findings, no `[tool.ruff]` config) — a one-time
  sweep or a config file is needed for CI to go green.

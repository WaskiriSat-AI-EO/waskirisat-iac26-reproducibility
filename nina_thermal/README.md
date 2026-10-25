# nina_thermal — NINA fire classifier

**Owner:** Osvaldo. Pipeline: `datagen/` → `training/` → `quantization/` → `evaluation/`.
Design rationale: `docs/CLAUDE_CODE_BRIEF.md`; training: `docs/TRAINING_REPORT.md`; regeneration: `docs/RUNBOOK_regenerate_frp20000.md`.

## Delivery checklist
- [x] `configs/paper.yaml`: FRP threshold (20000), seeds, epochs (QAT budget 40), the model (MobileNetV3-Small @ 160×120)
- [x] Verify the runbook regenerates `../models/thermal/frp20000/` from a clean checkout; note what is bit-exact and what is not (QAT). See [Regeneration](#regeneration-what-is-and-is-not-bit-exact)
- [ ] FP32 checkpoint (`mobilenetv3s_fire_best.pt`) and INT8 `.tflite` → release; `.tflite` sha256 must match `6e486fc7…01e8` (already the HIL artifact). sha256 of both recorded in `../models/MANIFEST.json` and checked by `make verify-models`; **hosting pending (#17)**
- [x] Evaluation: ROC-AUC 0.787 (granule-level 90% CI 0.644–0.943), PR-AUC 0.702, thresholds 0.456 (FP32) / 0.2875 (INT8, HIL artifact) via `evaluation/pipelines/run_evaluate.py`
- [x] Reproduce the acceptance gate: it **fails 3 of 4** criteria. Keep the result visible, do not tune it away (`quantization/pipelines/run_gate.py`)
- [x] Dataset composition for Table 5: `../data/manifests/thermal_v1_*` (granule lists, split registry, FIRMS files and dates). FIRMS covers Aug 2023, Aug–Sep 2024 and Sep 2025; **June 2023 granules have no FIRMS data** (see `DATASET_CARD.md`)
- [x] `make table6` and `make table5` produce the paper values for this payload (`python -m reproduce.table5 thermal`, `… table6 thermal`, `… thresholds thermal`; the full targets wait on the cloud rows, #14)

Model and data: [`MODEL_CARD.md`](MODEL_CARD.md), [`DATASET_CARD.md`](DATASET_CARD.md). Table 2 (profiles A/B/C) comes from `datagen/config.py` `DEGRADATION_PROFILES`.

## Run

Reproduces the paper's thermal numbers from the published artifacts. No training; a CPU is enough.

**1. Data.** The v1 roots `data/`, `data_202408/`, `data_202409/`, `data_202509/`, each with
`04_dataset/` (images + `balanced_manifest.json`) and the FIRMS CSVs that existed at
training time (2026-08-30). Those CSVs are the `firms_old_narrow/` folders, **without** the
seven June 2023 SP files added on 2026-10-03; with them, 195 v1 negatives become
ambiguous and the test set collapses to 37 patches. The code reads `<root>/firms/*.csv`,
so build a view with symlinks:

```bash
V1=../v1_view
for r in data data_202408 data_202409 data_202509; do
  mkdir -p $V1/$r/firms
  ln -sfn $PWD/$r/04_dataset $V1/$r/04_dataset
  find $PWD/$r/firms_old_narrow -name '*.csv' ! -newermt '2026-08-30 20:47' -exec ln -sfn {} $V1/$r/firms/ \;
done
ROOTS="$V1/data $V1/data_202408 $V1/data_202409 $V1/data_202509"
```

**2. Artifacts.** Put `mobilenetv3s_fire_best.pt` and `mobilenetv3s_fire_int8.tflite` in
`models/thermal/frp20000/`, then `make verify-models` (sha256 against `models/MANIFEST.json`).

**3. Test evaluation, FP32 and INT8** (Table 6):

```bash
M=models/thermal/frp20000
python -m nina_thermal.evaluation.pipelines.run_evaluate --roots $ROOTS \
    --checkpoint $M/mobilenetv3s_fire_best.pt --training_report $M/training_report.json \
    --frp_threshold 20000 --tflite $M/mobilenetv3s_fire_int8.tflite \
    --tflite_threshold 0.28752797842025757 --out_dir /tmp/thermal_eval
```

Expected, profile B, 82 patches (36 fire): FP32 recall 0.778, precision 0.651, ROC-AUC 0.787
[0.644, 0.943], PR-AUC 0.702; INT8 recall 0.972, precision 0.547, ROC-AUC 0.694, PR-AUC 0.539.
The output equals `models/thermal/frp20000/evaluation_summary.json` (FP32 scores bit-exact).

**4. Acceptance gate** (validation, 36 patches / 11 granules / 17 fire):

```bash
python -m nina_thermal.quantization.pipelines.run_gate --roots $ROOTS \
    --checkpoint $M/mobilenetv3s_fire_best.pt --tflite $M/mobilenetv3s_fire_int8.tflite \
    --out /tmp/thermal_gate.json
```

Exit code 1: the gate **fails 3 of 4** (recall drop 0 passes; F-beta drop 0.047 > 0.010,
SQNR 11.0 dB < 30 dB, KL 3.90 > 0.05 nats). Same as the `acceptance` block in
`models/thermal/frp20000/quantization_report.json` and `gate_validation.json`.

**5. Dataset manifests** (Table 5):

```bash
python -m nina_thermal.training.pipelines.run_dataset_manifests --roots $ROOTS \
    --out_dir data/manifests --prefix thermal_v1 --firms_bbox=-67.5,-20.5,-59.7,-11.8
python -m reproduce.table5 thermal && python -m reproduce.table6 thermal && python -m reproduce.thresholds thermal
```

## Regeneration: what is and is not bit-exact

`docs/RUNBOOK_regenerate_frp20000.md` regenerates the folder from the checkpoint.
Checked on 2026-10-22 with the v1 data view above:

- **FP32 evaluation: bit-exact.** `main`'s code gives the same per-patch scores as the
  2026-08-30 run, so the same metrics and CI (seed 42 bootstrap).
- **FP32 training:** seeded (42) and expected to be deterministic. It needs the v1 data view
  (old FIRMS) and v1-era label code: later code (coverage guards, low-activity negatives)
  builds a different dataset. Not re-run for #15.
- **QAT / INT8: not bit-exact.** QAT on GPU is not bit-reproducible (TRAINING_REPORT_v3 §8.4).
  A second quantization of the same checkpoint on 2026-09-03 gave a different file
  (`fdbdf938…`, val threshold 0.181; see `models/thermal/frp20000/REQUANTIZATION_NOTE.md`).
  A regenerated INT8 will not reproduce `6e486fc7…`. This is why the HIL-validated file is
  published as an artifact, not regenerated.

The HIL firmware compares the INT8 score with a fixed threshold of 0.5, not 0.2875;
Table 8 compares board and PC at that same 0.5.

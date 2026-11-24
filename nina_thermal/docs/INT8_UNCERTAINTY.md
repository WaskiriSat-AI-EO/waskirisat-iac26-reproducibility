# INT8 test metrics with uncertainty (v1 and v4)

**Date:** 2026-11-22 · **Code:** `evaluation/evaluate.py::int8_metrics_block`, tests in
`evaluation/tests/test_int8_block.py`

Each INT8 test block now carries the same uncertainty as the fp32 block:

- **Confidence intervals:** granule-level cluster bootstrap, 1000 resamples,
  90% (`metrics.cluster_bootstrap_ci`). Granules are resampled rather than
  patches because 50% tile overlap makes neighbouring patches share pixels.
- **Projected precision:** Bayes' rule at prevalences π = 0.10, 0.50 and 0.90,
  with precision(π) = π·TPR / (π·TPR + (1−π)·FPR) (`metrics.precision_at_prevalence`).

Profile B (flight configuration). Each model uses its own validation threshold.
Point estimates are followed by the 90% CI in brackets.

| model | threshold | n / fire | recall | precision | F-β (β = √3) | ROC-AUC | precision @ π = 0.10 / 0.50 / 0.90 | TP / FP / FN / TN |
|---|---|---|---|---|---|---|---|---|
| v1 fp32 | 0.4564 | 82 / 36 | 0.778 [0.675, 0.889] | 0.651 [0.386, 0.941] | 0.742 [0.598, 0.849] | 0.787 [0.644, 0.943] | 0.209 / 0.705 / 0.955 | 28 / 15 / 8 / 31 |
| v1 INT8, HIL/paper (`6e486fc7…`) | 0.2875 | 82 / 36 | 0.972 [0.913, 1.000] | 0.547 [0.319, 0.816] | 0.814 [0.638, 0.930] | 0.694 [0.508, 0.943] | 0.146 / 0.607 / 0.933 | 35 / 29 / 1 / 17 |
| v1 INT8, re-run 2026-09-03 (`fdbdf938…`) | 0.1809 | 82 / 36 | 1.000 [1.000, 1.000] | 0.514 [0.305, 0.772] | 0.809 [0.637, 0.931] | 0.660 [0.474, 0.926] | 0.131 / 0.575 / 0.924 | 36 / 34 / 0 / 12 |
| v4 fp32 | 0.6266 | 714 / 359 | 0.958 [0.922, 0.987] | 0.831 [0.781, 0.876] | 0.923 [0.891, 0.949] | 0.945 [0.918, 0.967] | 0.351 / 0.829 / 0.978 | 344 / 70 / 15 / 285 |
| v4 INT8 (`954388ef…`) | 0.4818 | 714 / 359 | 0.961 [0.933, 0.985] | 0.804 [0.751, 0.853] | 0.916 [0.887, 0.940] | 0.935 [0.909, 0.956] | 0.311 / 0.802 / 0.973 | 345 / 84 / 14 / 271 |

## Reading the table

- **v1 is too small to measure the INT8 loss.** The test set has 21 granules,
  and the ROC-AUC intervals are 0.3–0.45 wide. fp32 [0.64, 0.94] and both INT8
  artifacts overlap almost entirely. On this set the 0.787 → 0.694 drop cannot
  be told apart from noise. Both INT8 artifacts reach recall near 1.0 only
  because their thresholds are low. They pay for it in false alarms: 29–34 FP
  against 15 for fp32, and precision at π = 0.10 is 0.13–0.15 against 0.21.
  The re-run's recall interval [1.000, 1.000] is degenerate: every fire scores
  above 0.18.
- **v4 resolves it.** The test set has 245 granules, and the intervals are
  0.05–0.10 wide. INT8 keeps recall (0.961 against 0.958). It costs 14 more
  false alarms and about 0.01 ROC-AUC (0.935 against 0.945), inside
  overlapping intervals.
- **Operational precision depends on season.** When fire is rare (π = 0.10),
  only about 1 alarm in 3 is a fire, even for v4. At the Aug–Sep peak
  (π = 0.90) it is above 0.97.

## Which v1 artifact is which

The test ROC-AUC of 0.694 quoted for v1 INT8 belongs to the **HIL/paper**
artifact (`6e486fc7…`, in `nina_thermal_models_backup/` only). The gate
numbers quoted with it (SQNR 8.3 dB, KL 11.07 nats, F-β drop 5.8 pp) belong
to the **2026-09-03 re-run** (`fdbdf938…`,
`models/nina_thermal_frp20000/quantization_report.json`). On test, the re-run
scores ROC-AUC 0.660. Both artifacts come from the same fp32 checkpoint
`fe7ff10f…`.

## Reproduce

**v1:** use `main`'s code (the v4 branch's FIRMS footprint changes v1's
labels: 36 test patches, all fire) and the pre-2026-08-30 FIRMS view.

```bash
V=../v1_view_pre0830; M=models/nina_thermal_frp20000
python -m nina_thermal.evaluation.pipelines.run_evaluate \
  --roots $V/data $V/data_202408 $V/data_202409 $V/data_202509 \
  --checkpoint $M/mobilenetv3s_fire_best.pt --training_report $M/training_report.json \
  --frp_threshold 20000 \
  --tflite <int8.tflite> --tflite_threshold <0.28752797842025757 HIL | 0.18085384368896484 re-run> \
  --out_dir $M/eval_int8_<hil|rerun>
```

fp32 reproduces the v1 report bit for bit (82 patches, 36 fire, ROC-AUC
0.786836). Outputs: `models/nina_thermal_frp20000/eval_int8_{hil,rerun}/`.

**v4:** `models/nina_thermal_frp20000_v4_abc_qat_sel/eval_all/evaluation_summary.json`
(command in `TRAINING_REPORT_v4.md`).

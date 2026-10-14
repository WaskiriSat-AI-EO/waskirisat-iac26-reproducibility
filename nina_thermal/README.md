# nina_thermal — NINA fire classifier

**Owner:** Osvaldo. Pipeline: `datagen/` → `training/` → `quantization/` → `evaluation/`.
Design rationale: `docs/CLAUDE_CODE_BRIEF.md`; training: `docs/TRAINING_REPORT.md`; regeneration: `docs/RUNBOOK_regenerate_frp20000.md`.

## Delivery checklist
- [ ] `configs/paper.yaml`: FRP threshold (20000), seeds, epochs (QAT budget 40), the model (MobileNetV3-Small @ 160×120)
- [ ] Verify the runbook regenerates `../models/thermal/frp20000/` from a clean checkout; note what is bit-exact and what is not (QAT)
- [ ] FP32 checkpoint (`mobilenetv3s_fire_best.pt`) and INT8 `.tflite` → release; `.tflite` sha256 must match `6e486fc7…01e8` (already the HIL artifact)
- [ ] Evaluation: ROC-AUC 0.787 (granule-level 90% CI 0.644–0.943), PR-AUC 0.702, thresholds 0.456 (FP32) / 0.181 (INT8) via `evaluation/pipelines/run_evaluate.py`
- [ ] Reproduce the acceptance gate: it **fails 3 of 4** criteria. Keep the result visible, do not tune it away
- [ ] Dataset composition for Table 5: `../data/manifests/` (granule lists, split registry, FIRMS extraction dates Aug–Sep 2023–2025)
- [ ] `make table6` and `make table5` produce the paper values for this payload

## Run
_TODO (Osvaldo): commands from clean checkout to the INT8 artifact and its evaluation._

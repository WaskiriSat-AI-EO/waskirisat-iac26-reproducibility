# nina_cloud — IQC cloud/clear classifier

**Owner:** Leyla. Pipeline: `datagen/` → `training/` → `quantization/` → `evaluation/`.

## Delivery checklist
- [ ] `configs/paper.yaml`: the exact configuration of the paper run (width multiplier, **α = 0.75** per the tflite graph — the code here still documents α = 1.0 in `docs/quantization_methodology.md`; reconcile), seeds, `pos_weight`, epochs, input resolution/normalisation
- [ ] Commit hash (or branch) of the code that produced the delivered model, if not this repo
- [ ] FP32 checkpoint and INT8 `.tflite` → `../models/cloud/` + entry in `../models/MANIFEST.json` (sha256 of INT8 must be `f528086c…6d5e`, the file used in HIL)
- [ ] Evaluation report: ROC-AUC / PR-AUC per degradation profile A/B/C (paper: ROC-AUC 0.976–0.980, PR-AUC 0.970–0.977) via `evaluation/pipelines/run_evaluate.py`
- [ ] Derivation of the INT8 threshold t* = 0.934 (script or report)
- [ ] Dataset composition for Table 5 (tiles, scenes, per-split counts, leakage audit): `../data/manifests/`
- [ ] Preprocessing contract for the device path (RGB565 + resize difference, see paper §6 / C-19): document the exact training preprocessing
- [ ] `make table6` and `make table5` produce the paper values for this payload

## Run
_TODO (Leyla): commands from clean checkout to the INT8 artifact and its evaluation._

# Thermal INT8 gate failure — diagnosis (Task A1)

**Scope:** v1 (`models/nina_thermal_frp20000`, paper/HIL) and v4 (`models/nina_thermal_frp20000_v4_abc`).
**Branch:** `feat/thermal-int8-diagnosis` (stacked on `feat/v4-data-expansion`). Raw sweep outputs: `docs/int8_diagnosis/`.

> **Resumen (ES).** El INT8 de v1 falla porque nunca se entrenó con QAT real: el cuantizador no recibía
> `is_qat=True`, 34 BatchNorm quedaron sin plegar y los observadores seguían actualizándose durante la
> validación. Con activaciones por tensor, la señal se pierde en la primera convolución depthwise (error
> relativo 31 %, salida final 178 %). Corregido el pipeline (ramas v3/v4), QAT recupera el v1 (AUC test
> 0.81–0.93 vs fp32 0.787) y el v4 INT8 queda a 0.01–0.025 AUC del fp32 en test (IC solapados) y **pasa F-β, recall y KL**. El único
> criterio que sigue fallando es SQNR ≥ 30 dB, y **también lo falla un ajuste fino fp32 sin cuantizar**
> (10–16 dB): ese umbral mide la deriva del ajuste fino, no el error de cuantización. Propuesta: medir
> SQNR/KL contra el grafo fake-quant y mantener F-β/recall contra fp32.

## 1. Two artifacts were being mixed

The gate numbers in the task (F-β drop 5.8 pp, SQNR 8.29 dB, KL 11.07, val 36 / 17 fire) come from the
**2026-09-03 re-run** (`models/nina_thermal_frp20000/mobilenetv3s_fire_int8.tflite`, sha `fdbdf938…`,
threshold 0.181). The test ROC-AUC 0.694 belongs to the **paper/HIL** artifact (sha `6e486fc7…`,
`nina_thermal_models_backup/`, threshold 0.288); the re-run scores 0.660. Both come from the same fp32
checkpoint (`fe7ff10f`, test ROC-AUC 0.787 [0.644, 0.943], reproduced exactly).

## 2. Cause (v1)

1. **No real QAT.** `build_quantizer` was never passed `is_qat=True`; under torch 2.12 the 34
   BatchNorm nodes stayed unfolded, and observers kept updating while validation scored the model.
   The exported graph has 234 float32 ops (BN arithmetic as SUB/MUL/ADD) against 77 INT8 ops; v4 has 44 vs 97.
   (Documented in `TRAINING_REPORT_v3.md` §8; fixed on v3/v4 with `freeze_observers` and checkpoint
   selection on real-INT8 validation AUC.)
2. **Where the signal dies.** Per-node trace (PTQ, INT8 vs fp32, mean of 8 val images,
   `run_quant_diagnose --trace`): stem conv 3.6 % → **first depthwise conv 31 %** (first jump) →
   block-2 convs 69–97 % → logit **178 %** (logit correlation 0.40, SQNR 1.0 dB). Observers alone
   track fp32 at corr 0.9999, so the loss is rounding, not export.
3. **Why there.** Activations are per-tensor uint8. The stem output channels span 14.9× (max/median):
   6 % of channels sit below one INT8 step and 37.5 % below four. Folded depthwise weights span
   18.6× / 10.8× / 44.5×. Small-range channels are quantized to a handful of levels; without QAT the
   network never learns to compensate.

**Not the cause — calibration vs AGC.** Calibration uses all 219 v1 training images (augmented loader).
The input QUANTIZE sits after normalization and its range clips above ~314–322 K (v1) / 323–332 K (v4),
within the AGC 258.15–358.15 K. Val/test pixels peak at 305–315 K (0.001 % of test pixels > 314 K, one
image), so clipping does not explain this failure — but it **is a flight risk** for hot fires (§5).
More calibration images help PTQ (v1 val AUC 0.60 → 0.72; v4 0.62 → 0.74) but never close the gap.

**Float boundary (CPU vs NPU on the N6).** Input is float32, output INT8 (`fully_quantized=false`).
Float islands that would run on the Cortex-M55: v1 — BN arithmetic + input normalization
(SUB 35, MUL 122, ADD 49, HARD_SWISH 18, SUM 10); v4 — input normalization + squeeze-excite
(SUB 1, MUL 20, ADD 13, SUM 10).

## 3. Variant sweep (validation)

Gate: recall drop ≤ 0.5 pp, F-β drop ≤ 1 pp, SQNR ≥ 30 dB, KL ≤ 0.05 nats; SQNR/KL on sigmoid scores
INT8 vs the original fp32. "fp32 ctrl" = identical fine-tuning loop with **no quantization**
(`run_quantize --float_control`): the drift any fine-tune causes.

| model | variant | INT8 val AUC | F-β drop | recall drop | SQNR dB | KL | SQNR vs fake-quant |
|---|---|---|---|---|---|---|---|
| v1 (fp32 0.836, n=36) | PTQ, 16 / 64 / 219 calib imgs | 0.60 / 0.64 / 0.72 | +10.0 / +10.9 / +5.8 | 0 | 12.8 / 13.5 / 14.2 | 5.9–8.3 | 18–20 |
| | QAT e10 lr 5e-5 | 0.64 | +13.7 | 0 | 10.2 | 9.4 | 15.9 |
| | QAT e40 lr 5e-5 (seeds 42/1/2) | 0.83 / 0.89 / 0.83 | +4.7 / +3.6 / +6.9 | 0 | 10.5 / 7.8 / 12.7 | 7.6–14.2 | 16.8–17.6 |
| | QAT e40 lr 1e-5 / 1e-4 | 0.83 / 0.89 | −1.3 / +3.6 | 0 | 11.2 / 7.3 | 6.0 / 13.0 | 15.9 / 17.1 |
| | QAT e100 lr 5e-5 | 0.90 | +2.4 | 0 | 6.2 | 13.7 | 15.0 |
| | **fp32 ctrl e10 / e40** | 0.85 / 0.86 | +3.6 / +2.4 | 0 | **15.6 / 10.4** | **4.0 / 8.3** | — |
| v4 (fp32 0.917, n=1742) | PTQ, 256 / 8274 calib imgs | 0.62 / 0.74 | +12.6 / +9.3 | 0 / −0.4 | 4.3 / 5.7 | 11.7 / 11.8 | 9.8 / 11.3 |
| | QAT e10 lr 5e-5 | 0.900 | +1.5 | −0.1 | 11.3 | 0.131 | 17.1 |
| | QAT e40 lr 1e-5 | 0.895 | +1.8 | −0.1 | 10.2 | 0.184 | 15.3 |
| | QAT e40 lr 5e-5 | 0.906 | +1.2 | −0.4 | 11.7 | 0.044 | 17.0 |
| | **QAT e40 lr 5e-5, all 8274 calib** | **0.908** | **+0.4 ✓** | **0 ✓** | 10.8 ✗ | **0.030 ✓** | 17.4 |
| | **fp32 ctrl e40** | 0.917 | −0.2 | 0 | **13.1** | **0.091** | — |

## 4. Test results (granule bootstrap, 1000 resamples, 90 % CI)

| model | artifact | test ROC-AUC [90 % CI] | FP | notes |
|---|---|---|---|---|
| v1 (82 patches, 36 fire, 21 granules) | fp32 | 0.787 [0.644, 0.943] | 15 | reference |
| | HIL (paper) INT8 | 0.694 [0.508, 0.943] | — | broken pipeline |
| | 2026-09-03 re-run INT8 | 0.660 [0.474, 0.926] | — | broken pipeline |
| | fixed PTQ (all calib) | 0.699 | 27 | |
| | fixed QAT e40 lr 5e-5 (seeds 42/1/2) | 0.843 / 0.910 / 0.863 | 24 / 21 / 30 | |
| | fixed QAT e40 lr 1e-5 / 1e-4 | 0.811 / 0.929 [0.875, 0.988] | 20 / 22 | |
| v4 (profile B, 714 patches, 359 fire) | fp32 | 0.945 [0.918, 0.967] | 70 | recall 0.958; P@π=0.1/0.5/0.9 = 0.35/0.83/0.98 |
| | existing `v4_abc_qat_sel` INT8 | 0.935 [0.909, 0.956] | 84 | recall 0.961; P@π = 0.31/0.80/0.97 |
| | PTQ, 256 / 8274 calib imgs | 0.622 [0.580, 0.664] / 0.815 [0.775, 0.850] | 305 / 223 | recall 0.928 / 0.933 |
| | QAT e10 lr 5e-5 | 0.930 [0.901, 0.954] | 96 | recall 0.955; P@π = 0.28/0.78/0.97 |
| | QAT e40 lr 1e-5 | 0.934 [0.906, 0.957] | 117 | recall 0.967; P@π = 0.25/0.75/0.96 |
| | QAT e40 lr 5e-5 | 0.933 [0.908, 0.956] | 93 | recall 0.950; P@π = 0.29/0.78/0.97 |
| | QAT e40 lr 5e-5, all calib (best on val) | 0.921 [0.889, 0.946] | **78** | recall 0.942; P@π = 0.32/0.81/0.97 |

All v4 QAT variants overlap each other and fp32 within their 90 % CIs (ΔAUC −0.011 to −0.024); the
differences between them are not resolvable on this test set. PTQ alone is not viable for v4.

The v1 "fixed" INT8 models beat their own fp32 on test; that is fine-tuning plus a 36-image threshold on
flawed v1 labels (v3 §11.5: v1 negatives encode geography), not a property of INT8. Treat v1 numbers as
"the fixed pipeline no longer destroys the model", not as accuracy claims.

## 5. Conclusions and recommended fixes

1. **The v1 failure is a pipeline bug, already fixed** on v3/v4 (real QAT, folded BN, frozen observers,
   selection on real-INT8 AUC). v1 is recoverable by re-running QAT on the fixed pipeline, but v1's
   labels make that of limited value; v4 is the model to ship.
2. **v4 INT8 is within 0.01–0.025 test AUC of fp32 (CIs overlap) and passes the gate's recall, F-β
   and KL criteria on validation** with QAT 40 epochs, lr 5e-5, full-train calibration
   (`--calib_batches` ≥ all). On test that run trades 1.6 pp recall for the fewest false positives (78 vs
   fp32's 70); the existing `v4_abc_qat_sel` artifact remains an equally valid choice.
3. **Change the gate.** SQNR ≥ 30 dB / KL ≤ 0.05 against the *original* fp32 cannot be met by any
   fine-tuned model: the fp32 control (no quantization) gets 10–16 dB and KL 0.09 (v4) to 8.3 (v1).
   Proposal: keep recall/F-β drop vs fp32 (the operational criteria); measure SQNR/KL of INT8 **vs the
   fake-quant graph it was trained as** (now reported as `parity.vs_fake_quant`, 15–17 dB on sigmoid
   scores; a 30 dB target needs a logit-space metric or a looser threshold); report KL only when
   validation has ≥ 20 fire (v1's 36-image KL is noise; A3 now flags this).
4. **Input range.** Fix the input quantization to the full AGC range ([0, 1] → 258.15–358.15 K) instead of
   the calibrated range, or take uint8 input with normalization folded into the stem. This removes the
   float input op and the ~315–330 K clip.
5. **Float islands.** Quantize squeeze-excite (or replace with an INT8-friendly variant) to keep v4's
   remaining 44 float ops off the Cortex-M55.

**Caveats.** v1 validation is 36 images (17 fire); its thresholds and gate numbers have very wide error.
QAT is not bit-reproducible on GPU (v1 seeds spread 0.84–0.91 test AUC). Per-channel activation
quantization is not available on the target, so the per-tensor spread issue can only be absorbed by QAT.

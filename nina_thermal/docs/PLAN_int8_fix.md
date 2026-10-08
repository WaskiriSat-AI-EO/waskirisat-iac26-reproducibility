# Implementation plan — fixing the int8 conversion

**Status (2026-10-08): int8 fixed** — root cause was that QAT never ran (config
bug + torch 2.12 incompatibility); see `TRAINING_REPORT_v3.md` §8–§9. Phases 0–2
4 and 6 done (plus an observer-leak fix found on the way), 3 and 5 not needed
for accuracy. Final artifact: `models/nina_thermal_frp20000_v3_bc_qat_sel/`.

**Date:** 2026-10-08 · **Context:** `TRAINING_REPORT_v3.md` §8–§9 · **Model:** `models/nina_thermal_frp20000_v3_bc/`

## Problem statement

The fp32 model (test ROC-AUC 0.959, accuracy 0.894) and its QAT simulation
(val ROC-AUC 0.912, corr 0.987 with fp32) are sound. The int8 artifact is not
(test accuracy 0.540, ROC-AUC 0.717). The loss happens inside `convert_pt2e`,
before TFLite export.

## What is already established (2026-10-03 / 10-08)

| finding | evidence |
|---|---|
| Training and QAT simulation are fine | fake-quant val ROC-AUC 0.912, corr 0.987 |
| TFLite export is faithful to its input | converted-PT2E vs TFLite scores agree (AUC 0.66 vs 0.68) |
| **Defect A — QAT path does not fold BatchNorm.** `convert_pt2e` should fold each BN into its conv; after conversion all **34 `aten.batch_norm` nodes remain** | op count of the converted graph; BN is in eval mode and output is batch-independent, so it is not a train-mode BN issue |
| **Defect B — a second, shared defect.** The PTQ path folds BN correctly (0 BN nodes) and is still broken (AUC 0.66, corr 0.25) | stage-by-stage scoring |
| Error is already large in the first layers | per-node trace (PTQ): conv 6%, first depthwise 19%, >50% from block 2 on; typical int8 is 1–3% |
| Dead squeeze-excite ReLUs calibrate to [0, 0] | widening those ranges alone did not fix the output |
| `ai-edge-quantizer` route has its own calibration defect | averaged per-sample min/max collapses single-element tensors; patched output range, still constant output (also with int16 activations — unexplained) |

## Strategy

Fix B first (it blocks both paths), then A, then harden the pipeline so a
broken conversion can never again pass silently. Run the `ai-edge-quantizer`
route in parallel as a second chance, and keep an architecture fallback.

---

## Phase 0 — Diagnostic harness (½ day)

Make the investigation reproducible instead of ad-hoc scratch scripts.

| task | file |
|---|---|
| `compare_graphs(prepared, converted, images)`: per-node relative error and correlation, in execution order, matched by node name; flags the first node whose error jumps > 3× its input's error | new `nina_thermal/quantization/diagnostics.py` |
| `compare_tflite(float_tflite, int8_tflite, image)`: same for TFLite with `experimental_preserve_all_tensors` | same |
| `stage_report(checkpoint)`: fp32 → fake-quant → converted → TFLite AUC/corr table (what §8.4 of the report shows) | same |
| CLI: `python -m nina_thermal.quantization.pipelines.run_quant_diagnose --checkpoint …` | new pipeline |

**Done when:** one command reproduces the §8.4 table and prints the first
diverging node for PTQ and for QAT.

## Phase 1 — Isolate defect B (1 day)

Each hypothesis is a small, reversible change to the export; measure **converted
vs fake-quant correlation** on validation (target > 0.99). Run them in this
order (cheapest and most likely first):

| # | hypothesis | test |
|---|---|---|
| H1 | **In-place ops** (`hardswish_`, `relu_`, `add_`) mutate tensors that inserted quantize/dequantize nodes still reference | set `inplace=False` on every activation module (and replace `+=` residuals) before export |
| H2 | **Squeeze-excite** (`adaptive_avg_pool → conv → relu → conv → hardsigmoid → mul`) annotated or converted wrongly | exclude `SqueezeExcitation` from the quantizer (keep it float), re-measure |
| H3 | **`hardswish` / `hardsigmoid`** have no correct quantized pattern in this quantizer | exclude those op types from annotation |
| H4 | **`fold_quantize=False`** leaves a graph whose q/dq semantics differ from the fake-quant graph | compare with `fold_quantize=True` in PyTorch only (TFLite export may reject it — that's fine for the test) |
| H5 | **Degenerate ranges** (dead units, all-zero weight channels) combined with any of the above | apply the range-widening patch on top of the winning change |
| H6 | **Library bug** in `torchao` 0.17 / `torch` 2.12 PT2E convert | reproduce on torchvision's stock `mobilenet_v3_small`; try the previous `torchao` release in a throwaway venv |

**Decision rule:** the first hypothesis that brings correlation > 0.99 becomes
the fix; if none does, H6 tells us whether it's upstream (→ pin a working
version, or go to Phase 3 / 5).

## Phase 2 — Fix defect A: BatchNorm folding in QAT (½–1 day)

Two options; try in order:

1. **Explicit fold after convert:** call torchao's QAT conv-BN fold pass
   directly, or locate why its pattern match fails on `aten.batch_norm.default`
   (the export in torch 2.12 emits this op rather than the legacy
   `_native_batch_norm_legit*` the matcher may expect).
2. **Fold before QAT (standard fallback):** fold BN into conv weights in the
   eager fp32 model, then export and run QAT on the BN-free graph. Loses BN
   statistics adaptation during QAT, which matters little for a 40-epoch
   fine-tune from a converged checkpoint.

**Done when:** converted QAT graph has 0 BN nodes and matches fake-quant
(corr > 0.99).

## Phase 3 — Parallel route: `ai-edge-quantizer` (1 day, alongside 1–2)

Attractive because it yields **int8 input/output** (full-NPU graph, open issue #6).

| task | detail |
|---|---|
| Fix calibration | override zero-width QSVs (single-element tensors, dead units) before `quantize()` |
| Explain the constant output | trace the int16 run node by node; it should be near-lossless, so a constant output means another collapsed tensor or an I/O scaling error in our scoring |
| QAT-trained weights | once Phase 2 works, export the QAT-trained float weights to float TFLite and quantize them with this route |

**Done when:** int8-I/O artifact passes the gate, or the remaining blocker is identified.

## Phase 4 — Harden the pipeline (1 day, after a working conversion)

This is where the supervisor's "train with quantization" requirement lands:
measure the **real** int8 model during training, not just the simulation.

| task | file |
|---|---|
| **Conversion parity guard:** after `convert`, score converted vs fake-quant on val; abort with a clear error if corr < 0.99 | `run_quantize.py` |
| **Real-int8 checkpoint selection:** every N QAT epochs, convert a copy, score it on val, keep the epoch with the best int8 ROC-AUC (not the last epoch) | `run_quantize.py`, `qat.py` |
| **int8 test-set evaluation** in the standard evaluator (today there is no pipeline path) | `run_evaluate.py --tflite …` |
| Regression tests: tiny MobileNetV3-like model with SE + hardswish + BN; assert converted ≈ fake-quant | `quantization/tests/` |
| Runbook + report update | `RUNBOOK_*`, `TRAINING_REPORT_v3.md` §8 |

## Phase 5 — Fallback: quantization-friendly backbone (2–3 days, only if 1–3 fail)

Retrain MobileNetV3-Small **without squeeze-excite and with ReLU instead of
hardswish** (or a comparable small backbone), same v3 data and protocol.
Expect a small fp32 drop; in exchange the graph contains only conv/BN/ReLU/add,
which PT2E and TFLite quantize reliably.

## Phase 6 — Validate and report (½ day)

- `run_quantize` → gate verdict; int8 scored on test A/B/C.
- Update `TRAINING_REPORT_v3.md` §8 tables; checksum; mark artifact deployable.

---

## Success criteria

| criterion | target |
|---|---|
| converted vs fake-quant correlation (val) | > 0.99 |
| acceptance gate | recall drop ≤ 0.5 pp, F-beta drop ≤ 1 pp, SQNR ≥ 30 dB (KL tolerance to be recalibrated for n = 256) |
| int8 test ROC-AUC, profile B | ≥ 0.949 (within 0.01 of fp32 0.959) |
| int8 test accuracy, profile B | within 2 pp of fp32 (0.894) |
| artifact size | ≤ 2 MB; int8 I/O preferred |

## Schedule

| phase | effort | depends on |
|---|---|---|
| 0 Harness | ½ day | — |
| 1 Defect B | 1 day | 0 |
| 2 Defect A | ½–1 day | 0 (independent of 1) |
| 3 aeq route | 1 day | 0; parallel |
| 4 Harden | 1 day | a working conversion from 1–3 |
| 5 Fallback | 2–3 days | only if 1–3 fail |
| 6 Validate | ½ day | 4 |
| **Total** | **≈ 4–5 days**, +2–3 if fallback | |

## Risks

| risk | mitigation |
|---|---|
| Defect is upstream in `torchao`/`litert-torch` and not patchable locally | H6 identifies it early; pin a working version or use Phase 3 |
| Fix works in PyTorch but TFLite export rejects the graph | Phase 3 route starts from a float TFLite and avoids PT2E export entirely |
| int8 still loses a few points after the fix | Phase 4 real-int8 checkpoint selection; Phase 5 backbone |

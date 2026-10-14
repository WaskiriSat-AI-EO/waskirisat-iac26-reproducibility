# Model card — <payload> classifier

| | |
|---|---|
| Task | <e.g. cloud / clear tile classification> |
| Architecture | <backbone, width multiplier, input shape/layout> |
| Artifacts | FP32 `<file>` (sha256 …), INT8 `<file>` (sha256 …) — see `models/MANIFEST.json` |
| Code / config | commit `<sha>`, `configs/paper.yaml` |
| Owner | <name> |

## Intended use
<what it is for in the mission; what decision it supports>

## Not intended for
<explicit non-uses, e.g. not flight-cleared, not validated on real detector imagery>

## Training data
<one paragraph + link to DATASET_CARD.md>

## Performance (with evidence tier)
| Metric | Value | Split / profile | Tier (T1–T4) | Reproduced by |
|---|---|---|---|---|
| ROC-AUC | | | T2 | `make table6` |

## Quantization and thresholds
<method, parity result, acceptance-gate verdict, threshold and how it was derived>

## Known limitations
<synthetic data; preprocessing differences on device; gate result; anything the paper states>

## Ladder status
<rungs V0–V7 claimed for this model, from docs/evidence-and-ladder.md>

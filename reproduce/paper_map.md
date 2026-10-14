# Paper → code map

Numbering follows the submitted manuscript (9 tables, 2 figures; source: `Transcribe/` of the vault, D-038). Fill "Command" and "Status" as each owner delivers. `expected/` holds the paper's values (JSON) that CI compares against.

| Paper item | Content | Payload | Owner | Command | Inputs | Status |
|---|---|---|---|---|---|---|
| Fig. 1 | Payload block diagram | — | Christian | `make figures` | `reproduce/make_figures.py` | todo |
| Fig. 2 | Dataset synthesis pipeline | — | Christian | `make figures` | same | todo |
| Table 1 | Payload specification (T1/T2/T3 per row) | both | Christian | n/a (cited datasheets) | `docs/` | todo |
| Table 2 | Degradation profiles A/B/C (all T3) | both | Leyla / Osvaldo | `make table2` | `shared/`, `*/datagen/config.py` | todo |
| Table 3 | Evidence tiers | — | Christian | n/a | `docs/evidence-and-ladder.md` | todo |
| Table 4 | Verification ladder and status | — | Christian | n/a | `docs/evidence-and-ladder.md` | todo |
| Table 5 | Dataset composition (thermal 219/36/82 frames; cloud val 998, test 1,408 per profile) | both | Leyla / Osvaldo | `make table5` | `data/manifests/` | todo |
| Table 6 | Test performance (T2): thermal FP32 / INT8, cloud INT8 A/B/C | both | Leyla / Osvaldo | `make table6` | `models/`, test split | todo |
| §4 | Thresholds (cloud t*=0.934; thermal 0.456 → 0.181) and acceptance gate | both | Leyla / Osvaldo | `make parity` | `models/` | todo |
| Table 7 | Surrogate execution: latency, arena, min free heap | both | Misael | `make table7` | `hil/results/test1/` | **done** (computed from raw HIL results) |
| Table 8 | Device-path fidelity (94/100, 69/100, 97/100) | both | Misael | `make table8` | `hil/results/test1/`, `models/` | **partial**: rows 1-2 computed from raw HIL results; row 3 (97/100) needs the exact-tensor raw data |
| §6.6 | OBC integration (Test 2): 199 samples, 117,840 packets, 0 integrity errors | both | Misael | `make test2-summary` | `hil/results/test2/` | **partial**: 185/185 reproduced; the paper's 199/200 needs the earlier interrupted logs |
| Table 9 | Deferred campaign | — | Christian | n/a | `docs/limitations.md`, `docs/deferred-protocols/` | todo |

Note: Table 5 has no cloud training row in the paper. State the training counts in `DATASET_CARD.md` anyway.

Evidence tier (T1–T4) and ladder rung (V0–V7) for each item: see `docs/evidence-and-ladder.md`.

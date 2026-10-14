# Paper → code map

Fill the "Command" and "Status" columns as each owner delivers. `expected/` holds the paper's values (JSON) that CI compares against.

| Paper item | Content | Payload | Owner | Command | Inputs | Status |
|---|---|---|---|---|---|---|
| Table 5 | Dataset composition | both | Leyla / Osvaldo | `make table5` | `data/manifests/` | todo |
| Table 6 | Test performance (T2), FP32, profiles A/B/C | both | Leyla / Osvaldo | `make table6` | `models/`, test split | todo |
| §4 | INT8 parity, thresholds (cloud t*=0.934; thermal 0.456→0.181), gate | both | Leyla / Osvaldo | `make parity` | `models/` | todo |
| Table 7 | Surrogate execution: latency, arena, free heap | both | Misael | `make table7` | `hil/results/test1/` | todo |
| Table 8 | Device-path fidelity vs desktop (94 / 69 / 97 %) | both | Misael | `make table8` | `hil/results/test1/`, `models/` | todo |
| §6 (Test 2) | OBC integration: 199/200 reproduced, 117,840 packets | both | Misael | `make test2-summary` | `hil/results/test2/` | todo |
| Fig. 1 | Payload block diagram | — | Christian | `make figures` | `reproduce/make_figures.py` | todo (move script) |
| Fig. 2 | Synthesis pipeline | — | Christian | `make figures` | same | todo |

Evidence tier (T1–T4) and ladder rung (V0–V7) for each item: see `docs/evidence-and-ladder.md`.

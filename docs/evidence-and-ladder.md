# Evidence tiers and verification ladder

This page mirrors §5 of the paper. Every number in the paper carries one evidence tier (T1–T4) and, when it concerns verification, sits on one rung of the ladder (V0–V7). The flight hardware (OpenMV N6, PAG7936 RGB sensor, FLIR Lepton 3.5) was not available at submission, so the paper makes the evidential status of each claim explicit instead of weakening the claims silently.

## Evidence tiers

| Tier | Name | Definition | Admissible as |
|---|---|---|---|
| **T1** | Traceable | Calibrated third-party measurement or datasheet constant; a reader can verify it without access to this project | Evidence |
| **T2** | Modeled | Documented transform of T1 quantities; correct in form, uncertain in parameter | Evidence, with stated uncertainty |
| **T3** | Assumed | Parameter chosen without direct measurement; where ambiguous, the value that makes the classifier's task harder | Assumption, never evidence |
| **T4** | Deferred | Needs flight hardware or an environmental facility; stated as a requirement with an acceptance criterion | Requirement only |

Rules: a result inherits the worst tier of its inputs (an exact metric on synthetic data is T2); T3 values appear in a bracketing sweep, not as a point estimate; T4 claims never appear in a results table.

## Verification ladder

The ladder is monotonic: each rung presupposes the ones below it. The paper **claims V0–V5 and specifies V6–V7**.

| Rung | Name | What it establishes | Status in this repo |
|---|---|---|---|
| V0 | Unit / physics invariants | Each pipeline step is correct and conserves what it claims to | `make test` |
| V1 | Dataset validation | Labels agree with the authoritative source, no leakage between splits | `data/manifests/` (thermal), `nina_cloud_results/dataset/` (cloud) |
| V2 | Model-in-the-loop | FP32 model meets its metric on held-out synthetic data across profiles A/B/C | Table 6 (`make table6`) |
| V3 | Numerics-in-the-loop | INT8 preserves the metric within the acceptance gate | §4 thresholds (`make parity`); **thermal gate fails 3 of 4** |
| V4 | Toolchain-in-the-loop | The artifact compiles for the N6 and maps to the NPU | Not run (scope cut); see `docs/deferred-protocols/` |
| V5 | Surrogate hardware-in-the-loop | The artifact executes on real STM32 silicon through the flight deployment API, under OBC control over a framed link | `hil/`, Tables 7 and 8, `make test2-summary` |
| V6 | Target hardware-in-the-loop | Real N6 and flight sensors | Deferred |
| V7 | Environmental qualification | TVAC, vibration, radiation | Deferred |

### Why V5 bounds the target

The OpenMV H7 Plus shares the flight board's vendor, its `ml.Model()` deployment API and its INT8 TFLite format, with strictly weaker compute (Cortex-M7, TFLM interpreter, no NPU). For a quantity that is monotone in compute (latency, peak memory), the surrogate measurement is an upper bound for the N6. Energy is not monotone in capability and is **not** claimed.

The artifact is shared between surrogate and target; the execution engine is not. The paper says so, and so does this repo.

## What reproduces without hardware

| Paper item | Reproducible here | Needs |
|---|---|---|
| Tables 5, 6 and §4 thresholds | Yes, from committed reports and manifests | Nothing beyond `pip install -r requirements.txt` |
| Tables 7, 8 and §6.6 | Yes, recomputed from the raw HIL results in `hil/results/` | Nothing (the board is only needed to *regenerate* the raw results) |
| Fig. 1 and 2 | Yes (`make figures`) | Nothing |
| Re-training the models | Partly: the recipe and configs are here; QAT is not bit-reproducible on GPU | A GPU and the datasets (see each payload's `DATASET_CARD.md`) |
| Re-running the HIL tests | No | OpenMV H7 Plus + ESP32, see `hil/ENVIRONMENT.md` |
| V4, V6, V7, energy | No, deferred | N6 hardware and facilities, see `docs/deferred-protocols/` |

See `reproduce/paper_map.md` for the exact command and status of every table and figure, and `docs/limitations.md` for the limits the paper declares.

# Known limitations

The limits below are stated in the paper or were found while preparing this repository. Each one says what it affects and where to look.

## Declared in the paper

- **Synthetic data only.** Both models are trained and tested on data synthesized from Sentinel-2 (cloud) and Sentinel-3 SLSTR + NASA FIRMS (thermal) through degradation profiles A/B/C. The profile multipliers are assumptions (T3). Profile C is a noise-equivalent proxy for South Atlantic Anomaly transit, not a radiation-transport model. No real PAG7936 or Lepton frames have been used; comparing them with the synthetic output is the V6 fidelity check.
- **Thermal INT8 fails the acceptance gate (3 of 4).** On the HIL artifact `6e486fc7…01e8`, the recall drop passes (0), while the F-beta drop (0.047), SQNR (11.0 dB) and KL (3.90 nats) fail. The artifact is published as the HIL-validated file and is **not released for flight**. Evidence: `models/thermal/frp20000/gate_validation.json`.
- **Cloud on-device preprocessing differs from training (C-19).** The device path uses RGB565 plus a resize, which is not numerically equivalent to the training preprocessing. The thermal path does match. Table 8 reports the resulting device fidelity (94 / 69 / 97 %).
- **V5 uses a surrogate board.** The OpenMV H7 Plus runs the artifact through TFLM, not the N6 Neural-ART engine. Latency and peak memory are upper bounds on the target. Energy was not measured.
- **Fault paths were not exercised.** The HIL campaign had no fault injection, and the protocol's error-detection paths were never triggered on hardware because no error occurred. Emulated SEU injection was a scope cut.
- **Test 2 reproduces 199 of 200 samples** (117,840 packets, 0 integrity errors), with two payload-side aborts whose root cause is open.

## Found while preparing the repository

- **June 2023 has no FIRMS data.** The thermal v1 dataset contains 42 June 2023 granules. Their patches were labelled "no fire" because no detections exist, not because no fire was observed. They supply all 46 test negatives, 17 of 19 validation negatives and 138 of 152 train negatives, while every positive comes from August or September. The model may therefore separate seasons rather than fire. The paper's wording "FIRMS Aug–Sep 2023–2025" must be read with this in mind.
- **The thermal v1 headline should not be read as a fire-detection measure.** The v3 report (PR #26) states that ROC-AUC 0.787 should not be cited as one, for the label problems above plus a wide-bbox FIRMS query and mixed NRT/SP products. v3 and v4 fix them (PRs #26 and #27, drafts), but they are **not the paper's model** and are not part of this release.
- **The QAT acceptance gate is undefined for QAT models.** In v3 and v4 the INT8 test metrics match FP32 while SQNR and KL still fail, so the gate criteria may not suit QAT. Until it is defined, no INT8 thermal artifact is released for flight.
- **QAT is not bit-reproducible on GPU.** The thermal INT8 was re-quantized on 2026-09-03 and produced a different file (`fdbdf938…`, threshold 0.181) than the HIL one. See `ERRATA.md`.
- **Warm-scene confound.** Scene mean temperature alone reaches test ROC-AUC 0.903 on the v3 test set, so part of the thermal skill may be scene temperature.
- **Datasets are not yet on an open archive.** The cloud dataset is available through a provisional OneDrive link (see `nina_cloud/DATASET_CARD.md`). Both datasets will move to an open repository such as Hugging Face; until then reproducing training needs a request to the owners. Evaluation and tables reproduce from the committed reports and manifests.
- **Binaries live partly in git.** `nina_cloud_results/` carries about 76 MB of `.keras`, `.pt` and `.tflite` files, contrary to the policy in `models/MANIFEST.json`. They are kept as delivered.
- **No DOI yet.** Zenodo archiving is pending; see `CITATION.cff`.

## Deferred (Table 9)

V4 static analysis, V6 target HIL, V7 environmental qualification, power and energy measurement, and SEU injection are specified in `docs/deferred-protocols/` with acceptance criteria.

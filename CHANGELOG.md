# Changelog

## iac26-v1.0 (2026-10-25)

First release of the reproducibility repository for the IAC-26 paper *Onboard Edge AI Wildfire Monitoring*.

### Included
- Cloud (NINA Cloud / IQC): delivered training, quantization and evaluation results, model card, dataset card, `paper.yaml`, and the Table 5/6 and §4 reproduction.
- Thermal (NINA Thermal): v1 pipeline, the HIL-validated INT8 artifact with its failed acceptance gate, evaluation, dataset manifests, cards, `paper.yaml`.
- HIL: delivered OpenMV/ESP32 code and results of Test 1 and Test 2, environment, protocol and catalogue documentation, and the recomputation of Tables 7 and 8.
- `reproduce/`: one command per paper item, compared against the paper's values in `reproduce/expected/`.
- `docs/`: evidence tiers and verification ladder, limitations, deferred campaign.
- `ERRATA.md`: discrepancies between sources.

### Not included (declared)
- Hosting of datasets and binaries on an open archive (Hugging Face or similar), and the Zenodo DOI. The cloud dataset is reachable through a provisional OneDrive link.
- Thermal v3 and v4 (PRs #26 and #27, drafts): corrected labels and real QAT, not the paper's model.
- Definition of the acceptance gate for QAT models.
- Table 2 reproduction script, Dockerfile, thermal FP32 checkpoint.
- V4, V6, V7, energy and SEU: see `docs/deferred-protocols/`.

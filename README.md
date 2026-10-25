# WaskiriSat — AI payload: models, data pipelines and reproducibility

Code, models and verification artifacts behind the IAC-26 paper
**"Onboard edge AI for wildfire monitoring using visible and LWIR imagery on Bolivia's first nanosatellite"** (IAC-2026, paper ID 111635, session B4.4) on the
two AI payloads of WaskiriSat, Bolivia's first CubeSat (525 km sun-synchronous orbit).
Paper: [IAF Digital Library, IAC-2026 paper 111635](https://dl.iafastro.directory/event/IAC-2026/paper/111635/) (the manuscript PDF there is access-restricted).
Archive of this repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23647727.svg)](https://doi.org/10.5281/zenodo.23647727) (all versions); exact version for the paper: [10.5281/zenodo.23647728](https://doi.org/10.5281/zenodo.23647728) (`iac26-v1.0.1`).

> **Status: frozen at the paper.** This repository is dedicated to the IAC-26 paper and is kept as the reproducibility record of release [`iac26-v1.0.1`](https://github.com/WaskiriSat-AI-EO/waskirisat-pl-aidev/releases/tag/iac26-v1.0.1) (Zenodo DOI above). Later development (newer thermal models, the flight gate, further hardware-in-the-loop work) continues in other repositories of the organization. Known gaps of this record are listed in [`docs/limitations.md`](docs/limitations.md) and [`CHANGELOG.md`](CHANGELOG.md).

| Payload | Sensor | Input | Task | Code |
|---|---|---|---|---|
| **NINA** — thermal | FLIR Lepton 3.5 | 160×120, 3,563 m/px | Wildfire hotspot detection | [`nina_thermal/`](nina_thermal/) |
| **IQC** — visible | PAG7936 RGB | 1280×800, 562.5 m/px | Cloud / clear screening for downlink | [`nina_cloud/`](nina_cloud/) |

Everything is **pre-flight**: datasets are synthetic proxies built by degrading calibrated
Sentinel-2/-3 and NASA FIRMS products to the flight sensors' characteristics. The paper's
claims are graded by evidence tier (T1–T4) and verification rung (V0–V7); see
[`docs/evidence-and-ladder.md`](docs/evidence-and-ladder.md) and the known limits in
[`docs/limitations.md`](docs/limitations.md).

## Repository map

```
nina_thermal/   NINA fire classifier: datagen → training → quantization → evaluation   (owner: Osvaldo)
nina_cloud/     IQC cloud classifier: datagen → training → quantization → evaluation   (owner: Leyla)
hil/            Surrogate hardware-in-the-loop (OpenMV H7 Plus + ESP32 OBC): code and raw results (owner: Misael)
models/         Published artifacts; MANIFEST.json holds the sha256 of every binary
reproduce/      One script per paper table/figure, compared against the paper's values
shared/         Utilities used by both payloads (CDSE access, optics, seeding, logging)
data/manifests/ Versioned dataset manifests and the scene/split registry (data itself is not in git)
docs/           Evidence taxonomy, limitations, deferred campaign, model/dataset card templates
ERRATA.md       Discrepancies between sources and which value prevails
CHANGELOG.md    What each release contains and what it leaves out
```

## Quick start

```bash
git clone git@github.com:WaskiriSat-AI-EO/waskirisat-pl-aidev.git
cd waskirisat-pl-aidev
python3.13 -m venv .venv && source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
make test          # unit tests
make reproduce     # recompute every paper item and compare with the paper
```

`make reproduce` prints PASS / FAIL / PENDING per item. Tables 5, 6 and 7, §4 thresholds, and Table 8
(rows 1 and 2) and the Test 2 summary (185 of the 199 samples) reproduce from the reports, manifests and raw HIL
results committed in the repo. What does not reproduce, and why, is in
[`reproduce/paper_map.md`](reproduce/paper_map.md), [`docs/limitations.md`](docs/limitations.md) and
[`ERRATA.md`](ERRATA.md).

## Reproducing the paper

| Paper item | Command | Status |
|---|---|---|
| Tables 5, 6, §4 thresholds | `make table5 table6 parity` | reproduces (thermal §4 INT8 threshold is 0.2875, see [`ERRATA.md`](ERRATA.md) #2) |
| Tables 7, 8, §6.6 | `make table7 table8 test2-summary` | Table 7 reproduces; Table 8 row 3 and the 199th Test 2 sample need raw data not delivered |
| Table 2 | `make table2` | not implemented (values are in each payload's `datagen/config.py`) |
| Figures 1, 2 | `make figures` | reproduces (SVG) |
| Binary integrity | `make verify-models` | cloud artifacts OK (tracked in git); thermal ones need the Release assets |

`models/MANIFEST.json` records the sha256 of each model artifact and `make verify-models` checks
them. The cloud model files are tracked in git under `nina_cloud_results/`; the thermal binaries
(`.pt`, `.tflite`) are attached to the GitHub Release.

The datasets are **not in git**. The cloud dataset is available through a provisional OneDrive link
(see [`nina_cloud/DATASET_CARD.md`](nina_cloud/DATASET_CARD.md)); both datasets will be moved to an open
archive (Hugging Face or similar), and a Zenodo DOI is pending. Dataset manifests, split registries and
evaluation reports in git are enough to reproduce the tables without the data.

## Rebuilding the datasets

Both pipelines run the same seven stages from the repo root. Copernicus credentials are free at
[dataspace.copernicus.eu](https://dataspace.copernicus.eu); copy `.env.template` to `.env` and fill in
`CDSE_USER` / `CDSE_PASSWORD`.

### NINA thermal (Sentinel-3 SLSTR + FIRMS → Lepton 3.5 patches)

```bash
python -m nina_thermal.datagen.pipelines.run_acquisition --roi nina_thermal/resources/bo_nina_roi_simplified.geojson --start 20230601 --end 20230930 --output_dir data/01_raw
python -m nina_thermal.datagen.pipelines.run_extraction  --input_dir data/01_raw --output_dir data/02_extracted
python -m nina_thermal.datagen.pipelines.run_processing  --input_dir data/02_extracted --output_dir data/03_patches --roi nina_thermal/resources/bo_nina_roi_simplified.geojson --firms_dir data/firms
python -m nina_thermal.datagen.pipelines.run_simulation  --input_dir data/03_patches --output_dir data/04_dataset
python -m nina_thermal.datagen.pipelines.run_split       --dataset_dir data/04_dataset
python -m nina_thermal.datagen.pipelines.run_balance     --dataset_dir data/04_dataset
python -m nina_thermal.datagen.pipelines.run_report      --dataset_dir data/04_dataset
```

### IQC cloud (Sentinel-2 → PAG7936 tiles)

```bash
python -m nina_cloud.datagen.pipelines.run_acquisition --roi nina_thermal/resources/bo_nina_roi_simplified.geojson --start 20230601 --end 20230930 --output_dir data/cloud/01_raw
python -m nina_cloud.datagen.pipelines.run_extraction  --input_dir data/cloud/01_raw --output_dir data/cloud/02_extracted
python -m nina_cloud.datagen.pipelines.run_processing  --input_dir data/cloud/02_extracted --output_dir data/cloud/03_tiles
python -m nina_cloud.datagen.pipelines.run_simulation  --input_dir data/cloud/03_tiles --output_dir data/cloud/04_dataset
python -m nina_cloud.datagen.pipelines.run_split       --dataset_dir data/cloud/04_dataset
python -m nina_cloud.datagen.pipelines.run_balance     --dataset_dir data/cloud/04_dataset
python -m nina_cloud.datagen.pipelines.run_report      --dataset_dir data/cloud/04_dataset
```

Training, quantization and evaluation entry points are documented per payload in
[`nina_thermal/README.md`](nina_thermal/README.md) and [`nina_cloud/README.md`](nina_cloud/README.md).

### Incremental, multi-operator acquisition (cloud)

Split assignment is deterministic and permanent: each scene is hashed to `train` / `val` / `test`
the first time it is registered, and the mapping lives in the scene registry. Re-running or adding
scenes never moves a scene between splits. Profile B is simulated for train/val scenes; test scenes get
A, B and C. Balancing (50/50 cloud:clear) applies to the train split only.

To share the registry between operators, commit it as `data/manifests/scene_registry.json` and pass
`--registry data/manifests/scene_registry.json` to the acquisition, split and simulation stages. Always
`git pull` before acquiring so already-registered scenes are skipped, set `NINA_OPERATOR` to record
provenance, and commit the updated registry afterwards.

## Hardware-in-the-loop (V5)

[`hil/`](hil/) contains the OpenMV H7 Plus payload code, the ESP32 stand-in OBC firmware and the raw
results of Test 1 (surrogate execution) and Test 2 (OBC-driven integration). The code is kept exactly
as it ran. See [`hil/README.md`](hil/README.md).

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md). Each area has an owner and a delivery checklist in its README;
open work is tracked in the `iac26-repro` milestone.

## Affiliation and license

Universidad Católica Boliviana "San Pablo".
Code is BSD 3-Clause ([`LICENSE`](LICENSE)). Proposed licence for model weights and derived datasets: CC BY 4.0,
with the Copernicus and NASA FIRMS attribution given in the model and dataset cards. Cite with
[`CITATION.cff`](CITATION.cff) (Zenodo DOI above).

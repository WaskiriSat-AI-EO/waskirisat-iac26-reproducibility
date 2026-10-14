# WaskiriSat — AI payload: models, data pipelines and reproducibility

Code, models and verification artifacts behind the IAC-26 paper
**"Onboard Edge AI for Wildfire Monitoring …"** (paper ID 111635, session B4.4) on the
two AI payloads of WaskiriSat, Bolivia's first CubeSat (525 km sun-synchronous orbit).
Paper: _link / DOI to be added when indexed_.

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
docs/           Evidence taxonomy, limitations, model/dataset card templates
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

`make reproduce` prints PASS / FAIL / PENDING per item. Tables 7 and 8 and the Test 2 summary
already reproduce from the raw HIL results committed in `hil/results/`. Items still marked
PENDING are waiting for their owner; the map is in [`reproduce/paper_map.md`](reproduce/paper_map.md).

## Reproducing the paper

| Paper item | Command | Status |
|---|---|---|
| Tables 5, 6, §4 thresholds | `make table5 table6 parity` | pending (payload owners) |
| Tables 7, 8, §6.6 | `make table7 table8 test2-summary` | reproduces |
| Figures 1, 2 | `make figures` | script available |
| Binary integrity | `make verify-models` | needs the published binaries |

Models (`.tflite`, checkpoints) and the full datasets are **not stored in git**. They are
published outside the repository (Zenodo / GitHub Release); `models/MANIFEST.json` records
the expected sha256 of each artifact and `make verify-models` checks them. Links will be added
here once published.

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

Universidad Católica Boliviana "San Pablo" · BIRDS Program · Kyushu Institute of Technology.
Code is BSD 3-Clause ([`LICENSE`](LICENSE)). Licenses for models and derived datasets, and the required
attribution for Copernicus and NASA FIRMS data, are documented in the model and dataset cards.

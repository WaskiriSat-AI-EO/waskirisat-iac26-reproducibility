# WaskiriSat — AI/ML Development

Bolivia's first university-developed CubeSat (UCB San Pablo / BIRDS / KyuTech), carrying two onboard AI payloads for Earth observation from a 525 km sun-synchronous orbit.

---

## Mission

**WaskiriSat** addresses Bolivia's critical need for real-time wildfire monitoring. The country loses millions of hectares to fire annually in the Chiquitanía and Beni regions — yet has no sovereign space-based observation capability.

The satellite's AI payload, **NINA** (Neural Infrared Anomaly Observer), is a FLIR Lepton 3.5 thermal camera paired with an onboard inference engine. It detects wildfire hotspots autonomously and downlinks only the relevant frames, enabling near-real-time alerts to Bolivian civil protection authorities.

---

## Onboard AI Missions

| Payload | Task | Sensor |
|---------|------|--------|
| **NINA** | Wildfire hotspot detection | FLIR Lepton 3.5 (thermal, 160×120 px) |
| **IQC** | Image quality control for downlink prioritisation | Optical camera |

---

## This Repository

Unified workspace for the full AI/ML workflow: training dataset generation, model development, and model registry.

```
datagen/                  — Dataset generation pipelines
image_processing_notebooks/  — Landsat 9 prototype (reference)
```

**→ [`datagen/README.md`](datagen/README.md)** — start here to run the pipeline.

---

## Quickstart

```bash
git clone <repo-url> && cd waskirisat-pl-aidev
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add CDSE_USER and CDSE_PASSWORD
```

Copernicus credentials are free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).

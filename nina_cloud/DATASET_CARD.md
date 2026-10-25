# Dataset card: NINA cloud / clear tiles (`dataset_final_repo`)

| | |
|---|---|
| Content | 7,910 unique 224 × 224 RGB tiles, rendered under degradation profiles A, B and C |
| Source | Copernicus Sentinel-2 L2A, 41 scenes, 2020–2022 acquisitions over Bolivia (30 MGRS tiles) |
| Labels | `cloud` / `clear` from the scene cloud fraction of each tile |
| Owner | Leyla (WaskiriSat AI-EO) |
| Access (provisional) | [OneDrive](https://1drv.ms/f/c/a557f4573269976b/IgA9srXbmUr3SoiSf9RvRRu0AZyDhbR1KdDjQZ7Z3IK1QHw?e=6ybTYa), also on the shared Drive as `Cloud_dataset_pipeline2` |
| Planned hosting | An open archive (Hugging Face or similar). **Not done yet** |
| Metadata in git | `../nina_cloud_results/dataset/dataset_final_repo/final_manifest.json`, `../nina_cloud_results/configs/scene_registry.json` |
| Licence | **Proposal, to be confirmed:** derived data CC BY 4.0, with the Sentinel-2 attribution below |

## Composition

| Split | Profile | Clear | Cloud | Tiles |
|---|---|---:|---:|---:|
| Train | B | 2,653 | 2,851 | 5,504 |
| Validation | B | 468 | 530 | 998 |
| Test | A | 834 | 574 | 1,408 |
| Test | B | 834 | 574 | 1,408 |
| Test | C | 834 | 574 | 1,408 |

The 1,408 test tiles are the same tiles rendered three times, so the test split has 4,224 images. Paper Table 5 lists validation and test only; the training counts are given here. All counts are recomputed from the manifest by `make table5`.

## Labelling

A tile is `cloud` if its cloud fraction is above 0.70 and `clear` if it is below 0.30. Tiles in between are ambiguous and excluded (`nina_cloud/datagen/config.py`). In the manifest, cloud tiles span 0.70–1.00 and clear tiles 0.00–0.30.

## Splits and leakage audit

Splitting is by scene: none of the 41 scenes appears in more than one split (checked on the manifest; the training report records `scene_leakage: 0`). Test data was not used for training, calibration or threshold selection.

## Degradation profiles

A (lower), B (intermediate) and C (highest degradation). B is used for training and validation; A, B and C for testing. The multipliers are assumptions (T3); see paper Table 2. The profile definitions live in `datagen/config.py`.

## Attribution and terms of use

Contains modified Copernicus Sentinel-2 data (2020–2022), processed by WaskiriSat AI-EO. Copernicus Sentinel data are free and open under the Copernicus data policy, which requires attribution and allows redistribution of derived products. Please cite the source as "Copernicus Sentinel data [year]".

## Limitations

- Synthetic degradation of real Sentinel-2 imagery, not real payload imagery.
- 41 scenes from one region and few seasons: the model's behaviour elsewhere is not tested.
- Test tiles are three renderings of the same 1,408 tiles, so the three profiles are not independent samples.
- The OneDrive link is a personal share and may disappear; the manifest and the reports in git are the durable record until the archive exists.

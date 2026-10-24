# Sample catalogue

[`manifest.csv`](manifest.csv) records the 200 evaluation PNG identities: 100 Cloud and 100 Fire, each balanced at 50 negative and 50 positive samples. Images remain outside git. Test Payload uses these same datasets.

| Column | Meaning |
|---|---|
| `sample_id` | Test Payload identity: mode, zero-based six-digit index, filename |
| `dataset`, `index` | Dataset folder and zero-based position within its deterministic catalog |
| `class`, `class_id` | Ground-truth folder and negative=0 / positive=1 |
| `path` | Relative to `paper_tests/`, using forward slashes |
| `sha256` | SHA-256 of the complete PNG file bytes |
| `size_bytes` | Original file size |
| `width`, `height`, `bit_depth`, `png_color_type` | PNG IHDR fields; color type 0 is grayscale, 2 RGB |

All 200 filenames, class labels and dimensions were checked against Test Surrogate result records. Rows follow the device order: Cloud `clear` then `cloud`; Fire `no_fire` then `fire`; lexicographic filenames within each class. Test Surrogate result indices are one-based, while these indices and Test Payload IDs are zero-based.

The legacy dataset manifest hashes are:

| Dataset | Legacy manifest SHA-256 |
|---|---|
| Cloud | `595211136ed731e4085fc60cd8db5f2fd14e96fb697acfa1d354f2ba36c4b553` |
| Fire | `c6e1489a88323230a97e3273d21ac1c469a8a847bcd78adc07397f8d28781be5` |

These were independently reproduced by hashing the ordered UTF-8 lines `class/filename\n`. They cover names and order, not pixel contents. The new CSV adds per-file content hashes and does not replace the device's legacy ordering check. Public image download location and redistribution licensing still need to accompany the dataset release.

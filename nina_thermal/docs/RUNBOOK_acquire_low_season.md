# Runbook — acquire fire-season (Aug 2022) and low-season (Feb–Apr 2023) granules and retrain FRP20000

**For:** Osvaldo (the CDSE download and the 5-stage per-root pipeline both run
on your machine, where the credentials and the existing data roots live).

**Status as of 2026-09-24: not started.** Steps below are verified against
the live repo (CLI flags checked via `--help`, existing roots' actual granule
dates confirmed from disk) but **not yet executed end-to-end** — this is the
plan to run, not a record of a run.

**Prerequisite:** `main` branch, `.env` with `CDSE_USER` / `CDSE_PASSWORD` at
the repo's parent directory (`load_dotenv()` walks up and finds it).

---

## 0. Why these dates

Every granule downloaded so far sits inside Bolivia's fire season:

| root | months actually on disk (from granule folder names, not just the manifest's last-run `args`) |
|---|---|
| `data/` | 2023-06, 2023-08 |
| `data_202408/` | 2024-08 |
| `data_202409/` | 2024-09 |
| `data_202509/` | 2025-09 |

The model has never seen a genuinely low-fire-season scene, so every clean
negative it has trained on is also a hot, dry day — the confound documented
in `frp_labels.py` and `TRAINING_REPORT.md` §10. **Feb–Apr 2023** is
Bolivia's wet season: low FIRMS activity, and it doesn't overlap any month
already downloaded, so no split-reconciliation conflicts are expected this
time (unlike `data_202408`, which genuinely duplicated granules already in
`data/`).

Training has also shifted toward accumulated-fire scenes, which needs more
fire-rich data. **Aug 2022** is peak fire season, a year not yet on disk, so
it adds fire-heavy scenes with no overlap. The two new roots are therefore
complementary: `data_202208` (fire-rich) and `data_202302` (clean negatives).

---

## 1. Preflight

```bash
cd ~/Documents/IA+EO/waskirisat-pl-aidev
git status --short          # confirm clean before starting a long job
df -h /                     # 128 GB free as of 2026-09-24
```

**Disk budget.** Each existing single-month root downloaded 50 granules at
~400–700 MB/granule ≈ 20 GB. A 3-month window at `--max_products 60` could
reach ~30–40 GB. Two windows ≈ **60–80 GB**, against 128 GB free — workable
but roughly half to two-thirds of what remains. Three `.zip` archives at the
repo root (`data_202408.zip`, `data_202409.zip`, `data_202509.zip`, ~59 GB
combined) look like they duplicate already-extracted directories and may be
safe to remove first — verify before deleting anything.

---

## 2. Download — two new roots

```bash
# Fire-rich lote: Aug 2022 (peak fire season, no overlap with existing roots)
python -m nina_thermal.datagen.pipelines.run_acquisition \
    --roi nina_thermal/resources/bolivia_roi.geojson \
    --start 20220801 --end 20220831 \
    --output_dir data_202208/01_raw --max_products 50 --use_firms

# Low-season lote: Feb–Apr 2023 (wet season, clean negatives)
python -m nina_thermal.datagen.pipelines.run_acquisition \
    --roi nina_thermal/resources/bolivia_roi.geojson \
    --start 20230201 --end 20230430 \
    --output_dir data_202302/01_raw --max_products 60
```

The Aug 2022 command uses `--use_firms` so the download keeps only
FIRMS-active dates (we want fire-heavy scenes for the accumulated-fire focus).
The Feb–Apr 2023 command omits it — for a low-season window we want the full
date range, not filtered down to FIRMS-active dates (there should barely be
any).

This is the slow step: CDSE throughput, not compute, is the bottleneck.

### 2b. FIRMS CSVs — not written by `run_acquisition`

`run_acquisition --use_firms` only uses FIRMS to pick dates; it never saves the
CSVs that `run_processing` and the FRP relabel read from `$ROOT/firms/`. Fetch
them separately, and get two things right (both were wrong before 2026-10-03
and silently produced fake clean negatives):

- **bbox must cover every patch footprint**, not just Bolivia's lowlands. A
  patch is 640 × 480 km, so footprints reach lon −76.2…−52.7, lat −28.2…−5.6.
  Use `-77,-29,-52,-5`. With the old `-67.5,-20.5,-59.7,-11.8` box, 70% of
  "zero-detection" negatives were simply outside the queried area.
- **one product per date** (`VIIRS_SNPP_SP`; NRT and SP rows are not
  de-duplicated against each other and double-count FRP). Suomi-NPP was
  offline 2022-07-20…08-10 — use `VIIRS_NOAA20_SP` for those dates.

Each root's `firms/coverage.json` declares the queried bbox
(`{"bbox": [-77, -29, -52, -5]}`); `frp_labels` refuses to call a patch a
negative when its footprint falls outside it, or when its date has no FIRMS
detection anywhere.

---

## 3. Per new root — five pipeline stages

Run this block once for `ROOT=data_202208`, then again for `ROOT=data_202302`.

```bash
ROOT=data_202208   # then data_202302

python -m nina_thermal.datagen.pipelines.run_extraction \
    --input_dir $ROOT/01_raw --output_dir $ROOT/02_extracted

python -m nina_thermal.datagen.pipelines.run_processing \
    --input_dir $ROOT/02_extracted --output_dir $ROOT/03_patches \
    --firms_dir $ROOT/firms

python -m nina_thermal.datagen.pipelines.run_simulation \
    --input_dir $ROOT/03_patches --output_dir $ROOT/04_dataset

python -m nina_thermal.datagen.pipelines.run_split \
    --dataset_dir $ROOT/04_dataset

python -m nina_thermal.datagen.pipelines.run_balance \
    --dataset_dir $ROOT/04_dataset
```

`data_202208` should be fire-rich. Low season (`data_202302`) likely yields
close to zero fire patches, so `run_balance`'s fire-ratio balancing may leave
that root almost entirely no-fire. **That is expected and is the point** —
these are the clean negatives the confound analysis needs.

---

## 4. Merge — no code changes required

`dataset.py::load_merged_records()` already accepts an arbitrary `--roots`
list; extend it to include the two new roots:

```bash
--roots data data_202408 data_202409 data_202509 data_202208 data_202302
```

Since Aug 2022 and Feb–Apr 2023 don't overlap any existing granule, no split
conflicts are expected — but the merge still checks and will raise if one
ever appears, exactly as it did for `data_202408`.

---

## 5. Retrain, requantize, re-evaluate

The training pool changes, so the current headline (test ROC-AUC 0.787) can
move. Write to a new directory rather than overwriting the existing result,
so the two remain comparable:

```bash
ROOTS="data data_202408 data_202409 data_202509 data_202208 data_202302"
OUT=models/nina_thermal_frp20000_v2

python -m nina_thermal.training.pipelines.run_baseline \
    --label frp --frp_threshold 20000 --roots $ROOTS --out_dir $OUT

python -m nina_thermal.training.train_thermal \
    --frp_threshold 20000 --roots $ROOTS --out_dir $OUT

python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 --roots $ROOTS --out_dir $OUT --figures

python -m nina_thermal.quantization.pipelines.run_quantize \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt \
    --frp_threshold 20000 --roots $ROOTS --out_dir $OUT \
    --epochs 40 --lr 5e-5 --batch_size 16 --num_workers 2 --no_smoke_test
```

**Pass `--roots $ROOTS` to every command.** Each one defaults to the four
original roots (`DEFAULT_ROOTS` in `train_thermal.py`); on 2026-10-02 evaluate
and quantize silently ran on the old roots while training used all seven.
Likewise any `--frp_negative_max` must be identical across all four commands.

**`--batch_size 16 --num_workers 2` on `run_quantize` is not optional.** At
the defaults (32/4) this OOM-killed silently on this machine on 2026-09-03 —
no traceback, just a dead process after "Stage 2: QAT preparation …".

---

## 6. Update the report

Add to `TRAINING_REPORT.md`:

- §5 (Dataset) — new root sizes, new total granule/patch counts
- §6.5-style baseline table — re-run `run_baseline --label frp` now includes
  true low-season negatives
- §10 (confound discussion) — this is the number reviewers will ask for: a
  clean-negative count drawn from genuinely low-fire-season scenes, not just
  "zero FIRMS detections in an otherwise hot month"

---

## Time estimate

Steps 2–3 are the long pole — download plus five pipeline stages × 2 roots,
likely 1–3 hours depending on CDSE throughput. Step 5's quantization alone
took ~1–2 min on this GPU in the 2026-09-03 run; training and evaluation are
each under a minute. The wait is almost entirely acquisition, not compute.

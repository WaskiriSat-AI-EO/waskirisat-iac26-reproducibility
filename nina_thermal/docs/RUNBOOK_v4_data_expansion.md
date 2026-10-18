# Runbook — v4 data expansion: download, process, retrain

**For:** Osvaldo (CDSE/FIRMS credentials and all data roots live on this machine).
**Written:** 2026-10-17 · **Status:** Step 1 done (2026-10-17, branch
`feat/v4-data-expansion`); Step 2 pilot done (§3.1); Step 3 done (§4.1); Steps 4–6 not started. All commands below exist and
were checked with `--help`. Step 1 was verified live against CDSE and FIRMS (§2.1).
**Starts from:** `TRAINING_REPORT_v3.md` (v3_bc fp32 + v3_bc_qat_sel int8).
**Goal:** roughly double the labeled pool. Prioritize months that contain both
classes, so that v4 learns fire rather than season (v3 open issues #2, #3, #5).
Then train, quantize and report v4.

---

## 0. Where we are

### 0.1 Disk and raw data

On 2026-10-13 all raw `.SEN3` granules and the five `.zip` snapshots were
deleted (~557 GB). Free space is now ~594 GB. Every root still has
`02_extracted/` onward. `01_raw/` now holds only `acquisition_manifest.json`
and `granule_inventory.txt` (the list of deleted granules).

Consequences:

- **Do not re-run `run_extraction` on existing roots.** There is nothing left
  to extract. `02_extracted/` is now the source of truth.
- Reading other bands or files from the old granules means downloading them again.
- Raw data is ~93% of a root (≈380 MB per granule; `02_extracted` ≈27 MB per granule).
  New downloads delete raw right after extraction (Step 1, A3).

### 0.2 What is on disk (from `granule_inventory.txt`)

Acquisition sorted by date and stopped at `--max_products`. Most months
therefore only cover **their first 9 days**:

| root | month | granules | days covered |
|---|---|---|---|
| `data/` | 2023-06, 2023-08, 2024-08 | 49 / 50 / 50 | 1–9 |
| `data_202208/` | 2022-08 | 46 | 1–22 |
| `data_202302/` | 2023-02 | 57 | 1–25 |
| `data_202309/` | 2023-09 | 54 | 1–26 |
| `data_202408/` | 2024-08 | 50 | 1–9 (same granules as in `data/`) |
| `data_202409/` | 2024-09 | 50 | 1–9 |
| `data_202509/` | 2025-09 | 50 | 1–9 |
| `datagen/data/` | 14 months, 2023-01 … 2025-09 | 368 | mostly 1–9; **318 not in any training root** |

Sentinel-3A + 3B give ~5–6 granules per day over the ROI, so a full month is
~160–170 granules.

### 0.3 Why more data, and which data

- v3 labels by month: 2023-02 and 2023-06 have **no fire** patches, and 2024-09 has
  **no no-fire** patches. Those months teach season, not fire. Mixed months are
  what improve within-month AUC (v3: 0.889, few patches).
- About 2.6 labeled patches per granule: 51% of patches fall in the excluded
  1–20 GW band.
- Wet-season negatives are already plentiful (2023-02: 295). Don't prioritize more.

### 0.4 ROI: use `bo_nina_roi_simplified.geojson`

`nina_thermal/resources/bolivia_roi.geojson` is a **0.1° × 0.1° test box**
near Cochabamba, not Bolivia. The pre-v4 runbook used it, so `data_202208`,
`data_202302` and `data_202309` were searched with it. Their granules are valid,
but the search found ~40% of what the country polygon gives (2024-09: 70 products
vs 177). The other roots used `bo_nina_roi_simplified.geojson` (Bolivia,
lon −67.5…−59.7, lat −20.5…−11.8). `run_acquisition` now refuses ROIs narrower
than 1° (`--allow_small_roi` to override). Batch 1 tops up those three months (§5).

---

## 1. Rules for every step (lessons from v1–v3)

1. **Pass `--roots` to every training, evaluation and quantization command.** The default
   is the 4 original roots (`DEFAULT_ROOTS` in `train_thermal.py`).
2. **The same `--frp_threshold 20000 --frp_negative_max 1000` everywhere.**
3. **FIRMS:** always through `run_fetch_firms`, which uses bbox `-77,-29,-52,-5`
   and **one product per date** (`VIIRS_SNPP_SP`; `VIIRS_NOAA20_SP` for the S-NPP
   outages in `config.FIRMS_SNPP_OUTAGES`), fetches `[start − 1 day, end + 1 day]`,
   **checks every day** (0 detections, or < 40% of NOAA-20, means missing or
   partial) and writes `firms/coverage.json`. Never mix NRT and SP. If it
   flags a day, add it to `FIRMS_SNPP_OUTAGES` and re-run. Audit a root with
   `run_fetch_firms --check_only --out_dir $ROOT/firms --start 20000101 --end 20000101`.
4. **No granule may appear in two roots.** Split reconciliation is test > val > train
   (`dataset.py::reconcile_splits`), so an overlapping granule can move a v3 train
   granule into test and silently change the v3 test set.
5. **Freeze the v3 test set.** Never modify the 7 v3 roots' `04_dataset/` splits.
   v4 is always reported on the v3 test set *and* on the new test granules.
6. `run_quantize`: `--batch_size 16 --num_workers 2` (it was OOM-killed at the defaults).
7. New roots get new directory names. Never write into a v3 root.

---

## 2. Step 1 — Acquisition code changes ✅ done 2026-10-17

Branch: `feat/v4-data-expansion`. Each change gets a unit test under
`nina_thermal/datagen/tests/`.

| id | change | file(s) | acceptance |
|---|---|---|---|
| A1 | **Even sampling over the window.** `--sampling {even,oldest}` (default `even`): search the whole window without a cap, then keep `--max_products` granules spread evenly over the days (round-robin per day). | `acquisition/cdse_search.py`, `pipelines/run_acquisition.py` | a 30-day window with cap 90 gives ~3 granules/day on every day that has data |
| A2 | **Skip already-held granules.** `--skip_inventory GLOB` (e.g. `'data*/01_raw/granule_inventory.txt'`) drops any product whose name is already listed, before the cap is applied. | `run_acquisition.py` | a dry run on 2024-09 lists 0 of the 50 held granules |
| A3 | **Stream: download → extract → delete raw.** `--extract_dir DIR --delete_raw`: after each granule, run `extract_channels` into `DIR` and remove the `.SEN3` folder only once the 4 `.npy` files exist. | `run_acquisition.py` (reuses `ingestion/extract_channels.py`) | peak disk use < 2 GB for a whole month; `02_extracted` identical to a plain `run_extraction` on one test granule |
| A4 | **Write `granule_inventory.txt`** with every granule downloaded, plus `--dry_run` (search and list only, no download). | `run_acquisition.py` | the inventory matches the `02_extracted` filenames |
| A5 | **FIRMS fetch CLI** `run_fetch_firms --start --end --out_dir ROOT/firms`: 5-day area-API chunks over `[start−1, end+1]`, bbox `-77,-29,-52,-5`, `VIIRS_SNPP_SP` with NOAA-20 for the outage, writes `coverage.json`. It fails loudly if SP is not available for a date, and never falls back to NRT silently. | new `pipelines/run_fetch_firms.py`, `acquisition/firms_api.py` | CSV names and coverage match `data_202309/firms/`; refetching 2023-09 reproduces the same detections |

Also: `--allow_small_roi` guard (§0.4); the acquisition manifest now records
the `.SEN3` path correctly (it used to append `.SEN3` twice) and the new
arguments. Manifest and inventory are rewritten after every granule, so an
interrupted run resumes with the same command. The root's own inventory never
counts as "held", even when a `--skip_inventory` glob matches it.

Tests: `test_acquisition_v4.py` (17), `test_fetch_firms.py` (10);
`pytest nina_thermal -q` → **304 passed**. `black`/`ruff` clean on the touched
files. The 7 black and 9 mypy findings elsewhere predate this branch
(int8-fix commit).

### 2.1 Live verification (2026-10-17)

| check | result |
|---|---|
| CDSE dry run, 2024-09, country ROI, cap 60 | 177 in catalog; **all 50 held skipped**; 60 selected over 22 days (2–3/day, days 9–30) |
| one real granule (2024-09-15), `--extract_dir --delete_raw` | 350 MB download → 4 `.npy` (28 MB); raw deleted; inventory + both manifests written; values match `data_202409` (F2 median 270 K vs 274 K, same lat range) |
| **download speed** | **≈ 1 min 50 s per granule (≈ 3.3 MB/s)** → ~1.8 h per 60-granule root |
| `run_fetch_firms` 2023-09-01…26 | same 6 file names as `data_202309/firms/`; **identical detections and summed FRP** in every file (only row order differs) |

Optional (not required for v4): download only the 4 needed NetCDFs per
granule through CDSE OData `Nodes(...)/$value` (~20 MB instead of ~350 MB).
At the measured 3.3 MB/s, Batches 1+2 (~1,500 granules) take ~45 h of
downloading; the 4-file route could cut that roughly 10×. Decide after the pilot.

---

## 3. Step 2 — Pilot: one month, end to end (≈ ½ day)

Window: **2024-09-10 … 2024-09-30** (gap fill; 2024-09 is all-fire on days 1–9).

```bash
cd ~/Documents/IA+EO/waskirisat-pl-aidev
source ../venv/bin/activate
ROOT=data_202409b

# dry run: how many products, how many skipped
python -m nina_thermal.datagen.pipelines.run_acquisition \
    --roi nina_thermal/resources/bo_nina_roi_simplified.geojson \
    --start 20240910 --end 20240930 --max_products 60 \
    --sampling even --skip_inventory 'data*/01_raw/granule_inventory.txt' \
    --skip_inventory 'datagen/data/01_raw/granule_inventory.txt' \
    --output_dir $ROOT/01_raw --dry_run                                

# real run (time it)
time python -m nina_thermal.datagen.pipelines.run_acquisition \
    --roi nina_thermal/resources/bo_nina_roi_simplified.geojson \
    --start 20240910 --end 20240930 --max_products 60 \
    --sampling even --skip_inventory 'data*/01_raw/granule_inventory.txt' \
    --skip_inventory 'datagen/data/01_raw/granule_inventory.txt' \
    --output_dir $ROOT/01_raw --extract_dir $ROOT/02_extracted --delete_raw

python -m nina_thermal.datagen.pipelines.run_fetch_firms \
    --start 20240910 --end 20240930 --out_dir $ROOT/firms
```

Then run §6 (per-root pipeline) and §7 (QC) for `data_202409b`.

**Start only one download per root.** Two runs on the same root delete each
other's partial ZIPs. Before starting or re-running:
`pgrep -f "^python -m nina_thermal.datagen.pipelines.run_acquisition"`.

**Record from the pilot** (they size everything that follows):

- minutes per granule (download + extract), which gives hours per batch
- labeled patches per granule and the fire / no-fire / ambiguous split
- whether days 10–30 give no-fire patches in a month that was all-fire

### 3.1 Pilot result (2026-10-17) ✅

| | value |
|---|---|
| granules | 60 / 60 (6 failed on the first pass: 3 from an accidental double start, 3 dropped connections; all recovered by re-running; retries are now automatic) |
| time | 2 h 10 min + 10 min re-run → **≈ 2.2 min per granule** |
| patches (with `--roi`) | 228 from 59 granules (1 fully masked) |
| labels | **115 fire / 24 no-fire / 89 ambiguous** → 2.3 labeled patches per granule (v3: 2.6) |
| 2024-09 total | 222 fire / 40 no-fire (days 1–9 alone: 107 / 16) |
| merged pool | 883 → **1,022 labeled** (fire 316 → 431, no-fire 567 → 591) |
| checks | 0 duplicate granules; v3 reconciliation unchanged (19 granules / 58 patches); `no_frp_data` 0 |

What this means for the batches:

- **Peak-season months add mostly fire.** Days 10–30 of Sep 2024 gave 4.8 fire
  for every no-fire. No-fire is now the scarce class, so Batch 2 priorities 1–2
  (July, October: shoulder months) matter more than more Aug–Sep.
- **Sizing:** +139 labeled per 60 granules. Doubling the v3 pool (+~900
  labeled) needs ~400 granules ≈ 15 h of downloading.

If more than 60% of patches end up ambiguous, consider a FIRMS pre-screen
(rank granules by expected summed FRP from their CDSE footprint before
download). It is not needed otherwise.

---

## 4. Step 3 — Recover `datagen/data` (no download, ≈ 2–3 h)

318 granules, already extracted and never used in training. Extraction code is
unchanged since 2026-05-31, and these were extracted on 2026-06-20, so
`02_extracted` is valid. Move them into a new root **with the 50 overlapping
granules removed** (rule 4):

```bash
ROOT=data_extra
mkdir -p $ROOT/01_raw $ROOT/02_extracted
cat data/01_raw/granule_inventory.txt data_20*/01_raw/granule_inventory.txt | sort -u > /tmp/used.txt
comm -23 <(sort datagen/data/01_raw/granule_inventory.txt) /tmp/used.txt > $ROOT/01_raw/granule_inventory.txt
wc -l $ROOT/01_raw/granule_inventory.txt                 # expect 318
while read g; do s=${g%.SEN3}; cp datagen/data/02_extracted/${s}_*.npy $ROOT/02_extracted/; done \
    < $ROOT/01_raw/granule_inventory.txt
ls $ROOT/02_extracted/*_F2_BT.npy | wc -l                 # expect 318
```

### 4.1 Result (2026-10-17) ✅

`data_extra`: 318 granules → 1,274 patches (all kept, `--fire_ratio 0.01`).
FIRMS: 26 CSVs, 0 flagged days after the outage fixes below.

| month | fire | no-fire | ambiguous |
|---|---|---|---|
| 2023-01 / 2024-01 / 2025-01 | 0 | 170 / 98 / 132 | 184 |
| 2023-07 | 4 | 68 | 217 |
| **2024-07** | **13** | **39** | 126 |
| 2025-07 | 0 | 35 | 23 |
| 2024-08 (30–31) | 11 | 4 | 6 |
| **2024-10** | **38** | **2** | 21 |
| 2023-09/10, 2025-08 | 11 | 10 | 62 |
| **total** | **81** | **558** | 639 |

**Merged pool (9 roots): 883 → 1,662 labeled patches** (fire 316 → 514,
no-fire 567 → 1,148); test 339 / val 253 / train 1,070 patches. The class
balance has flipped: no-fire is now 69%. 400 of the new no-fire patches are
wet-season January scenes (easy, cold negatives). The valuable additions are
July's warm, dry-season negatives (142) and the mixed July 2024 month.
Fire-rich months are now what the pool needs most (§5 priorities).

**FIRMS gaps found by the per-day check** (all S-NPP, filled from NOAA-20; the
7 v3 roots passed `--check_only` with 0 flagged days):

| dates | S-NPP vs NOAA-20 | root affected |
|---|---|---|
| 2024-07-10 … 07-16 | 0 (07-16: 844 vs 4,566) | `data_extra` |
| 2024-07-24 … 07-28 | 0 (07-24: 782 vs 13,045) | none yet (Batch 2 July 2024) |
| 2024-09-26 … 09-27 | 0 / 2,741 vs 7,073 | `data_202409b` (pilot relabelled: 115 → 121 fire) |
| 2025-07-04 … 07-05 | 311 vs 1,235 / 385 vs 1,200 | `data_extra` |

Months (new granules): 2023-01 50, **2023-07 60**, 2023-09 9, 2023-10 5, 2024-01 49,
**2024-07 50**, 2024-08 4, **2024-10 16**, 2025-01 48, **2025-07 18**, 2025-08 9.

FIRMS: run `run_fetch_firms` once per contiguous month range into
`$ROOT/firms/` (the CSVs can share one directory; `coverage.json` holds the
bbox). Then §6 and §7.

After this step and the pilot pass QC, `datagen/data/` can be deleted (~10 GB).

---

## 5. Step 4 — Download batches

Batches use `--sampling even --max_products 90` (~3 granules/day), always
with both `--skip_inventory` globs, `--extract_dir … --delete_raw`, the
country ROI, and a FIRMS fetch per root. One root per window. Template:

```bash
ROOT=data_202306b; START=20230610; END=20230630
nohup python -m nina_thermal.datagen.pipelines.run_acquisition \
    --roi nina_thermal/resources/bo_nina_roi_simplified.geojson \
    --start $START --end $END --max_products 90 --sampling even \
    --skip_inventory 'data*/01_raw/granule_inventory.txt' \
    --skip_inventory 'datagen/data/01_raw/granule_inventory.txt' \
    --output_dir $ROOT/01_raw --extract_dir $ROOT/02_extracted --delete_raw \
    > logs_acq_$ROOT.log 2>&1 &
python -m nina_thermal.datagen.pipelines.run_fetch_firms \
    --start $START --end $END --out_dir $ROOT/firms
```

If the run stops (network, token, reboot), run the same command again. It
resumes from the inventory. Exit code 1 means at least one granule failed; re-run.

### Batch 1 — fill the gaps in months we already have (~550 granules, ~17 h)

| root | window | why |
|---|---|---|
| `data_202306b` | 2023-06-10 … 06-30 | June false alarms (12 of 15 v3 test FPs) |
| `data_202308b` | 2023-08-10 … 08-31 | peak season, only 19 labeled patches today |
| `data_202408b` | 2024-08-10 … 08-31 | near all-fire on days 1–9 |
| `data_202509b` | 2025-09-10 … 09-30 | only 25 labeled patches today |
| `data_202208b` | 2022-08-01 … 08-31 | rest of Aug 2022 (v3 root used the tiny ROI, §0.4) |
| `data_202309b` | 2023-09-01 … 09-30 | rest of Sep 2023 (tiny ROI); mixed month (64 fire / 22 no-fire) |
| `data_202302b` | 2023-02-01 … 02-28 | low priority: wet season (cap 40) |

(`data_202409b` was the pilot.)

### Batch 2 — new periods (~900–1,100 granules)

Priority order; stop when the pool reaches the target (§8.1).

| priority | roots | windows | why |
|---|---|---|---|
| 1 | `data_YYYY07` for 2019–2022, `data_202307b` / `data_202407b` / `data_202507b` (rest of July) | July | early season: warm and dry with mixed activity, the June-FP confound |
| 2 | `data_YYYY10` for 2019–2025 | October | end of season, rain onset, mixed classes |
| 3 | `data_201908`, `data_201909`, `data_202008`, `data_202009`, `data_202108`, `data_202109` | Aug–Sep | new fire years (2019 Chiquitanía, 2020 large) |
| 4 | `data_2023mar` (2023-03-01 … 04-30) | wet season | closes open issue #5; low value for the model |

**Check in the pilot of each new year** (first root of 2019): products are still
`SL_1_RBT___` with the same `S8_BT_in` / `F2_BT_in` / `geodetic_in` / `flags_in`
layout. Older baselines and processing centres (PS1, PS2, ESA) appear in names.
Compare one extracted granule's value range against a 2023 one.

### Budget

- Disk: ~1.6 GB per 60-granule root after raw deletion. Batches 1+2 total < 50 GB.
- Time: ~1 min 50 s per granule × ~1,500 ≈ 45 h of downloading. Run batches in
  the background (template above). CDSE tokens are refreshed per download.
  Running 2–3 roots in parallel should help if CDSE allows concurrent
  downloads per account. Check the speed with two at once before relying on it.

---

## 6. Per-root pipeline (each new root)

```bash
ROOT=data_202409b   # each new root

# (only if A3 was not used) python -m nina_thermal.datagen.pipelines.run_extraction \
#     --input_dir $ROOT/01_raw --output_dir $ROOT/02_extracted

python -m nina_thermal.datagen.pipelines.run_processing \
    --input_dir $ROOT/02_extracted --output_dir $ROOT/03_patches --firms_dir $ROOT/firms \
    --roi nina_thermal/resources/bo_nina_roi_simplified.geojson

python -m nina_thermal.datagen.pipelines.run_simulation \
    --input_dir $ROOT/03_patches --output_dir $ROOT/04_dataset

python -m nina_thermal.datagen.pipelines.run_split   --dataset_dir $ROOT/04_dataset
python -m nina_thermal.datagen.pipelines.run_balance --dataset_dir $ROOT/04_dataset --fire_ratio 0.01

# Profiles A and C for train/val (default simulation renders only B for train/val).
python -m nina_thermal.datagen.pipelines.run_simulation \
    --input_dir $ROOT/03_patches --output_dir $ROOT/04_dataset_ac \
    --profiles A,C --agc_modes fixed
cp --update=none $ROOT/04_dataset_ac/images/*_fixed_A.png $ROOT/04_dataset_ac/images/*_fixed_C.png \
    $ROOT/04_dataset/images/
```

**`--fire_ratio 0.01` on `run_balance` for new roots.** Balance undersamples
no-fire by the *datagen pixel* label, before the FRP relabel used in
training. At the default 0.2 it dropped 1,029 of `data_extra`'s 1,274 patches
(the v3 roots kept the default; only `data_202309` lost patches, 257, and is
left as is so the frozen v3 test set does not move). 0.01 keeps every patch;
training handles imbalance with `pos_weight` and the val-swept threshold.

**`--roi` on `run_processing` is required.** The v3 roots were processed with
it (logs in `data/logs/run_*/run_processing_*.log`), so only patches centred in
Bolivia are kept. Without it a granule yields ~10 patches instead of ~3–4,
reaching Brazil and Paraguay (lon −48°): a different distribution from v3, and
partly outside the FIRMS bbox (`no_frp_data`). Found in the pilot (§3.1).

The loader finds images by name (`{patch}_{agc}_{profile}.png` under
`04_dataset/images/`), so copied A/C images are picked up. Splits come from
`04_dataset/balanced_manifest.json` only. `cp --update=none` leaves the existing test
A/C images untouched (simulation is deterministic per patch, so they are
identical anyway).

**Also once for the 7 v3 roots:** render profile **A** (they have B and C) so
that v4 can train on A+B+C (open issue #4):

```bash
for ROOT in data data_202408 data_202409 data_202509 data_202208 data_202302 data_202309; do
  python -m nina_thermal.datagen.pipelines.run_simulation \
      --input_dir $ROOT/03_patches --output_dir $ROOT/04_dataset_a --profiles A --agc_modes fixed
  cp --update=none $ROOT/04_dataset_a/images/*_fixed_A.png $ROOT/04_dataset/images/
done
```

This adds images only. It does not touch any manifest or split, so the v3 test set is unchanged.

---

## 7. QC checklist per root (before it joins training)

- [ ] `granule_inventory.txt` count = `02_extracted/*_F2_BT.npy` count
- [ ] no granule in two roots:
      `cat data*/01_raw/granule_inventory.txt | sort | uniq -d | wc -l` → 0
- [ ] `firms/coverage.json` bbox `[-77,-29,-52,-5]`; CSVs cover `[start−1, end+1]`;
      one product per date
- [ ] patches with labels by month: fire / no-fire / ambiguous / `no_frp_data`
      (via `train_thermal --audit --roots <v3 roots> $ROOT`)
- [ ] the audit reports 0 granules with conflicting splits
- [ ] A, B, C images present for train/val/test: count `*_fixed_{A,B,C}.png`
- [ ] spot-check 5 `viz` PNGs (orientation, no NaN stripes)
- [ ] record the root in the table in §8.2

---

## 8. Step 5 — Train v4

### 8.1 When to train

- **Checkpoint 1:** after Step 3 + pilot + Batch 1. Train v4a to measure how
  much the new data helps.
- **Checkpoint 2:** after Batch 2, or once the pool reaches **≥ 1,800 labeled
  patches with ≥ 600 fire and ≥ 6 mixed months**. Train v4 (the final model).

### 8.2 Roots

```bash
ROOTS_V3="data data_202408 data_202409 data_202509 data_202208 data_202302 data_202309"
ROOTS_NEW="data_extra data_202409b data_202306b data_202308b data_202408b data_202509b data_202208b"  # + batch 2
ROOTS="$ROOTS_V3 $ROOTS_NEW"
COMMON="--frp_threshold 20000 --frp_negative_max 1000"
OUT=models/nina_thermal_frp20000_v4_abc
```

### 8.3 Commands

```bash
# audit the merged pool (counts, conflicts, splits)
python -m nina_thermal.training.train_thermal $COMMON --roots $ROOTS --audit

# feasibility baseline (scene-temperature control, §7 of the v3 report)
python -m nina_thermal.training.pipelines.run_baseline \
    --label frp $COMMON --roots $ROOTS --out_dir $OUT

# fp32: A+B+C, recall target 0.95
python -m nina_thermal.training.train_thermal $COMMON --roots $ROOTS \
    --profiles A B C --recall_target 0.95 --out_dir $OUT --num_workers 2

# evaluate on the FROZEN v3 test set (v3 roots only), then on everything
python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt $COMMON \
    --roots $ROOTS_V3 --out_dir $OUT/eval_v3test --figures
python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt $COMMON \
    --roots $ROOTS --out_dir $OUT/eval_all --figures

# int8: QAT with real-int8 checkpoint selection (same recipe as v3_bc_qat_sel)
QOUT=${OUT}_qat_sel
mkdir -p $QOUT && cp $OUT/training_report.json $QOUT/
python -m nina_thermal.quantization.pipelines.run_quantize \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt $COMMON --roots $ROOTS \
    --profiles A B C --out_dir $QOUT --epochs 40 --lr 5e-5 \
    --batch_size 16 --num_workers 2 --select_every 5 --no_smoke_test
python -m nina_thermal.evaluation.pipelines.run_evaluate \
    --checkpoint $OUT/mobilenetv3s_fire_best.pt --tflite $QOUT/mobilenetv3s_fire_int8.tflite \
    $COMMON --roots $ROOTS_V3 --out_dir $QOUT/eval_v3test
```

`run_evaluate --roots $ROOTS_V3` gives exactly the v3 test set: the v3 roots
and their splits are untouched, and no new root contains their granules
(rule 4). Note that the v3 report's within-month AUC and per-month tables
were computed separately (by hand). Reproduce them for v4 the same way, or
add them to `run_evaluate` (recommended, ≈1 h).

### 8.4 Experiments (run after the main model)

| id | what | why |
|---|---|---|
| E1 | QAT with seeds 42 / 1 / 2 | QAT varies run to run (v3: val AUC 0.865–0.915) |
| E2 | `--frp_negative_max` 500 / 1000 / 2000 | v3 open issue #7 |
| E3 | v4 B+C vs v4 A+B+C | does A in training fix A recall without hurting B? |
| E4 | v3_bc evaluated on the new test granules | how well v3 generalizes to unseen months |

### 8.5 Acceptance for v4 (vs v3_bc on the frozen v3 test set, profile B)

- ROC-AUC ≥ 0.959 − 0.01; recall ≥ 0.94 at the val-chosen threshold
- **within-month AUC above the scene-temperature baseline** by a wider margin than v3 (0.889 vs 0.843)
- fewer June-type false alarms (v3: 12 of 15 test FPs)
- profile A recall ≥ 0.85 (v3 int8: 0.70)
- int8: recall drop ≤ 0.005 and F-beta drop ≤ 0.010 vs fp32. Before QAT runs,
  decide whether SQNR/KL compare against fake-quant (v3 open issue #1).

---

## 9. Step 6 — Report

Write `docs/TRAINING_REPORT_v4.md`, using the v3 report's structure:

- data table: every root, months, days covered, granules, labeled patches
- labels-by-month table: the number of mixed months is the headline data metric
- frozen-v3-test results next to v3_bc (fp32 and int8), plus extended-test results
- within-month AUC vs scene temperature, as a headline metric
- E1–E4, and the gate decision
- update the open-issues table in the v3 report and mark what v4 closes

Commit the code first (Step 1), then models and report. Keep `.tflite` and
`.sha256`, and decide whether models go in git or are ignored (the v3 artifacts
are currently untracked).

---

## 10. Order and time

| step | work | blocking? |
|---|---|---|
| 1 | acquisition code changes A1–A5 + tests ✅ | done |
| 3 | `data_extra` (no download) | can run during Step 1 once A5 (FIRMS) exists |
| 2 | pilot `data_202409b` ✅ | done (§3.1) |
| — | render profile A for the 7 v3 roots | anytime |
| 4 | Batch 1 → **train v4a (checkpoint 1)** | |
| 4 | Batch 2 (background, days) | |
| 5–6 | train v4, QAT, experiments, report | |

Open questions before Batch 2 (answer after the pilot):

1. Download speed: is the 4-file node download (§2, optional) worth building?
2. Cap per month: 90 (≈3/day), or the full month (~165)? Consecutive granules on
   the same day are correlated, so 90 is the default.
3. Should 2019 be a **held-out year** (all of its granules in test) for an
   out-of-time evaluation? This needs a `--force_split test` option on
   `run_split` (≈1 h). Recommended: it is the strongest test of the season confound.

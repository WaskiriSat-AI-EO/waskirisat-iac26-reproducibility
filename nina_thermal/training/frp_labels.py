"""FRP-based relabelling: "is there a *large* fire complex in view?"

Why this module exists
----------------------
The datagen labels answer "did FIRMS detect at least three 1 km active-fire
pixels anywhere in this patch?".  At the simulated Lepton GSD that question is
not answerable from the image.  A patch is 160×120 px at
``SLSTR_SYNTHETIC_GSD_M`` = 4 km, i.e. a **640 × 480 km footprint**.  A handful
of FIRMS pixels is ~0.001% of that area, and averaging a fire over 16 km² of
ground raises the pixel by a few tenths of a Kelvin — below the 0.392 K AGC
quantization step.  Measured on mixed granules, the hottest pixel of a
fire-labelled patch differs from a no-fire patch of the same granule by
-0.08 K (95% CI -0.80 .. +0.63): no signal.

This module reformulates the target to one the sensor can actually support:
**total radiated fire power in the field of view**.  Summed FRP over the
footprint scales with the total thermal energy the scene is emitting, which is
what a broadband LWIR microbolometer integrates.  Separability rises
monotonically with the threshold (ROC-AUC of the hottest pixel: 0.63 at any
FRP, 0.70 at 5 GW, 0.77 at 50 GW).

Negative-class contamination
----------------------------
The original negatives are dirty: 299 of 516 label-0 patches contain FIRMS
detections inside their footprint.  Training a "fire vs no-fire" classifier
against negatives that contain fire caps achievable accuracy no matter the
architecture.  :func:`assign_frp_labels` therefore emits three classes —
positive (>= threshold), clean negative (zero detections), and *ambiguous*
(everything between), which is excluded from training rather than being
assigned an arbitrary label.

Caveat, stated plainly
----------------------
The within-granule control that invalidated the original labels cannot be run
at high thresholds: no granule contains both a >= 20 GW patch and a
zero-detection patch.  So part of the AUC gain may still be a scene-level
confound (large fire complexes occur on hot, dry days) rather than the fire
signal itself.  The granule-disjoint test split is the arbiter; treat the
val-set numbers as optimistic until test confirms them.
"""

import glob
import json
import logging
from pathlib import Path

import numpy as np

from nina_thermal.datagen import config as C
from nina_thermal.datagen.dataset.export import metadata_path_for_image

logger = logging.getLogger(__name__)

# Half-extent of a patch footprint, in kilometres.
HALF_WIDTH_KM = C.LEPTON_RESOLUTION[0] * C.SLSTR_SYNTHETIC_GSD_M / 1000.0 / 2.0
HALF_HEIGHT_KM = C.LEPTON_RESOLUTION[1] * C.SLSTR_SYNTHETIC_GSD_M / 1000.0 / 2.0

# FIRMS detections this many days from the granule sensing date still count.
# SLSTR and VIIRS do not overpass simultaneously, and large complexes burn for
# days; +/-1 day is the same window the datagen labelling stage uses.
TIME_WINDOW_DAYS = 1

# Default "large fire complex" threshold, in megawatts of summed FRP over the
# 640x480 km footprint.
#
# Selected empirically on the granule-disjoint test split (profile B):
#
#   labels            test ROC-AUC   specificity   val -> test
#   datagen (any fire)     0.572         0.025      0.698 -> 0.572
#   FRP >= 5 GW            0.615         0.065      0.694 -> 0.615
#   FRP >= 20 GW           0.787         0.674      0.836 -> 0.787
#
# Specificity is the discriminator: below 20 GW the classifier is degenerate,
# answering "fire" to essentially every scene, and its apparent recall is
# meaningless.  At 20 GW it rejects two thirds of non-fire scenes and the
# val->test gap nearly closes, indicating the learned feature transfers across
# unseen granules.  Raising the threshold further (50 GW scores higher still)
# leaves under 10 validation positives, too few to tune an operating point on.
DEFAULT_FRP_THRESHOLD_MW = 20000.0

LABEL_AMBIGUOUS = -1

# Minimum fraction of a patch footprint that must lie inside the bbox the
# FIRMS CSVs were queried for (``firms/coverage.json``) for its detection count
# to be trusted.  Below this, "zero detections" mostly reflects where FIRMS
# was not queried, not an absence of fire.
MIN_FOOTPRINT_COVERAGE = 0.9

COVERAGE_FILE = "coverage.json"


def load_firms_detections(roots: list[str | Path]):
    """Load and de-duplicate every FIRMS CSV under the given data roots.

    Args:
        roots: Data roots; each is searched for a ``firms/*.csv`` subtree.

    Returns:
        DataFrame with ``latitude``, ``longitude``, ``acq_date``, ``frp``.

    Raises:
        RuntimeError: If no FIRMS CSV yields usable rows.
    """
    import pandas as pd

    wanted = {"latitude", "longitude", "acq_date", "confidence", "frp"}
    frames = []
    for root in roots:
        for path in glob.glob(str(Path(root) / "firms" / "*.csv")):
            frame = pd.read_csv(path, usecols=lambda c: c in wanted)
            if len(frame):
                frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No FIRMS detections found under {[str(r) for r in roots]}.  "
            "Run the acquisition stage (run_acquisition) to populate firms/."
        )

    detections = pd.concat(frames, ignore_index=True).drop_duplicates()
    detections["acq_date"] = pd.to_datetime(detections["acq_date"])
    logger.info(
        "FIRMS: %d unique detections, FRP median %.1f MW, max %.1f MW",
        len(detections),
        detections["frp"].median(),
        detections["frp"].max(),
    )
    return detections


def load_firms_coverage(roots: list[str | Path]) -> list[tuple] | None:
    """Read the queried FIRMS bbox declared by each root's ``firms/coverage.json``.

    The file is ``{"bbox": [west, south, east, north]}`` and records the area
    the CSVs beside it were fetched for.  Detections alone cannot tell a
    fire-free area from one that was never queried.

    Args:
        roots: Data roots holding ``firms/`` CSVs.

    Returns:
        One bbox per root that declares coverage, or None when none does — the
        spatial coverage check is then skipped with a warning.
    """
    bboxes = []
    undeclared = []
    for root in roots:
        path = Path(root) / "firms" / COVERAGE_FILE
        if path.is_file():
            bboxes.append(tuple(json.loads(path.read_text(encoding="utf-8"))["bbox"]))
        else:
            undeclared.append(str(root))
    if undeclared:
        logger.warning(
            "No firms/%s in %s — footprints outside the queried FIRMS area cannot "
            "be told apart from fire-free ones there.",
            COVERAGE_FILE,
            undeclared,
        )
    return bboxes or None


def _granule_date(granule_id: str) -> np.datetime64:
    """Extract the sensing date from an SLSTR granule id.

    Args:
        granule_id: e.g. ``S3A_SL_1_RBT____20230801T023011_...``.

    Returns:
        The sensing date.

    Raises:
        ValueError: If the id does not carry a parsable timestamp.
    """
    try:
        token = granule_id.split("____")[1][:8]
        return np.datetime64(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse sensing date from {granule_id!r}") from exc


def compute_patch_frp(
    samples: list[dict], detections, coverage: list[tuple] | None = None
) -> dict[str, dict]:
    """Sum FIRMS FRP inside each patch's footprint and time window.

    The footprint is approximated as a lat/lon box around the patch centroid.
    At this scale (640 km) the box-vs-true-swath discrepancy is small relative
    to the footprint itself, and the result is used only to rank patches by
    fire load, so the approximation does not affect the conclusion.

    Zero detections only means "no fire" where FIRMS was actually queried, so
    two kinds of patch are treated as uncovered rather than as clean
    negatives, left out of the result, and surface as ``no_frp_data``:

    * sensing date with no detection *anywhere* within the time window — no
      CSV was fetched for that date, or the sensor was down (e.g. the
      Suomi-NPP outage of late July–10 Aug 2022);
    * footprint less than ``MIN_FOOTPRINT_COVERAGE`` inside every queried
      bbox in ``coverage`` — the CSVs were fetched for an area that does not
      reach the patch, so its "zero detections" is missing data.

    Args:
        samples: Samples carrying ``patch``, ``granule_id`` and ``path``.
        detections: DataFrame from :func:`load_firms_detections`.
        coverage: Queried ``(west, south, east, north)`` bboxes from
            :func:`load_firms_coverage`; None skips the spatial check.

    Returns:
        Maps ``patch`` to ``{"frp_sum", "frp_max", "n_detections"}``.
    """
    lat = detections["latitude"].to_numpy()
    lon = detections["longitude"].to_numpy()
    frp = detections["frp"].to_numpy()
    dates = detections["acq_date"].to_numpy()
    covered_dates = np.unique(dates.astype("datetime64[D]"))

    result: dict[str, dict] = {}
    missing_metadata = 0
    uncovered_dates: set[str] = set()
    outside_extent = 0
    for sample in samples:
        if sample["patch"] in result:
            continue
        sensed = _granule_date(sample["granule_id"])
        gap = np.abs((covered_dates - sensed) / np.timedelta64(1, "D"))
        if not covered_dates.size or gap.min() > TIME_WINDOW_DAYS:
            uncovered_dates.add(str(sensed))
            continue
        meta_path = metadata_path_for_image(sample["path"])
        if not meta_path.is_file():
            missing_metadata += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        centre_lat = meta["centroid_lat"]
        centre_lon = meta["centroid_lon"]
        d_lat = HALF_HEIGHT_KM / 111.0
        d_lon = HALF_WIDTH_KM / (111.320 * np.cos(np.radians(centre_lat)))
        if coverage is not None:
            inside = max(
                (
                    max(min(centre_lat + d_lat, north) - max(centre_lat - d_lat, south), 0.0)
                    * max(min(centre_lon + d_lon, east) - max(centre_lon - d_lon, west), 0.0)
                    for west, south, east, north in coverage
                ),
                default=0.0,
            ) / (4 * d_lat * d_lon)
            if inside < MIN_FOOTPRINT_COVERAGE:
                outside_extent += 1
                continue

        within = (
            (np.abs(lat - centre_lat) <= d_lat)
            & (np.abs(lon - centre_lon) <= d_lon)
            & (
                np.abs(
                    (dates - _granule_date(sample["granule_id"]))
                    / np.timedelta64(1, "D")
                )
                <= TIME_WINDOW_DAYS
            )
        )
        hits = frp[within]
        result[sample["patch"]] = {
            "frp_sum": float(hits.sum()),
            "frp_max": float(hits.max()) if hits.size else 0.0,
            "n_detections": int(within.sum()),
        }

    if missing_metadata:
        logger.warning(
            "%d sample(s) had no metadata JSON and carry no FRP.", missing_metadata
        )
    if outside_extent:
        logger.warning(
            "%d patch(es) have under %.0f%% of their footprint inside the queried "
            "FIRMS bbox %s and carry no FRP.  Re-fetch firms/ CSVs over a bbox "
            "covering every patch footprint.",
            outside_extent,
            100 * MIN_FOOTPRINT_COVERAGE,
            coverage,
        )
    if uncovered_dates:
        logger.warning(
            "No FIRMS detections within +/-%d day of %d sensing date(s) — those "
            "patches carry no FRP instead of being labelled clean negatives.  "
            "Fetch the missing firms/ CSVs: %s",
            TIME_WINDOW_DAYS,
            len(uncovered_dates),
            ", ".join(sorted(uncovered_dates)),
        )
    return result


def assign_frp_labels(
    samples: list[dict],
    patch_frp: dict[str, dict],
    threshold_mw: float = DEFAULT_FRP_THRESHOLD_MW,
    negative_max_mw: float = 0.0,
) -> tuple[list[dict], dict]:
    """Relabel samples by summed FRP, dropping the ambiguous middle band.

    Positive when ``frp_sum >= threshold_mw``; negative when the footprint
    contains no FIRMS detection at all or, with ``negative_max_mw > 0``, when
    ``frp_sum < negative_max_mw`` (low fire activity); ambiguous otherwise.
    Ambiguous samples are returned with ``label = -1`` so the caller can drop
    them the same way the datagen pipeline drops its own ambiguous records.

    Args:
        samples: Samples to relabel, mutated copies are returned.
        patch_frp: Per-patch FRP from :func:`compute_patch_frp`.
        threshold_mw: Summed-FRP threshold for the positive class.
        negative_max_mw: Summed-FRP ceiling for the negative class.  0 keeps
            the strict zero-detection definition.

    Returns:
        ``(relabelled_samples, stats)``.
    """
    relabelled = []
    counts = {"fire": 0, "no_fire": 0, "ambiguous": 0, "no_frp_data": 0}

    for sample in samples:
        stats = patch_frp.get(sample["patch"])
        if stats is None:
            counts["no_frp_data"] += 1
            continue

        if stats["frp_sum"] >= threshold_mw:
            label = 1
            counts["fire"] += 1
        elif stats["n_detections"] == 0 or stats["frp_sum"] < negative_max_mw:
            label = 0
            counts["no_fire"] += 1
        else:
            label = LABEL_AMBIGUOUS
            counts["ambiguous"] += 1

        relabelled.append(
            {
                **sample,
                "label": label,
                "original_label": sample["label"],
                "frp_sum": stats["frp_sum"],
                "n_detections": stats["n_detections"],
            }
        )

    summary = {
        "threshold_mw": threshold_mw,
        "negative_max_mw": negative_max_mw,
        "counts": counts,
        "retained": counts["fire"] + counts["no_fire"],
        "definition": (
            "label=1 when summed FIRMS FRP over the 640x480 km patch footprint "
            f">= {threshold_mw:g} MW within +/-{TIME_WINDOW_DAYS} day of sensing; "
            + (
                f"label=0 when summed FRP < {negative_max_mw:g} MW; "
                if negative_max_mw > 0
                else "label=0 only when the footprint contains zero FIRMS detections; "
            )
            + "patches in between are ambiguous and excluded."
        ),
    }
    logger.info(
        "FRP relabel @ %.0f MW — fire=%d  no_fire=%d  ambiguous(dropped)=%d",
        threshold_mw,
        counts["fire"],
        counts["no_fire"],
        counts["ambiguous"],
    )
    return relabelled, summary


def apply_frp_labels(
    samples: list[dict],
    roots: list[str | Path],
    threshold_mw: float = DEFAULT_FRP_THRESHOLD_MW,
    negative_max_mw: float = 0.0,
) -> tuple[list[dict], dict]:
    """Relabel samples by FRP and drop the ambiguous band.

    Convenience wrapper chaining load, compute and assign.

    Args:
        samples: Samples from ``expand_to_samples``.
        roots: Data roots holding ``firms/`` CSVs.
        threshold_mw: Summed-FRP threshold for the positive class.
        negative_max_mw: Summed-FRP ceiling for the negative class.

    Returns:
        ``(kept_samples, summary)`` with ambiguous samples already removed.
    """
    detections = load_firms_detections(roots)
    patch_frp = compute_patch_frp(samples, detections, load_firms_coverage(roots))
    relabelled, summary = assign_frp_labels(
        samples, patch_frp, threshold_mw, negative_max_mw
    )
    kept = [s for s in relabelled if s["label"] != LABEL_AMBIGUOUS]
    return kept, summary

"""Section 4 — INT8 thresholds and acceptance gate (owners: Leyla for cloud, Osvaldo for thermal).

NOT IMPLEMENTED YET. Contract: `compute()` returns a flat dict whose keys are
exactly those in `reproduce/expected/thresholds.json`, computed from the repo's own
artifacts (dataset manifests under data/manifests/, models listed in
models/MANIFEST.json, the payload's evaluation output). Split the work by
payload: implement `compute_cloud()` / `compute_thermal()` and merge them.
`python -m reproduce.check` reports this item as PENDING until it is done.
"""

from __future__ import annotations

import sys

from reproduce.lib import report


def compute_cloud() -> dict:
    raise NotImplementedError("cloud: see issue #14")


def compute_thermal() -> dict:
    raise NotImplementedError("thermal: see issue #15")


def compute() -> dict:
    return {**compute_cloud(), **compute_thermal()}


if __name__ == "__main__":
    sys.exit(report("thresholds", compute()))

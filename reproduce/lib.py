"""Shared helpers for the paper-reproduction scripts.

Each ``tableN.py`` computes its numbers from raw artifacts in the repo and
returns a flat ``{key: value}`` dict. ``check`` compares that dict with the
paper's values stored in ``reproduce/expected/<name>.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "reproduce" / "expected"
HIL_RESULTS = ROOT / "hil" / "results"


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_expected(name: str) -> dict:
    return load_json(EXPECTED / f"{name}.json")


def compare(name: str, got: dict) -> list[str]:
    """Return a list of mismatch descriptions (empty when all values agree).

    ``expected/<name>.json`` maps key -> {"value": x, "tol": t}; ``tol`` is an
    absolute tolerance (0 for exact match; strings compare exactly). Keys with
    ``"value": null`` are not yet delivered and are reported, not ignored.
    """
    problems = []
    for key, spec in load_expected(name).items():
        want, tol = spec["value"], spec.get("tol", 0)
        if want is None:
            problems.append(f"{key}: expected value not filled in yet")
        elif key not in got:
            problems.append(f"{key}: not produced (want {want})")
        elif isinstance(want, str) or isinstance(got[key], str):
            if got[key] != want:
                problems.append(f"{key}: got {got[key]!r}, paper says {want!r}")
        elif not math.isclose(got[key], want, rel_tol=0, abs_tol=tol):
            problems.append(f"{key}: got {got[key]}, paper says {want} (tol {tol})")
    return problems


def report(name: str, got: dict) -> int:
    """Print computed values and the comparison; return a process exit code."""
    for key, value in got.items():
        print(f"  {key:45s} {value}")
    problems = compare(name, got)
    if problems:
        print(f"\n{name}: {len(problems)} issue(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\n{name}: all values match the paper")
    return 0

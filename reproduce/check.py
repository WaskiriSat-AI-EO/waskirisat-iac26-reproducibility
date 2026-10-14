"""Run every paper-reproduction item and compare with the paper's values.

    python -m reproduce.check            # PENDING items do not fail the run
    python -m reproduce.check --strict   # PENDING items fail the run

Prints one line per item: PASS, FAIL (values differ from the paper) or
PENDING (script not implemented yet).
"""

from __future__ import annotations

import importlib
import sys

from reproduce.lib import compare

ITEMS = ["table5", "table6", "thresholds", "table7", "table8", "test2_summary"]


def main(strict: bool = False) -> int:
    failed = pending = 0
    for name in ITEMS:
        module = importlib.import_module(f"reproduce.{name}")
        try:
            problems = compare(name, module.compute())
        except NotImplementedError as exc:
            print(f"PENDING  {name}  ({exc})")
            pending += 1
            continue
        if problems:
            print(f"FAIL     {name}")
            for p in problems:
                print(f"           - {p}")
            failed += 1
        else:
            print(f"PASS     {name}")
    print(f"\n{len(ITEMS) - failed - pending} pass, {failed} fail, {pending} pending")
    return 1 if failed or (strict and pending) else 0


if __name__ == "__main__":
    sys.exit(main(strict="--strict" in sys.argv))

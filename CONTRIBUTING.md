# Contributing

This repository is the presentation and reproducibility repo of the IAC-26 paper. Each area has an
owner and a delivery checklist in its README; open work is tracked in the
[`iac26-repro`](https://github.com/WaskiriSat-AI-EO/waskirisat-pl-aidev/milestone/1) milestone.

| Area | Owner | Checklist | Issue |
|---|---|---|---|
| `nina_cloud/`, `models/cloud/` | Leyla | `nina_cloud/README.md` | #14 |
| `nina_thermal/`, `models/thermal/` | Osvaldo | `nina_thermal/README.md` | #15 |
| `hil/` | Misael | `hil/README.md` | #16 |
| `reproduce/`, `docs/`, root files | Christian | issues #17–#19 | #17–#19 |

## Ground rules

1. **Code, comments, docstrings, commit messages and docs are written in English.**
2. **Never merge with a red CI.** `main` is not branch-protected, so this is a convention: wait for the
   `ci` check on your PR to pass before merging.
3. **Do not commit binaries or datasets.** No `.tflite`, `.pt`, images or raw data. Publish them outside git
   and record the file name, size and sha256 in `models/MANIFEST.json` (models) or a manifest under
   `data/manifests/` (data).
4. **Do not tune the results.** The paper reports some negative results on purpose (the thermal INT8
   acceptance gate fails; the cloud device path is not numerically equivalent to training). Reproduce them as
   they are.
5. **Do not edit `hil/` code "to clean it up".** It must match the code that produced the published results. If
   it changes, the results must be re-run and `hil/results/` replaced.
6. **Evidence tiers.** Every number you add to a report or card carries its tier (T1 traceable, T2 modeled,
   T3 assumed, T4 deferred); definitions in `docs/evidence-and-ladder.md`.

## Workflow

```bash
git switch main && git pull
git switch -c <area>/<short-description>      # e.g. cloud/paper-config, thermal/eval-report, hil/protocol-spec
# ... work ...
make lint test reproduce                      # must be clean before opening the PR
git push -u origin <branch>
gh pr create --base main                      # or use the GitHub UI
```

- One branch per deliverable, named `cloud/…`, `thermal/…`, `hil/…`, `docs/…`, `ci/…` or `chore/…`.
- Open the PR against `main`, link the issue (`Closes #14` only when the whole checklist is done, otherwise
  `Refs #14`) and tick the checklist items the PR completes.
- `CODEOWNERS` requests the area owner as reviewer. Cross-area changes need both owners' approval.
- Squash-merge. The PR title becomes the commit message (`type(scope): summary`).

## Delivering a model or a result

1. Put code and the exact configuration in your area (`configs/paper.yaml`, copied from
   `paper.yaml.template`).
2. Publish the binary outside git; add its entry to `models/MANIFEST.json` (id, file, sha256, size, owner,
   url). Run `make verify-models`.
3. Implement the stub in `reproduce/` for your table (`table5.py`, `table6.py`, `thresholds.py`): `compute()`
   returns a flat dict with the keys of `reproduce/expected/<name>.json`. Run `make reproduce` until your item
   shows PASS.
4. Fill in the model card and dataset card from `docs/templates/` and place them in your area.
5. Update the status column of `reproduce/paper_map.md`.

If a value differs from the paper, **do not edit `reproduce/expected/`**. Open an issue describing the
discrepancy; the paper's figure is the reference until the discrepancy is resolved and recorded in the errata.

## Checks

| Command | What it does |
|---|---|
| `make test` | Unit tests (`pytest`) |
| `make lint` | `black --check` and `ruff check` |
| `make reproduce` | Recompute every paper item and compare with the paper (PENDING is allowed) |
| `make verify-models` | Check published binaries against `models/MANIFEST.json` |

CI runs the formatter, linter, mypy, the tests and `python -m reproduce.check` on every push and PR.

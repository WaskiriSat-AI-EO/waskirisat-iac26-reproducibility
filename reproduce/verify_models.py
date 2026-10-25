"""Check that every artifact listed in models/MANIFEST.json matches its sha256.

Files are looked up next to the manifest (models/<payload>/<file>) or at the
optional ``repo_path`` of the entry (relative to the repo root). Artifacts
whose binary is not present are reported as MISSING, not as failures, until a
download URL is filled in.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads((MODELS / "MANIFEST.json").read_text())
    bad = 0
    for art in manifest["artifacts"]:
        candidates = list((MODELS / art["payload"]).rglob(art["file"]))
        # some artifacts are tracked in git outside models/ (repo_path)
        if "repo_path" in art and (ROOT / art["repo_path"]).exists():
            candidates.append(ROOT / art["repo_path"])
        if not candidates:
            print(f"MISSING  {art['id']}  ({art['file']})")
            continue
        ok = sha256(candidates[0]) == art["sha256"]
        print(f"{'OK     ' if ok else 'MISMATCH'}  {art['id']}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

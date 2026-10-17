"""v4 acquisition: even sampling, skip-held, streamed extraction, inventory."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nina_thermal.datagen.acquisition.cdse_search import (
    drop_held_products,
    granule_key,
    load_held_granules,
    products_per_day,
    select_products,
)
from nina_thermal.datagen.pipelines import run_acquisition

ROI = str(
    Path(__file__).resolve().parents[2] / "resources" / "bo_nina_roi_simplified.geojson"
)
MOD = "nina_thermal.datagen.pipelines.run_acquisition"


def _product(day: int, slot: int, month: str = "2024-09") -> dict:
    stamp = f"{month.replace('-', '')}{day:02d}T{slot:02d}0000"
    return {
        "Id": f"id-{day}-{slot}",
        "Name": f"S3A_SL_1_RBT____{stamp}_X_PS1_O_NT_004.SEN3",
        "ContentDate": {"Start": f"{month}-{day:02d}T{slot:02d}:00:00.000Z"},
    }


def _month(days=range(10, 31), per_day=6) -> list:
    return [_product(d, s) for d in days for s in range(per_day)]


# ── select_products ──────────────────────────────────────────────────────────


def test_even_sampling_covers_every_day():
    products = _month()  # 21 days × 6
    chosen = select_products(products, 60, "even")
    per_day = Counter(p["ContentDate"]["Start"][:10] for p in chosen)
    assert len(chosen) == 60
    assert len(per_day) == 21
    assert min(per_day.values()) == 2 and max(per_day.values()) == 3


def test_even_sampling_spreads_partial_round():
    """The last partial round must not all land on the first days."""
    chosen = select_products(_month(), 60, "even")
    third = sorted(
        d
        for d, n in Counter(p["ContentDate"]["Start"][:10] for p in chosen).items()
        if n == 3
    )
    assert third[0][8:] == "10" and int(third[-1][8:]) >= 28


def test_oldest_sampling_is_pre_v4_behaviour():
    chosen = select_products(_month(), 12, "oldest")
    assert {p["ContentDate"]["Start"][:10] for p in chosen} == {
        "2024-09-10",
        "2024-09-11",
    }


def test_cap_above_available_returns_all_in_order():
    products = list(reversed(_month(days=range(1, 4), per_day=2)))
    chosen = select_products(products, 100, "even")
    starts = [p["ContentDate"]["Start"] for p in chosen]
    assert len(chosen) == 6 and starts == sorted(starts)


def test_unknown_sampling_rejected():
    with pytest.raises(ValueError):
        select_products(_month(), 5, "random")


def test_products_per_day():
    assert products_per_day(_month(days=[1, 2], per_day=3)) == {
        "2024-09-01": 3,
        "2024-09-02": 3,
    }


# ── held granules ────────────────────────────────────────────────────────────


def test_granule_key_strips_sen3():
    assert granule_key("ABC.SEN3\n") == "ABC" == granule_key("ABC")


def test_load_and_drop_held(tmp_path):
    products = _month(days=[10, 11], per_day=2)
    inv = tmp_path / "rootA" / "01_raw" / "granule_inventory.txt"
    inv.parent.mkdir(parents=True)
    inv.write_text(products[0]["Name"] + "\n" + granule_key(products[3]["Name"]) + "\n")
    held = load_held_granules(
        [str(tmp_path / "root*" / "01_raw" / "granule_inventory.txt")]
    )
    kept, skipped = drop_held_products(products, held)
    assert [p["Id"] for p in skipped] == [products[0]["Id"], products[3]["Id"]]
    assert len(kept) == 2


def test_load_held_unmatched_glob_is_empty(tmp_path):
    assert load_held_granules([str(tmp_path / "nothing*.txt")]) == set()


# ── run_acquisition end to end (CDSE mocked) ─────────────────────────────────


def _fake_download(access_token, product_id, product_name, output_dir, delete_zip):
    sen3 = Path(output_dir) / product_name
    sen3.mkdir(parents=True)
    (sen3 / "S8_BT_in.nc").write_bytes(b"x" * 10)
    return True


def _fake_extract(sen3_dir, output_dir, prefix=None):
    prefix = granule_key(Path(sen3_dir).name)
    paths = []
    for suffix in ("F2_BT", "S8_BT", "lat", "lon"):
        p = Path(output_dir) / f"{prefix}_{suffix}.npy"
        np.save(p, np.zeros((2, 2), dtype=np.float32))
        paths.append(p)
    return tuple(paths)


def _run(tmp_path, products, extra, download=_fake_download):
    with patch(f"{MOD}.time.sleep"):
        return _run_inner(tmp_path, products, extra, download)


def _run_inner(tmp_path, products, extra, download):
    argv = [
        "--roi",
        ROI,
        "--start",
        "20240910",
        "--end",
        "20240930",
        "--output_dir",
        str(tmp_path / "root" / "01_raw"),
        "--log_dir",
        str(tmp_path / "logs"),
    ] + extra
    with patch(f"{MOD}.get_access_token", return_value="tok"), patch(
        f"{MOD}.search_slstr_products", return_value=list(products)
    ), patch(f"{MOD}.download_and_extract", side_effect=download) as dl, patch(
        f"{MOD}.extract_thermal_channels", side_effect=_fake_extract
    ):
        code = run_acquisition.main(argv)
    return code, dl


def test_dry_run_downloads_nothing(tmp_path, capsys):
    code, dl = _run(tmp_path, _month(), ["--max_products", "30", "--dry_run"])
    assert code == 0 and dl.call_count == 0
    out = capsys.readouterr().out
    assert "30 selected (21 days)" in out
    assert not (tmp_path / "root" / "01_raw" / "granule_inventory.txt").exists()


def test_streamed_extract_deletes_raw_and_writes_inventory(tmp_path):
    products = _month(days=[10, 11, 12], per_day=2)
    raw = tmp_path / "root" / "01_raw"
    ext = tmp_path / "root" / "02_extracted"
    code, dl = _run(
        tmp_path,
        products,
        ["--max_products", "4", "--extract_dir", str(ext), "--delete_raw"],
    )
    assert code == 0 and dl.call_count == 4
    assert not list(raw.glob("*.SEN3"))  # raw deleted
    assert len(list(ext.glob("*_F2_BT.npy"))) == 4
    inventory = (raw / "granule_inventory.txt").read_text().split()
    assert len(inventory) == 4 and all(n.endswith(".SEN3") for n in inventory)
    manifest = json.loads((raw / "acquisition_manifest.json").read_text())
    assert manifest["total_downloaded"] == 4
    assert all(not g["sen3_dir"].endswith(".SEN3.SEN3") for g in manifest["granules"])
    assert manifest["args"]["sampling"] == "even"
    ext_manifest = json.loads((ext / "extraction_manifest.json").read_text())
    assert ext_manifest["total_extracted"] == 4


def test_rerun_resumes_without_redownloading(tmp_path):
    products = _month(days=[10, 11, 12], per_day=2)
    ext = tmp_path / "root" / "02_extracted"
    args = ["--max_products", "4", "--extract_dir", str(ext), "--delete_raw"]
    _run(tmp_path, products, args)
    # Own inventory matches the skip glob too: it must resume, not skip.
    skip = [
        "--skip_inventory",
        str(tmp_path / "*" / "01_raw" / "granule_inventory.txt"),
    ]
    code, dl = _run(tmp_path, products, args + skip)
    assert code == 0 and dl.call_count == 0
    manifest = json.loads(
        (tmp_path / "root" / "01_raw" / "acquisition_manifest.json").read_text()
    )
    assert manifest["total_downloaded"] == 4


def test_skip_inventory_of_other_root(tmp_path):
    products = _month(days=[10, 11], per_day=2)
    other = tmp_path / "other" / "01_raw" / "granule_inventory.txt"
    other.parent.mkdir(parents=True)
    other.write_text("\n".join(p["Name"] for p in products[:3]) + "\n")
    code, dl = _run(
        tmp_path,
        products,
        ["--max_products", "10", "--skip_inventory", str(other)],
    )
    assert code == 0 and dl.call_count == 1
    assert dl.call_args.kwargs["product_name"] == products[3]["Name"]


def test_failed_download_sets_exit_code(tmp_path):
    code, _ = _run(
        tmp_path,
        _month(days=[10], per_day=1),
        ["--max_products", "1"],
        download=lambda **kw: False,
    )
    assert code == 1


def test_delete_raw_requires_extract_dir(tmp_path):
    with pytest.raises(SystemExit):
        run_acquisition.main(
            ["--roi", ROI, "--output_dir", str(tmp_path), "--delete_raw"]
        )


def test_tiny_roi_rejected(tmp_path):
    tiny = str(Path(ROI).with_name("bolivia_roi.geojson"))
    with patch(f"{MOD}.get_access_token", return_value="tok"), patch(
        f"{MOD}.search_slstr_products"
    ) as search:
        code = run_acquisition.main(
            [
                "--roi",
                tiny,
                "--output_dir",
                str(tmp_path / "raw"),
                "--log_dir",
                str(tmp_path / "logs"),
            ]
        )
    assert code == 1 and search.call_count == 0

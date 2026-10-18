"""run_fetch_firms: chunk planning, product rules, coverage.json."""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from nina_thermal.datagen.acquisition.firms_api import (
    daily_counts,
    firms_csv_name,
    incomplete_days,
    parse_firms_csv_name,
    plan_firms_chunks,
    source_conflicts,
    unavailable_chunks,
)
from nina_thermal.datagen.pipelines import run_fetch_firms

API = "nina_thermal.datagen.acquisition.firms_api.requests.get"
HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight,type\n"
)
CSV = HEADER + "-15.0,-62.0,330.0,0.5,0.4,2023-09-01,433,N,VIIRS,n,2,290.0,12.5,N,0\n"


def _names(chunks):
    return [firms_csv_name(*c) for c in chunks]


def test_plan_matches_existing_root_layout():
    """Granules 2023-09-01…26 → the CSVs on disk in data_202309/firms/."""
    chunks = plan_firms_chunks(date(2023, 9, 1), date(2023, 9, 26))
    assert _names(chunks) == [
        "firms_VIIRS_SNPP_SP_2023-08-31_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-05_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-10_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-15_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-20_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-25_3d.csv",
    ]


def test_plan_switches_product_for_snpp_outage():
    chunks = plan_firms_chunks(date(2022, 8, 1), date(2022, 8, 22))
    assert _names(chunks)[:3] == [
        "firms_VIIRS_NOAA20_SP_2022-07-31_5d.csv",
        "firms_VIIRS_NOAA20_SP_2022-08-05_5d.csv",
        "firms_VIIRS_NOAA20_SP_2022-08-10_1d.csv",
    ]
    assert all(src == "VIIRS_SNPP_SP" for src, _, _ in chunks[3:])
    covered = sum(n for _, _, n in chunks)
    assert covered == 24  # 07-31 … 08-23, every day exactly once


def test_csv_name_round_trip():
    chunk = ("VIIRS_NOAA20_SP", date(2022, 8, 10), 1)
    assert parse_firms_csv_name(firms_csv_name(*chunk)) == chunk
    assert parse_firms_csv_name("coverage.json") is None


def test_source_conflicts_flags_mixed_products():
    chunks = plan_firms_chunks(date(2023, 8, 1), date(2023, 8, 9))
    existing = [
        "firms_VIIRS_SNPP_NRT_2023-08-01_5d.csv",  # overlaps, other product
        "firms_VIIRS_SNPP_SP_2023-08-05_5d.csv",  # same product: fine
        "firms_VIIRS_SNPP_NRT_2024-08-01_5d.csv",  # other dates: fine
    ]
    assert source_conflicts(existing, chunks) == [
        "firms_VIIRS_SNPP_NRT_2023-08-01_5d.csv"
    ]


def test_unavailable_chunks():
    chunks = [("VIIRS_SNPP_SP", date(2025, 9, 28), 5)]
    avail = {"VIIRS_SNPP_SP": (date(2012, 1, 20), date(2025, 9, 30))}
    assert unavailable_chunks(chunks, avail) == chunks
    assert (
        unavailable_chunks(
            chunks, {"VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 3, 31))}
        )
        == []
    )


# ── per-day completeness ─────────────────────────────────────────────────────


def _csv(per_day):
    rows = [
        f"-15.0,-62.0,330.0,0.5,0.4,{d.isoformat()},433,N,VIIRS,n,2,290.0,12.5,N,0\n"
        for d, n in per_day.items()
        for _ in range(n)
    ]
    return HEADER + "".join(rows)


def test_daily_counts():
    assert daily_counts(_csv({date(2024, 7, 9): 3, date(2024, 7, 10): 1})) == {
        date(2024, 7, 9): 3,
        date(2024, 7, 10): 1,
    }
    assert daily_counts("") == {}


def test_incomplete_days_flags_missing_and_partial():
    days = [date(2024, 7, 22) + timedelta(days=i) for i in range(4)]
    counts = {days[0]: 7121, days[1]: 10495, days[2]: 782}  # 07-25 missing
    reference = {days[0]: 7000, days[1]: 9010, days[2]: 13045, days[3]: 10746}
    flagged = incomplete_days(counts, days, reference)
    assert sorted(flagged) == [days[2], days[3]]
    assert "0 detections" in flagged[days[3]]
    # Without a reference only the empty day is caught.
    assert sorted(incomplete_days(counts, days)) == [days[3]]


def test_incomplete_days_ignores_quiet_reference():
    day = date(2023, 2, 10)
    assert incomplete_days({day: 20}, [day], {day: 150}) == {}


def test_outage_config_covers_july_2024():
    chunks = plan_firms_chunks(date(2024, 7, 1), date(2024, 7, 31))
    noaa = {
        day0 + timedelta(days=i)
        for src, day0, n in chunks
        if src == "VIIRS_NOAA20_SP"
        for i in range(n)
    }
    assert date(2024, 7, 10) in noaa and date(2024, 7, 16) in noaa
    assert date(2024, 7, 24) in noaa and date(2024, 7, 28) in noaa
    assert date(2024, 7, 20) not in noaa


# ── CLI (FIRMS mocked) ───────────────────────────────────────────────────────


def _response(text):
    r = MagicMock()
    r.text = text
    r.raise_for_status.return_value = None
    return r


def _get(count=lambda src, day: 300, max_date="2026-03-31"):
    """Fake FIRMS: *count(src, day)* detections on every requested day."""
    availability = (
        "data_id,min_date,max_date\n"
        f"VIIRS_SNPP_SP,2012-01-20,{max_date}\n"
        f"VIIRS_NOAA20_SP,2018-04-01,{max_date}\n"
    )

    def fake(url, timeout):
        if "data_availability" in url:
            return _response(availability)
        src, _bbox, n, day0 = url.split("/")[-4:]
        first = date.fromisoformat(day0)
        days = [first + timedelta(days=i) for i in range(int(n))]
        return _response(_csv({d: count(src, d) for d in days}))

    return fake


def _run(tmp_path, *extra, get=None):
    argv = [
        "--start",
        "20230901",
        "--end",
        "20230910",
        "--out_dir",
        str(tmp_path / "firms"),
        "--firms_key",
        "k",
        "--log_dir",
        str(tmp_path / "logs"),
    ] + list(extra)
    with patch(API, side_effect=get or _get()) as mock_get:
        return run_fetch_firms.main(argv), mock_get


def _area_urls(mock_get, source="VIIRS_SNPP_SP"):
    return [c.args[0] for c in mock_get.call_args_list if f"/{source}/" in c.args[0]]


def test_fetch_writes_csvs_and_coverage(tmp_path):
    code, mock_get = _run(tmp_path)
    assert code == 0
    firms = tmp_path / "firms"
    assert sorted(p.name for p in firms.glob("*.csv")) == [
        "firms_VIIRS_SNPP_SP_2023-08-31_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-05_5d.csv",
        "firms_VIIRS_SNPP_SP_2023-09-10_2d.csv",
    ]
    coverage = json.loads((firms / "coverage.json").read_text())
    assert coverage["bbox"] == [-77.0, -29.0, -52.0, -5.0]
    assert coverage["windows"] == [{"start": "20230901", "end": "20230910"}]
    assert "/VIIRS_SNPP_SP/-77,-29,-52,-5/5/2023-08-31" in _area_urls(mock_get)[0]
    assert len(_area_urls(mock_get, "VIIRS_NOAA20_SP")) == 3  # cross-check

    # Re-run keeps files and does not refetch them.
    code, mock_get = _run(tmp_path)
    assert code == 0
    assert not _area_urls(mock_get)


def test_missing_day_inside_chunk_fails(tmp_path):
    gap = date(2023, 9, 7)
    get = _get(count=lambda src, d: 0 if (src == "VIIRS_SNPP_SP" and d == gap) else 300)
    code, _ = _run(tmp_path, get=get)
    firms = tmp_path / "firms"
    assert code == 1
    assert not (firms / "firms_VIIRS_SNPP_SP_2023-09-05_5d.csv").exists()
    assert (firms / "firms_VIIRS_SNPP_SP_2023-08-31_5d.csv").exists()
    assert not (firms / "coverage.json").exists()


def test_partial_day_fails_unless_allowed(tmp_path):
    part = date(2023, 9, 2)
    get = _get(
        count=lambda src, d: 30 if (src == "VIIRS_SNPP_SP" and d == part) else 300
    )
    code, _ = _run(tmp_path, get=get)
    assert code == 1
    assert not (tmp_path / "firms" / "firms_VIIRS_SNPP_SP_2023-08-31_5d.csv").exists()
    code, _ = _run(tmp_path, "--allow_incomplete", get=get)
    assert code == 0


def test_no_crosscheck_skips_reference(tmp_path):
    code, mock_get = _run(tmp_path, "--no_crosscheck")
    assert code == 0 and not _area_urls(mock_get, "VIIRS_NOAA20_SP")


def test_check_only_audits_existing_files(tmp_path, capsys):
    firms = tmp_path / "firms"
    firms.mkdir()
    first = date(2023, 8, 31)
    good = {first + timedelta(days=i): 300 for i in range(5)}
    bad = dict(good)
    bad[first + timedelta(days=2)] = 10
    (firms / firms_csv_name("VIIRS_SNPP_SP", first, 5)).write_text(_csv(good))
    code, _ = _run(tmp_path, "--check_only")
    assert code == 0
    (firms / firms_csv_name("VIIRS_SNPP_SP", first, 5)).write_text(_csv(bad))
    code, mock_get = _run(tmp_path, "--check_only")
    out = capsys.readouterr().out
    assert code == 1 and "FLAG  firms_VIIRS_SNPP_SP_2023-08-31_5d.csv" in out
    assert not _area_urls(mock_get)  # nothing refetched, nothing written


def test_window_beyond_availability_fails(tmp_path):
    code, _ = _run(tmp_path, get=_get(max_date="2023-09-05"))
    assert code == 1


def test_bbox_mismatch_fails(tmp_path):
    firms = tmp_path / "firms"
    firms.mkdir()
    (firms / "coverage.json").write_text(
        json.dumps({"bbox": [-67.5, -20.5, -59.7, -11.8]})
    )
    code, _ = _run(tmp_path)
    assert code == 1


def test_existing_nrt_csv_blocks_fetch(tmp_path):
    firms = tmp_path / "firms"
    firms.mkdir()
    (firms / "firms_VIIRS_SNPP_NRT_2023-09-01_5d.csv").write_text(CSV)
    code, _ = _run(tmp_path)
    assert code == 1


def test_dry_run_makes_no_requests(tmp_path, capsys):
    code, mock_get = _run(tmp_path, "--dry_run")
    assert code == 0 and mock_get.call_count == 0
    assert "firms_VIIRS_SNPP_SP_2023-08-31_5d.csv" in capsys.readouterr().out

"""The HIL-derived items must reproduce the paper from the committed raw results."""

import pytest

from reproduce import check, table5, table6, table7, table8, test2_summary, thresholds
from reproduce.lib import compare


@pytest.mark.parametrize(
    "name,module",
    [("table7", table7), ("table8", table8), ("test2_summary", test2_summary)],
)
def test_hil_items_match_paper(name, module):
    assert compare(name, module.compute()) == []


@pytest.mark.parametrize(
    "name,module",
    [("table5", table5), ("table6", table6), ("thresholds", thresholds)],
)
def test_thermal_items_match_paper(name, module):
    assert compare(name, module.compute_thermal(), payload="thermal") == []


def test_compare_flags_mismatch_and_missing(tmp_path, monkeypatch):
    import json

    from reproduce import lib

    (tmp_path / "x.json").write_text(
        json.dumps({"a": {"value": 1.0, "tol": 0.1}, "b": {"value": None}})
    )
    monkeypatch.setattr(lib, "EXPECTED", tmp_path)
    problems = compare("x", {"a": 2.0})
    assert len(problems) == 2


def test_check_runs_and_table_items_pass(capsys):
    assert check.main(strict=False) == 0
    out = capsys.readouterr().out
    for item in ("table5", "table6", "thresholds", "table7"):
        assert f"PASS     {item}" in out

from unittest.mock import patch, MagicMock
import pandas as pd


from nina_thermal.datagen.acquisition.firms_api import (
    _normalise_confidence,
    query_firms_fire_dates,
)


def test_normalise_confidence_numeric():
    # MODIS-style numeric confidence
    s = pd.Series([10, 30, 85])
    normalized = _normalise_confidence(s)
    assert normalized.iloc[0] == "low"
    assert normalized.iloc[1] == "nominal"
    assert normalized.iloc[2] == "high"


def test_normalise_confidence_string():
    # VIIRS-style letter/word confidence
    s = pd.Series(["l", "n", "h", "low", "nominal", "high"])
    normalized = _normalise_confidence(s)
    assert list(normalized) == ["low", "nominal", "high", "low", "nominal", "high"]


@patch("nina_thermal.datagen.acquisition.firms_api.requests.get")
def test_query_firms_fire_dates_success(mock_get):
    # Mock a CSV response containing active fire detections
    mock_csv = (
        "latitude,longitude,acq_date,confidence,frp\n"
        "-15.0,-62.0,2023-08-12,nominal,15.5\n"
        "-15.1,-62.1,2023-08-12,high,25.0\n"
        "-15.2,-62.2,2023-08-13,low,9.0\n"  # ignored due to low confidence and low frp
        "-15.3,-62.3,2023-08-14,nominal,8.0\n"  # ignored due to low frp (< 10 MW)
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_csv
    mock_get.return_value = mock_resp

    bbox = (-65.0, -20.0, -60.0, -10.0)
    dates = query_firms_fire_dates(
        firms_key="dummy_key",
        bbox=bbox,
        start_date_str="20230810",
        end_date_str="20230814",
        source="VIIRS_SNPP_NRT",
        min_confidence="nominal",
        min_frp_mw=10.0,
    )

    # Should only return 20230812 because 20230813 and 20230814 are filtered out
    assert dates == ["20230812"]
    # Check that requests.get was called with correct parameters
    # The date range 20230810 to 20230814 is 5 days, so it should be requested in a single chunk
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert "dummy_key" in called_url
    assert "VIIRS_SNPP_NRT" in called_url
    assert "-65.0,-20.0,-60.0,-10.0" in called_url
    assert "/5/2023-08-10" in called_url


@patch("nina_thermal.datagen.acquisition.firms_api.requests.get")
def test_query_firms_fire_dates_chunking(mock_get):
    # Mock empty responses
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "latitude,longitude,acq_date,confidence,frp\n"
    mock_get.return_value = mock_resp

    # Request range of 12 days (Aug 1 to Aug 12)
    # With chunk size = 5, this should result in 3 requests (5, 5, 2 days)
    bbox = (-65.0, -20.0, -60.0, -10.0)
    query_firms_fire_dates(
        firms_key="dummy_key",
        bbox=bbox,
        start_date_str="20230801",
        end_date_str="20230812",
        source="VIIRS_SNPP_NRT",
    )

    assert mock_get.call_count == 3
    urls = [call[0][0] for call in mock_get.call_args_list]
    assert "/5/2023-08-01" in urls[0]
    assert "/5/2023-08-06" in urls[1]
    assert "/2/2023-08-11" in urls[2]

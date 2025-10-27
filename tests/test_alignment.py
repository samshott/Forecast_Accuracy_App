import pandas as pd

from src.matcher import align_daily_pairs


def test_align_daily_pairs_basic():
    forecasts = pd.DataFrame(
        {
            "var": ["TMAX", "TMAX", "PRCP"],
            "valid_time": [
                "2024-05-01T18:00:00Z",
                "2024-05-02T18:00:00Z",
                "2024-05-01T12:00:00Z",
            ],
            "issue_time": ["2024-04-30T12:00:00Z"] * 3,
            "lead_hours": [30, 54, 24],
            "value": [20.0, 22.0, 5.0],
        }
    )
    obs = pd.DataFrame(
        {
            "station_id": ["STA001", "STA001", "STA001"],
            "date": ["2024-05-01T00:00:00", "2024-05-02T00:00:00", "2024-05-01T00:00:00"],
            "var": ["TMAX", "TMAX", "PRCP"],
            "value": [18.0, 21.0, 4.5],
            "qc_flag": ["", "", ""],
        }
    )

    pairs = align_daily_pairs(forecasts, obs)

    assert len(pairs) == 3
    assert set(pairs["var"]) == {"TMAX", "PRCP"}
    assert pairs["station_id"].nunique() == 1
    prcp_pair = pairs[pairs["var"] == "PRCP"].iloc[0]
    assert prcp_pair["lead_hours"] == 24
    assert prcp_pair["f_value"] == 5.0
    assert prcp_pair["o_value"] == 4.5


def test_align_daily_pairs_empty():
    forecasts = pd.DataFrame(columns=["var", "valid_time", "issue_time", "lead_hours", "value"])
    obs = pd.DataFrame(columns=["station_id", "date", "var", "value", "qc_flag"])
    pairs = align_daily_pairs(forecasts, obs)
    assert pairs.empty

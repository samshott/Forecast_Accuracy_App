import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.metrics import bias, mae, rmse, score_pairs


def test_metric_functions():
    errors = np.array([1.0, -1.0, 2.0, -2.0])
    assert mae(errors) == pytest.approx(1.5)
    assert rmse(errors) == pytest.approx(np.sqrt(2.5))
    assert bias(errors) == pytest.approx(0.0)


def test_score_pairs_basic():
    pairs_df = pd.DataFrame(
        {
            "station_id": ["STA001", "STA001", "STA001"],
            "var": ["TMAX", "TMAX", "TMIN"],
            "valid_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-01"]),
            "lead_hours": [24, 48, 24],
            "issue_time": pd.to_datetime(["2024-04-30T12:00:00Z"] * 3),
            "f_value": [20.0, 22.0, 10.0],
            "o_value": [18.0, 21.0, 12.0],
        }
    )

    scores = score_pairs(pairs_df)

    assert len(scores) == 3
    tmax_day1 = scores[(scores["var"] == "TMAX") & (scores["lead_hours"] == 24)].iloc[0]
    assert_allclose(tmax_day1["mae"], 2.0)
    assert_allclose(tmax_day1["rmse"], 2.0)
    assert_allclose(tmax_day1["bias"], 2.0)
    assert tmax_day1["n"] == 1

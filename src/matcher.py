from __future__ import annotations

import pandas as pd


def _ensure_utc(series: pd.Series) -> pd.Series:
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC")
    return series.dt.tz_convert("UTC")


def align_daily_pairs(forecasts_df: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    if forecasts_df.empty or obs_df.empty:
        columns = [
            "station_id",
            "var",
            "valid_date",
            "lead_hours",
            "issue_time",
            "f_value",
            "o_value",
        ]
        return pd.DataFrame(columns=columns)

    forecasts = forecasts_df.copy()
    forecasts["valid_time"] = _ensure_utc(pd.to_datetime(forecasts["valid_time"]))
    forecasts["issue_time"] = _ensure_utc(pd.to_datetime(forecasts["issue_time"]))
    forecasts["valid_date"] = forecasts["valid_time"].dt.floor("D")

    obs = obs_df.copy()
    obs["date"] = _ensure_utc(pd.to_datetime(obs["date"]))
    obs["valid_date"] = obs["date"].dt.floor("D")
    obs.rename(columns={"value": "o_value"}, inplace=True)

    merged = forecasts.merge(
        obs[["station_id", "var", "valid_date", "o_value"]],
        on=["var", "valid_date"],
        how="inner",
    )
    merged.rename(columns={"value": "f_value"}, inplace=True)
    merged = merged[
        [
            "station_id",
            "var",
            "valid_date",
            "lead_hours",
            "issue_time",
            "f_value",
            "o_value",
        ]
    ]
    return merged.sort_values(["var", "valid_date", "lead_hours"]).reset_index(drop=True)


__all__ = ["align_daily_pairs"]

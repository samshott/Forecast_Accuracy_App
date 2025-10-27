from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd

from .config import Settings, load_settings
from .http import build_session


DATASET_ID = "GHCND"
DEFAULT_LIMIT = 1000

VALUE_CONVERSIONS = {
    "TMAX": lambda v: v / 10 if v is not None else None,
    "TMIN": lambda v: v / 10 if v is not None else None,
    "PRCP": lambda v: v / 10 if v is not None else None,
}


def _to_datestr(value: date | datetime | str) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def get_ncei_daily_obs(
    station_id: str,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    variables: Optional[Iterable[str]] = None,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    if settings is None:
        settings = load_settings()
    if not settings.api.ncei_token:
        raise EnvironmentError("NCEI_TOKEN is required to query NCEI APIs")
    vars_to_fetch = tuple(variables or ("TMAX", "TMIN", "PRCP"))
    params = {
        "datasetid": DATASET_ID,
        "stationid": station_id,
        "startdate": _to_datestr(start),
        "enddate": _to_datestr(end),
        "units": "metric",
        "limit": DEFAULT_LIMIT,
        "offset": 1,
    }
    if vars_to_fetch:
        params["datatypeid"] = vars_to_fetch

    session = build_session(settings.api.ncei_retries)
    headers = {"token": settings.api.ncei_token}
    results = []
    url = f"{settings.api.ncei_base_url}/data"
    while True:
        response = session.get(url, headers=headers, params=params, timeout=settings.api.ncei_timeout)
        response.raise_for_status()
        payload = response.json()
        results.extend(payload.get("results", []))
        metadata = payload.get("metadata", {}).get("resultset", {})
        total = metadata.get("count", len(results))
        params["offset"] += params["limit"]
        if params["offset"] > total:
            break

    if not results:
        columns = ["station_id", "date", "var", "value", "qc_flag", "network"]
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.rename(columns={"datatype": "var"}, inplace=True)
    df["station_id"] = df["station"].fillna(station_id)
    df["network"] = df["station_id"].str.split(":").str[0]
    df["qc_flag"] = df["attributes"].fillna("")
    df["value"] = df.apply(lambda row: VALUE_CONVERSIONS.get(row["var"], lambda v: v)(row["value"]), axis=1)
    return df[["station_id", "network", "date", "var", "value", "qc_flag"]]


__all__ = ["get_ncei_daily_obs"]

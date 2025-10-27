from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import numpy as np
from pyproj import Geod

from .config import Settings, load_settings
from .http import build_session


DEFAULT_VARIABLES = ("TMAX", "TMIN", "PRCP")
GEOD = Geod(ellps="WGS84")


def select_nearest_station(
    lat: float,
    lon: float,
    *,
    network: str = "GHCND",
    radius_km: float = 50.0,
    variables: Optional[Iterable[str]] = None,
    settings: Optional[Settings] = None,
) -> pd.Series:
    if settings is None:
        settings = load_settings()
    if not settings.api.ncei_token:
        raise EnvironmentError("NCEI_TOKEN is required to query NCEI APIs")

    session = build_session(settings.api.ncei_retries)
    headers = {"token": settings.api.ncei_token}
    params = {
        "datasetid": "GHCND",
        "latitude": lat,
        "longitude": lon,
        "radius": radius_km,
        "limit": 1000,
    }
    params_items = list(params.items())
    for var in variables or DEFAULT_VARIABLES:
        params_items.append(("datatypeid", var))

    response = session.get(
        f"{settings.api.ncei_base_url}/stations",
        headers=headers,
        params=params_items,
        timeout=settings.api.ncei_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        raise LookupError("No stations found within radius")

    df = pd.DataFrame(results)
    df = df[df["id"].str.startswith(f"{network}:")]
    if df.empty:
        raise LookupError(f"No stations found in network {network}")

    lons = np.full(len(df), lon)
    lats = np.full(len(df), lat)
    target_lons = df["longitude"].astype(float).to_numpy()
    target_lats = df["latitude"].astype(float).to_numpy()
    _, _, distances = GEOD.inv(lons, lats, target_lons, target_lats)
    df["distance_km"] = distances / 1000
    df.sort_values("distance_km", inplace=True)
    best = df.iloc[0]
    return best


__all__ = ["select_nearest_station"]

from __future__ import annotations

from typing import Iterable, Optional
import math

import pandas as pd
import numpy as np
from pyproj import Geod

from .config import Settings, load_settings
from .http import build_session


DEFAULT_VARIABLES = ("TMAX", "TMIN", "PRCP")
GEOD = Geod(ellps="WGS84")


def _compute_extent(lat: float, lon: float, radius_km: float) -> str:
    radius_deg_lat = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    radius_deg_lon = radius_km / (111.0 * cos_lat) if cos_lat else 360.0
    min_lat = max(-90.0, lat - radius_deg_lat)
    max_lat = min(90.0, lat + radius_deg_lat)
    min_lon = lon - radius_deg_lon
    max_lon = lon + radius_deg_lon
    if min_lon < -180.0:
        min_lon += 360.0
    if max_lon > 180.0:
        max_lon -= 360.0
    if min_lon > max_lon:
        min_lon, max_lon = -180.0, 180.0
    return f"{min_lat},{min_lon},{max_lat},{max_lon}"


def select_nearest_station(
    lat: float,
    lon: float,
    *,
    network: str = "GHCND",
    radius_km: float = 50.0,
    variables: Optional[Iterable[str]] = None,
    settings: Optional[Settings] = None,
    return_all: bool = False,
) -> pd.Series:
    if settings is None:
        settings = load_settings()
    if not settings.api.ncei_token:
        raise EnvironmentError("NCEI_TOKEN is required to query NCEI APIs")

    session = build_session(settings.api.ncei_retries)
    headers = {"token": settings.api.ncei_token}
    extent = _compute_extent(lat, lon, radius_km)
    params = {
        "datasetid": "GHCND",
        "extent": extent,
        "limit": 1000,
        "sortfield": "datacoverage",
        "sortorder": "desc",
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
        if radius_km < 300:
            return select_nearest_station(
                lat,
                lon,
                network=network,
                radius_km=radius_km * 2,
                variables=variables,
                settings=settings,
                return_all=return_all,
            )
        raise LookupError("No stations found within radius")

    df = pd.DataFrame(results)
    df = df[df["id"].str.startswith(f"{network}:")]
    if df.empty:
        if radius_km < 300:
            return select_nearest_station(
                lat,
                lon,
                network=network,
                radius_km=radius_km * 2,
                variables=variables,
                settings=settings,
                return_all=return_all,
            )
        raise LookupError(f"No stations found in network {network}")

    lons = np.full(len(df), lon, dtype=float)
    lats = np.full(len(df), lat, dtype=float)
    target_lons = df["longitude"].astype(float).to_numpy()
    target_lats = df["latitude"].astype(float).to_numpy()
    _, _, distances = GEOD.inv(lons, lats, target_lons, target_lats)
    df["distance_km"] = distances / 1000
    df.sort_values("distance_km", inplace=True)
    df["station_id"] = df["id"]
    if return_all:
        return df.reset_index(drop=True)
    return df.iloc[0]


__all__ = ["select_nearest_station"]

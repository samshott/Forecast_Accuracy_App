from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dateutil import parser as date_parser

from .config import Settings, load_settings
from .http import build_session


VARIABLE_MAP: Dict[str, str] = {
    "TMAX": "maxTemperature",
    "TMIN": "minTemperature",
    "PRCP": "qpf",
}


def _cache_file(cache_dir: Path, lat: float, lon: float) -> Path:
    date_key = datetime.now(timezone.utc).strftime("%Y%m%d")
    sanitized = f"{lat:.4f}_{lon:.4f}".replace(".", "d").replace("-", "m")
    return cache_dir / "nws" / date_key / f"gridpoint_{sanitized}.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_cache(path: Path) -> Optional[Dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _write_cache(path: Path, payload: Dict) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _parse_valid_time(valid_time: str) -> datetime:
    ts, *_ = valid_time.split("/")
    return date_parser.isoparse(ts)


def _extract_values(feature: Dict, issue_time: datetime, variables: Iterable[str]) -> List[Dict]:
    rows: List[Dict] = []
    for var in variables:
        grid_key = VARIABLE_MAP.get(var)
        if not grid_key:
            continue
        prop = feature["properties"].get(grid_key, {})
        for entry in prop.get("values", []):
            valid_time = _parse_valid_time(entry["validTime"])
            lead_hours = int(round((valid_time - issue_time).total_seconds() / 3600))
            value = entry["value"]
            if value is None:
                continue
            rows.append(
                {
                    "var": var,
                    "grid_key": grid_key,
                    "valid_time": valid_time,
                    "issue_time": issue_time,
                    "lead_hours": lead_hours,
                    "value": value,
                }
            )
    return rows


def get_current_gridpoint_forecast(
    lat: float,
    lon: float,
    *,
    variables: Optional[Iterable[str]] = None,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    if settings is None:
        settings = load_settings()
    vars_to_fetch = tuple(variables or settings.app.default_variables)
    cache_path = _cache_file(settings.data.cache_dir, lat, lon)
    payload = _load_cache(cache_path)
    session = build_session(settings.api.nws_retries)
    headers = {
        "User-Agent": settings.api.nws_user_agent,
        "Accept": "application/geo+json",
    }

    if payload is None:
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        point_resp = session.get(points_url, headers=headers, timeout=settings.api.nws_timeout)
        point_resp.raise_for_status()
        point_data = point_resp.json()
        grid_url = point_data["properties"]["forecastGridData"]
        grid_resp = session.get(grid_url, headers=headers, timeout=settings.api.nws_timeout)
        grid_resp.raise_for_status()
        payload = grid_resp.json()
        payload["metadata"] = {
            "grid_id": point_data["properties"]["gridId"],
            "grid_x": point_data["properties"]["gridX"],
            "grid_y": point_data["properties"]["gridY"],
            "cwa": point_data["properties"]["cwa"],
            "radar_station": point_data["properties"].get("radarStation"),
        }
        _write_cache(cache_path, payload)

    issue_time = date_parser.isoparse(payload["properties"]["updateTime"]).astimezone(timezone.utc)
    rows = _extract_values(payload, issue_time, vars_to_fetch)
    if not rows:
        return pd.DataFrame(columns=["var", "valid_time", "issue_time", "lead_hours", "value"])
    df = pd.DataFrame(rows)
    meta = payload.get("metadata", {})
    df["grid_id"] = meta.get("grid_id")
    df["grid_x"] = meta.get("grid_x")
    df["grid_y"] = meta.get("grid_y")
    df["cwa"] = meta.get("cwa")
    df["latitude"] = lat
    df["longitude"] = lon
    df["source_cycle"] = issue_time.strftime("%Y%m%d%H")
    return df.sort_values(["var", "valid_time"]).reset_index(drop=True)


__all__ = ["get_current_gridpoint_forecast"]

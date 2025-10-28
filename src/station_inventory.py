from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import Settings, load_settings
from .http import build_session

DATASET_ID = "GHCND"
DEFAULT_DATATYPES = ("TMAX", "TMIN", "PRCP")
PAGE_LIMIT = 1000


@dataclass
class StationRecord:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float | None
    datacoverage: float | None
    mindate: str | None
    maxdate: str | None


def _fetch_station_metadata(
    *,
    datatype_id: str,
    settings: Settings,
) -> pd.DataFrame:
    print(f"Fetching stations for {datatype_id}…", flush=True)
    session = build_session(settings.api.ncei_retries)
    headers = {"token": settings.api.ncei_token}
    params = {
        "datasetid": DATASET_ID,
        "datatypeid": datatype_id,
        "limit": PAGE_LIMIT,
        "offset": 1,
    }
    url = f"{settings.api.ncei_base_url}/stations"
    frames: List[pd.DataFrame] = []

    while True:
        response = session.get(url, headers=headers, params=params, timeout=settings.api.ncei_timeout)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break
        frames.append(pd.DataFrame(results))
        meta = payload.get("metadata", {}).get("resultset", {})
        total = meta.get("count", len(results))
        fetched = min(params["offset"] + params["limit"] - 1, total)
        print(f"  Retrieved {fetched} / {total} records", flush=True)
        params["offset"] += params["limit"]
        if params["offset"] > total:
            break

    if not frames:
        print(f"  No stations returned for {datatype_id}.", flush=True)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"id": "station_id"})
    print(f"Completed fetch for {datatype_id}: {len(df)} stations", flush=True)
    return df


def inventory_stations(
    datatypes: Iterable[str],
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    inventory_frames: List[pd.DataFrame] = []
    availability: Dict[str, set[str]] = {}

    for dtype in datatypes:
        df_dtype = _fetch_station_metadata(datatype_id=dtype, settings=settings)
        if df_dtype.empty:
            continue
        availability.setdefault(dtype, set()).update(df_dtype["station_id"].unique())
        inventory_frames.append(df_dtype)

    if not inventory_frames:
        return pd.DataFrame(columns=["station_id", "name", "latitude", "longitude"])

    combined = pd.concat(inventory_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["station_id"])

    combined["mindate"] = pd.to_datetime(combined.get("mindate"))
    combined["maxdate"] = pd.to_datetime(combined.get("maxdate"))

    combined["has_tmax"] = combined["station_id"].isin(availability.get("TMAX", set()))
    combined["has_tmin"] = combined["station_id"].isin(availability.get("TMIN", set()))
    combined["has_prcp"] = combined["station_id"].isin(availability.get("PRCP", set()))

    return combined[
        [
            "station_id",
            "name",
            "latitude",
            "longitude",
            "elevation",
            "datacoverage",
            "mindate",
            "maxdate",
            "has_tmax",
            "has_tmin",
            "has_prcp",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory NCEI stations with required datatypes.")
    parser.add_argument(
        "--datatypes",
        nargs="+",
        default=list(DEFAULT_DATATYPES),
        help="Datatype IDs to require (default: TMAX TMIN PRCP).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stations/station_inventory.parquet"),
        help="Output Parquet path (default: data/stations/station_inventory.parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    try:
        inventory = inventory_stations(args.datatypes, settings=settings)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Station inventory failed: {exc}")

    if inventory.empty:
        print("No stations matched the requested datatypes.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(args.output, index=False)
    summary = (
        inventory[["has_tmax", "has_tmin", "has_prcp"]]
        .sum()
        .rename({"has_tmax": "stations_with_tmax", "has_tmin": "stations_with_tmin", "has_prcp": "stations_with_prcp"})
    )
    print("Inventory saved to", args.output)
    print(summary.to_string())
    complete = inventory[inventory[["has_tmax", "has_tmin", "has_prcp"]].all(axis=1)]
    print(f"Stations with full variable availability: {len(complete)}")


if __name__ == "__main__":
    main()

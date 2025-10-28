from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

from .config import Settings, load_settings
from .ingest_forecast_archive import ingest_ndfd_archive

DEFAULT_BOUNDS = (24.0, -125.0, 50.0, -66.0)
DEFAULT_DAYS = 14
DEFAULT_STEP_HOURS = 6
TARGET_NETWORKS = {"USW", "USC"}
TARGET_VARS = ("TMAX", "TMIN", "PRCP")


def load_inventory(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("latitude", "longitude", "datacoverage"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_stations(
    df: pd.DataFrame,
    *,
    bounds: tuple[float, float, float, float] | None,
    min_datacoverage: float,
) -> pd.DataFrame:
    df = df.copy()
    df["network"] = df["station_id"].str.split(":", n=1).str[-1].str[:3]
    has_all = df[["has_tmax", "has_tmin", "has_prcp"]].all(axis=1)
    meets = has_all & df["network"].isin(TARGET_NETWORKS)
    meets &= df["datacoverage"].fillna(0) >= min_datacoverage

    if bounds is not None:
        min_lat, min_lon, max_lat, max_lon = bounds
        meets &= df["latitude"].between(min_lat, max_lat)
        meets &= df["longitude"].between(min_lon, max_lon)

    return df[meets].reset_index(drop=True)


def issue_times(start: datetime, end: datetime, step_hours: int) -> Sequence[datetime]:
    return pd.date_range(start, end, freq=f"{step_hours}H").to_pydatetime()


def ensure_parquet(
    issue_time: datetime,
    variables: Iterable[str],
    *,
    settings: Settings,
    bounds: tuple[float, float, float, float],
) -> None:
    missing_vars = []
    forecast_dir = settings.data.forecast_dir
    forecast_dir.mkdir(parents=True, exist_ok=True)
    for var in variables:
        var_dir = forecast_dir / var
        var_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = var_dir / f"{issue_time:%Y%m%d%H}.parquet"
        if not parquet_path.exists():
            missing_vars.append(var)
    if not missing_vars:
        return

    ingest_ndfd_archive(
        issue_time,
        variables=missing_vars,
        bbox=bounds,
        settings=settings,
        write_parquet=True,
        overwrite_download=False,
    )


def summarize_output(settings: Settings, variables: Iterable[str]) -> None:
    total_bytes = 0
    total_rows = 0
    for var in variables:
        var_dir = settings.data.forecast_dir / var
        if not var_dir.exists():
            continue
        for parquet_path in var_dir.glob("*.parquet"):
            total_bytes += parquet_path.stat().st_size
            total_rows += len(pd.read_parquet(parquet_path))
    print(f"Total parquet size: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Total rows written: {total_rows}")


def parse_datetime(value: str) -> datetime:
    dt = pd.to_datetime(value)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill a sample of NDFD forecasts for a station subset.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/stations/station_inventory.parquet"),
        help="Path to the station inventory parquet file.",
    )
    parser.add_argument(
        "--min-datacoverage",
        type=float,
        default=0.9,
        help="Minimum datacoverage required for stations (default: 0.9).",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Bounding box for stations (default: lower 48).",
    )
    parser.add_argument(
        "--no-bounds",
        action="store_true",
        help="Disable bounding-box filtering and include all stations.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (UTC) for backfill range (default: today minus N days).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of days to look back (default: 14).",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=DEFAULT_STEP_HOURS,
        help="Cycle spacing in hours (default: 6).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(TARGET_VARS),
        help="Variables to ingest (default: TMAX TMIN PRCP).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_path = args.inventory
    if not inventory_path.exists():
        raise SystemExit(f"Inventory file not found: {inventory_path}")

    settings = load_settings()
    inventory = load_inventory(inventory_path)

    bounds = None if args.no_bounds else (tuple(args.bounds) if args.bounds else DEFAULT_BOUNDS)
    stations = filter_stations(
        inventory,
        bounds=bounds,
        min_datacoverage=args.min_datacoverage,
    )

    if stations.empty:
        raise SystemExit("No stations matched the criteria; adjust bounds or datacoverage.")

    print(f"Stations matching filter: {len(stations)}")
    print(stations["network"].value_counts().to_string(name="count"))

    end_issue = parse_datetime(args.start_date) if args.start_date else datetime.now(timezone.utc)
    start_issue = end_issue - timedelta(days=args.days)
    issues = issue_times(start_issue, end_issue, args.step_hours)

    print(f"Processing {len(issues)} issue times from {start_issue} to {end_issue}...")

    progress_bounds = bounds or DEFAULT_BOUNDS
    for issue in tqdm(issues, desc="Ingesting issues", unit="cycle"):
        ensure_parquet(issue, args.variables, settings=settings, bounds=progress_bounds)

    summarize_output(settings, args.variables)


if __name__ == "__main__":
    main()

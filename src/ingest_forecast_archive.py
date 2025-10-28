from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import ForecastArchiveConfig, Settings, load_settings
from .http import build_session
logger = logging.getLogger(__name__)

VARIABLE_DATA_VARS: Dict[str, str] = {
    "TMAX": "tmax",
    "TMIN": "tmin",
    "PRCP": "unknown",
}

VARIABLE_BACKEND_KWARGS: Dict[str, Dict] = {
    "TMAX": {"filter_by_keys": {"shortName": "tmax"}},
    "TMIN": {"filter_by_keys": {"shortName": "tmin"}},
    "PRCP": {},
}


@dataclass(frozen=True)
class NDFDIngestResult:
    variable: str
    issue_time: datetime
    parquet_path: Optional[Path]
    source_path: Path
    row_count: int


def _build_remote_url(
    issue_time: datetime,
    variable: str,
    config: ForecastArchiveConfig,
) -> str:
    product_files = config.product_files.get(variable, ())
    if not product_files:
        raise ValueError(f"No product file configured for variable {variable}")
    product_file = product_files[0]
    base = config.base_url.rstrip("/")
    return f"{base}/{product_file.lstrip('/')}"


def _download_tgftp_file(
    variable: str,
    rel_path: str,
    *,
    session,
    headers: Dict[str, str],
    config: ForecastArchiveConfig,
    timeout: int,
) -> Path:
    url = f"{config.base_url.rstrip('/')}/{rel_path.lstrip('/')}"
    response = session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    tmp_dir = config.archive_dir / variable / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    filename = rel_path.replace("/", "_")
    tmp_path = tmp_dir / filename
    with tmp_path.open("wb") as fh:
        fh.write(response.content)
    return tmp_path


def _open_dataset(
    grib_path: Path,
    variable: str,
) -> xr.Dataset:
    backend_kwargs = VARIABLE_BACKEND_KWARGS.get(variable, {})
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)
    except ValueError as exc:
        raise RuntimeError(f"Unable to open GRIB file {grib_path}: {exc}") from exc
    return ds


def _extract_issue_time(ds: xr.Dataset, default: datetime) -> datetime:
    if "time" in ds.coords:
        time_values = ds["time"].values
        if np.isscalar(time_values):
            issue = pd.to_datetime(time_values)
        else:
            issue = pd.to_datetime(time_values[0])
        return _ensure_utc(issue.to_pydatetime())
    return default


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_dataframe(
    ds: xr.Dataset,
    variable: str,
    issue_time: datetime,
) -> pd.DataFrame:
    data_var_name = VARIABLE_DATA_VARS.get(variable)
    if data_var_name and data_var_name in ds:
        da = ds[data_var_name]
    else:
        if not ds.data_vars:
            raise ValueError(f"No data variables found in dataset for {variable}")
        da = next(iter(ds.data_vars.values()))
        data_var_name = da.name
    df = da.to_dataframe(name="value").reset_index()
    issue_time = _extract_issue_time(ds, issue_time)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df["time"] = issue_time
    if "step" in df.columns:
        df["step"] = pd.to_timedelta(df["step"])
    else:
        df["step"] = pd.to_timedelta(df["time"] - issue_time)
    df["valid_time"] = df["time"] + df["step"]
    df["lead_hours"] = df["step"].dt.total_seconds() / 3600.0
    df["issue_time"] = issue_time
    df["var"] = variable

    if "latitude" not in df.columns and "latitude" in ds:
        lat_df = ds["latitude"].to_dataframe(name="latitude").reset_index()
        df = df.merge(lat_df, on=[col for col in lat_df.columns if col != "latitude"], how="left")
    if "longitude" not in df.columns and "longitude" in ds:
        lon_df = ds["longitude"].to_dataframe(name="longitude").reset_index()
        df = df.merge(lon_df, on=[col for col in lon_df.columns if col != "longitude"], how="left")

    if "x" in df.columns:
        df.rename(columns={"x": "grid_x"}, inplace=True)
    if "y" in df.columns:
        df.rename(columns={"y": "grid_y"}, inplace=True)
    df["grid_x"] = df.get("grid_x")
    df["grid_y"] = df.get("grid_y")
    df["source"] = str(ds.attrs.get("source", "ndfd"))
    df["data_var"] = data_var_name

    df = df[
        [
            "var",
            "issue_time",
            "valid_time",
            "lead_hours",
            "grid_x",
            "grid_y",
            "latitude",
            "longitude",
            "value",
            "data_var",
            "source",
        ]
    ]
    df.sort_values(["var", "valid_time", "grid_y", "grid_x"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _filter_bbox(df: pd.DataFrame, bbox: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if bbox is None:
        return df
    min_lat, min_lon, max_lat, max_lon = bbox
    mask = pd.Series(True, index=df.index)
    if "latitude" in df:
        mask &= df["latitude"].between(min_lat, max_lat)
    if "longitude" in df:
        mask &= df["longitude"].between(min_lon, max_lon)
    return df[mask]


def ingest_ndfd_archive(
    issue_time: datetime,
    *,
    variables: Optional[Iterable[str]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    settings: Optional[Settings] = None,
    write_parquet: bool = True,
    overwrite_download: bool = False,
) -> Dict[str, NDFDIngestResult]:
    settings = settings or load_settings()
    vars_to_process = tuple(variables or settings.app.default_variables)
    session = build_session(settings.api.nws_retries)
    headers = {"User-Agent": settings.api.nws_user_agent}
    config = settings.forecast_archive
    results: Dict[str, NDFDIngestResult] = {}

    for variable in vars_to_process:
        rel_paths = config.product_files.get(variable, ())
        if not rel_paths:
            logger.warning("No product file configured for %s", variable)
            continue

        combined_frames: List[pd.DataFrame] = []
        cached_paths: List[Path] = []
        issue_time_for_var: Optional[datetime] = None

        for rel_path in rel_paths:
            try:
                tmp_path = _download_tgftp_file(
                    variable,
                    rel_path,
                    session=session,
                    headers=headers,
                    config=config,
                    timeout=settings.api.nws_timeout,
                )
            except Exception as exc:
                logger.error("Failed to fetch %s (%s): %s", variable, rel_path, exc)
                continue

            ds = _open_dataset(tmp_path, variable)
            issue_from_ds = _extract_issue_time(ds, issue_time)
            issue_time_for_var = issue_from_ds if issue_time_for_var is None else issue_time_for_var

            issue_key = issue_time_for_var.strftime("%Y%m%d%H")
            dest_dir = config.archive_dir / variable / issue_key
            dest_dir.mkdir(parents=True, exist_ok=True)

            df_part = _to_dataframe(ds, variable, issue_time_for_var)
            ds.close()
            dest_path = dest_dir / tmp_path.name
            tmp_path.replace(dest_path)
            cached_paths.append(dest_path)
            combined_frames.append(df_part)

        if not combined_frames or issue_time_for_var is None:
            continue

        df = pd.concat(combined_frames, ignore_index=True)
        if bbox:
            df = _filter_bbox(df, bbox)

        parquet_path: Optional[Path] = None
        if write_parquet and not df.empty:
            parquet_dir = settings.data.forecast_dir / variable
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / f"{issue_time_for_var:%Y%m%d%H}.parquet"
            df.to_parquet(parquet_path, index=False)

        results[variable] = NDFDIngestResult(
            variable=variable,
            issue_time=_ensure_utc(issue_time_for_var),
            parquet_path=parquet_path,
            source_path=cached_paths[0] if cached_paths else Path(),
            row_count=len(df),
        )
    return results


def _issue_sequence(
    start: datetime,
    end: datetime,
    step_hours: int,
) -> Sequence[datetime]:
    issues: List[datetime] = []
    current = start
    while current <= end:
        issues.append(current)
        current += timedelta(hours=step_hours)
    return issues


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NDFD archive forecasts into Parquet.")
    parser.add_argument(
        "--issue",
        help="Forecast issue time in UTC (e.g., 2024-05-01T00).",
    )
    parser.add_argument(
        "--start",
        help="Start issue time (inclusive) in UTC for backfill range.",
    )
    parser.add_argument(
        "--end",
        help="End issue time (inclusive) in UTC for backfill range.",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=6,
        help="Spacing in hours between issue times when using --start/--end (default: 6).",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Variables to ingest (default: configured default variables).",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Optional bounding box to subset the grid.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude for automatic bounding box (requires --lon).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude for automatic bounding box (requires --lat).",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=75.0,
        help="Radius in km used when deriving bbox from lat/lon (default: 75 km).",
    )
    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download of GRIB files even if cached locally.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing Parquet output; useful for dry runs.",
    )
    return parser.parse_args()


def _parse_issue_time(value: str) -> datetime:
    issue = pd.to_datetime(value)
    if issue.tzinfo is None:
        issue = issue.tz_localize("UTC")
    return issue.to_pydatetime()


def _bbox_from_center(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    radius_deg_lat = radius_km / 111.0
    cos_lat = np.cos(np.radians(lat))
    radius_deg_lon = radius_km / (111.0 * cos_lat) if cos_lat else 360.0
    min_lat = max(-90.0, lat - radius_deg_lat)
    max_lat = min(90.0, lat + radius_deg_lat)
    min_lon = lon - radius_deg_lon
    max_lon = lon + radius_deg_lon
    if min_lon < -180.0:
        min_lon += 360.0
    if max_lon > 180.0:
        max_lon -= 360.0
    return (min_lat, min_lon, max_lat, max_lon)


def _determine_bbox(args: argparse.Namespace) -> Optional[Tuple[float, float, float, float]]:
    if args.bbox:
        return tuple(args.bbox)  # type: ignore[return-value]
    if args.lat is not None and args.lon is not None:
        return _bbox_from_center(args.lat, args.lon, args.radius_km)
    return None


def _resolve_issue_times(args: argparse.Namespace) -> Sequence[datetime]:
    if args.issue:
        return [_parse_issue_time(args.issue)]
    if args.start and args.end:
        start = _parse_issue_time(args.start)
        end = _parse_issue_time(args.end)
        if end < start:
            raise ValueError("--end must be on or after --start")
        return _issue_sequence(start, end, args.step_hours)
    raise ValueError("Provide either --issue or both --start and --end")


def main() -> None:
    args = _parse_cli_args()
    try:
        issue_times = _resolve_issue_times(args)
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc

    bbox = _determine_bbox(args)
    variables = args.vars
    for issue_time in issue_times:
        logger.info("Processing issue %s", issue_time.isoformat())
        results = ingest_ndfd_archive(
            issue_time,
            variables=variables,
            bbox=bbox,
            write_parquet=not args.no_parquet,
            overwrite_download=args.overwrite_download,
        )
        for var, result in results.items():
            msg = f"{var}: rows={result.row_count}, source={result.source_path}"
            if result.parquet_path:
                msg += f", parquet={result.parquet_path}"
            logger.info(msg)


if __name__ == "__main__":
    main()


__all__ = ["ingest_ndfd_archive", "NDFDIngestResult"]

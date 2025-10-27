from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import ForecastArchiveConfig, Settings, load_settings
from .http import build_session

logger = logging.getLogger(__name__)

VARIABLE_DATA_VARS: Dict[str, str] = {
    "TMAX": "tmax",
    "TMIN": "tmin",
    "PRCP": "tp",
}

VARIABLE_BACKEND_KWARGS: Dict[str, Dict] = {
    "TMAX": {"filter_by_keys": {"typeOfLevel": "surface", "shortName": "tmax"}},
    "TMIN": {"filter_by_keys": {"typeOfLevel": "surface", "shortName": "tmin"}},
    "PRCP": {"filter_by_keys": {"typeOfLevel": "surface", "shortName": "tp"}},
}


@dataclass(frozen=True)
class NDFDIngestResult:
    variable: str
    issue_time: datetime
    parquet_path: Optional[Path]
    source_path: Path
    row_count: int


def _issue_to_parts(issue_time: datetime) -> Tuple[str, str]:
    issue_time_utc = issue_time.astimezone(timezone.utc)
    return issue_time_utc.strftime("%Y%m%d"), issue_time_utc.strftime("%H")


def _build_remote_url(
    issue_time: datetime,
    variable: str,
    config: ForecastArchiveConfig,
) -> str:
    date_str, hour_str = _issue_to_parts(issue_time)
    product_file = config.product_files.get(variable)
    if not product_file:
        raise ValueError(f"No product file configured for variable {variable}")
    base = config.base_url.rstrip("/")
    sector = config.sector
    grid = config.grid
    return f"{base}/ndfd.{date_str}/{hour_str}/{sector}/{grid}.{product_file}"


def _local_grib_path(
    issue_time: datetime,
    variable: str,
    config: ForecastArchiveConfig,
) -> Path:
    date_str, hour_str = _issue_to_parts(issue_time)
    filename = f"{variable.lower()}_{date_str}{hour_str}.grib2"
    target_dir = config.archive_dir / variable / date_str
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def download_ndfd_grib(
    issue_time: datetime,
    variable: str,
    *,
    settings: Optional[Settings] = None,
    overwrite: bool = False,
) -> Path:
    settings = settings or load_settings()
    config = settings.forecast_archive
    local_path = _local_grib_path(issue_time, variable, config)
    if local_path.exists() and not overwrite:
        logger.debug("Using cached GRIB for %s at %s", variable, local_path)
        return local_path

    url = _build_remote_url(issue_time, variable, config)
    session = build_session(settings.api.nws_retries)
    headers = {"User-Agent": settings.api.nws_user_agent}
    logger.info("Downloading %s forecast from %s", variable, url)
    response = session.get(url, headers=headers, timeout=settings.api.nws_timeout)
    response.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as fh:
        fh.write(response.content)
    return local_path


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
    results: Dict[str, NDFDIngestResult] = {}
    for variable in vars_to_process:
        try:
            grib_path = download_ndfd_grib(
                issue_time,
                variable,
                settings=settings,
                overwrite=overwrite_download,
            )
        except Exception as exc:
            logger.error("Failed to fetch %s archive: %s", variable, exc)
            continue
        ds = _open_dataset(grib_path, variable)
        df = _to_dataframe(ds, variable, issue_time)
        ds.close()
        if bbox:
            df = _filter_bbox(df, bbox)
        parquet_path: Optional[Path] = None
        if write_parquet and not df.empty:
            parquet_dir = settings.data.forecast_dir / variable
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / f"{issue_time:%Y%m%d%H}.parquet"
            df.to_parquet(parquet_path, index=False)
        results[variable] = NDFDIngestResult(
            variable=variable,
            issue_time=_ensure_utc(issue_time),
            parquet_path=parquet_path,
            source_path=grib_path,
            row_count=len(df),
        )
    return results


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NDFD archive forecasts into Parquet.")
    parser.add_argument(
        "--issue",
        required=True,
        help="Forecast issue time in UTC (e.g., 2024-05-01T00).",
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


def main() -> None:
    args = _parse_cli_args()
    issue_time = _parse_issue_time(args.issue)
    results = ingest_ndfd_archive(
        issue_time,
        variables=args.vars,
        bbox=tuple(args.bbox) if args.bbox else None,
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


__all__ = ["ingest_ndfd_archive", "download_ndfd_grib", "NDFDIngestResult"]
